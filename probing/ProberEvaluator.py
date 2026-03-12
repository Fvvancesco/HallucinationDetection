import os
import json
import torch
import logging
from typing import List, Dict, Tuple, Optional, Any
from sklearn.model_selection import train_test_split

from probing.LinearProber import LinearProber
from utils import Utils as ut
from analysis.VisualisationUtils import VisualisationUtils

logger = logging.getLogger(__name__)


class ProberEvaluator:
    ACTIVATION_TARGETS = ["hidden", "mlp", "attn"]

    def __init__(self, project_dir: str, dataset: Any, dataset_name: str, target_layers: List[int],
                 random_seed: int = 42, cache_dir_name: str = "activation_cache"):
        self.project_dir = project_dir
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.target_layers = target_layers
        self.random_seed = random_seed
        self.cache_dir_name = cache_dir_name

        # Nuove costanti locali per le directory di predizione
        self.prediction_dir = "predictions"
        self.predictions_file_name = "predictions_layer{layer}.jsonl"

    def train_and_evaluate_probers(self, llm_name: str, test_size: float = 0.2, epochs: int = 30) -> None:
        logger.info("\n" + "=" * 50 + "\n🛠️ Avvio Addestramento Probers (Hallucination Detection)\n" + "=" * 50)

        if not self.dataset:
            raise ValueError("Dataset non fornito all'inizializzazione del ProberEvaluator.")

        id_to_label = {iid: label for _, label, iid in self.dataset}

        llm_short_name = llm_name.split("/")[-1]
        generations_dir = os.path.join(self.project_dir, self.cache_dir_name, llm_short_name, self.dataset_name,
                                       "generations")

        # Mapping: 1.0 = Allucinazione, 0.0 = Corretta
        id_to_hallucination: Dict[int, float] = {}
        for iid, ground_truth in id_to_label.items():
            gen_file = os.path.join(generations_dir, f"generation_{iid}.json")
            if os.path.exists(gen_file):
                with open(gen_file, "r", encoding="utf-8") as f:
                    model_answer = json.load(f).get("generated_output", "").strip().lower()
                    pred_label = "yes" if ("true" in model_answer or "yes" in model_answer) else "no"
                    id_to_hallucination[iid] = 1.0 if pred_label != ground_truth else 0.0
            else:
                id_to_hallucination[iid] = 0.0

        metrics_results: List[Dict[str, Any]] = []
        base_results_dir = os.path.join(self.project_dir, self.cache_dir_name, llm_short_name, self.dataset_name)

        for target in self.ACTIVATION_TARGETS:
            logger.info(f"\n--- Training su {target.upper()} ---")
            for layer in self.target_layers:
                act_path = os.path.join(base_results_dir, f"activation_{target}", f"layer{layer}_activations.pt")
                ids_path = os.path.join(base_results_dir, f"activation_{target}", f"layer{layer}_instance_ids.json")

                if not os.path.exists(act_path) or not os.path.exists(ids_path):
                    continue

                activations = torch.load(act_path, map_location="cuda", weights_only=True)
                with open(ids_path, "r") as f:
                    instance_ids = json.load(f)

                labels = torch.tensor([id_to_hallucination[i] for i in instance_ids], dtype=torch.float32).cuda()
                activations_balanced, labels_balanced = self._balance_classes(activations, labels)

                if activations_balanced is None or labels_balanced is None:
                    logger.warning(f"⚠️ Errore al layer {layer}: Una delle classi è vuota. Salto.")
                    continue

                X_train, X_test, y_train, y_test = train_test_split(
                    activations_balanced, labels_balanced, test_size=test_size,
                    random_state=self.random_seed, stratify=labels_balanced.cpu()
                )

                prober = LinearProber(self.project_dir, activation=target, layer=layer, load_pretrained=False,
                                      input_dim=activations_balanced.shape[1])
                acc = prober.train(X_train, y_train, X_test, y_test, epochs=epochs)
                prober.save_model()

                metrics_results.append({"target": target, "layer": layer, "accuracy": acc})
                logger.info(f"Layer {layer:02d} | Accuracy: {acc:.4f} | Samples totali: {len(labels_balanced)}")

        if metrics_results:
            # Deleghiamo il salvataggio dei grafici alla classe preposta
            VisualisationUtils.save_prober_results(metrics_results, self.project_dir)

    @torch.no_grad()
    def predict_prober(self, target: str, layer: int, llm_name: str, label: int = 1) -> None:
        if not self.dataset:
            raise RuntimeError("Setup non completato. Impossibile avviare il probing.")

        # Carichiamo il modello del prober "on the fly"
        prober_model = LinearProber(self.project_dir, activation=target, layer=layer, load_pretrained=True)
        llm_short_name = llm_name.split("/")[-1]

        activations, instance_ids = ut.load_activations(
            model_name=llm_short_name,
            data_name=self.dataset_name,
            analyse_activation=target,
            layer_idx=layer,
            results_dir=os.path.join(self.project_dir, self.cache_dir_name)
        )

        preds = [
            {
                "instance_id": iid,
                "lang": self.dataset.get_language_by_instance_id(iid) if hasattr(self.dataset,
                                                                                 'get_language_by_instance_id') else "en",
                "prediction": prober_model.predict(act).item(),
                "label": label
            }
            for act, iid in zip(activations, instance_ids)
        ]

        save_path = os.path.join(self.project_dir, self.prediction_dir, llm_short_name, self.dataset_name, target,
                                 self.predictions_file_name.format(layer=layer))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                existing_preds = json.load(f)
            preds.extend(existing_preds)

        with open(save_path, "w") as f:
            json.dump(preds, f, indent=4)
        logger.info(f"\t -> Predictions saved to {save_path}")

    def _balance_classes(self, activations: torch.Tensor, labels: torch.Tensor) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Esegue undersampling per bilanciare 50/50 le classi."""
        # Setup generatore per la riproducibilità locale (usando il random_seed dell'istanza)
        g = torch.Generator(device=labels.device)
        g.manual_seed(self.random_seed)

        idx_0 = (labels == 0.0).nonzero(as_tuple=True)[0]
        idx_1 = (labels == 1.0).nonzero(as_tuple=True)[0]
        min_samples = min(len(idx_0), len(idx_1))

        if min_samples == 0:
            return None, None

        idx_0_bal = idx_0[torch.randperm(len(idx_0), generator=g, device=labels.device)[:min_samples]]
        idx_1_bal = idx_1[torch.randperm(len(idx_1), generator=g, device=labels.device)[:min_samples]]

        balanced_indices = torch.cat([idx_0_bal, idx_1_bal])
        # Rimescolamento finale
        balanced_indices = balanced_indices[torch.randperm(len(balanced_indices), generator=g, device=labels.device)]

        return activations[balanced_indices], labels[balanced_indices]