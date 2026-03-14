import os
import json
import torch
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

from probing.LinearProber import LinearProber
from analysis.VisualisationUtils import VisualisationUtils
from core.StorageManager import StorageManager

logger = logging.getLogger(__name__)


class ProberEvaluator:
    ACTIVATION_TARGETS = ["hidden", "mlp", "attn"]

    def __init__(self, project_dir: str, dataset: Any, dataset_name: str, target_layers: List[int],
                 random_seed: int = 42, cache_dir_name: str = "activation_cache", prompt_id: str = "base_v1"):
        self.project_dir = project_dir
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.target_layers = target_layers
        self.random_seed = random_seed
        self.cache_dir_name = cache_dir_name
        self.prompt_id = prompt_id

        # Directory di predizione ripristinata
        self.prediction_dir = "predictions"
        self.predictions_file_name = "predictions_layer{layer}.json"

    def _balance_classes(self, activations: torch.Tensor, labels: torch.Tensor, ids: torch.Tensor) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Esegue undersampling per bilanciare 50/50 le classi, portandosi dietro anche gli ID."""
        g = torch.Generator(device=labels.device)
        g.manual_seed(self.random_seed)

        idx_0 = (labels == 0.0).nonzero(as_tuple=True)[0]
        idx_1 = (labels == 1.0).nonzero(as_tuple=True)[0]
        min_samples = min(len(idx_0), len(idx_1))

        if min_samples == 0:
            return None, None, None

        idx_0_bal = idx_0[torch.randperm(len(idx_0), generator=g, device=labels.device)[:min_samples]]
        idx_1_bal = idx_1[torch.randperm(len(idx_1), generator=g, device=labels.device)[:min_samples]]

        balanced_indices = torch.cat([idx_0_bal, idx_1_bal])
        return activations[balanced_indices], labels[balanced_indices], ids[balanced_indices]

    # FIX COMPATIBILITÀ: Ripristinato prompt_id nella firma del metodo
    def train_and_evaluate_probers(self, llm_name: str, test_size: float = 0.2, prompt_id: str = "base_v1",
                                   epochs: int = 30,
                                   use_undersampling: bool = False, tune_wd: bool = True) -> None:

        # Sovrascriviamo il prompt_id di classe se ne viene passato uno nuovo
        self.prompt_id = prompt_id

        logger.info("\n" + "=" * 50 + "\n🛠️ Avvio Addestramento Probers (Hallucination Detection)\n" + "=" * 50)
        llm_short_name = llm_name.split("/")[-1]

        id_to_label = {}
        for i in range(len(self.dataset)):
            _, label_str, instance_id = self.dataset[i]
            val = 1.0 if str(label_str).lower() in ["yes", "1", "true"] else 0.0
            id_to_label[instance_id] = val

        metrics_results = []

        for target in self.ACTIVATION_TARGETS:
            logger.info(f"\n--- Training su {target.upper()} ---")

            for layer in self.target_layers:
                try:
                    activations, instance_ids = StorageManager.load_activations(
                        model_name=llm_short_name,
                        data_name=self.dataset_name,
                        prompt_id=self.prompt_id,
                        analyse_activation=target,
                        layer_idx=layer,
                        results_dir=os.path.join(self.project_dir, self.cache_dir_name)
                    )
                except FileNotFoundError as e:
                    # AGGIUNGI QUESTO LOG
                    logger.debug(f"Layer {layer} [{target}]: File non trovato. Percorso: {e.filename}")
                    continue

                valid_mask = [i for i, iid in enumerate(instance_ids) if iid in id_to_label]

                if not valid_mask:
                    logger.warning(f"Layer {layer} [{target}]: Nessun ID corrispondente trovato nel dataset. Salto...")
                    continue

                    # Filtra i tensori delle attivazioni e la lista degli ID
                activations = activations[valid_mask]
                instance_ids = [instance_ids[i] for i in valid_mask]

                # Ora la creazione del tensore labels non fallirà MAI, anche se la cache è sporca
                labels = torch.tensor([id_to_label[iid] for iid in instance_ids], dtype=torch.float32)

                #labels = torch.tensor([id_to_label[iid] for iid in instance_ids], dtype=torch.float32)
                ids_tensor = torch.tensor(instance_ids, dtype=torch.int32)  # Tensorizziamo gli ID

                if use_undersampling:
                    # Facciamo passare anche gli ID dal bilanciatore
                    X, y, ids = self._balance_classes(activations, labels, ids_tensor)
                    if X is None:
                        continue
                    pos_weight = None
                else:
                    X, y, ids = activations, labels, ids_tensor
                    num_pos = (y == 1.0).sum().item()
                    num_neg = (y == 0.0).sum().item()
                    pos_weight = num_neg / num_pos if num_pos > 0 else 1.0

                if len(torch.unique(y)) < 2:
                    continue

                # FIX PREDICITON: Passiamo anche "ids" a train_test_split per sapere chi è chi!
                try:
                    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
                        X, y, ids, test_size=test_size, random_state=self.random_seed, stratify=y.numpy()
                    )
                except ValueError:
                    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
                        X, y, ids, test_size=test_size, random_state=self.random_seed
                    )

                prober = LinearProber(self.project_dir, activation=target, layer=layer, input_dim=X.shape[-1])
                results = prober.train(X_train, y_train, X_test, y_test, epochs=epochs,
                                       pos_weight=pos_weight, tune_wd=tune_wd)

                y_test_np = y_test.numpy()
                probs = results["probabilities"]
                preds_binary = (probs > 0.5).astype(int)

                # --- 🔴 FIX: RIPRISTINO SALVATAGGIO PREDIZIONI ---
                predictions_to_save = []
                for i in range(len(id_test)):
                    predictions_to_save.append({
                        "instance_id": int(id_test[i]),
                        "true_label": int(y_test_np[i]),
                        "probability": float(probs[i]),
                        "predicted_label": int(preds_binary[i])
                    })

                pred_dir = os.path.join(self.project_dir, self.prediction_dir, llm_short_name, self.dataset_name,
                                        self.prompt_id, target)
                os.makedirs(pred_dir, exist_ok=True)
                pred_path = os.path.join(pred_dir, self.predictions_file_name.format(layer=layer))

                # Legge file vecchi e aggiunge i nuovi (Comportamento Originale)
                if os.path.exists(pred_path):
                    with open(pred_path, "r") as f:
                        existing_preds = json.load(f)
                    predictions_to_save.extend(existing_preds)

                with open(pred_path, "w") as f:
                    json.dump(predictions_to_save, f, indent=4)
                # ------------------------------------------------

                try:
                    auroc = roc_auc_score(y_test_np, probs)
                    auprc = average_precision_score(y_test_np, probs)
                except ValueError:
                    auroc = 0.5
                    auprc = 0.5

                logger.info(
                    f"Layer {layer:02d} | Acc: {results['accuracy']:.4f} | AUROC: {auroc:.4f} | AUPRC: {auprc:.4f} | Best WD: {results['best_wd']}")

                metrics_results.append({
                    "target": target,
                    "layer": layer,
                    "accuracy": results["accuracy"],
                    "auroc": auroc,
                    "auprc": auprc,
                    "best_wd": results["best_wd"]
                })

        if metrics_results:
            VisualisationUtils.save_prober_results(metrics_results, self.project_dir,
                                                   use_undersampling=use_undersampling)
        else:
            logger.warning("Nessun modello addestrato con successo.")
