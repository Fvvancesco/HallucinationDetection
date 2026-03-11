import os
import json
import logging
import torch
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Any
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

import model.utils as ut
from logical_datasets.BeliefBankDataset import BeliefBankDataset
from model.InspectOutputContext import InspectOutputContext
from prober.LinearProber import LinearProber

# -------------
# Configurazione Logging
# -------------
logger = logging.getLogger(__name__)

# -------------
# Prompts Globali
# -------------
SYSTEM_PROMPT = "You are a helpful assistant. Answer truthfully with only one word: TRUE or FALSE."
USER_PROMPT_TEMPLATE = "Is the following statement true: {fact}?"


class HallucinationDetection:
    # -------------
    # Constants
    # -------------
    TARGET_LAYERS: List[int] = list(range(0, 32))
    MAX_NEW_TOKENS: int = 5
    DEFAULT_DATASET: str = "beliefbank"
    CACHE_DIR_NAME: str = "activation_cache"
    ACTIVATION_TARGET: List[str] = ["hidden", "mlp", "attn"]
    PREDICTION_DIR: str = "predictions"
    RESULTS_DIR: str = "results"
    PREDICTIONS_FILE_NAME: str = "kc_predictions_layer{layer}.jsonl"

    # 4. Rimozione Magic Numbers
    RANDOM_SEED: int = 42

    LABELS: Dict[int, str] = {0: "no", 1: "yes"}

    def __init__(self, project_dir: str) -> None:
        self.project_dir: str = project_dir
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        # State variables
        self.dataset: Optional[BeliefBankDataset] = None
        self.dataset_name: Optional[str] = None
        self.llm: Any = None
        self.tokenizer: Any = None
        self.llm_name: Optional[str] = None
        self.label: int = 1
        self.prober_model: Optional[LinearProber] = None
        self.prober_layer: Optional[int] = None
        self.dirs: Dict[str, str] = {}

        # Buffer RAM per il chunking
        self.activation_buffer: Dict[str, Dict[int, List[torch.Tensor]]] = {}
        self.instance_ids_buffer: Dict[int, List[int]] = {}

    # -------------
    # Setup Methods
    # -------------
    def load_dataset(self, dataset_name: str = DEFAULT_DATASET, use_local: bool = False, label: int = 1) -> None:
        logger.info(f"{'--' * 25} Loading dataset {dataset_name} {'--' * 25}")
        self.label = label
        if dataset_name in ["beliefbank", "belief"]:
            self.dataset_name = "beliefbank"
            self.dataset = BeliefBankDataset(
                project_root=self.project_dir,
                data_type="constraints",
                label=label,
                shuffle=False
            )
        else:
            raise ValueError(f"Dataset {dataset_name} not supported.")

    def load_llm(self, llm_name: str, use_local: bool = False, dtype: torch.dtype = torch.bfloat16,
                 use_device_map: bool = True, use_flash_attn: bool = False) -> None:
        logger.info(f"{'--' * 25} Loading LLM {llm_name} {'--' * 25}")
        self.llm_name = llm_name
        self.tokenizer = ut.load_tokenizer(llm_name, local=use_local)
        self.llm = ut.load_llm(
            llm_name,
            ut.create_bnb_config(),
            local=use_local,
            dtype=dtype,
            use_device_map=use_device_map,
            use_flash_attention=use_flash_attn
        )
        self.device = self.llm.device

    def load_linear_prober(self, activation: str, layer: int) -> None:
        logger.info(f"{'--' * 25} Loading Linear Prober for layer {layer} ({activation}) {'--' * 25}")
        self.prober_layer = layer
        self.prober_model = LinearProber(self.project_dir, activation=activation, layer=layer)

    def setup_directories(self) -> None:
        """Centralizza la creazione di tutte le directory di output necessarie."""
        if not self.llm_name or not self.dataset_name:
            raise ValueError("LLM e Dataset devono essere caricati prima del setup delle directory.")

        model_name = self.llm_name.split("/")[-1]
        base_dir = os.path.join(self.project_dir, self.CACHE_DIR_NAME, model_name, self.dataset_name)

        self.dirs = {
            "hidden": os.path.join(base_dir, "activation_hidden"),
            "mlp": os.path.join(base_dir, "activation_mlp"),
            "attn": os.path.join(base_dir, "activation_attn"),
            "generations": os.path.join(base_dir, "generations"),
            "logits": os.path.join(base_dir, "logits"),
            "attr_hidden": os.path.join(base_dir, "attributions_hidden"),
            "attr_mlp": os.path.join(base_dir, "attributions_mlp"),
            "attr_attn": os.path.join(base_dir, "attributions_attn")
        }

        for path in self.dirs.values():
            os.makedirs(path, exist_ok=True)

    # -------------
    # Main Prediction & Evaluation
    # -------------
    @torch.no_grad()
    def predict_llm(self, llm_name: str, data_name: str = DEFAULT_DATASET, label: int = 1,
                    use_local: bool = False, use_chat_template: bool = True) -> None:
        self.load_dataset(dataset_name=data_name, use_local=use_local, label=label)
        self.load_llm(llm_name, use_local=use_local)
        self.setup_directories()

        logger.info(f"[1] Saving {self.llm_name} activations for layers {self.TARGET_LAYERS}")
        self.save_activations(use_chat_template=use_chat_template)

    @torch.no_grad()
    def predict_prober(self, target: str, layer: int, llm_name: str, data_name: str = DEFAULT_DATASET,
                       use_local: bool = False, label: int = 1) -> None:
        self.load_linear_prober(target, layer)
        self.load_dataset(dataset_name=data_name, use_local=use_local, label=label)

        if not self.dataset_name or not self.prober_model:
            raise RuntimeError("Setup non completato. Impossibile avviare il prober.")

        activations, instance_ids = ut.load_activations(
            model_name=llm_name,
            data_name=self.dataset_name,
            analyse_activation=target,
            layer_idx=self.prober_layer,
            results_dir=os.path.join(self.project_dir, self.CACHE_DIR_NAME)
        )

        preds = [
            {
                "instance_id": iid,
                "lang": self.dataset.get_language_by_instance_id(iid) if hasattr(self.dataset,
                                                                                 'get_language_by_instance_id') else "en",
                "prediction": self.prober_model.predict(act).item(),
                "label": label
            }
            for act, iid in zip(activations, instance_ids)
        ]

        save_path = os.path.join(self.project_dir, self.PREDICTION_DIR, llm_name, self.dataset_name, target,
                                 self.PREDICTIONS_FILE_NAME.format(layer=layer))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                existing_preds = json.load(f)
            preds.extend(existing_preds)

        with open(save_path, "w") as f:
            json.dump(preds, f, indent=4)
        logger.info(f"\t -> Predictions saved to {save_path}")

    # -------------
    # Core Extraction Methods
    # -------------
    def save_activations(self, use_chat_template: bool = True) -> None:
        """Salvataggio classico (file singolo per istanza). Lento su disco, ma robusto."""
        module_names = self._get_target_modules()
        if not self.dataset:
            raise ValueError("Dataset non caricato.")

        for idx in tqdm(range(len(self.dataset)), desc="Saving activations"):
            fact, _, instance_id = self.dataset[idx]
            formatted_prompt, tokens = self._prepare_inputs(fact, use_chat_template)

            with InspectOutputContext(self.llm, module_names, save_generation=True,
                                      save_dir=self.dirs["generations"]) as inspect:
                self._generate_text_and_logits(tokens, formatted_prompt, instance_id)

            self._save_tensors_to_disk(inspect.catcher, instance_id, is_attribution=False, save_last=False)

        self.combine_activations()

    def save_activations_pure_forward(self, use_chat_template: bool = True) -> None:
        module_names = self._get_target_modules()
        if not self.dataset:
            raise ValueError("Dataset non caricato.")

        for idx in tqdm(range(len(self.dataset)), desc="Saving pure activations"):
            fact, _, instance_id = self.dataset[idx]
            formatted_prompt, tokens = self._prepare_inputs(fact, use_chat_template)

            # Fase 1: Generazione libera
            self._generate_text_and_logits(tokens, formatted_prompt, instance_id)

            # Fase 2: Estrazione pura (forward solo sul prompt)
            with InspectOutputContext(self.llm, module_names) as inspect:
                _ = self.llm(input_ids=tokens["input_ids"], attention_mask=tokens.get("attention_mask"))

            self._save_tensors_to_disk(inspect.catcher, instance_id, is_attribution=False, save_last=False)

        self.combine_activations()

    def save_attributions_and_grads(self) -> None:
        logger.info(f"{'--' * 25} Saving Attributions (Act x Grad) {'--' * 25}")
        torch.set_grad_enabled(True)
        self.llm.eval()
        for param in self.llm.parameters():
            param.requires_grad = False

        if not self.dataset:
            raise ValueError("Dataset non caricato.")

        module_names = self._get_target_modules()
        pos_token = self.tokenizer.encode(" TRUE", add_special_tokens=False)[-1]
        neg_token = self.tokenizer.encode(" FALSE", add_special_tokens=False)[-1]

        target_dirs = {"hidden": self.dirs["attr_hidden"], "mlp": self.dirs["attr_mlp"], "attn": self.dirs["attr_attn"]}

        for idx in tqdm(range(len(self.dataset)), desc="Computing attributions"):
            fact, _, instance_id = self.dataset[idx]
            _, tokens = self._prepare_inputs(fact, use_chat_template=True)

            with InspectOutputContext(self.llm, module_names, track_grads=True) as inspect:
                outputs = self.llm(input_ids=tokens["input_ids"], attention_mask=tokens.get("attention_mask"))
                metric = outputs.logits[0, -1, pos_token] - outputs.logits[0, -1, neg_token]

                self.llm.zero_grad()
                metric.backward()

            self._save_tensors_to_disk(inspect.catcher, instance_id, is_attribution=True, target_dirs=target_dirs,
                                       save_last=False)

        self.combine_activations(is_attribution=True)

    # -------------
    # Chunking Methods (Ripristinati & Ottimizzati)
    # -------------
    def _init_buffers(self) -> None:
        """Inizializza o resetta i buffer in RAM per il chunking."""
        self.activation_buffer = {
            target: {layer: [] for layer in self.TARGET_LAYERS}
            for target in self.ACTIVATION_TARGET
        }
        self.instance_ids_buffer = {layer: [] for layer in self.TARGET_LAYERS}

    def _bufferize_tensors(self, catcher: Dict[str, Any], instance_id: int, is_attribution: bool = False) -> None:
        """Salva i tensori nel dizionario in RAM pre-processandoli."""
        for module, tensor in catcher.items():
            if tensor is None or (is_attribution and getattr(tensor, 'grad', None) is None):
                continue

            # Processamento
            if is_attribution:
                t_act = tensor[0, -1].detach().float()
                t_grad = tensor.grad[0, -1].detach().float()
                tensor_to_save = (t_act * t_grad).cpu()
            else:
                # bfloat16 in RAM salva moltissimo spazio
                tensor_to_save = tensor[0, -1].detach().cpu().to(torch.bfloat16)

            layer_idx = int(module.split(".")[2])

            if "mlp" in module:
                self.activation_buffer["mlp"][layer_idx].append(tensor_to_save)
            elif "self_attn" in module:
                self.activation_buffer["attn"][layer_idx].append(tensor_to_save)
            else:
                self.activation_buffer["hidden"][layer_idx].append(tensor_to_save)
                # L'ID si salva solo una volta per il target hidden come master
                self.instance_ids_buffer[layer_idx].append(instance_id)

    def _flush_buffer_to_disk(self, chunk_idx: int, prefix: str = "activation") -> None:
        """Prende tutto ciò che è in RAM, fa lo stack e salva il chunk su disco."""
        logger.info(f"Saving chunk {chunk_idx} to disk...")
        model_name = self.llm_name.split("/")[-1]
        results_dir = os.path.join(self.project_dir, self.CACHE_DIR_NAME, model_name, self.dataset_name)

        for target in self.ACTIVATION_TARGET:
            # Scegli la directory giusta (activation_ o attributions_)
            dir_name = f"{prefix}_{target}"
            act_dir = os.path.join(results_dir, dir_name)
            os.makedirs(act_dir, exist_ok=True)

            for layer_idx in self.TARGET_LAYERS:
                if not self.activation_buffer[target][layer_idx]:
                    continue

                stacked_acts = torch.stack(self.activation_buffer[target][layer_idx])
                save_path = os.path.join(act_dir, f"layer{layer_idx}_chunk{chunk_idx}.pt")
                torch.save(stacked_acts, save_path)

                if target == "hidden":
                    ids_save_path = os.path.join(act_dir, f"layer{layer_idx}_ids_chunk{chunk_idx}.json")
                    with open(ids_save_path, "w") as f:
                        json.dump(self.instance_ids_buffer[layer_idx], f, indent=4)

    def save_activations_chunked(self, use_chat_template: bool = True, chunk_size: int = 1000) -> None:
        """Versione I/O ottimizzata: usa la RAM per raggruppare migliaia di attivazioni prima di salvare."""
        module_names = self._get_target_modules()
        if not self.dataset:
            raise ValueError("Dataset non caricato.")

        self._init_buffers()
        chunk_idx = 0

        for idx in tqdm(range(len(self.dataset)), desc="Saving activations (Chunked)"):
            fact, _, instance_id = self.dataset[idx]
            formatted_prompt, tokens = self._prepare_inputs(fact, use_chat_template)

            with InspectOutputContext(self.llm, module_names, save_generation=True,
                                      save_dir=self.dirs["generations"]) as inspect:
                self._generate_text_and_logits(tokens, formatted_prompt, instance_id)

            self._bufferize_tensors(inspect.catcher, instance_id, is_attribution=False)

            if (idx + 1) % chunk_size == 0:
                self._flush_buffer_to_disk(chunk_idx, prefix="activation")
                chunk_idx += 1
                self._init_buffers()

        # Flush finale
        if self.instance_ids_buffer[self.TARGET_LAYERS[0]]:
            self._flush_buffer_to_disk(chunk_idx, prefix="activation")

    # -------------
    # Helpers & Utilities
    # -------------
    def _generate_text_and_logits(self, tokens: Dict[str, torch.Tensor], formatted_prompt: str,
                                  instance_id: int) -> None:
        output = self.llm.generate(
            input_ids=tokens["input_ids"],
            max_new_tokens=self.MAX_NEW_TOKENS,
            attention_mask=tokens.get("attention_mask"),
            do_sample=False,
            temperature=0.,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )

        gen_ids = output.sequences[0][tokens["input_ids"].shape[1]:]
        gen_list = gen_ids.cpu().tolist() if hasattr(gen_ids, "cpu") else list(gen_ids)
        gen_text = self.tokenizer.decode(gen_list, skip_special_tokens=True).strip()

        ut.save_generation_output(gen_text, formatted_prompt, instance_id, self.dirs["generations"])

        if hasattr(output, 'scores') and output.scores:
            ut.save_model_logits(torch.stack(output.scores, dim=1), instance_id, self.dirs["logits"])

    def _prepare_inputs(self, fact: str, use_chat_template: bool) -> Tuple[str, Dict[str, torch.Tensor]]:
        user_prompt = USER_PROMPT_TEMPLATE.format(fact=fact)
        if use_chat_template:
            messages = ut.build_messages(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt, k=0)
            tokens = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                        return_tensors="pt", return_dict=True)
            return self.tokenizer.decode(tokens["input_ids"][0]), {k: v.to(self.device) for k, v in tokens.items()}

        tokens = self.tokenizer(user_prompt, return_tensors="pt").to(self.device)
        return user_prompt, tokens

    def _get_target_modules(self) -> List[str]:
        modules = [f'model.layers.{idx}' for idx in self.TARGET_LAYERS]
        modules += [f'model.layers.{idx}.self_attn' for idx in self.TARGET_LAYERS]
        modules += [f'model.layers.{idx}.mlp' for idx in self.TARGET_LAYERS]
        return modules

    def _save_tensors_to_disk(self, catcher: Dict[str, Any], instance_id: int, is_attribution: bool = False,
                              target_dirs: Optional[Dict[str, str]] = None, save_last: bool = True) -> None:
        dirs = target_dirs or {"hidden": self.dirs["hidden"], "mlp": self.dirs["mlp"], "attn": self.dirs["attn"]}

        for module, tensor in catcher.items():
            if tensor is None or (is_attribution and getattr(tensor, 'grad', None) is None): continue

            idx_slice = -1 if save_last else slice(None)
            t_act = tensor[0, idx_slice].detach().float()

            if is_attribution:
                t_grad = tensor.grad[0, idx_slice].detach().float()
                tensor_to_save = t_act * t_grad
            else:
                tensor_to_save = t_act

            layer_idx = int(module.split(".")[2])
            save_name = f"layer{layer_idx}-id{instance_id}.pt"

            sub_dir = dirs["mlp"] if "mlp" in module else dirs["attn"] if "self_attn" in module else dirs["hidden"]
            torch.save(tensor_to_save.cpu(), os.path.join(sub_dir, save_name))

    def combine_activations(self, is_attribution: bool = False) -> None:
        """
        Fonde i singoli file creati.
        1. Rischio OOM risolto: Pre-alloca il tensore finale invece di caricare migliaia di tensori in liste RAM.
        5. Sicurezza path: Controlla che siano file '.pt'.
        """
        if not self.llm_name or not self.dataset_name:
            raise RuntimeError("Dati modello/dataset mancanti.")

        model_name = self.llm_name.split("/")[-1]
        results_dir = os.path.join(self.project_dir, self.CACHE_DIR_NAME)
        prefix = "attributions" if is_attribution else "activation"

        for aa in self.ACTIVATION_TARGET:
            act_dir = os.path.join(results_dir, model_name, self.dataset_name, f"{prefix}_{aa}")
            if not os.path.exists(act_dir): continue

            # Sicurezza: solo file .pt con la giusta struttura
            act_files = [f for f in os.listdir(act_dir) if f.endswith(".pt") and len(f.split("-")) == 2]
            layer_group_files: Dict[int, List[Tuple[str, int]]] = {lid: [] for lid in self.TARGET_LAYERS}

            for act_f in act_files:
                layer_id, instance_id = ut.parse_layer_id_and_instance_id(act_f)
                layer_group_files[layer_id].append((act_f, instance_id))

            for layer_id, files in layer_group_files.items():
                if not files: continue
                files.sort(key=lambda x: x[1])  # Ordina per instance_id

                loaded_paths = [os.path.join(act_dir, f[0]) for f in files]
                instance_ids = [f[1] for f in files]

                # --- PREVENZIONE OOM (Out-Of-Memory) ---
                # 1. Carichiamo solo il primo tensore per scoprirne la forma (shape) e il tipo (dtype)
                first_tensor = torch.load(loaded_paths[0], map_location="cpu", weights_only=True)

                # 2. Pre-allochiamo uno spazio vuoto esatto nella RAM per tutti gli elementi
                combined_acts = torch.empty((len(files), *first_tensor.shape), dtype=first_tensor.dtype, device="cpu")

                # 3. Riempiamo progressivamente gli indici (senza usare stack/append massivi)
                for idx, path_to_load in enumerate(loaded_paths):
                    combined_acts[idx] = torch.load(path_to_load, map_location="cpu", weights_only=True)

                # Salvataggio
                torch.save(combined_acts, os.path.join(act_dir, f"layer{layer_id}_activations.pt"))
                with open(os.path.join(act_dir, f"layer{layer_id}_instance_ids.json"), "w") as f:
                    json.dump(instance_ids, f, indent=4)

                # Pulizia file parziali
                for p in loaded_paths:
                    os.remove(p)

    # -------------
    # Prober Training
    # -------------
    def train_and_evaluate_probers(self, llm_name: str, data_name: str = DEFAULT_DATASET,
                                   test_size: float = 0.2, epochs: int = 30) -> None:
        logger.info("\n" + "=" * 50 + "\n🛠️ Avvio Addestramento Probers (Hallucination Detection)\n" + "=" * 50)
        self.load_dataset(data_name)
        if not self.dataset:
            raise ValueError("Impossibile caricare il dataset.")

        id_to_label = {iid: label for _, label, iid in self.dataset}

        llm_short_name = llm_name.split("/")[-1]
        generations_dir = os.path.join(self.project_dir, self.CACHE_DIR_NAME, llm_short_name, data_name, "generations")

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
        base_results_dir = os.path.join(self.project_dir, self.CACHE_DIR_NAME, llm_short_name, data_name)

        for target in self.ACTIVATION_TARGET:
            logger.info(f"\n--- Training su {target.upper()} ---")
            for layer in self.TARGET_LAYERS:
                act_path = os.path.join(base_results_dir, f"activation_{target}", f"layer{layer}_activations.pt")
                ids_path = os.path.join(base_results_dir, f"activation_{target}", f"layer{layer}_instance_ids.json")

                if not os.path.exists(act_path) or not os.path.exists(ids_path): continue

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
                    random_state=self.RANDOM_SEED, stratify=labels_balanced.cpu()
                )

                prober = LinearProber(self.project_dir, activation=target, layer=layer, load_pretrained=False,
                                      input_dim=activations_balanced.shape[1])
                acc = prober.train(X_train, y_train, X_test, y_test, epochs=epochs)
                prober.save_model()

                metrics_results.append({"target": target, "layer": layer, "accuracy": acc})
                logger.info(f"Layer {layer:02d} | Accuracy: {acc:.4f} | Samples totali: {len(labels_balanced)}")

        if metrics_results:
            self._save_prober_results(metrics_results)

    def _balance_classes(self, activations: torch.Tensor, labels: torch.Tensor) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Esegue undersampling per bilanciare 50/50 le classi."""
        # Setup generatore per la riproducibilità locale (usando il RANDOM_SEED)
        g = torch.Generator(device=labels.device)
        g.manual_seed(self.RANDOM_SEED)

        idx_0 = (labels == 0.0).nonzero(as_tuple=True)[0]
        idx_1 = (labels == 1.0).nonzero(as_tuple=True)[0]
        min_samples = min(len(idx_0), len(idx_1))

        if min_samples == 0: return None, None

        idx_0_bal = idx_0[torch.randperm(len(idx_0), generator=g)[:min_samples]]
        idx_1_bal = idx_1[torch.randperm(len(idx_1), generator=g)[:min_samples]]

        balanced_indices = torch.cat([idx_0_bal, idx_1_bal])
        # Rimescolamento finale
        balanced_indices = balanced_indices[torch.randperm(len(balanced_indices), generator=g)]

        return activations[balanced_indices], labels[balanced_indices]

    def _save_prober_results(self, metrics_results: List[Dict[str, Any]]) -> None:
        df_metrics = pd.DataFrame(metrics_results)
        df_metrics.to_csv(os.path.join(self.project_dir, self.RESULTS_DIR, "prober_training_results.csv"), index=False)

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_metrics, x='layer', y='accuracy', hue='target', marker='o')
        plt.title('Prober Validation Accuracy per Layer (Classes Balanced 50/50)')
        plt.grid(True)

        plot_path = os.path.join(self.project_dir, self.RESULTS_DIR, "prober_accuracy_profile.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        logger.info(f"\n✅ Training completato! Profilo salvato in: {plot_path}")