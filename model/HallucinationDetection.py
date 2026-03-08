import os
import json
import torch
import pandas as pd
from tqdm import tqdm

import model.utils as ut
from logical_datasets.BeliefBankDataset import BeliefBankDataset
from model.InspectOutputContext import InspectOutputContext
from prober.KCProbing import KCProbing

# -------------
# Prompts Globali
# -------------
SYSTEM_PROMPT = "You are a helpful assistant. Answer truthfully with only one word: TRUE or FALSE."
USER_PROMPT_TEMPLATE = "Is the following statement true: {fact}?"


class HallucinationDetection:
    # -------------
    # Constants
    # -------------
    TARGET_LAYERS = list(range(0, 32))  # Upper bound excluded
    MAX_NEW_TOKENS = 5  # Ridotto poiché ci aspettiamo solo TRUE/FALSE
    DEFAULT_DATASET = "beliefbank"
    CACHE_DIR_NAME = "activation_cache"
    ACTIVATION_TARGET = ["hidden", "mlp", "attn"]
    PREDICTION_DIR = "predictions"
    RESULTS_DIR = "results"
    PREDICTIONS_FILE_NAME = "kc_predictions_layer{layer}.jsonl"

    LABELS = {
        0: "no",
        1: "yes"
    }

    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------
    # Setup Methods
    # -------------
    def load_dataset(self, dataset_name=DEFAULT_DATASET, use_local=False, label=1):
        print("--" * 25 + f" Loading dataset {dataset_name} " + "--" * 25)
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
            raise ValueError(f"Dataset {dataset_name} not supported in this adapted version.")

    def load_llm(self, llm_name, use_local=False, dtype=torch.bfloat16, use_device_map=True, use_flash_attn=False):
        print("--" * 25 + f" Loading LLM {llm_name} " + "--" * 25)
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
        self.device = self.llm.device  # Aggiorna il device basato su dove è stato caricato il modello

    def load_kc_probing(self, activation, layer):
        print("--" * 25 + f" Loading KC probing for layer {layer} ({activation}) " + "--" * 25)
        self.kc_layer = layer
        self.kc_model = KCProbing(self.project_dir, activation=activation, layer=layer)

    # -------------
    # Main Methods
    # -------------
    @torch.no_grad()
    def predict_llm(self, llm_name, data_name=DEFAULT_DATASET, label=1, use_local=False,
                    dtype=torch.bfloat16, use_device_map=True, use_flash_attn=False, use_chat_template=True):
        self.load_dataset(dataset_name=data_name, use_local=use_local, label=label)
        self.load_llm(llm_name, use_local=use_local, dtype=dtype, use_device_map=use_device_map,
                      use_flash_attn=use_flash_attn)

        print(f"\n[0] Preparing folders for label {self.LABELS[label]}")
        self._create_folders_if_not_exists(label=label)

        print(f"[1] Saving {self.llm_name} activations for layers {self.TARGET_LAYERS}")
        self.save_activations(use_chat_template=use_chat_template)

    @torch.no_grad()
    def predict_kc(self, target, layer, llm_name, data_name=DEFAULT_DATASET, use_local=False, label=1):
        self.load_kc_probing(target, layer)
        self.load_dataset(dataset_name=data_name, use_local=use_local, label=label)

        result_path = os.path.join(self.project_dir, self.PREDICTION_DIR, llm_name)
        activations, instance_ids = ut.load_activations(
            model_name=llm_name,
            data_name=self.dataset_name,
            analyse_activation=target,
            activation_type=self.LABELS[label],
            layer_idx=self.kc_layer,
            results_dir=os.path.join(self.project_dir, self.CACHE_DIR_NAME)
        )

        preds = []
        for activation, instance_id in zip(activations, instance_ids):
            preds.append({
                "instance_id": instance_id,
                "lang": self.dataset.get_language_by_instance_id(instance_id),
                "prediction": self.kc_model.predict(activation).item(),
                "label": label
            })

        path_to_save = os.path.join(result_path, self.dataset_name, target, self.PREDICTIONS_FILE_NAME.format(layer=self.kc_layer))
        os.makedirs(os.path.dirname(path_to_save), exist_ok=True)

        if os.path.exists(path_to_save):
            existing_preds = json.load(open(path_to_save, "r"))
            preds.extend(existing_preds)

        with open(path_to_save, "w") as f:
            json.dump(preds, f, indent=4)
        print(f"\t -> Predictions saved to {path_to_save}")

    def eval(self, target, llm_name, data_name=DEFAULT_DATASET):
        result_path = os.path.join(self.project_dir, self.PREDICTION_DIR, llm_name)
        preds_path = os.path.join(result_path, data_name, target, self.PREDICTIONS_FILE_NAME.format(layer=self.kc_layer))

        if not os.path.exists(preds_path):
            raise FileNotFoundError(f"Predictions file not found: {preds_path}")

        preds = json.load(open(preds_path, "r"))
        metrics = self.compute_all_metrics(preds, data_name)
        self._save_metrics(metrics, target, data_name, llm_name)

    # -------------
    # Core Extraction Methods
    # -------------
    def save_activations(self, use_chat_template=True):
        module_names = self._get_target_modules()

        for idx in tqdm(range(len(self.dataset)), desc="Saving activations"):
            fact, fact_label, instance_id = self.dataset[idx]

            formatted_prompt, tokens = self._prepare_inputs(fact, use_chat_template)
            attention_mask = tokens.get("attention_mask")

            with InspectOutputContext(self.llm, module_names, save_generation=True,
                                      save_dir=self.generation_save_dir) as inspect:
                output = self.llm.generate(
                    input_ids=tokens["input_ids"],
                    max_new_tokens=self.MAX_NEW_TOKENS,
                    max_length=None,
                    attention_mask=attention_mask,
                    do_sample=False,
                    temperature=0.,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )


                generated_ids = output.sequences[0][tokens["input_ids"].shape[1]:]
                """
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True) #si fa prima passare alla cpu per evitare bug di decoding, un po' più lento però
                """
                generated_ids_list = generated_ids.cpu().tolist() if hasattr(generated_ids, "cpu") else list(generated_ids)
                generated_text = self.tokenizer.decode(generated_ids_list, skip_special_tokens=True).strip()
                ut.save_generation_output(generated_text, formatted_prompt, instance_id, self.generation_save_dir)

                if hasattr(output, 'scores') and output.scores:
                    logits = torch.stack(output.scores, dim=1)
                    ut.save_model_logits(logits, instance_id, self.logits_save_dir)

            #self._save_tensors_to_disk(inspect.catcher, instance_id, is_attribution=False)
            self._save_tensors_to_disk(inspect.catcher, instance_id, is_attribution=False, save_last=False)

        self.combine_activations()

    def _init_buffers(self):
        """Inizializza o resetta i buffer in RAM per il chunking."""
        self.activation_buffer = {
            target: {layer: [] for layer in self.TARGET_LAYERS}
            for target in self.ACTIVATION_TARGET
        }
        self.instance_ids_buffer = {layer: [] for layer in self.TARGET_LAYERS}

    def save_activations_only_ram(self, use_chat_template=True, chunk_size=1000):
        module_names = self._get_target_modules()

        # Inizializziamo il primo buffer vuoto
        self._init_buffers()
        chunk_idx = 0

        for idx in tqdm(range(len(self.dataset)), desc="Saving activations"):
            fact, fact_label, instance_id = self.dataset[idx]

            model_input, tokens = self._prepare_inputs(fact, use_chat_template)
            attention_mask = tokens.get("attention_mask")

            with InspectOutputContext(self.llm, module_names, save_generation=True,
                                      save_dir=self.generation_save_dir) as inspect:
                output = self.llm.generate(
                    input_ids=tokens["input_ids"],
                    max_new_tokens=self.MAX_NEW_TOKENS,
                    max_length=None, #
                    attention_mask=attention_mask,
                    do_sample=False,
                    temperature=0.,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )

                generated_ids = output.sequences[0][tokens["input_ids"].shape[1]:]
                generated_ids_list = generated_ids.cpu().tolist() if hasattr(generated_ids, "cpu") else list(
                    generated_ids)
                generated_text = self.tokenizer.decode(generated_ids_list, skip_special_tokens=True).strip()
                ut.save_generation_output(generated_text, model_input, instance_id, self.generation_save_dir)

                if hasattr(output, 'scores') and output.scores:
                    logits = torch.stack(output.scores, dim=1)
                    ut.save_model_logits(logits, instance_id, self.logits_save_dir)

            # Accodiamo in RAM
            self._bufferize_tensors(inspect.catcher, instance_id)

            # --- LOGICA DI CHUNKING ---
            # Se abbiamo raggiunto la dimensione del blocco, salviamo su disco e resettiamo
            if (idx + 1) % chunk_size == 0:
                self._flush_buffer_to_disk(chunk_idx)
                chunk_idx += 1
                self._init_buffers()  # Svuota la RAM per il prossimo blocco

        # --- SALVATAGGIO FINALE ---
        # Alla fine del ciclo, salviamo eventuali dati rimanenti (l'ultimo blocco parziale)
        if len(self.instance_ids_buffer[self.TARGET_LAYERS[0]]) > 0:
            self._flush_buffer_to_disk(chunk_idx)

    def _bufferize_tensors(self, catcher, instance_id):
        """Prende i tensori catturati e li salva nel dizionario in RAM."""
        for module, tensor in catcher.items():
            if tensor is None:
                continue

            # Prende l'ultimo token, lo sposta su CPU e usa bfloat16 per dimezzare il peso in RAM
            tensor_to_save = tensor[0, -1].detach().cpu().to(torch.bfloat16)
            layer_idx = int(module.split(".")[2])

            # Smistamento nel target giusto (hidden, mlp, attn)
            if "mlp" in module:
                self.activation_buffer["mlp"][layer_idx].append(tensor_to_save)
            elif "self_attn" in module:
                self.activation_buffer["attn"][layer_idx].append(tensor_to_save)
            else:
                self.activation_buffer["hidden"][layer_idx].append(tensor_to_save)
                # Salviamo l'instance id una volta sola (usiamo il blocco 'hidden' come riferimento)
                self.instance_ids_buffer[layer_idx].append(instance_id)

    def _flush_buffer_to_disk(self, chunk_idx):
        """Prende tutto ciò che è in RAM, lo impila (stack) e crea i file finali con l'indice del chunk."""
        print(f"\nSaving chunk {chunk_idx} to disk...")
        model_name = self.llm_name.split("/")[-1]
        results_dir = os.path.join(self.project_dir, self.CACHE_DIR_NAME)

        for target in self.ACTIVATION_TARGET:
            act_dir = os.path.join(results_dir, model_name, self.dataset_name, f"activation_{target}")
            os.makedirs(act_dir, exist_ok=True)

            for layer_idx in self.TARGET_LAYERS:
                # Impila tutte le attivazioni di questo layer per il chunk corrente
                stacked_acts = torch.stack(self.activation_buffer[target][layer_idx])

                # Salva il tensore aggiungendo l'indice del blocco al nome del file
                save_path = os.path.join(act_dir, f"layer{layer_idx}_activations_chunk{chunk_idx}.pt")
                torch.save(stacked_acts, save_path)

                # Salva gli IDs corrispondenti
                if target == "hidden":
                    ids_save_path = os.path.join(act_dir, f"layer{layer_idx}_instance_ids_chunk{chunk_idx}.json")
                    with open(ids_save_path, "w") as f:
                        json.dump(self.instance_ids_buffer[layer_idx], f, indent=4)

        print(f"Chunk {chunk_idx} saved successfully!")

    def save_attributions_and_grads(self):
        print("--" * 25 + " Saving Attributions (Act x Grad) " + "--" * 25)
        torch.set_grad_enabled(True)
        self.llm.eval()
        for param in self.llm.parameters():
            param.requires_grad = False

        module_names = self._get_target_modules()

        # Setup directories
        attr_dirs = {
            "hidden": os.path.join(self.project_dir, self.CACHE_DIR_NAME, "attributions_hidden"),
            "mlp": os.path.join(self.project_dir, self.CACHE_DIR_NAME, "attributions_mlp"),
            "attn": os.path.join(self.project_dir, self.CACHE_DIR_NAME, "attributions_attn")
        }
        for d in attr_dirs.values():
            os.makedirs(d, exist_ok=True)

        for idx in tqdm(range(len(self.dataset)), desc="Computing attributions"):
            fact, fact_label, instance_id = self.dataset[idx]

            _, tokens = self._prepare_inputs(fact, use_chat_template=True)

            pos_token = self.tokenizer.encode(" TRUE", add_special_tokens=False)[-1]
            neg_token = self.tokenizer.encode(" FALSE", add_special_tokens=False)[-1]

            with InspectOutputContext(self.llm, module_names, track_grads=True) as inspect:
                outputs = self.llm(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens.get("attention_mask")
                )

                logits = outputs.logits
                last_token_logits = logits[0, -1, :]
                metric = last_token_logits[pos_token] - last_token_logits[neg_token] #metrica differenza dei logit

                self.llm.zero_grad()
                metric.backward()

            # Pass directories dict explicitly for attributions
            self._save_tensors_to_disk(inspect.catcher, instance_id, is_attribution=True, target_dirs=attr_dirs)
    """
    def save_attributions_and_grads_chunking(self, chunk_size=1000):
        print("--" * 25 + " Saving Attributions (Act x Grad) " + "--" * 25)
        torch.set_grad_enabled(True)
        self.llm.eval()
        for param in self.llm.parameters():
            param.requires_grad = False

        module_names = self._get_target_modules()

        # 1. Spostiamo la tokenizzazione fuori dal ciclo!
        pos_token = self.tokenizer.encode(" TRUE", add_special_tokens=False)[-1]
        neg_token = self.tokenizer.encode(" FALSE", add_special_tokens=False)[-1]

        # 2. Inizializziamo il buffer in RAM per le attribuzioni
        self._init_buffers()
        chunk_idx = 0

        for idx in tqdm(range(len(self.dataset)), desc="Computing attributions"):
            fact, fact_label, instance_id = self.dataset[idx]

            _, tokens = self._prepare_inputs(fact, use_chat_template=True)

            with InspectOutputContext(self.llm, module_names, track_grads=True) as inspect:
                outputs = self.llm(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens.get("attention_mask")
                )

                logits = outputs.logits
                last_token_logits = logits[0, -1, :]
                metric = last_token_logits[pos_token] - last_token_logits[neg_token]

                self.llm.zero_grad()
                metric.backward()

            # 3. Accodiamo in RAM usando un nuovo metodo dedicato (Act x Grad)
            self._bufferize_attributions(inspect.catcher, instance_id)

            # --- LOGICA DI CHUNKING ---
            if (idx + 1) % chunk_size == 0:
                # Nota: passiamo 'attributions' come prefisso per salvare nelle cartelle giuste
                self._flush_buffer_to_disk(chunk_idx, prefix="attributions")
                chunk_idx += 1
                self._init_buffers()

        # Salvataggio finale per il blocco parziale rimanente
        if len(self.instance_ids_buffer[self.TARGET_LAYERS[0]]) > 0:
            self._flush_buffer_to_disk(chunk_idx, prefix="attributions")"""

    # -------------
    # Helpers & Utilities
    # -------------
    def _prepare_inputs(self, fact, use_chat_template):
        """Prepara e tokenizza il prompt, restituendo stringa pulita per i log e dict di tensori."""
        user_prompt = USER_PROMPT_TEMPLATE.format(fact=fact)

        if use_chat_template:
            # Sfruttiamo build_messages importato da utils
            messages = ut.build_messages(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                k=0
            )

            tokens = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True
            )
            model_input = self.tokenizer.decode(tokens["input_ids"][0])
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
        else:
            model_input = user_prompt
            tokens = self.tokenizer(model_input, return_tensors="pt").to(self.device)

        return model_input, tokens

    def _get_target_modules(self):
        """Restituisce la lista dei moduli di cui estrarre le attivazioni."""
        modules = [f'model.layers.{idx}' for idx in self.TARGET_LAYERS]
        modules += [f'model.layers.{idx}.self_attn' for idx in self.TARGET_LAYERS]
        modules += [f'model.layers.{idx}.mlp' for idx in self.TARGET_LAYERS]
        return modules

    def _save_tensors_to_disk(self, catcher, instance_id, is_attribution=False, target_dirs=None, save_last=True):
        """Salva i tensori estratti dalle callback su disco."""
        for module, tensor in catcher.items():
            if tensor is None or (is_attribution and getattr(tensor, 'grad', None) is None):
                continue

            if save_last:
                if is_attribution:
                    act_last = tensor[0, -1].detach().float()
                    grad_last = tensor.grad[0, -1].detach().float()
                    tensor_to_save = act_last * grad_last
                else:
                    tensor_to_save = tensor[0, -1].detach().float()
            else:
                if is_attribution:
                    # Usiamo ':' invece di '-1' per prendere TUTTI i token della sequenza
                    act_all = tensor[0, :].detach().float()
                    grad_all = tensor.grad[0, :].detach().float()
                    tensor_to_save = act_all * grad_all
                else:
                    tensor_to_save = tensor[0, -1].detach().float()

            tensor_to_save = tensor_to_save.cpu()
            layer_idx = int(module.split(".")[2])
            save_name = f"layer{layer_idx}-id{instance_id}.pt"

            if target_dirs:  # Directory per le attribuzioni
                if "mlp" in module:
                    save_path = os.path.join(target_dirs["mlp"], save_name)
                elif "self_attn" in module:
                    save_path = os.path.join(target_dirs["attn"], save_name)
                else:
                    save_path = os.path.join(target_dirs["hidden"], save_name)
            else:  # Directory per le attivazioni classiche
                if "mlp" in module:
                    save_path = os.path.join(self.mlp_save_dir, save_name)
                elif "self_attn" in module:
                    save_path = os.path.join(self.attn_save_dir, save_name)
                else:
                    save_path = os.path.join(self.hidden_save_dir, save_name)

            torch.save(tensor_to_save, save_path)


    def combine_activations(self):
        results_dir = os.path.join(self.project_dir, self.CACHE_DIR_NAME)
        model_name = self.llm_name.split("/")[-1]

        #task = self._get_task_name(self.label)

        for aa in self.ACTIVATION_TARGET:
            act_dir = os.path.join(results_dir, model_name, self.dataset_name, f"activation_{aa}")

            act_files = list(os.listdir(act_dir))
            act_files = [f for f in act_files if len(f.split("-")) == 2]

            act_files_layer_idx_instance_idx = [
                [act_f, ut.parse_layer_id_and_instance_id(os.path.basename(act_f))]
                for act_f in act_files
            ]

            layer_group_files = {lid: [] for lid in self.TARGET_LAYERS}
            for act_f, (layer_id, instance_id) in act_files_layer_idx_instance_idx:
                layer_group_files[layer_id].append([act_f, instance_id])

            for layer_id in self.TARGET_LAYERS:
                layer_group_files[layer_id] = sorted(layer_group_files[layer_id], key=lambda x: x[1])

                acts = []
                loaded_paths = []
                instance_ids = []
                for idx, (act_f, instance_id) in enumerate(layer_group_files[layer_id]):
                    path_to_load = os.path.join(act_dir, act_f)
                    acts.append(torch.load(path_to_load))
                    loaded_paths.append(path_to_load)
                    instance_ids.append(instance_id)

                acts = torch.stack(acts)
                save_path = os.path.join(act_dir, f"layer{layer_id}_activations.pt")
                torch.save(acts, save_path)

                ids_save_path = os.path.join(act_dir, f"layer{layer_id}_instance_ids.json")
                json.dump(instance_ids, open(ids_save_path, "w"), indent=4)

                for p in loaded_paths:
                    os.remove(p)

    @staticmethod
    def compute_all_metrics(preds, data_name):
        preds_df = pd.DataFrame(preds)

        all_preds = preds_df["prediction"].tolist()

        # Adattato per non dipendere da Mushroom: BeliefBank ha sempre delle label reali
        labels = preds_df["label"].tolist()

        metrics = HallucinationDetection.compute_metrics(all_preds, labels)

        return metrics

    @staticmethod
    def compute_metrics(preds, labels):
        # Nota: Affinché queste metriche funzionino, `preds` e `labels` devono essere array numerici (0, 1)
        # e non stringhe ("yes", "no"). Ho mantenuto la logica intatta assumendo che KCModel restituisca numeri
        correct = sum([1 for p, l in zip(preds, labels) if p == l])

        # Se importi sklearn, scommenta queste righe. Assicurati che preds contenga probabilità per AUC.
        # AUC = roc_auc_score(labels, preds)
        # precision, recall, thresholds = precision_recall_curve(labels, preds)
        # AUPRC = auc(recall, precision)

        ACC = correct / len(labels)

        return {"ACC": ACC}  # Ritorno solo ACC per ora, aggiungi AUC e AUPRC se scommenti l'import

    # -------------
    # Utility Methods
    # -------------
    def _get_task_name(self, label):
        return self.LABELS[label]

    def _create_folders_if_not_exists(self, label=1):
        model_name = self.llm_name.split("/")[-1]

        results_dir = os.path.join(self.project_dir, self.CACHE_DIR_NAME)

        """task = self._get_task_name(label=label)
        print(f"Task: {task}")"""

        self.hidden_save_dir = os.path.join(results_dir, model_name, self.dataset_name, "activation_hidden")
        self.mlp_save_dir = os.path.join(results_dir, model_name, self.dataset_name, "activation_mlp")
        self.attn_save_dir = os.path.join(results_dir, model_name, self.dataset_name, "activation_attn")

        self.generation_save_dir = os.path.join(results_dir, model_name, self.dataset_name, "generations")
        self.logits_save_dir = os.path.join(results_dir, model_name, self.dataset_name, "logits")

        for sd in [self.hidden_save_dir, self.mlp_save_dir, self.attn_save_dir, self.generation_save_dir,
                   self.logits_save_dir]:
            print(f"Creating directory: {sd}")
            if not os.path.exists(sd):
                os.makedirs(sd)

        print("\n\n")

    def _save_metrics(self, metrics, target, data_name, llm_name):
        metrics_path = os.path.join(self.project_dir, self.RESULTS_DIR, llm_name, data_name, target,
                                    f"metrics_layer{self.kc_layer}.json")

        if not os.path.exists(os.path.dirname(metrics_path)):
            os.makedirs(os.path.dirname(metrics_path))
        else:
            if os.path.exists(metrics_path):
                existing_metrics = json.load(open(metrics_path, "r"))
                metrics.update(existing_metrics)

        json.dump(metrics, open(metrics_path, "w"), indent=4)

        print(f"\t -> Metrics saved to {metrics_path}")
