import os
import json
import torch
import pandas as pd
from tqdm import tqdm
#from wandb.sdk.lib.wbauth import prompt

import model.utils as ut
# from model.KCProbing import KCProbing
# from model.InspectOutputContext import InspectOutputContext
# from model.prompts import PROMPT_CORRECT as prompt

from logical_datasets.BeliefBankDataset import BeliefBankDataset
from model.InspectOutputContext import InspectOutputContext
from prober.KCProbing import KCProbing


# from sklearn.metrics import auc
# from sklearn.metrics import roc_auc_score, precision_recall_curve

prompt = "Answer with only one word, is the following statement true: {fact}?\nAnswer:"

class HallucinationDetection:
    # -------------
    # Constants
    # -------------
    TARGET_LAYERS = list(range(0, 22))  # Upper bound excluded
    MAX_NEW_TOKENS = 100
    DEFAULT_DATASET = "beliefbank"
    CACHE_DIR_NAME = "activation_cache"
    ACTIVATION_TARGET = ["hidden", "mlp", "attn"]
    PREDICTION_DIR = "predictions"
    RESULTS_DIR = "results"
    PREDICTIONS_FILE_NAME = "kc_predictions_layer{layer}.jsonl"

    # Adattiamo le etichette per BeliefBank (0=no, 1=yes)
    LABELS = {
        0: "no",
        1: "yes"
    }

    # -------------
    # Constructor
    # -------------
    def __init__(self, project_dir):
        self.project_dir = project_dir

    def load_dataset(self, dataset_name=DEFAULT_DATASET, use_local=False, label=1):
        """Carica il BeliefBankDataset. Ignoriamo use_local perché BeliefBank usa file locali di default."""
        print("--" * 50)
        print(f"Loading dataset {dataset_name}")
        print("--" * 50)
        self.label = label

        if dataset_name == "beliefbank" or dataset_name == "belief":
            self.dataset_name = "beliefbank"
            # Inizializza il dataset. Nota: 'label' qui filtra/imposta il target nel tuo dataset
            self.dataset = BeliefBankDataset(
                project_root=self.project_dir,
                data_type="facts",
                label=label,
                shuffle=False  # Mantieni False per estrarre le attivazioni in ordine
            )
        else:
            raise ValueError(f"Dataset {dataset_name} not supported in this adapted version.")

    def load_llm(self, llm_name, use_local=False, dtype=torch.bfloat16, use_device_map=True, use_flash_attn=False):
        print("--" * 50)
        print(f"Loading LLM {llm_name}")
        print("--" * 50)
        self.llm_name = llm_name
        self.tokenizer = ut.load_tokenizer(llm_name, local=use_local)
        self.llm = ut.load_llm(llm_name, ut.create_bnb_config(), local=use_local, dtype=dtype,
                               use_device_map=use_device_map, use_flash_attention=use_flash_attn)
        print("--" * 50)

    def load_kc_probing(self, activation, layer):
        print("--" * 50)
        print(f"Loading KC probing model for layer {layer} of {activation} activations")
        print("--" * 50)
        self.kc_layer = layer
        self.kc_model = KCProbing(self.project_dir, activation=activation, layer=layer)
        print("--" * 50)

    # -------------
    # Main Methods
    # -------------
    @torch.no_grad()
    def predict_llm(self, llm_name, data_name=DEFAULT_DATASET, label=1, use_local=False, dtype=torch.bfloat16,
                    use_device_map=True, use_flash_attn=False):
        self.load_dataset(dataset_name=data_name, use_local=use_local, label=label)
        self.load_llm(llm_name, use_local=use_local, dtype=dtype, use_device_map=use_device_map,
                      use_flash_attn=use_flash_attn)

        print("--" * 50)
        print("Hallucination Detection - Saving LLM's activations")
        print("--" * 50)

        print("\n0. Prepare folders")
        self._create_folders_if_not_exists(label=label)

        print(f"\n1. Saving {self.llm_name} activations for layers {self.TARGET_LAYERS}")
        self.save_activations()

        print("--" * 50)

    @torch.no_grad()
    def predict_kc(self, target, layer, llm_name, data_name=DEFAULT_DATASET, use_local=False, label=1):
        self.load_kc_probing(target, layer)
        self.load_dataset(dataset_name=data_name, use_local=use_local, label=label)

        print("--" * 50)
        print("Hallucination Detection - Saving KC Probing Predictions")
        print(f"Activation: {target}, Layer: {self.kc_layer}, Label: {self.LABELS[label]}")
        print("--" * 50)

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
            pred = {
                "instance_id": instance_id,
                "lang": self.dataset.get_language_by_instance_id(instance_id),
                "prediction": self.kc_model.predict(activation).item(),
                "label": label  # Qui salviamo 0 o 1
            }
            preds.append(pred)

        path_to_save = os.path.join(result_path, self.dataset_name, target,
                                    self.PREDICTIONS_FILE_NAME.format(layer=self.kc_layer))

        if not os.path.exists(os.path.dirname(path_to_save)):
            os.makedirs(os.path.dirname(path_to_save))
        else:
            if os.path.exists(path_to_save):
                existing_preds = json.load(open(path_to_save, "r"))
                preds.extend(existing_preds)

        json.dump(preds, open(path_to_save, "w"), indent=4)

        print(f"\t -> Predictions saved to {path_to_save}")

    def eval(self, target, llm_name, data_name=DEFAULT_DATASET):
        result_path = os.path.join(self.project_dir, self.PREDICTION_DIR, llm_name)
        print("--" * 50)
        print("Hallucination Detection - Evaluation")
        print(f"Activation: {target}, Layer: {self.kc_layer}")
        print("--" * 50)

        print(f"\n1. Load predictions for layer {self.kc_layer}")
        preds_path = os.path.join(result_path, data_name, target,
                                  self.PREDICTIONS_FILE_NAME.format(layer=self.kc_layer))
        if not os.path.exists(preds_path):
            raise FileNotFoundError(f"Predictions file not found: {preds_path}")

        preds = json.load(open(preds_path, "r"))

        print("\n2. Compute metrics")
        metrics = HallucinationDetection.compute_all_metrics(preds, data_name)

        print("\n3. Save results")
        self._save_metrics(metrics, target, data_name, llm_name)

        print("--" * 50)

    def save_activations(self):
        module_names = []
        module_names += [f'model.layers.{idx}' for idx in self.TARGET_LAYERS]
        module_names += [f'model.layers.{idx}.self_attn' for idx in self.TARGET_LAYERS]
        module_names += [f'model.layers.{idx}.mlp' for idx in self.TARGET_LAYERS]

        for idx in tqdm(range(len(self.dataset)), desc="Saving activations"):
            # Adattato al return del tuo BeliefBankDataset: fact, computed_label, id
            fact, fact_label, instance_id = self.dataset[idx]

            # In BeliefBank non abbiamo una coppia "question/answer" classica, ma un "fact".
            # Assicurati che il tuo `prompt` (in model.prompts) sia formattato per accettare
            # una singola frase, es: "Is the following statement true: {fact}?\nAnswer:"
            model_input = prompt.format(fact=fact)
            tokens = self.tokenizer(model_input, return_tensors="pt")
            attention_mask = tokens["attention_mask"].to("cuda") if "attention_mask" in tokens else None

            with InspectOutputContext(self.llm, module_names, save_generation=True,
                                      save_dir=self.generation_save_dir) as inspect:
                output = self.llm.generate(
                    input_ids=tokens["input_ids"].to("cuda"),
                    max_new_tokens=self.MAX_NEW_TOKENS,
                    attention_mask=attention_mask,
                    do_sample=False,
                    top_p=None,
                    temperature=0.,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )

                generated_ids = output.sequences[0][tokens["input_ids"].shape[1]:]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

                ut.save_generation_output(generated_text, model_input, instance_id, self.generation_save_dir)

                if hasattr(output, 'scores') and output.scores:
                    logits = torch.stack(output.scores, dim=1)
                    ut.save_model_logits(logits, instance_id, self.logits_save_dir)

            for module, ac in inspect.catcher.items():
                ac_last = ac[0, -1].float()
                layer_idx = int(module.split(".")[2])

                save_name = f"layer{layer_idx}-id{instance_id}.pt"
                if "mlp" in module:
                    save_path = os.path.join(self.mlp_save_dir, save_name)
                elif "self_attn" in module:
                    save_path = os.path.join(self.attn_save_dir, save_name)
                else:
                    save_path = os.path.join(self.hidden_save_dir, save_name)

                torch.save(ac_last, save_path)

        self.combine_activations()

    def combine_activations(self):
        results_dir = os.path.join(self.project_dir, self.CACHE_DIR_NAME)
        model_name = self.llm_name.split("/")[-1]

        task = self._get_task_name(self.label)

        for aa in self.ACTIVATION_TARGET:
            act_dir = os.path.join(results_dir, model_name, self.dataset_name, f"activation_{aa}", task)

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

        task = self._get_task_name(label=label)
        print(f"Task: {task}")

        self.hidden_save_dir = os.path.join(results_dir, model_name, self.dataset_name, "activation_hidden", task)
        self.mlp_save_dir = os.path.join(results_dir, model_name, self.dataset_name, "activation_mlp", task)
        self.attn_save_dir = os.path.join(results_dir, model_name, self.dataset_name, "activation_attn", task)

        self.generation_save_dir = os.path.join(results_dir, model_name, self.dataset_name, "generations", task)
        self.logits_save_dir = os.path.join(results_dir, model_name, self.dataset_name, "logits", task)

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


import os
import torch

# Assicurati che l'import corrisponda alla struttura delle tue cartelle
from model.HallucinationDetection import HallucinationDetection


def main():
    # 1. Configurazione Iniziale
    # Imposta la root del progetto alla cartella in cui si trova main.py
    PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logical_datasets"))

    # Inserisci qui il nome del modello HuggingFace (o il percorso locale) che vuoi testare
    # Esempi: "meta-llama/Llama-2-7b-hf", "mistralai/Mistral-7B-v0.1", o un modello più piccolo
    LLM_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # 2. Inizializzazione della classe
    detector = HallucinationDetection(project_dir=PROJECT_DIR)
    print("🚀 Inizio pipeline di test Hallucination Detection")

    # 3. Caricamento Dataset
    # Impostiamo label=1 (che nel nostro nuovo setup significa fatti veri, "yes")
    detector.load_dataset(dataset_name="beliefbank", label=1)

    # ---------------------------------------------------------
    # TRUCCO PER IL TEST: Limitiamo a 10 prompt
    # Usiamo la funzione che hai scritto tu stesso nel Dataset!
    # ---------------------------------------------------------
    detector.dataset.get_sample(max_samples=10)
    print(f"\n📊 Dataset limitato a {len(detector.dataset)} elementi per il test rapido.\n")

    # 4. Caricamento Modello Linguistico
    # (Imposta use_device_map=True per usare la GPU automaticamente)
    detector.load_llm(
        llm_name=LLM_NAME,
        use_local=False,
        dtype=torch.float16,  # Usa float16 o bfloat16 a seconda della tua GPU
        use_device_map=True
    )

    # 5. Creazione delle cartelle di output
    # Questo creerà 'activation_cache/nome_modello/beliefbank/...'
    detector._create_folders_if_not_exists(label=1)

    # 6. Estrazione e Salvataggio Attivazioni
    print("\n🧠 Inizio generazione e salvataggio attivazioni...")
    detector.save_activations()

    print("\n✅ Test completato con successo!")
    print(f"Controlla la cartella '{detector.CACHE_DIR_NAME}' per vedere i tensori salvati.")


if __name__ == '__main__':
    # Disabilita eventuali warning noiosi sui symlink di HuggingFace (opzionale)
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    print(torch.cuda.is_available())

    main()