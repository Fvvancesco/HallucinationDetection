import os
import json
import torch
import logging
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)


class StorageManager:
    # Costante per le sottocartelle
    ACTIVATION_TARGETS = ["hidden", "mlp", "attn"]

    def __init__(self, project_dir: str, llm_name: str, dataset_name: str,
                 target_layers: List[int], prompt_id: str = "base_v1", cache_dir_name: str = "activation_cache"):
        self.project_dir = project_dir
        self.llm_name = llm_name.split("/")[-1]
        self.dataset_name = dataset_name
        self.prompt_id = prompt_id
        self.cache_dir_name = cache_dir_name
        self.target_layers = target_layers

        # Dizionario per i percorsi salvati
        self.dirs: Dict[str, str] = {}

        # Buffer RAM per il chunking
        self.activation_buffer: Dict[str, Dict[int, List[torch.Tensor]]] = {}
        self.instance_ids_buffer: Dict[int, List[int]] = {}

    def setup_directories(self) -> None:
        """Centralizza la creazione di tutte le directory di output necessarie."""
        if not self.llm_name or not self.dataset_name:
            raise ValueError("LLM e Dataset devono essere passati al costruttore.")

        base_dir = os.path.join(self.project_dir, self.cache_dir_name, self.llm_name, self.dataset_name, self.prompt_id)
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

    @staticmethod
    def save_generation_output(output_text: str, input_text: str, instance_id: int, save_dir: str) -> str:
        os.makedirs(save_dir, exist_ok=True)

        output_data = {
            "instance_id": instance_id,
            "input": input_text,
            "generated_output": output_text,
            "timestamp": str(torch.cuda.current_stream().query())
        }

        save_path = os.path.join(save_dir, f"generation_{instance_id}.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        return save_path

    @staticmethod
    def save_model_logits(logits: torch.Tensor, instance_id: int, save_dir: str) -> str:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"logits_{instance_id}.pt")
        torch.save(logits.cpu(), save_path)
        return save_path

    @staticmethod
    def parse_layer_id_and_instance_id(s: str) -> Tuple[int, int]:
        instance_idx = -1
        layer_idx = -1
        try:
            layer_s, id_s = s.split("-")
            layer_idx = int(layer_s[len("layer"):])
            instance_idx = int(id_s[len("id"):-len(".pt")])
        except Exception as e:
            logger.error(f"Errore parsing stringa {s}: {e}")
        return layer_idx, instance_idx

    @staticmethod
    def load_activations(model_name: str, data_name: str, analyse_activation: str, layer_idx: int, results_dir: str) -> \
    Tuple[torch.Tensor, List[int]]:
        model_name = model_name.split("/")[-1]
        base_dir = os.path.join(results_dir, model_name, data_name, f"activation_{analyse_activation}")

        act_path = os.path.join(base_dir, f"layer{layer_idx}_activations.pt")
        ids_path = os.path.join(base_dir, f"layer{layer_idx}_instance_ids.json")

        if not os.path.exists(act_path) or not os.path.exists(ids_path):
            raise FileNotFoundError(f"File mancanti per il layer {layer_idx} in {base_dir}")

        with open(ids_path, "r") as f:
            instance_ids = json.load(f)

        activations = torch.load(act_path, map_location="cpu", weights_only=True)
        return activations, instance_ids

    def save_tensors_to_disk(self, catcher: Dict[str, Any], instance_id: int, is_attribution: bool = False,
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

    def init_buffers(self) -> None:
        """Inizializza o resetta i buffer in RAM per il chunking."""
        self.activation_buffer = {
            target: {layer: [] for layer in self.target_layers}
            for target in self.ACTIVATION_TARGETS
        }
        self.instance_ids_buffer = {layer: [] for layer in self.target_layers}

    def bufferize_tensors(self, catcher: Dict[str, Any], instance_id: int, is_attribution: bool = False) -> None:
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
                tensor_to_save = tensor[0, -1].detach().cpu().to(torch.bfloat16)

            layer_idx = int(module.split(".")[2])

            if "mlp" in module:
                self.activation_buffer["mlp"][layer_idx].append(tensor_to_save)
            elif "self_attn" in module:
                self.activation_buffer["attn"][layer_idx].append(tensor_to_save)
            else:
                self.activation_buffer["hidden"][layer_idx].append(tensor_to_save)
                self.instance_ids_buffer[layer_idx].append(instance_id)

    def flush_buffer_to_disk(self, chunk_idx: int, prefix: str = "activation") -> None:
        """Prende tutto ciò che è in RAM, fa lo stack e salva il chunk su disco."""
        logger.info(f"Saving chunk {chunk_idx} to disk...")
        results_dir = os.path.join(self.project_dir, self.cache_dir_name, self.llm_name, self.dataset_name)

        for target in self.ACTIVATION_TARGETS:
            dir_name = f"{prefix}_{target}"
            act_dir = os.path.join(results_dir, dir_name)
            os.makedirs(act_dir, exist_ok=True)

            for layer_idx in self.target_layers:
                if not self.activation_buffer[target][layer_idx]:
                    continue

                stacked_acts = torch.stack(self.activation_buffer[target][layer_idx])
                save_path = os.path.join(act_dir, f"layer{layer_idx}_chunk{chunk_idx}.pt")
                torch.save(stacked_acts, save_path)

                if target == "hidden":
                    ids_save_path = os.path.join(act_dir, f"layer{layer_idx}_ids_chunk{chunk_idx}.json")
                    with open(ids_save_path, "w") as f:
                        json.dump(self.instance_ids_buffer[layer_idx], f, indent=4)

    def combine_activations(self, is_attribution: bool = False) -> None:
        """Fonde i singoli file creati pre-allocando i tensori per prevenire OOM."""
        results_dir = os.path.join(self.project_dir, self.cache_dir_name)
        prefix = "attributions" if is_attribution else "activation"

        for aa in self.ACTIVATION_TARGETS:
            act_dir = os.path.join(results_dir, self.llm_name, self.dataset_name, f"{prefix}_{aa}")
            if not os.path.exists(act_dir): continue

            act_files = [f for f in os.listdir(act_dir) if f.endswith(".pt") and len(f.split("-")) == 2]
            layer_group_files: Dict[int, List[Tuple[str, int]]] = {lid: [] for lid in self.target_layers}

            for act_f in act_files:
                layer_id, instance_id = self.parse_layer_id_and_instance_id(act_f)
                layer_group_files[layer_id].append((act_f, instance_id))

            for layer_id, files in layer_group_files.items():
                if not files: continue
                files.sort(key=lambda x: x[1])

                loaded_paths = [os.path.join(act_dir, f[0]) for f in files]
                instance_ids = [f[1] for f in files]

                first_tensor = torch.load(loaded_paths[0], map_location="cpu", weights_only=True)
                combined_acts = torch.empty((len(files), *first_tensor.shape), dtype=first_tensor.dtype, device="cpu")

                for idx, path_to_load in enumerate(loaded_paths):
                    combined_acts[idx] = torch.load(path_to_load, map_location="cpu", weights_only=True)

                torch.save(combined_acts, os.path.join(act_dir, f"layer{layer_id}_activations.pt"))
                with open(os.path.join(act_dir, f"layer{layer_id}_instance_ids.json"), "w") as f:
                    json.dump(instance_ids, f, indent=4)

                for p in loaded_paths:
                    os.remove(p)