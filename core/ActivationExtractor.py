import os

import torch
import logging
from tqdm import tqdm
from typing import List, Dict, Tuple, Any

from utils import Utils as ut
from utils.InspectOutputContext import InspectOutputContext

logger = logging.getLogger(__name__)

class ActivationExtractor:
    def __init__(self, llm: Any, tokenizer: Any, dataset: Any, storage_manager: Any, system_prompt: str, user_prompt_template: str, pos_token_str: str, neg_token_str: str, max_new_tokens: int = 5):
        self.llm = llm
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.storage_manager = storage_manager  # Iniettiamo il manager dell'I/O
        self.device = self.llm.device
        self.pos_token_str = pos_token_str
        self.neg_token_str = neg_token_str
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template

        # 🚀 LETTURA DINAMICA DEI LAYER
        if hasattr(self.llm.config, "num_hidden_layers"):
            self.num_layers = self.llm.config.num_hidden_layers
        else:
            self.num_layers = 32  # Fallback di sicurezza

        self.target_layers = list(range(self.num_layers))

        # Prompts (idealmente andrebbero in un config.py)
        #self.SYSTEM_PROMPT = "You are a helpful assistant. Answer truthfully with only one word: TRUE or FALSE."
        #self.USER_PROMPT_TEMPLATE = "Is the following statement true: {fact}?"

    def _get_target_modules(self) -> List[str]:
        """Usa self.target_layers calcolati dinamicamente."""
        modules = [f'model.layers.{idx}' for idx in self.target_layers]
        modules += [f'model.layers.{idx}.self_attn' for idx in self.target_layers]
        modules += [f'model.layers.{idx}.mlp' for idx in self.target_layers]
        return modules

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

            if self._is_already_processed(instance_id, is_attribution=False):
                continue

            formatted_prompt, tokens = self._prepare_inputs(fact, use_chat_template)

            with InspectOutputContext(self.llm, module_names, save_generation=True,
                                      save_dir=self.storage_manager.dirs["generations"]) as inspect:
                self._generate_text_and_logits(tokens, formatted_prompt, instance_id)

            # Deleghiamo il salvataggio allo Storage Manager
            self.storage_manager.save_tensors_to_disk(inspect.catcher, instance_id, is_attribution=False, save_last=False)

        self.storage_manager.combine_activations()

    def save_activations_pure_forward(self, use_chat_template: bool = True) -> None:
        """Salvataggio classico (file singolo per istanza). Lento su disco, ma robusto. Salva fino alla fine del prompt all'utente"""
        module_names = self._get_target_modules()
        if not self.dataset:
            raise ValueError("Dataset non caricato.")

        for idx in tqdm(range(len(self.dataset)), desc="Saving pure activations"):
            fact, _, instance_id = self.dataset[idx]

            if self._is_already_processed(instance_id, is_attribution=False):
                continue

            formatted_prompt, tokens = self._prepare_inputs(fact, use_chat_template)

            # Fase 1: Generazione libera
            self._generate_text_and_logits(tokens, formatted_prompt, instance_id)

            # Fase 2: Estrazione pura (forward solo sul prompt)
            with InspectOutputContext(self.llm, module_names) as inspect:
                _ = self.llm(input_ids=tokens["input_ids"], attention_mask=tokens.get("attention_mask"))

            self.storage_manager.save_tensors_to_disk(inspect.catcher, instance_id, is_attribution=False, save_last=False)

        self.storage_manager.combine_activations()


    def save_activations_chunked(self, use_chat_template: bool = True, chunk_size: int = 1000) -> None:
        module_names = self._get_target_modules()
        if not self.dataset: raise ValueError("Dataset non caricato.")

        self.storage_manager.init_buffers()
        chunk_idx = 0
        items_in_buffer = 0  # <--- 1. NUOVO CONTATORE

        for idx in tqdm(range(len(self.dataset)), desc="Saving activations (Chunked)"):
            fact, _, instance_id = self.dataset[idx]

            # Se è già processato, saltiamo tutto (il contatore non cresce)
            if self._is_already_processed(instance_id, is_attribution=False):
                continue

            formatted_prompt, tokens = self._prepare_inputs(fact, use_chat_template)

            with InspectOutputContext(self.llm, module_names, save_generation=True,
                                      save_dir=self.storage_manager.dirs["generations"]) as inspect:
                self._generate_text_and_logits(tokens, formatted_prompt, instance_id)

            self.storage_manager.bufferize_tensors(inspect.catcher, instance_id, is_attribution=False)
            items_in_buffer += 1  # <--- 2. INCREMENTIAMO IL CONTATORE

            # 3. CONTROLLIAMO IL BUFFER, NON L'INDICE ASSOLUTO
            if items_in_buffer == chunk_size:
                self.storage_manager.flush_buffer_to_disk(chunk_idx, prefix="activation")
                chunk_idx += 1
                self.storage_manager.init_buffers()
                items_in_buffer = 0  # Resettiamo il contatore

        # Flush finale per gli elementi rimasti (anche se sono meno di 1000)
        if items_in_buffer > 0:
            self.storage_manager.flush_buffer_to_disk(chunk_idx, prefix="activation")

    def save_attributions_and_grads(self, metric_type: str = "hallucination") -> None:
        logger.info(f"{'--' * 25} Saving Attributions [{metric_type.upper()}] {'--' * 25}")
        torch.set_grad_enabled(True)
        self.llm.eval()
        for param in self.llm.parameters():
            param.requires_grad = False

        if not self.dataset:
            raise ValueError("Dataset non caricato.")

        module_names = self._get_target_modules()

        # 1. I token bersaglio ora vengono letti dinamicamente!
        pos_id = self.tokenizer.encode(self.pos_token_str, add_special_tokens=False)[-1]
        neg_id = self.tokenizer.encode(self.neg_token_str, add_special_tokens=False)[-1]

        target_dirs = {
            "hidden": self.storage_manager.dirs["attr_hidden"],
            "mlp": self.storage_manager.dirs["attr_mlp"],
            "attn": self.storage_manager.dirs["attr_attn"]
        }

        # 2. DEFINIZIONE DELLE METRICHE COME FUNZIONI (lambda)
        # Ogni metrica riceve: i logits(tensor), l'id positivo, l'id negativo, e la verità (label)
        metrics_dict = {
            "hallucination": lambda logits, p, n, label: (
                (logits[0, -1, n] - logits[0, -1, p]) if str(label).lower() == "yes"
                else (logits[0, -1, p] - logits[0, -1, n])
            ),
            "true_vs_false": lambda logits, p, n, label: (
                    logits[0, -1, p] - logits[0, -1, n]
            )
        }

        if metric_type not in metrics_dict:
            raise ValueError(f"Metrica '{metric_type}' non trovata nel dizionario.")
        metric_function = metrics_dict[metric_type]

        for idx in tqdm(range(len(self.dataset)), desc=f"Attributions ({metric_type})"):
            fact, label, instance_id = self.dataset[idx]

            if self._is_already_processed(instance_id, is_attribution=True):
                continue

            _, tokens = self._prepare_inputs(fact, use_chat_template=True)

            with InspectOutputContext(self.llm, module_names, track_grads=True) as inspect:
                outputs = self.llm(input_ids=tokens["input_ids"], attention_mask=tokens.get("attention_mask"))

                # 3. Applichiamo la funzione passata matematicamente
                metric = metric_function(outputs.logits, pos_id, neg_id, label)

                self.llm.zero_grad()
                metric.backward()

            self.storage_manager.save_tensors_to_disk(inspect.catcher, instance_id, is_attribution=True,
                                                      target_dirs=target_dirs, save_last=False)

        self.storage_manager.combine_activations(is_attribution=True)

    # -------------
    # Helpers & Utilities
    # -------------
    def _generate_text_and_logits(self, tokens: Dict[str, torch.Tensor], formatted_prompt: str,
                                  instance_id: int) -> None:
        output = self.llm.generate(
            input_ids=tokens["input_ids"],
            max_new_tokens=self.max_new_tokens,
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

        # Usiamo i metodi statici dello StorageManager
        self.storage_manager.save_generation_output(
            gen_text, formatted_prompt, instance_id, self.storage_manager.dirs["generations"]
        )

        if hasattr(output, 'scores') and output.scores:
            self.storage_manager.save_model_logits(
                torch.stack(output.scores, dim=1), instance_id, self.storage_manager.dirs["logits"]
            )

    def _prepare_inputs(self, fact: str, use_chat_template: bool) -> Tuple[str, Dict[str, torch.Tensor]]:
        # Prendiamo il nome del dataset dallo storage manager
        if self.storage_manager.dataset_name == "entailmentbank":
            user_prompt = fact
        else:
            user_prompt = self.user_prompt_template.format(fact=fact)
            #user_prompt = self.USER_PROMPT_TEMPLATE.format(fact=fact) formato a prompt fisso

        if use_chat_template:
            # ut.build_messages rimane corretto perché vive in utils.py
            messages = ut.build_messages(system_prompt=self.system_prompt, user_prompt=user_prompt, k=0)
            #messages = ut.build_messages(system_prompt=self.SYSTEM_PROMPT, user_prompt=user_prompt, k=0) formato a prompt fisso
            tokens = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                        return_tensors="pt", return_dict=True)
            return self.tokenizer.decode(tokens["input_ids"][0]), {k: v.to(self.device) for k, v in tokens.items()}

        tokens = self.tokenizer(user_prompt, return_tensors="pt").to(self.device)
        return user_prompt, tokens

    def _is_already_processed(self, instance_id: int, is_attribution: bool) -> bool:
        """Controlla su disco se questo specifico elemento è già stato elaborato in precedenza."""
        if is_attribution:
            # Se stiamo calcolando le attribuzioni, verifichiamo se esiste il file del primo layer
            path = os.path.join(self.storage_manager.dirs["attr_hidden"], f"layer0-id{instance_id}.pt")
            return os.path.exists(path)
        else:
            # Per le estrazioni standard/chunked, verifichiamo se esiste già il testo generato
            path = os.path.join(self.storage_manager.dirs["generations"], f"generation_{instance_id}.json")
            return os.path.exists(path)