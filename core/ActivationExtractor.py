import torch
import logging
from tqdm import tqdm
from typing import List, Dict, Tuple, Any

from utils import Utils as ut
from utils.InspectOutputContext import InspectOutputContext

logger = logging.getLogger(__name__)

class ActivationExtractor:
    def __init__(self, llm: Any, tokenizer: Any, dataset: Any, storage_manager: Any, system_prompt: str, user_prompt_template: str, max_new_tokens: int = 5):
        self.llm = llm
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.storage_manager = storage_manager  # Iniettiamo il manager dell'I/O
        self.device = self.llm.device
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
            formatted_prompt, tokens = self._prepare_inputs(fact, use_chat_template)

            with InspectOutputContext(self.llm, module_names, save_generation=True,
                                      save_dir=self.storage_manager.dirs["generations"]) as inspect:
                self._generate_text_and_logits(tokens, formatted_prompt, instance_id)

            # Deleghiamo il salvataggio allo Storage Manager
            self.storage_manager.save_tensors_to_disk(inspect.catcher, instance_id, is_attribution=False, save_last=False)

        self.storage_manager.combine_activations()

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

            self.storage_manager.save_tensors_to_disk(inspect.catcher, instance_id, is_attribution=False, save_last=False)

        self.storage_manager.combine_activations()

    def save_attributions_and_grads(self) -> None:
        logger.info(f"{'--' * 25} Saving Attributions (Act x Grad) {'--' * 25}")
        torch.set_grad_enabled(True)
        self.llm.eval()
        for param in self.llm.parameters():
            param.requires_grad = False

        if not self.dataset:
            raise ValueError("Dataset non caricato.")

        module_names = self._get_target_modules()

        # Rinomino per chiarezza: questi sono i token per "TRUE" e "FALSE"
        token_true = self.tokenizer.encode(" TRUE", add_special_tokens=False)[-1]
        token_false = self.tokenizer.encode(" FALSE", add_special_tokens=False)[-1]

        target_dirs = {
            "hidden": self.storage_manager.dirs["attr_hidden"],
            "mlp": self.storage_manager.dirs["attr_mlp"],
            "attn": self.storage_manager.dirs["attr_attn"]
        }

        for idx in tqdm(range(len(self.dataset)), desc="Computing attributions"):
            # 1. Ora estraiamo la 'label' (la verità) invece di ignorarla con '_'
            fact, label, instance_id = self.dataset[idx]
            _, tokens = self._prepare_inputs(fact, use_chat_template=True)

            # 2. Definiamo qual è l'allucinazione in base alla ground truth
            # Nei tuoi dataset, le label sono stringhe "yes" o "no"
            if str(label).lower() == "yes":
                correct_token = token_true
                hallucinated_token = token_false
            else:
                correct_token = token_false
                hallucinated_token = token_true

            with InspectOutputContext(self.llm, module_names, track_grads=True) as inspect:
                outputs = self.llm(input_ids=tokens["input_ids"], attention_mask=tokens.get("attention_mask"))

                # 3. NUOVA METRICA: Logit(Allucinazione) - Logit(Risposta Corretta)
                metric = outputs.logits[0, -1, hallucinated_token] - outputs.logits[0, -1, correct_token]

                self.llm.zero_grad()
                metric.backward()

            self.storage_manager.save_tensors_to_disk(inspect.catcher, instance_id, is_attribution=True,
                                                      target_dirs=target_dirs,
                                                      save_last=False)

        self.storage_manager.combine_activations(is_attribution=True)

    def save_attributions_and_grads_true_vs_false(self) -> None:
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

        target_dirs = {
            "hidden": self.storage_manager.dirs["attr_hidden"],
            "mlp": self.storage_manager.dirs["attr_mlp"],
            "attn": self.storage_manager.dirs["attr_attn"]
        }

        for idx in tqdm(range(len(self.dataset)), desc="Computing attributions"):
            fact, _, instance_id = self.dataset[idx]
            _, tokens = self._prepare_inputs(fact, use_chat_template=True)

            with InspectOutputContext(self.llm, module_names, track_grads=True) as inspect:
                outputs = self.llm(input_ids=tokens["input_ids"], attention_mask=tokens.get("attention_mask"))
                metric = outputs.logits[0, -1, pos_token] - outputs.logits[0, -1, neg_token]

                self.llm.zero_grad()
                metric.backward()

            self.storage_manager.save_tensors_to_disk(inspect.catcher, instance_id, is_attribution=True, target_dirs=target_dirs,
                                       save_last=False)

        self.storage_manager.combine_activations(is_attribution=True)

    def save_activations_chunked(self, use_chat_template: bool = True, chunk_size: int = 1000) -> None:
        """Versione I/O ottimizzata: usa la RAM per raggruppare migliaia di attivazioni prima di salvare."""
        module_names = self._get_target_modules()
        if not self.dataset:
            raise ValueError("Dataset non caricato.")

        # Diciamo allo storage manager di preparare la RAM
        self.storage_manager.init_buffers()
        chunk_idx = 0

        for idx in tqdm(range(len(self.dataset)), desc="Saving activations (Chunked)"):
            fact, _, instance_id = self.dataset[idx]
            formatted_prompt, tokens = self._prepare_inputs(fact, use_chat_template)

            with InspectOutputContext(self.llm, module_names, save_generation=True,
                                      save_dir=self.storage_manager.dirs["generations"]) as inspect:
                self._generate_text_and_logits(tokens, formatted_prompt, instance_id)

            # Inviamo i tensori in RAM invece che su disco
            self.storage_manager.bufferize_tensors(inspect.catcher, instance_id, is_attribution=False)

            if (idx + 1) % chunk_size == 0:
                self.storage_manager.flush_buffer_to_disk(chunk_idx, prefix="activation")
                chunk_idx += 1
                self.storage_manager.init_buffers()

        # Flush finale per gli elementi rimasti
        if self.storage_manager.instance_ids_buffer[self.target_layers[0]]:
            self.storage_manager.flush_buffer_to_disk(chunk_idx, prefix="activation")

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