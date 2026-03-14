import os
import logging
import torch
from typing import Union, Optional, Any

from utils import Utils as ut
from core.StorageManager import StorageManager
from core.ActivationExtractor import ActivationExtractor
from probing.ProberEvaluator import ProberEvaluator
from logical_datasets.BeliefBankDataset import BeliefBankDataset
from config.prompts import PROMPT_REGISTRY

logger = logging.getLogger(__name__)

class HallucinationPipeline:
    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = None
        self.dataset_name = None
        self.llm = None
        self.tokenizer = None
        self.llm_name = None
        self.storage_manager = None
        self.extractor = None

    def load_dataset(self, dataset_name: str, label: Union[int, str] = 1) -> None:
        logger.info(f"Loading dataset {dataset_name}...")
        self.dataset_name = dataset_name
        if dataset_name in ["beliefbank", "belief"]:
            self.dataset = BeliefBankDataset(project_root=self.project_dir, data_type="constraints", label=label)
        elif dataset_name == "entailmentbank":
            from logical_datasets.EntailmentBankDataset import EntailmentBankDataset
            self.dataset = EntailmentBankDataset(project_root=self.project_dir, label=label)
        elif dataset_name in ["logic"]:
            from logical_datasets.LogicDataset import LogicDataset
            self.dataset = LogicDataset(project_root=self.project_dir, max_samples_per_type=200)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported.")

    def load_llm(self, llm_name: str, prompt_id: str = "base_v1", use_local: bool = False) -> None:
        logger.info(f"Loading LLM {llm_name} with Prompt: {prompt_id}...")
        self.llm_name = llm_name
        self.tokenizer = ut.load_tokenizer(llm_name, local=use_local)
        self.llm = ut.load_llm(llm_name, ut.create_bnb_config(), local=use_local)

        num_layers = getattr(self.llm.config, "num_hidden_layers", 32)
        target_layers = list(range(num_layers))

        # Estraiamo i dati del prompt
        prompt_data = PROMPT_REGISTRY.get(prompt_id, PROMPT_REGISTRY["base_v1"])

        self.storage_manager = StorageManager(self.project_dir, self.llm_name, self.dataset_name, target_layers, prompt_id=prompt_id)
        self.storage_manager.setup_directories()

        #self.extractor = ActivationExtractor(self.llm, self.tokenizer, self.dataset, self.storage_manager,
                                             #system_prompt=prompt_data["system"],
                                             #user_prompt_template=prompt_data["user"])

        self.extractor = ActivationExtractor(
            self.llm, self.tokenizer, self.dataset, self.storage_manager,
            system_prompt=prompt_data["system"],
            user_prompt_template=prompt_data["user"],
            pos_token_str=prompt_data.get("pos_token", " TRUE"),
            neg_token_str=prompt_data.get("neg_token", " FALSE")
        )

    def run_extraction(self, method_name: str, **kwargs) -> None:
        if not self.extractor: raise RuntimeError("LLM non caricato.")
        getattr(self.extractor, method_name)(**kwargs)

    def train_probers(self, llm_name: str, prompt_id: str = "base_v1", test_size: float = 0.2, epochs: int = 30) -> None:
        target_layers = self.extractor.target_layers if self.extractor else list(range(32))
        evaluator = ProberEvaluator(self.project_dir, self.dataset, self.dataset_name, target_layers, prompt_id=prompt_id)
        evaluator.train_and_evaluate_probers(llm_name=llm_name, test_size=test_size, epochs=epochs)

    def predict_prober(self, target: str, layer: int, llm_name: str, prompt_id: str = "base_v1", label: int = 1) -> None:
        target_layers = self.extractor.target_layers if self.extractor else list(range(32))
        evaluator = ProberEvaluator(self.project_dir, self.dataset, self.dataset_name, target_layers, prompt_id=prompt_id)
        evaluator.predict_prober(target=target, layer=layer, llm_name=llm_name, label=label)