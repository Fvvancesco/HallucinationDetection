import os
import json
import logging
import pandas as pd
from typing import List, Dict, Tuple, Union
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class EntailmentBankDataset(Dataset):
    LABELS: Dict[int, str] = {0: "no", 1: "yes"}

    def __init__(self, project_root: str, label: Union[int, str] = "all", shuffle: bool = False) -> None:
        """
        Inizializza il dataset EntailmentBank.
        :param label: 1 per solo veri, 0 per solo falsi generati, "all" per entrambi.
        """
        self.label = label
        self.all_data = self.get_dataset(project_root=project_root)
        self.dataset = self.format_dataset()

        if shuffle:
            self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[str, str, str]:
        item = self.dataset.iloc[idx]

        instance_id: str = item['instance_id']
        text: str = item['text']
        label_val: int = item['label']

        computed_label: str = self.LABELS[label_val]

        return text, computed_label, instance_id


    def get_dataset(self, project_root: str) -> List[Dict]:
        all_data = []
        data_dir = os.path.join(project_root, "logical_datasets", "data", "entailmentbank")

        if not os.path.exists(data_dir):
            logger.warning(f"⚠️ Cartella dataset non trovata: {data_dir}")
            return all_data

        for file in os.listdir(data_dir):
            if file.endswith(".jsonl"):
                with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
                    all_data.extend([json.loads(s) for s in f])

        return all_data


    def format_dataset(self) -> pd.DataFrame:
        records = []

        # Estraiamo tutte le ipotesi per creare gli esempi negativi
        all_hypotheses = [item["hypothesis"] for item in self.all_data]
        # Shifting: l'ipotesi dell'elemento i-esimo diventa quella dell'elemento i+1
        shifted_hypotheses = all_hypotheses[1:] + [all_hypotheses[0]] if all_hypotheses else []

        for i, item in enumerate(self.all_data):
            triples = item.get("meta", {}).get("triples", {})
            # Uniamo i fatti in una premessa unica
            premises = " ".join(triples.values())

            # --- 1. ESEMPI POSITIVI (Label 1) ---
            if self.label in [1, "all"]:
                hypothesis_true = item["hypothesis"]
                prompt_pos = f"Given these premises: {premises}\nIs the following hypothesis true: {hypothesis_true}?"

                records.append({
                    "instance_id": f"{item['id']}_pos",
                    "text": prompt_pos,
                    "label": 1
                })

            # --- 2. ESEMPI NEGATIVI (Label 0) ---
            if self.label in [0, "all"] and shifted_hypotheses:
                hypothesis_false = shifted_hypotheses[i]
                prompt_neg = f"Given these premises: {premises}\nIs the following hypothesis true: {hypothesis_false}?"

                records.append({
                    "instance_id": f"{item['id']}_neg",
                    "text": prompt_neg,
                    "label": 0
                })

        return pd.DataFrame(records).drop_duplicates(subset=['instance_id'])

    def get_language_by_instance_id(self, instance_id: str) -> str:
        return "EN"