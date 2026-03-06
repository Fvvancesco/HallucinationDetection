import os
import json
import string
import pandas as pd
from torch.utils.data import Dataset

LABELS = ["no", "yes"]


class EntailmentBankDataset(Dataset):
    def __init__(self, project_root, recreate_ids=True, label=0):
        self.label = label
        self.all_data = self.get_dataset(project_root=project_root)
        self.dataset = self.format_dataset()
        # Create instance ids
        if ('instance_id' not in self.dataset.columns) or recreate_ids:
            self.dataset = self.create_instance_ids()


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        item = self.dataset.iloc[idx]

        id = item['instance_id'].item()
        reasoning = item['reasoning']
        
        label = item['label'].item()
        computed_label = LABELS[label] if self.label == 0 else LABELS[1 - label]

        return reasoning, computed_label, id

        
    def get_dataset(self, project_root):
        all_data = []
        data_dir = os.path.join(project_root, "data", "entailmentbank")

        for file in os.listdir(data_dir):
            if file.endswith(".jsonl"):
                d_entailmentbank = [json.loads(s) for s in list(open(os.path.join(data_dir, file)))]
                all_data.extend(d_entailmentbank)    

        return all_data
    

    def format_dataset(self):
        records = []

        for item in self.all_data:
            records.append({
                "reasoning": self.format_item(item),
                "label": 1  # All instances are positive examples
            })

        return pd.DataFrame(records).drop_duplicates()


    def format_item(self, item):
        reasoning = item["full_text_proof"]
        triples = item["meta"]["triples"]
        intermediate_conclusions = item["meta"]["intermediate_conclusions"]
        sentences = {**triples, **intermediate_conclusions}
        
        for key, value in sentences.items():
            reasoning = reasoning.replace(f"{key}: ", value)
            reasoning = reasoning.replace(key, value)

        return reasoning


    def get_language_by_instance_id(self, instance_id):
        return "EN"  # BeliefBank is in English, so we return "EN" directly

    
    def create_instance_ids(self):
        instance_ids = list(range(len(self.dataset)))

        if "instance_id" in self.dataset.columns:
            self.dataset = self.dataset.drop(columns="instance_id")

        self.dataset = self.dataset.assign(instance_id=instance_ids)

        return self.dataset
