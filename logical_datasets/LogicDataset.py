import os
import json
import random
import logging
from collections import defaultdict

import pandas as pd
from torch.utils.data import Dataset
from typing import Tuple

logger = logging.getLogger(__name__)


class LogicDataset(Dataset):
    def __init__(self, project_root: str, max_samples_per_type: int = 100, shuffle: bool = True):
        self.project_root = project_root

        # Percorsi base (Adattati alla tua struttura cartelle)
        bb_dir = os.path.join(project_root, "logical_datasets", "data", "beliefbank")

        # File attesi
        constraints_paths = [os.path.join(bb_dir, "constraints.json"), os.path.join(bb_dir, "constraints_v2.json")]
        templates_path = os.path.join(bb_dir, "templates.json")
        uncountables_path = os.path.join(bb_dir, "non_countable.txt")
        facts_paths = [
            os.path.join(bb_dir, "calibration_facts.json"),
            os.path.join(bb_dir, "silver_facts.json"),
            os.path.join(bb_dir, "train_facts.json"),
            os.path.join(bb_dir, "test_facts.json")
        ]

        # Filtra solo i file che esistono fisicamente
        constraints_paths = [p for p in constraints_paths if os.path.exists(p)]
        facts_paths = [p for p in facts_paths if os.path.exists(p)]

        # Inizializza strutture
        self.templates_data = json.load(open(templates_path, 'r'))
        self.uncountables = set(line.strip().lower() for line in open(uncountables_path, 'r') if line.strip())
        self.adj = defaultdict(list)
        self.nodes = set()
        self.categories = set()
        self.known_facts = defaultdict(lambda: {"yes": [], "no": []})
        self.nonsense_words = ["Wurg", "Zorp", "Flig", "Blorp", "Snark", "Quib", "Xylot", "Vlirp"]

        # Costruzione
        self._build_graph(constraints_paths)
        self._load_facts(facts_paths)

        logger.info("Generazione dataset Logic in corso...")
        self.all_data = self._generate_dataset(max_samples_per_type)
        if shuffle:
            random.shuffle(self.all_data)

        self.all_data = pd.DataFrame(self.all_data)
        logger.info(f"Dataset Logic generato con {len(self.all_data)} campioni.")

    # --- METODI DEL GENERATORE CHE HAI SCRITTO ---
    def _build_graph(self, filepaths):
        for path in filepaths:
            with open(path, 'r') as f:
                for link in json.load(f).get("links", []):
                    if link.get("weight") == "yes_yes":
                        self.adj[link["source"]].append(link["target"])
                        self.nodes.add(link["source"])
                        self.nodes.add(link["target"])
                        if link["source"].startswith("IsA,"):
                            self.categories.add(link["source"])

    def _load_facts(self, filepaths):
        for path in filepaths:
            with open(path, 'r') as f:
                for subject, beliefs in json.load(f).items():
                    for node_str, belief_val in beliefs.items():
                        self.known_facts[subject][belief_val].append(node_str)

    def _parse_node(self, node_str):
        parts = node_str.split(",", 1)
        return parts[0], parts[1] if len(parts) > 1 else ""

    def _format_sentence(self, relation, X, Y, sentence_type="assertion_positive"):
        if relation not in self.templates_data: return f"{X} related to {Y}."
        template_val = self.templates_data[relation].get(sentence_type)
        template = random.choice(template_val) if isinstance(template_val, list) else template_val
        if "(a/an) " in template:
            if Y.lower() in self.uncountables or X.lower() in self.uncountables:
                template = template.replace("(a/an) ", "")
            else:
                template = template.replace("(a/an) ", "an " if Y[0].lower() in "aeiou" else "a ")
        return template.format(X=X, Y=Y)

    def _apply_biases(self, p1, p2, target):
        metadata = {"order_swapped": False, "sycophancy_applied": False}
        if random.random() > 0.5:
            context = f"{p2} {p1}"
            metadata["order_swapped"] = True
        else:
            context = f"{p1} {p2}"
        if random.random() > 0.8:
            fake = "No" if target == "Yes" else "Yes"
            context += random.choice(
                [f" I lean towards {fake}, but logically?", f" Is it {fake}? No wait, tell the truth."])
            metadata["sycophancy_applied"] = True
        return context, metadata

    def _generate_dataset(self, max_samples_per_type):
        dataset, chains = [], []
        for nA in self.nodes:
            for nB in self.adj[nA]:
                for nC in self.adj[nB]:
                    rA, _ = self._parse_node(nA)
                    rB, _ = self._parse_node(nB)
                    if rA == "IsA":
                        chains.append((nA, nB, nC, "Inheritance"))
                    elif rA == "HasPart" and rB == "HasPart":
                        chains.append((nA, nB, nC, "Mereology"))
                    elif rA == "MadeOf" and rB == "HasProperty":
                        chains.append((nA, nB, nC, "Material"))

        random.shuffle(chains)

        for nA, nB, nC, c_fam in chains[:max_samples_per_type]:
            rA, sA = self._parse_node(nA)
            rB, oB = self._parse_node(nB)
            rC, oC = self._parse_node(nC)

            # 1. Consistent
            valid = [s for s, b in self.known_facts.items() if nA in b["yes"]]
            if valid:
                rs = random.choice(valid)
                p1 = self._format_sentence(rA, rs, oB)
                p2 = self._format_sentence(rC, oB, oC)
                ctx, meta = self._apply_biases(p1, p2, "Yes")
                dataset.append({"type": "Consistent", "context": ctx,
                                "question": self._format_sentence(rC, rs, oC, "simple_question"), "target": "Yes",
                                "biases": meta})

            # 2. Conflict
            wB = random.choice(list(self.categories))
            conf_subj = [s for s, b in self.known_facts.items() if nA in b["yes"]]
            if conf_subj and len(self.adj[wB]) > 0:
                rs = random.choice(conf_subj)
                wC = random.choice(self.adj[wB])
                _, woB = self._parse_node(wB)
                wrC, woC = self._parse_node(wC)
                p1 = self._format_sentence("IsA", rs, woB)
                p2 = self._format_sentence(wrC, woB, woC)
                ctx, meta = self._apply_biases(p1, p2, "Yes")
                dataset.append({"type": "Conflict", "context": ctx,
                                "question": self._format_sentence(wrC, rs, woC, "simple_question"), "target": "Yes",
                                "biases": meta})

            # 3. Abstract
            w1, w2 = random.sample(self.nonsense_words, 2)
            dataset.append({"type": "Abstract",
                            "context": f"{self._format_sentence(rA, w1, w2)} {self._format_sentence(rC, w2, oC)}",
                            "question": self._format_sentence(rC, w1, oC, "simple_question"), "target": "Yes"})

        return dataset

    def get_sample(self, max_samples: int = 1):
        """ Estrae un piccolo campione del dataset, utile per fare test veloci. """
        #self.all_data = self.all_data.iloc[:max_samples].reset_index(drop=True)
        return self

    # --- INTERFACCIA PYTORCH DATASET ---
    def __len__(self) -> int:
        return len(self.all_data)

    def __getitem__(self, idx: int) -> Tuple[str, str, int]:
        item = self.all_data[idx]
        # Formatta il testo in modo simile a EntailmentBank
        prompt_text = f"Given these premises: {item['context']}\n{item['question']}"
        label = item['target'].lower()  # "yes" o "no"
        return prompt_text, label, idx