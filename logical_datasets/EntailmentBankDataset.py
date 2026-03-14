import os
import json
import random
import logging
import pandas as pd
import spacy
from typing import List, Dict, Tuple, Union
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class EntailmentBankDataset(Dataset):
    LABELS: Dict[int, str] = {0: "no", 1: "yes"}

    def __init__(self, project_root: str, label: Union[int, str] = "all", shuffle: bool = False,
                 neg_strategy: str = "negate") -> None:
        """
        Strategie mirate per l'analisi delle allucinazioni logiche:
        - "negate": Usa spaCy (NLP) per una negazione grammaticalmente perfetta del verbo principale.
        - "premise_deletion": Usa `core_concepts` per eliminare in modo chirurgico la regola fondamentale.
        - "distractor_injection": Usa `delete_list` per iniettare rumore altamente correlato (adversarial).
        - "shift": Scambia l'ipotesi con quella della riga successiva (baseline fuori contesto).
        """
        self.label = label
        self.neg_strategy = neg_strategy

        # Carica il modello spaCy (lo fa una sola volta per non rallentare l'esecuzione)
        try:
            self.nlp = spacy.load("en_core_web_trf")
        except OSError:
            logger.error("Modello spaCy non trovato. Esegui: python -m spacy download en_core_web_sm")
            raise

        self.all_data = self.get_dataset(project_root=project_root)
        self.dataset = self.format_dataset()
        if shuffle:
            self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[str, str, int]:
        item = self.dataset.iloc[idx]
        return item['text'], self.LABELS[item['label']], int(item['instance_id'])

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

    # --- NLP E DATA AUGMENTATION ---

    def _negate_hypothesis_spacy(self, text: str) -> str:
        """
        Esegue il Dependency Parsing avanzato tramite Transformer per trovare
        il verbo radice (ROOT) e i suoi ausiliari.
        """
        # 1. SCUDO ANTI-DOPPIO NEGATIVO (Logico, non sintattico)
        # Se la frase è già negativa, la invertiamo per farla diventare falsa nel contesto
        if any(neg in text.lower() for neg in
               [" not ", "n't", "cannot", "does not", "do not", "did not", "is not", "are not"]):
            pos_text = text.replace(" cannot ", " can ").replace("cannot ", "can ")
            pos_text = pos_text.replace(" does not ", " ").replace(" do not ", " ").replace(" did not ", " ")
            pos_text = pos_text.replace(" is not ", " is ").replace(" are not ", " are ")
            pos_text = pos_text.replace(" not ", " ")
            return " ".join(pos_text.split())

        doc = self.nlp(text)
        root = None
        for token in doc:
            if token.dep_ == "ROOT":
                root = token
                break

        if not root:
            return "It is false that " + text

        if root.pos_ in ["NOUN", "PROPN", "ADJ"]:
            return "It is incorrect that " + text

        # 2. RICERCA DEGLI AUSILIARI
        aux_token = None
        for child in root.children:
            if child.dep_ in ["aux", "auxpass"]:
                aux_token = child
                break

        # CASO A: C'è un ausiliare (can, may, is, was, has, will, ecc.)
        if aux_token:
            insert_idx = aux_token.idx + len(aux_token.text)

            # Gestione specifica di "can" per formare "cannot "
            if aux_token.text.lower() == "can":
                # Nessun .strip() qui, manteniamo lo spaziatura naturale di spaCy!
                return text[:aux_token.idx] + "cannot" + text[insert_idx:]

            return text[:insert_idx] + " not" + text[insert_idx:]

        # CASO B: Nessun ausiliare, è il verbo nudo.
        start_idx = root.idx
        end_idx = start_idx + len(root.text)

        if root.lemma_ == "be":
            return text[:end_idx] + " not" + text[end_idx:]

        else:
            if root.tag_ == "VBD":
                aux = "did not "
            elif root.tag_ == "VBP":
                aux = "do not "
            else:
                aux = "does not "

            verb = root.lemma_
            if root.text.istitle():
                aux = aux.capitalize()

            negated_verb_phrase = aux + verb

            # Applicata correttamente la spaziatura per evitare che verbo e avverbio si incollino
            return text[:start_idx] + negated_verb_phrase + " " + text[end_idx:].strip()

    def _get_adversarial_distractor(self, item: Dict) -> str:
        """Estrae un fatto fuorviante ma semanticamente affine dalla delete_list."""
        delete_list = item.get("meta", {}).get("delete_list", [])
        if delete_list:
            return random.choice(delete_list).get("fact", "")
        # Fallback se non ci sono distrattori nativi
        return random.choice(self.all_data).get("hypothesis", "")

    # --- CREAZIONE DEL DATASET ---

    def format_dataset(self) -> pd.DataFrame:
        records = []
        all_hypotheses = [item["hypothesis"] for item in self.all_data]
        total_hypotheses = len(all_hypotheses)

        for i, item in enumerate(self.all_data):
            triples = item.get("meta", {}).get("triples", {})
            original_hypothesis = item["hypothesis"]
            premises_str = ", ".join(triples.values())

            # 1. CASO POSITIVO (Vero)
            if self.label in [1, "all"]:
                pos_premises = premises_str

                # Se la strategia è distractor, testiamo la robustezza aggiungendo rumore
                if self.neg_strategy == "distractor_injection":
                    pos_premises += f" {self._get_adversarial_distractor(item)}"

                prompt_pos = f"Given these premises: {pos_premises}\nIs the following hypothesis true: {original_hypothesis}?"
                records.append({"original_id": f"{item['id']}_pos", "text": prompt_pos, "label": 1})

            # 2. CASO NEGATIVO (Falso / Neutro)
            if self.label in [0, "all"] and total_hypotheses > 1:
                neg_premises = premises_str
                hypothesis_false = original_hypothesis  # Default prima della manipolazione

                if self.neg_strategy == "shift":
                    hypothesis_false = all_hypotheses[(i + 1) % total_hypotheses]

                elif self.neg_strategy == "negate":
                    hypothesis_false = self._negate_hypothesis_spacy(original_hypothesis)

                elif self.neg_strategy == "premise_deletion":
                    # Eliminazione chirurgica basata sul core_concept
                    core_concepts = item.get("meta", {}).get("core_concepts", [])
                    keys_to_keep = list(triples.keys())

                    if core_concepts:
                        core_text = core_concepts[0].strip().lower()
                        target_key = next((k for k, v in triples.items() if v.strip().lower() == core_text), None)
                        if target_key:
                            keys_to_keep.remove(target_key)
                        elif len(keys_to_keep) > 0:
                            keys_to_keep.pop(random.randint(0, len(keys_to_keep) - 1))
                    elif len(keys_to_keep) > 0:
                        keys_to_keep.pop(random.randint(0, len(keys_to_keep) - 1))

                    if not keys_to_keep:
                        neg_premises = "No relevant context available."
                    else:
                        neg_premises = " ".join([triples[k] for k in keys_to_keep])
                    # L'ipotesi resta identica, il LLM allucina se risponde 1 (Vero)

                elif self.neg_strategy == "distractor_injection":
                    # Manteniamo la logica fallace per creare un negativo: un'ipotesi sbagliata + rumore avversario
                    hypothesis_false = all_hypotheses[(i + 1) % total_hypotheses]
                    neg_premises += f" {self._get_adversarial_distractor(item)}"

                prompt_neg = f"Given these premises: {neg_premises}\nIs the following hypothesis true: {hypothesis_false}?"
                records.append({"original_id": f"{item['id']}_neg", "text": prompt_neg, "label": 0})

        df = pd.DataFrame(records).drop_duplicates(subset=['original_id'])
        df['instance_id'] = range(len(df))
        return df