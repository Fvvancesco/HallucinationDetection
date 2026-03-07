import os
import json
import string
import random
import pandas as pd
from typing import List
from torch.utils.data import Dataset

SPLITS = ["calibration", "silver"]
LABELS = ["no", "yes"]


# ===================================================================================
# Sostituto per la vecchia libreria `torchdataset`
# ===================================================================================
class TorchDataset(Dataset):
    """ Wrappa una semplice lista in un Dataset PyTorch nativo """

    def __init__(self, data: List[dict]):
        """ Inizializza il dataset salvando la lista di dati grezzi. """
        self.data = data

    def __len__(self):
        """ Restituisce il numero totale di elementi nel dataset. Requisito di PyTorch. """
        return len(self.data)

    def __getitem__(self, idx):
        """ Restituisce l'elemento alla posizione 'idx'. Requisito di PyTorch. """
        return self.data[idx]


# ===================================================================================
# Main Dataset Class
# ===================================================================================
class BeliefBankDataset(Dataset):
    """ Classe principale che coordina il caricamento e la formattazione dei dati di BeliefBank. """

    def __init__(self, project_root, recreate_ids=True, data_type="facts", label=0, shuffle=False):
        """
        Costruttore del dataset.
        - project_root: Percorso della cartella principale del progetto.
        - recreate_ids: Se True, rigenera gli ID univoci per le istanze.
        - data_type: "facts" per le frasi dirette, "constraints" per le regole logiche.
        - label: Determina come mappare l'etichetta (0 per default).
        - shuffle: Se True, mischia i dati casualmente.
        """
        # Carica i dati grezzi dai file JSON
        self.all_data = self.get_dataset(project_root=project_root)
        # Formatta i dati in un DataFrame Pandas in base al tipo richiesto (fatti o vincoli logici)
        self.dataset = self.format_dataset(data_type=data_type)
        self.label = label

        # Crea gli ID univoci per ogni riga se non esistono
        if ('instance_id' not in self.dataset.columns) or recreate_ids:
            self.dataset = self.create_instance_ids()

        # Mischia i dati se richiesto
        if shuffle:
            self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        """ Restituisce il numero totale di frasi pronte per l'addestramento. """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Estrae un singolo esempio di addestramento.
        Restituisce una tupla: (Testo della frase, Etichetta "yes"/"no", ID istanza)
        """
        item = self.dataset.iloc[idx]

        id = item['instance_id'].item()
        fact = item['fact']

        # Converte l'etichetta numerica (0 o 1) nella stringa corrispondente ("no" o "yes")
        label = item['belief'].item()
        computed_label = LABELS[label] if self.label == 0 else LABELS[1 - label] #inverts label, idk why

        return fact, computed_label, id

    def get_sample(self, max_samples: int = 1):
        """ Estrae un piccolo campione del dataset, utile per fare test veloci. """
        self.dataset = self.dataset.iloc[:max_samples].reset_index(drop=True)
        return self

    def get_dataset(self, project_root):
        """
        Legge fisicamente i file JSON dal disco e divide i dati in
        set di Addestramento (train), Validazione (val) e Test.
        """
        # Costruisce i percorsi assoluti dei file
        constraints_path = os.path.join(project_root, "logical_datasets", "data", "beliefbank", "constraints_v2.json")
        calibration_facts_path = os.path.join(project_root, "logical_datasets", "data", "beliefbank", "calibration_facts.json")
        silver_facts_path = os.path.join(project_root, "logical_datasets", "data", "beliefbank", "silver_facts.json")

        # Inizializza le classi di supporto per gestire i file JSON
        constraints = Constraints(project_root=project_root, constraints_path=constraints_path)
        silver_facts = Facts(project_root=project_root, constraints=constraints, facts_path=silver_facts_path)
        calibration_facts = Facts(project_root=project_root, constraints=constraints, facts_path=calibration_facts_path)

        # Ottiene i dati già divisi in Train/Test/Val
        silver_splits = silver_facts.get_splits()
        calibration_splits = calibration_facts.get_splits()

        # Raggruppa tutto in un grosso dizionario formattando le liste tramite TorchDataset
        train_constraints = TorchDataset(
            constraints.get_grounded_constraints(facts=calibration_splits["train"], path=constraints_path))
        train_calibration_facts = TorchDataset(calibration_splits["train"])
        train_silver_facts = TorchDataset(silver_splits["train"])

        val_constraints = TorchDataset(
            constraints.get_grounded_constraints(facts=calibration_splits["val"], path=constraints_path))
        val_calibration_facts = TorchDataset(calibration_splits["val"])

        test_calibration_facts = TorchDataset(calibration_splits["test"])
        test_silver_facts = TorchDataset(silver_splits["test"])

        return {
            "constraints": {
                "train": train_constraints,
                "all": constraints,
            },
            "facts": {
                "calibration": {
                    "train": train_calibration_facts,
                    "test": test_calibration_facts,
                    "val": val_calibration_facts,
                    "complete": TorchDataset(calibration_facts.get_whole_set()),
                },
                "silver": {
                    "train": train_silver_facts,
                    "test": test_silver_facts,
                    "complete": TorchDataset(silver_facts.get_whole_set())
                }
            }
        }

    def format_dataset(self, data_type):
        """ Instrada la formattazione a seconda se vogliamo estrarre i Fatti o le Regole Logiche. """
        if data_type == "facts":
            return self.format_facts()
        elif data_type == "constraints":
            return self.format_constraints()
        else:
            raise NotImplementedError(f"Data type {data_type} not implemented.")

    def format_facts(self):
        """
        Prende i fatti estratti da "calibration" e "silver", li unisce in un DataFrame Pandas
        e genera le versioni negate (data augmentation) per raddoppiare i dati.
        """
        dataset = []
        for split in SPLITS:
            dataset.extend(self.all_data["facts"][split]["complete"].data)
        dataset = pd.DataFrame(dataset)

        return BeliefBankDataset.extend_with_negated_facts(dataset)

    def format_constraints(self):
        """
        Formatta i vincoli logici (es. "Se X è un uccello, allora X sa volare").
        Trasforma le regole logiche astratte in vere e proprie stringhe in inglese.
        """
        dataset = []

        for row in self.all_data["constraints"]["train"].data:
            # Crea la stringa per l'implicazione positiva
            fact, belief = BeliefBankDataset.get_implication(row)
            dataset.append({"fact": fact, "belief": int(belief)})

            # Crea la stringa per l'implicazione con conseguente negato
            neg_fact, neg_belief = BeliefBankDataset.get_negated_implication(row)
            dataset.append({"fact": neg_fact, "belief": int(neg_belief)})

        return pd.DataFrame(dataset)

    @staticmethod
    def get_implication(instance):
        """
        Costruisce una frase condizionale in inglese "If [antecedente], then [conseguente]".
        Calcola anche la verità (label) dell'implicazione usando la logica booleana (non A oppure B).
        """
        antecedent = BeliefBankDataset.remove_punc(instance["antecedent"])
        consequent = BeliefBankDataset.remove_punc(instance["consequent"])
        ant_label = bool(instance["s_antecedent"])
        con_label = bool(instance["s_consequent"])

        implication_string = f"If {antecedent}, then {consequent}"
        implication_label = not ant_label or con_label

        return implication_string, implication_label

    @staticmethod
    def get_negated_implication(instance):
        """ Simile a get_implication, ma costruisce la frase negando il conseguente. """
        antecedent = BeliefBankDataset.remove_punc(instance["antecedent"])
        neg_consequent = BeliefBankDataset.remove_punc(instance["neg_consequent"])
        ant_label = bool(instance["s_antecedent"])
        neg_con_label = not bool(instance["s_consequent"])

        implication_string = f"If {antecedent}, then {neg_consequent}"
        implication_label = not ant_label or neg_con_label

        return implication_string, implication_label

    @staticmethod
    def remove_punc(text):
        """ Utility per rimuovere la punteggiatura da una stringa. """
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def get_language_by_instance_id(self, instance_id):
        """ Ritorna la lingua del dataset. """
        return "EN"

    def create_instance_ids(self):
        """ Crea una colonna 'instance_id' con un numero sequenziale univoco per ogni riga. """
        instance_ids = list(range(len(self.dataset)))
        if "instance_id" in self.dataset.columns:
            self.dataset = self.dataset.drop(columns="instance_id")
        self.dataset = self.dataset.assign(instance_id=instance_ids)
        return self.dataset

    @staticmethod
    def extend_with_negated_facts(dataset):
        """
        Data Augmentation: per ogni riga (es. "Un cane è un animale" -> yes),
        prende la versione negata generata precedentemente ("Un cane non è un animale")
        e la aggiunge al dataset con l'etichetta invertita ("no").
        """
        negated_df = dataset.copy()
        negated_df["fact"] = dataset["negated_fact"]
        negated_df["belief"] = 1 - dataset["belief"]

        dataset = dataset.drop(columns=["negated_fact"])
        negated_df = negated_df.drop(columns=["negated_fact"])

        dataset = pd.concat([dataset, negated_df], ignore_index=True)
        return dataset


# ===================================================================================
# Utility classes for BeliefBank dataset
# ===================================================================================
class Constraints(Dataset):
    """ Gestisce il caricamento e la logica del file constraints_v2.json. """

    def __init__(self, project_root, constraints_path: str):
        self.project_root = project_root
        # Estrae i link logici (es. IsA,dog -> IsA,animal)
        self.samples = Constraints.get_links(path=constraints_path)

    def __len__(self):
        """ Restituisce il numero totale di vincoli logici estratti. Requisito PyTorch. """
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """ Restituisce il vincolo logico alla posizione specificata. Requisito PyTorch. """
        return self.samples[idx]

    def get_grounded_constraints(self, facts: dict, path: str) -> List[object]:
        """
        'Mette a terra' i vincoli astratti. Incrocia le regole logiche generali
        (es. Uccello->Vola) con i fatti concreti estratti sui soggetti (es. Albatro è un Uccello),
        generando le 4 varianti testuali (antecedenti e conseguenti, positivi e negativi).
        """
        hash_facts = dict()
        for fact in facts: hash_facts.setdefault(fact["predicate"], dict())[fact["subject"]] = fact

        templates, uncountables = Facts.get_language_templates(
            templates_path=os.path.join(self.project_root,"logical_datasets", "data", "beliefbank", "templates.json"),
            uncountables_path=os.path.join(self.project_root,"logical_datasets", "data", "beliefbank", "non_countable.txt"))

        general_constraints = Constraints.get_links(path)
        samples = []
        for constraint in general_constraints:
            knows_antecedent = constraint["antecedent"] in hash_facts.keys()
            knows_consequent = constraint["consequent"] in hash_facts.keys()

            if knows_antecedent:
                for subj, belief in hash_facts[constraint["antecedent"]].items():
                    rel, obj = constraint["consequent"].split(",")
                    sample = {
                        "antecedent": belief["fact"],
                        "neg_antecedent": belief["negated_fact"],
                        "consequent": Facts.implication2string(templates=templates, uncountables=uncountables,
                                                               subject=subj, relation=rel, symbol=True, obj=obj),
                        "neg_consequent": Facts.implication2string(templates=templates, uncountables=uncountables,
                                                                   subject=subj, relation=rel, symbol=False, obj=obj),
                        "s_antecedent": int(constraint["s_antecedent"]),
                        "s_consequent": int(constraint["s_consequent"]),
                        "g_antecedent": belief["belief"],
                        "g_consequent": -1
                    }
                    if knows_consequent and subj in hash_facts[constraint["consequent"]]:
                        sample["g_consequent"] = hash_facts[constraint["consequent"]][subj]["belief"]
                    samples.append(sample)
            elif knows_consequent:
                for subj, belief in hash_facts[constraint["consequent"]].items():
                    rel, obj = constraint["antecedent"].split(",")
                    sample = {
                        "consequent": belief["fact"],
                        "neg_consequent": belief["negated_fact"],
                        "antecedent": Facts.implication2string(templates=templates, uncountables=uncountables,
                                                               subject=subj, relation=rel, symbol=True, obj=obj),
                        "neg_antecedent": Facts.implication2string(templates=templates, uncountables=uncountables,
                                                                   subject=subj, relation=rel, symbol=False, obj=obj),
                        "s_antecedent": int(constraint["s_antecedent"]),
                        "s_consequent": int(constraint["s_consequent"]),
                        "g_antecedent": -1,
                        "g_consequent": belief["belief"]
                    }
                    samples.append(sample)
        return samples

    @staticmethod
    def get_links(path) -> (List[object]):
        """ Legge il file constraints_v2.json ed estrae le relazioni direzionali (source -> target). """
        with open(path) as f:
            constraints = json.load(f)
        links = []
        for rel in constraints["links"]:
            if rel["direction"] == "forward":
                source = rel["source"]
                source_symbol = rel["weight"].split("_")[0] == "yes"
                target = rel["target"]
                target_symbol = rel["weight"].split("_")[1] == "yes"
                sample = {"antecedent": source, "consequent": target, "s_antecedent": source_symbol,
                          "s_consequent": target_symbol}
                links.append(sample)
        return links


class Facts():
    """ Gestisce la conversione delle triple (Soggetto, Relazione, Oggetto) in frasi grammaticalmente corrette. """

    def __init__(self, project_root, facts_path, constraints):
        self.project_root = project_root
        self.constraints = constraints
        self.facts_path = facts_path

    def get_whole_set(self) -> dict:
        """ Estrae tutti i fatti (soggetto, predicato) e genera sia la frase vera che quella negata. """
        templates, uncountables = Facts.get_language_templates(
            templates_path=os.path.join(self.project_root, "logical_datasets","data", "beliefbank", "templates.json"),
            uncountables_path=os.path.join(self.project_root, "logical_datasets","data", "beliefbank", "non_countable.txt"))
        with open(self.facts_path) as f:
            facts = json.load(f)
            samples = []
            for subject, subject_facts in facts.items():
                for key, belief in subject_facts.items():
                    relation, obj = key.split(",")
                    # Genera la stringa inglese positiva
                    fact = Facts.implication2string(templates=templates, uncountables=uncountables, subject=subject,
                                                    relation=relation, symbol=True, obj=obj)
                    # Genera la stringa inglese negativa
                    negated_fact = Facts.implication2string(templates=templates, uncountables=uncountables,
                                                            subject=subject, relation=relation, symbol=False, obj=obj)
                    sample = {"subject": subject, "predicate": key, "fact": fact, "negated_fact": negated_fact,
                              "belief": int(belief == "yes")}
                    samples.append(sample)
        return samples

    def get_splits(self) -> dict:
        """
        Divide i fatti in Train, Val (10% del train) e Test in modo intelligente.
        I fatti che sono solo 'conseguenze' (target) finiscono nel Test set.
        I fatti che sono 'cause' (source) finiscono nel Train set.
        Questo serve a valutare se il modello, imparando le cause, sa dedurre le conseguenze.
        """

        templates, uncountables = Facts.get_language_templates(
            templates_path=os.path.join(self.project_root,"logical_datasets", "data", "beliefbank", "templates.json"),
            uncountables_path=os.path.join(self.project_root, "logical_datasets","data", "beliefbank", "non_countable.txt")
        )

        with open(self.facts_path) as f:
            facts = json.load(f)
            c_antecedents = [c["antecedent"] for c in self.constraints]
            c_consequents = [c["consequent"] for c in self.constraints]
            ant_facts = []
            con_facts = []

            for subject, subject_facts in facts.items():
                for key, belief in subject_facts.items():
                    relation, obj = key.split(",")
                    fact = Facts.implication2string(templates=templates, uncountables=uncountables, subject=subject,
                                                    relation=relation, symbol=True, obj=obj)
                    negated_fact = Facts.implication2string(templates=templates, uncountables=uncountables,
                                                            subject=subject, relation=relation, symbol=False, obj=obj)
                    sample = {"subject": subject, "predicate": key, "fact": fact, "negated_fact": negated_fact,
                              "belief": int(belief == "yes")}

                    # Se è un target ma NON una source, va nel set di Test
                    if key in c_consequents and key not in c_antecedents:
                        con_facts.append(sample)
                    else:
                        ant_facts.append(sample)

        # Prende il 10% casuale del set di training (ant_facts) per usarlo come Validazione
        idx_val = random.sample(range(0, len(ant_facts) - 1), int(0.1 * len(ant_facts)))
        train_ant_facts = [ant_facts[idx] for idx in range(len(ant_facts)) if idx not in idx_val]
        val_ant_facts = [ant_facts[idx] for idx in idx_val]

        return {"train": train_ant_facts, "test": con_facts, "val": val_ant_facts}

    @staticmethod
    def implication2string(templates, uncountables, subject, relation, symbol, obj):
        """
        Prende soggetto e oggetto, corregge gli articoli con noun_fluenterer,
        e li inserisce all'interno del template corrispondente alla relazione (es. {X} is {Y}).
        """
        this_template = templates[relation]
        X = Facts.noun_fluenterer(subject, uncountables)
        Y = Facts.noun_fluenterer(obj, uncountables, relation)

        if symbol:
            nl_question = this_template['assertion_positive'].format(X=X, Y=Y)
        else:
            nl_question = this_template['assertion_negative'].format(X=X, Y=Y)
        return nl_question

    @staticmethod
    def get_language_templates(templates_path, uncountables_path):
        """ Carica in memoria i file JSON/TXT con i template per le frasi e la lista delle parole non numerabili. """
        with open(templates_path) as f:
            natural_relations = json.load(f)
        with open(uncountables_path) as f:
            uncountables = f.read().split('\n')
        return natural_relations, uncountables

    @staticmethod
    def noun_fluenterer(noun, uncountables, relation=None):
        """
        Funzione grammaticale geniale che applica gli articoli ai nomi in inglese.
        1. Se la parola è non numerabile (es. "water"), non aggiunge articoli.
        2. Se la relazione indica un'azione o proprietà astratta, non aggiunge articoli.
        3. Se inizia per vocale aggiunge "an ".
        4. Altrimenti aggiunge "a ".
        """
        if noun in uncountables:
            return noun

        if relation is not None:
            if relation in ['CapableOf', 'MadeOf', 'HasProperty']:
                return noun

        if noun[0] in ['a', 'e', 'i', 'o', 'u']:
            return 'an ' + noun

        return 'a ' + noun

    def get_multihop_splits(self) -> dict:
        """
        Divide i fatti in Train e Test per testare il "Multihop reasoning" (ragionamento a più salti).
        Cerca i fatti il cui predicato è sia un 'conseguente' di una regola che un 'antecedente' di un'altra.
        Esempio: Cane -> Mammifero -> Animale.
        Se un fatto si trova "in mezzo" a questa catena, va nel Test set per vedere se il modello
        riesce a dedurre l'intera catena da solo. Gli altri vanno nel Train set.
        """
        templates, uncountables = Facts.get_language_templates(
            templates_path=os.path.join(self.project_root, "data", "beliefbank", "templates.json"),
            uncountables_path=os.path.join(self.project_root, "data", "beliefbank", "non_countable.txt"))

        with open(self.facts_path) as f:
            facts = json.load(f)
            # Estrae tutti gli antecedenti e conseguenti noti
            c_antecedents = [c["antecedent"] for c in self.constraints]
            c_consequents = [c["consequent"] for c in self.constraints]
            train_facts = []
            test_facts = []

            for subject, subject_facts in facts.items():
                for key, belief in subject_facts.items():
                    relation, obj = key.split(",")
                    fact = Facts.implication2string(templates=templates, uncountables=uncountables, subject=subject,
                                                    relation=relation, symbol=True, obj=obj)
                    negated_fact = Facts.implication2string(templates=templates, uncountables=uncountables,
                                                            subject=subject, relation=relation, symbol=False, obj=obj)
                    sample = {"subject": subject, "predicate": key, "fact": fact, "negated_fact": negated_fact,
                              "belief": int(belief == "yes")}

                    # Controllo Multihop: è sia causa che effetto in regole diverse?
                    if key in c_consequents and key in c_antecedents:
                        test_facts.append(sample)
                    else:
                        train_facts.append(sample)

        return {"train": train_facts, "test": test_facts}
