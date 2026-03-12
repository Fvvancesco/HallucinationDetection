import json
import os

import networkx as nx
from collections import Counter

from main import PROJECT_DIR

def analyze_beliefbank(facts_file, constraints_file):
    print(f"Caricamento dati da {facts_file} e {constraints_file}...\n")

    # 1. Caricamento dei dati
    with open(facts_file, 'r') as f:
        facts_data = json.load(f)

    with open(constraints_file, 'r') as f:
        constraints_data = json.load(f)

    # --- ANALISI GENERALE DEL DATASET ---
    entities = list(facts_data.keys())
    num_entities = len(entities)

    all_facts = []
    true_facts = 0
    false_facts = 0
    unique_relations_targets = set()

    for entity, facts in facts_data.items():
        for rel_target, label in facts.items():
            all_facts.append((entity, rel_target, label))
            unique_relations_targets.add(rel_target)
            if label == "yes":
                true_facts += 1
            elif label == "no":
                false_facts += 1

    num_total_facts = len(all_facts)
    num_unique_rel_targets = len(unique_relations_targets)
    possible_combinations = num_entities * num_unique_rel_targets
    sparsity = 1.0 - (num_total_facts / possible_combinations) if possible_combinations > 0 else 0

    print("=== 1. METRICHE GENERALI DEL DATASET ===")
    print(f"Entità uniche: {num_entities}")
    print(f"Asserzioni/Query uniche: {num_unique_rel_targets}")
    print(f"Fatti totali annotati: {num_total_facts}")
    print(f"Class Imbalance - Fatti Veri: {true_facts} ({(true_facts / num_total_facts) * 100:.2f}%)")
    print(f"Class Imbalance - Fatti Falsi: {false_facts} ({(false_facts / num_total_facts) * 100:.2f}%)")
    print(f"Sparsità del dataset: {sparsity * 100:.2f}%\n")

    # --- ANALISI STRUTTURALE DEL GRAFO (VINCOLI) ---
    G = nx.DiGraph()
    for node in constraints_data.get("nodes", []):
        G.add_node(node["id"])

    implication_types = Counter()
    implications = []

    for link in constraints_data.get("links", []):
        source = link["source"]
        target = link["target"]
        weight = link["weight"]  # Es. 'yes_yes' (A->B) o 'yes_no' (A->!B)
        G.add_edge(source, target, weight=weight)
        implication_types[weight] += 1
        implications.append((source, target, weight))

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    # Grado dei vincoli
    degrees = dict(G.degree())
    avg_degree = sum(degrees.values()) / num_nodes if num_nodes > 0 else 0

    # Copertura (Quanti fatti nel dataset hanno almeno un vincolo?)
    constrained_facts = sum(1 for rel in unique_relations_targets if rel in G.nodes and G.degree(rel) > 0)
    coverage = (constrained_facts / num_unique_rel_targets) * 100 if num_unique_rel_targets > 0 else 0

    # Verifica cicli
    has_cycles = not nx.is_directed_acyclic_graph(G)

    print("=== 2. METRICHE STRUTTURALI DEI VINCOLI ===")
    print(f"Nodi nel grafo dei vincoli: {num_nodes}")
    print(f"Vincoli totali (Archi): {num_edges}")
    print(f"Grado medio (vincoli per nodo): {avg_degree:.2f}")
    print(f"Tipi di implicazione: {dict(implication_types)}")
    print(f"Il grafo contiene cicli logici? {'Sì' if has_cycles else 'No'}")
    print(f"Copertura dei vincoli sulle relazioni uniche: {coverage:.2f}%\n")

    # --- ANALISI DEGLI ERRORI (SANITY CHECK SUL GROUND TRUTH) ---
    contradictions = 0
    total_checked = 0

    for entity, facts in facts_data.items():
        for source, target, weight in implications:
            if source in facts and target in facts:
                source_val = facts[source]
                target_val = facts[target]
                total_checked += 1

                # Implicazione Positiva (Se A=yes, B deve essere yes)
                if weight == 'yes_yes':
                    if source_val == 'yes' and target_val == 'no':
                        contradictions += 1

                # Implicazione Negativa (Se A=yes, B deve essere no)
                elif weight == 'yes_no':
                    if source_val == 'yes' and target_val == 'yes':
                        contradictions += 1

    contradiction_rate = (contradictions / total_checked) * 100 if total_checked > 0 else 0

    print("=== 3. SANITY CHECK SUL GROUND TRUTH ===")
    print(f"Coppie di vincoli verificate sulle label vere: {total_checked}")
    print(f"Contraddizioni trovate nelle label Ground Truth: {contradictions} ({contradiction_rate:.2f}%)")


if __name__ == "__main__":
    facts = os.path.join(PROJECT_DIR, "logical_datasets", "data", "beliefbank", 'silver_facts.json')
    constraints = os.path.join(PROJECT_DIR, "logical_datasets", "data", "beliefbank", 'constraints_v2.json')
    # Esecuzione
    analyze_beliefbank(facts, constraints)