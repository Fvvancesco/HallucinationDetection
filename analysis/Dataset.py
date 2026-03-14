import json
import os

import networkx as nx
from collections import Counter

from main import PROJECT_DIR

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

def analyze_entailmentbank():
    # 1. Caricamento dei dati
    files = {'train': 'train.jsonl', 'dev': 'dev.jsonl', 'test': 'test.jsonl'}
    for split, fp in files.items():
        files[split] =os.path.join(PROJECT_DIR, "logical_datasets", "data", "entailmentbank", fp)
    dfs = []

    for split, file_path in files.items():
        try:
            df = pd.read_json(file_path, lines=True)
            df['split'] = split
            dfs.append(df)
        except Exception as e:
            print(f"Errore nel caricamento di {file_path}: {e}")

    if not dfs:
        print("Nessun file caricato. Controlla i percorsi.")
        exit()

    df_all = pd.concat(dfs, ignore_index=True)

    # 2. Estrazione delle metriche di complessità logica e rumore
    # Numero di premesse utili
    df_all['num_triples'] = df_all['meta'].apply(lambda x: len(x.get('triples', {})) if isinstance(x, dict) else 0)

    # Numero di conclusioni intermedie
    df_all['num_intermediates'] = df_all['meta'].apply(
        lambda x: len(x.get('intermediate_conclusions', {})) if isinstance(x, dict) else 0)

    # Numero di distrattori (fatti irrilevanti presenti nel prompt ma da ignorare)
    df_all['num_distractors'] = df_all['meta'].apply(
        lambda x: len(x.get('delete_list', [])) if isinstance(x, dict) else 0)

    # Rapporto rumore/segnale (Distrattori su Premesse Utili)
    df_all['distractor_ratio'] = df_all['num_distractors'] / df_all['num_triples'].replace(0, np.nan)

    # Estrazione del Branching Factor (quante premesse sono combinate in un singolo step logico)
    def get_max_branching(step_proof):
        if not isinstance(step_proof, str) or not step_proof.strip():
            return 0
        # Dividiamo i vari step separati da punto e virgola
        steps = [s.strip() for s in step_proof.split(';') if '->' in s]
        max_branch = 0
        for step in steps:
            premises = step.split('->')[0]
            # Il numero di elementi combinati è il numero di '&' + 1
            branch_factor = premises.count('&') + 1
            if branch_factor > max_branch:
                max_branch = branch_factor
        return max_branch

    df_all['max_branching_factor'] = df_all['meta'].apply(
        lambda x: get_max_branching(x.get('step_proof', '')) if isinstance(x, dict) else 0
    )

    # 3. Analisi di Base
    print("=== CONTEGGIO SAMPLE ===")
    print(df_all['split'].value_counts())
    print("-" * 40)

    # 4. Statistiche Descrittive (Medie, Deviazioni Standard, ecc.)
    print("\n=== STATISTICHE DESCRITTIVE PER SPLIT ===")
    metrics_to_analyze = [
        'depth_of_proof', 'length_of_proof', 'num_triples',
        'num_intermediates', 'num_distractors', 'max_branching_factor', 'distractor_ratio'
    ]

    # Calcoliamo le statistiche e arrotondiamo a 2 decimali
    stats = df_all.groupby('split')[metrics_to_analyze].agg(['mean', 'std', 'median', 'min', 'max']).round(2)

    # Stampiamo le statistiche in modo leggibile
    for metric in metrics_to_analyze:
        print(f"\n--- {metric.upper()} ---")
        print(stats[metric])

    # 5. Visualizzazione Grafica (Opzionale ma molto utile)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Analisi della Complessità Logica (EntailmentBank)", fontsize=16)

    # Plot 1: Depth of Proof
    '''sns.boxplot(data=df_all, x='split', y='depth_of_proof', ax=axes[0, 0], palette='Set2')
    axes[0, 0].set_title('Profondità dell\'albero (Depth)')

    # Plot 2: Length of Proof
    sns.boxplot(data=df_all, x='split', y='length_of_proof', ax=axes[0, 1], palette='Set2')
    axes[0, 1].set_title('Step totali (Length)')'''

    # Esempio per il Plot 1 (applica lo stesso schema agli altri boxplot)
    sns.boxplot(data=df_all, x='split', y='depth_of_proof', ax=axes[0, 0], palette='Set2', hue='split', legend=False)

    # Esempio per il Plot 2
    sns.boxplot(data=df_all, x='split', y='length_of_proof', ax=axes[0, 1], palette='Set2', hue='split', legend=False)



    # Plot 3: Max Branching Factor
    sns.boxplot(data=df_all, x='split', y='max_branching_factor', ax=axes[0, 2], palette='Set2', hue='split')
    axes[0, 2].set_title('Branching Factor Max (Rischio Allucinazione)')

    # Plot 4: Premesse vs Intermedi
    sns.scatterplot(data=df_all, x='num_triples', y='num_intermediates', hue='split', alpha=0.6, ax=axes[1, 0])
    axes[1, 0].set_title('Premesse vs Conclusioni Intermedie')

    # Plot 5: Numero di Distrattori
    sns.boxplot(data=df_all, x='split', y='num_distractors', ax=axes[1, 1], palette='Set2', hue='split')
    axes[1, 1].set_title('Numero di Distrattori (Rumore)')

    # Plot 6: Distractor Ratio
    sns.boxplot(data=df_all, x='split', y='distractor_ratio', ax=axes[1, 2], palette='Set2', hue='split')
    axes[1, 2].set_title('Rapporto Distrattori / Premesse Utili')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    facts = os.path.join(PROJECT_DIR, "logical_datasets", "data", "beliefbank", 'silver_facts.json')
    constraints = os.path.join(PROJECT_DIR, "logical_datasets", "data", "beliefbank", 'constraints_v2.json')
    # Esecuzione
    #analyze_beliefbank(facts, constraints)
    analyze_entailmentbank()