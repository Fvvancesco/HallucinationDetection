import os
import torch

from logical_datasets.BeliefBankDataset import BeliefBankDataset
from model.HallucinationDetection_refactored import HallucinationDetection

from huggingface_hub import login
"""
predict.py

import os
import argparse
from model.HallucinationDetection import HallucinationDetection

PROJECT_ROOT = os.getcwd()


def main(args):
    model_name = args.model_name
    data_name = args.data_name
    use_local = args.use_local
    hallucination_detector = HallucinationDetection(project_dir=PROJECT_ROOT)

    llm_name = model_name.split("/")[-1]

    for label, desc in HallucinationDetection.LABELS.items():
        if data_name == "mushroom" and label == 0:
            continue

        print("==" * 50)
        print(f"Predicting {desc} instances")
        print("==" * 50)
        hallucination_detector.predict_llm(llm_name=model_name, use_local=use_local, data_name=data_name, label=label)

        for activation in HallucinationDetection.ACTIVATION_TARGET:
            for layer in HallucinationDetection.TARGET_LAYERS:
                hallucination_detector.predict_kc(target=activation, layer=layer, data_name=data_name,
                                                  use_local=use_local, label=label, llm_name=llm_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hallucination Detection predictions.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B",
                        help="Name of the LLM model to use.")
    parser.add_argument("--data_name", type=str, default=HallucinationDetection.DEFAULT_DATASET,
                        help="Name of the dataset to use.")
    parser.add_argument("--use_local", action="store_true", help="Use local model instead of remote.")

    args = parser.parse_args()
    main(args)
"""

#Configurazione cartelle
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(PROJECT_DIR, "logical_datasets")

#Modello per il testing
LLM_NAME = "meta-llama/Meta-Llama-3-8B-Instruct" #"TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Numero di elementi per ogni batch caricato da PyTorch
TEST_SIZE = 1000

if __name__ == "__main__":
    c = 2#input("Scegli la modalita' di testing:\n1. Dataset\n2. LLM\nScelta: ")
    if c == "1":
        from torch.utils.data import DataLoader

        # 1. Definisci la root del progetto
        # (Sostituisci "." con il percorso assoluto se i file si trovano altrove)
        PROJECT_ROOT = "."

        print(f"🛠️ Inizializzazione del dataset da: {os.path.abspath(PROJECT_ROOT)}")

        try:
            # 2. Istanzia il dataset
            dataset = BeliefBankDataset(
                project_root=PROJECT_ROOT,
                recreate_ids=True,
                data_type="constraints",  # Cambia in "constraints" per testare l'estrazione logica
                label=0,
                shuffle=True
            )

            print(f"✅ Dataset caricato con successo! Elementi totali: {len(dataset)}")

            # 3. Test di accesso singolo (__getitem__)
            print("\n--- 🔍 Test Accesso Singolo ---")
            fact, label, instance_id = dataset[0]
            print(f"ID Istanza : {instance_id}")
            print(f"Fatto      : {fact}")
            print(f"Etichetta  : {label}")

            # 4. Test con PyTorch DataLoader (Simulazione Training)
            print(f"\n--- 📦 Test DataLoader (Batch Size = {TEST_SIZE}) ---")
            # Il DataLoader raggruppa i dati in blocchi (batch) ottimali per la GPU
            dataloader = DataLoader(dataset, batch_size=TEST_SIZE, shuffle=True)

            # Estrai il primo batch (gruppo) di dati
            batch_facts, batch_labels, batch_ids = next(iter(dataloader))

            # Stampa i risultati del primo batch
            for i in range(TEST_SIZE):
                print(f"[{batch_ids[i].item():04d}] {batch_facts[i]} --> {batch_labels[i]}")

        except FileNotFoundError as e:
            # Gestione errori se i file JSON non sono nella cartella corretta
            print(f"\n❌ Errore di percorso: Non riesco a trovare i file JSON.")
            print(f"Assicurati di avere la seguente struttura:")
            print(f"{PROJECT_ROOT}/data/beliefbank/constraints_v2.json")
            print(f"{PROJECT_ROOT}/data/beliefbank/calibration_facts.json")
            print(f"Dettaglio: {e}")
        except Exception as e:
            print(f"\n❌ Errore inaspettato durante l'esecuzione: {e}")
    else:
        # Disabilita eventuali warning noiosi sui symlink di HuggingFace (opzionale)
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

        print(
            "-------------------------------------------------- Hallucination detection -------------------------------------------------- ")
        # Controllo utilizzo GPU
        print("Cuda disponibile (utilizzo la GPU): " + str(torch.cuda.is_available()))

        # Login su hf per usare i modelli su licenza
        print("Eseguo il login con il token letto da file")
        login(open("token.txt").read().strip())

        detector = HallucinationDetection(project_dir=PROJECT_DIR)  # Passiamo la root al detector
        print("Inizio pipeline di test Hallucination Detection")

        # Impostiamo il percorso corretto per i file del dataset
        detector.load_dataset(dataset_name="beliefbank", label=0)  # attenzione, label inverte le etichette

        detector.dataset.get_sample(max_samples=TEST_SIZE)
        print(f"\nDataset limitato a {len(detector.dataset)} elementi per il test rapido.\n")

        # Caricamento Modello Linguistico
        # (Imposta use_device_map=True per usare la GPU automaticamente)
        detector.load_llm(
            llm_name=LLM_NAME,
            use_local=False,
            dtype=torch.bfloat16,  # Usa float16 o bfloat16 (rtx30) a seconda della tua GPU (rtx20)
            use_device_map=True
        )

        # 5. Creazione delle cartelle di output
        # Questo creerà 'activation_cache/nome_modello/beliefbank/...'
        detector._create_folders_if_not_exists()

        # 6. Estrazione e Salvataggio Attivazioni
        print("\n🧠 Inizio generazione e salvataggio attivazioni...")
        detector.save_activations()

        print("\n✅ Test completato con successo!")
        print(f"Controlla la cartella '{detector.CACHE_DIR_NAME}' per vedere i tensori salvati.")
