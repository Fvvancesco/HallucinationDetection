import os
import argparse
import torch
from torch.utils.data import DataLoader
from huggingface_hub import login

from logical_datasets.BeliefBankDataset import BeliefBankDataset
from model.HallucinationDetection import HallucinationDetection

# Configurazione cartelle dinamica
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def setup_huggingface_login():
    """Utility per il login sicuro su HuggingFace."""
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    token_path = os.path.join(PROJECT_DIR, "token.txt")

    if os.path.exists(token_path):
        print("🔑 Eseguo il login su HuggingFace con il token letto da file...")
        login(open(token_path).read().strip())
    else:
        print(
            "⚠️ Attenzione: File 'token.txt' non trovato. Assicurati di aver fatto il login o che il modello sia pubblico.")


def test_dataset(args):
    """Testa esclusivamente il caricamento e l'iterazione del dataset."""
    print(f"🛠️ Inizializzazione del dataset da: {PROJECT_DIR}")

    try:
        dataset = BeliefBankDataset(
            project_root=PROJECT_DIR,
            recreate_ids=True,
            data_type=args.data_type,
            label=args.label,
            shuffle=True
        )

        print(f"✅ Dataset caricato con successo! Elementi totali: {len(dataset)}")

        # Test di accesso singolo
        print("\n--- 🔍 Test Accesso Singolo ---")
        fact, label, instance_id = dataset[0]
        print(f"ID Istanza : {instance_id}")
        print(f"Fatto      : {fact}")
        print(f"Etichetta  : {label}")

        # Test con PyTorch DataLoader
        print(f"\n--- 📦 Test DataLoader (Batch Size = {args.batch_size}) ---")
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        batch_facts, batch_labels, batch_ids = next(iter(dataloader))

        for i in range(min(args.batch_size, len(batch_facts))):
            print(f"[{batch_ids[i].item():04d}] {batch_facts[i]} --> {batch_labels[i]}")

    except Exception as e:
        print(f"\n❌ Errore durante il test del dataset: {e}")


def test_extraction(args):
    """Testa la pipeline di caricamento modello ed estrazione delle attivazioni."""
    setup_huggingface_login()

    print("\n" + "-" * 50 + " Hallucination Detection Test " + "-" * 50)
    print(f"🖥️ Cuda disponibile (utilizzo la GPU): {torch.cuda.is_available()}")

    detector = HallucinationDetection(project_dir=PROJECT_DIR)

    # 1. Caricamento Dataset
    detector.load_dataset(dataset_name=args.data_name, label=args.label)

    # Limitiamo il dataset se richiesto per un test veloce
    if args.test_size > 0:
        detector.dataset.get_sample(max_samples=args.test_size)
        print(f"\n✂️ Dataset limitato a {len(detector.dataset)} elementi per il test rapido.\n")

    # 2. Caricamento Modello
    detector.load_llm(
        llm_name=args.model_name,
        use_local=args.use_local,
        dtype=torch.bfloat16,
        use_device_map=True
    )

    # 3. Setup Cartelle e Generazione
    detector._create_folders_if_not_exists(label=args.label)

    print("\n🧠 Inizio generazione e salvataggio attivazioni...")
    # Qui usiamo la logica di estrazione. (Usa il chunk_size se hai implementato l'ultima versione di cui parlavamo)
    detector.save_activations(use_chat_template=True)

    print("\n✅ Estrazione completata con successo!")
    print(f"📁 Controlla la cartella '{detector.CACHE_DIR_NAME}' per vedere i tensori salvati.")

def test_attribution(args):
    """Testa la pipeline di caricamento modello ed estrazione delle attivazioni."""
    setup_huggingface_login()

    print("\n" + "-" * 50 + " Hallucination Detection Test " + "-" * 50)
    print(f"🖥️ Cuda disponibile (utilizzo la GPU): {torch.cuda.is_available()}")

    detector = HallucinationDetection(project_dir=PROJECT_DIR)

    # 3. Setup Cartelle e Generazione
    detector._create_folders_if_not_exists(label=args.label)

    print("\n🧠 Inizio calcolo e salvataggio attribuzioni...")
    # Qui usiamo la logica di estrazione. (Usa il chunk_size se hai implementato l'ultima versione di cui parlavamo)
    detector.save_attributions_and_grads()

    print("\n✅ Attribuzioni calcolate con successo!")
    print(f"📁 Controlla la cartella '{detector.CACHE_DIR_NAME}' per vedere le attribuzioni.")


def run_full_pipeline(args):
    """Esegue l'intero ciclo di predizione LLM e probing KC (dal codice commentato originale)."""
    setup_huggingface_login()

    hallucination_detector = HallucinationDetection(project_dir=PROJECT_DIR)
    llm_name = args.model_name.split("/")[-1]

    for label, desc in hallucination_detector.LABELS.items():
        if args.data_name == "beliefbank" and label == 0:
            continue

        print("\n" + "==" * 25)
        print(f"Predicting {desc} instances (Label: {label})")
        print("==" * 25)

        # Estrazione LLM
        hallucination_detector.predict_llm(
            llm_name=args.model_name,
            use_local=args.use_local,
            data_name=args.data_name,
            label=label
        )
        """ da implementare
        # Probing
        for activation in hallucination_detector.ACTIVATION_TARGET:
            for layer in hallucination_detector.TARGET_LAYERS:
                hallucination_detector.predict_kc(
                    target=activation,
                    layer=layer,
                    data_name=args.data_name,
                    use_local=args.use_local,
                    label=label,
                    llm_name=llm_name
                )
        """


def main():
    parser = argparse.ArgumentParser(description="Toolkit per Test ed Estrazione Hallucination Detection")

    # Scegli la modalità di esecuzione
    parser.add_argument("--mode", type=str, required=True,
                        choices=["test_dataset", "test_extraction", "full_pipeline"],
                        help="Scegli quale test o pipeline eseguire.")

    # Parametri globali
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="Nome del modello LLM.")
    parser.add_argument("--use_local", action="store_true", help="Usa i pesi in locale invece di scaricarli.")
    parser.add_argument("--data_name", type=str, default="beliefbank", help="Nome del dataset.")
    parser.add_argument("--data_type", type=str, default="constraints", choices=["constraints", "facts"],
                        help="Tipo di dati per BeliefBank.")
    parser.add_argument("--label", type=int, default=0, help="Etichetta da filtrare (0 o 1).")

    # Parametri di utilità per i test
    parser.add_argument("--test_size", type=int, default=100,
                        help="Numero massimo di campioni per un test rapido (usa 0 per tutto il dataset).")
    parser.add_argument("--batch_size", type=int, default=10, help="Dimensione del batch per il test del Dataloader.")

    args = parser.parse_args()

    # Routing alla funzione corretta in base alla scelta
    if args.mode == "test_dataset":
        test_dataset(args)
    elif args.mode == "test_extraction":
        test_extraction(args)
    elif args.mode == "test_attribution":
        test_attribution(args)
    elif args.mode == "full_pipeline":
        run_full_pipeline(args)
    else:
        print("Modalità non riconosciuta.")


if __name__ == "__main__":
    main()