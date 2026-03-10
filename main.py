import os
import argparse
import torch
from torch.utils.data import DataLoader
from huggingface_hub import login

from attributions.analysis import AttributionAnalyzer
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

    print("\n🧠 Inizio calcolo e salvataggio attribuzioni...")
    # Qui usiamo la logica di estrazione. (Usa il chunk_size se hai implementato l'ultima versione di cui parlavamo)
    detector.save_attributions_and_grads()

    print("\n✅ Attribuzioni calcolate con successo!")
    print(f"📁 Controlla la cartella '{detector.CACHE_DIR_NAME}' per vedere le attribuzioni.")

def run_probing_only(args):
        """Esegue solo la fase di probing su attivazioni già estratte."""
        detector = HallucinationDetection(project_dir=PROJECT_DIR)
        llm_short_name = args.model_name.split("/")[-1]

        targets = ["hidden", "mlp", "attn"]
        layers = list(range(0, 32))

        print(f"\n🚀 Avvio Probing per il modello: {llm_short_name}")

        detector.load_dataset(dataset_name=args.data_name, label=args.label)

        for target in targets:
            print(f"\n--- Analisi Target: {target} ---")
            for layer in layers:
                try:
                    detector.predict_prober(
                        target=target,
                        layer=layer,
                        llm_name=llm_short_name,
                        label=args.label
                    )
                except FileNotFoundError:
                    print(f"⚠️ Modello di prober non trovato per layer {layer}, salto...")
                except Exception as e:
                    print(f"❌ Errore al layer {layer}: {e}")

def run_train_probers(args):
    detector = HallucinationDetection(project_dir=PROJECT_DIR)
    detector.train_and_evaluate_probers(
        llm_name=args.model_name,
        data_name=args.data_name,
        test_size=0.2,
        epochs=30 # Puoi esporlo ad argparse se vuoi
    )


def run_full_pipeline(args):
    """Esegue l'intero ciclo di predizione LLM e probing Lineare."""
    setup_huggingface_login()

    detector = HallucinationDetection(project_dir=PROJECT_DIR)
    llm_name = args.model_name.split("/")[-1]

    for label, desc in detector.LABELS.items():
        if args.data_name == "beliefbank" and label == 0:
            continue

        print("\n" + "==" * 25)
        print(f"Predicting {desc} instances (Label: {label})")
        print("==" * 25)

        detector.predict_llm(
            llm_name=args.model_name,
            use_local=args.use_local,
            data_name=args.data_name,
            label=label
        )

        # Probing post-estrazione
        for target in detector.ACTIVATION_TARGET:
            for layer in detector.TARGET_LAYERS:
                try:
                    detector.predict_prober(
                        target=target,
                        layer=layer,
                        data_name=args.data_name,
                        use_local=args.use_local,
                        label=label,
                        llm_name=llm_name
                    )
                except Exception as e:
                    print(f"Errore al layer {layer} con target {target}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Toolkit per Test ed Estrazione Hallucination Detection")

    parser.add_argument("--mode", type=str, required=True,
                        choices=["test_dataset", "test_extraction", "test_attribution", "full_pipeline", "analyze", "train_probers", "probing"],
                        help="Scegli quale test o pipeline eseguire.")

    # Aggiungi questo parametro per l'analizzatore
    parser.add_argument("--instance_id", type=int, default=0, help="ID della frase da analizzare.")

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
    parser.add_argument("--batch_size", type=int, default=100000, help="Dimensione del batch per il test del Dataloader.")

    args = parser.parse_args()

    # Routing alla funzione corretta in base alla scelta
    if args.mode == "test_dataset":
        test_dataset(args)
    elif args.mode == "test_extraction":
        test_extraction(args)
    elif args.mode == "test_attribution":
        test_attribution(args)
    elif args.mode == "probing":
        run_probing_only(args)
    elif args.mode == "train_probers":
        run_train_probers(args)
    elif args.mode == "analyze":
        print("\n🔍 Avvio Analizzatore di Attribuzioni...")
        analyzer = AttributionAnalyzer(PROJECT_DIR, args.model_name, data_name=args.data_name, label=args.label)

        # 1. Dove si trova la conoscenza logica?
        analyzer.plot_layer_profile(args.instance_id)

        #analyzer.plot_word_heatmap(args.instance_id)
        #analyzer.plot_word_barh(args.instance_id)

        for i in range(32):
            # 2. Quando (su quale parola) il layer 20 si è "svegliato"?
            analyzer.plot_word_barh(args.instance_id, module="attn", layer=i)
            analyzer.plot_word_heatmap(args.instance_id, module="attn", layer=i)
            analyzer.plot_text_saliency(args.instance_id, module="hidden", layer=i)

        # 3. Chi (quali canali) ha spinto la decisione finale nell'ultimo strato?
        for i in range(32):
            analyzer.hunt_top_dimensions(args.instance_id, module="attn", layer=i, top_k=5)
            analyzer.hunt_top_dimensions(args.instance_id, module="mlp", layer=i, top_k=5)
            analyzer.hunt_top_dimensions(args.instance_id, module="hidden", layer=i, top_k=5)
    elif args.mode == "full_pipeline":
        run_full_pipeline(args)
    else:
        print("Modalità non riconosciuta.")


if __name__ == "__main__":
    main()