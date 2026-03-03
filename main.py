import os
import torch

# Assicurati che l'import corrisponda alla struttura delle tue cartelle
from model.HallucinationDetection import HallucinationDetection


def main():
    # 1. Configurazione Iniziale
    # Puntiamo direttamente alla cartella logical_datasets
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))

    LLM_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    detector = HallucinationDetection(project_dir=BASE_DIR)  # Passiamo la root al detector
    print("🚀 Inizio pipeline di test Hallucination Detection")

    # Impostiamo il percorso corretto per i file del dataset
    detector.load_dataset(dataset_name="beliefbank", label=1)

    # ---------------------------------------------------------
    # TRUCCO PER IL TEST: Limitiamo a 10 prompt
    # Usiamo la funzione che hai scritto tu stesso nel Dataset!
    # ---------------------------------------------------------
    detector.dataset.get_sample(max_samples=10)
    print(f"\n📊 Dataset limitato a {len(detector.dataset)} elementi per il test rapido.\n")

    # 4. Caricamento Modello Linguistico
    # (Imposta use_device_map=True per usare la GPU automaticamente)
    detector.load_llm(
        llm_name=LLM_NAME,
        use_local=False,
        dtype=torch.float16,  # Usa float16 o bfloat16 a seconda della tua GPU
        use_device_map=True
    )

    # 5. Creazione delle cartelle di output
    # Questo creerà 'activation_cache/nome_modello/beliefbank/...'
    detector._create_folders_if_not_exists(label=1)

    # 6. Estrazione e Salvataggio Attivazioni
    print("\n🧠 Inizio generazione e salvataggio attivazioni...")
    detector.save_activations()

    print("\n✅ Test completato con successo!")
    print(f"Controlla la cartella '{detector.CACHE_DIR_NAME}' per vedere i tensori salvati.")


if __name__ == '__main__':
    # Disabilita eventuali warning noiosi sui symlink di HuggingFace (opzionale)
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    print(torch.cuda.is_available())

    main()