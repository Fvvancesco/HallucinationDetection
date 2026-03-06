import os
import torch

from model.HallucinationDetection import HallucinationDetection

from huggingface_hub import login
"""
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

def main():
    print("Cuda available (using GPU): " + str(torch.cuda.is_available()))

    print("Eseguo il login con il token letto da file")
    login(open("token.txt").read())

    # 1. Configurazione Iniziale
    # Puntiamo direttamente alla cartella logical_datasets
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.join(BASE_DIR, "logical_datasets")

    LLM_NAME = "meta-llama/Meta-Llama-3-8B-Instruct" #"TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    detector = HallucinationDetection(project_dir=PROJECT_DIR) # Passiamo la root al detector
    print("🚀 Inizio pipeline di test Hallucination Detection")

    # Impostiamo il percorso corretto per i file del dataset
    detector.load_dataset(dataset_name="beliefbank", label=0) #label inverte
    #for i in range(100):
        #print(detector.dataset[i])

    detector.dataset.get_sample(max_samples=5)
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
    detector._create_folders_if_not_exists(label=0)

    # 6. Estrazione e Salvataggio Attivazioni
    print("\n🧠 Inizio generazione e salvataggio attivazioni...")
    detector.save_activations()

    print("\n✅ Test completato con successo!")
    print(f"Controlla la cartella '{detector.CACHE_DIR_NAME}' per vedere i tensori salvati.")


if __name__ == '__main__':
    # Disabilita eventuali warning noiosi sui symlink di HuggingFace (opzionale)
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    main()