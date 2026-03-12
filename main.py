import os
import argparse
import logging
import functools
import torch
from typing import Dict, Callable, Any
from torch.utils.data import DataLoader
from huggingface_hub import login

from core.HallucinationPipeline import HallucinationPipeline
from logical_datasets.BeliefBankDataset import BeliefBankDataset
from analysis.Attributions import AttributionAnalyzer

from config.prompts import PROMPT_REGISTRY

# Configurazione Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def setup_huggingface_login() -> None:
    """Esegue il login su HuggingFace in modo sicuro."""
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    token_path = os.path.join(PROJECT_DIR, "token.txt")

    if os.path.exists(token_path):
        logger.info("🔑 Eseguo il login su HuggingFace con il token letto da file...")
        with open(token_path, "r") as f:
            login(f.read().strip())
    else:
        logger.warning("⚠️ File 'token.txt' non trovato. Assicurati che il modello sia pubblico.")


def setup_pipeline(args: argparse.Namespace, current_prompt_id=None, require_llm: bool = True) -> HallucinationPipeline:
    """Utility centralizzata per inizializzare il workflow."""
    if require_llm:
        setup_huggingface_login()

    logger.info("-" * 40 + " Hallucination Detection Pipeline " + "-" * 40)
    logger.info(f"🖥️ Cuda disponibile: {torch.cuda.is_available()}")

    p_id = current_prompt_id or args.prompt_id
    prompt_data = PROMPT_REGISTRY[p_id]

    pipeline = HallucinationPipeline(project_dir=PROJECT_DIR)
    pipeline.load_dataset(dataset_name=args.data_name, label=args.label)

    if args.test_size > 0:
        pipeline.dataset.get_sample(max_samples=args.test_size)
        logger.info(f"✂️ Dataset limitato a {len(pipeline.dataset)} elementi per il test rapido.")

    if require_llm:
        pipeline.load_llm(llm_name=args.model_name, use_local=args.use_local)

    return pipeline


def test_dataset(args: argparse.Namespace) -> None:
    """Testa il caricamento e l'iterazione del dataset."""
    logger.info(f"🛠️ Inizializzazione del dataset da: {PROJECT_DIR}")
    try:
        if args.data_name == "entailmentbank":
            from logical_datasets.EntailmentBankDataset import EntailmentBankDataset
            dataset = EntailmentBankDataset(
                project_root=PROJECT_DIR,
                label=args.label if args.label in [0, 1] else "all",
                shuffle=True
            )
        else:
            dataset = BeliefBankDataset(
                project_root=PROJECT_DIR, recreate_ids=True,
                data_type=args.data_type, label=args.label, shuffle=True
            )

        logger.info(f"✅ Dataset caricato con successo! Elementi totali: {len(dataset)}")
        fact, label, instance_id = dataset[0]
        logger.info(f"--- 🔍 Test Accesso Singolo ---\nID: {instance_id}\nFatto/Prompt: \n{fact}\nEtichetta: {label}")

        logger.info(f"\n--- 📦 Test DataLoader (Batch Size = {args.batch_size}) ---")
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        batch_facts, batch_labels, batch_ids = next(iter(dataloader))

        for i in range(min(args.batch_size, len(batch_facts))):
            logger.info(f"[{batch_ids[i]}] {batch_facts[i][:80]}... --> {batch_labels[i]}")

    except Exception as e:
        logger.error(f"❌ Errore durante il test del dataset: {e}", exc_info=True)


def run_extraction(args: argparse.Namespace, method_name: str, success_msg: str, **kwargs: Any) -> None:
    """Esegue un metodo di estrazione dinamico sulla pipeline."""
    pipeline = setup_pipeline(args, require_llm=True)
    logger.info(f"🧠 Inizio operazione sui tensori chiamando '{method_name}'...")

    try:
        pipeline.run_extraction(method_name=method_name, **kwargs)
        logger.info(f"✅ {success_msg}")
        logger.info(f"📁 Controlla la cartella cache.")
    except Exception as e:
        logger.error(f"❌ Errore durante l'estrazione: {e}", exc_info=True)


def run_probing_only(args: argparse.Namespace) -> None:
    """Esegue solo la fase di probing su attivazioni già estratte."""
    pipeline = setup_pipeline(args, require_llm=False)
    llm_short_name = args.model_name.split("/")[-1]

    logger.info(f"🚀 Avvio Probing per il modello: {llm_short_name}")

    # Fallback su default se llm non è caricato
    targets = ["hidden", "mlp", "attn"]
    layers = range(32)

    for target in targets:
        for layer in layers:
            try:
                pipeline.predict_prober(target=target, layer=layer, llm_name=llm_short_name, label=args.label)
            except FileNotFoundError:
                logger.warning(f"⚠️ Modello di probing non trovato per target {target}, layer {layer}. Salto...")
            except Exception as e:
                logger.error(f"❌ Errore al layer {layer} con target {target}: {e}")


def run_train_probers(args: argparse.Namespace) -> None:
    """Avvia l'addestramento dei probing."""
    pipeline = setup_pipeline(args, require_llm=False)
    logger.info("⚙️ Avvio addestramento e valutazione probers...")
    pipeline.train_probers(llm_name=args.model_name, test_size=0.2, epochs=30)


def run_analyze(args: argparse.Namespace) -> None:
    """Avvia l'analizzatore di attribuzioni."""

    logger.info("🔍 Avvio Analizzatore di Attribuzioni...")

    setup_huggingface_login()

    analyzer = AttributionAnalyzer(PROJECT_DIR, args.model_name, data_name=args.data_name, label=args.label)

    layers = range(32)  # Eventualmente dinamico se l'analyzer supporta l'Huggingface config
    logger.info(f"📊 Generazione profili per {len(layers)} layers...")

    analyzer.plot_layer_profile(args.instance_id)
    for i in layers:
        analyzer.plot_word_barh(args.instance_id, module="attn", layer=i)
        analyzer.plot_word_heatmap(args.instance_id, module="attn", layer=i)
        analyzer.plot_text_saliency(args.instance_id, module="hidden", layer=i)
        for module in ["attn", "mlp", "hidden"]:
            analyzer.hunt_top_dimensions(args.instance_id, module=module, layer=i, top_k=5)
    logger.info("✅ Analisi completata.")


def run_full_pipeline(args: argparse.Namespace) -> None:
    """Esegue estrazione chunked e probing completo."""
    logger.info("Inizio esecuzione Full Pipeline...")
    run_extraction(args, method_name="save_activations_chunked",
                   success_msg="Estrazione Chunked completata!",
                   use_chat_template=True, chunk_size=args.chunk_size)
    run_train_probers(args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Toolkit Hallucination Detection")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["test_dataset", "test_extraction", "test_extraction_for_probers",
                                 "test_attribution", "full_pipeline", "analyze", "train_probers", "probing"])
    parser.add_argument("--instance_id", type=int, default=0, help="ID della frase da analizzare.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--use_local", action="store_true")
    parser.add_argument("--data_name", type=str, default="beliefbank")
    parser.add_argument("--data_type", type=str, default="constraints", choices=["constraints", "facts"])
    parser.add_argument("--label", type=str, default="1", help="Label del dataset (0, 1, o 'all')")
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=100000)
    parser.add_argument("--chunk_size", type=int, default=1000, help="Dimensione blocco per salvataggio RAM.")
    parser.add_argument("--prompt_id", type=str, default="base_v1", help="ID del prompt dal registry")
    parser.add_argument("--run_all_prompts", action="store_true", help="Esegue la pipeline su tutti i prompt in sequenza")

    args = parser.parse_args()

    if args.label.isdigit():
        args.label = int(args.label)

    # Routing dinamico
    modes: Dict[str, Callable[[argparse.Namespace], None]] = {
        "test_dataset": test_dataset,
        "test_extraction": functools.partial(run_extraction, method_name="save_activations_chunked",
                                             success_msg="Estrazione attivazioni (Chunked) completata!",
                                             use_chat_template=True, chunk_size=args.chunk_size),
        "test_extraction_for_probers": functools.partial(run_extraction, method_name="save_activations_pure_forward",
                                                         success_msg="Estrazione pure forward completata!",
                                                         use_chat_template=True),
        "test_attribution": functools.partial(run_extraction, method_name="save_attributions_and_grads",
                                              success_msg="Attribuzioni calcolate con successo!"),
        "probing": run_probing_only,
        "train_probers": run_train_probers,
        "analyze": run_analyze,
        "full_pipeline": run_full_pipeline
    }

    try:
        modes[args.mode](args)
    except KeyError:
        logger.error(f"Modalità '{args.mode}' non supportata.")
    except Exception as e:
        logger.critical(f"Errore critico non gestito nell'esecuzione di {args.mode}: {e}", exc_info=True)


if __name__ == "__main__":
    main()