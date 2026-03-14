import torch
import logging
from tqdm import tqdm
from typing import Dict, Any, List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from utils import Utils as ut

logger = logging.getLogger(__name__)


class EntailmentEvaluator:
    def __init__(self, pipeline: Any, max_new_tokens: int = 10):
        """
        Prende in input l'oggetto HallucinationPipeline già inizializzato con llm, tokenizer e dataset.
        """
        self.pipeline = pipeline
        self.llm = pipeline.llm
        self.tokenizer = pipeline.tokenizer
        self.dataset = pipeline.dataset
        self.device = self.llm.device
        self.max_new_tokens = max_new_tokens

        # Recupera il prompt dal registry usando lo storage_manager
        self.prompt_data = pipeline.storage_manager.prompt_id

    def _parse_response(self, text: str) -> int:
        """Parsa la risposta testuale del LLM in 1 (Yes/True), 0 (No/False) o -1 (Unclear)."""
        text_lower = text.lower().strip()
        # Parole chiave per il "Sì"
        if any(w in text_lower for w in ["yes", "true", "correct", "1"]):
            return 1
        # Parole chiave per il "No"
        elif any(w in text_lower for w in ["no", "false", "incorrect", "0", "not true"]):
            return 0
        else:
            return -1  # Risposta ambigua o rifiuto

    @torch.no_grad()
    def evaluate(self) -> Dict[str, Any]:
        """Esegue l'inferenza sull'intero dataset e calcola le metriche."""
        if not self.dataset:
            raise ValueError("Dataset non caricato nella pipeline.")

        logger.info(f"🚀 Avvio inferenza veloce su {len(self.dataset)} campioni...")
        self.llm.eval()

        predictions = []
        ground_truths = []
        ambiguous_count = 0

        # Il tuo dataset restituisce: text, label (1 o 0, ma formatta 'yes'/'no' da LABELS. Adattiamo il cast), instance_id
        for idx in tqdm(range(len(self.dataset)), desc="Inference & Evaluation"):
            prompt_text, string_label, instance_id = self.dataset[idx]

            # Converte la label 'yes'/'no' in 1/0
            true_label = 1 if string_label.lower() == "yes" else 0
            ground_truths.append(true_label)

            # 1. Preparazione dell'input con Chat Template
            messages = ut.build_messages(
                system_prompt="You are a helpful logical assistant. Answer only with Yes or No.",
                user_prompt=prompt_text,
                k=0
            )
            tokens = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
            ).to(self.device)

            # 2. Generazione Pura (senza Context Manager per salvare attivazioni)
            output = self.llm.generate(
                input_ids=tokens["input_ids"],
                attention_mask=tokens.get("attention_mask"),
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Avido per la valutazione logica
                temperature=0.,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # 3. Estrazione e Parsing del Testo
            gen_ids = output[0][tokens["input_ids"].shape[1]:]
            gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            pred_label = self._parse_response(gen_text)

            if pred_label == -1:
                ambiguous_count += 1
                # Per il calcolo metriche puro, penalizziamo le risposte ambigue considerandole sbagliate
                pred_label = 0 if true_label == 1 else 1

            predictions.append(pred_label)

        # 4. Calcolo Metriche
        return self._calculate_metrics(ground_truths, predictions, ambiguous_count, len(self.dataset))

    def _calculate_metrics(self, y_true: List[int], y_pred: List[int], ambiguous_count: int, total: int) -> Dict[
        str, Any]:
        """Calcola e stampa un report dettagliato delle metriche."""
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        # Metriche mirate alle allucinazioni
        hallucination_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # FPR
        over_conservatism = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # FNR
        ambiguity_rate = ambiguous_count / total

        strategy = self.dataset.neg_strategy if hasattr(self.dataset, 'neg_strategy') else "Unknown"

        logger.info("\n" + "=" * 50)
        logger.info(f"📊 REPORT VALUTAZIONE LLM - EntailmentBank")
        logger.info(f"Strategia di distorsione testata: {strategy.upper()}")
        logger.info("=" * 50)
        logger.info(f"✅ Overall Accuracy:      {acc * 100:.2f}%")
        logger.info(f"🎯 F1-Score:              {f1 * 100:.2f}%")
        logger.info(f"🔎 Precision:             {precision * 100:.2f}%")
        logger.info(f"🔄 Recall:                {recall * 100:.2f}%")
        logger.info("-" * 50)
        logger.info(f"👻 Hallucination Rate (FP): {hallucination_rate * 100:.2f}%  <-- Risponde 'Yes' quando è Falso!")
        logger.info(f"🛡️ Over-Conservatism (FN):  {over_conservatism * 100:.2f}%  <-- Risponde 'No' quando è Vero")
        logger.info(f"🤷 Ambiguità (non parsabile): {ambiguity_rate * 100:.2f}% ({ambiguous_count}/{total})")
        logger.info("=" * 50)

        return {
            "accuracy": acc, "f1": f1, "precision": precision, "recall": recall,
            "hallucination_rate": hallucination_rate, "over_conservatism": over_conservatism,
            "ambiguous_rate": ambiguity_rate, "strategy": strategy
        }