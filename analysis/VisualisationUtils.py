import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class VisualisationUtils:
    @staticmethod
    def save_prober_results(metrics_results: List[Dict[str, Any]], project_dir: str,
                            results_dir: str = "results", use_undersampling: bool = False) -> None:
        """
        Salva i risultati dell'addestramento in CSV e genera una dashboard visiva con Accuracy, AUROC e AUPRC.
        """
        df_metrics = pd.DataFrame(metrics_results)

        # 1. Salvataggio su CSV
        save_path_csv = os.path.join(project_dir, results_dir, "prober_training_results.csv")
        os.makedirs(os.path.dirname(save_path_csv), exist_ok=True)
        df_metrics.to_csv(save_path_csv, index=False)

        # 2. Impostazione della Dashboard Grafica
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)

        # Titolo dinamico per tracciare l'esperimento
        balance_text = "Classes Balanced 50/50" if use_undersampling else "Full Dataset + Class Weights"
        fig.suptitle(f'Prober Validation Metrics per Layer\n({balance_text})', fontsize=16, fontweight='bold', y=1.05)

        # --- Pannello 1: Accuracy ---
        sns.lineplot(data=df_metrics, x='layer', y='accuracy', hue='target', marker='o', ax=axes[0])
        axes[0].set_title('Accuracy', fontsize=14)
        axes[0].set_ylim(0.0, 1.05)
        axes[0].axhline(0.5, ls='--', color='gray', alpha=0.5)
        axes[0].set_ylabel('Score')
        axes[0].set_xlabel('Layer')

        # --- Pannello 2: AUROC ---
        sns.lineplot(data=df_metrics, x='layer', y='auroc', hue='target', marker='o', ax=axes[1])
        axes[1].set_title('AUROC (Area Under ROC)', fontsize=14)
        axes[1].set_ylim(0.0, 1.05)
        axes[1].axhline(0.5, ls='--', color='gray', alpha=0.5)
        axes[1].set_ylabel('')
        axes[1].set_xlabel('Layer')

        # --- Pannello 3: AUPRC ---
        sns.lineplot(data=df_metrics, x='layer', y='auprc', hue='target', marker='o', ax=axes[2])
        axes[2].set_title('AUPRC (Precision-Recall)', fontsize=14)
        axes[2].set_ylim(0.0, 1.05)
        axes[2].set_ylabel('')
        axes[2].set_xlabel('Layer')

        # Ottimizzazione layout e salvataggio
        plt.tight_layout()
        plot_path = os.path.join(project_dir, results_dir, "prober_metrics_dashboard.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()

        logger.info(f"\n✅ Training completato!")
        logger.info(f"📊 Dati numerici (con Hyperparameters) salvati in: {save_path_csv}")
        logger.info(f"📈 Dashboard visiva salvata in: {plot_path}")