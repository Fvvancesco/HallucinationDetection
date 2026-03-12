import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class VisualisationUtils:
    @staticmethod
    def save_prober_results(metrics_results: List[Dict[str, Any]], project_dir: str, results_dir: str = "results") -> None:
        df_metrics = pd.DataFrame(metrics_results)
        save_path_csv = os.path.join(project_dir, results_dir, "prober_training_results.csv")
        os.makedirs(os.path.dirname(save_path_csv), exist_ok=True)
        df_metrics.to_csv(save_path_csv, index=False)

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_metrics, x='layer', y='accuracy', hue='target', marker='o')
        plt.title('Prober Validation Accuracy per Layer (Classes Balanced 50/50)')
        plt.grid(True)

        plot_path = os.path.join(project_dir, results_dir, "prober_accuracy_profile.png")
        plt.savefig(plot_path)
        logger.info(f"\n✅ Training completato! Profilo salvato in: {plot_path}")