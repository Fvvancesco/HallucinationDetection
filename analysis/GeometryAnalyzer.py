import os
import torch
import pandas as pd
import logging
import torch.nn.functional as F
from typing import List, Dict, Any

from core.StorageManager import StorageManager
from analysis.SkewnessAnalyzer import SkewnessAnalyzer

#se il computer è un mammifero, (#se x è un mammifero allora non depone le uova) allora il computer non depone le uova

logger = logging.getLogger(__name__)


class GeometryAnalyzer:
    ACTIVATION_TARGETS = ["hidden", "mlp", "attn"]

    def __init__(self, project_dir: str, dataset: Any, dataset_name: str, target_layers: List[int],
                 cache_dir_name: str = "activation_cache", prompt_id: str = "base_v1"):
        self.project_dir = project_dir
        self.dataset = dataset
        self.dataset_name = dataset_name
        # Ordiniamo i layer per assicurarci che il calcolo della traiettoria (L vs L-1) abbia senso
        self.target_layers = sorted(target_layers)
        self.cache_dir_name = cache_dir_name
        self.prompt_id = prompt_id

    def run_analysis(self, llm_name: str) -> pd.DataFrame:
        """
        Esegue l'analisi statistica e geometrica layer-by-layer, confrontando
        le distribuzioni delle frasi Vere con quelle delle Allucinazioni.
        """
        logger.info("\n" + "=" * 50 + "\n📐 Avvio Analisi Geometrica (Non-Parametrica)\n" + "=" * 50)
        llm_short_name = llm_name.split("/")[-1]

        # 1. Costruiamo la mappa degli ID -> Label dal dataset
        id_to_label = {}
        for i in range(len(self.dataset)):
            _, label_str, instance_id = self.dataset[i]
            val = 1.0 if str(label_str).lower() in ["yes", "1", "true"] else 0.0
            id_to_label[instance_id] = val

        results = []

        for target in self.ACTIVATION_TARGETS:
            logger.info(f"\n--- Estrazione Metriche su {target.upper()} ---")

            # Memoria temporanea per calcolare la traiettoria (Similarità tra Layer L e L-1)
            prev_activations = None
            prev_ids = None

            for layer in self.target_layers:
                try:
                    activations, instance_ids = StorageManager.load_activations(
                        model_name=llm_short_name,
                        data_name=self.dataset_name,
                        prompt_id=self.prompt_id,
                        analyse_activation=target,
                        layer_idx=layer,
                        results_dir=os.path.join(self.project_dir, self.cache_dir_name)
                    )
                except FileNotFoundError:
                    continue

                # Mappiamo le label per dividere i tensori tra "Veri" e "Allucinati"
                labels = torch.tensor([id_to_label[iid] for iid in instance_ids], dtype=torch.float32)

                idx_true = (labels == 1.0).nonzero(as_tuple=True)[0]
                idx_false = (labels == 0.0).nonzero(as_tuple=True)[0]

                # Estraiamo i sottogruppi
                act_true = activations[idx_true] if len(idx_true) > 0 else None
                act_false = activations[idx_false] if len(idx_false) > 0 else None

                # ==========================================================
                # 1. ANALISI SKEWNESS & MAGNITUDO (Dal paper su Knowledge Conflicts)
                # ==========================================================
                stats_true = SkewnessAnalyzer.analyze_all(act_true) if act_true is not None else {}
                stats_false = SkewnessAnalyzer.analyze_all(act_false) if act_false is not None else {}

                # ==========================================================
                # 2. GEOMETRIA DELLA VERITÀ (Distanza tra i Centroidi)
                # ==========================================================
                centroid_dist = None
                if act_true is not None and act_false is not None:
                    # Troviamo il baricentro (media) delle due classi
                    c_true = act_true.mean(dim=0)
                    c_false = act_false.mean(dim=0)
                    # Distanza Euclidea tra il Concetto di "Verità" e quello di "Allucinazione"
                    centroid_dist = torch.norm(c_true - c_false, p=2).item()

                # ==========================================================
                # 3. ANALISI TRAIETTORIALE (Flusso Semantico)
                # ==========================================================
                traj_sim_true = None
                traj_sim_false = None

                # Calcoliamo la Similarità Coseno solo se abbiamo il layer precedente
                # e gli ID corrispondono (nessun dato mancante)
                if prev_activations is not None and prev_ids == instance_ids:
                    # Coseno tra Vettore al layer L e Vettore al layer L-1
                    sims = F.cosine_similarity(prev_activations, activations, dim=-1)

                    if act_true is not None:
                        traj_sim_true = sims[idx_true].mean().item()
                    if act_false is not None:
                        traj_sim_false = sims[idx_false].mean().item()

                # Aggiorniamo la memoria per il prossimo ciclo
                prev_activations = activations
                prev_ids = instance_ids

                # Salviamo tutto il blocco di metriche
                results.append({
                    "target": target,
                    "layer": layer,
                    "l1_true": stats_true.get("l1_norm", None),
                    "l1_false": stats_false.get("l1_norm", None),
                    "kurtosis_true": stats_true.get("kurtosis", None),
                    "kurtosis_false": stats_false.get("kurtosis", None),
                    "gini_true": stats_true.get("gini_index", None),
                    "gini_false": stats_false.get("gini_index", None),
                    "centroid_distance": centroid_dist,
                    "traj_sim_true": traj_sim_true,
                    "traj_sim_false": traj_sim_false
                })

                logger.info(
                    f"Layer {layer:02d} | Dist. Centroidi: {centroid_dist:.4f} " if centroid_dist else f"Layer {layer:02d} | Metriche Calcolate")

        # 4. Esportazione Dati
        df_geometry = pd.DataFrame(results)
        save_path = os.path.join(self.project_dir, "results", "geometry_analysis.csv")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_geometry.to_csv(save_path, index=False)

        logger.info(f"\n✅ Analisi Geometrica e Statistica completata!")
        logger.info(f"📁 Dati salvati in: {save_path}")

        return df_geometry