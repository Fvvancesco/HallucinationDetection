import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

import model.utils as ut
from logical_datasets.BeliefBankDataset import BeliefBankDataset


class AttributionAnalyzer:
    """
    Classe per l'analisi delle attribuzioni (Act x Grad) estratte da LLMs.
    Gestisce la mappatura dei token, l'analisi per layer (Dove), per token (Quando)
    e la caccia alle singole feature (Chi).
    """

    def __init__(self, project_dir: str, model_name: str, data_name: str = "beliefbank", label: int = 1):
        self.project_dir = project_dir
        self.model_name = model_name
        self.cache_dir = os.path.join(project_dir, "activation_cache")
        self.modules = ["hidden", "mlp", "attn"]

        print("⏳ Caricamento Tokenizer e Dataset per l'analisi...")
        self.tokenizer = ut.load_tokenizer(model_name, local=False)
        self.dataset = BeliefBankDataset(project_root=project_dir, data_type="constraints", label=label, shuffle=False)
        print("✅ Setup Analizzatore completato.")

    # ==========================================
    # Metodi di Supporto (Utilities)
    # ==========================================
    def _get_tensor(self, module: str, layer: int, instance_id: int) -> torch.Tensor:
        """Carica il tensore di attribuzione [Sequenza, 4096] dal disco."""
        file_path = os.path.join(self.cache_dir, f"attributions_{module}", f"layer{layer}-id{instance_id}.pt")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File non trovato: {file_path}. Assicurati di aver fatto l'estrazione.")

        # Ritorniamo il tensore in formato float32 per i calcoli analitici
        return torch.load(file_path, map_location="cpu", weights_only=False).float()

    def _get_tokens_from_instance(self, instance_id: int) -> Tuple[str, List[str]]:
        """Recupera la frase originale dal dataset e la scompone nei token testuali esatti."""
        # Trova la riga nel dataset originale
        row = self.dataset.dataset[self.dataset.dataset['instance_id'] == instance_id]
        if row.empty:
            raise ValueError(f"Instance ID {instance_id} non trovato nel dataset.")

        fact = row.iloc[0]['fact']

        # Ricostruisce lo stesso identico prompt usato in HallucinationDetection
        system_prompt = "You are a helpful assistant. Answer truthfully with only one word: TRUE or FALSE."
        user_prompt = f"Is the following statement true: {fact}?"
        messages = ut.build_messages(system_prompt=system_prompt, user_prompt=user_prompt, k=0)

        token_dict = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )

        input_ids = token_dict["input_ids"][0].tolist()

        # Decodifica ogni singolo token ID in una stringa leggibile
        token_strings = []
        for tok_id in input_ids:
            tok_str = self.tokenizer.decode([tok_id])
            # Pulizia per evitare stringhe vuote o invisibili nei grafici
            tok_str = tok_str.replace('\n', '\\n').strip()
            if not tok_str:
                tok_str = f"[{tok_id}]"
            token_strings.append(tok_str)

        return fact, token_strings

    # ==========================================
    # FASE A: Il "Quando" (Mappa di Calore Parole)
    # ==========================================
    def plot_word_heatmap(self, instance_id: int, module: str = "mlp", layer: int = 20):
        """Disegna un grafico a barre che mostra l'importanza logica di ogni parola della frase."""
        print(f"\n--- FASE A: Analisi Token (Quando) | Layer {layer} | Modulo {module} ---")

        fact, tokens = self._get_tokens_from_instance(instance_id)
        tensor = self._get_tensor(module, layer, instance_id)  # Shape: [Seq_Len, 4096]

        assert len(tokens) == tensor.shape[0], f"Mismatch! Tokens: {len(tokens)}, Tensor Seq: {tensor.shape[0]}"

        # Somma il valore assoluto su tutti e 4096 i canali per ogni parola
        importanza_per_parola = torch.abs(tensor).sum(dim=1).numpy()

        df = pd.DataFrame({'Token': tokens, 'Importanza': importanza_per_parola})

        plt.figure(figsize=(14, 6))
        plt.bar(df.index, df['Importanza'], color='skyblue', edgecolor='black')
        plt.xticks(df.index, df['Token'], rotation=45, ha='right', fontsize=12)
        plt.title(f"Importanza Parole (Layer {layer} - {module.upper()})\nFatto: {fact}", fontsize=14)
        plt.ylabel("Energia Attribuzione", fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ==========================================
    # FASE B: Il "Dove" (Profilo dei Layer)
    # ==========================================
    def plot_layer_profile(self, instance_id: int, total_layers: int = 32):
        """Calcola l'energia totale di ogni strato per individuare dove il modello prende la decisione."""
        print(f"\n--- FASE B: Analisi Layer (Dove) | Instance ID: {instance_id} ---")
        fact, _ = self._get_tokens_from_instance(instance_id)

        risultati = {mod: [] for mod in self.modules}

        for mod in self.modules:
            for layer in range(total_layers):
                try:
                    tensor = self._get_tensor(mod, layer, instance_id)
                    energia_totale = torch.abs(tensor).sum().item()
                    risultati[mod].append(energia_totale)
                except FileNotFoundError:
                    risultati[mod].append(0.0)

        df = pd.DataFrame(risultati)

        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['hidden'], label='Hidden (Autostrada)', color='black', linewidth=2, linestyle='--')
        plt.plot(df.index, df['mlp'], label='MLP (Fatti)', color='blue', linewidth=2)
        plt.plot(df.index, df['attn'], label='Attention (Contesto)', color='red', linewidth=2)

        plt.title(f"Profilo dei Layer\nFatto: {fact}", fontsize=14)
        plt.xlabel("Numero Layer (0 -> 31)", fontsize=12)
        plt.ylabel("Energia Totale (Somma Assoluta)", fontsize=12)
        plt.xticks(np.arange(0, total_layers, 2))
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ==========================================
    # FASE C: Caccia alle Feature (Il "Chi")
    # ==========================================
    def hunt_top_dimensions(self, instance_id: int, module: str = "hidden", layer: int = 31, top_k: int = 5):
        """Trova le dimensioni (canali) con il maggiore impatto positivo e negativo sull'ultima parola."""
        print(f"\n--- FASE C: Caccia alle Feature (Chi) | Layer {layer} | Modulo {module} ---")

        tensor = self._get_tensor(module, layer, instance_id)

        # Concentriamoci ESCLUSIVAMENTE sull'ultimo token della frase
        ultimo_token_attr = tensor[-1, :]  # Shape: [4096]

        # Ordiniamo i valori per trovare i massimi e i minimi reali (non assoluti)
        valori, indici = torch.sort(ultimo_token_attr, descending=True)

        # Top K Positivi (Promotori del TRUE)
        top_positivi = [(indici[i].item(), valori[i].item()) for i in range(top_k)]

        # Top K Negativi (Promotori del FALSE)
        top_negativi = [(indici[-(i + 1)].item(), valori[-(i + 1)].item()) for i in range(top_k)]

        print("\n🟩 TOP PROMOTORI (Spingono verso la Verità / TRUE):")
        for rank, (dim_idx, score) in enumerate(top_positivi, 1):
            print(f"  {rank}. Dimensione {dim_idx:04d} --> Forza: +{score:.4f}")

        print("\n🟥 TOP DETRATTORI (Spingono verso la Falsità / FALSE):")
        for rank, (dim_idx, score) in enumerate(top_negativi, 1):
            print(f"  {rank}. Dimensione {dim_idx:04d} --> Forza: {score:.4f}")


# Esempio di utilizzo se eseguito come script indipendente
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", type=str, default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--instance_id", type=int, default=0)
    args = parser.parse_args()

    # Inizializza l'analizzatore
    analyzer = AttributionAnalyzer(args.project_dir, args.model_name)

    # Lancia le analisi!
    analyzer.plot_layer_profile(args.instance_id)
    # analyzer.plot_word_heatmap(args.instance_id, module="mlp", layer=20)
    # analyzer.hunt_top_dimensions(args.instance_id, module="hidden", layer=31)