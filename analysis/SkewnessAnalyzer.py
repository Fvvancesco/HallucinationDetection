import torch


class SkewnessAnalyzer:
    """
    Motore matematico per calcolare metriche di distribuzione, sparsità
    e magnitudo sui tensori di attivazione estratti dagli LLM.
    """

    @staticmethod
    def calculate_norms(activations: torch.Tensor) -> tuple:
        """
        Calcola la Norma L1 (Somma assoluta) e L2 (Energia Euclidea).
        Indica l'energia complessiva che attraversa il layer.
        """
        # attivazioni in formato (batch_size, hidden_dim)
        l1_norm = torch.norm(activations, p=1, dim=-1).mean().item()
        l2_norm = torch.norm(activations, p=2, dim=-1).mean().item()
        return l1_norm, l2_norm

    @staticmethod
    def calculate_kurtosis(activations: torch.Tensor) -> float:
        """
        Calcola la Curtosi di Fisher.
        - Valori > 0 indicano che il segnale è concentrato su pochissimi neuroni.
        - Valori < 0 indicano che il segnale è spalmato su quasi tutti i neuroni.
        """
        mean = activations.mean(dim=-1, keepdim=True)
        var = activations.var(dim=-1, unbiased=False, keepdim=True)
        std = torch.sqrt(var + 1e-8)  # Evita divisioni per zero

        # Calcolo del momento centrale di ordine 4 (E[(X - mu)^4])
        fourth_moment = ((activations - mean) ** 4).mean(dim=-1)

        # Curtosi di Fisher (sottraendo 3)
        kurtosis = (fourth_moment / (std.squeeze(-1) ** 4)) - 3.0
        return kurtosis.mean().item()

    @staticmethod
    def calculate_gini(activations: torch.Tensor) -> float:
        """
        Calcola l'Indice di Gini.
        Misura la disuguaglianza (sparsità) delle attivazioni.
        0 = tutti i neuroni hanno la stessa energia; 1 = un solo neurone ha tutta l'energia.
        """
        # La sparsità si valuta sulla magnitudo assoluta
        x = torch.abs(activations)

        # Ordiniamo lungo l'ultima dimensione (le 4096 hidden features)
        x_sorted, _ = torch.sort(x, dim=-1)
        n = x.shape[-1]

        # Vettore degli indici (1, 2, ..., n)
        index = torch.arange(1, n + 1, device=activations.device, dtype=activations.dtype)

        # Formula vettoriale ultra-veloce del Gini
        sum_x = x_sorted.sum(dim=-1)
        sum_x = torch.where(sum_x == 0, torch.tensor(1e-8, device=activations.device), sum_x)  # Protezione Zeri

        gini = ((2 * (x_sorted * index).sum(dim=-1)) / (n * sum_x)) - ((n + 1) / n)
        return gini.mean().item()

    @staticmethod
    def analyze_all(activations: torch.Tensor) -> dict:
        """
        Esegue tutti i calcoli in un colpo solo su un batch di attivazioni.
        """
        l1, l2 = SkewnessAnalyzer.calculate_norms(activations)
        kurtosis = SkewnessAnalyzer.calculate_kurtosis(activations)
        gini = SkewnessAnalyzer.calculate_gini(activations)

        return {
            "l1_norm": l1,
            "l2_norm": l2,
            "kurtosis": kurtosis,
            "gini_index": gini
        }