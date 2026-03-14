import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, use_bias):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=use_bias)

    def forward(self, x):
        return torch.sigmoid(self.linear(x.float())).squeeze()

    @torch.inference_mode()
    def predict(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.to(self.linear.weight.device)
        return torch.sigmoid(self.linear(x.float())).squeeze()


class LinearProber:
    MODEL_DIR = "prob_{activation}_layer{layer}.pt"
    MODEL_NAME = "prob_model_list_{layer}_L1factor3.pt"

    def __init__(self, project_dir, activation="hidden", layer=16, load_pretrained=True, input_dim=4096):
        self.layer = layer
        self.activation = activation
        self.model_name = self.MODEL_NAME.format(layer=layer)
        self.model_path = os.path.join(project_dir, "models", self.MODEL_DIR.format(activation=activation, layer=layer),
                                       self.model_name)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = LogisticRegression(input_dim=input_dim, use_bias=True).to(device)

        if load_pretrained:
            self.load_probing_model()

    def load_probing_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        saved_data = torch.load(self.model_path, weights_only=True)
        # ... (mantieni qui la tua logica esistente di load_state_dict)
        if isinstance(saved_data, dict):
            self.model.load_state_dict(saved_data)
        elif isinstance(saved_data, list) and len(saved_data) > 0:
            if hasattr(saved_data[0], 'state_dict'):
                self.model.load_state_dict(saved_data[0].state_dict())
            else:
                self.model.load_state_dict(saved_data[0])

        self.model.eval()

    def train_basic(self, X_train, y_train, X_val, y_val, epochs=50, lr=1e-3, batch_size=64):
        """Addestra il probing e calcola l'accuracy sul validation set."""
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # Valutazione
        self.model.eval()
        with torch.no_grad():
            val_outputs = self.model(X_val)
            val_preds = (val_outputs > 0.5).float()
            acc = accuracy_score(y_val.cpu(), val_preds.cpu())

        return acc

    def train(self, X_train, y_train, X_val, y_val, epochs=50, lr=1e-3, batch_size=64, pos_weight=None, tune_wd=True):
        """Addestra il probing e calcola l'accuracy sul validation set con Regolarizzazione."""
        weight_tensor = torch.tensor([pos_weight], device=X_train.device) if pos_weight else None
        criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)

        #criterion = nn.BCELoss()

        # Usiamo AdamW con un forte Weight Decay (L2 Regularization)
        # Questo impedisce al probing di overfittare le 4096 dimensioni.
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.05)

        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # Valutazione
        self.model.eval()
        with torch.no_grad():
            val_outputs = self.model(X_val)
            val_preds = (val_outputs > 0.5).float()
            acc = accuracy_score(y_val.cpu(), val_preds.cpu())

        return {
            "accuracy": acc,
            "probabilities": val_outputs.detach().cpu().numpy(),
            "best_wd": 0.01
        }

    def save_model(self):
        """Salva i pesi addestrati."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)