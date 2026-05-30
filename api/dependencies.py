# api/dependencies.py
import os
import sys
from pathlib import Path

import joblib
import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


# ============================================================
# LA CLASE DEBE COINCIDIR EXACTAMENTE CON LA DEL NOTEBOOK
# ============================================================
class TransportLSTM(nn.Module):
    def __init__(
        self,
        num_routes,
        num_climas,
        input_size=4,
        hidden_size=128,
        num_layers=2,
        route_embedding_dim=8,
        clima_embedding_dim=4,
        dropout=0.2,
    ):
        super().__init__()

        self.route_embedding = nn.Embedding(num_routes, route_embedding_dim)
        self.clima_embedding = nn.Embedding(num_climas, clima_embedding_dim)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        fc_input = hidden_size + route_embedding_dim + clima_embedding_dim

        self.fc = nn.Sequential(
            nn.Linear(fc_input, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x, route_ids, clima_ids):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        route_embed = self.route_embedding(route_ids)
        clima_embed = self.clima_embedding(clima_ids)
        combined = torch.cat([lstm_out, route_embed, clima_embed], dim=1)
        return self.fc(combined)

# ============================================================
# FUNCIÓN DE CARGA GLOBAL
# ============================================================
MODEL_DIR = os.path.join("models", "demand")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Variables globales para el Singleton
_model = None
_scalers = {}
_distraction_classifier = None

def get_demand_model_infra():
    """Carga el modelo y scalers una sola vez en memoria (Pattern Singleton)"""
    global _model, _scalers

    if _model is None:
        try:
            _scalers["feature"] = joblib.load(os.path.join(MODEL_DIR, "feature_scaler.pkl"))
            _scalers["target"] = joblib.load(os.path.join(MODEL_DIR, "target_scaler.pkl"))
            _scalers["route"] = joblib.load(os.path.join(MODEL_DIR, "route_encoder.pkl"))
            _scalers["clima"] = joblib.load(os.path.join(MODEL_DIR, "clima_encoder.pkl"))

            _model = TransportLSTM(
                num_routes=len(_scalers["route"].classes_),
                num_climas=len(_scalers["clima"].classes_)
            ).to(DEVICE)

            weights = torch.load(os.path.join(MODEL_DIR, "best_model.pth"), map_location=DEVICE, weights_only=True)
            _model.load_state_dict(weights)
            _model.eval()
            print(f"✓ Modelo de demanda cargado exitosamente en {DEVICE} desde dependencies.py")
        except Exception as e:
            print(f"❌ Error cargando infraestructura de demanda: {e}")

    return _model, _scalers, DEVICE


def get_distraction_classifier():
    """Carga el clasificador de distracciones una sola vez si existe el checkpoint."""
    global _distraction_classifier

    if _distraction_classifier is None:
        try:
            from src.module2_distraction.classifier import DriverDistractionClassifier

            checkpoint_path = ROOT_DIR / "models" / "module2_distraction" / "best_model.pth"
            if not checkpoint_path.exists():
                checkpoint_path = ROOT_DIR / "api" / "models" / "module2_distraction" / "best_model.pth"
            if not checkpoint_path.exists():
                return None
            _distraction_classifier = DriverDistractionClassifier(
                checkpoint_path=checkpoint_path,
                device=DEVICE,
            )
            print(f"Modelo de distraccion cargado en {DEVICE}: {checkpoint_path}")
        except Exception as e:
            print(f"Error cargando modelo de distraccion: {e}")
            return None

    return _distraction_classifier
