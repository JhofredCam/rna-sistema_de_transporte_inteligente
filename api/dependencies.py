# api/dependencies.py
import os
import sys
from pathlib import Path

import joblib
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
ROOT_DIR = Path(PROJECT_ROOT)

from src.module1_demand.model import TransportLSTM

# ============================================================
# FUNCIÓN DE CARGA GLOBAL
# ============================================================
MODEL_DIR = os.path.join(API_DIR, "models", "demand")
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
