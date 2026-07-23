"""
model_loader.py
================
Loads all 5 CatBoost models once at API startup and keeps them in memory.
"""
import os
from catboost import CatBoostRegressor

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

MODEL_FILES = {
    "energy": "catboost_energy_model.cbm",
    "soc": "catboost_soc_model.cbm",
    "congestion": "catboost_congestion_model.cbm",
    "demand": "catboost_demand_model.cbm",
    "leftover": "catboost_leftover_model.cbm",
}

_models = {}


def load_all_models():
    """Load every .cbm model into memory. Call once at API startup."""
    for key, filename in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        model = CatBoostRegressor()
        model.load_model(path)
        _models[key] = model
    return _models


def get_model(name: str):
    if name not in _models:
        raise RuntimeError(f"Model '{name}' is not loaded. Did startup run correctly?")
    return _models[name]
