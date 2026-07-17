"""
SoC Prediction
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.model_training import run_training_pipeline

CONFIG = {
    "input_file": "data/processed/features_soc_model.csv",
    "output_dir": "outputs/soc_model",
    "model_file": "models/catboost_soc_model.cbm",
    "target_col": "soc_pct",
    "title": "SoC Prediction",
    "model_name": "soc",
    "cat_features": ["charging_station_id", "charger_id"],
    "clip_range": (0, 100),
    "units": "%",
    "compute_mape": False,
    "palette": "magma",
    "plots": ["importance", "avp"],
    "params": {
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": 42,
        "verbose": 50,
        "early_stopping_rounds": 50,
    },
}

if __name__ == "__main__":
    run_training_pipeline(CONFIG)
