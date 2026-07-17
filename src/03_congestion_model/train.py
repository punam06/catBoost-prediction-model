"""
Station Congestion Prediction
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.model_training import run_training_pipeline

CONFIG = {
    "input_file": "data/processed/features_station_model.csv",
    "output_dir": "outputs/congestion_model",
    "model_file": "models/catboost_congestion_model.cbm",
    "target_col": "active_chargers",
    "title": "Station Congestion Prediction",
    "model_name": "congestion",
    "exclude": ["split", "timestamp", "total_power_kw"],
    "cat_features": ["charging_station_id"],
    "clip_range": (0, None),
    "units": "chargers",
    "compute_mape": False,
    "palette": "rocket",
    "plots": ["importance", "avp"],
    "params": {
        "iterations": 1500,
        "learning_rate": 0.05,
        "depth": 8,
        "l2_leaf_reg": 3,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": 42,
        "verbose": 100,
        "early_stopping_rounds": 100,
    },
}

if __name__ == "__main__":
    run_training_pipeline(CONFIG)
