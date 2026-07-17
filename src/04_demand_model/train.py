"""
Demand Forecasting
====================
Target: total_energy_wh (station-level hourly demand)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.model_training import run_training_pipeline

CONFIG = {
    "input_file": "data/processed/features_demand_model.csv",
    "output_dir": "outputs/demand_model",
    "model_file": "models/catboost_demand_model.cbm",
    "target_col": "total_energy_wh",
    "title": "Station-Level Hourly Demand Prediction",
    "model_name": "demand",
    "cat_features": ["charging_station_id"],
    "clip_range": (0, None),
    "units": "Wh",
    "compute_mape": True,
    "palette": "mako",
    "plots": ["importance", "avp", "residuals", "shap"],
    "params": {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 8,
        "l2_leaf_reg": 3,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": 42,
        "verbose": 100,
        "early_stopping_rounds": 50,
    },
}

if __name__ == "__main__":
    run_training_pipeline(CONFIG)
