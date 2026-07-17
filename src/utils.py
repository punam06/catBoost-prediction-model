"""
Utility Module
===============
Shared helpers for prediction scripts and data processing.
"""

import math
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


def load_catboost_model(model_path):
    model = CatBoostRegressor()
    model.load_model(model_path)
    return model


def compute_temporal_features(hour, date_str):
    dt = pd.to_datetime(date_str)
    return {
        "hour": hour,
        "day_of_week": dt.dayofweek,
        "month": dt.month,
        "is_weekend": int(dt.dayofweek >= 5),
        "hour_sin": round(math.sin(2 * math.pi * hour / 24), 4),
        "hour_cos": round(math.cos(2 * math.pi * hour / 24), 4),
        "collection_period": 3,
    }


def interactive_input(prompt, default=None, cast=str):
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"{prompt}{suffix}: ").strip()
    if not val and default is not None:
        return cast(default) if cast else default
    return cast(val)
