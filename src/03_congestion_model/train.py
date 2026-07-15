"""
09_train_station_model.py
=========================
Trains a CatBoostRegressor to predict Station Congestion (active_chargers).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# ─── CONFIG ─────────────────────────────────────────────────────────────────
INPUT_FILE   = "data/processed/features_station_model.csv"
MODEL_DIR    = "models"
OUTPUT_DIR   = "outputs/congestion_model"
MODEL_FILE   = os.path.join(MODEL_DIR, "catboost_congestion_model.cbm")
RANDOM_SEED  = 42
TARGET_COL   = "active_chargers"

CATBOOST_PARAMS = {
    "iterations": 1500,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 3,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": RANDOM_SEED,
    "verbose": 100,
    "early_stopping_rounds": 100
}
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(filepath: str):
    print(f"\n{'='*60}")
    print("  CatBoost — Station Congestion Prediction")
    print(f"{'='*60}\n")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[ERROR] '{filepath}' not found.")
    df = pd.read_csv(filepath)
    print(f"[LOAD]  {filepath} ({len(df):,} rows)")
    return df

def split_and_prepare(df: pd.DataFrame):
    # Exclude leakages (total_power_kw) and metadata
    exclude = ["split", "timestamp", "total_power_kw", TARGET_COL]
    feature_cols = [c for c in df.columns if c not in exclude]
    cat_features = ["charging_station_id"]
    
    for col in feature_cols:
        if df[col].dtype == 'float64' and df[col].isna().sum() > 0:
            df[col] = df[col].fillna(-999)

    train_df = df[df["split"] == "train"]
    val_df   = df[df["split"] == "val"]
    test_df  = df[df["split"] == "test"]

    print(f"[SPLIT] Using pre-defined time-based splits:")
    print(f"        Train: {len(train_df):>8,} rows")
    print(f"        Val  : {len(val_df):>8,} rows")
    print(f"        Test : {len(test_df):>8,} rows\n")

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    X_val   = val_df[feature_cols]
    y_val   = val_df[TARGET_COL]
    X_test  = test_df[feature_cols]
    y_test  = test_df[TARGET_COL]

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool   = Pool(X_val, y_val, cat_features=cat_features)
    test_pool  = Pool(X_test, y_test, cat_features=cat_features)

    return train_pool, val_pool, test_pool, X_test, y_test, feature_cols

def train_model(train_pool, val_pool):
    print(f"[TRAIN] CatBoost Params: {CATBOOST_PARAMS}\n")
    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(train_pool, eval_set=val_pool)
    print(f"\n        Best iteration: {model.get_best_iteration()}")
    print(f"        Best val RMSE : {model.get_best_score()['validation']['RMSE']:,.3f} chargers\n")
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None) # Cannot have negative active chargers
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    
    print(f"[EVAL]  Test Set Metrics:")
    print(f"        RMSE  : {rmse:>12,.3f} chargers")
    print(f"        MAE   : {mae:>12,.3f} chargers")
    print(f"        R²    : {r2:>12.4f}\n")

    metrics_path = os.path.join(OUTPUT_DIR, "catboost_congestion_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("CatBoost — Station Congestion Prediction\n")
        f.write("=" * 40 + "\n")
        f.write(f"RMSE  : {rmse:,.3f} chargers\n")
        f.write(f"MAE   : {mae:,.3f} chargers\n")
        f.write(f"R2    : {r2:.4f}\n")
    
    return y_pred

def plot_feature_importance(model, feature_cols):
    importance = model.get_feature_importance()
    df_imp = pd.DataFrame({'feature': feature_cols, 'importance': importance})
    df_imp = df_imp.sort_values('importance', ascending=False).head(20)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=df_imp, palette='rocket')
    plt.title("CatBoost (Congestion) — Feature Importances", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "catboost_congestion_feature_importance.png"), dpi=150)
    plt.close()

def plot_actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize=(8, 7))
    plt.scatter(y_test, y_pred, alpha=0.3, s=15, color="teal")
    lim = max(y_test.max(), y_pred.max())
    plt.plot([0, lim], [0, lim], "r--", linewidth=1.5)
    plt.xlabel("Actual Active Chargers")
    plt.ylabel("Predicted Active Chargers")
    plt.title("CatBoost (Congestion) — Actual vs Predicted", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "catboost_congestion_actual_vs_predicted.png"), dpi=150)
    plt.close()

def save_model(model):
    model.save_model(MODEL_FILE)
    print(f"[SAVE]  Model saved to {MODEL_FILE}")

def main():
    ensure_dirs()
    df = load_data(INPUT_FILE)
    train_pool, val_pool, test_pool, X_test, y_test, feature_cols = split_and_prepare(df)
    
    model = train_model(train_pool, val_pool)
    y_pred = evaluate_model(model, X_test, y_test)
    
    print("[PLOTS] Generating visualisations...")
    plot_feature_importance(model, feature_cols)
    plot_actual_vs_predicted(y_test, y_pred)
    save_model(model)
    print("\n[DONE] Pipeline complete.")

if __name__ == "__main__":
    main()
