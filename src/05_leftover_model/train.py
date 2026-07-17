"""
train_leftover_model.py
=========================
Trains a CatBoostRegressor to predict Leftover Energy (leftover_energy_wh).

Pipeline:
  1. Load pre-split features
  2. Train CatBoost Regressor
  3. Evaluate: RMSE, MAE, R2
  4. Visualise: Feature importance, Actual vs Predicted, Residuals, SHAP
  5. Save model → models/catboost_leftover_model.cbm
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
INPUT_FILE   = "data/processed/features_leftover_model.csv"
MODEL_DIR    = "models"
OUTPUT_DIR   = "outputs/leftover_model"
MODEL_FILE   = os.path.join(MODEL_DIR, "catboost_leftover_model.cbm")
RANDOM_SEED  = 42
TARGET_COL   = "leftover_energy_wh"

CATBOOST_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 3,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": RANDOM_SEED,
    "verbose": 100,
    "early_stopping_rounds": 50
}
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(filepath: str):
    print(f"\n{'='*60}")
    print("  CatBoost — Leftover Energy Prediction")
    print(f"{'='*60}\n")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[ERROR] '{filepath}' not found. Run prepare_leftover_features.py first.")
    df = pd.read_csv(filepath)
    print(f"[LOAD]  {filepath} ({len(df):,} rows)")
    return df

def split_and_prepare(df: pd.DataFrame):
    exclude = ["split", "timestamp", TARGET_COL]
    feature_cols = [c for c in df.columns if c not in exclude]
    cat_features = ["charging_station_id", "charger_id"]

    for col in feature_cols:
        if df[col].dtype == "float64" and df[col].isna().sum() > 0:
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
    print(f"        Best val RMSE : {model.get_best_score()['validation']['RMSE']:,.2f} Wh\n")
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print(f"[EVAL]  Test Set Metrics:")
    print(f"        RMSE  : {rmse:>12,.2f} Wh")
    print(f"        MAE   : {mae:>12,.2f} Wh")
    print(f"        R²    : {r2:>12.4f}\n")

    metrics_path = os.path.join(OUTPUT_DIR, "catboost_leftover_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("CatBoost — Leftover Energy Prediction\n")
        f.write("=" * 40 + "\n")
        f.write(f"RMSE  : {rmse:,.2f} Wh\n")
        f.write(f"MAE   : {mae:,.2f} Wh\n")
        f.write(f"R2    : {r2:.4f}\n")

    return y_pred

def plot_feature_importance(model, feature_cols):
    importance = model.get_feature_importance()
    df_imp = pd.DataFrame({"feature": feature_cols, "importance": importance})
    df_imp = df_imp.sort_values("importance", ascending=False).head(20)

    plt.figure(figsize=(10, 8))
    sns.barplot(x="importance", y="feature", data=df_imp, palette="crest")
    plt.title("CatBoost (Leftover Energy) — Feature Importances", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "catboost_leftover_feature_importance.png"), dpi=150)
    plt.close()

def plot_actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize=(8, 7))
    plt.scatter(y_test, y_pred, alpha=0.3, s=15, color="seagreen")
    lims = [0, max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", linewidth=1.5)
    plt.xlabel("Actual Leftover Energy (Wh)")
    plt.ylabel("Predicted Leftover Energy (Wh)")
    plt.title("CatBoost (Leftover Energy) — Actual vs Predicted", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "catboost_leftover_actual_vs_predicted.png"), dpi=150)
    plt.close()

def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    plt.figure(figsize=(9, 5))
    sns.histplot(residuals, bins=80, kde=True, color="seagreen")
    plt.axvline(0, color="black", linestyle="--")
    plt.xlabel("Residual (Actual - Predicted) Wh")
    plt.title("CatBoost (Leftover Energy) — Residual Distribution", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "catboost_leftover_residuals.png"), dpi=150)
    plt.close()

def plot_shap(model, test_pool, feature_cols):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_pool)

    plt.figure()
    shap.summary_plot(shap_values, test_pool.get_features(), feature_names=feature_cols, show=False, max_display=15)
    plt.title("CatBoost (Leftover Energy) — SHAP Feature Contributions", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "catboost_leftover_shap_summary.png"), dpi=150, bbox_inches="tight")
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
    plot_residuals(y_test, y_pred)
    try:
        plot_shap(model, test_pool, feature_cols)
    except Exception as e:
        print(f"        SHAP plot failed (skipping): {e}")

    save_model(model)
    print("\n[DONE] Pipeline complete.")

if __name__ == "__main__":
    main()
