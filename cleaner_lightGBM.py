"""
lightgbm_energy_prediction.py
==============================
LightGBM Energy Consumption Prediction Model.
Target  : max_energy_wh  (Wh consumed per charger per interval)
Input   : All_Merger.csv (raw — this script cleans it internally)

Why separate cleaning for LightGBM?
  - LightGBM natively handles NaN in FEATURES → no imputation needed
  - LightGBM requires categorical columns as 'category' dtype or int
  - LightGBM is less sensitive to outliers (tree-based splits) →
    we use a softer P99.5 cap instead of P99
  - We keep the soc_pct column with NaNs as-is (LightGBM tolerates it)
  - No station capping needed here — LightGBM handles imbalance via
    sample_weight or is robust to it through regularisation

Pipeline:
  1. Load raw data
  2. LightGBM-specific cleaning
  3. Feature engineering
  4. Encode categoricals (category dtype)
  5. Period-aware train / validation / test split
  6. Train LightGBM Regressor with early stopping
  7. Evaluate: RMSE, MAE, R2, MAPE
  8. Visualise: Feature importance, Actual vs Predicted, Residuals, SHAP
  9. Save model → models/lightgbm_energy_model.txt
"""

import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ─── CONFIG ─────────────────────────────────────────────────────────────────
RAW_FILE     = "All_Merger.csv"
MODEL_DIR    = "models"
OUTPUT_DIR   = "outputs"
MODEL_FILE   = os.path.join(MODEL_DIR, "lightgbm_energy_model.txt")
RANDOM_SEED  = 42
TARGET_COL   = "max_energy_wh"

# LightGBM hyperparameters
LGBM_PARAMS = {
    "objective"         : "regression",
    "metric"            : ["rmse", "mae"],
    "boosting_type"     : "gbdt",
    "num_leaves"        : 63,        # Tree complexity (higher = more complex)
    "max_depth"         : -1,        # No depth limit; num_leaves controls it
    "learning_rate"     : 0.05,
    "n_estimators"      : 1000,
    "min_child_samples" : 20,        # Min data points per leaf (anti-overfitting)
    "subsample"         : 0.8,       # Row sub-sampling fraction per tree
    "colsample_bytree"  : 0.8,       # Feature sub-sampling fraction per tree
    "reg_alpha"         : 0.1,       # L1 regularisation
    "reg_lambda"        : 0.1,       # L2 regularisation
    "random_state"      : RANDOM_SEED,
    "n_jobs"            : -1,
    "verbose"           : -1,
}
EARLY_STOPPING_ROUNDS = 50
# ─────────────────────────────────────────────────────────────────────────────


# =============================================================================
# STEP 1 — LOAD RAW DATA
# =============================================================================
def load_raw(filepath: str) -> pd.DataFrame:
    """Load All_Merger.csv and parse the timestamp column."""
    print(f"\n{'='*62}")
    print("  LightGBM — Energy Consumption Prediction")
    print(f"{'='*62}\n")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[ERROR] '{filepath}' not found.")

    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)

    print(f"[1. LOAD]   {filepath}")
    print(f"            {len(df):,} rows  x  {df.shape[1]} columns\n")
    return df


# =============================================================================
# STEP 2 — LIGHTGBM-SPECIFIC DATA CLEANING
# =============================================================================
def clean_for_lgbm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data cleaning tailored for LightGBM.

    Key differences from the CatBoost cleaner (cleaner.py):
      - NaN in feature columns are KEPT (LightGBM handles them internally
        by learning the best split direction for missing values)
      - Outlier cap is P99.5 instead of P99 (LightGBM is more robust to
        extreme values because splits are based on thresholds, not distances)
      - NO station row capping — LightGBM regularisation (reg_alpha,
        reg_lambda, min_child_samples) handles station imbalance
      - soc_pct NaNs are left as-is (it is only a future target, not a feature)
    """
    print(f"[2. CLEAN]  Applying LightGBM-specific cleaning ...")
    print(f"            Raw rows: {len(df):,}")
    print(f"            {'─'*46}")

    # 2a. Remove exact full-row duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"            Exact duplicates removed  : -{before - len(df):>6,}  → {len(df):,}")

    # 2b. Resolve near-duplicates (same station+charger+timestamp)
    #     Keep the row with the highest max_energy_wh
    before = len(df)
    key = ["charging_station_id", "charger_id", "timestamp"]
    df = (df.sort_values(TARGET_COL, ascending=False)
            .drop_duplicates(subset=key, keep="first")
            .reset_index(drop=True))
    print(f"            Near-duplicates resolved  : -{before - len(df):>6,}  → {len(df):,}")

    # 2c. Remove physically impossible negative values
    #     (negative power or energy violates physics)
    before = len(df)
    df = df[(df["avg_pwr_kw"] >= 0) & (df[TARGET_COL] >= 0)].reset_index(drop=True)
    print(f"            Negative values removed   : -{before - len(df):>6,}  → {len(df):,}")

    # 2d. Remove rows where active_session_count == 0 (no charging session)
    before = len(df)
    df = df[df["active_session_count"] != 0].reset_index(drop=True)
    print(f"            No-session rows removed   : -{before - len(df):>6,}  → {len(df):,}")

    # 2e. Remove dead rows: avg_pwr_kw == 0 AND max_energy_wh == 0
    #     These have no signal for prediction
    before = len(df)
    df = df[~((df["avg_pwr_kw"] == 0) & (df[TARGET_COL] == 0))].reset_index(drop=True)
    print(f"            Dead rows removed (0/0)   : -{before - len(df):>6,}  → {len(df):,}")

    # 2f. Outlier capping at P99.5 (softer than CatBoost's P99)
    #     LightGBM handles extreme values better; we cap less aggressively
    cap_energy = df[TARGET_COL].quantile(0.995)
    cap_power  = df["avg_pwr_kw"].quantile(0.995)
    df[TARGET_COL]   = df[TARGET_COL].clip(upper=cap_energy)
    df["avg_pwr_kw"] = df["avg_pwr_kw"].clip(upper=cap_power)
    print(f"            Outlier cap (P99.5):  energy ≤ {cap_energy:,.0f} Wh | power ≤ {cap_power:,.2f} kW")
    print(f"            (clip only — no rows removed)")

    # 2g. Fix integer columns stored as float64
    int_cols = ["hour", "day_of_week", "month", "is_weekend", "active_session_count"]
    for col in int_cols:
        df[col] = df[col].astype(int)

    # 2h. Add collection_period label (non-continuous temporal windows)
    #     CRITICAL: data has gaps of 4 days (Nov→Dec) and 113 days (Dec→Apr)
    #     This label lets the model distinguish each collection window
    period_map = {11: 1, 12: 2, 4: 3}
    df["collection_period"] = df["month"].map(period_map).fillna(3).astype(int)

    print(f"            {'─'*46}")
    print(f"            Clean rows : {len(df):,}")
    print(f"            Null counts (NaN kept by design for LightGBM):")
    for col in df.columns:
        n = df[col].isnull().sum()
        if n > 0:
            print(f"              {col:<28}: {n:,} ({n/len(df)*100:.1f}%)")
    print()
    return df


# =============================================================================
# STEP 3 — FEATURE ENGINEERING
# =============================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features optimised for LightGBM.

    IMPORTANT: No lag/rolling features across collection periods.
    The dataset is non-continuous (113-day gap between Dec 2025 and Apr 2026).
    All engineered features must be within-row or static aggregates.
    """
    print(f"[3. FEAT]   Engineering features ...")

    # --- Within-row features ---
    # Hour bucket: 0=midnight, 1=early-night, 2=late-night (only hours 0,1,2 exist)
    df["is_late_night"] = (df["hour"] >= 2).astype(int)

    # Power-to-energy ratio: how fast energy was delivered per kW
    df["power_to_energy_ratio"] = np.where(
        df[TARGET_COL] > 0,
        df["avg_pwr_kw"] / (df[TARGET_COL] / 1000.0),
        np.nan   # NaN when target is zero — LightGBM handles this
    )

    # Interaction: collection_period × hour
    df["period_hour_interaction"] = df["collection_period"] * 10 + df["hour"]

    # --- Station-level aggregate features ---
    st = (df.groupby("charging_station_id")
            .agg(
                station_mean_energy_wh    = (TARGET_COL,   "mean"),
                station_median_energy_wh  = (TARGET_COL,   "median"),
                station_std_energy_wh     = (TARGET_COL,   "std"),
                station_mean_power_kw     = ("avg_pwr_kw", "mean"),
                station_max_power_kw      = ("avg_pwr_kw", "max"),
                station_charger_count     = ("charger_id", "nunique"),
            )
            .reset_index())
    df = df.merge(st, on="charging_station_id", how="left")

    # --- Charger-level aggregate features ---
    ch = (df.groupby("charger_id")
            .agg(
                charger_mean_energy_wh = (TARGET_COL,   "mean"),
                charger_max_energy_wh  = (TARGET_COL,   "max"),
                charger_mean_power_kw  = ("avg_pwr_kw", "mean"),
            )
            .reset_index())
    df = df.merge(ch, on="charger_id", how="left")

    # Estimated charger capacity (max seen × 1.1 buffer) + leftover energy
    df["estimated_capacity_wh"] = df["charger_max_energy_wh"] * 1.1
    df["leftover_energy_wh"]    = (
        df["estimated_capacity_wh"] - df[TARGET_COL]
    ).clip(lower=0)

    print(f"            Features built. Shape: {df.shape}\n")
    return df


# =============================================================================
# STEP 4 — ENCODE CATEGORICALS FOR LIGHTGBM
# =============================================================================
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert string categorical columns to pandas 'category' dtype.
    LightGBM uses this dtype to apply its own internal optimal categorical
    handling (finds best splits across categories efficiently).
    This is better than label encoding for high-cardinality categoricals
    like charging_station_id (1,286 values) and charger_id (7,600 values).
    """
    print(f"[4. ENCODE] Converting categoricals to 'category' dtype ...")
    cat_cols = ["charging_station_id", "charger_id"]
    for col in cat_cols:
        df[col] = df[col].astype("category")
        print(f"            {col}: {df[col].nunique():,} unique categories")
    print()
    return df


# =============================================================================
# STEP 5 — TRAIN / VALIDATION / TEST SPLIT
# =============================================================================
def split_data(df: pd.DataFrame) -> tuple:
    """
    Period-aware chronological split to prevent data leakage.

    Data windows (non-continuous):
      Period 1 — Nov 2025  (2 unique days)
      Period 2 — Dec 2025  (4 unique days)
      Period 3 — Apr 2026  (5 unique days)

    Split strategy:
      Train      : Period 1 + Period 2 + first 80% of Period 3
      Validation : Middle 10% of Period 3  (LightGBM early stopping)
      Test       : Last 10% of Period 3    (final evaluation)
    """
    feature_cols = [
        # Identifiers (category dtype)
        "charging_station_id", "charger_id",
        # Time
        "collection_period", "hour", "day_of_week", "month",
        "is_weekend", "is_late_night", "hour_sin", "hour_cos",
        "period_hour_interaction",
        # Raw metrics
        "avg_pwr_kw", "active_session_count",
        # Engineered
        "power_to_energy_ratio",
        "station_mean_energy_wh", "station_median_energy_wh",
        "station_std_energy_wh",  "station_mean_power_kw",
        "station_max_power_kw",   "station_charger_count",
        "charger_mean_energy_wh", "charger_max_energy_wh",
        "charger_mean_power_kw",
        "estimated_capacity_wh",
        # soc_pct is NOT a feature for this model (96.5% null, future target)
    ]

    # Periods 1 & 2 → all to train
    train_base = df[df["collection_period"].isin([1, 2])]

    # Period 3 → sort chronologically → 80/10/10
    p3   = df[df["collection_period"] == 3].sort_values("timestamp")
    n3   = len(p3)
    c80  = int(n3 * 0.80)
    c90  = int(n3 * 0.90)

    p3_train = p3.iloc[:c80]
    p3_val   = p3.iloc[c80:c90]
    p3_test  = p3.iloc[c90:]

    train_df = pd.concat([train_base, p3_train], axis=0)
    val_df   = p3_val
    test_df  = p3_test

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    X_val   = val_df[feature_cols]
    y_val   = val_df[TARGET_COL]
    X_test  = test_df[feature_cols]
    y_test  = test_df[TARGET_COL]

    print(f"[5. SPLIT]")
    print(f"            Train      : {len(X_train):>8,} rows  "
          f"(Period 1+2 + 80% of Period 3)")
    print(f"            Validation : {len(X_val):>8,} rows  "
          f"(10% of Period 3 — early stopping)")
    print(f"            Test       : {len(X_test):>8,} rows  "
          f"(last 10% of Period 3)\n")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


# =============================================================================
# STEP 6 — TRAIN MODEL
# =============================================================================
def train_model(X_train, y_train, X_val, y_val) -> lgb.LGBMRegressor:
    """
    Train LightGBM with early stopping on the validation set.
    category dtype columns are auto-detected by LightGBM.
    """
    print(f"[6. TRAIN]")
    print(f"            Params: {LGBM_PARAMS}\n")

    model = lgb.LGBMRegressor(**LGBM_PARAMS)

    callbacks = [
        lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True),
        lgb.log_evaluation(period=100),
    ]

    model.fit(
        X_train, y_train,
        eval_set  = [(X_val, y_val)],
        eval_names= ["validation"],
        callbacks = callbacks,
    )

    print(f"\n            Best iteration : {model.best_iteration_}")
    print(f"            Best val RMSE  : {model.best_score_['validation']['rmse']:,.2f} Wh\n")
    return model


# =============================================================================
# STEP 7 — EVALUATE
# =============================================================================
def evaluate_model(model, X_test, y_test) -> tuple:
    """Compute and print RMSE, MAE, R², MAPE on the held-out test set."""
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)   # No negative energy predictions

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    nonzero = y_test > 0
    mape = (np.abs((y_test[nonzero] - y_pred[nonzero]) / y_test[nonzero]).mean() * 100)

    print(f"[7. EVAL]   Test set results:")
    print(f"            RMSE  : {rmse:>12,.2f} Wh")
    print(f"            MAE   : {mae:>12,.2f} Wh")
    print(f"            R²    : {r2:>12.4f}")
    print(f"            MAPE  : {mape:>12.2f} %  (non-zero actuals only)\n")

    # Save metrics
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "lgbm_energy_metrics.txt")
    with open(path, "w") as f:
        f.write("LightGBM — Energy Consumption Prediction\n")
        f.write("=" * 40 + "\n")
        f.write(f"RMSE  : {rmse:,.2f} Wh\n")
        f.write(f"MAE   : {mae:,.2f} Wh\n")
        f.write(f"R2    : {r2:.4f}\n")
        f.write(f"MAPE  : {mape:.2f}%\n")
    print(f"            Metrics saved → {path}")

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}, y_pred


# =============================================================================
# STEP 8 — VISUALISE
# =============================================================================
def plot_feature_importance(model, feature_cols: list) -> None:
    """Top-25 feature importances (split gain)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    imp = pd.DataFrame({
        "feature"   : feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(25)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=imp, x="importance", y="feature", palette="viridis")
    plt.title("LightGBM — Top 25 Feature Importances\n(Energy Consumption)", fontsize=13)
    plt.xlabel("Importance (split gain)")
    plt.ylabel("Feature")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "lgbm_feature_importance.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"            Feature importance  → {out}")


def plot_actual_vs_predicted(y_test, y_pred) -> None:
    """Scatter plot: actual vs predicted energy."""
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.figure(figsize=(8, 7))
    plt.scatter(y_test, y_pred, alpha=0.3, s=10, color="steelblue")
    plt.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
    plt.xlabel("Actual max_energy_wh (Wh)")
    plt.ylabel("Predicted max_energy_wh (Wh)")
    plt.title("LightGBM — Actual vs Predicted\n(Energy Consumption)", fontsize=13)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "lgbm_actual_vs_predicted.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"            Actual vs Predicted → {out}")


def plot_residuals(y_test, y_pred) -> None:
    """Residual distribution (actual − predicted)."""
    residuals = y_test.values - y_pred
    plt.figure(figsize=(9, 5))
    sns.histplot(residuals, bins=80, kde=True, color="coral")
    plt.axvline(0, color="black", linewidth=1.5, linestyle="--")
    plt.xlabel("Residual (Actual − Predicted) in Wh")
    plt.ylabel("Count")
    plt.title("LightGBM — Residual Distribution\n(Energy Consumption)", fontsize=13)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "lgbm_residuals.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"            Residuals          → {out}")


def plot_shap(model, X_test, feature_cols: list) -> None:
    """SHAP summary (sampled to 500 rows for speed)."""
    sample      = X_test.sample(n=min(500, len(X_test)), random_state=RANDOM_SEED)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    plt.figure()
    shap.summary_plot(shap_values, sample, feature_names=feature_cols,
                      show=False, max_display=20)
    plt.title("LightGBM — SHAP Feature Contributions", fontsize=12)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "lgbm_shap_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"            SHAP summary       → {out}")


# =============================================================================
# STEP 9 — SAVE MODEL
# =============================================================================
def save_model(model: lgb.LGBMRegressor) -> None:
    """Persist the trained booster as a .txt file."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.booster_.save_model(MODEL_FILE)
    print(f"\n[9. SAVE]   Model → {MODEL_FILE}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    # 1. Load raw data
    df = load_raw(RAW_FILE)

    # 2. LightGBM-specific cleaning
    df = clean_for_lgbm(df)

    # 3. Feature engineering
    df = engineer_features(df)

    # 4. Encode categoricals
    df = encode_categoricals(df)

    # 5. Train / val / test split
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = split_data(df)

    # 6. Train model
    model = train_model(X_train, y_train, X_val, y_val)

    # 7. Evaluate
    metrics, y_pred = evaluate_model(model, X_test, y_test)

    # 8. Visualise
    print(f"\n[8. PLOTS]")
    plot_feature_importance(model, feature_cols)
    plot_actual_vs_predicted(y_test, y_pred)
    plot_residuals(y_test, y_pred)
    plot_shap(model, X_test, feature_cols)

    # 9. Save model
    save_model(model)

    print(f"\n{'='*62}")
    print(f"  Pipeline complete.")
    print(f"  Model   : {MODEL_FILE}")
    print(f"  Outputs : {OUTPUT_DIR}/lgbm_*.png  |  lgbm_energy_metrics.txt")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
