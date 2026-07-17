"""
Shared CatBoost Training Pipeline
==================================
Reusable functions for loading data, splitting, training, and saving models.
"""

import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore")


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def load_feature_data(filepath, title):
    print(f"\n{'='*60}")
    print(f"  CatBoost — {title}")
    print(f"{'='*60}\n")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[ERROR] '{filepath}' not found.")
    df = pd.read_csv(filepath)
    print(f"[LOAD]  {filepath} ({len(df):,} rows)")
    return df


def split_and_prepare(df, target_col, exclude=None, cat_features=None,
                      drop_target_nans=False, log_target=False):
    if exclude is None:
        exclude = ["split", "timestamp", target_col]
    elif target_col not in exclude:
        exclude = exclude + [target_col]

    if cat_features is None:
        cat_features = ["charging_station_id"]

    if drop_target_nans:
        before = len(df)
        df = df.dropna(subset=[target_col])
        if len(df) < before:
            print(f"[CLEAN] Dropped {before - len(df):,} rows with NaN target ({target_col})")

    feature_cols = [c for c in df.columns if c not in exclude]

    for col in feature_cols:
        if df[col].dtype == "float64" and df[col].isna().sum() > 0:
            df[col] = df[col].fillna(-999)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    print(f"[SPLIT] Using pre-defined time-based splits:")
    print(f"        Train: {len(train_df):>8,} rows")
    print(f"        Val  : {len(val_df):>8,} rows")
    print(f"        Test : {len(test_df):>8,} rows\n")

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    if log_target:
        y_train = np.log1p(y_train)
        y_val = np.log1p(y_val)
        print(f"[LOG]   Target log1p-transformed for training\n")

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    test_pool = Pool(X_test, y_test, cat_features=cat_features)

    return train_pool, val_pool, test_pool, X_test, y_test, feature_cols


def train_catboost(train_pool, val_pool, params):
    print(f"[TRAIN] CatBoost Params: {params}\n")
    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=val_pool)
    best_rmse = model.get_best_score()["validation"]["RMSE"]
    print(f"\n        Best iteration: {model.get_best_iteration()}")
    print(f"        Best val RMSE : {best_rmse:,.2f}\n")
    return model


def evaluate_model(model, X_test, y_test, clip_range=(0, None),
                   units="Wh", output_dir=None, model_name="model",
                   compute_mape=False, log_target=False, y_test_raw=None):
    y_pred = model.predict(X_test)

    if log_target:
        y_pred = np.expm1(y_pred)
        if y_test_raw is not None:
            y_test = y_test_raw

    y_pred = np.clip(y_pred, clip_range[0], clip_range[1])

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"[EVAL]  Test Set Metrics:")
    print(f"        RMSE  : {rmse:>12,.2f} {units}")
    print(f"        MAE   : {mae:>12,.2f} {units}")
    print(f"        R²    : {r2:>12.4f}")

    metrics = {"rmse": rmse, "mae": mae, "r2": r2}

    if compute_mape:
        nonzero = y_test > 0
        if nonzero.sum() > 0:
            mape = (np.abs((y_test[nonzero] - y_pred[nonzero]) / y_test[nonzero]).mean() * 100)
        else:
            mape = np.nan
        print(f"        MAPE  : {mape:>12.2f} %")
        metrics["mape"] = mape
    else:
        mape = None

    print()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, f"catboost_{model_name}_metrics.txt")
        with open(metrics_path, "w") as f:
            f.write(f"CatBoost — {model_name.replace('_', ' ').title()} Prediction\n")
            f.write("=" * 40 + "\n")
            f.write(f"RMSE  : {rmse:,.2f} {units}\n")
            f.write(f"MAE   : {mae:,.2f} {units}\n")
            f.write(f"R2    : {r2:.4f}\n")
            if mape is not None:
                f.write(f"MAPE  : {mape:.2f}%\n")

    return y_pred


def save_catboost_model(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save_model(filepath)
    print(f"[SAVE]  Model saved to {filepath}")


def run_training_pipeline(config):
    """Run full train→evaluate→plot→save pipeline from a config dict.

    config keys:
        input_file, output_dir, model_file, target_col, title,
        params (CatBoost params dict),
        exclude (list, optional), cat_features (list, optional),
        clip_range (tuple, optional), units (str, optional),
        compute_mape (bool, optional), palette (str, optional),
        drop_target_nans (bool, optional), log_target (bool, optional),
        plots (list of str: "importance", "avp", "residuals", "shap")
    """
    ensure_dirs(os.path.dirname(config["model_file"]), config["output_dir"])

    log_target = config.get("log_target", False)
    df = load_feature_data(config["input_file"], config["title"])

    train_pool, val_pool, test_pool, X_test, y_test, feature_cols = split_and_prepare(
        df,
        target_col=config["target_col"],
        exclude=config.get("exclude"),
        cat_features=config.get("cat_features"),
        drop_target_nans=config.get("drop_target_nans", False),
        log_target=log_target,
    )

    # Keep raw y_test for inverse transform during evaluation
    y_test_raw = None
    if log_target:
        y_test_raw = df[df["split"] == "test"][config["target_col"]].values

    model = train_catboost(train_pool, val_pool, config["params"])

    y_pred = evaluate_model(
        model, X_test, y_test,
        clip_range=config.get("clip_range", (0, None)),
        units=config.get("units", "Wh"),
        output_dir=config["output_dir"],
        model_name=config.get("model_name", "model"),
        compute_mape=config.get("compute_mape", False),
        log_target=log_target,
        y_test_raw=y_test_raw,
    )

    from src.evaluation import plot_feature_importance, plot_actual_vs_predicted, \
        plot_residuals, plot_shap

    plots = config.get("plots", ["importance", "avp"])
    palette = config.get("palette", "viridis")
    model_name = config.get("model_name", "model")
    title = config.get("title", model_name)

    # Use raw y_test for plots when log_target is used
    y_test_plot = y_test_raw if log_target else y_test

    print("[PLOTS] Generating visualisations...")
    if "importance" in plots:
        plot_feature_importance(model, feature_cols, config["output_dir"],
                                model_name, palette, title)
    if "avp" in plots:
        plot_actual_vs_predicted(y_test_plot, y_pred, config["output_dir"],
                                 model_name, title,
                                 config.get("units", "Wh"),
                                 config.get("clip_range", (0, None)))
    if "residuals" in plots:
        plot_residuals(y_test_plot, y_pred, config["output_dir"],
                       model_name, config.get("units", "Wh"), palette, title)
    if "shap" in plots:
        try:
            plot_shap(model, test_pool, feature_cols, config["output_dir"],
                      model_name, title)
        except Exception as e:
            print(f"        SHAP plot failed (skipping): {e}")

    save_catboost_model(model, config["model_file"])
    print("\n[DONE] Pipeline complete.")

    return model, y_pred
