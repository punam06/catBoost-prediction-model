"""
Evaluation Module
==================
Metrics computation and visualization for CatBoost models.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap


def plot_feature_importance(model, feature_cols, output_dir, model_name,
                            palette="viridis", title=None):
    importance = model.get_feature_importance()
    df_imp = pd.DataFrame({"feature": feature_cols, "importance": importance})
    df_imp = df_imp.sort_values("importance", ascending=False).head(20)

    plt.figure(figsize=(10, 8))
    sns.barplot(x="importance", y="feature", data=df_imp, palette=palette)
    plt.title(f"CatBoost ({title or model_name}) — Feature Importances", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"catboost_{model_name}_feature_importance.png"), dpi=150)
    plt.close()


def plot_actual_vs_predicted(y_test, y_pred, output_dir, model_name,
                              title=None, units="Wh", clip_range=(0, None)):
    plt.figure(figsize=(8, 7))
    alpha = 0.5 if clip_range[1] == 100 else 0.3
    color = "purple" if clip_range[1] == 100 else "steelblue"
    plt.scatter(y_test, y_pred, alpha=alpha, s=15, color=color)

    if clip_range[1] == 100:
        plt.plot([0, 100], [0, 100], "r--", linewidth=1.5)
        plt.xlim(0, 100)
        plt.ylim(0, 100)
    else:
        lims = [0, max(y_test.max(), y_pred.max())]
        plt.plot(lims, lims, "r--", linewidth=1.5)

    plt.xlabel(f"Actual {units}")
    plt.ylabel(f"Predicted {units}")
    plt.title(f"CatBoost ({title or model_name}) — Actual vs Predicted", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"catboost_{model_name}_actual_vs_predicted.png"), dpi=150)
    plt.close()


def plot_residuals(y_test, y_pred, output_dir, model_name,
                   units="Wh", palette="viridis", title=None):
    residuals = y_test - y_pred
    color_map = {"viridis": "coral", "mako": "darkorange", "crest": "seagreen",
                 "magma": "purple", "rocket": "teal"}
    color = color_map.get(palette, "steelblue")
    plt.figure(figsize=(9, 5))
    sns.histplot(residuals, bins=80, kde=True, color=color)
    plt.axvline(0, color="black", linestyle="--")
    plt.xlabel(f"Residual (Actual - Predicted) {units}")
    plt.title(f"CatBoost ({title or model_name}) — Residual Distribution", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"catboost_{model_name}_residuals.png"), dpi=150)
    plt.close()


def plot_shap(model, test_pool, feature_cols, output_dir, model_name, title=None):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_pool)

    plt.figure()
    shap.summary_plot(shap_values, test_pool.get_features(),
                      feature_names=feature_cols, show=False, max_display=15)
    plt.title(f"CatBoost ({title or model_name}) — SHAP Feature Contributions", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"catboost_{model_name}_shap_summary.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
