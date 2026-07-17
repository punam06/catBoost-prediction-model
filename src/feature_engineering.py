"""
Feature Engineering Module
============================
Reusable helpers for feature preparation scripts.
"""

import numpy as np
import pandas as pd


def add_idle_flag(df):
    df = df.copy()
    df["is_idle_charger"] = ((df.get("active_session_count", 0) == 0) |
                              (df.get("session_count", 0) == 0)).astype(int)
    return df


def compute_hour_sin_cos(hour):
    return {
        "hour_sin": round(np.sin(2 * np.pi * hour / 24), 4),
        "hour_cos": round(np.cos(2 * np.pi * hour / 24), 4),
    }


def period_aware_split(df, p3_train_pct=0.8):
    """Time-based split: P1+P2+80%P3 → train, next 10%P3 → val, last 10%P3 → test."""
    df = df.copy()
    df["collection_period"] = df["collection_period"].astype(int)
    periods = sorted(df["collection_period"].unique())

    if len(periods) >= 3:
        p1_p2 = df[df["collection_period"].isin(periods[:2])]
        p3 = df[df["collection_period"] == periods[2]]
        split_idx = int(len(p3) * p3_train_pct)
        val_idx = int(len(p3) * (p3_train_pct + (1 - p3_train_pct) / 2))

        train = pd.concat([p1_p2, p3.iloc[:split_idx]])
        val = p3.iloc[split_idx:val_idx]
        test = p3.iloc[val_idx:]
    else:
        n = len(df)
        train = df.iloc[:int(n * 0.8)]
        val = df.iloc[int(n * 0.8):int(n * 0.9)]
        test = df.iloc[int(n * 0.9):]

    train = train.copy(); train["split"] = "train"
    val = val.copy(); val["split"] = "val"
    test = test.copy(); test["split"] = "test"

    return pd.concat([train, val, test]).reset_index(drop=True)


def compute_train_aggregates(train_df, group_col, agg_dict, val_df, test_df):
    """Compute aggregates on train split only, then merge to all splits."""
    aggs = train_df.groupby(group_col).agg(agg_dict).reset_index()
    return aggs
