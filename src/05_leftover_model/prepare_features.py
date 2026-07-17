"""
prepare_leftover_features.py
==============================
Prepares features for Leftover Energy Prediction at charger level.
Target: leftover_energy_wh = (charger_max_energy_wh * 1.1) - max_energy_wh

Data leakage prevention:
  - charger_max_energy_wh computed on TRAINING split only
  - soc_pct excluded (96.5% null, not relevant)

Source  : data/processed/cleaned_data.csv
Output  : data/processed/features_leftover_model.csv
"""

import os
import pandas as pd
import numpy as np

# ─── CONFIG ───────────────────────────────────────────────────────────────────
INPUT_FILE  = "data/processed/cleaned_data.csv"
OUTPUT_FILE = "data/processed/features_leftover_model.csv"
RANDOM_SEED = 42
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  prepare_leftover_features.py — Leftover Energy Prep")
    print("=" * 62 + "\n")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Missing {INPUT_FILE}. Run 00_data_cleaning/cleaner.py first.")

    df = pd.read_csv(INPUT_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Drop soc_pct — 96.5% null, future model target
    if "soc_pct" in df.columns:
        df = df.drop(columns=["soc_pct"])
        print("[INIT]   Dropped soc_pct (96.5% null)")

    print(f"[LOAD]  Loaded {len(df):,} charger-level rows.\n")

    # ── 1. ADD IDLE FLAG ───────────────────────────────────────────────────
    df["is_idle_charger"] = (df["avg_pwr_kw"] == 0).astype(int)
    idle_count = df["is_idle_charger"].sum()
    print(f"[FEAT]  is_idle_charger flag added")
    print(f"        Idle rows: {idle_count:,} ({idle_count / len(df) * 100:.1f}%)")
    print(f"        Active   : {len(df) - idle_count:,} ({(len(df) - idle_count) / len(df) * 100:.1f}%)\n")

    # ── 2. TIME-BASED SPLIT FIRST (needed for train-only aggregates) ───────
    df = df.sort_values(by="timestamp").reset_index(drop=True)

    period_1_2 = df[df["collection_period"].isin([1, 2])].copy()
    period_3   = df[df["collection_period"] == 3].copy()

    p3_total      = len(period_3)
    p3_train_end  = int(p3_total * 0.8)
    p3_val_end    = int(p3_total * 0.9)

    p3_train = period_3.iloc[:p3_train_end]
    p3_val   = period_3.iloc[p3_train_end:p3_val_end]
    p3_test  = period_3.iloc[p3_val_end:]

    train_df = pd.concat([period_1_2, p3_train]).copy()
    val_df   = p3_val.copy()
    test_df  = p3_test.copy()

    print(f"[SPLIT] Period-aware temporal split:")
    print(f"        Train: {len(train_df):>8,} rows")
    print(f"        Val  : {len(val_df):>8,} rows")
    print(f"        Test : {len(test_df):>8,} rows\n")

    # ── 3. TRAIN-ONLY AGGREGATES ───────────────────────────────────────────
    print("[FEAT]  Computing charger aggregates on TRAIN split only...")

    charger_agg = (
        train_df.groupby("charger_id")
        .agg(
            charger_max_energy_wh=("max_energy_wh", "max"),
            charger_mean_energy_wh=("max_energy_wh", "mean"),
        )
        .reset_index()
    )

    print(f"        Charger aggregates built from {len(train_df):,} train rows.")
    print(f"        Unique chargers in train: {charger_agg['charger_id'].nunique():,}")

    # Join aggregates to all splits
    for split_df in [train_df, val_df, test_df]:
        split_df.drop(
            columns=[c for c in split_df.columns if c in charger_agg.columns and c != "charger_id"],
            inplace=True,
            errors="ignore",
        )

    train_df = train_df.merge(charger_agg, on="charger_id", how="left")
    val_df   = val_df.merge(charger_agg, on="charger_id", how="left")
    test_df  = test_df.merge(charger_agg, on="charger_id", how="left")

    val_new  = val_df["charger_max_energy_wh"].isna().sum()
    test_new = test_df["charger_max_energy_wh"].isna().sum()
    print(f"        Val  rows with unseen charger (NaN): {val_new}")
    print(f"        Test rows with unseen charger (NaN): {test_new}\n")

    # ── 4. COMPUTE TARGET ──────────────────────────────────────────────────
    # estimated_capacity_wh = charger's historical max * 1.1 buffer
    # leftover_energy_wh = capacity - current session energy
    for split_df in [train_df, val_df, test_df]:
        split_df["estimated_capacity_wh"] = split_df["charger_max_energy_wh"] * 1.1
        split_df["leftover_energy_wh"] = (
            split_df["estimated_capacity_wh"] - split_df["max_energy_wh"]
        ).clip(lower=0)

    print("[TARGET] leftover_energy_wh computed for all splits.")
    print(f"         Train target mean: {train_df['leftover_energy_wh'].mean():>10,.0f} Wh")
    print(f"         Val   target mean: {val_df['leftover_energy_wh'].mean():>10,.0f} Wh")
    print(f"         Test  target mean: {test_df['leftover_energy_wh'].mean():>10,.0f} Wh\n")

    # ── 5. LABEL SPLITS AND COMBINE ────────────────────────────────────────
    train_df["split"] = "train"
    val_df["split"]   = "val"
    test_df["split"]  = "test"

    final_df = pd.concat([train_df, val_df, test_df], axis=0).reset_index(drop=True)

    # ── 6. COLUMN ORDER ────────────────────────────────────────────────────
    feature_cols = [
        # Identifiers
        "charging_station_id", "charger_id",
        # Temporal
        "hour", "day_of_week", "month", "is_weekend",
        "hour_sin", "hour_cos", "collection_period",
        # Metrics
        "avg_pwr_kw", "active_session_count", "is_idle_charger", "max_energy_wh",
        # Train-only aggregates
        "charger_max_energy_wh", "charger_mean_energy_wh", "estimated_capacity_wh",
        # Metadata
        "split", "timestamp",
        # TARGET
        "leftover_energy_wh",
    ]
    final_df = final_df[feature_cols]

    # ── 7. SAVE ────────────────────────────────────────────────────────────
    final_df.to_csv(OUTPUT_FILE, index=False)

    # ── 8. SUMMARY ─────────────────────────────────────────────────────────
    print(f"{'='*62}")
    print(f"  DONE — {OUTPUT_FILE}")
    print(f"{'='*62}")
    print(f"\n  Shape   : {final_df.shape}")
    print(f"  Columns : {list(final_df.columns)}")
    print(f"\n  Split distribution:")
    print(f"  {final_df['split'].value_counts().to_string()}")
    print(f"\n  Null counts:")
    nulls = final_df.isnull().sum()
    if nulls.sum() == 0:
        print("    None — fully clean!")
    else:
        for col, n in nulls[nulls > 0].items():
            print(f"    {col:<35}: {n:,} ({n / len(final_df) * 100:.1f}%)")
    print(f"\n  Target (leftover_energy_wh) stats:")
    print(f"    mean   : {final_df['leftover_energy_wh'].mean():>12,.0f} Wh")
    print(f"    median : {final_df['leftover_energy_wh'].median():>12,.0f} Wh")
    print(f"    min    : {final_df['leftover_energy_wh'].min():>12,.0f} Wh")
    print(f"    max    : {final_df['leftover_energy_wh'].max():>12,.0f} Wh")


if __name__ == "__main__":
    main()
