"""
prepare_demand_features.py
===========================
Aggregates charger-level data to station-level for Demand Forecasting.
Target: total_energy_wh (sum of energy consumed across all chargers at a station per hour).

Data leakage prevention:
  - total_power_kw excluded (directly correlates with target)
  - station_avg_hourly_demand and station_peak_hourly_demand computed on TRAIN only

Source  : data/processed/cleaned_data.csv
Output  : data/processed/features_demand_model.csv
"""

import os
import pandas as pd
import numpy as np

# ─── CONFIG ───────────────────────────────────────────────────────────────────
INPUT_FILE  = "data/processed/cleaned_data.csv"
OUTPUT_FILE = "data/processed/features_demand_model.csv"
RANDOM_SEED = 42
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  prepare_demand_features.py — Demand Forecasting Prep")
    print("=" * 62 + "\n")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Missing {INPUT_FILE}. Run 00_data_cleaning/cleaner.py first.")

    df = pd.read_csv(INPUT_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"[LOAD]  Loaded {len(df):,} charger-level rows.")

    # ── 1. AGGREGATE TO STATION LEVEL ──────────────────────────────────────
    def count_active(series):
        return (series > 0).sum()

    print("[AGG]   Aggregating to station level per timestamp...")
    station_df = df.groupby(["charging_station_id", "timestamp"]).agg(
        # Target: total energy consumed at station
        total_energy_wh=("max_energy_wh", "sum"),
        # Station metrics
        total_chargers=("charger_id", "count"),
        active_chargers=("avg_pwr_kw", count_active),
        avg_energy_per_charger=("max_energy_wh", "mean"),
        peak_power_kw=("avg_pwr_kw", "max"),
        # Time features (same across all chargers in a group)
        hour=("hour", "first"),
        day_of_week=("day_of_week", "first"),
        month=("month", "first"),
        is_weekend=("is_weekend", "first"),
        hour_sin=("hour_sin", "first"),
        hour_cos=("hour_cos", "first"),
        collection_period=("collection_period", "first"),
    ).reset_index()

    # Derived feature
    station_df["utilization_rate"] = station_df["active_chargers"] / station_df["total_chargers"]

    print(f"        Resulted in {len(station_df):,} station-level snapshots.\n")

    # ── 2. TIME-BASED SPLIT (Train: P1, P2 + 80% P3 | Val: 10% P3 | Test: 10% P3) ──
    station_df = station_df.sort_values(by="timestamp").reset_index(drop=True)

    period_1_2 = station_df[station_df["collection_period"].isin([1, 2])]
    period_3   = station_df[station_df["collection_period"] == 3]

    p3_total      = len(period_3)
    p3_train_end  = int(p3_total * 0.8)
    p3_val_end    = int(p3_total * 0.9)

    p3_train = period_3.iloc[:p3_train_end]
    p3_val   = period_3.iloc[p3_train_end:p3_val_end]
    p3_test  = period_3.iloc[p3_val_end:]

    train_df = pd.concat([period_1_2, p3_train])
    val_df   = p3_val
    test_df  = p3_test

    print(f"[SPLIT] Period-aware temporal split:")
    print(f"        Train: {len(train_df):>8,} rows  (P1+P2 + 80% of P3)")
    print(f"        Val  : {len(val_df):>8,} rows  (10% of P3)")
    print(f"        Test : {len(test_df):>8,} rows  (10% of P3)\n")

    # ── 3. TRAIN-ONLY AGGREGATES (prevent future data leakage) ─────────────
    print("[FEAT]  Computing station aggregates on TRAIN split only...")

    station_demand_agg = (
        train_df.groupby("charging_station_id")
        .agg(
            station_avg_hourly_demand=("total_energy_wh", "mean"),
            station_peak_hourly_demand=("total_energy_wh", "max"),
        )
        .reset_index()
    )

    print(f"        Aggregates built from {len(train_df):,} train rows.")

    # Join aggregates to all splits (unseen stations get NaN — CatBoost handles it)
    for split_df in [train_df, val_df, test_df]:
        split_df.drop(
            columns=[c for c in split_df.columns if c in station_demand_agg.columns and c != "charging_station_id"],
            inplace=True,
            errors="ignore",
        )

    train_df = train_df.merge(station_demand_agg, on="charging_station_id", how="left")
    val_df   = val_df.merge(station_demand_agg, on="charging_station_id", how="left")
    test_df  = test_df.merge(station_demand_agg, on="charging_station_id", how="left")

    val_new  = val_df["station_avg_hourly_demand"].isna().sum()
    test_new = test_df["station_avg_hourly_demand"].isna().sum()
    print(f"        Val  rows with unseen station (NaN): {val_new}")
    print(f"        Test rows with unseen station (NaN): {test_new}\n")

    # ── 4. LABEL SPLITS AND COMBINE ────────────────────────────────────────
    train_df["split"] = "train"
    val_df["split"]   = "val"
    test_df["split"]  = "test"

    final_df = pd.concat([train_df, val_df, test_df], axis=0).reset_index(drop=True)

    # ── 5. COLUMN ORDER ────────────────────────────────────────────────────
    feature_cols = [
        # Identifier
        "charging_station_id",
        # Station metrics
        "total_chargers", "active_chargers", "utilization_rate",
        "avg_energy_per_charger", "peak_power_kw",
        # Train-only aggregates
        "station_avg_hourly_demand", "station_peak_hourly_demand",
        # Temporal
        "hour", "day_of_week", "month", "is_weekend",
        "hour_sin", "hour_cos", "collection_period",
        # Metadata
        "split", "timestamp",
        # TARGET
        "total_energy_wh",
    ]
    final_df = final_df[feature_cols]

    # ── 6. SAVE ────────────────────────────────────────────────────────────
    final_df.to_csv(OUTPUT_FILE, index=False)

    # ── 7. SUMMARY ─────────────────────────────────────────────────────────
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
    print(f"\n  Target (total_energy_wh) stats:")
    print(f"    mean   : {final_df['total_energy_wh'].mean():>12,.0f} Wh")
    print(f"    median : {final_df['total_energy_wh'].median():>12,.0f} Wh")
    print(f"    min    : {final_df['total_energy_wh'].min():>12,.0f} Wh")
    print(f"    max    : {final_df['total_energy_wh'].max():>12,.0f} Wh")


if __name__ == "__main__":
    main()
