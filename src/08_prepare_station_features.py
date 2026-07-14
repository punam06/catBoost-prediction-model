"""
08_prepare_station_features.py
==============================
Aggregates the charger-level dataset up to the station-level to predict Congestion.
Target: active_chargers (number of chargers currently in use at the station).
"""

import os
import pandas as pd
import numpy as np

# ─── CONFIG ───────────────────────────────────────────────────────────────────
INPUT_FILE  = "data/cleaned_data.csv"
OUTPUT_FILE = "data/features_station_model.csv"
RANDOM_SEED = 42
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("==============================================================")
    print("  prepare_station_features.py — Congestion Prediction Prep")
    print("==============================================================\n")
    
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Missing {INPUT_FILE}. Run 01_cleaner.py first.")
        
    df = pd.read_csv(INPUT_FILE)
    initial_rows = len(df)
    print(f"[LOAD]  Loaded {initial_rows:,} charger-level rows.")
    
    # 1. Group by Station and Timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Helper to count active chargers (where power > 0)
    def count_active(series):
        return (series > 0).sum()
        
    print("[AGG]   Aggregating data to the station level...")
    station_df = df.groupby(["charging_station_id", "timestamp"]).agg(
        active_chargers=("avg_pwr_kw", count_active),
        total_chargers=("charger_id", "count"),
        total_power_kw=("avg_pwr_kw", "sum"),
        # Extract time features directly from the first row in the group 
        # (they are the same for all chargers at the same timestamp)
        hour=("hour", "first"),
        day_of_week=("day_of_week", "first"),
        month=("month", "first"),
        is_weekend=("is_weekend", "first"),
        hour_sin=("hour_sin", "first"),
        hour_cos=("hour_cos", "first"),
        collection_period=("collection_period", "first")
    ).reset_index()
    
    print(f"        Resulted in {len(station_df):,} station-level snapshots.\n")
    
    # 2. Time-based Split (Train: P1, P2 + 80% P3 | Val: 10% P3 | Test: 10% P3)
    station_df = station_df.sort_values(by='timestamp').reset_index(drop=True)
    
    period_1_2 = station_df[station_df['collection_period'].isin([1, 2])]
    period_3   = station_df[station_df['collection_period'] == 3]
    
    p3_total = len(period_3)
    p3_train_end = int(p3_total * 0.8)
    p3_val_end   = int(p3_total * 0.9)
    
    p3_train = period_3.iloc[:p3_train_end]
    p3_val   = period_3.iloc[p3_train_end:p3_val_end]
    p3_test  = period_3.iloc[p3_val_end:]
    
    train_df = pd.concat([period_1_2, p3_train])
    val_df   = p3_val
    test_df  = p3_test
    
    # Add split column
    station_df['split'] = 'train'
    station_df.loc[val_df.index, 'split'] = 'val'
    station_df.loc[test_df.index, 'split'] = 'test'
    
    print(f"[SPLIT] Period-aware temporal split applied:")
    print(f"        Train: {len(train_df):>8,} rows")
    print(f"        Val  : {len(val_df):>8,} rows")
    print(f"        Test : {len(test_df):>8,} rows\n")
    
    # 3. Save
    station_df.to_csv(OUTPUT_FILE, index=False)
    print("==============================================================")
    print(f"  DONE — Saved to {OUTPUT_FILE}")
    print("==============================================================")

if __name__ == "__main__":
    main()
