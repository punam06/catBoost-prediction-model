"""
05_prepare_soc_features.py
==========================
Prepares the dataset specifically for predicting the State of Charge (soc_pct).
Filters out null values, engineers time features, and splits the data.
"""

import os
import pandas as pd
import numpy as np

# ─── CONFIG ───────────────────────────────────────────────────────────────────
INPUT_FILE  = "data/cleaned_data.csv"
OUTPUT_FILE = "data/features_soc_model.csv"
RANDOM_SEED = 42
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("==============================================================")
    print("  prepare_soc_features.py — SoC Prediction Prep")
    print("==============================================================\n")
    
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Missing {INPUT_FILE}. Run 01_cleaner.py first.")
        
    df = pd.read_csv(INPUT_FILE)
    initial_rows = len(df)
    
    # 1. Filter for non-null soc_pct
    df = df.dropna(subset=['soc_pct']).copy()
    print(f"[FILTER]  Dropped rows missing 'soc_pct'")
    print(f"          Remaining rows: {len(df):,} (out of {initial_rows:,})\n")
    
    if len(df) == 0:
        print("[ERROR] No rows with soc_pct found! Cannot proceed.")
        return
        
    # 2. Basic temporal sorting
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    # 3. Create active session flag
    df['is_idle_charger'] = (df['avg_pwr_kw'] == 0).astype(int)
    
    # 4. Train / Val / Test Split (Time-based since it's a timeseries)
    # Since we only have ~3,145 rows, we will do an 80/10/10 time-based split
    total_rows = len(df)
    train_end = int(total_rows * 0.8)
    val_end   = int(total_rows * 0.9)
    
    df['split'] = 'train'
    df.iloc[train_end:val_end, df.columns.get_loc('split')] = 'val'
    df.iloc[val_end:, df.columns.get_loc('split')] = 'test'
    
    print(f"[SPLIT]   Time-based split applied:")
    print(f"          Train: {len(df[df['split']=='train']):>6,} rows")
    print(f"          Val  : {len(df[df['split']=='val']):>6,} rows")
    print(f"          Test : {len(df[df['split']=='test']):>6,} rows\n")
    
    # 5. Save features
    columns_to_keep = [
        "charging_station_id", "charger_id", "timestamp", "split", "collection_period",
        "hour", "day_of_week", "month", "is_weekend", "hour_sin", "hour_cos", 
        "avg_pwr_kw", "is_idle_charger", "max_energy_wh", "soc_pct"
    ]
    df = df[columns_to_keep]
    
    df.to_csv(OUTPUT_FILE, index=False)
    print("==============================================================")
    print(f"  DONE — Saved to {OUTPUT_FILE}")
    print("==============================================================")

if __name__ == "__main__":
    main()
