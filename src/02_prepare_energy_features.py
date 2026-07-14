"""
prepare_energy_model.py
========================
Fixes all critical data issues in features_charger.csv and produces a
clean, model-ready file: features_energy_model.csv

Issues fixed:
  [1] DATA LEAKAGE — 3 columns derived from the target are removed:
        • estimated_capacity_wh  (= charger_max_energy * 1.1)
        • leftover_energy_wh     (= estimated_capacity_wh - max_energy_wh)
        • power_to_energy_ratio  (= avg_pwr_kw / (max_energy_wh / 1000))

  [2] IDLE PHASE BIAS — 86.8% of rows have avg_pwr_kw == 0 (idle chargers).
        These rows are VALID (max_energy_wh = cumulative session energy stored).
        Fix: add explicit binary flag  is_idle_charger  so the model can learn
        the idle vs active distinction cleanly.

  [3] AGGREGATE LEAKAGE — station_mean_energy_wh / station_mean_power_kw were
        computed on the full dataset (including future val/test rows).
        Fix: compute all aggregates on the TRAINING split only, then join
        to val and test rows (they will see only past/train-derived means).

Source  : features_charger.csv  (88,808 rows — NOT modified)
Output  : features_energy_model.csv  (new file — clean and leak-free)
"""

import pandas as pd
import numpy as np
import os

# ─── CONFIG ───────────────────────────────────────────────────────────────────
INPUT_FILE  = "data/features_charger.csv"
OUTPUT_FILE = "data/features_energy_model.csv"
RANDOM_SEED = 42
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 62)
print("  prepare_energy_model.py — Critical Issue Fixer")
print("=" * 62)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE)
df["timestamp"] = pd.to_datetime(df["timestamp"])
print(f"\n[LOAD]  {INPUT_FILE} → {len(df):,} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────────────────────────────────────
# FIX 1: REMOVE LEAKED COLUMNS
# These 3 columns are algebraically derived from max_energy_wh (the target).
# Including them would let the model see the answer — making metrics fake.
# ─────────────────────────────────────────────────────────────────────────────
leaked_cols = ["estimated_capacity_wh", "leftover_energy_wh", "power_to_energy_ratio"]
df = df.drop(columns=leaked_cols)
print(f"\n[FIX 1] Leaked columns removed: {leaked_cols}")
print(f"        Remaining columns: {df.shape[1]}")

# Also drop soc_pct — it's a future model's target, not a feature here,
# and is 96.5% null anyway.
df = df.drop(columns=["soc_pct"])
print(f"        soc_pct dropped (96.5% null — future Model 4 target)")
print(f"        Columns now: {list(df.columns)}")

# ─────────────────────────────────────────────────────────────────────────────
# FIX 2: IDLE PHASE — add explicit binary flag
# 86.8% of rows have avg_pwr_kw == 0 (charger idle within the interval).
# max_energy_wh is cumulative session energy, so idle rows still carry signal.
# We add is_idle_charger so the model can split cleanly on this dominant state.
# ─────────────────────────────────────────────────────────────────────────────
df["is_idle_charger"] = (df["avg_pwr_kw"] == 0).astype(int)
idle_count = df["is_idle_charger"].sum()
print(f"\n[FIX 2] is_idle_charger flag added")
print(f"        Idle rows (avg_pwr_kw = 0): {idle_count:,} ({idle_count/len(df)*100:.1f}%)")
print(f"        Active rows               : {len(df)-idle_count:,} ({(len(df)-idle_count)/len(df)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# FIX 3: PERIOD-AWARE SPLIT FIRST, then compute aggregates on TRAIN only
#
# Data windows (NON-CONTINUOUS — do NOT random split):
#   Period 1 → Nov 2025  (2 days)
#   Period 2 → Dec 2025  (4 days)
#   Period 3 → Apr 2026  (5 days)
#
# Split:
#   Train : Period 1 + Period 2 + first 80% of Period 3
#   Val   : middle 10% of Period 3 (used for early stopping)
#   Test  : last 10% of Period 3 (held-out final evaluation)
# ─────────────────────────────────────────────────────────────────────────────

# Drop old aggregate columns (computed on full dataset — leaky)
df = df.drop(columns=["station_mean_energy_wh", "station_mean_power_kw"])
print(f"\n[FIX 3] Dropped full-dataset aggregates (station_mean_energy_wh, station_mean_power_kw)")

# --- Split ---
p1p2_df = df[df["collection_period"].isin([1, 2])].copy()
p3_df   = df[df["collection_period"] == 3].sort_values("timestamp").copy()

n3      = len(p3_df)
cut80   = int(n3 * 0.80)
cut90   = int(n3 * 0.90)

train_df = pd.concat([p1p2_df, p3_df.iloc[:cut80]], axis=0).copy()
val_df   = p3_df.iloc[cut80:cut90].copy()
test_df  = p3_df.iloc[cut90:].copy()

print(f"\n        Period-aware split:")
print(f"          Train : {len(train_df):>8,} rows  (Period 1+2 + 80% of Period 3)")
print(f"          Val   : {len(val_df):>8,} rows  (10% of Period 3)")
print(f"          Test  : {len(test_df):>8,} rows  (last 10% of Period 3)")

# --- Compute aggregates on TRAIN only ---
print(f"\n        Computing aggregates on TRAIN set only ...")

# Station-level: mean energy and mean power per station (from train rows)
station_agg = (train_df.groupby("charging_station_id")
               .agg(
                   station_mean_energy_wh = ("max_energy_wh", "mean"),
                   station_median_energy_wh = ("max_energy_wh", "median"),
                   station_mean_power_kw  = ("avg_pwr_kw", "mean"),
                   station_max_power_kw   = ("avg_pwr_kw", "max"),
                   station_charger_count  = ("charger_id", "nunique"),
               )
               .reset_index())

# Charger-level: mean energy per charger (from train rows)
charger_agg = (train_df.groupby("charger_id")
               .agg(
                   charger_mean_energy_wh = ("max_energy_wh", "mean"),
                   charger_median_energy_wh = ("max_energy_wh", "median"),
               )
               .reset_index())

print(f"          Station aggregates built from {len(train_df):,} train rows")
print(f"          Charger aggregates built from {len(train_df):,} train rows")

# --- Join aggregates to all splits ---
# Val/test stations/chargers not in train get NaN → CatBoost handles gracefully
for split_df in [train_df, val_df, test_df]:
    split_df.drop(columns=[c for c in split_df.columns
                            if c in station_agg.columns and c != "charging_station_id"],
                  inplace=True, errors="ignore")
    split_df.drop(columns=[c for c in split_df.columns
                            if c in charger_agg.columns and c != "charger_id"],
                  inplace=True, errors="ignore")

train_df = train_df.merge(station_agg, on="charging_station_id", how="left")
train_df = train_df.merge(charger_agg, on="charger_id", how="left")
val_df   = val_df.merge(station_agg,   on="charging_station_id", how="left")
val_df   = val_df.merge(charger_agg,   on="charger_id", how="left")
test_df  = test_df.merge(station_agg,  on="charging_station_id", how="left")
test_df  = test_df.merge(charger_agg,  on="charger_id", how="left")

# Check new-station coverage in val/test (stations not seen in train)
val_new_stations  = val_df["station_mean_energy_wh"].isna().sum()
test_new_stations = test_df["station_mean_energy_wh"].isna().sum()
print(f"\n          Val  rows with unseen station (NaN agg): {val_new_stations}")
print(f"          Test rows with unseen station (NaN agg): {test_new_stations}")
print(f"          (CatBoost handles NaN natively — no imputation needed)")

# ─────────────────────────────────────────────────────────────────────────────
# ADD split LABEL and recombine into one file
# The split column lets the modelling script separate train/val/test
# without re-computing the split (ensures perfect reproducibility)
# ─────────────────────────────────────────────────────────────────────────────
train_df["split"] = "train"
val_df["split"]   = "val"
test_df["split"]  = "test"

final_df = pd.concat([train_df, val_df, test_df], axis=0).reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# FINAL COLUMN ORDER
# ─────────────────────────────────────────────────────────────────────────────
feature_cols = [
    # Identifiers (CatBoost categorical)
    "charging_station_id", "charger_id",
    # Temporal
    "timestamp", "split", "collection_period",
    "hour", "day_of_week", "month", "is_weekend", "is_late_night",
    "hour_sin", "hour_cos",
    # Raw metrics (no leakage)
    "avg_pwr_kw", "active_session_count", "is_idle_charger",
    # Train-only aggregates (no leakage)
    "station_mean_energy_wh", "station_median_energy_wh",
    "station_mean_power_kw",  "station_max_power_kw",
    "station_charger_count",
    "charger_mean_energy_wh", "charger_median_energy_wh",
    # TARGET
    "max_energy_wh",
]
final_df = final_df[feature_cols]

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
final_df.to_csv(OUTPUT_FILE, index=False)

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
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
    for col, n in nulls[nulls>0].items():
        print(f"    {col:<35}: {n:,} ({n/len(final_df)*100:.1f}%)")
print(f"\n  Target (max_energy_wh) stats:")
print(f"    mean   : {final_df['max_energy_wh'].mean():>12,.0f} Wh")
print(f"    median : {final_df['max_energy_wh'].median():>12,.0f} Wh")
print(f"    min    : {final_df['max_energy_wh'].min():>12,.0f} Wh")
print(f"    max    : {final_df['max_energy_wh'].max():>12,.0f} Wh")
print(f"\n  Fixes applied:")
print(f"    [1] ✅ Leaked columns removed : {leaked_cols}")
print(f"    [2] ✅ is_idle_charger flag added ({idle_count:,} idle rows flagged)")
print(f"    [3] ✅ Aggregates recomputed on train split only (no future leakage)")
print(f"\n  Source file unchanged: {INPUT_FILE}")
print(f"  New file created     : {OUTPUT_FILE}")
