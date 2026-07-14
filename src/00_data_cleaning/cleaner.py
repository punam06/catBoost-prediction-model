"""
cleaner.py
==========
Data cleaning pipeline for All_Merger.csv (EV Charging Station Dataset).

Steps:
  1. Remove exact full-row duplicates
  2. Resolve near-duplicates (same station+charger+timestamp -> keep highest max_energy_wh)
  3. Remove physically invalid negative values
  4. Cap extreme outliers at the 99th percentile (value capping)
  5. Remove dead rows (avg_pwr_kw == 0 AND max_energy_wh == 0)
  6. Remove no-session rows (active_session_count == 0)
  7. Fix data types + add collection_period label
  8. Cap station representation at 300 rows per station

Output: cleaned_data.csv (~88,808 rows)
"""

import pandas as pd
import numpy as np
import os

# ─── CONFIG ───────────────────────────────────────────────────────────────────
INPUT_FILE         = "data/raw/All_Merger.csv"
OUTPUT_FILE        = "data/processed/cleaned_data.csv"
STATION_ROW_CAP    = 300       # Max rows per charging station
OUTLIER_PERCENTILE = 0.99      # Winsorize cap threshold
RANDOM_SEED        = 42
# ──────────────────────────────────────────────────────────────────────────────


def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw CSV file."""
    print(f"\n{'='*60}")
    print("  EV Charging Station — Data Cleaning Pipeline")
    print(f"{'='*60}\n")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")

    df = pd.read_csv(filepath)
    print(f"[LOADED]  {filepath}")
    print(f"          {len(df):>10,} rows  x  {df.shape[1]} columns")
    print(f"\n  Columns: {list(df.columns)}\n")
    return df


def step1_remove_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Step 1: Drop rows that are 100% identical across all columns."""
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    print(f"[Step 1]  Remove exact duplicates        | -{removed:>6,} rows  ->  {len(df):,} remaining")
    return df


def step2_resolve_near_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2: Resolve near-duplicates.
    Same (charging_station_id, charger_id, timestamp) but different metric values.
    Keep the row with the highest max_energy_wh (most informative reading).
    """
    before = len(df)
    key_cols = ["charging_station_id", "charger_id", "timestamp"]
    df = (df.sort_values("max_energy_wh", ascending=False)
            .drop_duplicates(subset=key_cols, keep="first")
            .reset_index(drop=True))
    removed = before - len(df)
    print(f"[Step 2]  Resolve near-duplicates        | -{removed:>6,} rows  ->  {len(df):,} remaining")
    return df


def step3_remove_negatives(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3: Drop rows with physically impossible negative values."""
    before = len(df)
    mask = (df["avg_pwr_kw"] >= 0) & (df["max_energy_wh"] >= 0)
    df = df[mask].reset_index(drop=True)
    removed = before - len(df)
    print(f"[Step 3]  Remove negative values         | -{removed:>6,} rows  ->  {len(df):,} remaining")
    return df


def step4_cap_outliers(df: pd.DataFrame, percentile: float = OUTLIER_PERCENTILE) -> pd.DataFrame:
    """
    Step 4: Value capping (Winsorization) at the given percentile.
    Clips extreme values to the cap threshold — no rows are removed.
    """
    cap_cols = ["max_energy_wh", "avg_pwr_kw"]
    caps = {}
    for col in cap_cols:
        cap_val = df[col].quantile(percentile)
        df[col] = df[col].clip(upper=cap_val)
        caps[col] = cap_val

    print(f"[Step 4]  Cap outliers at P{int(percentile*100)}            |   (clip)        ->  {len(df):,} remaining")
    print(f"            max_energy_wh capped at : {caps['max_energy_wh']:>15,.0f} Wh")
    print(f"            avg_pwr_kw    capped at : {caps['avg_pwr_kw']:>15,.2f} kW")
    return df


def step5_remove_dead_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Step 5: Remove rows where avg_pwr_kw == 0 AND max_energy_wh == 0 (no useful signal)."""
    before = len(df)
    mask = ~((df["avg_pwr_kw"] == 0) & (df["max_energy_wh"] == 0))
    df = df[mask].reset_index(drop=True)
    removed = before - len(df)
    print(f"[Step 5]  Remove dead rows (0kW & 0Wh)  | -{removed:>6,} rows  ->  {len(df):,} remaining")
    return df


def step6_remove_no_session_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Step 6: Remove rows where active_session_count == 0 (no active charging session)."""
    before = len(df)
    df = df[df["active_session_count"] != 0].reset_index(drop=True)
    removed = before - len(df)
    print(f"[Step 6]  Remove no-session rows         | -{removed:>6,} rows  ->  {len(df):,} remaining")
    return df


def step7_fix_types_and_add_period(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 7: Fix data types and add collection_period label.
    - Parse timestamp as UTC datetime, strip timezone for model compatibility
    - Cast integer columns from float64 to int64
    - Add collection_period: 1=Nov2025, 2=Dec2025, 3=Apr2026
    """
    # Parse timestamp (strip timezone for compatibility)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)

    # Fix integer columns (were stored as float64 due to mixed-type merging)
    int_cols = ["hour", "day_of_week", "month", "is_weekend", "active_session_count"]
    for col in int_cols:
        df[col] = df[col].astype(int)

    # Add collection_period — distinguishes the 3 non-continuous data windows
    # This is critical: data is NOT continuous (gaps of 4 days and 113 days between periods)
    period_map = {11: 1, 12: 2, 4: 3}
    df["collection_period"] = df["month"].map(period_map).fillna(3).astype(int)

    print(f"[Step 7]  Fix dtypes + add period label  |   (no rows)    ->  {len(df):,} remaining")
    print(f"            collection_period distribution:")
    dist = df["collection_period"].value_counts().sort_index()
    period_names = {1: "Nov 2025", 2: "Dec 2025", 3: "Apr 2026"}
    for period, count in dist.items():
        print(f"              Period {period} ({period_names[period]}): {count:,} rows")

    return df


def step8_cap_station_rows(df: pd.DataFrame, cap: int = STATION_ROW_CAP,
                            seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Step 8: Station representation capping.
    Any station with more than `cap` rows is randomly downsampled to exactly `cap` rows.
    Stations below the cap are kept in full to preserve diversity.
    """
    before = len(df)
    rng = np.random.default_rng(seed)

    idx_keep = []
    over_cap_stations = 0

    for sid, grp in df.groupby("charging_station_id"):
        if len(grp) > cap:
            chosen = rng.choice(grp.index, size=cap, replace=False)
            idx_keep.extend(chosen)
            over_cap_stations += 1
        else:
            idx_keep.extend(grp.index)

    df = df.loc[idx_keep].reset_index(drop=True)
    removed = before - len(df)
    print(f"[Step 8]  Station cap ({cap} rows/station) | -{removed:>6,} rows  ->  {len(df):,} remaining")
    print(f"            Stations reduced to {cap} rows: {over_cap_stations}")
    return df


def print_summary(raw_count: int, df: pd.DataFrame) -> None:
    """Print a final cleaning summary report."""
    print(f"\n{'='*60}")
    print("  CLEANING COMPLETE — Final Report")
    print(f"{'='*60}")
    print(f"  Raw rows        : {raw_count:>10,}")
    print(f"  Cleaned rows    : {len(df):>10,}")
    print(f"  Rows removed    : {raw_count - len(df):>10,}  ({(raw_count - len(df)) / raw_count * 100:.1f}% reduction)")
    print(f"  Columns         : {df.shape[1]}")

    print(f"\n  Null counts (non-zero only):")
    null_counts = df.isnull().sum()
    has_nulls = False
    for col, count in null_counts.items():
        if count > 0:
            print(f"    {col:<30} {count:>8,}  ({count/len(df)*100:.1f}%)")
            has_nulls = True
    if not has_nulls:
        print("    None")

    print(f"\n  Key column statistics:")
    for col in ["avg_pwr_kw", "max_energy_wh"]:
        s = df[col].describe()
        print(f"\n    {col}:")
        print(f"      count={s['count']:,.0f}  mean={s['mean']:,.2f}  "
              f"min={s['min']:,.2f}  max={s['max']:,.2f}")

    print(f"\n  Dataset diversity:")
    print(f"    Unique stations : {df['charging_station_id'].nunique():,}")
    print(f"    Unique chargers : {df['charger_id'].nunique():,}")
    print(f"\n  Collection periods:")
    period_names = {1: "Nov 2025", 2: "Dec 2025", 3: "Apr 2026"}
    for p in sorted(df["collection_period"].unique()):
        count = (df["collection_period"] == p).sum()
        print(f"    Period {p} ({period_names[p]}): {count:,} rows")


def run_pipeline() -> pd.DataFrame:
    """Execute the full 8-step cleaning pipeline and save the output."""

    df = load_data(INPUT_FILE)
    raw_count = len(df)

    print(f"{'─'*60}")

    df = step1_remove_exact_duplicates(df)
    df = step2_resolve_near_duplicates(df)
    df = step3_remove_negatives(df)
    df = step4_cap_outliers(df)
    df = step5_remove_dead_rows(df)
    df = step6_remove_no_session_rows(df)
    df = step7_fix_types_and_add_period(df)
    df = step8_cap_station_rows(df)

    print(f"{'─'*60}")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[SAVED]   {OUTPUT_FILE}  ({len(df):,} rows x {df.shape[1]} columns)\n")

    print_summary(raw_count, df)

    return df


if __name__ == "__main__":
    cleaned_df = run_pipeline()
