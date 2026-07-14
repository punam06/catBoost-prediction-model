"""
04_predict.py
=============
Interactive terminal script to predict energy consumption for a specific charger.
It automatically looks up the required historical aggregates for the requested station/charger.

Usage:
  source venv/bin/activate
  python3 src/04_predict.py
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings("ignore")

MODEL_PATH = "models/catboost_energy_model.cbm"
DATA_PATH  = "data/features_energy_model.csv"

def load_resources():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Data not found at {DATA_PATH} (Needed to look up historical aggregates)")
        sys.exit(1)
        
    print("Loading model and historical data database...")
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)
    
    # Load dataset just to extract unique station and charger historical aggregates
    df = pd.read_csv(DATA_PATH)
    return model, df

def get_aggregates(df, station_id, charger_id):
    """Fetch the historical averages for the given station and charger."""
    # Find the first row that matches the station to get its static aggregates
    station_data = df[df["charging_station_id"] == station_id]
    if station_data.empty:
        print(f"\n[WARNING] Station '{station_id}' is unknown. Using global averages.")
        st_mean_e = df["station_mean_energy_wh"].mean()
        st_med_e  = df["station_median_energy_wh"].mean()
        st_mean_p = df["station_mean_power_kw"].mean()
        st_max_p  = df["station_max_power_kw"].mean()
        st_count  = 1
    else:
        row = station_data.iloc[0]
        st_mean_e = row["station_mean_energy_wh"]
        st_med_e  = row["station_median_energy_wh"]
        st_mean_p = row["station_mean_power_kw"]
        st_max_p  = row["station_max_power_kw"]
        st_count  = row["station_charger_count"]

    # Find charger
    charger_data = df[df["charger_id"] == charger_id]
    if charger_data.empty:
        print(f"[WARNING] Charger '{charger_id}' is unknown. Using station averages.")
        ch_mean_e = st_mean_e
        ch_med_e  = st_med_e
    else:
        row = charger_data.iloc[0]
        ch_mean_e = row["charger_mean_energy_wh"]
        ch_med_e  = row["charger_median_energy_wh"]
        
    # Replace NaN with -999 for CatBoost consistency
    aggs = {
        "station_mean_energy_wh": -999 if pd.isna(st_mean_e) else st_mean_e,
        "station_median_energy_wh": -999 if pd.isna(st_med_e) else st_med_e,
        "station_mean_power_kw": -999 if pd.isna(st_mean_p) else st_mean_p,
        "station_max_power_kw": -999 if pd.isna(st_max_p) else st_max_p,
        "station_charger_count": -999 if pd.isna(st_count) else st_count,
        "charger_mean_energy_wh": -999 if pd.isna(ch_mean_e) else ch_mean_e,
        "charger_median_energy_wh": -999 if pd.isna(ch_med_e) else ch_med_e,
    }
    return aggs

def main():
    print("\n" + "="*50)
    print(" ⚡ ENERGY CONSUMPTION PREDICTOR (Terminal API) ⚡")
    print("="*50 + "\n")
    
    model, df = load_resources()
    
    # Provide a sample station/charger to make it easy for the user
    sample_station = df["charging_station_id"].dropna().iloc[0]
    sample_charger = df["charger_id"].dropna().iloc[0]
    
    print(f"\n[Tip] You can just press ENTER to use defaults for testing.")
    
    try:
        # 1. Inputs
        st_id = input(f"Enter Station ID (default: {sample_station}): ").strip() or sample_station
        ch_id = input(f"Enter Charger ID (default: {sample_charger}): ").strip() or sample_charger
        
        target_date_str = input("Enter Date (YYYY-MM-DD) [default: tomorrow]: ").strip()
        if not target_date_str:
            target_date = datetime.now() + timedelta(days=1)
        else:
            target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
            
        hour_str = input("Enter Hour (0-23) [default: 18]: ").strip()
        hour = int(float(hour_str)) if hour_str else 18
        
        pwr_str = input("Enter expected Avg Power (kW) [default: 25.0]: ").strip()
        avg_pwr_kw = float(pwr_str) if pwr_str else 25.0
        
        # 2. Compute Features
        day_of_week = target_date.weekday()
        month = target_date.month
        is_weekend = 1 if day_of_week >= 5 else 0
        is_late_night = 1 if hour >= 2 else 0
        
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        
        is_idle_charger = 1 if avg_pwr_kw == 0 else 0
        active_session_count = 1
        
        # We assign period=3 (April 2026 patterns) to all future dates as it's the most recent
        collection_period = 3 
        
        # 3. Get Historical Aggregates
        aggs = get_aggregates(df, st_id, ch_id)
        
        # 4. Construct DataFrame matching the model's expected columns
        feature_cols = [
            "charging_station_id", "charger_id", "collection_period",
            "hour", "day_of_week", "month", "is_weekend", "is_late_night",
            "hour_sin", "hour_cos", "avg_pwr_kw", "active_session_count", "is_idle_charger",
            "station_mean_energy_wh", "station_median_energy_wh",
            "station_mean_power_kw",  "station_max_power_kw",
            "station_charger_count",
            "charger_mean_energy_wh", "charger_median_energy_wh"
        ]
        
        input_data = {
            "charging_station_id": [st_id],
            "charger_id": [ch_id],
            "collection_period": [collection_period],
            "hour": [hour],
            "day_of_week": [day_of_week],
            "month": [month],
            "is_weekend": [is_weekend],
            "is_late_night": [is_late_night],
            "hour_sin": [hour_sin],
            "hour_cos": [hour_cos],
            "avg_pwr_kw": [avg_pwr_kw],
            "active_session_count": [active_session_count],
            "is_idle_charger": [is_idle_charger]
        }
        # Merge with aggregates
        for k, v in aggs.items():
            input_data[k] = [v]
            
        X_predict = pd.DataFrame(input_data)[feature_cols]
        
        # 5. Predict
        predicted_energy = model.predict(X_predict)[0]
        predicted_energy = max(0, predicted_energy) # No negative energy
        
        print("\n" + "="*50)
        print(" 🔮 PREDICTION RESULT 🔮")
        print("="*50)
        print(f"  Target Date  : {target_date.strftime('%Y-%m-%d')} at {hour:02d}:00")
        print(f"  Station      : {st_id}")
        print(f"  Charger      : {ch_id}")
        print(f"  Assumed Pwr  : {avg_pwr_kw} kW")
        print(f"  -------------------------------------")
        print(f"  Estimated Energy Consumption : {predicted_energy:,.2f} Wh")
        print(f"                               : {predicted_energy/1000:,.2f} kWh")
        print("="*50 + "\n")

    except ValueError as e:
        print(f"\n[ERROR] Invalid input: {e}")
    except KeyboardInterrupt:
        print("\nPrediction cancelled.")

if __name__ == "__main__":
    main()
