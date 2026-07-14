"""
07_predict_soc.py
=================
Interactive terminal script to predict the State of Charge (SoC %) of an EV.

Usage:
  source venv/bin/activate
  python3 src/07_predict_soc.py
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings("ignore")

MODEL_PATH = "models/catboost_soc_model.cbm"
DATA_PATH  = "data/features_soc_model.csv"

def load_resources():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}")
        sys.exit(1)
        
    print("Loading SoC prediction model...")
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)
    
    # Load just for getting a sample row to offer defaults
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        df = None
        
    return model, df

def main():
    print("\n" + "="*50)
    print(" 🔋 STATE OF CHARGE (SoC %) PREDICTOR 🔋")
    print("="*50 + "\n")
    
    model, df = load_resources()
    
    # Defaults
    sample_station = "008a2054178b149a52d9893112adf164"
    sample_charger = "T124-IT1-1423-083"
    
    if df is not None and not df.empty:
        sample_station = df["charging_station_id"].iloc[0]
        sample_charger = df["charger_id"].iloc[0]
    
    print(f"\n[Tip] You can just press ENTER to use defaults for testing.")
    
    try:
        # 1. Inputs
        st_id = input(f"Enter Station ID (default: {sample_station}): ").strip() or sample_station
        ch_id = input(f"Enter Charger ID (default: {sample_charger}): ").strip() or sample_charger
        
        target_date_str = input("Enter Date (YYYY-MM-DD) [default: 2026-04-11]: ").strip()
        if not target_date_str:
            target_date = datetime(2026, 4, 11)
        else:
            target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
            
        hour_str = input("Enter Hour (0-23) [default: 1]: ").strip()
        hour = int(float(hour_str)) if hour_str else 1
        
        pwr_str = input("Enter expected Avg Power (kW) [default: 18.4]: ").strip()
        avg_pwr_kw = float(pwr_str) if pwr_str else 18.4
        
        energy_str = input("Enter current Energy stored (Wh) [default: 52560]: ").strip()
        max_energy_wh = float(energy_str) if energy_str else 52560.0
        
        # 2. Compute Features
        day_of_week = target_date.weekday()
        month = target_date.month
        is_weekend = 1 if day_of_week >= 5 else 0
        
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        
        is_idle_charger = 1 if avg_pwr_kw == 0 else 0
        
        # We assign period=3 (April 2026 patterns) to all future dates as it's the most recent
        collection_period = 3 
        
        # 3. Construct DataFrame matching the model's expected columns
        feature_cols = [
            "charging_station_id", "charger_id", "collection_period",
            "hour", "day_of_week", "month", "is_weekend", "hour_sin", "hour_cos", 
            "avg_pwr_kw", "is_idle_charger", "max_energy_wh"
        ]
        
        input_data = {
            "charging_station_id": [st_id],
            "charger_id": [ch_id],
            "collection_period": [collection_period],
            "hour": [hour],
            "day_of_week": [day_of_week],
            "month": [month],
            "is_weekend": [is_weekend],
            "hour_sin": [hour_sin],
            "hour_cos": [hour_cos],
            "avg_pwr_kw": [avg_pwr_kw],
            "is_idle_charger": [is_idle_charger],
            "max_energy_wh": [max_energy_wh]
        }
            
        X_predict = pd.DataFrame(input_data)[feature_cols]
        
        # 4. Predict
        predicted_soc = model.predict(X_predict)[0]
        predicted_soc = max(0, min(100, predicted_soc)) # SoC is 0-100%
        
        print("\n" + "="*50)
        print(" 🔮 SoC PREDICTION RESULT 🔮")
        print("="*50)
        print(f"  Target Date  : {target_date.strftime('%Y-%m-%d')} at {hour:02d}:00")
        print(f"  Station      : {st_id}")
        print(f"  Charger      : {ch_id}")
        print(f"  Assumed Pwr  : {avg_pwr_kw} kW")
        print(f"  Assumed Energy: {max_energy_wh} Wh")
        print(f"  -------------------------------------")
        print(f"  Estimated State of Charge : {predicted_soc:.2f} %")
        print("="*50 + "\n")

    except ValueError as e:
        print(f"\n[ERROR] Invalid input: {e}")
    except KeyboardInterrupt:
        print("\nPrediction cancelled.")

if __name__ == "__main__":
    main()
