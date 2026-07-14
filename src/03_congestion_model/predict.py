"""
10_predict_congestion.py
========================
Interactive terminal script to predict Station Congestion (active_chargers).

Usage:
  source venv/bin/activate
  python3 src/10_predict_congestion.py
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings("ignore")

MODEL_PATH = "models/catboost_congestion_model.cbm"
DATA_PATH  = "data/processed/features_station_model.csv"

def load_resources():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}")
        sys.exit(1)
        
    print("Loading Congestion Prediction model...")
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)
    
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        df = None
        
    return model, df

def get_station_info(df, station_id):
    if df is not None and not df.empty:
        station_data = df[df["charging_station_id"] == station_id]
        if not station_data.empty:
            return int(station_data["total_chargers"].max())
    return 1 # Fallback default

def main():
    print("\n" + "="*50)
    print(" 🚦 STATION CONGESTION PREDICTOR 🚦")
    print("="*50 + "\n")
    
    model, df = load_resources()
    
    # Provide a sample station
    sample_station = "008a2054178b149a52d9893112adf164"
    if df is not None and not df.empty:
        sample_station = df["charging_station_id"].iloc[0]
        
    print(f"\n[Tip] You can just press ENTER to use defaults for testing.")
    
    try:
        # 1. Inputs
        st_id = input(f"Enter Station ID (default: {sample_station}): ").strip() or sample_station
        
        target_date_str = input("Enter Date (YYYY-MM-DD) [default: tomorrow]: ").strip()
        if not target_date_str:
            target_date = datetime.now() + timedelta(days=1)
        else:
            target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
            
        hour_str = input("Enter Hour (0-23) [default: 17]: ").strip()
        hour = int(float(hour_str)) if hour_str else 17
        
        # 2. Compute Features
        day_of_week = target_date.weekday()
        month = target_date.month
        is_weekend = 1 if day_of_week >= 5 else 0
        
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        
        # We assign period=3 (April 2026 patterns) to all future dates
        collection_period = 3 
        
        # Get total chargers for this station to help the model
        total_chargers = get_station_info(df, st_id)
        
        # 3. Construct DataFrame matching the model's expected columns
        feature_cols = [
            "charging_station_id", "total_chargers", "hour", "day_of_week", 
            "month", "is_weekend", "hour_sin", "hour_cos", "collection_period"
        ]
        
        input_data = {
            "charging_station_id": [st_id],
            "total_chargers": [total_chargers],
            "hour": [hour],
            "day_of_week": [day_of_week],
            "month": [month],
            "is_weekend": [is_weekend],
            "hour_sin": [hour_sin],
            "hour_cos": [hour_cos],
            "collection_period": [collection_period]
        }
            
        X_predict = pd.DataFrame(input_data)[feature_cols]
        
        # 4. Predict
        predicted_active = model.predict(X_predict)[0]
        # Must be between 0 and total_chargers
        predicted_active = max(0, min(total_chargers, predicted_active))
        
        print("\n" + "="*50)
        print(" 🔮 CONGESTION PREDICTION RESULT 🔮")
        print("="*50)
        print(f"  Target Date    : {target_date.strftime('%Y-%m-%d')} at {hour:02d}:00")
        print(f"  Station        : {st_id}")
        print(f"  Total Chargers : {total_chargers}")
        print(f"  -------------------------------------")
        print(f"  Estimated Active Chargers : {predicted_active:.2f}")
        
        # Calculate percentage occupancy
        occupancy = (predicted_active / total_chargers) * 100
        print(f"  Predicted Occupancy Rate  : {occupancy:.1f}%")
        
        if occupancy > 80:
            print("  ⚠️ High Congestion Expected!")
        elif occupancy < 30:
            print("  ✅ Wide Open Availability Expected")
            
        print("="*50 + "\n")

    except ValueError as e:
        print(f"\n[ERROR] Invalid input: {e}")
    except KeyboardInterrupt:
        print("\nPrediction cancelled.")

if __name__ == "__main__":
    main()
