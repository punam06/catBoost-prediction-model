"""
predict_demand.py
==================
Interactive terminal script to predict Station-Level Hourly Demand (total_energy_wh).

Usage:
  python src/04_demand_model/predict.py
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings("ignore")

MODEL_PATH = "models/catboost_demand_model.cbm"
DATA_PATH  = "data/processed/features_demand_model.csv"

def load_resources():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}")
        sys.exit(1)

    print("Loading Demand Forecasting model...")
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
            info = {
                "total_chargers": int(station_data["total_chargers"].max()),
                "station_avg_hourly_demand": station_data["station_avg_hourly_demand"].mean(),
                "station_peak_hourly_demand": station_data["station_peak_hourly_demand"].max(),
            }
            return info
    return {"total_chargers": 1, "station_avg_hourly_demand": 0, "station_peak_hourly_demand": 0}

def main():
    print("\n" + "=" * 55)
    print("  STATION DEMAND FORECASTER")
    print("=" * 55 + "\n")

    model, df = load_resources()

    sample_station = "008a2054178b149a52d9893112adf164"
    if df is not None and not df.empty:
        sample_station = df["charging_station_id"].iloc[0]

    print("[Tip] Press ENTER to use defaults for testing.\n")

    try:
        st_id = input(f"Enter Station ID (default: {sample_station}): ").strip() or sample_station

        target_date_str = input("Enter Date (YYYY-MM-DD) [default: tomorrow]: ").strip()
        if not target_date_str:
            target_date = datetime.now() + timedelta(days=1)
        else:
            target_date = datetime.strptime(target_date_str, "%Y-%m-%d")

        hour_str = input("Enter Hour (0-23) [default: 17]: ").strip()
        hour = int(float(hour_str)) if hour_str else 17

        day_of_week = target_date.weekday()
        month = target_date.month
        is_weekend = 1 if day_of_week >= 5 else 0
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        collection_period = 3

        station_info = get_station_info(df, st_id)

        feature_cols = [
            "charging_station_id", "total_chargers", "active_chargers",
            "utilization_rate", "avg_energy_per_charger", "peak_power_kw",
            "station_avg_hourly_demand", "station_peak_hourly_demand",
            "hour", "day_of_week", "month", "is_weekend",
            "hour_sin", "hour_cos", "collection_period"
        ]

        input_data = {
            "charging_station_id": [st_id],
            "total_chargers": [station_info["total_chargers"]],
            "active_chargers": [0],
            "utilization_rate": [0.0],
            "avg_energy_per_charger": [0.0],
            "peak_power_kw": [0.0],
            "station_avg_hourly_demand": [station_info["station_avg_hourly_demand"]],
            "station_peak_hourly_demand": [station_info["station_peak_hourly_demand"]],
            "hour": [hour],
            "day_of_week": [day_of_week],
            "month": [month],
            "is_weekend": [is_weekend],
            "hour_sin": [hour_sin],
            "hour_cos": [hour_cos],
            "collection_period": [collection_period],
        }

        X_predict = pd.DataFrame(input_data)[feature_cols]

        predicted_demand_wh = model.predict(X_predict)[0]
        predicted_demand_wh = max(0, predicted_demand_wh)
        predicted_demand_kwh = predicted_demand_wh / 1000.0

        total_chargers = station_info["total_chargers"]
        avg_per_charger = predicted_demand_wh / total_chargers if total_chargers > 0 else 0

        print("\n" + "=" * 55)
        print("  DEMAND FORECAST RESULT")
        print("=" * 55)
        print(f"  Target Date         : {target_date.strftime('%Y-%m-%d')} at {hour:02d}:00")
        print(f"  Station             : {st_id}")
        print(f"  Total Chargers      : {total_chargers}")
        print(f"  -------------------------------------")
        print(f"  Predicted Total Demand : {predicted_demand_wh:,.0f} Wh ({predicted_demand_kwh:,.2f} kWh)")
        print(f"  Avg per Charger        : {avg_per_charger:,.0f} Wh")

        if station_info["station_avg_hourly_demand"] > 0:
            ratio = predicted_demand_wh / station_info["station_avg_hourly_demand"]
            print(f"  vs Historical Avg      : {ratio:.1f}x average")

        print("=" * 55 + "\n")

    except ValueError as e:
        print(f"\n[ERROR] Invalid input: {e}")
    except KeyboardInterrupt:
        print("\nPrediction cancelled.")

if __name__ == "__main__":
    main()
