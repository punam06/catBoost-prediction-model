"""
predict_leftover.py
====================
Interactive terminal script to predict Leftover Energy (leftover_energy_wh).

Usage:
  python src/05_leftover_model/predict.py
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings("ignore")

MODEL_PATH = "models/catboost_leftover_model.cbm"
DATA_PATH  = "data/processed/features_leftover_model.csv"

def load_resources():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}")
        sys.exit(1)

    print("Loading Leftover Energy Prediction model...")
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)

    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        df = None

    return model, df

def get_charger_info(df, charger_id):
    if df is not None and not df.empty:
        charger_data = df[df["charger_id"] == charger_id]
        if not charger_data.empty:
            info = {
                "charger_max_energy_wh": charger_data["charger_max_energy_wh"].max(),
                "charger_mean_energy_wh": charger_data["charger_mean_energy_wh"].mean(),
                "charging_station_id": charger_data["charging_station_id"].iloc[0],
            }
            return info
    return None

def main():
    print("\n" + "=" * 55)
    print("  LEFTOVER ENERGY PREDICTOR")
    print("=" * 55 + "\n")

    model, df = load_resources()

    sample_charger = None
    if df is not None and not df.empty:
        sample_charger = df["charger_id"].iloc[0]

    print("[Tip] Press ENTER to use defaults for testing.\n")

    try:
        ch_id = input(f"Enter Charger ID (default: {sample_charger}): ").strip() or sample_charger

        target_date_str = input("Enter Date (YYYY-MM-DD) [default: tomorrow]: ").strip()
        if not target_date_str:
            target_date = datetime.now() + timedelta(days=1)
        else:
            target_date = datetime.strptime(target_date_str, "%Y-%m-%d")

        hour_str = input("Enter Hour (0-23) [default: 17]: ").strip()
        hour = int(float(hour_str)) if hour_str else 17

        power_str = input("Enter current power in kW (default: 0): ").strip()
        current_power = float(power_str) if power_str else 0.0

        energy_str = input("Enter energy consumed so far in Wh (default: 0): ").strip()
        energy_consumed = float(energy_str) if energy_str else 0.0

        charger_info = get_charger_info(df, ch_id)

        if charger_info is None:
            print(f"\n[WARN] Charger '{ch_id}' not found in training data. Using fallback estimates.")
            charger_max = energy_consumed * 1.5
            charger_mean = energy_consumed
            st_id = "unknown"
        else:
            charger_max = charger_info["charger_max_energy_wh"]
            charger_mean = charger_info["charger_mean_energy_wh"]
            st_id = charger_info["charging_station_id"]

        day_of_week = target_date.weekday()
        month = target_date.month
        is_weekend = 1 if day_of_week >= 5 else 0
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        collection_period = 3
        is_idle = 1 if current_power == 0 else 0
        active_sessions = 1 if current_power > 0 else 0
        estimated_capacity = charger_max * 1.1

        feature_cols = [
            "charging_station_id", "charger_id",
            "hour", "day_of_week", "month", "is_weekend",
            "hour_sin", "hour_cos", "collection_period",
            "avg_pwr_kw", "active_session_count", "is_idle_charger", "max_energy_wh",
            "charger_max_energy_wh", "charger_mean_energy_wh", "estimated_capacity_wh",
        ]

        input_data = {
            "charging_station_id": [st_id],
            "charger_id": [ch_id],
            "hour": [hour],
            "day_of_week": [day_of_week],
            "month": [month],
            "is_weekend": [is_weekend],
            "hour_sin": [hour_sin],
            "hour_cos": [hour_cos],
            "collection_period": [collection_period],
            "avg_pwr_kw": [current_power],
            "active_session_count": [active_sessions],
            "is_idle_charger": [is_idle],
            "max_energy_wh": [energy_consumed],
            "charger_max_energy_wh": [charger_max],
            "charger_mean_energy_wh": [charger_mean],
            "estimated_capacity_wh": [estimated_capacity],
        }

        X_predict = pd.DataFrame(input_data)[feature_cols]

        predicted_leftover_wh = model.predict(X_predict)[0]
        predicted_leftover_wh = max(0, predicted_leftover_wh)

        capacity_kwh = estimated_capacity / 1000.0
        consumed_kwh = energy_consumed / 1000.0
        leftover_kwh = predicted_leftover_wh / 1000.0
        utilization = (energy_consumed / estimated_capacity * 100) if estimated_capacity > 0 else 0

        print("\n" + "=" * 55)
        print("  LEFTOVER ENERGY PREDICTION RESULT")
        print("=" * 55)
        print(f"  Target Date          : {target_date.strftime('%Y-%m-%d')} at {hour:02d}:00")
        print(f"  Charger              : {ch_id}")
        print(f"  Station              : {st_id}")
        print(f"  -------------------------------------")
        print(f"  Estimated Capacity   : {estimated_capacity:,.0f} Wh ({capacity_kwh:,.2f} kWh)")
        print(f"  Energy Consumed      : {energy_consumed:,.0f} Wh ({consumed_kwh:,.2f} kWh)")
        print(f"  Predicted Leftover   : {predicted_leftover_wh:,.0f} Wh ({leftover_kwh:,.2f} kWh)")
        print(f"  Utilization          : {utilization:.1f}%")

        if utilization > 90:
            print("  Session nearly complete — minimal energy remaining")
        elif utilization > 50:
            print("  Session in progress — moderate energy remaining")
        else:
            print("  Session early stage — significant energy remaining")

        print("=" * 55 + "\n")

    except ValueError as e:
        print(f"\n[ERROR] Invalid input: {e}")
    except KeyboardInterrupt:
        print("\nPrediction cancelled.")

if __name__ == "__main__":
    main()
