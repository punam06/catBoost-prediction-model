"""
features.py
============
Recreates the exact feature-engineering logic from each model's original
predict.py script (src/0X_*_model/predict.py), so the API produces the same
inputs the models were validated against.
"""
import numpy as np
import pandas as pd


def _time_features(target_date, hour):
    day_of_week = target_date.weekday()
    month = target_date.month
    is_weekend = 1 if day_of_week >= 5 else 0
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    return day_of_week, month, is_weekend, hour_sin, hour_cos


COLLECTION_PERIOD = 3  # matches original scripts: "period=3 (April 2026 patterns)"


def build_energy_features(req):
    day_of_week, month, is_weekend, hour_sin, hour_cos = _time_features(req.target_date, req.hour)
    is_late_night = 1 if req.hour >= 2 else 0
    is_idle_charger = 1 if req.avg_pwr_kw == 0 else 0

    data = {
        "charging_station_id": [req.station_id],
        "charger_id": [req.charger_id],
        "collection_period": [COLLECTION_PERIOD],
        "hour": [req.hour],
        "day_of_week": [day_of_week],
        "month": [month],
        "is_weekend": [is_weekend],
        "is_late_night": [is_late_night],
        "hour_sin": [hour_sin],
        "hour_cos": [hour_cos],
        "avg_pwr_kw": [req.avg_pwr_kw],
        "active_session_count": [1],
        "is_idle_charger": [is_idle_charger],
        "station_mean_energy_wh": [req.station_mean_energy_wh],
        "station_median_energy_wh": [req.station_median_energy_wh],
        "station_mean_power_kw": [req.station_mean_power_kw],
        "station_max_power_kw": [req.station_max_power_kw],
        "station_charger_count": [req.station_charger_count],
        "charger_mean_energy_wh": [req.charger_mean_energy_wh],
        "charger_median_energy_wh": [req.charger_median_energy_wh],
    }
    cols = [
        "charging_station_id", "charger_id", "collection_period",
        "hour", "day_of_week", "month", "is_weekend", "is_late_night",
        "hour_sin", "hour_cos", "avg_pwr_kw", "active_session_count", "is_idle_charger",
        "station_mean_energy_wh", "station_median_energy_wh",
        "station_mean_power_kw", "station_max_power_kw",
        "station_charger_count",
        "charger_mean_energy_wh", "charger_median_energy_wh",
    ]
    return pd.DataFrame(data)[cols]


def build_soc_features(req):
    day_of_week, month, is_weekend, hour_sin, hour_cos = _time_features(req.target_date, req.hour)
    is_idle_charger = 1 if req.avg_pwr_kw == 0 else 0

    data = {
        "charging_station_id": [req.station_id],
        "charger_id": [req.charger_id],
        "collection_period": [COLLECTION_PERIOD],
        "hour": [req.hour],
        "day_of_week": [day_of_week],
        "month": [month],
        "is_weekend": [is_weekend],
        "hour_sin": [hour_sin],
        "hour_cos": [hour_cos],
        "avg_pwr_kw": [req.avg_pwr_kw],
        "is_idle_charger": [is_idle_charger],
        "max_energy_wh": [req.max_energy_wh],
    }
    cols = [
        "charging_station_id", "charger_id", "collection_period",
        "hour", "day_of_week", "month", "is_weekend", "hour_sin", "hour_cos",
        "avg_pwr_kw", "is_idle_charger", "max_energy_wh",
    ]
    return pd.DataFrame(data)[cols]


def build_congestion_features(req):
    day_of_week, month, is_weekend, hour_sin, hour_cos = _time_features(req.target_date, req.hour)
    total_chargers = req.total_chargers if req.total_chargers is not None else 1

    data = {
        "charging_station_id": [req.station_id],
        "total_chargers": [total_chargers],
        "hour": [req.hour],
        "day_of_week": [day_of_week],
        "month": [month],
        "is_weekend": [is_weekend],
        "hour_sin": [hour_sin],
        "hour_cos": [hour_cos],
        "collection_period": [COLLECTION_PERIOD],
        "lag_1_active": [req.lag_1_active],
        "lag_2_active": [req.lag_2_active],
        "lag_1_power_kw": [req.lag_1_power_kw],
        "roll_3_active": [req.roll_3_active],
        "roll_6_active": [req.roll_6_active],
    }
    cols = [
        "charging_station_id", "total_chargers", "hour", "day_of_week",
        "month", "is_weekend", "hour_sin", "hour_cos", "collection_period",
        "lag_1_active", "lag_2_active", "lag_1_power_kw", "roll_3_active", "roll_6_active",
    ]
    return pd.DataFrame(data)[cols], total_chargers


def build_demand_features(req):
    day_of_week, month, is_weekend, hour_sin, hour_cos = _time_features(req.target_date, req.hour)

    data = {
        "charging_station_id": [req.station_id],
        "total_chargers": [req.total_chargers],
        "active_chargers": [req.active_chargers],
        "utilization_rate": [req.utilization_rate],
        "avg_energy_per_charger": [req.avg_energy_per_charger],
        "peak_power_kw": [req.peak_power_kw],
        "station_avg_hourly_demand": [req.station_avg_hourly_demand],
        "station_peak_hourly_demand": [req.station_peak_hourly_demand],
        "hour": [req.hour],
        "day_of_week": [day_of_week],
        "month": [month],
        "is_weekend": [is_weekend],
        "hour_sin": [hour_sin],
        "hour_cos": [hour_cos],
        "collection_period": [COLLECTION_PERIOD],
    }
    cols = [
        "charging_station_id", "total_chargers", "active_chargers",
        "utilization_rate", "avg_energy_per_charger", "peak_power_kw",
        "station_avg_hourly_demand", "station_peak_hourly_demand",
        "hour", "day_of_week", "month", "is_weekend",
        "hour_sin", "hour_cos", "collection_period",
    ]
    return pd.DataFrame(data)[cols]


def build_leftover_features(req):
    day_of_week, month, is_weekend, hour_sin, hour_cos = _time_features(req.target_date, req.hour)
    is_idle = 1 if req.current_power_kw == 0 else 0
    active_sessions = 1 if req.current_power_kw > 0 else 0

    # Same fallback the original script uses when charger history is unknown
    if req.charger_max_energy_wh is not None:
        charger_max = req.charger_max_energy_wh
    else:
        charger_max = req.energy_consumed_wh * 1.5

    if req.charger_mean_energy_wh is not None:
        charger_mean = req.charger_mean_energy_wh
    else:
        charger_mean = req.energy_consumed_wh

    station_id = req.station_id if req.station_id else "unknown"
    estimated_capacity = charger_max * 1.1

    data = {
        "charging_station_id": [station_id],
        "charger_id": [req.charger_id],
        "hour": [req.hour],
        "day_of_week": [day_of_week],
        "month": [month],
        "is_weekend": [is_weekend],
        "hour_sin": [hour_sin],
        "hour_cos": [hour_cos],
        "collection_period": [COLLECTION_PERIOD],
        "avg_pwr_kw": [req.current_power_kw],
        "active_session_count": [active_sessions],
        "is_idle_charger": [is_idle],
        "max_energy_wh": [req.energy_consumed_wh],
        "charger_max_energy_wh": [charger_max],
        "charger_mean_energy_wh": [charger_mean],
        "estimated_capacity_wh": [estimated_capacity],
    }
    cols = [
        "charging_station_id", "charger_id",
        "hour", "day_of_week", "month", "is_weekend",
        "hour_sin", "hour_cos", "collection_period",
        "avg_pwr_kw", "active_session_count", "is_idle_charger", "max_energy_wh",
        "charger_max_energy_wh", "charger_mean_energy_wh", "estimated_capacity_wh",
    ]
    return pd.DataFrame(data)[cols], estimated_capacity
