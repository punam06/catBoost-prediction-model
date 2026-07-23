"""
schemas.py
==========
Pydantic request/response models for every prediction endpoint.
Field defaults mirror the fallback values used in the original predict.py
scripts, so the API works even without the (gitignored) historical CSVs.
"""
from datetime import date
from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. Energy Consumption Model
# ---------------------------------------------------------------------------
class EnergyRequest(BaseModel):
    station_id: str = Field(..., description="Charging station ID")
    charger_id: str = Field(..., description="Charger ID")
    target_date: date = Field(..., description="Prediction date, YYYY-MM-DD")
    hour: int = Field(18, ge=0, le=23, description="Hour of day (0-23)")
    avg_pwr_kw: float = Field(25.0, description="Expected average power draw in kW")

    # Historical aggregates (from data/processed/features_energy_model.csv).
    # Optional — fall back to -999 (CatBoost's "missing" sentinel in this project)
    # if you don't have the friend's historical aggregates handy.
    station_mean_energy_wh: float = -999
    station_median_energy_wh: float = -999
    station_mean_power_kw: float = -999
    station_max_power_kw: float = -999
    station_charger_count: float = -999
    charger_mean_energy_wh: float = -999
    charger_median_energy_wh: float = -999


class EnergyResponse(BaseModel):
    predicted_energy_wh: float
    predicted_energy_kwh: float


# ---------------------------------------------------------------------------
# 2. State of Charge (SoC) Model
# ---------------------------------------------------------------------------
class SocRequest(BaseModel):
    station_id: str
    charger_id: str
    target_date: date
    hour: int = Field(1, ge=0, le=23)
    avg_pwr_kw: float = 18.4
    max_energy_wh: float = 52560.0


class SocResponse(BaseModel):
    predicted_soc_pct: float


# ---------------------------------------------------------------------------
# 3. Station Congestion Model
#    NOTE: the model was trained with 5 lag/rolling features that are absent
#    from the original repo's predict.py. They're exposed here as optional
#    inputs (default 0.0) — see README_API.md for details.
# ---------------------------------------------------------------------------
class CongestionRequest(BaseModel):
    station_id: str
    target_date: date
    hour: int = Field(17, ge=0, le=23)
    total_chargers: Optional[int] = Field(
        None, description="If omitted, defaults to 1 (unknown station fallback)"
    )

    # Present in the trained model but missing from the original script.
    # Provide real values if available; otherwise defaults are used and the
    # prediction should be treated as provisional (see README_API.md).
    lag_1_active: float = 0.0
    lag_2_active: float = 0.0
    lag_1_power_kw: float = 0.0
    roll_3_active: float = 0.0
    roll_6_active: float = 0.0


class CongestionResponse(BaseModel):
    predicted_active_chargers: float
    total_chargers: int
    predicted_occupancy_pct: float
    note: Optional[str] = None


# ---------------------------------------------------------------------------
# 4. Station Demand Model
# ---------------------------------------------------------------------------
class DemandRequest(BaseModel):
    station_id: str
    target_date: date
    hour: int = Field(17, ge=0, le=23)

    total_chargers: float = 1
    active_chargers: float = 0
    utilization_rate: float = 0.0
    avg_energy_per_charger: float = 0.0
    peak_power_kw: float = 0.0
    station_avg_hourly_demand: float = 0.0
    station_peak_hourly_demand: float = 0.0


class DemandResponse(BaseModel):
    predicted_demand_wh: float
    predicted_demand_kwh: float
    avg_per_charger_wh: float


# ---------------------------------------------------------------------------
# 5. Leftover Energy Model
# ---------------------------------------------------------------------------
class LeftoverRequest(BaseModel):
    charger_id: str
    station_id: Optional[str] = Field(None, description="If unknown, pass null")
    target_date: date
    hour: int = Field(17, ge=0, le=23)
    current_power_kw: float = 0.0
    energy_consumed_wh: float = 0.0

    # From historical charger aggregates; if you don't have them, leave null
    # and they'll be derived the same way the original script falls back:
    # charger_max_energy_wh = energy_consumed_wh * 1.5, charger_mean = energy_consumed_wh
    charger_max_energy_wh: Optional[float] = None
    charger_mean_energy_wh: Optional[float] = None


class LeftoverResponse(BaseModel):
    predicted_leftover_wh: float
    predicted_leftover_kwh: float
    estimated_capacity_wh: float
    utilization_pct: float
