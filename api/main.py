"""
main.py
=======
FastAPI service wrapping the 5 CatBoost models from
punam06/catBoost-prediction-model:
  - Energy consumption predictor
  - State of Charge (SoC) predictor
  - Station congestion predictor
  - Station demand forecaster
  - Leftover energy predictor

Run locally:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Then open http://localhost:8000/docs for interactive Swagger UI.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.model_loader import load_all_models, get_model
from api import features
from api.schemas import (
    EnergyRequest, EnergyResponse,
    SocRequest, SocResponse,
    CongestionRequest, CongestionResponse,
    DemandRequest, DemandResponse,
    LeftoverRequest, LeftoverResponse,
)

app = FastAPI(
    title="EV Charging Prediction API",
    description="REST API wrapping 5 CatBoost models for EV charging station prediction.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    load_all_models()


@app.get("/")
def root():
    return {"status": "ok", "message": "EV Charging Prediction API is running. See /docs."}


@app.get("/health")
def health():
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# 1. Energy Consumption
# ---------------------------------------------------------------------------
@app.post("/predict/energy", response_model=EnergyResponse)
def predict_energy(req: EnergyRequest):
    try:
        model = get_model("energy")
        X = features.build_energy_features(req)
        pred = max(0.0, float(model.predict(X)[0]))
        return EnergyResponse(predicted_energy_wh=pred, predicted_energy_kwh=pred / 1000.0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# 2. State of Charge
# ---------------------------------------------------------------------------
@app.post("/predict/soc", response_model=SocResponse)
def predict_soc(req: SocRequest):
    try:
        model = get_model("soc")
        X = features.build_soc_features(req)
        pred = float(model.predict(X)[0])
        pred = max(0.0, min(100.0, pred))
        return SocResponse(predicted_soc_pct=pred)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# 3. Station Congestion
# ---------------------------------------------------------------------------
@app.post("/predict/congestion", response_model=CongestionResponse)
def predict_congestion(req: CongestionRequest):
    try:
        model = get_model("congestion")
        X, total_chargers = features.build_congestion_features(req)
        pred = float(model.predict(X)[0])
        pred = max(0.0, min(total_chargers, pred))
        occupancy = (pred / total_chargers) * 100 if total_chargers > 0 else 0.0

        note = None
        if all(v == 0.0 for v in [req.lag_1_active, req.lag_2_active, req.lag_1_power_kw, req.roll_3_active, req.roll_6_active]):
            note = (
                "lag/rolling features were not provided and defaulted to 0. "
                "This model was trained on real historical lag values not present "
                "in the original repo's predict.py — treat this prediction as provisional "
                "until real lag values are supplied."
            )

        return CongestionResponse(
            predicted_active_chargers=pred,
            total_chargers=total_chargers,
            predicted_occupancy_pct=occupancy,
            note=note,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# 4. Station Demand
# ---------------------------------------------------------------------------
@app.post("/predict/demand", response_model=DemandResponse)
def predict_demand(req: DemandRequest):
    try:
        model = get_model("demand")
        X = features.build_demand_features(req)
        pred = max(0.0, float(model.predict(X)[0]))
        avg_per_charger = pred / req.total_chargers if req.total_chargers > 0 else 0.0
        return DemandResponse(
            predicted_demand_wh=pred,
            predicted_demand_kwh=pred / 1000.0,
            avg_per_charger_wh=avg_per_charger,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# 5. Leftover Energy
# ---------------------------------------------------------------------------
@app.post("/predict/leftover", response_model=LeftoverResponse)
def predict_leftover(req: LeftoverRequest):
    try:
        model = get_model("leftover")
        X, estimated_capacity = features.build_leftover_features(req)
        pred = max(0.0, float(model.predict(X)[0]))
        utilization = (req.energy_consumed_wh / estimated_capacity * 100) if estimated_capacity > 0 else 0.0
        return LeftoverResponse(
            predicted_leftover_wh=pred,
            predicted_leftover_kwh=pred / 1000.0,
            estimated_capacity_wh=estimated_capacity,
            utilization_pct=utilization,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
