# CatBoost Prediction Models — EV Charging Stations

A complete machine learning pipeline for electric vehicle (EV) charging station analytics, built with **CatBoost**. From raw Elasticsearch telemetry data to a production REST API — this project covers data extraction, cleaning, feature engineering, model training, evaluation, and deployment.

---

## Table of Contents

1. [Data Extraction: Elasticsearch & Kibana](#1-data-extraction-elasticsearch--kibana)
2. [JSON to CSV Conversion](#2-json-to-csv-conversion)
3. [CSV Merging](#3-csv-merging)
4. [Data Cleaning](#4-data-cleaning)
5. [Feature Engineering & Processing](#5-feature-engineering--processing)
6. [Model Training](#6-model-training)
7. [Accuracy Measurement: Graphs & Terminal Scripts](#7-accuracy-measurement-graphs--terminal-scripts)
8. [Outputs & Responses](#8-outputs--responses)
9. [REST API](#9-rest-api)
10. [API Response Verification](#10-api-response-verification)
11. [Common Bugs & Fixes](#11-common-bugs--fixes)
12. [Usage Guide](#12-usage-guide)
13. [Project Structure](#13-project-structure)

---

## 1. Data Extraction: Elasticsearch & Kibana

### What is Elasticsearch?

Elasticsearch is a distributed search and analytics engine. Our EV charging station hardware (chargers, meters, controllers) sends telemetry data in real-time — power draw, energy consumed, session counts, timestamps — which gets indexed into Elasticsearch clusters.

### What is Kibana?

Kibana is the visualization layer on top of Elasticsearch. It provides a query interface (Dev Tools) and dashboards to explore, filter, and export data.

### Data Extraction Process

1. **Connect to Kibana** — Open the Kibana Dev Tools console (`http://<host>:5601/app/dev_tools`)
2. **Write an SQL-like query** — Elasticsearch supports SQL queries through the `_xpack/sql` endpoint. Example:

```sql
SELECT
  charging_station_id,
  charger_id,
  timestamp,
  avg_pwr_kw,
  max_energy_wh,
  active_session_count,
  soc_pct
FROM ev_charging_telemetry
WHERE timestamp >= '2025-11-01'
  AND timestamp < '2026-05-01'
ORDER BY timestamp ASC
```

3. **Export as JSON** — The query results are returned in JSON format. We copy the response and save it into `.json` files locally.
4. **Multiple extractions** — Data was extracted in batches corresponding to three collection periods:
   - **Period 1:** November 2025 (2 days of data)
   - **Period 2:** December 2025 (4 days of data)
   - **Period 3:** April 2026 (5 days of data)

> These periods are non-continuous — there are gaps of 4 days and 113 days between them. This is critical for later splitting strategy.

---

## 2. JSON to CSV Conversion

The raw Elasticsearch responses are nested JSON objects. A Python script parses them into flat CSV format.

### Script
```
data/raw/All_Merger.csv  ← (output of JSON-to-CSV conversion)
```

### What it does
- Reads each `.json` file extracted from Kibana
- Flattens nested JSON fields (e.g., `hits.hits._source.*`)
- Maps field names to clean column names (`charging_station_id`, `charger_id`, `avg_pwr_kw`, etc.)
- Parses timestamps into UTC datetime format
- Concatenates all batch files into a single `All_Merger.csv`

### Input
```json
{
  "hits": {
    "hits": [
      {
        "_source": {
          "station_id": "abc123",
          "charger_id": "CH-001",
          "timestamp": "2025-11-01T10:30:00Z",
          "power_kw": 22.5,
          "energy_wh": 15000
        }
      }
    ]
  }
}
```

### Output
| charging_station_id | charger_id | timestamp | avg_pwr_kw | max_energy_wh | ... |
|---|---|---|---|---|---|
| abc123 | CH-001 | 2025-11-01 10:30:00 | 22.5 | 15000 | ... |

---

## 3. CSV Merging

Multiple period-specific CSVs are merged into one master file.

### Input files
```
data/raw/period1_nov2025.json  →  period1.csv
data/raw/period2_dec2025.json  →  period2.csv
data/raw/period3_apr2026.json  →  period3.csv
```

### Output
```
data/raw/All_Merger.csv  (154,271 rows × 13 columns)
```

### Columns in All_Merger.csv
| Column | Description |
|---|---|
| `charging_station_id` | Unique station identifier |
| `charger_id` | Unique charger identifier |
| `timestamp` | UTC datetime of the reading |
| `avg_pwr_kw` | Average power draw in kilowatts |
| `max_energy_wh` | Cumulative energy consumed in watt-hours |
| `active_session_count` | Number of active charging sessions |
| `soc_pct` | State of charge percentage (96.5% null) |
| `hour` | Hour of day (0-23) |
| `day_of_week` | Day of week (0=Monday, 6=Sunday) |
| `month` | Month number (11, 12, or 4) |
| `is_weekend` | Binary flag (1 if Saturday/Sunday) |
| `is_late_night` | Binary flag (1 if hour >= 2) |
| `collection_period` | Data window label (1=Nov, 2=Dec, 3=Apr) |

---

## 4. Data Cleaning

### Script
```bash
python3 src/00_data_cleaning/cleaner.py
```

### Input
```
data/raw/All_Merger.csv  (154,271 rows)
```

### Output
```
data/processed/cleaned_data.csv  (86,615 rows × 14 columns)
```

### 8-Step Cleaning Pipeline

| Step | Action | Why |
|---|---|---|
| 1 | Remove exact full-row duplicates | Sensor double-logging creates duplicates |
| 2 | Resolve near-duplicates (same station+charger+timestamp → take mean) | Sensor glitches produce conflicting values at the same instant |
| 3 | Remove physically invalid negative values | `avg_pwr_kw < 0` or `max_energy_wh < 0` is impossible |
| 4 | Cap extreme outliers at 99th percentile | A few readings are orders of magnitude above normal |
| 5 | Remove dead rows (power=0 AND energy=0) | No signal — charger was completely offline |
| 6 | Remove no-session rows (active_session_count=0) | No charging session active, no useful data |
| 7 | Fix data types + add collection_period label | Cast floats to ints, add period 1/2/3 labels |
| 8 | Cap station representation at 300 rows/station | Prevents any single station from dominating training |

### Result
- **154,271 → 86,615 rows** (43.8% reduction)
- **21,871 exact duplicates removed**
- **62 negative power values removed**
- **Clean, model-ready dataset with no physically invalid readings**

---

## 5. Feature Engineering & Processing

Each model requires its own feature preparation script because different models predict different targets and need different input features.

### 5a. Charger-Level Feature Preparation

#### Energy Model Features
```bash
python3 src/01_energy_model/prepare_features.py
```
- **Input:** `data/processed/features_charger.csv` (88,808 rows × 20 columns)
- **Output:** `data/processed/features_energy_model.csv` (88,808 rows × 23 columns)
- **Target:** `max_energy_wh` (energy consumed during session)

**Critical fixes applied:**
1. **Data Leakage removed** — 3 columns derived from the target were deleted:
   - `estimated_capacity_wh` (= charger_max_energy × 1.1)
   - `leftover_energy_wh` (= estimated_capacity - max_energy)
   - `power_to_energy_ratio` (= avg_pwr_kw / (max_energy_wh / 1000))
2. **Idle phase flag added** — `is_idle_charger` binary flag (86.8% of rows are idle)
3. **Aggregate leakage fixed** — Station/charger historical averages are now computed on the TRAINING split only, then joined to val/test rows. This prevents the model from "seeing the future."

**Features generated:**

| Feature | Type | Description |
|---|---|---|
| `charging_station_id` | Categorical | Station identifier (CatBoost native) |
| `charger_id` | Categorical | Charger identifier (CatBoost native) |
| `collection_period` | Integer | Data window (1, 2, or 3) |
| `hour` | Integer | Hour of day (0-23) |
| `day_of_week` | Integer | Day of week (0-6) |
| `month` | Integer | Month (11, 12, or 4) |
| `is_weekend` | Binary | 1 if Saturday/Sunday |
| `is_late_night` | Binary | 1 if hour >= 2 |
| `hour_sin` | Float | `sin(2π × hour/24)` — cyclic time encoding |
| `hour_cos` | Float | `cos(2π × hour/24)` — cyclic time encoding |
| `avg_pwr_kw` | Float | Average power draw (kW) |
| `active_session_count` | Integer | Active charging sessions |
| `is_idle_charger` | Binary | 1 if power = 0 |
| `station_mean_energy_wh` | Float | Mean energy per station (train-only aggregate) |
| `station_median_energy_wh` | Float | Median energy per station |
| `station_mean_power_kw` | Float | Mean power per station |
| `station_max_power_kw` | Float | Max power per station |
| `station_charger_count` | Integer | Number of chargers at station |
| `charger_mean_energy_wh` | Float | Mean energy per charger (train-only aggregate) |
| `charger_median_energy_wh` | Float | Median energy per charger |
| `max_energy_wh` | Float | **TARGET** — energy consumed |

#### SoC Model Features
```bash
python3 src/02_soc_model/prepare_features.py
```
- **Input:** `data/processed/cleaned_data.csv` (filtered to rows where `soc_pct` is not null)
- **Output:** `data/processed/features_soc_model.csv` (2,064 rows × 15 columns)
- **Target:** `soc_pct` (state of charge percentage)

#### Station-Level Features (Congestion & Demand)
```bash
python3 src/03_congestion_model/prepare_features.py
```
- **Input:** `data/processed/cleaned_data.csv` (charger-level rows)
- **Output:** `data/processed/features_station_model.csv` (21,482 station-level snapshots)
- **Process:** Aggregates charger-level data to station-level, then generates time-series lag and rolling features

**Lag/Rolling features for congestion:**

| Feature | Description |
|---|---|
| `lag_1_active` | Active chargers 1 interval ago |
| `lag_2_active` | Active chargers 2 intervals ago |
| `lag_1_power_kw` | Power draw 1 interval ago |
| `roll_3_active` | Rolling average active chargers over last 3 intervals |
| `roll_6_active` | Rolling average active chargers over last 6 intervals |

#### Demand Model Features
```bash
python3 src/04_demand_model/prepare_features.py
```
- **Output:** `data/processed/features_demand_model.csv`
- **Target:** Total hourly energy demand (Wh) per station

#### Leftover Model Features
```bash
python3 src/05_leftover_model/prepare_features.py
```
- **Output:** `data/processed/features_leftover_model.csv`
- **Target:** `leftover_energy_wh` (unused capacity)

### Data Splitting Strategy

All models use a **period-aware temporal split** (not random):

```
Period 1 (Nov 2025) ─┐
                      ├──→ TRAIN (80%)
Period 2 (Dec 2025) ─┘
Period 3 (Apr 2026) ─┬→ First 80%  → TRAIN
                     ├→ Middle 10% → VAL (early stopping)
                     └→ Last 10%   → TEST (final evaluation)
```

This ensures the model never trains on future data and is evaluated on the most recent held-out period.

---

## 6. Model Training

### Shared Training Pipeline
```bash
# Train a specific model (example: energy)
python3 src/01_energy_model/train_catboost.py

# Or retrain ALL models
python3 qa_full_verification.py
```

### Training Process (each model)

1. **Load** the feature CSV
2. **Split** using pre-defined train/val/test labels
3. **Create CatBoost Pools** — CatBoost handles categorical features (station/charger IDs) natively, no one-hot encoding needed
4. **Train** with early stopping:
   - Up to 1,000 decision tree iterations
   - Learning rate: 0.05
   - Early stopping: halts if validation RMSE doesn't improve for 50 rounds
5. **Evaluate** on the held-out test set
6. **Generate plots** — Feature importance, actual vs predicted, residuals, SHAP
7. **Save** the trained model as a `.cbm` binary file

### CatBoost Hyperparameters (shared across models)

| Parameter | Value | Purpose |
|---|---|---|
| `iterations` | 1000 | Maximum number of decision trees |
| `learning_rate` | 0.05 | Step size for each tree |
| `depth` | 8 | Maximum tree depth |
| `l2_leaf_reg` | 3 | L2 regularization |
| `loss_function` | RMSE | Training objective |
| `early_stopping_rounds` | 50 | Stop if no improvement for 50 rounds |
| `random_seed` | 42 | Reproducibility |

---

## 7. Accuracy Measurement: Graphs & Terminal Scripts

### 7a. Terminal Metrics (automatic on every training run)

Every `train.py` script prints metrics to the terminal:

```
[EVAL]  Test Set Metrics:
        RMSE  :    20,682.62 Wh
        MAE   :     5,468.58 Wh
        R²    :       0.9977
```

**Metrics explained:**

| Metric | What it measures | Good value |
|---|---|---|
| **RMSE** (Root Mean Squared Error) | Average prediction error, penalizes large errors | Lower is better |
| **MAE** (Mean Absolute Error) | Average absolute error in original units | Lower is better |
| **R²** (R-Squared) | Proportion of variance explained (0-1) | Closer to 1 is better |

### 7b. Generated Graphs (saved to `outputs/`)

Each training run automatically generates:

#### Actual vs Predicted Scatter Plot
```
outputs/energy_model/catboost_energy_actual_vs_predicted.png
```
- X-axis: actual values, Y-axis: predicted values
- Perfect predictions fall on the red dashed diagonal line
- Tight clustering around the diagonal = high accuracy

#### Feature Importance Bar Chart
```
outputs/energy_model/catboost_energy_feature_importance.png
```
- Shows which features the model relied on most
- Helps with model interpretability and debugging

#### Residual Distribution
```
outputs/energy_model/catboost_energy_residuals.png
```
- Histogram of prediction errors (actual - predicted)
- Should be centered at 0 with a bell-curve shape
- Skewed residuals indicate systematic bias

#### SHAP Summary Plot
```
outputs/energy_model/catboost_energy_shap_summary.png
```
- Advanced explainability: shows how each feature pushes predictions up or down
- Color = feature value (red = high, blue = low)

### 7c. QA Verification Script
```bash
python3 qa_full_verification.py
```
This is the master verification tool that:
1. Inspects all CSV files for nulls, duplicates, and data quality issues
2. **Re-runs** every feature preparation pipeline from scratch
3. **Re-trains** all 3 core models (Energy, SoC, Congestion)
4. Prints a final **accuracy comparison table**:

```
  Model                    RMSE          MAE         R²    Test Rows
  ──────────────────── ──────────── ──────────── ────────── ────────────
  Energy               20,682.62     5,468.58     0.9977        8,882
  SoC                       7.95         3.91     0.8237          207
  Congestion                0.50         0.15     0.8133        2,149
```

---

## 8. Outputs & Responses

### Model Binary Files
```
models/
├── catboost_energy_model.cbm       # Energy consumption predictor
├── catboost_soc_model.cbm          # State of charge predictor
├── catboost_congestion_model.cbm   # Station congestion predictor
├── catboost_demand_model.cbm       # Station demand forecaster
└── catboost_leftover_model.cbm     # Leftover energy predictor
```

### CLI Prediction Outputs

**Energy Model:**
```
Target Date  : 2026-04-12 at 18:00
Station      : 008a2054178b149a52d9893112adf164
Charger      : T124-IT1-1423-083
Assumed Pwr  : 25.0 kW
Estimated Energy Consumption : 5,851.83 Wh
                             : 5.85 kWh
```

**Congestion Model:**
```
Target Date    : 2026-04-12 at 18:00
Station        : 008a2054178b149a52d9893112adf164
Total Chargers : 6
Estimated Active Chargers : 2.37
Predicted Occupancy Rate  : 39.5%
```

**Leftover Model:**
```
Estimated Capacity   : 24,750 Wh (24.75 kWh)
Energy Consumed      : 15,000 Wh (15.00 kWh)
Predicted Leftover   : 8,669 Wh (8.67 kWh)
Utilization          : 60.6%
```

### Final Model Accuracy

| Model | RMSE | MAE | R² | Interpretation |
|---|---|---|---|---|
| Energy | 20,683 Wh | 5,469 Wh | 0.9977 | 99.7% of variance explained |
| SoC | 7.95% | 3.91% | 0.8237 | Predicts battery % within ~4% |
| Congestion | 0.50 chargers | 0.15 chargers | 0.8133 | Accurate to within 0.15 chargers |
| Demand | 82,318 Wh | 18,311 Wh | 0.9723 | 97.2% of variance explained |
| Leftover | 6,115 Wh | 1,417 Wh | 0.9578 | 95.8% of variance explained |

---

## 9. REST API

### Overview
The API wraps all 5 CatBoost models into a FastAPI service. It loads models at startup and serves predictions via HTTP POST endpoints.

### How It Was Created

1. **`api/model_loader.py`** — Loads all 5 `.cbm` files into memory at server startup using CatBoost's `load_model()`
2. **`api/features.py`** — Recreates the exact feature engineering logic from each CLI `predict.py` script. Each `build_*_features()` function takes a Pydantic request object and returns a pandas DataFrame matching the model's expected input columns
3. **`api/schemas.py`** — Pydantic request/response models with field validation and sensible defaults
4. **`api/main.py`** — FastAPI app with 5 POST endpoints, CORS middleware, and startup event

### Architecture
```
HTTP Request → FastAPI Endpoint → Build Features → CatBoost Model → JSON Response
     ↓              ↓                  ↓                ↓               ↓
  JSON body    /predict/energy    features.py     model.predict()  EnergyResponse
```

### Starting the API
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Health check |
| `/health` | GET | Service status |
| `/docs` | GET | Interactive Swagger UI |
| `/predict/energy` | POST | Predict energy consumption (Wh) |
| `/predict/soc` | POST | Predict state of charge (%) |
| `/predict/congestion` | POST | Predict active chargers at station |
| `/predict/demand` | POST | Predict total hourly demand (Wh) |
| `/predict/leftover` | POST | Predict leftover energy (Wh) |

### Example API Request (Energy)
```bash
curl -X POST http://localhost:8000/predict/energy \
  -H "Content-Type: application/json" \
  -d '{
    "station_id": "008a2054178b149a52d9893112adf164",
    "charger_id": "T124-IT1-1423-083",
    "target_date": "2026-04-12",
    "hour": 18,
    "avg_pwr_kw": 25
  }'
```

### Example API Response
```json
{
  "predicted_energy_wh": 5851.83,
  "predicted_energy_kwh": 5.85
}
```

---

## 10. API Response Verification

### Comparison Script
A dedicated script compares CLI predictions with API predictions to verify zero accuracy loss:

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 &
python3 compare_cli_vs_api.py
```

### Results (all EXACT matches)
```
  Model                            CLI            API       Diff   % Diff  Status
  ───────────────────────────── ──────────── ──────────── ──────── ─────── ──────
  Energy Consumption (Wh)       5851.8317    5851.8317   0.000000  0.000%  EXACT
  State of Charge (%)             62.2932      62.2932   0.000000  0.000%  EXACT
  Congestion (active)              0.2165       0.2165   0.000000  0.000%  EXACT
  Demand (Wh)                   1342.1403    1342.1403   0.000000  0.000%  EXACT
  Leftover Energy (Wh)          8669.4119    8669.4119   0.000000  0.000%  EXACT
```

The API uses the same `.cbm` model files and identical feature engineering logic as the CLI scripts, producing byte-identical predictions.

---

## 11. Common Bugs & Fixes

### Bug 1: Congestion Model Missing Features (CRITICAL)

**Problem:** The congestion model was trained with 5 lag/rolling features (`lag_1_active`, `lag_2_active`, `lag_1_power_kw`, `roll_3_active`, `roll_6_active`) that were **not present** in the original CLI predict.py script. Running the script would crash:

```
CatBoostError: Feature lag_1_active is present in model but not in pool.
```

**Fix applied to CLI** (`src/03_congestion_model/predict.py`):
- Added input prompts for all 5 lag/rolling features with default values of 0
- Added a warning note when defaults are used, indicating predictions are provisional

**Fix applied to API** (`api/schemas.py`):
- All 5 features exposed as optional fields with `default=0.0`
- API returns a `note` field when default values are used

### Bug 2: Data Leakage in Energy Model

**Problem:** 3 columns in the training data were algebraically derived from the target variable (`max_energy_wh`):
- `estimated_capacity_wh` = `charger_max_energy × 1.1`
- `leftover_energy_wh` = `estimated_capacity - max_energy`
- `power_to_energy_ratio` = `avg_pwr_kw / (max_energy_wh / 1000)`

Including these made R² appear artificially high (the model was "seeing the answer").

**Fix** (`src/01_energy_model/prepare_features.py`):
- Removed all 3 leaked columns before training
- Also dropped `soc_pct` (96.5% null, future model target)

### Bug 3: Aggregate Leakage in Energy Model

**Problem:** Station and charger historical averages were computed on the **full dataset** (including future val/test rows). The model could infer future patterns from these aggregates.

**Fix** (`src/01_energy_model/prepare_features.py`):
- Split data into train/val/test FIRST
- Compute aggregates on TRAIN split only
- Join aggregated values to val/test rows (unseen stations get NaN — CatBoost handles this natively)

### Bug 4: SOC Prediction Target Mismatch

**Problem:** The SoC model was predicting `max_energy_wh` (energy) instead of `soc_pct` (battery percentage).

**Fix** (`src/02_soc_model/`):
- Changed target column to `soc_pct`
- Added `drop_target_nans=True` to filter out the 96.5% null rows
- Updated feature set to match SoC-specific inputs

---

## 12. Usage Guide

### Prerequisites
- Python 3.10+
- pip

### Installation
```bash
git clone https://github.com/punam06/catBoost-prediction-model.git
cd catBoost-prediction-model
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Requirements
All packages are verified to install cleanly and work together:

| Package | Version | Purpose |
|---|---|---|
| `catboost` | >=1.2 | Core ML algorithm (CatBoost Regressor) |
| `pandas` | >=2.0 | Data manipulation and DataFrame operations |
| `numpy` | >=1.24 | Numerical computing and math operations |
| `scikit-learn` | >=1.3 | Metrics (RMSE, MAE, R²) and evaluation |
| `matplotlib` | >=3.7 | Graph generation (scatter plots, histograms) |
| `seaborn` | >=0.13 | Statistical visualization (feature importance) |
| `shap` | >=0.52 | Model explainability (SHAP feature contributions) |
| `fastapi` | >=0.110 | REST API framework |
| `uvicorn[standard]` | >=0.29 | ASGI server to run the FastAPI app |
| `pydantic` | >=2.0 | Request/response validation and serialization |

```bash
# Verify installation
pip list | grep -iE "catboost|pandas|numpy|sklearn|matplotlib|seaborn|shap|fastapi|uvicorn|pydantic"
```

### Option A: Interactive CLI Menu
```bash
python3 master_predictor.py
```
Select a model (1-5) or start the API server (6).

### Option B: Individual CLI Scripts
```bash
# Energy prediction
python3 src/01_energy_model/predict.py

# SoC prediction
python3 src/02_soc_model/predict.py

# Congestion prediction
python3 src/03_congestion_model/predict.py

# Demand prediction
python3 src/04_demand_model/predict.py

# Leftover prediction
python3 src/05_leftover_model/predict.py
```

### Option C: REST API
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
Open http://localhost:8000/docs for the interactive Swagger UI.

### Option D: QA Verification
```bash
python3 qa_full_verification.py
```
Re-runs the entire pipeline: data prep → training → evaluation → accuracy table.

### Retrain a Specific Model
```bash
python3 src/01_energy_model/train_catboost.py
python3 src/02_soc_model/train.py
python3 src/03_congestion_model/train.py
python3 src/04_demand_model/train.py
python3 src/05_leftover_model/train.py
```

---

## 13. Project Structure

```
catBoost-prediction-model/
│
├── master_predictor.py                  # Unified CLI hub (options 1-6)
├── qa_full_verification.py              # Full pipeline QA verification
├── compare_cli_vs_api.py               # Optional: CLI vs API comparison (not in repo)
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
├── model_training_architecture.md       # ML methodology deep-dive
│
├── api/                                 # FastAPI REST API
│   ├── __init__.py
│   ├── main.py                          # FastAPI app + 5 endpoints
│   ├── schemas.py                       # Pydantic request/response models
│   ├── features.py                      # Feature engineering (mirrors CLI scripts)
│   └── model_loader.py                  # Loads all 5 .cbm models at startup
│
├── src/                                 # Source code
│   ├── __init__.py
│   ├── model_training.py                # Shared training pipeline
│   ├── evaluation.py                    # Metrics + graph generation
│   ├── feature_engineering.py           # Reusable feature helpers
│   ├── utils.py                         # Utility functions
│   │
│   ├── 00_data_cleaning/
│   │   └── cleaner.py                   # 8-step data cleaning pipeline
│   │
│   ├── 01_energy_model/
│   │   ├── prepare_features.py          # Feature engineering + leakage fixes
│   │   ├── train_catboost.py            # CatBoost training + evaluation
│   │   ├── train_lightgbm.py            # Legacy LightGBM comparison (not in requirements)
│   │   └── predict.py                   # Interactive CLI prediction
│   │
│   ├── 02_soc_model/
│   │   ├── prepare_features.py          # SoC feature preparation
│   │   ├── train.py                     # SoC model training
│   │   └── predict.py                   # Interactive CLI prediction
│   │
│   ├── 03_congestion_model/
│   │   ├── prepare_features.py          # Station-level aggregation + lag features
│   │   ├── train.py                     # Congestion model training
│   │   └── predict.py                   # Interactive CLI prediction (fixed)
│   │
│   ├── 04_demand_model/
│   │   ├── prepare_features.py          # Demand feature preparation
│   │   ├── train.py                     # Demand model training
│   │   └── predict.py                   # Interactive CLI prediction
│   │
│   └── 05_leftover_model/
│       ├── prepare_features.py          # Leftover feature preparation
│       ├── train.py                     # Leftover model training
│       └── predict.py                   # Interactive CLI prediction
│
├── models/                              # Trained CatBoost binaries
│   ├── catboost_energy_model.cbm
│   ├── catboost_soc_model.cbm
│   ├── catboost_congestion_model.cbm
│   ├── catboost_demand_model.cbm
│   └── catboost_leftover_model.cbm
│
├── data/
│   ├── raw/
│   │   └── All_Merger.csv              # Raw merged telemetry (154,271 rows)
│   └── processed/
│       ├── cleaned_data.csv             # Cleaned dataset (86,615 rows)
│       ├── features_charger.csv         # Charger-level features
│       ├── features_energy_model.csv    # Energy model ready dataset
│       ├── features_soc_model.csv       # SoC model ready dataset
│       ├── features_station_model.csv   # Station-level features
│       ├── features_demand_model.csv    # Demand model ready dataset
│       └── features_leftover_model.csv  # Leftover model ready dataset
│
└── outputs/                             # Auto-generated evaluation artifacts
    ├── energy_model/
    │   ├── catboost_energy_metrics.txt
    │   ├── catboost_energy_actual_vs_predicted.png
    │   ├── catboost_energy_feature_importance.png
    │   └── catboost_energy_residuals.png
    ├── soc_model/
    │   ├── catboost_soc_metrics.txt
    │   ├── catboost_soc_actual_vs_predicted.png
    │   └── catboost_soc_feature_importance.png
    ├── congestion_model/
    │   ├── catboost_congestion_metrics.txt
    │   ├── catboost_congestion_actual_vs_predicted.png
    │   └── catboost_congestion_feature_importance.png
    ├── demand_model/
    │   ├── catboost_demand_metrics.txt
    │   ├── catboost_demand_actual_vs_predicted.png
    │   ├── catboost_demand_feature_importance.png
    │   └── catboost_demand_residuals.png
    └── leftover_model/
        ├── catboost_leftover_metrics.txt
        ├── catboost_leftover_actual_vs_predicted.png
        ├── catboost_leftover_feature_importance.png
        └── catboost_leftover_residuals.png
```

---

## Author

**Punam** — [@punam06](https://github.com/punam06)
