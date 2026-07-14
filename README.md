# ⚡ CatBoost Prediction Models — EV Charging Stations

A suite of machine learning prediction models built with **CatBoost** for electric vehicle (EV) charging station analytics. The project uses real-world charging station telemetry data to forecast energy consumption, demand patterns, charging states, and station load.

---

## 📋 Project Overview

This project develops **5 predictive models** using CatBoost gradient boosting, each targeting a different aspect of EV charging station operations:

| # | Model | Target | Status |
|---|-------|--------|--------|
| 1 | **Energy Consumption Prediction** | `max_energy_wh` | 🔨 In Progress |
| 2 | **Demand Forecasting** | Active session count / hourly demand | 📋 Planned |
| 3 | **Charging State (SoC) Prediction** | `soc_pct` | 📋 Planned |
| 4 | **Leftover Energy Estimation** | Remaining capacity | 📋 Planned |
| 5 | **Congestion / Load Forecasting** | Aggregated station load | 📋 Planned |

---

## 📊 Dataset

**File**: `merged.csv`

| Property | Value |
|----------|-------|
| Rows | 90,507 |
| Columns | 13 |
| Unique Stations | 1,284 |
| Unique Chargers | 7,600 |
| Time Range | Nov 2025 — Apr 2026 |
| Granularity | 15-minute intervals |

### Column Descriptions

| Column | Type | Description |
|--------|------|-------------|
| `charging_station_id` | string | Unique station identifier |
| `charger_id` | string | Unique charger identifier |
| `timestamp` | datetime | Observation timestamp (UTC) |
| `avg_pwr_kw` | float | Average power draw (kW) |
| `max_energy_wh` | float | Maximum energy consumed (Wh) |
| `active_session_count` | int | Number of active charging sessions |
| `soc_pct` | float | State of charge (%), ~96% missing |
| `hour` | int | Hour of day (0–23) |
| `day_of_week` | int | Day of week (0=Mon, 6=Sun) |
| `month` | int | Month of year |
| `hour_sin` | float | Cyclical hour encoding (sine) |
| `hour_cos` | float | Cyclical hour encoding (cosine) |
| `is_weekend` | int | Weekend flag (0/1) |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/punam06/catBoost-prediction-model.git
   cd catBoost-prediction-model
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # macOS / Linux
   # venv\Scripts\activate          # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| catboost | ≥ 1.2 | Gradient boosting model |
| pandas | ≥ 2.0 | Data manipulation |
| numpy | ≥ 1.24 | Numerical computing |
| scikit-learn | ≥ 1.3 | Metrics & splitting |
| matplotlib | ≥ 3.7 | Visualization |
| seaborn | ≥ 0.13 | Statistical plots |
| shap | ≥ 0.52 | Model interpretability |

---

## 📁 Project Structure

```
catBoost-prediction-model/
│
├── merged.csv                           # Raw dataset
├── requirements.txt                     # Python dependencies
├── README.md                            # Project documentation
├── energy_consumption_prediction.py     # Model 1 — main pipeline
│
├── src/                                 # Source modules
│   ├── __init__.py
│   ├── data_preprocessing.py            # Data cleaning & loading
│   ├── feature_engineering.py           # Feature creation & transformation
│   ├── model_training.py               # CatBoost training & configuration
│   ├── evaluation.py                   # Metrics, plots & interpretation
│   └── utils.py                        # Helper functions & constants
│
├── models/                             # Saved CatBoost models (.cbm)
│
└── outputs/                            # Generated plots & metrics
    ├── feature_importance.png
    ├── actual_vs_predicted.png
    ├── residuals.png
    └── evaluation_metrics.txt
```

---

## 🏗️ Model Pipeline

Each prediction model follows the same structured pipeline:

```
Data Loading → Cleaning → Feature Engineering → Train/Test Split → CatBoost Training → Evaluation → Export
```

1. **Data Preprocessing** — Handle missing values, remove negatives/outliers, parse timestamps
2. **Feature Engineering** — Lag features, rolling averages, peak hour flags, station-level aggregations
3. **Train/Test Split** — Time-based split to prevent data leakage
4. **Model Training** — CatBoost with native categorical feature handling
5. **Evaluation** — RMSE, MAE, R², MAPE + visualizations
6. **Interpretation** — Feature importance & SHAP analysis

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Squared Error — penalizes large errors |
| **MAE** | Mean Absolute Error — average error magnitude |
| **R²** | Coefficient of determination — variance explained |
| **MAPE** | Mean Absolute Percentage Error |

---

## 🛠️ Tech Stack

- **Model**: [CatBoost](https://catboost.ai/) — gradient boosting on decision trees
- **Language**: Python 3
- **Visualization**: Matplotlib, Seaborn
- **Interpretability**: SHAP

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Punam**

- GitHub: [@punam06](https://github.com/punam06)