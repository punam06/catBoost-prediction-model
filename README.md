# ⚡ CatBoost Prediction Models — EV Charging Stations

A complete suite of machine learning prediction models built with **CatBoost** for electric vehicle (EV) charging station analytics. This project uses real-world telemetry data to accurately forecast **energy consumption**, **state of charge (SoC)**, and **station congestion**.

---

## 📋 Project Overview

We have built and deployed **3 production-grade predictive models** to help optimize EV charging operations:

| # | Model | Goal | Accuracy Metric |
|---|-------|------|-----------------|
| 1 | **Energy Consumption** | Predict how much energy (Wh) a car will consume during its session. | MAE: ~24.5 kWh ($R^2$: 85%) |
| 2 | **State of Charge (SoC)** | Predict the current battery percentage (%) of a plugged-in EV. | MAE: 9.7% |
| 3 | **Station Congestion** | Predict the exact number of active chargers in use at an entire station. | MAE: 0.29 Chargers ($R^2$: 75%) |

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
   # venv\Scripts\activate         # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎮 Interactive Testing (Master Predictor)

You can easily test all three AI models straight from your terminal! We have built a unified CLI Hub.

Run the following command:
```bash
python3 master_predictor.py
```

This will open an interactive menu:
```text
 ⚡ MULYTIC EV AI PREDICTION HUB ⚡
============================================================
Select a prediction model to run:
  [1] Energy Consumption Predictor (Charger Level)
  [2] State of Charge (SoC %) Predictor (Charger Level)
  [3] Station Congestion Predictor (Station Level)
  [q] Quit
```
Just select a model, hit Enter to use the default realistic testing values, and watch the model predict the future!

---

## 📁 Project Structure

The codebase is cleanly organized by Model Domain:

```text
catBoost-prediction-model/
│
├── master_predictor.py                  # CLI Hub to test all models
├── requirements.txt                     # Dependencies
├── README.md                            # Documentation
│
├── data/                                # Local Data Storage
│   ├── raw/                             # Original uncleaned CSVs
│   └── processed/                       # Cleaned data and engineered features
│
├── src/                                 # Python Scripts
│   ├── 00_data_cleaning/                # Initial data sanitization script
│   ├── 01_energy_model/                 # Energy Consumption scripts
│   ├── 02_soc_model/                    # State of Charge scripts
│   └── 03_congestion_model/             # Station Congestion scripts
│
├── models/                              # Trained CatBoost binaries (.cbm)
│   ├── catboost_energy_model.cbm
│   ├── catboost_soc_model.cbm
│   └── catboost_congestion_model.cbm
│
└── outputs/                             # Evaluation Metrics and Graphs
    ├── energy_model/                    # Feature importance, actual vs predicted
    ├── soc_model/
    └── congestion_model/
```

---

## 🏗️ Model Pipeline

Each of the three models strictly follows the same robust pipeline to prevent data leakage and ensure real-world accuracy:

1. **Data Aggregation/Prep** — Grouping data by Charger or Station, handling missing values.
2. **Feature Engineering** — Generating cyclic time features (hour sine/cosine) and historical aggregates.
3. **Time-Based Splitting** — Splitting train/val/test sequentially through time (never randomly) to simulate real forecasting.
4. **Model Training** — Using `CatBoostRegressor` natively handling categorical IDs without heavy one-hot encoding.
5. **Evaluation** — Generating RMSE, MAE, R² metrics and SHAP/Importance visualizations.

---

## 🛠️ Tech Stack

- **Model**: [CatBoost](https://catboost.ai/) — fast, scalable gradient boosting on decision trees.
- **Language**: Python 3
- **Visualization**: Matplotlib, Seaborn
- **Interpretability**: Feature Importance

---

## 👤 Author

**Punam**
- GitHub: [@punam06](https://github.com/punam06)