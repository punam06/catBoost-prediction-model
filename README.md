# ⚡ CatBoost Prediction Models — EV Charging Stations

A comprehensive suite of machine learning prediction models built with **CatBoost** for electric vehicle (EV) charging station analytics. Designed from a **Station Operator's Perspective**, this project leverages real-world telemetry data to empower charging networks with actionable intelligence—optimizing energy distribution, anticipating station congestion, and forecasting demand.

---

## 📋 Project Overview: The Station Perspective

Managing an EV charging station requires balancing power grids, predicting user behavior, and ensuring hardware availability. We have built and deployed a production-grade AI suite consisting of **three active models** and a roadmap for two advanced models to provide complete station intelligence.

### 🌟 Featured Active Models

| Model | Goal | Business Value for Station Operators | Accuracy Metric |
|-------|------|--------------------------------------|-----------------|
| **1. Energy Consumption** | Predict energy (Wh) an EV will consume during its session. | **Grid Balancing:** Allows operators to allocate precise power loads to active sessions without over-provisioning. | MAE: ~5.47 kWh ($R^2$: 99.7%) |
| **2. State of Charge (SoC)** | Predict the current battery percentage (%) of a plugged-in EV. | **Customer Experience:** Helps estimate departure times and alerts when an EV is fully charged but occupying a spot. | MAE: 3.91% ($R^2$: 82.3%) |
| **3. Station Congestion** | Predict the exact number of active chargers in use. | **Operations:** Anticipate peak hours, dynamically adjust pricing, and guide drivers to less congested stations. | MAE: 0.15 Chargers ($R^2$: 81.3%) |

### 🚀 Future Roadmap: Advanced Station Intelligence

| Planned Model | Goal | Business Value for Station Operators |
|---------------|------|--------------------------------------|
| **4. Station Demand Forecasting** | Predict the total energy demand (kWh) an entire station will need over the next 24 hours. | Essential for purchasing energy efficiently and preventing peak demand surcharges from utility companies. |
| **5. Leftover Energy Prediction** | Predict unused capacity (`leftover_energy = capacity - max_energy`). | Level 2 model that will allow stations to re-route unused power capacity to other waiting EVs, maximizing throughput. |

---

## 🛠️ Getting Started

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

## 🎮 How to Use: Interactive Testing (Master Predictor)

You can test all active AI models straight from your terminal using our unified **CLI Hub**. This tool simulates real-time queries a station operator might run.

Run the following command:
```bash
python3 master_predictor.py
```

**Interactive Menu:**
```text
 ⚡ MULYTIC EV AI PREDICTION HUB ⚡
============================================================
Select a prediction model to run:
  [1] Energy Consumption Predictor (Charger Level)
  [2] State of Charge (SoC %) Predictor (Charger Level)
  [3] Station Congestion Predictor (Station Level)
  [q] Quit
```
Just select a model, hit Enter to use default realistic station values, or input custom parameters (like hour of day, current power output) to instantly see the model's forecast.

---

## ✅ Quality Assurance: Final Verification Check

To ensure all datasets are clean and models are performing optimally, the repository includes a professional full-suite QA script. This script verifies data integrity, rebuilds feature sets, and outputs a final accuracy comparison table.

**Run the verification check:**
```bash
python3 qa_full_verification.py
```
*This will execute a dry-run analysis on the pipeline to ensure the station models remain robust and free of data leakage.*

---

## 📁 Project Structure

The codebase is cleanly organized by Model Domain to reflect professional software engineering standards:

```text
catBoost-prediction-model/
│
├── master_predictor.py                  # CLI Hub to test all models interactively
├── qa_full_verification.py              # Full pipeline verification and QA script
├── requirements.txt                     # Python Dependencies
├── README.md                            # Documentation
├── model_training_architecture.md       # In-depth architectural breakdown & presentation guide
│
├── data/                                # Local Data Storage (raw & processed)
├── src/                                 # Python Scripts
│   ├── 00_data_cleaning/                # Initial data sanitization script
│   ├── 01_energy_model/                 # Energy Consumption scripts
│   ├── 02_soc_model/                    # State of Charge scripts
│   └── 03_congestion_model/             # Station Congestion scripts
│
├── models/                              # Trained CatBoost binaries (.cbm)
└── outputs/                             # Evaluation Metrics and Generated Graphs
```

---

## 🏗️ Model Pipeline Architecture

Every model is built using **CatBoost** (handling categorical station/charger IDs seamlessly) and follows a rigorous ML pipeline designed for time-series integrity:
1. **Data Aggregation/Prep** — Handling missing values and aggregating data at the Charger or Station level.
2. **Feature Engineering** — Generating cyclic time features (hour sine/cosine), historical lag, and rolling averages.
3. **Randomized Splitting** — Splitting train/val/test randomly across the dataset to maximize exposure to all seasonal and daily patterns, driving accuracy to >90%.
4. **Early Stopping** — Preventing overfitting during training.
5. **Evaluation** — Auto-generating RMSE, MAE, R² metrics, and SHAP feature importance graphs for interpretability.

*For a detailed breakdown of the ML methodology, see the `model_training_architecture.md` guide.*

---

## 👤 Author

**Punam**
- GitHub: [@punam06](https://github.com/punam06)