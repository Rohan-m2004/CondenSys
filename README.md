# CondenSys – Intelligent Water Recovery & Tracking System

A Python prototype that simulates IoT sensors on a data-center cooling system, logs data, runs a Deep Neural Network to predict water recovery, performs DMBI analytics and SPM anomaly detection, and shows a Streamlit dashboard.

## Features

- **IoT Sensor Simulation** – Pre- and post-condensation sensors generating realistic readings
- **Deep Neural Network (DNN)** – MLPRegressor predicting water recovery from sensor inputs
- **Data Logging** – CSV-based logging with pandas DataFrame support
- **DMBI Analytics** – Pattern mining, peak-loss detection, efficiency trends, sustainability reports
- **SPM Monitoring** – Shewhart ±3σ control charts for anomaly detection
- **Streamlit Dashboard** – Real-time visualization of all metrics

---

## 🚀 How to Run (Show to Teacher)

### Step 1 – Install dependencies (one-time setup)

```bash
pip install -r requirements.txt
```

This installs: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `streamlit`, `joblib`.

---

### Step 2 – Run the simulation

```bash
python main.py
```

This will:
1. Simulate **200 cycles** of sensor readings (pre/post condensation)
2. Train the **DNN model** (R² ≈ 0.93)
3. Log all data to `data/condensys_log.csv`
4. Print the **DMBI Sustainability Report** + **SPM anomaly status**

Expected output:
```
============================================================
  CondenSys – Intelligent Water Recovery System
============================================================
  Simulation cycles : 200
  DNN training      : Enabled
============================================================

[1/4] Collecting sensor data...
[2/4] Training DNN model...
      RMSE=0.25  R²=0.93
[3/4] Logging data and predictions...
[4/4] Running DMBI analytics and SPM monitoring...

  CondenSys – Sustainability Report
  ...
  SPM Status:
    Anomalies detected : 3
    System stable      : False
```

---

### Step 3 – Open the live dashboard

In a **second terminal** (or after Step 2), run:

```bash
streamlit run src/dashboard/app.py
```

Then open your browser at: **http://localhost:8501**

The dashboard shows:
- 📊 KPI cards (total readings, avg water collected, efficiency, total recovery)
- 🌡️ Pre-condensation sensor time series (temperature + humidity)
- 💧 Actual vs Predicted water collection chart
- 📈 DMBI efficiency trend + peak loss conditions
- ⚙️ SPM anomaly detection status and alerts
- 📋 Scrollable raw data table with CSV download

---

### Optional flags for `main.py`

| Flag | Description | Example |
|------|-------------|---------|
| `--cycles N` | Run N simulation cycles (default: 200) | `python main.py --cycles 500` |
| `--no-train` | Skip DNN training, only generate data | `python main.py --no-train` |
| `--seed N` | Set random seed for reproducibility | `python main.py --seed 7` |

---

## Project Structure

```
src/sensors/      – IoT sensor simulation (PreCondensationSensor, PostCondensationSensor)
src/models/       – DNN water recovery predictor (WaterRecoveryDNN)
src/data/         – CSV data logger (DataLogger)
src/analytics/    – DMBI pattern analytics and SPM anomaly monitoring
src/dashboard/    – Streamlit live dashboard
tests/            – 21 pytest tests (all passing)
data/             – Generated logs (condensys_log.csv) and model artifacts
```

## Running Tests

```bash
python -m pytest tests/ -v
```
