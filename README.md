# AquaTrace AI – Intelligent Water Recovery & Tracking System

A Python prototype that simulates IoT sensors on a data-center cooling system, logs data, runs a Deep Neural Network to predict water recovery, performs DMBI analytics and SPM anomaly detection, and shows a Streamlit dashboard.

## Features

- **IoT Sensor Simulation** – Pre- and post-condensation sensors generating realistic readings
- **Deep Neural Network (DNN)** – MLPRegressor predicting water recovery from sensor inputs
- **Data Logging** – CSV-based logging with pandas DataFrame support
- **DMBI Analytics** – Pattern mining, peak-loss detection, efficiency trends, sustainability reports
- **SPM Monitoring** – Shewhart ±3σ control charts for anomaly detection
- **Streamlit Dashboard** – Real-time visualization of all metrics

## Quick Start

```bash
pip install -r requirements.txt
python main.py --cycles 200
streamlit run src/dashboard/app.py
```

## Project Structure

```
src/sensors/      – IoT sensor simulation
src/models/       – DNN water recovery predictor
src/data/         – CSV data logger
src/analytics/    – DMBI and SPM analytics
src/dashboard/    – Streamlit dashboard
tests/            – pytest test suite
data/             – Generated logs and model artifacts
```

## Running Tests

```bash
python -m pytest tests/ -v
```