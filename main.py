#!/usr/bin/env python3
"""
AquaTrace AI – main simulation runner.

Generates N cycles of simulated sensor readings, trains the DNN,
logs predictions, runs analytics, and prints a summary report.

Usage:
    python main.py            # 200 simulation cycles (default)
    python main.py --cycles 500
    python main.py --no-train  # skip DNN training (just generate data)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.sensors.sensor_simulator import PreCondensationSensor, PostCondensationSensor
from src.models.dnn_model import WaterRecoveryDNN
from src.data.data_logger import DataLogger
from src.analytics.dmbi import DMBIAnalytics
from src.analytics.spm import SPMMonitor


def run_simulation(n_cycles: int = 200, train_model: bool = True, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    pre_sensor = PreCondensationSensor(rng=rng)
    post_sensor = PostCondensationSensor(rng=rng)
    logger = DataLogger()
    model = WaterRecoveryDNN()

    print(f"\n{'='*60}")
    print("  AquaTrace AI – Intelligent Water Recovery System")
    print(f"{'='*60}")
    print(f"  Simulation cycles : {n_cycles}")
    print(f"  DNN training      : {'Enabled' if train_model else 'Disabled'}")
    print(f"{'='*60}\n")

    print("[1/4] Collecting sensor data...")
    pre_readings = []
    post_readings = []
    for _ in range(n_cycles):
        pre = pre_sensor.read()
        post = post_sensor.read(pre)
        pre_readings.append(pre)
        post_readings.append(post)

    if train_model:
        print("[2/4] Training DNN model...")
        airflow = rng.uniform(1.0, 4.0, size=n_cycles)
        X = np.array([[r.temperature, r.humidity, a]
                       for r, a in zip(pre_readings, airflow)])
        y = np.array([r.water_collected for r in post_readings])
        metrics = model.train(X, y)
        print(f"      RMSE={metrics['rmse']:.4f}  R²={metrics['r2']:.4f}")
    else:
        print("[2/4] Skipping DNN training.")

    print("[3/4] Logging data and predictions...")
    logger.clear()  # fresh run
    for i, (pre, post) in enumerate(zip(pre_readings, post_readings)):
        airflow_val = float(rng.uniform(1.0, 4.0))
        if train_model:
            pred = model.predict(pre.temperature, pre.humidity, airflow_val)
        else:
            pred = {
                "predicted_water_recovery": post.water_collected,
                "efficiency_score": 0.5,
                "recommendation": "Model not trained.",
            }
        logger.log(
            pre_temperature=pre.temperature,
            pre_humidity=pre.humidity,
            post_water_collected=post.water_collected,
            post_flow_rate=post.flow_rate,
            predicted_water_recovery=pred["predicted_water_recovery"],
            efficiency_score=pred["efficiency_score"],
            recommendation=pred["recommendation"],
            timestamp=pre.timestamp,
        )

    print("[4/4] Running DMBI analytics and SPM monitoring...\n")
    df = logger.load()
    dmbi = DMBIAnalytics(df)
    print(dmbi.generate_report())

    monitor = SPMMonitor(df)
    status = monitor.status()
    print(f"\n  SPM Status:")
    print(f"    Anomalies detected : {status['anomaly_count']}")
    print(f"    System stable      : {status['system_stable']}")
    if status["alerts"]:
        print("    Recent alerts:")
        for alert in status["alerts"][:5]:
            print(f"      {alert}")

    print(f"\n  Data saved to: {logger.log_path}")
    print("  Run the dashboard: streamlit run src/dashboard/app.py\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="AquaTrace AI – Water Recovery Simulation")
    parser.add_argument("--cycles", type=int, default=200, help="Number of simulation cycles")
    parser.add_argument("--no-train", action="store_true", help="Skip DNN training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    run_simulation(n_cycles=args.cycles, train_model=not args.no_train, seed=args.seed)


if __name__ == "__main__":
    main()
