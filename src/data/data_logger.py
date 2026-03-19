"""
DataLogger – appends sensor readings and model predictions to a CSV file
and provides utilities to reload data as a pandas DataFrame.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import pandas as pd

_DEFAULT_LOG_PATH = Path(__file__).parent.parent.parent / "data" / "aquatrace_log.csv"

_COLUMNS = [
    "timestamp",
    "pre_temperature",
    "pre_humidity",
    "post_water_collected",
    "post_flow_rate",
    "predicted_water_recovery",
    "efficiency_score",
    "recommendation",
]


class DataLogger:
    def __init__(self, log_path: Path | None = None):
        self.log_path = Path(log_path or _DEFAULT_LOG_PATH)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    def _ensure_header(self) -> None:
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            with open(self.log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_COLUMNS)
                writer.writeheader()

    def log(
        self,
        pre_temperature: float,
        pre_humidity: float,
        post_water_collected: float,
        post_flow_rate: float,
        predicted_water_recovery: float,
        efficiency_score: float,
        recommendation: str,
        timestamp: datetime | None = None,
    ) -> None:
        row = {
            "timestamp": (timestamp or datetime.utcnow()).isoformat(),
            "pre_temperature": round(pre_temperature, 3),
            "pre_humidity": round(pre_humidity, 3),
            "post_water_collected": round(post_water_collected, 3),
            "post_flow_rate": round(post_flow_rate, 3),
            "predicted_water_recovery": round(predicted_water_recovery, 3),
            "efficiency_score": round(efficiency_score, 4),
            "recommendation": recommendation,
        }
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_COLUMNS)
            writer.writerow(row)

    def load(self) -> pd.DataFrame:
        if not self.log_path.exists():
            return pd.DataFrame(columns=_COLUMNS)
        df = pd.read_csv(self.log_path, parse_dates=["timestamp"])
        return df

    def clear(self) -> None:
        with open(self.log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_COLUMNS)
            writer.writeheader()
