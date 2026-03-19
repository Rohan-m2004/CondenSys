"""
SPM – Statistical Process Monitoring for AquaTrace AI.

Implements Shewhart control charts (X-bar ± 3σ) to detect
anomalies in sensor readings and water recovery metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class ControlLimits:
    metric: str
    mean: float
    ucl: float   # upper control limit
    lcl: float   # lower control limit
    sigma: float


@dataclass
class AnomalyAlert:
    index: int
    metric: str
    value: float
    ucl: float
    lcl: float
    alert_type: str   # "HIGH" | "LOW"
    message: str


class SPMMonitor:
    """
    Statistical Process Monitor using Shewhart ±3σ control charts.

    Usage
    -----
    monitor = SPMMonitor(df)
    monitor.compute_control_limits()
    alerts = monitor.detect_anomalies()
    """

    MONITORED_METRICS = [
        "pre_temperature",
        "pre_humidity",
        "post_water_collected",
        "efficiency_score",
    ]

    def __init__(self, df: pd.DataFrame, sigma_multiplier: float = 3.0):
        self.df = df.copy()
        self.sigma_multiplier = sigma_multiplier
        self._limits: dict[str, ControlLimits] = {}

    def compute_control_limits(self) -> dict[str, ControlLimits]:
        for metric in self.MONITORED_METRICS:
            if metric not in self.df.columns or self.df[metric].dropna().empty:
                continue
            values = self.df[metric].dropna().values.astype(float)
            mean = float(np.mean(values))
            sigma = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            self._limits[metric] = ControlLimits(
                metric=metric,
                mean=round(mean, 4),
                ucl=round(mean + self.sigma_multiplier * sigma, 4),
                lcl=round(mean - self.sigma_multiplier * sigma, 4),
                sigma=round(sigma, 4),
            )
        return self._limits

    def detect_anomalies(self) -> list[AnomalyAlert]:
        if not self._limits:
            self.compute_control_limits()

        alerts: list[AnomalyAlert] = []
        for metric, limits in self._limits.items():
            if metric not in self.df.columns:
                continue
            for idx, val in self.df[metric].items():
                if pd.isna(val):
                    continue
                fval = float(val)
                if fval > limits.ucl:
                    alerts.append(AnomalyAlert(
                        index=int(idx),
                        metric=metric,
                        value=round(fval, 4),
                        ucl=limits.ucl,
                        lcl=limits.lcl,
                        alert_type="HIGH",
                        message=f"[HIGH] {metric}={fval:.3f} exceeds UCL={limits.ucl:.3f}",
                    ))
                elif fval < limits.lcl:
                    alerts.append(AnomalyAlert(
                        index=int(idx),
                        metric=metric,
                        value=round(fval, 4),
                        ucl=limits.ucl,
                        lcl=limits.lcl,
                        alert_type="LOW",
                        message=f"[LOW]  {metric}={fval:.3f} below LCL={limits.lcl:.3f}",
                    ))
        return alerts

    def status(self) -> dict:
        alerts = self.detect_anomalies()
        return {
            "total_readings": len(self.df),
            "anomaly_count": len(alerts),
            "system_stable": len(alerts) == 0,
            "alerts": [a.message for a in alerts[:10]],  # top 10
        }
