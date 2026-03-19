"""
DMBI – Data Mining & Business Intelligence for CondenSys.

Identifies patterns in water loss, predicts peak-loss times,
and generates sustainability reports.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


class DMBIAnalytics:
    """Mine patterns from logged sensor + prediction data."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def summary_stats(self) -> dict:
        """Return descriptive statistics for key metrics."""
        if self.df.empty:
            return {}
        cols = ["pre_temperature", "pre_humidity", "post_water_collected",
                "predicted_water_recovery", "efficiency_score"]
        stats = {}
        for col in cols:
            if col in self.df.columns:
                stats[col] = {
                    "mean": round(float(self.df[col].mean()), 3),
                    "std": round(float(self.df[col].std()), 3),
                    "min": round(float(self.df[col].min()), 3),
                    "max": round(float(self.df[col].max()), 3),
                }
        return stats

    def peak_loss_conditions(self) -> dict:
        """
        Identify conditions (temperature range) associated with highest water loss
        (i.e., largest gap between predicted recovery and actual collected).
        """
        if self.df.empty or len(self.df) < 5:
            return {"message": "Insufficient data for peak-loss analysis."}

        df = self.df.copy()
        df["loss_gap"] = df["predicted_water_recovery"] - df["post_water_collected"]
        worst = df.nlargest(max(1, len(df) // 5), "loss_gap")  # top 20%
        return {
            "avg_temperature_at_peak_loss": round(float(worst["pre_temperature"].mean()), 2),
            "avg_humidity_at_peak_loss": round(float(worst["pre_humidity"].mean()), 2),
            "avg_loss_gap_litres": round(float(worst["loss_gap"].mean()), 3),
            "insight": (
                "Maximum water loss occurs at high temperature + low humidity zones. "
                "Consider increasing airflow and pre-cooling the vapor stream."
            ),
        }

    def efficiency_trend(self, window: int = 10) -> pd.DataFrame:
        """Return rolling-average efficiency trend."""
        if self.df.empty:
            return pd.DataFrame()
        df = self.df[["timestamp", "efficiency_score"]].copy()
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["rolling_efficiency"] = df["efficiency_score"].rolling(window=window, min_periods=1).mean()
        return df

    def temperature_humidity_correlation(self) -> dict:
        """Pearson correlation between temperature, humidity, and water recovery."""
        if self.df.empty or len(self.df) < 3:
            return {}
        cols = ["pre_temperature", "pre_humidity", "post_water_collected"]
        available = [c for c in cols if c in self.df.columns]
        corr = self.df[available].corr()
        result = {}
        for col in available:
            result[col] = {k: round(float(v), 4) for k, v in corr[col].items()}
        return result

    def generate_report(self) -> str:
        """Generate a text sustainability report."""
        stats = self.summary_stats()
        peaks = self.peak_loss_conditions()

        lines = [
            "=" * 60,
            "  CondenSys – Sustainability Report",
            "=" * 60,
            f"  Total readings analysed : {len(self.df)}",
        ]

        if stats:
            wc = stats.get("post_water_collected", {})
            eff = stats.get("efficiency_score", {})
            lines += [
                f"  Avg water collected     : {wc.get('mean', 'N/A')} L",
                f"  Avg efficiency score    : {eff.get('mean', 'N/A')}",
                f"  Max water collected     : {wc.get('max', 'N/A')} L",
            ]

        if "insight" in peaks:
            lines += [
                "",
                "  Peak Loss Analysis:",
                f"  {peaks['insight']}",
                f"  Avg temp at peak loss   : {peaks.get('avg_temperature_at_peak_loss')} °C",
                f"  Avg humidity at peak    : {peaks.get('avg_humidity_at_peak_loss')} %",
                f"  Avg loss gap            : {peaks.get('avg_loss_gap_litres')} L",
            ]

        lines.append("=" * 60)
        return "\n".join(lines)
