import pytest
import numpy as np
import pandas as pd
from src.analytics.dmbi import DMBIAnalytics
from src.analytics.spm import SPMMonitor


def make_df(n=100, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="5min"),
        "pre_temperature": rng.normal(45, 5, n),
        "pre_humidity": rng.uniform(60, 95, n),
        "post_water_collected": rng.uniform(3, 9, n),
        "post_flow_rate": rng.uniform(0.6, 1.8, n),
        "predicted_water_recovery": rng.uniform(3.5, 8.5, n),
        "efficiency_score": rng.uniform(0.3, 0.9, n),
        "recommendation": ["OK"] * n,
    })


def test_summary_stats_keys(tmp_path):
    df = make_df()
    dmbi = DMBIAnalytics(df)
    stats = dmbi.summary_stats()
    assert "pre_temperature" in stats
    assert "efficiency_score" in stats


def test_peak_loss_conditions(tmp_path):
    df = make_df()
    dmbi = DMBIAnalytics(df)
    result = dmbi.peak_loss_conditions()
    assert "avg_temperature_at_peak_loss" in result
    assert "insight" in result


def test_efficiency_trend_length(tmp_path):
    df = make_df(50)
    dmbi = DMBIAnalytics(df)
    trend = dmbi.efficiency_trend(window=5)
    assert len(trend) == 50


def test_generate_report_contains_header(tmp_path):
    df = make_df()
    dmbi = DMBIAnalytics(df)
    report = dmbi.generate_report()
    assert "CondenSys" in report


def test_empty_df_summary(tmp_path):
    dmbi = DMBIAnalytics(pd.DataFrame())
    assert dmbi.summary_stats() == {}


def test_spm_control_limits(tmp_path):
    df = make_df()
    monitor = SPMMonitor(df)
    limits = monitor.compute_control_limits()
    assert "pre_temperature" in limits
    assert limits["pre_temperature"].ucl > limits["pre_temperature"].lcl


def test_spm_detects_anomalies(tmp_path):
    df = make_df()
    df.at[0, "pre_temperature"] = 999.0
    df.at[1, "efficiency_score"] = -50.0
    monitor = SPMMonitor(df)
    alerts = monitor.detect_anomalies()
    assert len(alerts) >= 2


def test_spm_status_structure(tmp_path):
    df = make_df()
    monitor = SPMMonitor(df)
    status = monitor.status()
    assert "total_readings" in status
    assert "anomaly_count" in status
    assert "system_stable" in status
