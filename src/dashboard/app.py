"""
CondenSys – Streamlit Dashboard
Run with: streamlit run src/dashboard/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.data.data_logger import DataLogger
from src.analytics.dmbi import DMBIAnalytics
from src.analytics.spm import SPMMonitor

st.set_page_config(
    page_title="CondenSys",
    page_icon="💧",
    layout="wide",
)

css_path = Path(__file__).with_name("style.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

st.title("💧 CondenSys – Intelligent Water Recovery & Tracking System")
st.caption("Real-time monitoring, AI predictions, and sustainability analytics for data-center cooling water recovery.")

logger = DataLogger()
df = logger.load()

if df.empty:
    st.warning("No data logged yet. Run `python main.py` to generate simulation data.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Readings", len(df))
col2.metric("Avg Water Collected (L)", f"{df['post_water_collected'].mean():.2f}")
col3.metric("Avg Efficiency Score", f"{df['efficiency_score'].mean():.2%}")
col4.metric("Total Water Recovered (L)", f"{df['post_water_collected'].sum():.1f}")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("🌡️ Pre-Condensation Sensor Readings")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df.index, df["pre_temperature"], label="Temperature (°C)", color="tomato")
    ax.plot(df.index, df["pre_humidity"], label="Humidity (%)", color="steelblue")
    ax.set_xlabel("Reading #")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("💧 Water Collection: Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df.index, df["post_water_collected"], label="Actual (L)", color="dodgerblue")
    ax.plot(df.index, df["predicted_water_recovery"], label="Predicted (L)", color="orange", linestyle="--")
    ax.set_xlabel("Reading #")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

st.divider()

st.subheader("📊 DMBI – Data Mining & Business Intelligence")
dmbi = DMBIAnalytics(df)

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Efficiency Trend (rolling average)**")
    trend = dmbi.efficiency_trend(window=max(5, len(df) // 10))
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(trend.index, trend["efficiency_score"], alpha=0.3, color="grey", label="Raw")
    ax.plot(trend.index, trend["rolling_efficiency"], color="green", label="Rolling avg")
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Reading #")
    ax.set_ylabel("Efficiency Score")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

with col_b:
    st.markdown("**Peak Loss Conditions**")
    peaks = dmbi.peak_loss_conditions()
    if "insight" in peaks:
        st.info(peaks["insight"])
        st.json({k: v for k, v in peaks.items() if k != "insight"})
    else:
        st.write(peaks.get("message", "Analysing..."))

st.markdown("**Sustainability Report**")
with st.expander("View full report"):
    st.code(dmbi.generate_report())

st.divider()

st.subheader("⚙️ SPM – Statistical Process Monitoring")
monitor = SPMMonitor(df)
status = monitor.status()

c1, c2, c3 = st.columns(3)
c1.metric("Total Readings", status["total_readings"])
c2.metric("Anomalies Detected", status["anomaly_count"])
c3.metric("System Stable", "✅ Yes" if status["system_stable"] else "⚠️ No")

if status["alerts"]:
    st.warning("Recent Anomaly Alerts:")
    for alert in status["alerts"]:
        st.text(alert)
else:
    st.success("No anomalies detected. System is within control limits.")

st.divider()

st.subheader("📋 Recent Logged Data")
st.dataframe(df.tail(50).sort_values("timestamp", ascending=False), use_container_width=True)
