"""Wai — Streamlit dashboard for coastal water-level visualization.

Run:
    streamlit run app.py

Features
--------
- Station selector and date-range filter
- Interactive water-level time series with high-water event markers
- Forecast vs actual overlay for the held-out test period
- Model performance metrics table
- Station location map
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Wai",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

METRICS_PATH = Path("reports/model_metrics.json")
DEMO_DATA_PATH = Path("data/demo/demo_water_levels.csv")


@st.cache_data
def load_data() -> pd.DataFrame:
    from src.data.loader import load_demo_data
    return load_demo_data()


@st.cache_data
def load_metrics() -> dict:
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}


@st.cache_data
def run_forecast(station_id: str, train_frac: float = 0.75):
    from src.models.baseline import HarmonicRidgeModel, PersistenceModel
    df = load_data()
    sub = df[df["station_id"] == station_id].sort_values("timestamp").reset_index(drop=True)
    n_train = int(len(sub) * train_frac)
    train, test = sub.iloc[:n_train], sub.iloc[n_train:]

    persist = PersistenceModel().fit(train["water_level"])
    persist_pred = persist.predict(len(test))

    harmonic = HarmonicRidgeModel(alpha=1.0).fit(train)
    harmonic_pred = harmonic.predict_on(test)

    return train, test, persist_pred, harmonic_pred


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌊 Wai")
    st.caption("Coastal Water-Level Forecasting · Demo")
    st.divider()

    if not DEMO_DATA_PATH.exists():
        st.error(
            "Demo data not found. Run:\n\n"
            "`python -m scripts.prepare_demo_data`"
        )
        st.stop()

    df_all = load_data()
    stations = sorted(df_all["station_id"].unique())
    station_id = st.selectbox("Station", stations)

    sub_all = df_all[df_all["station_id"] == station_id].sort_values("timestamp")
    date_min = sub_all["timestamp"].dt.date.min()
    date_max = sub_all["timestamp"].dt.date.max()

    date_range = st.date_input(
        "Date range",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )
    if len(date_range) == 2:
        start_dt = pd.Timestamp(date_range[0], tz="UTC")
        end_dt = pd.Timestamp(date_range[1], tz="UTC") + pd.Timedelta(days=1)
        sub = sub_all[(sub_all["timestamp"] >= start_dt) & (sub_all["timestamp"] < end_dt)]
    else:
        sub = sub_all.copy()

    high_water_pct = st.slider(
        "High-water threshold (std devs above mean)", 1.0, 4.0, 2.0, 0.25
    )
    show_forecast = st.checkbox("Show forecast overlay", value=True)
    st.divider()
    st.caption("Data: DEMO_SYNTHETIC only. No real sensor data.")

# ── Main ─────────────────────────────────────────────────────────────────────
st.header(f"Station: {station_id}")

col1, col2, col3, col4 = st.columns(4)
wl = sub["water_level"].dropna()
col1.metric("Observations", f"{len(sub):,}")
col2.metric("Mean water level", f"{wl.mean():.3f} m")
col3.metric("Max water level", f"{wl.max():.3f} m")
col4.metric("Min water level", f"{wl.min():.3f} m")

try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

threshold = float(wl.mean() + high_water_pct * wl.std())
anomalies = sub[sub["water_level"] >= threshold]

# ── Time-series plot ─────────────────────────────────────────────────────────
st.subheader("Water-Level Time Series")

if _HAS_PLOTLY:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub["timestamp"], y=sub["water_level"],
        name="Observed", line=dict(color="#1f77b4", width=1),
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.3f} m<extra></extra>",
    ))
    fig.add_hline(
        y=threshold, line_dash="dash", line_color="orange",
        annotation_text=f"High-water threshold ({threshold:.2f} m)",
        annotation_position="top right",
    )
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies["timestamp"], y=anomalies["water_level"],
            mode="markers", name=f"High-water ({len(anomalies)})",
            marker=dict(color="red", size=5, symbol="circle"),
        ))
    fig.update_layout(
        xaxis_title="Time (UTC)", yaxis_title="Water Level (m)",
        legend=dict(orientation="h", y=1.02, x=0),
        height=380, margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.line_chart(sub.set_index("timestamp")["water_level"])
    st.info("Install plotly for richer charts: `pip install plotly`")

# ── Forecast overlay ─────────────────────────────────────────────────────────
if show_forecast:
    st.subheader("Forecast vs Actual (held-out test set)")
    with st.spinner("Running forecast …"):
        try:
            train, test, persist_pred, harmonic_pred = run_forecast(station_id)
        except Exception as e:
            st.error(f"Forecast failed: {e}")
            train, test = None, None

    if test is not None and _HAS_PLOTLY:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=test["timestamp"], y=test["water_level"],
            name="Actual", line=dict(color="#1f77b4", width=1.5),
        ))
        aligned_test = test.iloc[-len(harmonic_pred):]
        fig2.add_trace(go.Scatter(
            x=aligned_test["timestamp"], y=harmonic_pred,
            name="HarmonicRidge forecast", line=dict(color="#2ca02c", width=1.5, dash="dash"),
        ))
        fig2.add_trace(go.Scatter(
            x=test["timestamp"], y=persist_pred,
            name="Persistence baseline", line=dict(color="#d62728", width=1, dash="dot"),
        ))
        fig2.update_layout(
            xaxis_title="Time (UTC)", yaxis_title="Water Level (m)",
            height=360, margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)
    elif test is not None:
        st.line_chart(test.set_index("timestamp")["water_level"])

# ── Metrics table ─────────────────────────────────────────────────────────────
st.subheader("Model Performance Metrics (synthetic demo data)")
all_metrics = load_metrics()
station_metrics = all_metrics.get(station_id, {})

if station_metrics:
    rows = []
    for model_name, m in station_metrics.items():
        if not isinstance(m, dict) or "mae" not in m:
            continue
        rows.append({
            "Model": model_name,
            "MAE (m)": f"{m.get('mae', float('nan')):.4f}",
            "RMSE (m)": f"{m.get('rmse', float('nan')):.4f}",
            "R²": f"{m.get('r2', float('nan')):.4f}",
            "NSE": f"{m.get('nse', float('nan')):.4f}",
            "Corr": f"{m.get('corr', float('nan')):.4f}",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info(
        "No metrics found. Run `python -m scripts.train_baseline` then refresh."
    )

# ── High-water events table ───────────────────────────────────────────────────
st.subheader(f"High-Water Events (≥ {threshold:.2f} m)")
if anomalies.empty:
    st.success("No high-water events in the selected date range.")
else:
    display = anomalies[["timestamp", "water_level"]].copy()
    display["timestamp"] = display["timestamp"].dt.strftime("%Y-%m-%d %H:%M UTC")
    display.columns = ["Timestamp (UTC)", "Water Level (m)"]
    st.dataframe(display.head(50), use_container_width=True, hide_index=True)

# ── Station map ───────────────────────────────────────────────────────────────
st.subheader("Station Locations")
station_coords = (
    df_all.groupby("station_id")[["lat", "lon"]].first().reset_index()
)
station_coords.columns = ["Station", "lat", "lon"]
st.map(station_coords.rename(columns={"lat": "latitude", "lon": "longitude"}))

st.caption(
    "Wai · Demo data (DEMO_SYNTHETIC) · "
    "No private or proprietary data is used."
)
