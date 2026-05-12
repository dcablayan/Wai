"""Wai — Streamlit dashboard for coastal water-level visualization.

Run:
    streamlit run app.py

Tabs
----
  Overview          — station summary stats and location map
  Forecasts         — time series with forecast overlay and conformal intervals
  Model Comparison  — metrics table for all pipeline models
  Alerts            — configurable high-water alert detection
  Uncertainty       — conformal prediction interval details
  Benchmark Results — prototype model RMSE on tidecast data
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
HORIZON_PATH = Path("reports/horizon_metrics.json")
BENCHMARK_PATH = Path("reports/benchmark_results.md")
DEMO_DATA_PATH = Path("data/demo/demo_water_levels.csv")

try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False


# ── Cached data loaders ───────────────────────────────────────────────────────

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
def load_horizon_metrics() -> dict:
    if HORIZON_PATH.exists():
        with open(HORIZON_PATH) as f:
            return json.load(f)
    return {}


@st.cache_data
def run_forecast(station_id: str, train_frac: float = 0.75):
    from src.models.baseline import HarmonicRidgeModel, PersistenceModel
    from src.models.gradient_boost import GradBoostModel
    from src.models.conformal import ConformalIntervals

    df = load_data()
    sub = df[df["station_id"] == station_id].sort_values("timestamp").reset_index(drop=True)
    n = len(sub)
    n_train = int(n * train_frac)
    n_cal = int(n_train * 0.15)  # last 15% of train for conformal calibration

    train_fit = sub.iloc[:n_train - n_cal]
    train_cal = sub.iloc[n_train - n_cal:n_train]
    test = sub.iloc[n_train:]

    persist = PersistenceModel().fit(sub.iloc[:n_train]["water_level"])
    persist_pred = persist.predict(len(test))

    harmonic = HarmonicRidgeModel(alpha=1.0).fit(train_fit)
    harmonic_pred = harmonic.predict_on(test)
    harmonic_cal_pred = harmonic.predict_on(train_cal)

    gradboost = GradBoostModel().fit(train_fit)
    gradboost_pred = gradboost.predict_on(test)
    gradboost_cal_pred = gradboost.predict_on(train_cal)

    # Conformal calibration — align to valid rows
    from src.features.engineering import build_feature_matrix
    _, y_cal_h = build_feature_matrix(train_cal)
    harmonic_ci = ConformalIntervals(coverage=0.90)
    harmonic_ci.calibrate(y_cal_h.values, harmonic_cal_pred[-len(y_cal_h):])

    gb_ci = ConformalIntervals(coverage=0.90)
    gb_ci.calibrate(y_cal_h.values, gradboost_cal_pred[-len(y_cal_h):])

    return train_fit, test, persist_pred, harmonic_pred, gradboost_pred, harmonic_ci, gb_ci


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

    st.divider()
    st.caption("Data: DEMO_SYNTHETIC only. No real sensor data.")


# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_overview, tab_forecasts, tab_comparison, tab_alerts, tab_uncertainty, tab_benchmark = st.tabs([
    "Overview",
    "Forecasts",
    "Model Comparison",
    "Alerts",
    "Uncertainty",
    "Benchmark Results",
])


# ── Tab 1: Overview ───────────────────────────────────────────────────────────

with tab_overview:
    st.header(f"Station: {station_id}")
    st.caption("All metrics are from DEMO_SYNTHETIC data — not real sensor readings.")

    col1, col2, col3, col4 = st.columns(4)
    wl = sub["water_level"].dropna()
    col1.metric("Observations", f"{len(sub):,}")
    col2.metric("Mean water level", f"{wl.mean():.3f} m")
    col3.metric("Max water level", f"{wl.max():.3f} m")
    col4.metric("Min water level", f"{wl.min():.3f} m")

    st.subheader("Water-Level Time Series")
    if _HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sub["timestamp"], y=sub["water_level"],
            name="Observed (synthetic)", line=dict(color="#1f77b4", width=1),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.3f} m<extra></extra>",
        ))
        fig.update_layout(
            xaxis_title="Time (UTC)", yaxis_title="Water Level (m)",
            height=360, margin=dict(t=30, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(sub.set_index("timestamp")["water_level"])

    st.subheader("Station Locations")
    station_coords = (
        df_all.groupby("station_id")[["lat", "lon"]].first().reset_index()
    )
    station_coords.columns = ["Station", "lat", "lon"]
    st.map(station_coords.rename(columns={"lat": "latitude", "lon": "longitude"}))


# ── Tab 2: Forecasts ──────────────────────────────────────────────────────────

with tab_forecasts:
    st.header("Forecasts vs Actual (held-out test set)")
    st.caption(
        "Train = first 75% of each station's data. "
        "Conformal calibration set = last 15% of training data. "
        "Test = last 25%."
    )

    show_persist = st.checkbox("Show Persistence baseline", value=True)
    show_gradboost = st.checkbox("Show GradBoost forecast", value=True)
    show_intervals = st.checkbox("Show 90% conformal intervals (HarmonicRidge)", value=True)

    with st.spinner("Running forecasts …"):
        try:
            train_fit, test, persist_pred, harmonic_pred, gradboost_pred, h_ci, gb_ci = (
                run_forecast(station_id)
            )
            forecast_ok = True
        except Exception as e:
            st.error(f"Forecast failed: {e}")
            forecast_ok = False

    if forecast_ok and _HAS_PLOTLY:
        fig2 = go.Figure()

        # Actual
        fig2.add_trace(go.Scatter(
            x=test["timestamp"], y=test["water_level"],
            name="Actual (synthetic)", line=dict(color="#1f77b4", width=1.5),
        ))

        # HarmonicRidge
        aligned_test = test.iloc[-len(harmonic_pred):]
        if show_intervals:
            h_lo, h_hi = h_ci.intervals(harmonic_pred)
            fig2.add_trace(go.Scatter(
                x=pd.concat([aligned_test["timestamp"], aligned_test["timestamp"][::-1]]),
                y=np.concatenate([h_hi, h_lo[::-1]]),
                fill="toself", fillcolor="rgba(44,160,44,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="HarmonicRidge 90% interval", showlegend=True,
            ))
        fig2.add_trace(go.Scatter(
            x=aligned_test["timestamp"], y=harmonic_pred,
            name="HarmonicRidge", line=dict(color="#2ca02c", width=1.5, dash="dash"),
        ))

        # GradBoost
        if show_gradboost:
            aligned_test_gb = test.iloc[-len(gradboost_pred):]
            fig2.add_trace(go.Scatter(
                x=aligned_test_gb["timestamp"], y=gradboost_pred,
                name="GradBoost", line=dict(color="#9467bd", width=1.5, dash="dot"),
            ))

        # Persistence
        if show_persist:
            fig2.add_trace(go.Scatter(
                x=test["timestamp"], y=persist_pred,
                name="Persistence", line=dict(color="#d62728", width=1, dash="dot"),
            ))

        fig2.update_layout(
            xaxis_title="Time (UTC)", yaxis_title="Water Level (m)",
            height=400, margin=dict(t=30, b=40),
            legend=dict(orientation="h", y=1.02),
        )
        st.plotly_chart(fig2, use_container_width=True)
    elif forecast_ok:
        st.line_chart(test.set_index("timestamp")["water_level"])
        st.info("Install plotly for richer charts: `pip install plotly`")


# ── Tab 3: Model Comparison ───────────────────────────────────────────────────

with tab_comparison:
    st.header("Model Performance — 1-Step (6 min) Horizon")
    st.caption("Metrics on held-out test set (synthetic demo data). All values in metres.")

    all_metrics = load_metrics()
    station_metrics = all_metrics.get(station_id, {})

    if station_metrics:
        from src.models.branding import DISPLAY_BY_KEY
        rows = []
        for model_name, m in station_metrics.items():
            if not isinstance(m, dict) or "mae" not in m:
                continue
            rows.append({
                "Model": DISPLAY_BY_KEY.get(model_name, model_name),
                "MAE (m)": f"{m.get('mae', float('nan')):.4f}",
                "RMSE (m)": f"{m.get('rmse', float('nan')):.4f}",
                "R²": f"{m.get('r2', float('nan')):.4f}",
                "NSE": f"{m.get('nse', float('nan')):.4f}",
                "Corr": f"{m.get('corr', float('nan')):.4f}",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Run `python -m scripts.train_baseline` then refresh.")

    st.subheader("Multi-Horizon Evaluation")
    st.caption(
        "Direct forecasting: separate model trained per horizon. "
        "WaveGRU evaluated at 1-step only."
    )

    horizon_metrics = load_horizon_metrics()
    station_horizon = horizon_metrics.get(station_id, {})

    if station_horizon:
        h_rows = []
        for h_name, models in station_horizon.items():
            for model_name, m in models.items():
                if not isinstance(m, dict) or "mae" not in m:
                    note = m.get("note", "—") if isinstance(m, dict) else "—"
                    h_rows.append({"Horizon": h_name, "Model": model_name,
                                   "MAE (m)": "—", "RMSE (m)": "—", "R²": note})
                    continue
                mae = m.get("mae")
                rmse = m.get("rmse")
                r2 = m.get("r2")
                h_rows.append({
                    "Horizon": h_name,
                    "Model": model_name,
                    "MAE (m)": f"{mae:.4f}" if mae and not np.isnan(mae) else "—",
                    "RMSE (m)": f"{rmse:.4f}" if rmse and not np.isnan(rmse) else "—",
                    "R²": f"{r2:.4f}" if r2 is not None and not np.isnan(r2) else "—",
                })
        if h_rows:
            st.dataframe(pd.DataFrame(h_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Run `python -m scripts.evaluate_horizons` then refresh.")


# ── Tab 4: Alerts ─────────────────────────────────────────────────────────────

with tab_alerts:
    st.header("High-Water Alert Detection")
    st.caption("Alerts are computed on the displayed date range. Thresholds are fit on all station data.")

    col_a, col_b, col_c = st.columns(3)
    alert_mode = col_a.selectbox(
        "Threshold mode",
        ["std", "absolute", "percentile"],
        index=0,
        help="std = mean + k·std · absolute = fixed value · percentile = p-th percentile",
    )
    k_val = col_b.slider("k (std multiplier)", 1.0, 4.0, 2.0, 0.25, disabled=(alert_mode != "std"))
    abs_val = col_b.number_input("Absolute threshold (m)", value=float(sub["water_level"].mean() + 2 * sub["water_level"].std()), disabled=(alert_mode != "absolute"))
    pct_val = col_c.slider("Percentile", 50, 99, 95, disabled=(alert_mode != "percentile"))

    from src.alerts import AlertConfig, detect_alerts, compute_threshold

    if alert_mode == "std":
        config = AlertConfig(mode="std", k=k_val)
    elif alert_mode == "absolute":
        config = AlertConfig(mode="absolute", absolute_threshold=abs_val)
    else:
        config = AlertConfig(mode="percentile", percentile=float(pct_val))

    threshold = compute_threshold(sub["water_level"], config)
    alerts = detect_alerts(sub, config)

    col1, col2, col3 = st.columns(3)
    col1.metric("Alert threshold", f"{threshold:.3f} m")
    col2.metric("Alert events", len(alerts))
    col3.metric("Alert rate", f"{100 * len(alerts) / max(len(sub), 1):.1f}%")

    if _HAS_PLOTLY:
        fig_a = go.Figure()
        fig_a.add_trace(go.Scatter(
            x=sub["timestamp"], y=sub["water_level"],
            name="Observed", line=dict(color="#1f77b4", width=1),
        ))
        fig_a.add_hline(
            y=threshold, line_dash="dash", line_color="orange",
            annotation_text=f"Threshold ({threshold:.2f} m)",
            annotation_position="top right",
        )
        if not alerts.empty:
            fig_a.add_trace(go.Scatter(
                x=alerts["timestamp"], y=alerts["water_level"],
                mode="markers", name=f"Alert ({len(alerts)})",
                marker=dict(color="red", size=6, symbol="circle"),
            ))
        fig_a.update_layout(
            xaxis_title="Time (UTC)", yaxis_title="Water Level (m)",
            height=360, margin=dict(t=30, b=40),
        )
        st.plotly_chart(fig_a, use_container_width=True)

    st.subheader(f"Alert Events (≥ {threshold:.2f} m)")
    if alerts.empty:
        st.success("No alert events in the selected date range.")
    else:
        display = alerts[["timestamp", "water_level"]].copy()
        display["timestamp"] = display["timestamp"].dt.strftime("%Y-%m-%d %H:%M UTC")
        display.columns = ["Timestamp (UTC)", "Water Level (m)"]
        st.dataframe(display.head(50), use_container_width=True, hide_index=True)


# ── Tab 5: Uncertainty ────────────────────────────────────────────────────────

with tab_uncertainty:
    st.header("Conformal Prediction Intervals")
    st.caption(
        "Split-conformal intervals with 90% nominal coverage. "
        "Calibration set = last 15% of training data. "
        "Coverage guarantee assumes exchangeability — may be lower for non-stationary series."
    )

    with st.spinner("Computing conformal intervals …"):
        try:
            train_fit, test, _, harmonic_pred, gradboost_pred, h_ci, gb_ci = (
                run_forecast(station_id)
            )
            ci_ok = True
        except Exception as e:
            st.error(f"Could not compute intervals: {e}")
            ci_ok = False

    if ci_ok:
        from src.features.engineering import build_feature_matrix
        aligned_test = test.iloc[-len(harmonic_pred):]
        _, y_test = build_feature_matrix(aligned_test)
        actual_test = y_test.values

        h_lo, h_hi = h_ci.intervals(harmonic_pred)
        h_cov = h_ci.empirical_coverage(actual_test, harmonic_pred[-len(actual_test):])

        aligned_test_gb = test.iloc[-len(gradboost_pred):]
        _, y_test_gb = build_feature_matrix(aligned_test_gb)
        actual_test_gb = y_test_gb.values
        gb_lo, gb_hi = gb_ci.intervals(gradboost_pred)
        gb_cov = gb_ci.empirical_coverage(actual_test_gb, gradboost_pred[-len(actual_test_gb):])

        col1, col2 = st.columns(2)
        col1.metric("HarmonicRidge qhat (±m)", f"±{h_ci.qhat:.4f}")
        col2.metric("HarmonicRidge empirical coverage", f"{h_cov:.1%}")
        col3, col4 = st.columns(2)
        col3.metric("GradBoost qhat (±m)", f"±{gb_ci.qhat:.4f}")
        col4.metric("GradBoost empirical coverage", f"{gb_cov:.1%}")

        st.caption(
            f"Calibration set size: {h_ci.n_cal} samples. "
            "Nominal coverage = 90%. Empirical coverage is on the same held-out test period."
        )

        if _HAS_PLOTLY:
            fig_u = go.Figure()
            fig_u.add_trace(go.Scatter(
                x=aligned_test["timestamp"], y=actual_test,
                name="Actual", line=dict(color="#1f77b4", width=1.5),
            ))
            fig_u.add_trace(go.Scatter(
                x=pd.concat([aligned_test["timestamp"], aligned_test["timestamp"][::-1]]),
                y=np.concatenate([h_hi, h_lo[::-1]]),
                fill="toself", fillcolor="rgba(44,160,44,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="HarmonicRidge 90% interval",
            ))
            fig_u.add_trace(go.Scatter(
                x=aligned_test["timestamp"], y=harmonic_pred,
                name="HarmonicRidge forecast", line=dict(color="#2ca02c", dash="dash"),
            ))
            fig_u.update_layout(
                xaxis_title="Time (UTC)", yaxis_title="Water Level (m)",
                height=380, margin=dict(t=30, b=40),
            )
            st.plotly_chart(fig_u, use_container_width=True)

        st.info(
            "**Known limitation**: Split-conformal guarantees assume exchangeability. "
            "Tidal series are non-stationary — empirical coverage may fall below 90%, "
            "especially at longer forecast horizons. Always report empirical coverage "
            "alongside nominal coverage."
        )


# ── Tab 6: Benchmark Results ──────────────────────────────────────────────────

with tab_benchmark:
    st.header("Prototype Benchmark — NOAA-Derived Tidecast Data")
    st.caption(
        "Metrics are on **NOAA-derived tidal predictions** (smooth, deterministic signal). "
        "Real sensor data would yield substantially higher RMSE. "
        "Prototype models are pure-Python research baselines, not production neural networks."
    )

    if BENCHMARK_PATH.exists():
        content = BENCHMARK_PATH.read_text()
        st.markdown(content)
    else:
        st.info("Run `python -m scripts.run_benchmark` then refresh.")

    st.subheader("Multi-Horizon Results")
    if HORIZON_PATH.exists():
        with open(HORIZON_PATH) as f:
            h_data = json.load(f)
        h_rows = []
        for sid, horizons in h_data.items():
            for h_name, models in horizons.items():
                for model_name, m in models.items():
                    if not isinstance(m, dict) or "mae" not in m:
                        continue
                    mae = m.get("mae")
                    rmse = m.get("rmse")
                    h_rows.append({
                        "Station": sid,
                        "Horizon": h_name,
                        "Model": model_name,
                        "MAE (m)": f"{mae:.4f}" if mae and not np.isnan(mae) else "—",
                        "RMSE (m)": f"{rmse:.4f}" if rmse and not np.isnan(rmse) else "—",
                    })
        if h_rows:
            st.dataframe(pd.DataFrame(h_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Run `python -m scripts.evaluate_horizons` then refresh.")

st.caption(
    "Wai · Demo data (DEMO_SYNTHETIC) · "
    "No private or proprietary data is used."
)
