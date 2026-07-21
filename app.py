"""Wai — Streamlit control panel for coastal water-level estimates.

Run:
    streamlit run app.py

Tabs
----
  Control Center    — estimate, uncertainty, accuracy, and operational status
  Overview          — station summary stats and location map
  Forecasts         — time series with forecast overlay and conformal intervals
  Model Comparison  — metrics table for all pipeline models
  Alerts            — configurable high-water alert detection
  Uncertainty       — conformal prediction interval details
  Benchmark Results — prototype model RMSE on tidecast data

Scientific protocol shown to users
----------------------------------
- Persistence baseline is rolling 1-step: pred[t] = observed[t-1]
  (matches scripts/train_baseline.rolling_persistence_1step). The previous
  constant-last-train baseline is retained only as a reference floor.
- Alert thresholds are fit on the *training* window for the selected station
  (75 % temporal split). The displayed date range is for visualisation only —
  the threshold never moves with the date filter.
- "Forecast" labels make the protocol explicit: 1-step forecasts use the most
  recent observed value as input ("online 1-step"); longer horizons come
  from the direct multi-horizon evaluation (separate model per horizon).
"""

from __future__ import annotations

import json
from html import escape
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
SUMMARY_PATH = Path("reports/summary.json")
DEMO_DATA_PATH = Path("data/demo/demo_water_levels.csv")

PALETTE = {
    "ink": "#172033",
    "muted": "#64748B",
    "grid": "#E2E8F0",
    "surface": "#FFFFFF",
    "blue": "#2563EB",
    "blue_light": "rgba(37, 99, 235, 0.13)",
    "gold": "#D08B18",
    "gold_light": "rgba(208, 139, 24, 0.16)",
    "orange": "#C45D24",
    "slate": "#8B98AA",
    "terrain": "#68783F",
    "sand": "#B9872F",
    "water": "rgba(37, 99, 235, 0.48)",
    "water_range": "rgba(111, 165, 255, 0.28)",
}

MODEL_VIEW = {
    "harmonic_ridge": {
        "label": "Harmonic Ridge",
        "prediction": "harmonic_pred",
        "lower": "harmonic_lower",
        "upper": "harmonic_upper",
        "coverage": "harmonic_coverage",
        "interval": "harmonic_ci",
        "color": PALETTE["gold"],
        "fill": PALETTE["gold_light"],
    },
    "grad_boost": {
        "label": "Gradient Boost",
        "prediction": "gradboost_pred",
        "lower": "gradboost_lower",
        "upper": "gradboost_upper",
        "coverage": "gradboost_coverage",
        "interval": "gradboost_ci",
        "color": PALETTE["blue"],
        "fill": PALETTE["blue_light"],
    },
}

DASHBOARD_MODEL_LABELS = {
    "persistence": "Persistence",
    "persistence_constant": "Constant holdout",
    "harmonic_ridge": "Harmonic Ridge",
    "grad_boost": "Gradient Boost",
    "wave_gru": "Fast Wave Adapter",
}

PLOTLY_CONFIG = {
    "displayModeBar": False,
    "displaylogo": False,
    "scrollZoom": False,
    "responsive": True,
}

# The original schematic used the full 0–100 horizontal span for its coastal
# slope. Compressing that profile into 0–50 makes the visible land footprint
# half as wide, then extends a shallow offshore bed through the remaining span.
SHORELINE_PROFILE_X = (0, 6, 12, 17.5, 22.5, 28, 34, 41, 50, 75, 100)
SHORELINE_DEPTH_FRACTIONS = (
    0.04,
    0.08,
    0.14,
    0.22,
    0.34,
    0.50,
    0.68,
    0.82,
    0.93,
    0.96,
    0.98,
)
SHORELINE_TICK_VALUES = (5, 28, 78)
TIDE_FRAME_DURATION_MS = 110
TIDE_TRANSITION_MS = 45

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
def load_summary() -> dict:
    if SUMMARY_PATH.exists():
        with open(SUMMARY_PATH) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=60)
def load_evidence_status() -> dict:
    """Return a live freshness verdict instead of trusting a static report flag."""

    try:
        from scripts.check_report_freshness import check_report_freshness

        verdict = check_report_freshness()
        return {
            "fresh": True,
            "fingerprint": verdict["current_source_fingerprint"],
            "message": "Evidence matches the current source",
        }
    except Exception as error:
        return {
            "fresh": False,
            "fingerprint": None,
            "message": str(error),
        }


@st.cache_data(ttl=300, show_spinner=False)
def load_live_noaa_snapshot(
    station,
    lookback_hours: int,
    datum: str,
    include_tide_predictions: bool,
):
    """Fetch a five-minute cached NOAA monitor snapshot with no mock fallback."""

    from src.data.noaa_live import fetch_live_noaa_snapshot

    return fetch_live_noaa_snapshot(
        station.station_id,
        lookback_hours=lookback_hours,
        datum=datum,
        include_tide_predictions=include_tide_predictions,
        station=station,
    )


@st.cache_data(ttl=300, show_spinner=False)
def load_live_noaa_guidance(
    station,
    history_hours: int,
    datum: str,
):
    """Fetch a five-minute cached NOAA OFS guidance window when supported."""

    from src.data.noaa_live import fetch_live_noaa_operational_guidance

    return fetch_live_noaa_operational_guidance(
        station.station_id,
        history_hours=history_hours,
        forecast_hours=48,
        datum=datum,
        station=station,
    )


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_live_noaa_station_catalog():
    """Discover all active NOAA water-level stations with a bundled fallback."""

    from src.data.noaa_catalog import load_noaa_station_catalog

    return load_noaa_station_catalog()


@st.cache_data
def run_forecast(station_id: str, train_frac: float = 0.75):
    """Run the dashboard's online 1-step forecast pipeline.

    Returns
    -------
    dict with aligned timestamps, actual values, model predictions, conformal
    intervals, and coverage summaries. All plotted arrays are aligned to the
    same feature-valid test timestamps.

    `persistence_pred` is the rolling 1-step persistence (matches
    `scripts/train_baseline.rolling_persistence_1step`) sampled at the same
    timestamps as the supervised models. `train_threshold` is mean + 2σ fit on
    the training window only — never on the displayed date range.
    """
    from src.models.baseline import HarmonicRidgeModel
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

    # Rolling 1-step persistence (matches scripts/train_baseline.py).
    # pred[0] = last value of full train; pred[t] = test[t-1] thereafter.
    test_vals = test["water_level"].values
    train_full = sub.iloc[:n_train]
    last_train = float(train_full["water_level"].dropna().iloc[-1])
    persist_pred = np.empty(len(test_vals))
    if len(test_vals):
        persist_pred[0] = last_train
        persist_pred[1:] = test_vals[:-1]

    harmonic = HarmonicRidgeModel(alpha=1.0).fit(train_fit)
    harmonic_cal = harmonic.predict_aligned(train_cal)
    harmonic_test = harmonic.predict_aligned(test)

    gradboost = GradBoostModel().fit(train_fit)
    gradboost_cal = gradboost.predict_aligned(train_cal)
    gradboost_test = gradboost.predict_aligned(test)

    if not harmonic_test["timestamp"].equals(gradboost_test["timestamp"]):
        raise RuntimeError("HarmonicRidge and GradBoost test timestamps are not aligned")
    if not harmonic_cal["timestamp"].equals(gradboost_cal["timestamp"]):
        raise RuntimeError("HarmonicRidge and GradBoost calibration timestamps are not aligned")

    # Conformal calibration on each model's already-aligned calibration rows.
    harmonic_ci = ConformalIntervals(coverage=0.90)
    harmonic_ci.calibrate(
        harmonic_cal["actual"].to_numpy(dtype=float),
        harmonic_cal["prediction"].to_numpy(dtype=float),
    )

    gb_ci = ConformalIntervals(coverage=0.90)
    gb_ci.calibrate(
        gradboost_cal["actual"].to_numpy(dtype=float),
        gradboost_cal["prediction"].to_numpy(dtype=float),
    )

    # Train-window-only alert threshold.
    train_wl = train_full["water_level"].dropna()
    train_threshold = float(train_wl.mean() + 2.0 * train_wl.std())

    rows = harmonic_test["_source_row"].to_numpy(dtype=int)
    timestamps = harmonic_test["timestamp"].reset_index(drop=True)
    actual = harmonic_test["actual"].to_numpy(dtype=float)
    harmonic_pred = harmonic_test["prediction"].to_numpy(dtype=float)
    gradboost_pred = gradboost_test["prediction"].to_numpy(dtype=float)
    persist_aligned = persist_pred[rows]
    h_lo, h_hi = harmonic_ci.intervals(harmonic_pred)
    gb_lo, gb_hi = gb_ci.intervals(gradboost_pred)

    return {
        "train_fit": train_fit,
        "train_cal": train_cal,
        "test": test,
        "timestamps": timestamps,
        "actual": actual,
        "persistence_pred": persist_aligned,
        "harmonic_pred": harmonic_pred,
        "gradboost_pred": gradboost_pred,
        "harmonic_lower": h_lo,
        "harmonic_upper": h_hi,
        "gradboost_lower": gb_lo,
        "gradboost_upper": gb_hi,
        "harmonic_ci": harmonic_ci,
        "gradboost_ci": gb_ci,
        "harmonic_coverage": harmonic_ci.stratified_coverage(
            actual, harmonic_pred, event_threshold=train_threshold,
        ),
        "gradboost_coverage": gb_ci.stratified_coverage(
            actual, gradboost_pred, event_threshold=train_threshold,
        ),
        "train_threshold": train_threshold,
    }


def build_estimate_frame(forecast: dict, model_key: str) -> pd.DataFrame:
    """Return one aligned, chart-ready estimate table for the control panel."""

    if model_key not in MODEL_VIEW:
        raise ValueError(f"Unsupported dashboard model: {model_key}")
    spec = MODEL_VIEW[model_key]
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(forecast["timestamps"], utc=True),
        "actual": np.asarray(forecast["actual"], dtype=float),
        "estimate": np.asarray(forecast[spec["prediction"]], dtype=float),
        "lower": np.asarray(forecast[spec["lower"]], dtype=float),
        "upper": np.asarray(forecast[spec["upper"]], dtype=float),
        "persistence": np.asarray(forecast["persistence_pred"], dtype=float),
    })
    frame["error"] = frame["estimate"] - frame["actual"]
    frame["absolute_error"] = frame["error"].abs()
    frame["inside_interval"] = frame["actual"].between(frame["lower"], frame["upper"])
    # Six hours at the demo's six-minute cadence. min_periods keeps the leading
    # edge honest instead of silently dropping it.
    frame["rolling_mae_6h"] = frame["absolute_error"].rolling(
        60, min_periods=1
    ).mean()
    return frame


def summarize_estimates(forecast: dict, model_key: str) -> dict:
    """Compute held-out accuracy and the latest replay estimate."""

    frame = build_estimate_frame(forecast, model_key)
    spec = MODEL_VIEW[model_key]
    persistence_mae = float(np.mean(np.abs(frame["persistence"] - frame["actual"])))
    mae = float(frame["absolute_error"].mean())
    rmse = float(np.sqrt(np.mean(np.square(frame["error"]))))
    coverage = float(frame["inside_interval"].mean())
    latest = frame.iloc[-1]
    return {
        "model_label": spec["label"],
        "n_samples": int(len(frame)),
        "mae": mae,
        "rmse": rmse,
        "persistence_mae": persistence_mae,
        "skill_vs_persistence": (
            (persistence_mae - mae) / persistence_mae
            if persistence_mae > 0
            else float("nan")
        ),
        "coverage": coverage,
        "interval_half_width": float(forecast[spec["interval"]].qhat),
        "latest_timestamp": latest["timestamp"],
        "latest_actual": float(latest["actual"]),
        "latest_estimate": float(latest["estimate"]),
        "latest_lower": float(latest["lower"]),
        "latest_upper": float(latest["upper"]),
        "latest_absolute_error": float(latest["absolute_error"]),
    }


def window_estimates(frame: pd.DataFrame, hours: int | None) -> pd.DataFrame:
    """Apply a recent-window filter without changing any fitted metric."""

    if hours is None or frame.empty:
        return frame.copy()
    cutoff = frame["timestamp"].max() - pd.Timedelta(hours=hours)
    return frame.loc[frame["timestamp"] >= cutoff].copy()


def model_accuracy_frame(metrics: dict, station_id: str) -> pd.DataFrame:
    """Shape model metrics for a lower-is-better ranked comparison."""

    rows = []
    for model_key, values in metrics.get(station_id, {}).items():
        if not isinstance(values, dict) or "mae" not in values:
            continue
        rows.append({
            "model_key": model_key,
            "model": DASHBOARD_MODEL_LABELS.get(
                model_key, model_key.replace("_", " ").title()
            ),
            "mae": float(values["mae"]),
            "rmse": float(values["rmse"]),
            "r2": float(values["r2"]),
        })
    if not rows:
        return pd.DataFrame(columns=["model_key", "model", "mae", "rmse", "r2"])
    return pd.DataFrame(rows).sort_values("mae", ascending=True).reset_index(drop=True)


def horizon_accuracy_frame(horizon_metrics: dict, station_id: str) -> pd.DataFrame:
    """Shape the four discrete forecast horizons for grouped RMSE bars."""

    horizon_order = {"1step_6min": 0, "6h": 1, "12h": 2, "24h": 3}
    rows = []
    for horizon, models in horizon_metrics.get(station_id, {}).items():
        if horizon.startswith("_") or not isinstance(models, dict):
            continue
        for model_key, values in models.items():
            if model_key.startswith("_") or not isinstance(values, dict) or "rmse" not in values:
                continue
            rows.append({
                "horizon": horizon,
                "horizon_order": horizon_order.get(horizon, 99),
                "model_key": model_key,
                "model": DASHBOARD_MODEL_LABELS.get(
                    model_key, model_key.replace("_", " ").title()
                ),
                "rmse": float(values["rmse"]),
                "mae": float(values["mae"]),
            })
    if not rows:
        return pd.DataFrame(
            columns=["horizon", "horizon_order", "model_key", "model", "rmse", "mae"]
        )
    return pd.DataFrame(rows).sort_values(
        ["horizon_order", "model"]
    ).reset_index(drop=True)


@st.cache_data(show_spinner=False, max_entries=16)
def build_tide_motion_figure(
    frame: pd.DataFrame,
    *,
    model_key: str,
    alert_threshold: float,
    max_frames: int = 72,
):
    """Build an animated shoreline cross-section linked to the estimate series.

    The shoreline profile is deliberately schematic. Its vertical axis shares
    the model's water-level datum so the moving surface, uncertainty band, and
    reference levels remain quantitatively meaningful.
    """

    if not _HAS_PLOTLY:
        raise RuntimeError("Plotly is required for the tide-motion explorer")
    if model_key not in MODEL_VIEW:
        raise ValueError(f"Unsupported dashboard model: {model_key}")
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1")
    if not np.isfinite(alert_threshold):
        raise ValueError("alert_threshold must be finite")

    required = ["timestamp", "actual", "estimate", "lower", "upper"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Tide-motion frame is missing columns: {missing}")

    work = frame[required].copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
    for column in required[1:]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna().sort_values("timestamp").reset_index(drop=True)
    if work.empty:
        raise ValueError("Tide-motion frame has no complete observations")

    from plotly.subplots import make_subplots

    spec = MODEL_VIEW[model_key]
    frame_count = min(max_frames, len(work))
    sample_indices = np.unique(
        np.linspace(0, len(work) - 1, frame_count, dtype=int)
    )

    time_hours = (
        (work["timestamp"] - work["timestamp"].iloc[0])
        .dt.total_seconds()
        .to_numpy(dtype=float)
        / 3600
    )
    if len(work) > 1:
        raw_rate = np.gradient(work["estimate"].to_numpy(dtype=float), time_hours)
        work["rate_cm_hour"] = (
            pd.Series(raw_rate).rolling(5, center=True, min_periods=1).mean() * 100
        )
    else:
        work["rate_cm_hour"] = 0.0

    all_levels = np.concatenate([
        work["lower"].to_numpy(dtype=float),
        work["upper"].to_numpy(dtype=float),
        work["actual"].to_numpy(dtype=float),
        np.array([float(alert_threshold)]),
    ])
    data_min = float(np.min(all_levels))
    data_max = float(np.max(all_levels))
    level_span = max(data_max - data_min, 0.5)
    scene_min = data_min - 0.22 * level_span
    scene_max = data_max + 0.18 * level_span
    scene_span = scene_max - scene_min
    status_y = scene_max + 0.10 * scene_span
    display_scene_max = scene_max + 0.20 * scene_span
    series_min = scene_min
    series_max = scene_max

    terrain_x = np.array(SHORELINE_PROFILE_X, dtype=float)
    terrain_y = scene_max - scene_span * np.array(
        SHORELINE_DEPTH_FRACTIONS, dtype=float
    )
    high_reference = float(work["actual"].quantile(0.90))
    low_reference = float(work["actual"].quantile(0.10))

    def shore_position(level: float) -> float:
        """Interpolate where the schematic terrain crosses a water level."""

        if level >= terrain_y[0]:
            return float(terrain_x[0])
        if level <= terrain_y[-1]:
            return float(terrain_x[-1])
        crossing = int(np.flatnonzero(terrain_y <= level)[0])
        left = crossing - 1
        fraction = (terrain_y[left] - level) / (
            terrain_y[left] - terrain_y[crossing]
        )
        return float(
            terrain_x[left] + fraction * (terrain_x[crossing] - terrain_x[left])
        )

    def water_polygon(level: float) -> tuple[list[float], list[float]]:
        return [0.0, 100.0, 100.0, 0.0], [level, level, scene_min, scene_min]

    def range_polygon(lower: float, upper: float) -> tuple[list[float], list[float]]:
        return [0.0, 100.0, 100.0, 0.0], [upper, upper, lower, lower]

    def phase_label(rate: float) -> str:
        if rate > 0.2:
            return "RISING"
        if rate < -0.2:
            return "FALLING"
        return "NEAR SLACK"

    def status_text(row: pd.Series) -> str:
        headroom = float(alert_threshold - row["estimate"])
        return (
            f"<b>{row['timestamp']:%b %d %H:%M} UTC · "
            f"{phase_label(float(row['rate_cm_hour']))} "
            f"{float(row['rate_cm_hour']):+.1f} cm/h</b><br>"
            f"Est {float(row['estimate']):.3f} · "
            f"Obs {float(row['actual']):.3f} · "
            f"90% {float(row['lower']):.3f}–{float(row['upper']):.3f} m · "
            f"Error {abs(float(row['estimate'] - row['actual'])):.3f} · "
            f"headroom {headroom:+.3f} m"
        )

    first = work.iloc[int(sample_indices[0])]
    first_water_x, first_water_y = water_polygon(float(first["estimate"]))
    first_range_x, first_range_y = range_polygon(
        float(first["lower"]), float(first["upper"])
    )
    first_shore = shore_position(float(first["estimate"]))

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.53, 0.47],
        vertical_spacing=0.18,
        subplot_titles=(
            "Animated shoreline cross-section",
            "History: observed solid · estimate dashed · 90% band",
        ),
    )
    fig.add_trace(go.Scatter(
        x=first_water_x,
        y=first_water_y,
        mode="lines",
        fill="toself",
        fillcolor=PALETTE["water"],
        line=dict(color="rgba(37, 99, 235, 0)"),
        hoverinfo="skip",
        showlegend=False,
        name="Estimated water volume",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=first_range_x,
        y=first_range_y,
        mode="lines",
        fill="toself",
        fillcolor=PALETTE["water_range"],
        line=dict(color="rgba(111, 165, 255, 0)"),
        hoverinfo="skip",
        showlegend=False,
        name="Moving 90% range",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=np.concatenate([terrain_x, [100, 0]]),
        y=np.concatenate([terrain_y, [scene_min, scene_min]]),
        mode="lines",
        fill="toself",
        fillcolor=PALETTE["terrain"],
        line=dict(color=PALETTE["terrain"], width=1),
        hoverinfo="skip",
        showlegend=False,
        name="Schematic terrain",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=terrain_x,
        y=terrain_y,
        mode="lines",
        line=dict(color=PALETTE["sand"], width=3),
        hoverinfo="skip",
        showlegend=False,
        name="Schematic shoreline",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[first_shore, 100],
        y=[float(first["estimate"]), float(first["estimate"])],
        mode="lines",
        line=dict(color=PALETTE["blue"], width=3),
        hovertemplate="Estimated surface %{y:.3f} m<extra></extra>",
        showlegend=False,
        name="Estimated surface",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[58, 100],
        y=[high_reference, high_reference],
        mode="lines+text",
        text=[None, "display high (90th pct.)"],
        textposition="top left",
        textfont=dict(color=PALETTE["muted"], size=11),
        line=dict(color=PALETTE["muted"], width=1, dash="dot"),
        hovertemplate=f"90th percentile {high_reference:.3f} m<extra></extra>",
        showlegend=False,
        name="Display high reference",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[58, 100],
        y=[low_reference, low_reference],
        mode="lines+text",
        text=[None, "display low (10th pct.)"],
        textposition="bottom left",
        textfont=dict(color=PALETTE["muted"], size=11),
        line=dict(color=PALETTE["muted"], width=1, dash="dot"),
        hovertemplate=f"10th percentile {low_reference:.3f} m<extra></extra>",
        showlegend=False,
        name="Display low reference",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[58, 100],
        y=[alert_threshold, alert_threshold],
        mode="lines+text",
        text=[None, "training alert threshold"],
        textposition="top left",
        textfont=dict(color=PALETTE["orange"], size=11),
        line=dict(color=PALETTE["orange"], width=1.5, dash="dash"),
        hovertemplate=f"Training threshold {alert_threshold:.3f} m<extra></extra>",
        showlegend=False,
        name="Training alert threshold",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[2],
        y=[status_y],
        mode="text",
        text=[status_text(first)],
        textposition="middle right",
        textfont=dict(color=PALETTE["ink"], size=10),
        hoverinfo="skip",
        showlegend=False,
        name="Current state",
    ), row=1, col=1)

    interval_x = pd.concat([work["timestamp"], work["timestamp"][::-1]])
    interval_y = pd.concat([work["upper"], work["lower"][::-1]])
    fig.add_trace(go.Scatter(
        x=interval_x,
        y=interval_y,
        fill="toself",
        fillcolor=spec["fill"],
        line=dict(color="rgba(255,255,255,0)"),
        name="90% interval",
        hoverinfo="skip",
        showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=work["timestamp"],
        y=work["actual"],
        name="Observed outcome",
        line=dict(color=PALETTE["ink"], width=1.6),
        hovertemplate="%{x|%b %d %H:%M}<br>Observed %{y:.3f} m<extra></extra>",
        showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=work["timestamp"],
        y=work["estimate"],
        name=f"{spec['label']} estimate",
        line=dict(color=spec["color"], width=2, dash="dash"),
        hovertemplate="%{x|%b %d %H:%M}<br>Estimate %{y:.3f} m<extra></extra>",
        showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[first["timestamp"], first["timestamp"]],
        y=[series_min, series_max],
        mode="lines",
        line=dict(color=PALETTE["orange"], width=1.5),
        hoverinfo="skip",
        showlegend=False,
        name="Selected time",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[first["timestamp"]],
        y=[first["actual"]],
        mode="markers",
        marker=dict(color=PALETTE["ink"], size=9, line=dict(color="#FFFFFF", width=2)),
        hovertemplate="Observed %{y:.3f} m<extra></extra>",
        showlegend=False,
        name="Selected observed",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[first["timestamp"]],
        y=[first["estimate"]],
        mode="markers",
        marker=dict(color=spec["color"], size=9, line=dict(color="#FFFFFF", width=2)),
        hovertemplate="Estimate %{y:.3f} m<extra></extra>",
        showlegend=False,
        name="Selected estimate",
    ), row=2, col=1)

    animation_frames = []
    slider_steps = []
    for sequence, work_index in enumerate(sample_indices):
        row = work.iloc[int(work_index)]
        estimate = float(row["estimate"])
        water_x, water_y = water_polygon(estimate)
        range_x, range_y = range_polygon(float(row["lower"]), float(row["upper"]))
        shore = shore_position(estimate)
        frame_name = f"tide-{sequence:03d}"
        animation_frames.append(go.Frame(
            name=frame_name,
            traces=[0, 1, 4, 8, 12, 13, 14],
            data=[
                go.Scatter(x=water_x, y=water_y),
                go.Scatter(x=range_x, y=range_y),
                go.Scatter(x=[shore, 100], y=[estimate, estimate]),
                go.Scatter(
                    x=[2],
                    y=[status_y],
                    text=[status_text(row)],
                ),
                go.Scatter(
                    x=[row["timestamp"], row["timestamp"]],
                    y=[series_min, series_max],
                ),
                go.Scatter(x=[row["timestamp"]], y=[row["actual"]]),
                go.Scatter(x=[row["timestamp"]], y=[row["estimate"]]),
            ],
        ))
        slider_steps.append({
            "args": [
                [frame_name],
                {
                    "frame": {"duration": 0, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
            "label": f"{row['timestamp']:%b %d %H:%M}",
            "method": "animate",
        })
    fig.frames = animation_frames

    style_figure(
        fig,
        title="Tide motion explorer",
        subtitle=(
            f"{len(work):,} six-minute samples · "
            f"{len(animation_frames)} motion states · model datum"
        ),
        height=760,
    )
    fig.update_layout(
        margin=dict(t=94, r=24, b=112, l=62),
        showlegend=False,
        updatemenus=[{
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {
                                "duration": TIDE_FRAME_DURATION_MS,
                                "redraw": False,
                            },
                            "fromcurrent": True,
                            "mode": "immediate",
                            "transition": {"duration": TIDE_TRANSITION_MS},
                        },
                    ],
                    "label": "▶ Play",
                    "method": "animate",
                },
                {
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": "Ⅱ Pause",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 48},
            "showactive": False,
            "type": "buttons",
            "x": 0,
            "xanchor": "left",
            "y": -0.08,
            "yanchor": "top",
        }],
        sliders=[{
            "active": 0,
            "currentvalue": {
                "font": {"color": PALETTE["ink"], "size": 12},
                "prefix": "Selected: ",
                "visible": True,
                "xanchor": "left",
            },
            "font": {"color": PALETTE["muted"], "size": 10},
            "len": 0.82,
            "pad": {"b": 0, "t": 48},
            "steps": slider_steps,
            "x": 0.18,
            "xanchor": "left",
            "y": -0.08,
            "yanchor": "top",
        }],
    )
    fig.update_xaxes(
        row=1,
        col=1,
        range=[0, 100],
        tickmode="array",
        tickvals=list(SHORELINE_TICK_VALUES),
        ticktext=["Land", "Shore", "Offshore"],
        title_text=None,
        fixedrange=True,
    )
    fig.update_yaxes(
        row=1,
        col=1,
        range=[scene_min, display_scene_max],
        title_text="Water level (m, model datum)",
        fixedrange=True,
    )
    fig.update_xaxes(row=2, col=1, title_text="Held-out time (UTC)")
    fig.update_yaxes(
        row=2,
        col=1,
        range=[series_min, series_max],
        title_text="Water level (m)",
        fixedrange=True,
    )
    for annotation in fig.layout.annotations:
        annotation.font = dict(color=PALETTE["ink"], size=12)
    return fig


@st.cache_data(show_spinner=False, max_entries=16)
def build_live_noaa_tide_figure(
    frame: pd.DataFrame,
    *,
    datum: str,
    max_frames: int = 24,
):
    """Animate the measured NOAA level against its astronomical tide prediction."""

    if not _HAS_PLOTLY:
        raise RuntimeError("Plotly is required for the NOAA tide-motion explorer")
    required = ["timestamp", "observed_m", "predicted_m"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Live tide frame is missing columns: {missing}")

    work = frame[required].copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
    work["observed_m"] = pd.to_numeric(work["observed_m"], errors="coerce")
    work["predicted_m"] = pd.to_numeric(work["predicted_m"], errors="coerce")
    work = work.dropna().sort_values("timestamp").reset_index(drop=True)
    if work.empty:
        raise ValueError("Live tide frame has no aligned NOAA samples")

    from plotly.subplots import make_subplots

    frame_count = min(max_frames, len(work))
    sample_indices = np.unique(np.linspace(0, len(work) - 1, frame_count, dtype=int))
    hours = (
        (work["timestamp"] - work["timestamp"].iloc[0])
        .dt.total_seconds()
        .to_numpy(dtype=float)
        / 3600
    )
    if len(work) > 1:
        rate = np.gradient(work["observed_m"].to_numpy(dtype=float), hours)
        work["rate_cm_hour"] = (
            pd.Series(rate).rolling(5, center=True, min_periods=1).mean() * 100
        )
    else:
        work["rate_cm_hour"] = 0.0
    work["residual_m"] = work["observed_m"] - work["predicted_m"]

    all_levels = np.concatenate([
        work["observed_m"].to_numpy(dtype=float),
        work["predicted_m"].to_numpy(dtype=float),
    ])
    data_min = float(np.min(all_levels))
    data_max = float(np.max(all_levels))
    level_span = max(data_max - data_min, 0.5)
    scene_min = data_min - 0.22 * level_span
    scene_max = data_max + 0.18 * level_span
    scene_span = scene_max - scene_min
    status_y = scene_max + 0.10 * scene_span
    display_scene_max = scene_max + 0.20 * scene_span
    terrain_x = np.array(SHORELINE_PROFILE_X, dtype=float)
    terrain_y = scene_max - scene_span * np.array(
        SHORELINE_DEPTH_FRACTIONS, dtype=float
    )
    high_reference = float(work["observed_m"].quantile(0.90))
    low_reference = float(work["observed_m"].quantile(0.10))

    def shore_position(level: float) -> float:
        if level >= terrain_y[0]:
            return float(terrain_x[0])
        if level <= terrain_y[-1]:
            return float(terrain_x[-1])
        crossing = int(np.flatnonzero(terrain_y <= level)[0])
        left = crossing - 1
        fraction = (terrain_y[left] - level) / (
            terrain_y[left] - terrain_y[crossing]
        )
        return float(
            terrain_x[left] + fraction * (terrain_x[crossing] - terrain_x[left])
        )

    def water_polygon(level: float) -> tuple[list[float], list[float]]:
        return [0.0, 100.0, 100.0, 0.0], [level, level, scene_min, scene_min]

    def phase_label(rate: float) -> str:
        if rate > 0.2:
            return "RISING"
        if rate < -0.2:
            return "FALLING"
        return "NEAR SLACK"

    def status_text(row: pd.Series) -> str:
        residual = float(row["residual_m"])
        return (
            f"<b>{row['timestamp']:%b %d %H:%M} UTC · "
            f"{phase_label(float(row['rate_cm_hour']))} "
            f"{float(row['rate_cm_hour']):+.1f} cm/h</b><br>"
            f"Obs {float(row['observed_m']):.3f} · "
            f"Tide {float(row['predicted_m']):.3f} · "
            f"Residual {residual:+.3f} m · {datum}"
        )

    first = work.iloc[int(sample_indices[0])]
    water_x, water_y = water_polygon(float(first["observed_m"]))
    first_shore = shore_position(float(first["observed_m"]))
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.53, 0.47],
        vertical_spacing=0.17,
        subplot_titles=(
            "Measured shoreline cross-section",
            "Synchronized history: observation solid · NOAA prediction dashed",
        ),
    )
    fig.add_trace(go.Scatter(
        x=water_x,
        y=water_y,
        mode="lines",
        fill="toself",
        fillcolor=PALETTE["water"],
        line=dict(color="rgba(37, 99, 235, 0)"),
        hoverinfo="skip",
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=np.concatenate([terrain_x, [100, 0]]),
        y=np.concatenate([terrain_y, [scene_min, scene_min]]),
        mode="lines",
        fill="toself",
        fillcolor=PALETTE["terrain"],
        line=dict(color=PALETTE["terrain"], width=1),
        hoverinfo="skip",
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=terrain_x,
        y=terrain_y,
        mode="lines",
        line=dict(color=PALETTE["sand"], width=3),
        hoverinfo="skip",
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[first_shore, 100],
        y=[float(first["observed_m"]), float(first["observed_m"])],
        mode="lines",
        line=dict(color=PALETTE["blue"], width=3),
        hovertemplate="Observed surface %{y:.3f} m<extra></extra>",
        showlegend=False,
    ), row=1, col=1)
    for reference, label, position in (
        (high_reference, "window high (90th pct.)", "top left"),
        (low_reference, "window low (10th pct.)", "bottom left"),
    ):
        fig.add_trace(go.Scatter(
            x=[60, 100],
            y=[reference, reference],
            mode="lines+text",
            text=[None, label],
            textposition=position,
            textfont=dict(color=PALETTE["muted"], size=11),
            line=dict(color=PALETTE["muted"], width=1, dash="dot"),
            hovertemplate=f"{label} {reference:.3f} m<extra></extra>",
            showlegend=False,
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[2],
        y=[status_y],
        mode="text",
        text=[status_text(first)],
        textposition="middle right",
        textfont=dict(color=PALETTE["ink"], size=11),
        hoverinfo="skip",
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=work["timestamp"],
        y=work["observed_m"],
        name="NOAA observed",
        line=dict(color=PALETTE["ink"], width=1.8),
        hovertemplate="%{x|%b %d %H:%M}<br>Observed %{y:.3f} m<extra></extra>",
        showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=work["timestamp"],
        y=work["predicted_m"],
        name="NOAA tide prediction",
        line=dict(color=PALETTE["gold"], width=2, dash="dash"),
        hovertemplate="%{x|%b %d %H:%M}<br>Prediction %{y:.3f} m<extra></extra>",
        showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[first["timestamp"], first["timestamp"]],
        y=[scene_min, scene_max],
        mode="lines",
        line=dict(color=PALETTE["orange"], width=1.5),
        hoverinfo="skip",
        showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[first["timestamp"]],
        y=[first["observed_m"]],
        mode="markers",
        marker=dict(color=PALETTE["ink"], size=9, line=dict(color="#FFFFFF", width=2)),
        hovertemplate="Observed %{y:.3f} m<extra></extra>",
        showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[first["timestamp"]],
        y=[first["predicted_m"]],
        mode="markers",
        marker=dict(color=PALETTE["gold"], size=9, line=dict(color="#FFFFFF", width=2)),
        hovertemplate="Prediction %{y:.3f} m<extra></extra>",
        showlegend=False,
    ), row=2, col=1)

    animation_frames = []
    slider_steps = []
    for sequence, work_index in enumerate(sample_indices):
        row = work.iloc[int(work_index)]
        level = float(row["observed_m"])
        water_x, water_y = water_polygon(level)
        shore = shore_position(level)
        frame_name = f"noaa-{sequence:03d}"
        animation_frames.append(go.Frame(
            name=frame_name,
            traces=[0, 3, 6, 9, 10, 11],
            data=[
                go.Scatter(x=water_x, y=water_y),
                go.Scatter(x=[shore, 100], y=[level, level]),
                go.Scatter(text=[status_text(row)]),
                go.Scatter(
                    x=[row["timestamp"], row["timestamp"]],
                    y=[scene_min, scene_max],
                ),
                go.Scatter(x=[row["timestamp"]], y=[row["observed_m"]]),
                go.Scatter(x=[row["timestamp"]], y=[row["predicted_m"]]),
            ],
        ))
        slider_steps.append({
            "args": [
                [frame_name],
                {
                    "frame": {"duration": 0, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
            "label": f"{row['timestamp']:%b %d %H:%M}",
            "method": "animate",
        })
    fig.frames = animation_frames
    style_figure(
        fig,
        title="Live NOAA tide motion",
        subtitle=(
            f"{len(work):,} aligned six-minute samples · "
            f"{len(animation_frames)} motion states · {datum}"
        ),
        height=720,
    )
    fig.update_layout(
        margin=dict(t=94, r=24, b=112, l=62),
        showlegend=False,
        updatemenus=[{
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {
                                "duration": TIDE_FRAME_DURATION_MS,
                                "redraw": False,
                            },
                            "fromcurrent": True,
                            "mode": "immediate",
                            "transition": {"duration": TIDE_TRANSITION_MS},
                        },
                    ],
                    "label": "▶ Play",
                    "method": "animate",
                },
                {
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": "Ⅱ Pause",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 48},
            "showactive": False,
            "type": "buttons",
            "x": 0,
            "xanchor": "left",
            "y": -0.08,
            "yanchor": "top",
        }],
        sliders=[{
            "active": 0,
            "currentvalue": {
                "font": {"color": PALETTE["ink"], "size": 12},
                "prefix": "Selected: ",
                "visible": True,
                "xanchor": "left",
            },
            "font": {"color": PALETTE["muted"], "size": 10},
            "len": 0.82,
            "pad": {"b": 0, "t": 48},
            "steps": slider_steps,
            "x": 0.18,
            "xanchor": "left",
            "y": -0.08,
            "yanchor": "top",
        }],
    )
    fig.update_xaxes(
        row=1,
        col=1,
        range=[0, 100],
        tickmode="array",
        tickvals=list(SHORELINE_TICK_VALUES),
        ticktext=["Land", "Shore", "Offshore"],
        title_text=None,
        fixedrange=True,
    )
    fig.update_yaxes(
        row=1,
        col=1,
        range=[scene_min, display_scene_max],
        title_text=f"Water level (m, {datum})",
        fixedrange=True,
    )
    # The animated slider already carries UTC timestamps below this axis; an
    # additional x-axis title competes with the compact play/pause controls.
    fig.update_xaxes(row=2, col=1, title_text=None)
    fig.update_yaxes(
        row=2,
        col=1,
        range=[scene_min, scene_max],
        title_text="Water level (m)",
        fixedrange=True,
    )
    for annotation in fig.layout.annotations:
        annotation.font = dict(color=PALETTE["ink"], size=12)
    return fig


@st.cache_data(show_spinner=False, max_entries=8)
def build_noaa_station_map(stations: tuple, selected_station_id: str):
    """Map the complete active NOAA catalog and highlight one station."""

    if not _HAS_PLOTLY:
        raise RuntimeError("Plotly is required for the nationwide station map")
    selected = [station.station_id == selected_station_id for station in stations]
    colors = [PALETTE["orange"] if active else PALETTE["blue"] for active in selected]
    sizes = [13 if active else 6 for active in selected]
    labels = [
        (
            f"<b>{escape(station.name)}</b><br>{escape(station.state)} · "
            f"{escape(station.station_id)}<br>"
            f"{'Tide predictions' if station.has_tide_predictions else 'Observation only'}"
        )
        for station in stations
    ]
    figure = go.Figure(go.Scattermap(
        lat=[station.latitude for station in stations],
        lon=[station.longitude for station in stations],
        mode="markers",
        marker=dict(size=sizes, color=colors, opacity=0.78),
        text=labels,
        hovertemplate="%{text}<extra></extra>",
    ))
    figure.update_layout(
        height=410,
        margin=dict(t=42, r=8, b=8, l=8),
        paper_bgcolor=PALETTE["surface"],
        map=dict(
            style="open-street-map",
            center=dict(lat=38.5, lon=-98.0),
            zoom=2.15,
        ),
        title=dict(
            text="Active NOAA water-level network",
            x=0.01,
            font=dict(size=16, color=PALETTE["ink"]),
        ),
        showlegend=False,
    )
    return figure


def style_figure(fig, *, title: str, subtitle: str, height: int = 360):
    """Apply one restrained chart system across the dashboard."""

    fig.update_layout(
        title={
            "text": f"{escape(title)}<br><sup>{escape(subtitle)}</sup>",
            "x": 0,
            "xanchor": "left",
            "y": 0.96,
            "yanchor": "top",
            "font": {"size": 17, "color": PALETTE["ink"]},
        },
        height=height,
        margin=dict(t=102, r=20, b=48, l=58),
        paper_bgcolor=PALETTE["surface"],
        plot_bgcolor=PALETTE["surface"],
        font=dict(family="Inter, ui-sans-serif, system-ui", color=PALETTE["ink"]),
        hoverlabel=dict(bgcolor=PALETTE["ink"], font_color="#FFFFFF"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(color=PALETTE["ink"], size=11),
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor=PALETTE["grid"],
        tickfont=dict(color=PALETTE["muted"]),
        title_font=dict(color=PALETTE["muted"]),
    )
    fig.update_yaxes(
        gridcolor=PALETTE["grid"],
        zerolinecolor=PALETTE["slate"],
        linecolor=PALETTE["grid"],
        tickfont=dict(color=PALETTE["muted"]),
        title_font=dict(color=PALETTE["muted"]),
    )
    return fig


def apply_dashboard_style() -> None:
    """Style native Streamlit surfaces as a compact control panel."""

    st.markdown(
        """
        <style>
        :root {
            --wai-ink: #172033;
            --wai-muted: #64748B;
            --wai-line: #E2E8F0;
            --wai-panel: #FFFFFF;
            --wai-canvas: #F4F7FB;
            --wai-blue: #2563EB;
            --wai-gold: #D08B18;
        }
        [data-testid="stAppViewContainer"] { background: var(--wai-canvas); }
        [data-testid="stHeader"] {
            background: transparent !important;
            border: 0 !important;
            height: 0 !important;
            min-height: 0 !important;
        }
        [data-testid="stAppDeployButton"],
        [data-testid="stMainMenu"],
        [data-testid="stToolbarActions"] {
            display: none !important;
        }
        [data-testid="stSidebar"] {
            background: #232630;
            border-right: 1px solid #343947;
        }
        [data-testid="stSidebar"] h1 {
            color: #F8FAFC !important;
        }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
            color: #BFCADF !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            background: #292E39;
            border-color: #3B4658;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary p,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary span {
            color: #E2E8F0 !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] p,
        [data-testid="stSidebar"] [data-testid="stExpander"] li {
            color: #CBD5E1 !important;
        }
        [data-testid="stSidebar"] hr {
            border-color: #3B4658 !important;
        }
        .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1500px; }
        .wai-hero {
            background: #172033;
            color: #FFFFFF;
            border: 1px solid #27344C;
            border-radius: 16px;
            padding: 1.15rem 1.35rem;
            margin-bottom: .8rem;
        }
        .wai-eyebrow { color: #AFC8FF; font-size: .72rem; font-weight: 700; letter-spacing: .12em; }
        .wai-hero h1 { color: #FFFFFF; font-size: 1.7rem; margin: .22rem 0 .3rem; }
        .wai-hero p { color: #CBD5E1; margin: 0; max-width: 850px; }
        .wai-status-row { display: flex; gap: .45rem; flex-wrap: wrap; margin-top: .8rem; }
        .wai-chip {
            border: 1px solid #41516D;
            border-radius: 999px;
            color: #E2E8F0;
            font-size: .75rem;
            padding: .28rem .58rem;
            flex: 0 0 auto;
            white-space: nowrap;
        }
        .wai-kpi-grid {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: .7rem;
            margin: .8rem 0 1rem;
        }
        .wai-kpi-card {
            background: var(--wai-panel);
            border: 1px solid var(--wai-line);
            border-radius: 12px;
            padding: .85rem 1rem;
            min-width: 0;
            transition: transform 110ms ease, border-color 110ms ease,
                        box-shadow 110ms ease;
            will-change: transform;
        }
        .wai-kpi-card:hover {
            transform: translateY(-1px);
            border-color: #B8C7DB;
            box-shadow: 0 5px 14px rgba(23, 32, 51, .07);
        }
        .wai-kpi-label { color: #536680; font-size: .76rem; min-height: 2.2em; }
        .wai-kpi-value {
            color: var(--wai-ink);
            font-size: clamp(1.25rem, 1.7vw, 1.8rem);
            line-height: 1.15;
            margin: .35rem 0 .42rem;
            white-space: nowrap;
        }
        .wai-kpi-meta { color: var(--wai-muted); font-size: .72rem; }
        [data-testid="stMetric"] {
            background: var(--wai-panel);
            border: 1px solid var(--wai-line);
            border-radius: 12px;
            padding: .8rem 1rem;
            min-height: 108px;
            transition: transform 110ms ease, border-color 110ms ease,
                        box-shadow 110ms ease;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-1px);
            border-color: #B8C7DB;
            box-shadow: 0 5px 14px rgba(23, 32, 51, .07);
        }
        [data-testid="stMetricLabel"] { color: var(--wai-muted); }
        [data-testid="stMetricValue"] { color: var(--wai-ink); }
        [data-testid="stPlotlyChart"] {
            background: var(--wai-panel);
            border: 1px solid var(--wai-line);
            border-radius: 12px;
            overflow: hidden;
        }
        [data-testid="stWidgetLabel"] p { color: var(--wai-ink) !important; }
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
            color: #E2E8F0 !important;
        }
        .stTabs [data-baseweb="tab-list"] { gap: .3rem; overflow-x: auto; }
        .stTabs [data-baseweb="tab"] {
            background: #E8EEF7;
            border: 1px solid #D6DEEA;
            border-radius: 8px;
            color: #334155 !important;
            font-weight: 650;
            padding: .42rem .62rem;
            transition: color 100ms ease, background-color 100ms ease,
                        border-color 100ms ease, transform 100ms ease;
        }
        .stTabs [data-baseweb="tab"]:active { transform: scale(.98); }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: #FFFFFF;
            border-color: #AFC5EF;
            color: #174FB7 !important;
        }
        button[data-baseweb="tab"] p { color: #334155 !important; }
        button[data-baseweb="tab"][aria-selected="true"] p {
            color: #174FB7 !important;
        }
        [data-testid="stTab"], button[role="tab"] {
            background: #E8EEF7 !important;
            border: 1px solid #D6DEEA !important;
            border-radius: 8px !important;
            color: #334155 !important;
            font-weight: 650 !important;
            padding: .42rem .62rem !important;
            transition: color 100ms ease, background-color 100ms ease,
                        border-color 100ms ease, transform 100ms ease !important;
        }
        [data-testid="stTab"]:active,
        button[role="tab"]:active { transform: scale(.98); }
        [data-testid="stTab"] p, button[role="tab"] p {
            color: #334155 !important;
            font-weight: 650 !important;
        }
        [data-testid="stTab"][aria-selected="true"],
        button[role="tab"][aria-selected="true"] {
            background: #FFFFFF !important;
            border-color: #AFC5EF !important;
            color: #174FB7 !important;
        }
        [data-testid="stTab"][aria-selected="true"] p,
        button[role="tab"][aria-selected="true"] p {
            color: #174FB7 !important;
        }
        [data-baseweb="tab-highlight"] {
            background-color: var(--wai-blue) !important;
        }
        .stButton button {
            transition: transform 90ms ease, box-shadow 100ms ease,
                        border-color 100ms ease, background-color 100ms ease;
        }
        .stButton button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 10px rgba(23, 32, 51, .10);
        }
        .stButton button:active {
            transform: translateY(0) scale(.985);
            box-shadow: none;
        }
        [data-baseweb="select"] > div {
            transition: border-color 100ms ease, box-shadow 100ms ease;
        }
        [data-baseweb="select"] > div:focus-within {
            border-color: #7EA5F5 !important;
            box-shadow: 0 0 0 2px rgba(37, 99, 235, .12);
        }
        @media (prefers-reduced-motion: reduce) {
            .wai-kpi-card,
            [data-testid="stMetric"],
            .stButton button,
            .stTabs [data-baseweb="tab"],
            [data-testid="stTab"],
            button[role="tab"],
            [data-baseweb="select"] > div {
                transition: none !important;
                transform: none !important;
            }
        }
        @media (max-width: 1100px) {
            [data-testid="stSidebar"] {
                min-width: 240px !important;
                max-width: 240px !important;
            }
            [data-testid="stSidebar"] > div { width: 240px !important; }
            .block-container {
                padding-left: 1.1rem !important;
                padding-right: 1.1rem !important;
            }
            .wai-hero h1 { font-size: 1.45rem; }
            .wai-kpi-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
            .stTabs [data-baseweb="tab"],
            [data-testid="stTab"],
            button[role="tab"] {
                font-size: .75rem;
                padding: .36rem .48rem !important;
            }
            .stTabs [role="tablist"] {
                flex-wrap: wrap !important;
                overflow: visible !important;
            }
        }
        @media (max-width: 700px) {
            .wai-kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_live_noaa_dashboard(
    station,
    catalog,
    *,
    lookback_hours: int,
    datum: str,
    guidance_mode: str,
) -> None:
    """Render a source-honest NOAA monitor without entering the model backtest."""

    from src.data.loader import NOAA_API_URL

    use_ofs = guidance_mode == "NOAA OFS guidance"
    include_tide_predictions = station.has_tide_predictions
    eyebrow = "LIVE NOAA OPERATIONAL GUIDANCE" if use_ofs else "LIVE PUBLIC DATA MONITOR"
    hero_description = (
        (
            "Compare NOAA CO-OPS observations, astronomical tide, and NOAA OFS "
            "water-level guidance on one station-aligned UTC timeline."
            if include_tide_predictions
            else
            "Compare NOAA CO-OPS observations with NOAA OFS water-level guidance "
            "on one station-aligned UTC timeline."
        )
        if use_ofs
        else (
            "Monitor NOAA CO-OPS six-minute observations from this active station. "
            "This station does not publish an astronomical tide baseline."
            if not include_tide_predictions
            else
            "Compare NOAA CO-OPS six-minute water-level observations with NOAA's "
            "astronomical tide prediction on the same station, datum, units, and UTC timeline."
        )
    )
    guidance_chip = (
        '<span class="wai-chip">NOAA OFS · up to 48h</span>'
        if use_ofs
        else (
            '<span class="wai-chip">Astronomical tide</span>'
            if include_tide_predictions
            else '<span class="wai-chip">Observations only</span>'
        )
    )
    station_heading = f"{station.name}, {station.state}"
    st.markdown(
        f"""
        <div class="wai-hero">
          <div class="wai-eyebrow">{escape(eyebrow)}</div>
          <h1>{escape(station_heading)} water-level control</h1>
          <p>{escape(hero_description)}</p>
          <div class="wai-status-row">
            <span class="wai-chip">NOAA station {escape(station.station_id)}</span>
            <span class="wai-chip">{escape(datum)} datum</span>
            <span class="wai-chip">Metric · UTC</span>
            {guidance_chip}
            <span class="wai-chip">{catalog.count} active stations</span>
            <span class="wai-chip">No mock fallback</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if use_ofs:
        st.info(
            "**NOAA OFS guidance is selected.** Public hydrodynamic model guidance; "
            "not a Wai machine-learning forecast or safety alert. Station availability varies."
        )
    elif include_tide_predictions:
        st.info(
            "Showing public NOAA observations with astronomical tide. Select "
            "**NOAA OFS guidance** for station-supported model guidance up to 48 hours ahead."
        )
    else:
        st.info(
            "**Observation-only station.** NOAA does not list an astronomical tide "
            "prediction for this gauge. Water levels remain live and use the station's "
            "compatible datum."
        )

    with st.expander("New to this view? Read it in 60 seconds", expanded=False):
        st.markdown(
            "**Observed level** is the station measurement. **Astronomical tide** is "
            "NOAA's harmonic baseline where available. **OFS guidance** is NOAA model "
            "output and is not a safety alert. **Observed minus guidance** shows how far "
            "the measurement is above or below the selected baseline. All values are "
            "meters on the selected vertical datum and all times are UTC."
        )

    with st.expander(
        f"Nationwide catalog · {catalog.count} active stations",
        expanded=False,
    ):
        st.caption(
            f"{catalog.tide_prediction_count} stations publish tide predictions · "
            f"{catalog.great_lakes_count} Great Lakes stations · "
            f"{len(catalog.regions)} regions · catalog source: {catalog.source}."
        )
        if _HAS_PLOTLY:
            station_map = build_noaa_station_map(
                catalog.stations,
                station.station_id,
            )
            st.plotly_chart(station_map, width="stretch", config=PLOTLY_CONFIG)

    refresh_col, status_col = st.columns([0.28, 0.72], vertical_alignment="center")
    if refresh_col.button("Refresh NOAA data", type="primary", width="stretch"):
        load_live_noaa_snapshot.clear()
        load_live_noaa_guidance.clear()
    status_col.caption(
        "Automatic cache: 5 minutes · refresh requests NOAA `water_level`"
        + (", `predictions`" if include_tide_predictions else "")
        + (", and `ofs_water_level`." if use_ofs else ".")
    )

    try:
        with st.spinner("Connecting to NOAA CO-OPS …"):
            snapshot = load_live_noaa_snapshot(
                station,
                lookback_hours,
                datum,
                include_tide_predictions,
            )
    except Exception as error:
        st.error(
            "NOAA CO-OPS could not be reached, so no live values are displayed. "
            f"The dashboard did not substitute demo data. Details: {error}"
        )
        st.caption(
            f"Request endpoint: `{NOAA_API_URL}` · station {station.station_id} · "
            f"datum {datum} · metric · GMT"
        )
        return

    guidance = None
    guidance_error = None
    if use_ofs:
        try:
            with st.spinner("Loading NOAA OFS operational guidance …"):
                guidance = load_live_noaa_guidance(
                    station,
                    min(lookback_hours, 24 * 7),
                    datum,
                )
        except Exception as error:
            guidance_error = str(error)
            st.warning(
                "NOAA does not currently return OFS water-level guidance for this "
                "station and datum. The observation monitor remains visible below and "
                "no replacement data was generated. "
                f"NOAA response: {guidance_error}"
            )

    frame = snapshot.frame
    paired = frame.dropna(subset=["observed_m", "predicted_m"]).copy()
    observed = frame.dropna(subset=["observed_m"]).copy()
    age_minutes = max(
        0.0,
        (snapshot.as_of - snapshot.latest_observed_at).total_seconds() / 60,
    )
    freshness_label = "Current" if age_minutes <= 30 else "Delayed"
    latest_qc = str(observed.iloc[-1].get("qc_status", "unknown")).lower()
    latest_qc = {"p": "preliminary", "v": "verified"}.get(latest_qc, latest_qc)
    observed_range = float(observed["observed_m"].max() - observed["observed_m"].min())

    comparison = pd.DataFrame(
        columns=["timestamp", "observed_m", "residual_m"]
    )
    comparison_name = "selected baseline"
    comparison_short = "baseline"
    second_kpi = (
        "Data layer",
        "Observed only",
        "no astronomical tide product at this station",
    )
    third_kpi = (
        "Baseline departure",
        "—",
        "select OFS guidance when NOAA supports it",
    )
    if not paired.empty:
        comparison = paired[["timestamp", "observed_m", "residual_m"]].copy()
        comparison_name = "astronomical tide"
        comparison_short = "tide"
        second_kpi = (
            "NOAA tide prediction",
            f"{snapshot.latest_prediction_m:.3f} m",
            "same timestamp and datum",
        )
        third_kpi = (
            "Observed − predicted",
            f"{snapshot.latest_residual_m:+.3f} m",
            "positive means observed is higher",
        )
    if guidance is not None:
        ofs_comparison = pd.merge(
            observed[["timestamp", "observed_m"]],
            guidance.frame,
            on="timestamp",
            how="inner",
        ).dropna(subset=["observed_m", "guidance_m"])
        ofs_comparison["residual_m"] = (
            ofs_comparison["observed_m"] - ofs_comparison["guidance_m"]
        )
        if not ofs_comparison.empty:
            comparison = ofs_comparison
            latest_comparison = comparison.iloc[-1]
            comparison_name = "OFS guidance"
            comparison_short = "OFS"
            second_kpi = (
                "NOAA OFS guidance",
                f"{float(latest_comparison['guidance_m']):.3f} m",
                f"aligned at {latest_comparison['timestamp']:%b %d %H:%M} UTC",
            )
            third_kpi = (
                "Observed − OFS",
                f"{float(latest_comparison['residual_m']):+.3f} m",
                "model departure at latest aligned sample",
            )
    if comparison.empty:
        quality_kpi = (
            "Samples in window",
            f"{len(observed):,}",
            "live six-minute NOAA observations",
        )
    else:
        comparison_mae = float(comparison["residual_m"].abs().mean())
        comparison_coverage = len(comparison) / max(len(observed), 1)
        quality_kpi = (
            "Aligned data quality",
            f"{comparison_coverage:.1%}",
            f"{len(comparison):,} pairs · {comparison_short} MAE {comparison_mae:.3f} m",
        )

    kpis = [
        (
            "Latest observed level",
            f"{snapshot.latest_observed_m:.3f} m",
            f"{snapshot.latest_observed_at:%b %d %H:%M} UTC · {latest_qc}",
        ),
        second_kpi,
        third_kpi,
        (
            "Observation age",
            f"{age_minutes:.0f} min",
            f"{freshness_label} at {snapshot.as_of:%H:%M} UTC",
        ),
        (
            f"{lookback_hours}h observed range",
            f"{observed_range:.3f} m",
            f"low {observed['observed_m'].min():.3f} · high {observed['observed_m'].max():.3f}",
        ),
        quality_kpi,
    ]
    kpi_html = "".join(
        "<div class='wai-kpi-card'>"
        f"<div class='wai-kpi-label'>{escape(label)}</div>"
        f"<div class='wai-kpi-value'>{escape(value)}</div>"
        f"<div class='wai-kpi-meta'>{escape(meta)}</div>"
        "</div>"
        for label, value, meta in kpis
    )
    st.markdown(f"<div class='wai-kpi-grid'>{kpi_html}</div>", unsafe_allow_html=True)

    if _HAS_PLOTLY:
        from plotly.subplots import make_subplots

        has_comparison = not comparison.empty
        if has_comparison:
            monitor_fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                row_heights=[0.68, 0.32],
                vertical_spacing=0.12,
                subplot_titles=(
                    "Water level on a common NOAA datum",
                    f"Observed minus {comparison_name}",
                ),
            )
        else:
            monitor_fig = make_subplots(
                rows=1,
                cols=1,
                subplot_titles=("Measured water level",),
            )
        monitor_fig.add_trace(go.Scatter(
            x=frame["timestamp"],
            y=frame["observed_m"],
            name="NOAA observed",
            line=dict(color=PALETTE["ink"], width=1.8),
            hovertemplate="%{x|%b %d %H:%M}<br>Observed %{y:.3f} m<extra></extra>",
        ), row=1, col=1)
        if not paired.empty:
            monitor_fig.add_trace(go.Scatter(
                x=frame["timestamp"],
                y=frame["predicted_m"],
                name="NOAA tide prediction",
                line=dict(color=PALETTE["gold"], width=2, dash="dash"),
                hovertemplate=(
                    "%{x|%b %d %H:%M}<br>Prediction %{y:.3f} m<extra></extra>"
                ),
            ), row=1, col=1)
        if guidance is not None:
            monitor_fig.add_trace(go.Scatter(
                x=guidance.frame["timestamp"],
                y=guidance.frame["guidance_m"],
                name="NOAA OFS guidance",
                line=dict(color=PALETTE["blue"], width=2.2),
                hovertemplate=(
                    "%{x|%b %d %H:%M}<br>OFS guidance %{y:.3f} m<extra></extra>"
                ),
            ), row=1, col=1)
        if has_comparison:
            monitor_fig.add_trace(go.Scatter(
                x=comparison["timestamp"],
                y=comparison["residual_m"],
                name="Residual",
                line=dict(color=PALETTE["blue"], width=1.6),
                fill="tozeroy",
                fillcolor=PALETTE["blue_light"],
                hovertemplate=(
                    "%{x|%b %d %H:%M}<br>Residual %{y:+.3f} m<extra></extra>"
                ),
                showlegend=False,
            ), row=2, col=1)
            monitor_fig.add_hline(
                y=0,
                line_color=PALETTE["slate"],
                line_width=1,
                row=2,
                col=1,
            )
        style_figure(
            monitor_fig,
            title=(
                "Live NOAA observation + OFS guidance"
                if guidance is not None
                else "Live NOAA observation monitor"
            ),
            subtitle=(
                f"{station.label} · last {lookback_hours} hours · {datum} · "
                "meters · UTC"
                + (" · guidance extends up to 48h" if guidance is not None else "")
            ),
            height=580 if has_comparison else 430,
        )
        monitor_fig.update_layout(
            legend=dict(orientation="h", y=1.04, x=0),
            hovermode="x unified",
        )
        monitor_fig.update_xaxes(
            title_text="Time (UTC)",
            row=2 if has_comparison else 1,
            col=1,
        )
        monitor_fig.update_yaxes(title_text="Water level (m)", row=1, col=1)
        if has_comparison:
            monitor_fig.update_yaxes(title_text="Residual (m)", row=2, col=1)
        st.plotly_chart(monitor_fig, width="stretch", config=PLOTLY_CONFIG)

        if not paired.empty:
            tide_fig = build_live_noaa_tide_figure(frame, datum=datum)
            st.plotly_chart(tide_fig, width="stretch", config=PLOTLY_CONFIG)
            st.caption(
                "The shoreline is schematic, not surveyed bathymetry. Its vertical level "
                "and motion are driven by the aligned NOAA observation series; the dashed "
                "history is NOAA's astronomical tide prediction."
            )
        else:
            st.caption(
                "The tide-motion cross-section is hidden because this station does not "
                "publish an astronomical tide baseline. The measured series above remains live."
            )
    else:
        fallback_chart = frame.set_index("timestamp")[["observed_m", "predicted_m"]]
        if guidance is not None:
            fallback_chart = fallback_chart.join(
                guidance.frame.set_index("timestamp")[["guidance_m"]],
                how="outer",
            )
        st.line_chart(fallback_chart.dropna(axis=1, how="all"))

    st.subheader("Latest aligned records")
    audit = observed[["timestamp", "observed_m", "qc_status"]].copy()
    if not paired.empty:
        audit = pd.merge(
            audit,
            paired[["timestamp", "predicted_m", "residual_m"]],
            on="timestamp",
            how="left",
        )
    if guidance is not None:
        audit = pd.merge(
            audit,
            guidance.frame,
            on="timestamp",
            how="left",
        )
        audit["ofs_residual_m"] = audit["observed_m"] - audit["guidance_m"]
    audit = audit.tail(20).sort_values("timestamp", ascending=False)
    audit["timestamp"] = audit["timestamp"].dt.strftime("%Y-%m-%d %H:%M UTC")
    audit = audit.rename(columns={
        "timestamp": "Timestamp",
        "observed_m": "Observed (m)",
        "predicted_m": "NOAA prediction (m)",
        "residual_m": "Observed − predicted (m)",
        "qc_status": "NOAA QC",
    })
    if guidance is not None:
        audit = audit.rename(columns={
            "guidance_m": "NOAA OFS guidance (m)",
            "ofs_residual_m": "Observed − OFS (m)",
        })
        audit_columns = [
            "Timestamp",
            "Observed (m)",
            "NOAA OFS guidance (m)",
            "Observed − OFS (m)",
            "NOAA QC",
        ]
        audit_format = {
            "Observed (m)": "{:.3f}",
            "NOAA OFS guidance (m)": "{:.3f}",
            "Observed − OFS (m)": "{:+.3f}",
        }
        if not paired.empty:
            audit_columns.insert(2, "NOAA prediction (m)")
            audit_format["NOAA prediction (m)"] = "{:.3f}"
    elif not paired.empty:
        audit_columns = [
            "Timestamp",
            "Observed (m)",
            "NOAA prediction (m)",
            "Observed − predicted (m)",
            "NOAA QC",
        ]
        audit_format = {
            "Observed (m)": "{:.3f}",
            "NOAA prediction (m)": "{:.3f}",
            "Observed − predicted (m)": "{:+.3f}",
        }
    else:
        audit_columns = [
            "Timestamp",
            "Observed (m)",
            "NOAA QC",
        ]
        audit_format = {
            "Observed (m)": "{:.3f}",
        }
    st.dataframe(
        audit[audit_columns].style.format(audit_format),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        f"Connected to the NOAA CO-OPS Data API at `{NOAA_API_URL}`. "
        f"Retrieved {snapshot.retrieved_at:%Y-%m-%d %H:%M:%S} UTC; "
        f"catalog={catalog.source}; units=metric; time_zone=gmt; cadence=6 minutes."
    )


# ── Sidebar ──────────────────────────────────────────────────────────────────

apply_dashboard_style()

with st.sidebar:
    st.title("Wai Control")
    st.caption("Coastal estimate console · research mode")
    with st.expander("Start here · beginner guide", expanded=False):
        st.markdown(
            "1. Choose **Live NOAA CO-OPS** for current public measurements.\n"
            "2. Filter a region, then type a station name or ID.\n"
            "3. Keep the suggested datum unless you know another reference is required.\n"
            "4. Start with astronomical tide; add OFS guidance only where NOAA supports it.\n\n"
            "Full setup, terminology, and troubleshooting: `docs/getting_started.md`."
        )
    st.divider()

    source_mode = st.selectbox(
        "Data source",
        ["Synthetic backtest", "Live NOAA CO-OPS"],
        help="Live NOAA is displayed separately from Wai's held-out model evidence.",
    )

    if source_mode == "Live NOAA CO-OPS":
        try:
            with st.spinner("Discovering active NOAA stations …"):
                live_catalog = load_live_noaa_station_catalog()
        except Exception as error:
            st.error(
                "The NOAA station catalog and bundled snapshot are unavailable. "
                f"Run `make noaa-stations` to rebuild the catalog. Details: {error}"
            )
            st.stop()

        if live_catalog.warning:
            st.warning(
                "Using the bundled NOAA station snapshot because live station "
                "discovery is temporarily unavailable."
            )
        live_region = st.selectbox(
            "Region",
            ["All regions", *live_catalog.regions],
            help="Includes U.S. states, territories, and NOAA partner regions.",
        )
        visible_stations = [
            station
            for station in live_catalog.stations
            if live_region == "All regions" or station.state == live_region
        ]
        station_options = [station.station_id for station in visible_stations]
        station_labels = {station.station_id: station.label for station in visible_stations}
        default_station = "1612340" if "1612340" in station_options else station_options[0]
        station_id = st.selectbox(
            "NOAA station",
            station_options,
            index=station_options.index(default_station),
            format_func=station_labels.get,
            help="Type a station name or seven-digit ID to search this region.",
        )
        live_station = live_catalog.by_id[station_id]
        lookback_label = st.selectbox(
            "Monitor window",
            ["24 hours", "72 hours", "7 days"],
            index=1,
        )
        live_lookback_hours = {
            "24 hours": 24,
            "72 hours": 72,
            "7 days": 24 * 7,
        }[lookback_label]
        live_datum = st.selectbox(
            "Vertical datum",
            live_station.datum_options,
            help=(
                "Vertical reference for every displayed level. The first option is "
                "the station-type default."
            ),
        )
        guidance_options = (
            ["Astronomical tide", "NOAA OFS guidance"]
            if live_station.has_tide_predictions
            else ["Observations only", "NOAA OFS guidance"]
        )
        live_guidance_mode = st.selectbox(
            "Guidance layer",
            guidance_options,
            help=(
                "OFS is NOAA hydrodynamic model guidance and is only available "
                "at stations inside supported forecast-system domains."
            ),
        )
        st.divider()
        st.caption(
            f"{live_catalog.count} active stations · {len(visible_stations)} in filter"
        )
        st.caption("NOAA CO-OPS · six-minute data · metric · UTC")
    else:
        if not DEMO_DATA_PATH.exists():
            st.error(
                "Demo data not found. Run:\n\n"
                "`uv run python -m scripts.prepare_demo_data`"
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
            sub = sub_all[
                (sub_all["timestamp"] >= start_dt)
                & (sub_all["timestamp"] < end_dt)
            ]
        else:
            sub = sub_all.copy()

        st.divider()
        st.caption("Source: DEMO_SYNTHETIC · cadence: 6 min")
        st.caption("No real sensor feed or operational alerting.")


if source_mode == "Live NOAA CO-OPS":
    render_live_noaa_dashboard(
        live_station,
        live_catalog,
        lookback_hours=live_lookback_hours,
        datum=live_datum,
        guidance_mode=live_guidance_mode,
    )
    st.caption(
        "Wai · Live public NOAA CO-OPS display · no API key · no mock fallback."
    )
    st.stop()


# ── Tabs ─────────────────────────────────────────────────────────────────────

(
    tab_control,
    tab_overview,
    tab_forecasts,
    tab_comparison,
    tab_alerts,
    tab_uncertainty,
    tab_benchmark,
) = st.tabs([
    "Control Center",
    "Overview",
    "Forecasts",
    "Model Comparison",
    "Alerts",
    "Uncertainty",
    "Benchmark Results",
])


# ── Tab 1: Control Center ─────────────────────────────────────────────────────

with tab_control:
    summary = load_summary()
    evidence = load_evidence_status()
    generated_at = summary.get("run_metadata", {}).get("run_at", "unknown")
    generated_label = str(generated_at).replace("T", " ")[:19]
    freshness_label = "Evidence fresh" if evidence["fresh"] else "Evidence stale"
    fingerprint = evidence.get("fingerprint")
    fingerprint_label = fingerprint[:10] if fingerprint else "unverified"

    st.markdown(
        f"""
        <div class="wai-hero">
          <div class="wai-eyebrow">BACKTEST CONTROL PANEL</div>
          <h1>{escape(station_id)} estimate console</h1>
          <p>Inspect held-out water-level estimates, uncertainty, observed error, and
          accuracy by horizon from one operational view.</p>
          <div class="wai-status-row">
            <span class="wai-chip">Historical replay</span>
            <span class="wai-chip">Synthetic source</span>
            <span class="wai-chip">{escape(freshness_label)}</span>
            <span class="wai-chip">Evidence {escape(fingerprint_label)}</span>
            <span class="wai-chip">Generated {escape(generated_label)} UTC</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not evidence["fresh"]:
        st.error(
            "The checked-in estimates do not match the current source. "
            "Run `make demo` before treating the displayed metrics as current."
        )
    else:
        st.info(
            "These are accurate **held-out backtest estimates** for synthetic data, "
            "not a live ocean reading or safety forecast. Actual outcomes are shown "
            "beside estimates so accuracy remains directly auditable."
        )

    control_a, control_b, control_c = st.columns([1.25, 1, 1])
    selected_model_label = control_a.selectbox(
        "Estimate model",
        [MODEL_VIEW[key]["label"] for key in MODEL_VIEW],
        index=0,
        help="Switching models recomputes the accuracy and uncertainty cards from the same test split.",
    )
    selected_model = next(
        key for key, spec in MODEL_VIEW.items() if spec["label"] == selected_model_label
    )
    window_label = control_b.selectbox(
        "Chart window",
        ["24 hours", "72 hours", "7 days", "Full test split"],
        index=1,
    )
    window_hours = {
        "24 hours": 24,
        "72 hours": 72,
        "7 days": 24 * 7,
        "Full test split": None,
    }[window_label]
    show_control_baseline = control_c.toggle(
        "Show persistence baseline",
        value=True,
        help="Rolling 1-step baseline: the previous observed value.",
    )

    with st.spinner("Preparing aligned estimates …"):
        try:
            control_forecast = run_forecast(station_id)
            estimate_frame = build_estimate_frame(control_forecast, selected_model)
            estimate_window = window_estimates(estimate_frame, window_hours)
            estimate_summary = summarize_estimates(control_forecast, selected_model)
            control_ok = True
        except Exception as error:
            st.error(f"Could not prepare the estimate console: {error}")
            control_ok = False

    if control_ok:
        latest_time = estimate_summary["latest_timestamp"]
        kpis = [
            (
                "Latest held-out estimate",
                f"{estimate_summary['latest_estimate']:.3f} m",
                f"{latest_time:%b %d %H:%M} UTC",
            ),
            (
                "Observed outcome",
                f"{estimate_summary['latest_actual']:.3f} m",
                f"absolute error {estimate_summary['latest_absolute_error']:.3f} m",
            ),
            (
                "90% estimate range",
                f"{estimate_summary['latest_lower']:.3f}–{estimate_summary['latest_upper']:.3f} m",
                f"half-width ±{estimate_summary['interval_half_width']:.3f} m",
            ),
            (
                "Held-out MAE",
                f"{estimate_summary['mae']:.3f} m",
                "lower is better",
            ),
            (
                "Skill vs persistence",
                f"{estimate_summary['skill_vs_persistence']:.1%}",
                f"n={estimate_summary['n_samples']:,} test samples",
            ),
            (
                "Empirical coverage",
                f"{estimate_summary['coverage']:.1%}",
                "nominal target 90%",
            ),
        ]
        kpi_html = "".join(
            "<div class='wai-kpi-card'>"
            f"<div class='wai-kpi-label'>{escape(label)}</div>"
            f"<div class='wai-kpi-value'>{escape(value)}</div>"
            f"<div class='wai-kpi-meta'>{escape(meta)}</div>"
            "</div>"
            for label, value, meta in kpis
        )
        st.markdown(
            f"<div class='wai-kpi-grid'>{kpi_html}</div>",
            unsafe_allow_html=True,
        )

        if _HAS_PLOTLY:
            spec = MODEL_VIEW[selected_model]
            estimate_fig = go.Figure()
            estimate_fig.add_trace(go.Scatter(
                x=pd.concat([estimate_window["timestamp"], estimate_window["timestamp"][::-1]]),
                y=pd.concat([estimate_window["upper"], estimate_window["lower"][::-1]]),
                fill="toself",
                fillcolor=spec["fill"],
                line=dict(color="rgba(255,255,255,0)"),
                name="90% range",
                hoverinfo="skip",
            ))
            estimate_fig.add_trace(go.Scatter(
                x=estimate_window["timestamp"],
                y=estimate_window["actual"],
                name="Observed",
                line=dict(color=PALETTE["ink"], width=1.8),
                hovertemplate="%{x|%b %d %H:%M}<br>Observed %{y:.3f} m<extra></extra>",
            ))
            estimate_fig.add_trace(go.Scatter(
                x=estimate_window["timestamp"],
                y=estimate_window["estimate"],
                name=spec["label"],
                line=dict(color=spec["color"], width=2, dash="dash"),
                hovertemplate="%{x|%b %d %H:%M}<br>Estimate %{y:.3f} m<extra></extra>",
            ))
            if show_control_baseline:
                estimate_fig.add_trace(go.Scatter(
                    x=estimate_window["timestamp"],
                    y=estimate_window["persistence"],
                    name="Persistence",
                    line=dict(color=PALETTE["slate"], width=1, dash="dot"),
                    hovertemplate="%{x|%b %d %H:%M}<br>Baseline %{y:.3f} m<extra></extra>",
                ))
            estimate_fig.add_hline(
                y=control_forecast["train_threshold"],
                line_color=PALETTE["orange"],
                line_dash="dash",
                annotation_text="training threshold",
                annotation_position="bottom right",
            )
            estimate_fig.update_xaxes(title_text="Held-out time (UTC)")
            estimate_fig.update_yaxes(title_text="Water level (m)")
            style_figure(
                estimate_fig,
                title="Estimate replay",
                subtitle=(
                    f"{window_label} · observed solid · estimate dashed · "
                    "90% interval shaded"
                ),
                height=440,
            )
            st.plotly_chart(
                estimate_fig, width="stretch", config=PLOTLY_CONFIG
            )

            tide_motion_fig = build_tide_motion_figure(
                estimate_window,
                model_key=selected_model,
                alert_threshold=control_forecast["train_threshold"],
            )
            st.plotly_chart(
                tide_motion_fig,
                width="stretch",
                config=PLOTLY_CONFIG,
            )
            st.caption(
                "The cross-section is schematic—not surveyed bathymetry. "
                "Its vertical water levels, uncertainty range, reference lines, "
                "and synchronized time series are expressed in metres on the model datum."
            )

            error_col, distribution_col = st.columns([1.45, 1])
            with error_col:
                error_fig = go.Figure()
                error_fig.add_trace(go.Scatter(
                    x=estimate_window["timestamp"],
                    y=estimate_window["absolute_error"],
                    name="Point absolute error",
                    mode="lines",
                    line=dict(color=PALETTE["slate"], width=1),
                    opacity=0.45,
                ))
                error_fig.add_trace(go.Scatter(
                    x=estimate_window["timestamp"],
                    y=estimate_window["rolling_mae_6h"],
                    name="Rolling MAE (6h)",
                    line=dict(color=spec["color"], width=2.2),
                ))
                error_fig.add_hline(
                    y=estimate_summary["mae"],
                    line_color=PALETTE["ink"],
                    line_dash="dot",
                    annotation_text="test MAE",
                )
                error_fig.update_xaxes(title_text="Held-out time (UTC)")
                error_fig.update_yaxes(title_text="Absolute error (m)", rangemode="tozero")
                style_figure(
                    error_fig,
                    title="Estimate error over time",
                    subtitle="Point error with a six-hour rolling mean; lower is better",
                    height=350,
                )
                st.plotly_chart(
                    error_fig, width="stretch", config=PLOTLY_CONFIG
                )

            with distribution_col:
                residual_fig = go.Figure(go.Histogram(
                    x=estimate_frame["error"],
                    nbinsx=40,
                    marker=dict(
                        color=spec["fill"].replace("0.16", "0.70").replace("0.13", "0.70"),
                        line=dict(color=spec["color"], width=1),
                    ),
                    name="Estimate − observed",
                    hovertemplate="Error %{x:.3f} m<br>Count %{y}<extra></extra>",
                ))
                residual_fig.add_vline(
                    x=0,
                    line_color=PALETTE["ink"],
                    line_width=1.5,
                    annotation_text="zero error",
                )
                residual_fig.update_xaxes(title_text="Signed error (m)")
                residual_fig.update_yaxes(title_text="Test samples", rangemode="tozero")
                style_figure(
                    residual_fig,
                    title="Residual error distribution",
                    subtitle=f"All {len(estimate_frame):,} held-out samples; centered at zero is ideal",
                    height=350,
                )
                residual_fig.update_layout(showlegend=False)
                st.plotly_chart(
                    residual_fig, width="stretch", config=PLOTLY_CONFIG
                )

            model_frame = model_accuracy_frame(load_metrics(), station_id)
            horizon_frame = horizon_accuracy_frame(load_horizon_metrics(), station_id)
            rank_col, horizon_col = st.columns(2)

            with rank_col:
                if not model_frame.empty:
                    ranked = model_frame.sort_values("mae", ascending=False)
                    rank_fig = go.Figure(go.Bar(
                        x=ranked["mae"],
                        y=ranked["model"],
                        orientation="h",
                        marker=dict(color=PALETTE["blue"]),
                        text=[f"{value:.3f} m" for value in ranked["mae"]],
                        textposition="outside",
                        cliponaxis=False,
                        hovertemplate="%{y}<br>MAE %{x:.4f} m<extra></extra>",
                    ))
                    rank_fig.update_xaxes(title_text="Mean absolute error (m)", rangemode="tozero")
                    rank_fig.update_yaxes(title_text="")
                    style_figure(
                        rank_fig,
                        title="Model accuracy ranking",
                        subtitle="One-step held-out MAE · shortest bar is best",
                        height=360,
                    )
                    rank_fig.update_layout(showlegend=False)
                    st.plotly_chart(
                        rank_fig, width="stretch", config=PLOTLY_CONFIG
                    )

            with horizon_col:
                if not horizon_frame.empty:
                    horizon_fig = go.Figure()
                    horizon_colors = {
                        "persistence": PALETTE["slate"],
                        "harmonic_ridge": PALETTE["gold"],
                        "grad_boost": PALETTE["blue"],
                    }
                    for model_key in ["persistence", "harmonic_ridge", "grad_boost"]:
                        model_rows = horizon_frame[horizon_frame["model_key"] == model_key]
                        if model_rows.empty:
                            continue
                        horizon_fig.add_trace(go.Bar(
                            x=model_rows["horizon"],
                            y=model_rows["rmse"],
                            name=model_rows["model"].iloc[0],
                            marker=dict(color=horizon_colors[model_key]),
                            hovertemplate="%{x}<br>RMSE %{y:.4f} m<extra>%{fullData.name}</extra>",
                        ))
                    horizon_fig.update_xaxes(
                        title_text="Forecast horizon",
                        categoryorder="array",
                        categoryarray=["1step_6min", "6h", "12h", "24h"],
                    )
                    horizon_fig.update_yaxes(title_text="RMSE (m)", rangemode="tozero")
                    style_figure(
                        horizon_fig,
                        title="Accuracy by forecast horizon",
                        subtitle="Direct held-out forecasts · grouped bars because horizons are discrete",
                        height=360,
                    )
                    horizon_fig.update_layout(barmode="group")
                    st.plotly_chart(
                        horizon_fig, width="stretch", config=PLOTLY_CONFIG
                    )

        st.subheader("Recent estimate audit")
        audit_rows = estimate_frame.tail(12).copy()
        audit_rows["timestamp"] = audit_rows["timestamp"].dt.strftime("%Y-%m-%d %H:%M UTC")
        audit_rows["interval_status"] = np.where(
            audit_rows["inside_interval"], "Inside 90% range", "Outside 90% range"
        )
        audit_rows = audit_rows.rename(columns={
            "timestamp": "Timestamp",
            "actual": "Observed (m)",
            "estimate": "Estimate (m)",
            "lower": "Lower 90% (m)",
            "upper": "Upper 90% (m)",
            "absolute_error": "Absolute error (m)",
            "interval_status": "Interval check",
        })
        st.dataframe(
            audit_rows[[
                "Timestamp",
                "Observed (m)",
                "Estimate (m)",
                "Lower 90% (m)",
                "Upper 90% (m)",
                "Absolute error (m)",
                "Interval check",
            ]].style.format({
                "Observed (m)": "{:.3f}",
                "Estimate (m)": "{:.3f}",
                "Lower 90% (m)": "{:.3f}",
                "Upper 90% (m)": "{:.3f}",
                "Absolute error (m)": "{:.3f}",
            }),
            width="stretch",
            hide_index=True,
        )


# ── Tab 2: Overview ───────────────────────────────────────────────────────────

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
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
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
    st.header("Online 1-Step Forecasts vs Actual (held-out test set)")
    st.caption(
        "**Protocol:** online 1-step forecasts using the previous observed "
        "value as the most recent input. Train = first 75 % of each station's "
        "data; conformal calibration = last 15 % of training. Test = last 25 %. "
        "No meteorological forcing is supplied to any model, so storm surge is "
        "*not* modelled."
    )
    st.caption(
        "Longer-horizon forecasts (6 h / 12 h / 24 h) come from direct "
        "multi-horizon training and are listed in the **Model Comparison** tab."
    )

    horizon_choice = st.selectbox(
        "Horizon",
        options=["1step_6min", "6h", "12h", "24h"],
        index=0,
        help="1-step (6 min) plots online forecasts. Longer horizons jump "
             "to the direct-forecasting metrics table below.",
    )

    show_persist = st.checkbox(
        "Show rolling 1-step persistence baseline", value=True,
        help="pred[t] = observed[t−1]. Same baseline used in scripts/train_baseline.py.",
    )
    show_gradboost = st.checkbox("Show GradBoost forecast", value=True)
    show_intervals = st.checkbox("Show 90% conformal intervals (HarmonicRidge)", value=True)

    with st.spinner("Running forecasts …"):
        try:
            forecast = run_forecast(station_id)
            forecast_ok = True
        except Exception as e:
            st.error(f"Forecast failed: {e}")
            forecast_ok = False

    if forecast_ok and horizon_choice != "1step_6min":
        st.info(
            f"Horizon **{horizon_choice}** uses direct forecasting (separate "
            "model per horizon, no recursive feedback). Metrics are in the "
            "Model Comparison tab; the time-series plot below shows the "
            "1-step online forecasts."
        )

    if forecast_ok and _HAS_PLOTLY:
        fig2 = go.Figure()
        timestamps = forecast["timestamps"]
        actual = forecast["actual"]
        harmonic_pred = forecast["harmonic_pred"]
        gradboost_pred = forecast["gradboost_pred"]
        persist_pred = forecast["persistence_pred"]

        # Actual
        fig2.add_trace(go.Scatter(
            x=timestamps, y=actual,
            name="Actual (synthetic)", line=dict(color="#1f77b4", width=1.5),
        ))

        # HarmonicRidge
        if show_intervals:
            fig2.add_trace(go.Scatter(
                x=pd.concat([timestamps, timestamps[::-1]]),
                y=np.concatenate([forecast["harmonic_upper"], forecast["harmonic_lower"][::-1]]),
                fill="toself", fillcolor="rgba(44,160,44,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="HarmonicRidge 90% interval", showlegend=True,
            ))
        fig2.add_trace(go.Scatter(
            x=timestamps, y=harmonic_pred,
            name="HarmonicRidge (online 1-step)",
            line=dict(color="#2ca02c", width=1.5, dash="dash"),
        ))

        # GradBoost
        if show_gradboost:
            fig2.add_trace(go.Scatter(
                x=timestamps, y=gradboost_pred,
                name="GradBoost (online 1-step)",
                line=dict(color="#9467bd", width=1.5, dash="dot"),
            ))

        # Rolling 1-step persistence
        if show_persist:
            fig2.add_trace(go.Scatter(
                x=timestamps, y=persist_pred,
                name="Persistence (rolling 1-step)",
                line=dict(color="#d62728", width=1, dash="dot"),
            ))

        fig2.update_layout(
            xaxis_title="Time (UTC)", yaxis_title="Water Level (m)",
            height=400, margin=dict(t=30, b=40),
            legend=dict(orientation="h", y=1.02),
        )
        st.plotly_chart(fig2, width="stretch", config=PLOTLY_CONFIG)
    elif forecast_ok:
        st.line_chart(pd.DataFrame({
            "timestamp": forecast["timestamps"],
            "water_level": forecast["actual"],
        }).set_index("timestamp")["water_level"])
        st.info("Install plotly for richer charts: `pip install plotly`")


# ── Tab 3: Model Comparison ───────────────────────────────────────────────────

with tab_comparison:
    st.header("Model Performance — 1-Step (6 min) Horizon")
    st.caption("Metrics on held-out test set (synthetic demo data). All values in metres.")

    all_metrics = load_metrics()
    station_metrics = all_metrics.get(station_id, {})

    if station_metrics:
        from src.models.branding import DISPLAY_BY_KEY

        def _fmt(val):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "—"
            return f"{val:.4f}"

        rows = []
        for model_name, m in station_metrics.items():
            if not isinstance(m, dict) or "mae" not in m:
                continue
            rows.append({
                "Model": DISPLAY_BY_KEY.get(model_name, model_name),
                "MAE (m)": _fmt(m.get("mae")),
                "RMSE (m)": _fmt(m.get("rmse")),
                "R²": _fmt(m.get("r2")),
                "NSE": _fmt(m.get("nse")),
                "Corr": _fmt(m.get("corr")),
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
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
        def _hfmt(val):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "—"
            return f"{val:.4f}"

        h_rows = []
        for h_name, models in station_horizon.items():
            if h_name.startswith("_") or not isinstance(models, dict):
                continue
            for model_name, m in models.items():
                if model_name.startswith("_"):
                    continue
                if not isinstance(m, dict) or "mae" not in m:
                    note = m.get("note", "—") if isinstance(m, dict) else "—"
                    h_rows.append({"Horizon": h_name, "Model": model_name,
                                   "MAE (m)": "—", "RMSE (m)": "—", "R²": note})
                    continue
                h_rows.append({
                    "Horizon": h_name,
                    "Model": model_name,
                    "MAE (m)": _hfmt(m.get("mae")),
                    "RMSE (m)": _hfmt(m.get("rmse")),
                    "R²": _hfmt(m.get("r2")),
                })
        if h_rows:
            st.dataframe(pd.DataFrame(h_rows), width="stretch", hide_index=True)
    else:
        st.info("Run `python -m scripts.evaluate_horizons` then refresh.")


# ── Tab 4: Alerts ─────────────────────────────────────────────────────────────

with tab_alerts:
    st.header("High-Water Alert Detection")
    st.caption(
        "Threshold is fit on the **training window** (first 75 % of each "
        "station's series) — never on the displayed date range. Changing the "
        "sidebar date filter only changes what is plotted; the threshold "
        "stays anchored to the training climatology."
    )

    # Training window for the selected station — independent of the
    # displayed filter range.
    _train_full = sub_all.sort_values("timestamp")
    _n_train_alert = int(len(_train_full) * 0.75)
    train_series = _train_full["water_level"].iloc[:_n_train_alert]
    train_start = _train_full["timestamp"].iloc[0]
    train_end = _train_full["timestamp"].iloc[_n_train_alert - 1] if _n_train_alert > 0 else _train_full["timestamp"].iloc[-1]

    col_a, col_b, col_c = st.columns(3)
    alert_mode = col_a.selectbox(
        "Threshold mode",
        ["std", "absolute", "percentile"],
        index=0,
        help="std = mean + k·std · absolute = fixed value · percentile = p-th percentile",
    )
    k_val = col_b.slider("k (std multiplier)", 1.0, 4.0, 2.0, 0.25, disabled=(alert_mode != "std"))
    abs_default = float(train_series.mean() + 2 * train_series.std())
    abs_val = col_b.number_input(
        "Absolute threshold (m)",
        value=abs_default,
        disabled=(alert_mode != "absolute"),
        help="Default is mean + 2σ of the training window.",
    )
    pct_val = col_c.slider("Percentile", 50, 99, 95, disabled=(alert_mode != "percentile"))

    from src.alerts import AlertConfig, detect_alerts, compute_threshold

    if alert_mode == "std":
        config = AlertConfig(mode="std", k=k_val)
    elif alert_mode == "absolute":
        config = AlertConfig(mode="absolute", absolute_threshold=abs_val)
    else:
        config = AlertConfig(mode="percentile", percentile=float(pct_val))

    threshold = compute_threshold(train_series, config)
    alerts = detect_alerts(sub, config, reference_series=train_series)

    st.caption(
        f"Threshold source: training window for **{station_id}** · "
        f"{train_start:%Y-%m-%d} → {train_end:%Y-%m-%d} "
        f"(n_train={len(train_series):,}). "
        f"Mode={alert_mode}; threshold = **{threshold:.3f} m**."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Alert threshold (train-fit)", f"{threshold:.3f} m")
    col2.metric("Alerts in displayed range", len(alerts))
    col3.metric("Alert rate (displayed)", f"{100 * len(alerts) / max(len(sub), 1):.1f}%")

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
        st.plotly_chart(fig_a, width="stretch", config=PLOTLY_CONFIG)

    st.subheader(f"Alert Events (≥ {threshold:.2f} m)")
    if alerts.empty:
        st.success("No alert events in the selected date range.")
    else:
        display = alerts[["timestamp", "water_level"]].copy()
        display["timestamp"] = display["timestamp"].dt.strftime("%Y-%m-%d %H:%M UTC")
        display.columns = ["Timestamp (UTC)", "Water Level (m)"]
        st.dataframe(display.head(50), width="stretch", hide_index=True)


# ── Tab 5: Uncertainty ────────────────────────────────────────────────────────

with tab_uncertainty:
    st.header("Conformal Prediction Intervals")
    st.caption(
        "Split-conformal intervals with 90 % nominal coverage. qhat uses the "
        "exact k-th smallest calibration residual (k = ⌈0.9·(n+1)⌉) — the "
        "finite-sample bound that gives marginal coverage ≥ 90 % under "
        "exchangeability. Calibration = last 15 % of training; test = held-out "
        "25 %. Empirical coverage is reported on the **future test split** "
        "and broken out by event / non-event so users can see where coverage "
        "degrades."
    )

    with st.spinner("Computing conformal intervals …"):
        try:
            forecast = run_forecast(station_id)
            ci_ok = True
        except Exception as e:
            st.error(f"Could not compute intervals: {e}")
            ci_ok = False

    if ci_ok:
        timestamps = forecast["timestamps"]
        actual_test = forecast["actual"]
        h_preds_test = forecast["harmonic_pred"]
        gb_preds_test = forecast["gradboost_pred"]
        h_lo, h_hi = forecast["harmonic_lower"], forecast["harmonic_upper"]
        h_ci = forecast["harmonic_ci"]
        gb_ci = forecast["gradboost_ci"]
        h_report = forecast["harmonic_coverage"]
        gb_report = forecast["gradboost_coverage"]
        train_threshold = forecast["train_threshold"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("HarmonicRidge qhat", f"±{h_ci.qhat:.4f} m")
        col2.metric("HR overall coverage", f"{h_report['coverage_overall']:.1%}")
        col3.metric(
            "HR coverage · event",
            f"{h_report['coverage_event']:.1%}" if not np.isnan(h_report['coverage_event']) else "—",
        )
        col4.metric(
            "HR coverage · non-event",
            f"{h_report['coverage_non_event']:.1%}" if not np.isnan(h_report['coverage_non_event']) else "—",
        )

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("GradBoost qhat", f"±{gb_ci.qhat:.4f} m")
        col6.metric("GB overall coverage", f"{gb_report['coverage_overall']:.1%}")
        col7.metric(
            "GB coverage · event",
            f"{gb_report['coverage_event']:.1%}" if not np.isnan(gb_report['coverage_event']) else "—",
        )
        col8.metric(
            "GB coverage · non-event",
            f"{gb_report['coverage_non_event']:.1%}" if not np.isnan(gb_report['coverage_non_event']) else "—",
        )

        st.caption(
            f"Calibration size n_cal={h_ci.n_cal} · k={h_ci.k} · "
            f"event threshold (train-fit) = {train_threshold:.3f} m. "
            f"Event samples in test: {h_report['n_event_samples']} of "
            f"{h_report['n_samples']}. Nominal coverage = 90 %."
        )

        if _HAS_PLOTLY:
            fig_u = go.Figure()
            fig_u.add_trace(go.Scatter(
                x=timestamps, y=actual_test,
                name="Actual", line=dict(color="#1f77b4", width=1.5),
            ))
            fig_u.add_trace(go.Scatter(
                x=pd.concat([timestamps, timestamps[::-1]]),
                y=np.concatenate([h_hi, h_lo[::-1]]),
                fill="toself", fillcolor="rgba(44,160,44,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="HarmonicRidge 90% interval",
            ))
            fig_u.add_trace(go.Scatter(
                x=timestamps, y=h_preds_test,
                name="HarmonicRidge forecast", line=dict(color="#2ca02c", dash="dash"),
            ))
            fig_u.update_layout(
                xaxis_title="Time (UTC)", yaxis_title="Water Level (m)",
                height=380, margin=dict(t=30, b=40),
            )
            st.plotly_chart(fig_u, width="stretch", config=PLOTLY_CONFIG)

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

        def _bfmt(val):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "—"
            return f"{val:.4f}"

        h_rows = []
        for sid, horizons in h_data.items():
            if not isinstance(horizons, dict):
                continue
            for h_name, models in horizons.items():
                if h_name.startswith("_") or not isinstance(models, dict):
                    continue
                for model_name, m in models.items():
                    if model_name.startswith("_"):
                        continue
                    if not isinstance(m, dict) or "mae" not in m:
                        continue
                    h_rows.append({
                        "Station": sid,
                        "Horizon": h_name,
                        "Model": model_name,
                        "MAE (m)": _bfmt(m.get("mae")),
                        "RMSE (m)": _bfmt(m.get("rmse")),
                    })
        if h_rows:
            st.dataframe(pd.DataFrame(h_rows), width="stretch", hide_index=True)
    else:
        st.info("Run `python -m scripts.evaluate_horizons` then refresh.")

st.caption(
    "Wai · Demo data (DEMO_SYNTHETIC) · "
    "No private or proprietary data is used."
)
