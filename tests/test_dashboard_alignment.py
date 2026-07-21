"""Dashboard forecast alignment tests."""

from __future__ import annotations

import os
import re

import numpy as np

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

from app import (  # noqa: E402
    build_estimate_frame,
    build_tide_motion_figure,
    horizon_accuracy_frame,
    load_horizon_metrics,
    load_metrics,
    model_accuracy_frame,
    run_forecast,
    summarize_estimates,
    window_estimates,
)


def test_dashboard_forecast_arrays_align_exactly():
    forecast = run_forecast("DEMO-HNL")
    keys = [
        "timestamps",
        "actual",
        "persistence_pred",
        "harmonic_pred",
        "gradboost_pred",
        "harmonic_lower",
        "harmonic_upper",
        "gradboost_lower",
        "gradboost_upper",
    ]
    lengths = {key: len(forecast[key]) for key in keys}
    assert len(set(lengths.values())) == 1, lengths
    assert lengths["timestamps"] > 0
    assert forecast["timestamps"].is_monotonic_increasing
    assert np.all(forecast["harmonic_lower"] <= forecast["harmonic_pred"])
    assert np.all(forecast["harmonic_upper"] >= forecast["harmonic_pred"])
    assert forecast["harmonic_coverage"]["n_samples"] == lengths["actual"]
    assert forecast["gradboost_coverage"]["n_samples"] == lengths["actual"]


def test_control_panel_estimates_are_auditable_and_windowed():
    forecast = run_forecast("DEMO-HNL")
    frame = build_estimate_frame(forecast, "harmonic_ridge")
    summary = summarize_estimates(forecast, "harmonic_ridge")
    recent = window_estimates(frame, 72)

    assert len(frame) == len(forecast["actual"])
    np.testing.assert_allclose(frame["error"], frame["estimate"] - frame["actual"])
    np.testing.assert_allclose(frame["absolute_error"], np.abs(frame["error"]))
    assert frame["timestamp"].is_monotonic_increasing
    assert frame["inside_interval"].dtype == bool
    assert np.isclose(summary["mae"], frame["absolute_error"].mean())
    assert np.isclose(summary["coverage"], frame["inside_interval"].mean())
    assert summary["skill_vs_persistence"] > 0
    assert recent["timestamp"].max() - recent["timestamp"].min() <= np.timedelta64(72, "h")
    assert 8 < len(recent) < len(frame)


def test_control_panel_comparison_frames_preserve_metric_grain():
    models = model_accuracy_frame(load_metrics(), "DEMO-HNL")
    horizons = horizon_accuracy_frame(load_horizon_metrics(), "DEMO-HNL")

    assert len(models) >= 4
    assert models["mae"].is_monotonic_increasing
    assert models[["mae", "rmse", "r2"]].notna().all().all()
    assert set(horizons["horizon"]) == {"1step_6min", "6h", "12h", "24h"}
    assert set(horizons["model_key"]) >= {"persistence", "harmonic_ridge", "grad_boost"}
    assert horizons[["mae", "rmse"]].gt(0).all().all()


def test_tide_motion_tracks_estimates_and_exposes_native_controls():
    forecast = run_forecast("DEMO-HNL")
    recent = window_estimates(
        build_estimate_frame(forecast, "harmonic_ridge"), 72
    )
    figure = build_tide_motion_figure(
        recent,
        model_key="harmonic_ridge",
        alert_threshold=forecast["train_threshold"],
        max_frames=24,
    )

    assert len(figure.frames) == 24
    assert len(figure.data) == 15
    assert len(figure.layout.updatemenus) == 1
    assert len(figure.layout.sliders) == 1
    assert len(figure.layout.sliders[0].steps) == 24
    assert figure.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] == 110
    assert list(figure.frames[0].traces) == [0, 1, 4, 8, 12, 13, 14]
    assert np.isclose(float(figure.data[0].y[0]), recent.iloc[0]["estimate"])
    assert np.isclose(float(figure.data[13].y[0]), recent.iloc[0]["actual"])
    assert np.isclose(float(figure.data[14].y[0]), recent.iloc[0]["estimate"])
    assert "tide motion explorer" in figure.layout.title.text.lower()
    assert figure.layout.showlegend is False
    assert figure.data[8].textposition == "middle right"
    assert figure.data[8].mode == "text"
    assert figure.layout.xaxis.title.text is None
    assert np.isclose(float(figure.data[3].x[-3]), 50.0)
    assert list(figure.layout.xaxis.tickvals) == [5, 28, 78]
    status = str(figure.data[8].text[0])
    displayed_rate = float(re.search(r"([+-][0-9.]+) cm/h", status).group(1))
    assert abs(displayed_rate) < 100
