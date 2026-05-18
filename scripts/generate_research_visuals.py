"""Generate lightweight SVG figures for the Wai research report.

The figures are intentionally built from the same artifacts used by the
reports, with no extra plotting dependency. They are portfolio visuals, not
new evidence tracks.

Outputs
-------
    docs/images/actual_vs_predicted.svg
    docs/images/error_by_horizon.svg
    docs/images/baseline_comparison.svg
    docs/images/residual_plot.svg
"""

from __future__ import annotations

import html
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_demo_data
from src.models.baseline import HarmonicRidgeModel


ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"
IMAGES_DIR = ROOT / "docs" / "images"

COLORS = {
    "Actual": "#1f77b4",
    "Rolling persistence": "#d62728",
    "HarmonicRidge": "#2ca02c",
    "GradBoost": "#9467bd",
    "WaveGRU": "#8c564b",
    "Residual": "#2f4858",
}


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `make demo` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_num(v: float, ndigits: int = 3) -> str:
    if abs(v) >= 1:
        return f"{v:.2f}"
    return f"{v:.{ndigits}f}"


def _nice_range(values: Iterable[float], pad_frac: float = 0.08) -> tuple[float, float]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return 0.0, 1.0
    lo, hi = min(vals), max(vals)
    if lo == hi:
        delta = abs(lo) * 0.1 or 1.0
        return lo - delta, hi + delta
    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad


def _svg(width: int, height: int, title: str, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(title)}">\n'
        "  <style>\n"
        "    .title { font: 700 20px Arial, sans-serif; fill: #1f2933; }\n"
        "    .subtitle { font: 12px Arial, sans-serif; fill: #52606d; }\n"
        "    .axis { stroke: #7b8794; stroke-width: 1; }\n"
        "    .grid { stroke: #d9e2ec; stroke-width: 1; }\n"
        "    .tick { font: 11px Arial, sans-serif; fill: #52606d; }\n"
        "    .label { font: 12px Arial, sans-serif; fill: #334e68; }\n"
        "    .legend { font: 12px Arial, sans-serif; fill: #243b53; }\n"
        "  </style>\n"
        f"{body}\n"
        "</svg>\n"
    )


def _line_path(x: np.ndarray, y: np.ndarray) -> str:
    parts = []
    for i, (xx, yy) in enumerate(zip(x, y)):
        cmd = "M" if i == 0 else "L"
        parts.append(f"{cmd}{xx:.1f},{yy:.1f}")
    return " ".join(parts)


def _axis_y_ticks(y_min: float, y_max: float, n: int = 5) -> list[float]:
    if y_min == y_max:
        return [y_min]
    return [y_min + (y_max - y_min) * i / (n - 1) for i in range(n)]


def _line_chart(
    path: Path,
    title: str,
    subtitle: str,
    series: list[tuple[str, pd.Series | np.ndarray]],
    x_labels: tuple[str, str],
    y_label: str,
) -> None:
    width, height = 920, 420
    left, right, top, bottom = 70, 28, 64, 62
    plot_w = width - left - right
    plot_h = height - top - bottom

    max_len = max(len(vals) for _, vals in series)
    y_values = np.concatenate([np.asarray(vals, dtype=float) for _, vals in series])
    y_min, y_max = _nice_range(y_values)

    def sx(idx: np.ndarray) -> np.ndarray:
        denom = max(max_len - 1, 1)
        return left + (idx / denom) * plot_w

    def sy(vals: np.ndarray) -> np.ndarray:
        return top + (y_max - vals) / (y_max - y_min) * plot_h

    body = [
        f'  <rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'  <text x="{left}" y="30" class="title">{html.escape(title)}</text>',
        f'  <text x="{left}" y="49" class="subtitle">{html.escape(subtitle)}</text>',
    ]
    for tick in _axis_y_ticks(y_min, y_max):
        y = sy(np.array([tick]))[0]
        body.append(f'  <line x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        body.append(f'  <text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="tick">{_fmt_num(tick)}</text>')
    body.append(f'  <line x1="{left}" x2="{left}" y1="{top}" y2="{height-bottom}" class="axis"/>')
    body.append(f'  <line x1="{left}" x2="{width-right}" y1="{height-bottom}" y2="{height-bottom}" class="axis"/>')
    body.append(f'  <text x="{left}" y="{height-20}" class="tick">{html.escape(x_labels[0])}</text>')
    body.append(f'  <text x="{width-right}" y="{height-20}" text-anchor="end" class="tick">{html.escape(x_labels[1])}</text>')
    body.append(f'  <text x="18" y="{top + plot_h / 2:.1f}" class="label" transform="rotate(-90 18 {top + plot_h / 2:.1f})">{html.escape(y_label)}</text>')

    legend_x = width - right - 170
    legend_y = 28
    for i, (name, vals) in enumerate(series):
        arr = np.asarray(vals, dtype=float)
        idx = np.arange(len(arr))
        color = COLORS.get(name, "#111827")
        body.append(
            f'  <path d="{_line_path(sx(idx), sy(arr))}" fill="none" '
            f'stroke="{color}" stroke-width="2" stroke-linejoin="round"/>'
        )
        ly = legend_y + i * 18
        body.append(f'  <line x1="{legend_x}" x2="{legend_x+22}" y1="{ly}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        body.append(f'  <text x="{legend_x+28}" y="{ly+4}" class="legend">{html.escape(name)}</text>')

    path.write_text(_svg(width, height, title, "\n".join(body)), encoding="utf-8")


def _bar_chart(
    path: Path,
    title: str,
    subtitle: str,
    groups: list[tuple[str, dict[str, float]]],
    model_order: list[str],
    y_label: str,
) -> None:
    width, height = 920, 430
    left, right, top, bottom = 70, 28, 68, 82
    plot_w = width - left - right
    plot_h = height - top - bottom
    all_vals = [v for _, rec in groups for v in rec.values() if math.isfinite(float(v))]
    y_min, y_max = 0.0, max(all_vals) * 1.15 if all_vals else 1.0

    def sy(v: float) -> float:
        return top + (y_max - v) / (y_max - y_min) * plot_h

    body = [
        f'  <rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'  <text x="{left}" y="30" class="title">{html.escape(title)}</text>',
        f'  <text x="{left}" y="49" class="subtitle">{html.escape(subtitle)}</text>',
    ]
    for tick in _axis_y_ticks(y_min, y_max):
        y = sy(tick)
        body.append(f'  <line x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        body.append(f'  <text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="tick">{_fmt_num(tick)}</text>')
    body.append(f'  <line x1="{left}" x2="{left}" y1="{top}" y2="{height-bottom}" class="axis"/>')
    body.append(f'  <line x1="{left}" x2="{width-right}" y1="{height-bottom}" y2="{height-bottom}" class="axis"/>')
    body.append(f'  <text x="18" y="{top + plot_h / 2:.1f}" class="label" transform="rotate(-90 18 {top + plot_h / 2:.1f})">{html.escape(y_label)}</text>')

    group_w = plot_w / max(len(groups), 1)
    bar_w = min(44, group_w / (len(model_order) + 1.4))
    for gi, (label, rec) in enumerate(groups):
        center = left + group_w * (gi + 0.5)
        start = center - (bar_w * len(model_order)) / 2
        for mi, model in enumerate(model_order):
            val = rec.get(model)
            if val is None or not math.isfinite(float(val)):
                continue
            x = start + mi * bar_w
            y = sy(float(val))
            h = height - bottom - y
            color = COLORS.get(model, "#475569")
            body.append(f'  <rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.78:.1f}" height="{h:.1f}" fill="{color}"/>')
            body.append(f'  <text x="{x + bar_w*0.39:.1f}" y="{y-5:.1f}" text-anchor="middle" class="tick">{_fmt_num(float(val))}</text>')
        body.append(f'  <text x="{center:.1f}" y="{height-52}" text-anchor="middle" class="tick">{html.escape(label)}</text>')

    legend_x = left
    legend_y = height - 25
    for i, model in enumerate(model_order):
        x = legend_x + i * 150
        color = COLORS.get(model, "#475569")
        body.append(f'  <rect x="{x}" y="{legend_y-10}" width="14" height="14" fill="{color}"/>')
        body.append(f'  <text x="{x+20}" y="{legend_y+1}" class="legend">{html.escape(model)}</text>')

    path.write_text(_svg(width, height, title, "\n".join(body)), encoding="utf-8")


def _scatter_chart(
    path: Path,
    title: str,
    subtitle: str,
    x_vals: pd.Series | np.ndarray,
    y_vals: pd.Series | np.ndarray,
    x_label: str,
    y_label: str,
) -> None:
    width, height = 920, 420
    left, right, top, bottom = 70, 28, 64, 62
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)
    x_min, x_max = _nice_range(x_arr)
    y_min, y_max = _nice_range(y_arr)

    def sx(v: np.ndarray) -> np.ndarray:
        return left + (v - x_min) / (x_max - x_min) * plot_w

    def sy(v: np.ndarray) -> np.ndarray:
        return top + (y_max - v) / (y_max - y_min) * plot_h

    body = [
        f'  <rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'  <text x="{left}" y="30" class="title">{html.escape(title)}</text>',
        f'  <text x="{left}" y="49" class="subtitle">{html.escape(subtitle)}</text>',
    ]
    for tick in _axis_y_ticks(y_min, y_max):
        y = sy(np.array([tick]))[0]
        body.append(f'  <line x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        body.append(f'  <text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="tick">{_fmt_num(tick)}</text>')
    for tick in _axis_y_ticks(x_min, x_max):
        x = sx(np.array([tick]))[0]
        body.append(f'  <text x="{x:.1f}" y="{height-42}" text-anchor="middle" class="tick">{_fmt_num(tick)}</text>')
    if y_min < 0 < y_max:
        zero_y = sy(np.array([0.0]))[0]
        body.append(f'  <line x1="{left}" x2="{width-right}" y1="{zero_y:.1f}" y2="{zero_y:.1f}" stroke="#d62728" stroke-width="1.5" stroke-dasharray="5 5"/>')
    body.append(f'  <line x1="{left}" x2="{left}" y1="{top}" y2="{height-bottom}" class="axis"/>')
    body.append(f'  <line x1="{left}" x2="{width-right}" y1="{height-bottom}" y2="{height-bottom}" class="axis"/>')
    body.append(f'  <text x="{left + plot_w / 2:.1f}" y="{height-18}" text-anchor="middle" class="label">{html.escape(x_label)}</text>')
    body.append(f'  <text x="18" y="{top + plot_h / 2:.1f}" class="label" transform="rotate(-90 18 {top + plot_h / 2:.1f})">{html.escape(y_label)}</text>')

    x_scaled = sx(x_arr)
    y_scaled = sy(y_arr)
    for xx, yy in zip(x_scaled, y_scaled):
        body.append(f'  <circle cx="{xx:.1f}" cy="{yy:.1f}" r="2.3" fill="{COLORS["Residual"]}" opacity="0.45"/>')

    path.write_text(_svg(width, height, title, "\n".join(body)), encoding="utf-8")


def _rolling_persistence(train_series: pd.Series, test_series: pd.Series) -> np.ndarray:
    vals = test_series.to_numpy(dtype=float)
    preds = np.empty(len(vals), dtype=float)
    preds[0] = float(train_series.dropna().iloc[-1])
    if len(vals) > 1:
        preds[1:] = vals[:-1]
    return preds


def _prediction_frame(station_id: str = "DEMO-HNL", max_points: int = 120) -> pd.DataFrame:
    df = load_demo_data()
    sub = df[df["station_id"] == station_id].sort_values("timestamp").reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"Station {station_id!r} not found in demo data")
    n_train = int(len(sub) * 0.75)
    train = sub.iloc[:n_train].copy()
    test = sub.iloc[n_train:].copy()
    model = HarmonicRidgeModel(alpha=1.0).fit(train)
    aligned = model.predict_aligned(test)
    persistence = _rolling_persistence(train["water_level"], test["water_level"])
    aligned["rolling_persistence"] = persistence[aligned["_source_row"].to_numpy(dtype=int)]
    aligned["residual"] = aligned["actual"] - aligned["prediction"]
    return aligned.head(max_points).copy()


def _avg_model_metrics() -> dict[str, float]:
    metrics = _read_json(REPORTS_DIR / "model_metrics.json")
    out: dict[str, list[float]] = {
        "Rolling persistence": [],
        "HarmonicRidge": [],
        "GradBoost": [],
        "WaveGRU": [],
    }
    key_map = {
        "Rolling persistence": "persistence",
        "HarmonicRidge": "harmonic_ridge",
        "GradBoost": "grad_boost",
        "WaveGRU": "wave_gru",
    }
    for station_rec in metrics.values():
        if not isinstance(station_rec, dict):
            continue
        for display, key in key_map.items():
            rec = station_rec.get(key, {})
            if isinstance(rec, dict) and isinstance(rec.get("mae"), (int, float)):
                out[display].append(float(rec["mae"]))
    return {name: float(np.mean(vals)) for name, vals in out.items() if vals}


def _avg_horizon_metrics() -> list[tuple[str, dict[str, float]]]:
    metrics = _read_json(REPORTS_DIR / "horizon_metrics.json")
    model_keys = {
        "Rolling persistence": "persistence",
        "HarmonicRidge": "harmonic_ridge",
        "GradBoost": "grad_boost",
    }
    labels = {
        "1step_6min": "1-step",
        "6h": "6 h",
        "12h": "12 h",
        "24h": "24 h",
    }
    groups: list[tuple[str, dict[str, float]]] = []
    for horizon_key in ("1step_6min", "6h", "12h", "24h"):
        rec: dict[str, list[float]] = {name: [] for name in model_keys}
        for station_rec in metrics.values():
            if not isinstance(station_rec, dict):
                continue
            h_rec = station_rec.get(horizon_key, {})
            if not isinstance(h_rec, dict):
                continue
            for display, key in model_keys.items():
                m = h_rec.get(key, {})
                if isinstance(m, dict) and isinstance(m.get("mae"), (int, float)):
                    rec[display].append(float(m["mae"]))
        groups.append((labels[horizon_key], {name: float(np.mean(vals)) for name, vals in rec.items() if vals}))
    return groups


def generate_visuals() -> list[Path]:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    pred = _prediction_frame()
    start = pd.to_datetime(pred["timestamp"].iloc[0]).strftime("%Y-%m-%d")
    end = pd.to_datetime(pred["timestamp"].iloc[-1]).strftime("%Y-%m-%d")

    actual_path = IMAGES_DIR / "actual_vs_predicted.svg"
    _line_chart(
        actual_path,
        "Actual vs predicted",
        "DEMO-HNL held-out synthetic test span, HarmonicRidge against rolling persistence",
        [
            ("Actual", pred["actual"]),
            ("HarmonicRidge", pred["prediction"]),
            ("Rolling persistence", pred["rolling_persistence"]),
        ],
        (start, end),
        "Water level (m)",
    )

    horizon_path = IMAGES_DIR / "error_by_horizon.svg"
    _bar_chart(
        horizon_path,
        "Error by forecast horizon",
        "Average synthetic MAE across DEMO-HNL and DEMO-SFO; lower is better",
        _avg_horizon_metrics(),
        ["Rolling persistence", "HarmonicRidge", "GradBoost"],
        "MAE (m)",
    )

    baseline_path = IMAGES_DIR / "baseline_comparison.svg"
    _bar_chart(
        baseline_path,
        "Baseline comparison",
        "Average 1-step synthetic MAE across demo stations; lower is better",
        [("1-step", _avg_model_metrics())],
        ["Rolling persistence", "HarmonicRidge", "GradBoost", "WaveGRU"],
        "MAE (m)",
    )

    sample_step = max(1, len(pred) // 80)
    residual_sample = pred.iloc[::sample_step]
    residual_path = IMAGES_DIR / "residual_plot.svg"
    _scatter_chart(
        residual_path,
        "Residual plot",
        "HarmonicRidge residuals on DEMO-HNL synthetic holdout; zero line marks unbiased error",
        residual_sample["prediction"],
        residual_sample["residual"],
        "Predicted water level (m)",
        "Actual - predicted (m)",
    )

    return [actual_path, horizon_path, baseline_path, residual_path]


def main() -> None:
    paths = generate_visuals()
    for path in paths:
        print(f"Saved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
