"""Assemble reports/summary.json from the latest generated reports.

The README and model card link to this file (rather than embedding numbers
in prose) so that the numbers always match the artifacts in `reports/`. Run
this after any of:

    python -m scripts.train_baseline
    python -m scripts.evaluate_horizons
    python -m scripts.evaluate_events
    python -m scripts.run_benchmark
    python -m scripts.evaluate_noaa_public

It is also wired into `make demo` so a full repro never drifts.

The summary writes a ``staleness`` block that compares the SHA recorded in
``reports/run_metadata.json`` against the current ``git HEAD``. If they
differ, ``summary.json`` is still produced but a warning is printed and
``staleness.fresh`` is set to ``false`` — a CI step (or a human reviewer)
can grep for this to refuse to ship stale committed reports.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
SUMMARY_PATH = REPORTS_DIR / "summary.json"


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _benchmark_table(md_path: Path) -> Optional[dict]:
    """Parse reports/benchmark_results.md into a structured dict.

    The benchmark report is small (~10 stations × 4 models) so re-parsing the
    Markdown is preferable to maintaining a second JSON output.
    """
    if not md_path.exists():
        return None
    text = md_path.read_text()

    stations = {}
    averages = {}
    header_models: list = []

    in_station_table = False
    in_avg_table = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            in_station_table = False
            in_avg_table = False
            continue
        if "Station" in line and "RMSE" in line and not in_avg_table:
            cols = [c.strip() for c in line.strip("|").split("|")]
            header_models = [
                re.sub(r"\s*RMSE\s*$", "", c).strip() for c in cols[1:]
            ]
            in_station_table = True
            continue
        if "Mean RMSE" in line:
            in_station_table = False
            in_avg_table = True
            continue
        if line.startswith("| ---") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if in_station_table and header_models and len(cells) >= 2:
            sid = cells[0]
            for i, model in enumerate(header_models):
                if i + 1 < len(cells):
                    try:
                        stations.setdefault(sid, {})[model] = float(cells[i + 1])
                    except ValueError:
                        pass
        elif in_avg_table and len(cells) >= 2:
            try:
                averages[cells[0]] = float(cells[1])
            except ValueError:
                pass

    # Recompute averages from station rows so the report and the index never
    # drift apart.
    recomputed: Dict[str, float] = {}
    if stations and header_models:
        for model in header_models:
            vals = [
                rec[model] for rec in stations.values() if model in rec
            ]
            if vals:
                recomputed[model] = round(sum(vals) / len(vals), 3)

    return {
        "stations": stations,
        "averages_from_report": averages,
        "averages_recomputed": recomputed,
        "mismatch": {
            model: {
                "in_report": averages.get(model),
                "recomputed": recomputed.get(model),
            }
            for model in recomputed
            if averages.get(model) is not None
            and abs(averages[model] - recomputed[model]) > 0.01
        },
    }


def _flatten_station_metrics(model_metrics: Optional[dict]) -> Optional[dict]:
    """Pull the headline 1-step MAE/RMSE/R^2 per station for the index."""
    if not model_metrics:
        return None
    out: Dict[str, dict] = {}
    for station, models in model_metrics.items():
        if not isinstance(models, dict):
            continue
        row = {}
        for model_name in (
            "persistence", "persistence_constant",
            "harmonic_ridge", "wave_gru", "grad_boost",
        ):
            m = models.get(model_name, {})
            if isinstance(m, dict) and "mae" in m:
                row[model_name] = {
                    "mae": m["mae"], "rmse": m["rmse"], "r2": m["r2"],
                    "mae_ci_95": m.get("mae_ci_95"),
                    "ci_method": (
                        m.get("mae_block_ci_95", {}).get("method")
                        if isinstance(m.get("mae_block_ci_95"), dict) else None
                    ),
                    "block_length": (
                        m.get("mae_block_ci_95", {}).get("block_length")
                        if isinstance(m.get("mae_block_ci_95"), dict) else None
                    ),
                }
        if row:
            out[station] = row
    return out


def _horizon_summary(horizon_metrics: Optional[dict]) -> Optional[dict]:
    """Per-horizon headline MAE for each station and model."""
    if not horizon_metrics:
        return None
    out: Dict[str, dict] = {}
    for station, horizons in horizon_metrics.items():
        if not isinstance(horizons, dict):
            continue
        station_out = {}
        for h_name, models in horizons.items():
            if h_name.startswith("_") or not isinstance(models, dict):
                continue
            station_out[h_name] = {}
            for model_name, m in models.items():
                if model_name.startswith("_"):
                    continue
                if isinstance(m, dict) and "mae" in m:
                    station_out[h_name][model_name] = {
                        "mae": m["mae"], "rmse": m["rmse"], "r2": m["r2"],
                    }
        if station_out:
            out[station] = station_out
    return out


def _event_summary(event_metrics: Optional[dict]) -> Optional[dict]:
    if not event_metrics:
        return None
    out = {}
    for station, rec in event_metrics.items():
        if station.startswith("_") or not isinstance(rec, dict):
            continue
        out[station] = {
            "train_threshold_m": rec.get("train_threshold_m"),
            "test_obs_episodes": rec.get("test_obs_episodes"),
            "harmonic_ridge_episode": rec.get("harmonic_ridge", {}).get("episode"),
            "grad_boost_episode": rec.get("grad_boost", {}).get("episode"),
            "persistence_episode": rec.get("persistence_rolling", {}).get("episode"),
        }
    return out


def _conformal_summary(conformal_metrics: Optional[dict]) -> Optional[dict]:
    if not conformal_metrics:
        return None
    out = {}
    for station, rec in conformal_metrics.items():
        if station.startswith("_") or not isinstance(rec, dict):
            continue
        out[station] = rec.get("models", {})
    return out


def _rolling_origin_summary(rolling_metrics: Optional[dict]) -> Optional[dict]:
    if not rolling_metrics:
        return None
    out = {}
    for station, rec in rolling_metrics.items():
        if station.startswith("_") or not isinstance(rec, dict):
            continue
        out[station] = {
            "n_folds": rec.get("n_folds"),
            "folds": [
                {
                    "fold": f.get("fold"),
                    "train_start": f.get("train_start"),
                    "train_end": f.get("train_end"),
                    "test_start": f.get("test_start"),
                    "test_end": f.get("test_end"),
                    "n_train": f.get("n_train"),
                    "n_test": f.get("n_test"),
                    "rolling_persistence_mae": f.get("rolling_persistence", {}).get("mae"),
                    "harmonic_ridge_mae": f.get("harmonic_ridge", {}).get("mae"),
                    "grad_boost_mae": f.get("grad_boost", {}).get("mae"),
                }
                for f in rec.get("folds", [])
            ],
        }
    return out


def _ablation_claims(ablation_metrics: Optional[dict]) -> Optional[dict]:
    """Generate ablation takeaways from current metrics, never hardcoded prose."""
    if not ablation_metrics:
        return None
    harmonics_r2 = []
    full_beats_harmonics = []
    best_by_station = {}
    for station, configs in ablation_metrics.items():
        if not isinstance(configs, dict):
            continue
        h = configs.get("harmonics_only", {})
        f = configs.get("full", {})
        if isinstance(h, dict) and isinstance(h.get("r2"), (int, float)):
            harmonics_r2.append(float(h["r2"]))
        if (
            isinstance(h, dict)
            and isinstance(f, dict)
            and isinstance(h.get("mae"), (int, float))
            and isinstance(f.get("mae"), (int, float))
        ):
            full_beats_harmonics.append(float(f["mae"]) < float(h["mae"]))
        scored = {
            cfg: m.get("mae")
            for cfg, m in configs.items()
            if isinstance(m, dict) and isinstance(m.get("mae"), (int, float))
        }
        if scored:
            best_by_station[station] = min(scored, key=scored.get)

    if not harmonics_r2:
        return None
    r2_min = min(harmonics_r2)
    r2_max = max(harmonics_r2)
    return {
        "stations_with_harmonics_only": int(len(harmonics_r2)),
        "harmonics_only_r2_min": round(r2_min, 6),
        "harmonics_only_r2_max": round(r2_max, 6),
        "harmonics_only_all_ge_0_98": bool(all(v >= 0.98 for v in harmonics_r2)),
        "full_mae_better_than_harmonics_only_all": bool(full_beats_harmonics and all(full_beats_harmonics)),
        "best_config_by_station": best_by_station,
        "statement": (
            "Current synthetic ablation shows harmonics-only R2 ranging from "
            f"{r2_min:.4f} to {r2_max:.4f}; full features improve MAE over "
            "harmonics-only for every station."
            if full_beats_harmonics and all(full_beats_harmonics)
            else
            "Current synthetic ablation does not support a blanket claim that "
            "full features improve MAE over harmonics-only at every station."
        ),
    }


def _noaa_summary(noaa_metrics: Optional[dict]) -> Optional[dict]:
    if not noaa_metrics:
        return None
    meta = noaa_metrics.get("_meta", {})
    stations = {}
    for sid, rec in noaa_metrics.items():
        if sid.startswith("_") or not isinstance(rec, dict):
            continue
        stations[sid] = {
            "station": rec.get("station"),
            "data_source": rec.get("data_source"),
            "mock_used": rec.get("mock_used"),
            "begin_date": rec.get("begin_date"),
            "end_date": rec.get("end_date"),
            "n_train": rec.get("n_train"),
            "n_test": rec.get("n_test"),
            "rolling_persistence_mae": rec.get("rolling_persistence", {}).get("mae"),
            "noaa_prediction_mae": rec.get("noaa_prediction", {}).get("mae"),
            "noaa_residual_persistence_mae": rec.get("noaa_residual_persistence", {}).get("mae"),
            "harmonic_ridge_mae": rec.get("harmonic_ridge", {}).get("mae"),
            "grad_boost_mae": rec.get("grad_boost", {}).get("mae"),
            "hybrid_residual_ridge_mae": rec.get("hybrid_residual_ridge", {}).get("mae"),
        }
    return {"meta": meta, "stations": stations}


def _current_head_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def _staleness(run_meta: Optional[dict]) -> dict:
    head = _current_head_sha()
    recorded = run_meta.get("git_sha") if run_meta else None
    return {
        "head_sha": head,
        "report_sha": recorded,
        "fresh": bool(head and recorded and head == recorded),
        "note": (
            "Committed reports are regenerated by `make demo`. If "
            "staleness.fresh is false, re-run `make demo` (or move the "
            "committed copies into reports/sample/ if they are meant to be "
            "frozen examples)."
        ),
    }


def build_summary() -> dict:
    REPORTS_DIR.mkdir(exist_ok=True)
    run_meta = _read_json(REPORTS_DIR / "run_metadata.json")
    ablation = _read_json(REPORTS_DIR / "ablation_metrics.json")
    summary = {
        "schema_version": 1,
        "generated_by": "scripts/build_summary.py",
        "run_metadata": run_meta,
        "staleness": _staleness(run_meta),
        "synthetic": {
            "model_metrics_1step": _flatten_station_metrics(
                _read_json(REPORTS_DIR / "model_metrics.json")
            ),
            "horizon_metrics": _horizon_summary(
                _read_json(REPORTS_DIR / "horizon_metrics.json")
            ),
            "event_metrics": _event_summary(
                _read_json(REPORTS_DIR / "event_metrics.json")
            ),
            "rolling_origin": _rolling_origin_summary(
                _read_json(REPORTS_DIR / "rolling_origin_metrics.json")
            ),
            "conformal": _conformal_summary(
                _read_json(REPORTS_DIR / "conformal_metrics.json")
            ),
            "ablation": ablation,
            "ablation_claims": _ablation_claims(ablation),
        },
        "tidecast": {
            "benchmark": _benchmark_table(REPORTS_DIR / "benchmark_results.md"),
        },
        "noaa_mock": _noaa_summary(
            _read_json(REPORTS_DIR / "noaa_mock_metrics.json")
        ),
        "noaa_live": _noaa_summary(
            _read_json(REPORTS_DIR / "noaa_live_metrics.json")
        ),
        "noaa_allow_mock": _noaa_summary(
            _read_json(REPORTS_DIR / "noaa_allow_mock_metrics.json")
        ),
        "scientific_evidence_audit": _read_json(
            REPORTS_DIR / "scientific_evidence_audit.json"
        ),
    }
    return summary


def main() -> None:
    summary = build_summary()
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Saved {SUMMARY_PATH}")

    benchmark = summary["tidecast"].get("benchmark")
    if benchmark and benchmark["mismatch"]:
        print("\nBenchmark average mismatch detected:")
        for model, vals in benchmark["mismatch"].items():
            print(f"  {model}: report={vals['in_report']} "
                  f"recomputed={vals['recomputed']}")

    stl = summary["staleness"]
    if not stl["fresh"]:
        print(
            f"\nWARNING: committed reports look stale.\n"
            f"  HEAD={stl['head_sha']}  report SHA={stl['report_sha']}.\n"
            f"  Re-run `make demo` so the artifacts match the current code."
        )


if __name__ == "__main__":
    main()
