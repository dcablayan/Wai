# Wai - Hybrid Water-Level Prediction Research Demo

> **Research demo, not an operational forecast system.**
> Wai is a lightweight, reproducible project for testing whether
> physics-informed tidal structure plus statistical/ML residual modeling can
> improve short-term water-level forecasts over serious baselines. Current
> synthetic results are sanity checks, not operational evidence.

![CI](https://github.com/dcablayan/Wai/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Thesis

In this repo, **hybrid** means:

1. Use known tidal physics as structure: NOAA tidal predictions and/or
   harmonic sin/cos features for astronomical constituents.
2. Learn the remaining residual statistically: Ridge, gradient boosting, or
   simple residual persistence.
3. Compare against baselines that are hard to beat: rolling persistence and
   NOAA tidal prediction.

Wai does **not** claim real-time forecasting, emergency alerting, validated
storm-surge modeling, or deep neural modeling. No PyTorch/TensorFlow dependency
is used or needed.

## Evidence Tracks

The project keeps evidence sources separate so mock and synthetic metrics are
not presented as real NOAA performance.

| Track | Data | Purpose | Main outputs |
| --- | --- | --- | --- |
| Synthetic sanity checks | `data/demo/demo_water_levels.csv` | Reproducible leakage, metrics, uncertainty, and event tests | `reports/model_metrics.json`, `reports/horizon_metrics.json`, `reports/event_metrics.json`, `reports/rolling_origin_metrics.json`, `reports/conformal_metrics.json`, `reports/ablation_metrics.json` |
| Tidecast prototype benchmark | `data/demo/tidecast/*.csv` NOAA-derived tidal predictions | Compare lightweight prototypes and persistence on a smooth tidal signal | `reports/benchmark_results.md` |
| NOAA mock evaluation | Synthetic fixtures shaped like NOAA API output | CI/offline check of NOAA evaluation code | `reports/noaa_mock_metrics.json`, `.md` |
| NOAA live evaluation | Public NOAA CO-OPS observations merged to NOAA predictions | Real-data baseline comparison when network access is available | `reports/noaa_live_metrics.json`, `.md` |
| Scientific evidence audit | Current checked-in reports | Machine-readable guardrail for live NOAA, mock, and forcing claims | `reports/scientific_evidence_audit.json`, `.md` |

`reports/summary.json` indexes these as `synthetic`, `tidecast`,
`noaa_mock`, and `noaa_live`.

## Research Answer

The current checked-in artifacts answer the portfolio question narrowly:

- On the **synthetic** benchmark, HarmonicRidge improves over rolling
  persistence at every evaluated horizon.
- On **rolling-origin synthetic** folds, HarmonicRidge improves in all six
  forward-in-time folds.
- On the **tidecast prototype** benchmark, TinyTide ties persistence rather
  than clearly beating it.
- On **NOAA mock** fixtures, hybrid residual Ridge does **not** beat the NOAA
  tidal prediction baseline. That is a mock/plumbing result, not live NOAA
  evidence.

Start with [docs/research-report.md](docs/research-report.md), then use
[docs/results-summary.md](docs/results-summary.md) for a compact table and
[docs/portfolio-case-study.md](docs/portfolio-case-study.md) for interview
framing.

## Models and Baselines

- `rolling_persistence`: one-step naive baseline, `pred[t] = observed[t-1]`.
- `noaa_prediction`: NOAA deterministic tidal prediction baseline.
- `noaa_residual_persistence`: NOAA prediction plus rolling residual
  persistence.
- `harmonic_ridge`: Ridge regression over tidal harmonic, temporal, lag, and
  rolling features.
- `grad_boost`: scikit-learn HistGradientBoostingRegressor on the same feature
  matrix.
- `hybrid_residual_ridge`: NOAA prediction plus Ridge-modeled residual.

Prototype names are compatibility labels, not capability claims. In reports
and docs, `WaveGRUPrototype` is described as smoothing, `SurgeNetPrototype` as
a residual heuristic, and `TsunamiSentinelPrototype` as an anomaly toy. Branding
details live in [docs/model_branding.md](docs/model_branding.md).

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

make demo
make test
make coverage
make dashboard
```

`make demo` regenerates the synthetic, tidecast, NOAA mock, rolling-origin,
conformal, research visual, scientific audit, and summary artifacts. The
dashboard is a local Streamlit viewer for the synthetic demo data.

## Forecast Orchestrator Foundation

Wai now includes a deterministic regional-to-local forecasting foundation under
`src/forecasting`, `src/orchestration`, `src/experts`, and the canonical
Hohonu/NOAA adapters in `src/data`. It converts local Hohonu observations and
NOAA observations/predictions into one canonical schema, routes to numerical
experts, combines successful forecasts, verifies safety constraints, and can
produce a historical replay table for future router training.

Run an offline example forecast:

```bash
python -m scripts.run_orchestrated_forecast --horizon-minutes 360
```

Generate a mocked historical replay dataset:

```bash
python -m scripts.run_historical_replay --output reports/routing_replay_mock.csv
```

Train an advisory learned router from replay rows:

```bash
python -m scripts.train_router --replay reports/routing_replay_mock.csv
```

See [docs/forecast_orchestrator.md](docs/forecast_orchestrator.md) for the data
flow, station pairing, datum rules, expert descriptions, environment variables,
example output, and limitations.

Generated research visuals are written to [docs/images](docs/images):

- `actual_vs_predicted.svg`
- `error_by_horizon.svg`
- `baseline_comparison.svg`
- `residual_plot.svg`

`make dashboard` starts Streamlit when localhost port binding is available.
In sandboxed environments that block local servers, it runs a dashboard smoke
check instead so the reproducible path still verifies the dashboard data flow.

Run a live NOAA evaluation only when network access is available:

```bash
python -m scripts.evaluate_noaa_public
```

Live mode writes `reports/noaa_live_metrics.*` and fails if any station would
fall back to mock data. Offline mode writes mock reports only:

```bash
python -m scripts.evaluate_noaa_public --offline
# or
NOAA_OFFLINE=1 python -m scripts.evaluate_noaa_public
```

Run the evidence audit after any report changes:

```bash
python -m scripts.audit_scientific_evidence
```

The audit currently records that the repo is forcing-ready but not
storm-surge-validated: numeric external columns such as `wind_speed_mps`,
`air_pressure_hpa`, `rainfall_mm`, and `wave_height_m` are accepted by the
feature matrix when supplied, but checked-in reports do not include real
meteorological covariates.

## Important Limitations

- Synthetic demo metrics are correctness checks, not evidence of operational
  water-level forecasting skill.
- NOAA-derived tidecast benchmark data is smooth deterministic tidal output,
  not noisy sensor observations.
- The live NOAA report uses short public API windows and no meteorological
  forcing, so it cannot validate storm-surge forecasting.
- If no `reports/noaa_live_metrics.*` artifact is present, the live NOAA claim
  remains open; mock reports are never a substitute.
- Conformal intervals are split-conformal intervals; empirical coverage is
  reported because tidal time series are not guaranteed exchangeable.
- Report thresholds are fit on train/reference windows, not the full displayed
  period.

See [docs/model_card.md](docs/model_card.md) for scope and failure modes, and
[docs/metrics_interpretation.md](docs/metrics_interpretation.md) for how to
read the generated numbers.

## Project Layout

```text
src/          data loaders, feature engineering, models, metrics, reports
scripts/      reproducible evaluations and report builders
tests/        pytest suite covering leakage, baselines, reports, and artifacts
data/demo/    synthetic and NOAA-derived demo inputs
reports/      generated metrics and HTML/Markdown reports
docs/         model card, modeling notes, metric interpretation, branding notes
app.py        Streamlit dashboard for the synthetic demo track
```

## License

MIT - see [LICENSE](LICENSE).
