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

`reports/summary.json` indexes these as `synthetic`, `tidecast`,
`noaa_mock`, and `noaa_live`.

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
conformal, and summary artifacts. The dashboard is a local Streamlit viewer
for the synthetic demo data.

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

## Important Limitations

- Synthetic demo metrics are correctness checks, not evidence of operational
  water-level forecasting skill.
- NOAA-derived tidecast benchmark data is smooth deterministic tidal output,
  not noisy sensor observations.
- The live NOAA report uses short public API windows and no meteorological
  forcing, so it cannot validate storm-surge forecasting.
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
