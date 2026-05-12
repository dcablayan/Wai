# Wai — Coastal Water-Level Forecasting

> End-to-end time-series pipeline for ingesting, validating, modeling, and
> visualizing coastal water-level data — built with NOAA CO-OPS awareness and
> a reproducible demo dataset.

![CI](https://github.com/dcablayan/Wai/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Problem Statement

Coastal flooding is accelerating. Sea-level rise combined with more frequent
storm surge and king-tide events is putting millions of people and billions in
infrastructure at risk. Accurate, station-level water-level forecasting is
a critical input for emergency management, insurance underwriting, stormwater
planning, and climate adaptation.

NOAA's tide-gauge network provides decades of high-quality observations, but
turning that data into actionable, multi-step-ahead forecasts requires a
reproducible pipeline — from raw sensor ingestion through feature engineering,
model evaluation, and stakeholder-facing reporting.

**Wai** is a clean, honest implementation of that pipeline.

---

## Why This Matters

- **King tides** — predictable but extreme high-water events driven by
  lunar/solar alignment — are a leading cause of nuisance flooding.
- **Storm surge** — meteorologically forced water-level rise — can exceed
  3–5 m above predicted tide, destroying infrastructure within hours.
- Forecasting even 6–24 hours ahead gives communities time to close surge
  barriers, pre-position pumps, and issue evacuation warnings.

---

## What I Built

Wai is a public-safe portfolio project demonstrating a complete coastal
water-level forecasting pipeline. It was inspired by operational coastal
monitoring work and built with NOAA CO-OPS awareness. The goal was to show the
full path from raw data to actionable outputs — without exposing any private
sensor data, credentials, or proprietary systems.

Key design choices:

- **Synthetic demo data** keeps the repo fully self-contained and public-safe
  while remaining physically realistic.
- **NOAA-derived tidecast predictions** provide a real-world harmonic signal
  to benchmark prototype models against, separate from the noisy synthetic series.
- **Pure-Python prototype models** demonstrate algorithmic ideas (harmonic
  fitting, exponential smoothing, anomaly detection) without requiring GPU
  infrastructure.
- **Production-style pipeline** (validation → feature engineering → Ridge →
  metrics → HTML reports → Streamlit dashboard) reflects how these models would
  integrate in an operational system.

No private sensor data, internal APIs, secrets, or proprietary forecasts are
present anywhere in this repository.

---

## Features

| Feature | Details |
|---------|---------|
| Demo data loader | Reproducible synthetic dataset (2 stations, 90 days, 6-min cadence) |
| NOAA CO-OPS loader | Live ingestion from the public API (no key required) |
| Tidecast reference data | NOAA-derived tidal predictions for 10 coastal stations (Hawaii) |
| Data validation | NaN, gap, duplicate, out-of-range, and timezone checks |
| Tidal feature engineering | 8-constituent sin/cos (M2/S2/K1/O1/N2/M4/M6/Mm) + temporal covariates + lags + rolling windows |
| Windowing utilities | Sliding-window builder and temporal train/val/test splits for prototype models |
| Persistence baseline | Last-value naive baseline |
| Harmonic Ridge model | Ridge-regularised linear regression over 8 tidal constituents + temporal covariates |
| WaveGRU adapter | DataFrame API wrapper for WaveGRUPrototype (bidirectional smoothing, no feature engineering required) |
| Prototype benchmark models | TinyTide, HarmonicNet, WaveGRU, SurgeNet, TsunamiSentinel — pure-Python research prototypes |
| Metrics | MAE, RMSE, R², NSE, Pearson correlation |
| Streamlit dashboard | Time series, forecast overlay, anomaly markers, station map |
| HTML report generation | Per-station report with stats, metrics, anomaly table |
| Benchmark reports | Prototype model RMSE across tidecast stations (`reports/benchmark_results.md`) |
| CI pipeline | GitHub Actions tests on Python 3.10 + 3.11 |

---

## Tech Stack

- **Python 3.10+** · numpy · pandas · scikit-learn · statsmodels
- **Streamlit** · Plotly — interactive dashboard
- **NOAA CO-OPS API** — public, no key required
- **pytest** — test suite
- **GitHub Actions** — CI

---

## Quickstart (< 5 minutes)

```bash
# 1. Clone and install
git clone https://github.com/dcablayan/Wai.git
cd Wai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Generate the synthetic demo dataset
python -m scripts.prepare_demo_data

# 3. Train baseline models and save metrics
python -m scripts.train_baseline

# 4. Run prototype benchmarks on tidecast data
python -m scripts.run_benchmark

# 5. Generate HTML station reports
python -m scripts.generate_report

# 6. Launch the interactive dashboard
streamlit run app.py
```

Open `http://localhost:8501` to explore the dashboard.

Or use `make` for convenience — see the [Makefile](Makefile).

---

## Demo Data Disclaimer

> **Data provenance:**
>
> - `data/demo/demo_water_levels.csv` — **entirely synthetic**, generated by
>   `scripts/prepare_demo_data.py`. Signal composed of M2/S2/K1/O1 constituents
>   with Gaussian noise, a synthetic storm-surge event, and a king-tide pulse.
>   Not real sensor measurements.
>
> - `data/demo/tidecast/*.csv` — **NOAA-derived tidal predictions** (not raw
>   sensor readings), originally published in
>   [dcablayan/tideformer](https://github.com/dcablayan/tideformer). These are
>   computed from NOAA harmonic constants — equivalent to calling the NOAA
>   CO-OPS `predictions` API for those station locations. The `hohonu-*` file
>   prefixes indicate the geographic location of public coastal monitoring sites;
>   no proprietary sensor data is present.
>
> No private data, API keys, credentials, or proprietary information is present
> anywhere in this repository. NOAA data loaded via `load_noaa_data()` is
> fetched live from the public API and is not stored.

---

## Fetch Real NOAA Data

```python
from src.data.loader import load_noaa_data

# San Francisco, January 2024, MLLW datum, metric units
df = load_noaa_data(
    station_id="9414290",
    begin_date="20240101",
    end_date="20240131",
    datum="MLLW",
    units="metric",
)
print(df.head())
```

Key NOAA station IDs:
- `9414290` — San Francisco, CA
- `1612340` — Honolulu, HI
- `8443970` — Boston, MA
- `8723214` — Virginia Key, FL

---

## Model Approach

### A. Production-style pipeline models

These fit the same DataFrame API and are evaluated on the synthetic demo data.
Metrics are saved to `reports/model_metrics.json`.

**1. Persistence baseline** — forecasts the last observed value for all future
steps. Provides a performance floor that any useful model must beat.

**2. Harmonic Ridge** — Ridge-regularised linear regression over 8 tidal
constituent sin/cos features (M2, S2, K1, O1, N2, M4, M6, Mm), temporal
covariates (hour-of-day, lunar phase), lagged observations at 6–240 minute
windows, and rolling mean/std features. Captures the dominant predictable
tidal signal without deep learning infrastructure.

**3. WaveGRU adapter** — DataFrame wrapper around `WaveGRUPrototype`.
Bidirectional double-exponential smoothing with attention-like weighting.
Operates on raw values; no tidal feature engineering is applied.

All three models are evaluated on a temporal hold-out (last 25% of data per
station) to prevent data leakage.

### B. Prototype benchmark models

These are pure-Python research prototypes ported from
[dcablayan/tideformer](https://github.com/dcablayan/tideformer). They use
stdlib math only — no PyTorch, TensorFlow, or scikit-learn. They are benchmark
baselines, not production deep-learning models. They are evaluated on
NOAA-derived tidecast predictions (smoother signal than raw sensor data; see
[`docs/metrics_interpretation.md`](docs/metrics_interpretation.md)).

| Model | Description | Mean RMSE (tidecast, ft) |
|-------|-------------|--------------------------|
| TinyTide | Two-layer MLP-style with tidal covariates and skip connection | 0.222 |
| HarmonicNet | Physics-informed 8-constituent harmonic projection | 6.687 |
| WaveGRU | Bidirectional double-exp smoothing + attention weighting | 0.911 |
| SurgeNet | Dual-head tide + surge residual estimator | 6.328 |
| TsunamiSentinel | Multi-scale high-pass anomaly detector | — |

Run the benchmark:

```bash
python -m scripts.run_benchmark
# Output: reports/benchmark_results.md
```

See [`docs/modeling.md`](docs/modeling.md) for full model details and
[`docs/metrics_interpretation.md`](docs/metrics_interpretation.md) for
guidance on interpreting these numbers honestly.

---

## Sample Outputs

After running the quickstart commands:

- `reports/model_metrics.json` — MAE/RMSE/R² for each pipeline model and station (synthetic data)
- `reports/benchmark_results.md` — RMSE table for prototype models across tidecast stations
- `reports/report_DEMO-HNL.html` — interactive HTML report for Honolulu station
- `reports/report_DEMO-SFO.html` — interactive HTML report for SF station

To add screenshots to this README:
1. Take a screenshot of the dashboard or a report
2. Save to `docs/images/`
3. Reference with `![Dashboard](docs/images/dashboard.png)`

---

## Project Structure

```
Wai/
├── src/
│   ├── data/
│   │   ├── loader.py          Demo + NOAA CO-OPS data loaders
│   │   ├── validation.py      Schema and quality validation
│   │   └── windowing.py       Sliding-window builder + tidecast CSV loader
│   ├── features/
│   │   └── engineering.py     8-constituent tidal harmonics, temporal covariates, lags, rolling windows
│   ├── models/
│   │   ├── baseline.py        PersistenceModel + HarmonicRidgeModel + WaveGRUModel adapter
│   │   ├── metrics.py         MAE / RMSE / R² / NSE / Pearson corr
│   │   └── prototypes.py      TinyTide, HarmonicNet, WaveGRU, SurgeNet, TsunamiSentinel (pure-Python)
│   └── reporting/
│       └── report.py          HTML report generation
├── scripts/
│   ├── prepare_demo_data.py   Generate synthetic demo CSV
│   ├── train_baseline.py      Train pipeline models, save metrics JSON
│   ├── generate_report.py     Build HTML reports from data + metrics
│   └── run_benchmark.py       Benchmark prototype models on tidecast data
├── app.py                     Streamlit dashboard entry point
├── data/demo/
│   ├── demo_water_levels.csv  Synthetic water-level data (safe to commit)
│   ├── README.md              Data provenance notes
│   └── tidecast/              NOAA-derived tidal predictions for 10 station locations
├── reports/
│   ├── model_metrics.json     Pipeline model metrics (synthetic data)
│   ├── benchmark_results.md   Prototype model RMSE table (tidecast data)
│   └── report_*.html          Per-station HTML reports
├── tests/                     pytest test suite
├── docs/
│   ├── architecture.md        Pipeline diagram and design decisions
│   ├── modeling.md            Model details, features, limitations
│   ├── metrics_interpretation.md  How to read synthetic and tidecast metrics honestly
│   └── images/                Screenshot placeholder
├── Makefile                   Common workflow targets
├── CONTRIBUTING.md            Local setup guide
├── .github/workflows/ci.yml   GitHub Actions CI
├── requirements.txt
└── pyproject.toml
```

---

## Limitations

- Demo pipeline results are on **synthetic data** and cannot be compared to
  published operational benchmarks.
- Tidecast benchmark metrics are on **NOAA-derived tidal predictions** (smooth,
  deterministic signal) — results will be substantially better than real-world
  performance on noisy sensor data.
- Prototype models are research baselines implemented in pure Python (stdlib
  math only); they are not production neural networks despite their names.
- The harmonic model does not incorporate meteorological forcing (wind,
  pressure), so it cannot predict storm surge from first principles.
- NOAA API requests are limited to 31-day windows per call; longer time series
  require multiple requests.
- No hyperparameter tuning is performed; Ridge alpha=1.0 is a reasonable
  default, not an optimised one.

See [`docs/metrics_interpretation.md`](docs/metrics_interpretation.md) for
a full discussion of how to interpret reported numbers.

---

## Future Roadmap

- [ ] Gradient boosting (XGBoost / LightGBM) — non-linear baseline
- [ ] LSTM encoder for multi-step-ahead forecasting
- [ ] Informer / Autoformer Transformer architecture
- [ ] Proper harmonic constituent fitting using `utide`
- [ ] Storm-surge covariate from NWS forecast API
- [ ] Uncertainty quantification via conformal prediction
- [ ] Multi-station spatial interpolation
- [ ] Real-time alerting for high-water thresholds

---

## Resume Bullet

> Built **Wai**, a reproducible coastal water-level forecasting pipeline in
> Python; implemented 8-constituent tidal harmonic feature engineering over
> NOAA CO-OPS data, trained Ridge and persistence baseline models (reporting
> MAE/RMSE/R² on held-out test sets), benchmarked five pure-Python prototype
> models against NOAA-derived tidal predictions, and shipped an interactive
> Streamlit dashboard with anomaly detection, station maps, and automated HTML
> reports — all with a CI-tested synthetic demo dataset and zero private data
> exposure.

---

## License

MIT — see [LICENSE](LICENSE).
