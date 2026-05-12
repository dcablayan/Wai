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

## Implemented Capabilities

| Capability | Details |
|-----------|---------|
| Demo data loader | Reproducible synthetic dataset (2 stations, 90 days, 6-min cadence) |
| NOAA CO-OPS loader | Live ingestion of observations and tidal predictions (no API key required) |
| Tidecast reference data | NOAA-derived tidal predictions for 10 coastal stations (Hawaii) |
| Data validation | NaN, gap, duplicate, out-of-range, and timezone checks |
| Tidal feature engineering | 8-constituent sin/cos (M2/S2/K1/O1/N2/M4/M6/Mm) + temporal covariates + lags + rolling windows |
| Windowing utilities | Sliding-window builder and temporal train/val/test splits for prototype models |
| Persistence baseline | Last-value naive baseline |
| Harmonic Ridge model | Ridge-regularised linear regression over 8 tidal constituents + temporal covariates |
| Gradient boosting baseline | HistGradientBoostingRegressor over the same feature matrix (non-linear baseline, scikit-learn only) |
| WaveGRU adapter | DataFrame API wrapper for WaveGRUPrototype (bidirectional smoothing) |
| Prototype benchmark models | TinyTide, HarmonicNet, WaveGRU, SurgeNet, TsunamiSentinel — pure-Python research prototypes |
| Multi-horizon evaluation | Direct forecasting at 1 step (6 min), 6 h, 12 h, 24 h horizons |
| Conformal uncertainty intervals | Split-conformal prediction intervals (distribution-free, 90% nominal coverage) |
| High-water alert detection | Configurable thresholds: mean + k·std, absolute value, or percentile |
| Spatial interpolation | Inverse-distance weighting across stations with lat/lon coordinates |
| Metrics | MAE, RMSE, R², NSE, Pearson correlation |
| Streamlit dashboard | Tabbed UI: Overview · Forecasts · Model Comparison · Alerts · Uncertainty · Benchmark |
| HTML report generation | Per-station report with stats, metrics, anomaly table, alert summary |
| Benchmark reports | Prototype model RMSE across tidecast stations |
| CI pipeline | GitHub Actions tests on Python 3.10 + 3.11 |

Advanced deep learning (LSTM, Transformer) and meteorological surge modeling (NWS/GFS covariates)
are intentionally excluded to keep the repo lightweight, dependency-free, and honest about scope.

---

## Tech Stack

- **Python 3.10+** · numpy · pandas · scikit-learn
- **Streamlit** · Plotly — interactive dashboard
- **NOAA CO-OPS API** — public, no key required
- **pytest** — test suite (110 tests)
- **GitHub Actions** — CI

---

## Quickstart (< 5 minutes)

```bash
# 1. Clone and install
git clone https://github.com/dcablayan/Wai.git
cd Wai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the full demo pipeline (data → train → report → benchmark)
make demo

# 3. Evaluate multi-horizon forecast skill
python -m scripts.evaluate_horizons

# 4. Run the test suite
make test

# 5. Launch the interactive dashboard
make dashboard
```

Open `http://localhost:8501` to explore the dashboard.

Or run individual steps:
```bash
python -m scripts.prepare_demo_data   # synthetic demo data
python -m scripts.train_baseline      # train + evaluate all models
python -m scripts.run_benchmark       # benchmark prototypes on tidecast data
python -m scripts.generate_report     # HTML station reports
```

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

| Display Name | Description | Mean RMSE (tidecast, ft) |
|-------------|-------------|--------------------------|
| ʻAle Iki (Ripple) | Two-layer MLP-style with tidal covariates and skip connection | 0.222 |
| Nalu Hoʻokani (Harmonic Wave) | Physics-informed 8-constituent harmonic projection | 9.389 |
| Nalu Holo (Fast Wave) | Bidirectional double-exp smoothing + attention weighting | 0.911 |
| ʻAle Piʻi (Rising Wave) | Dual-head tide + surge residual estimator | 8.582 |
| Kai Eʻe (Tsunami) | Multi-scale high-pass anomaly detector | — |

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

- `reports/model_metrics.json` — MAE/RMSE/R² for all four pipeline models per station
- `reports/horizon_metrics.json` / `.md` — multi-horizon evaluation at 1-step, 6h, 12h, 24h
- `reports/benchmark_results.md` — RMSE table for prototype models across tidecast stations
- `reports/report_DEMO-HNL.html` — interactive HTML report for Honolulu station
- `reports/report_DEMO-SFO.html` — interactive HTML report for SF station

---

## Project Structure

```
Wai/
├── src/
│   ├── data/
│   │   ├── loader.py          Demo + NOAA CO-OPS loaders (observations + predictions)
│   │   ├── validation.py      Schema and quality validation
│   │   └── windowing.py       Sliding-window builder + tidecast CSV loader
│   ├── features/
│   │   ├── engineering.py     8-constituent tidal harmonics, temporal covariates, lags, rolling windows
│   │   └── spatial.py         Inverse-distance weighting across stations with lat/lon
│   ├── models/
│   │   ├── baseline.py        PersistenceModel + HarmonicRidgeModel + WaveGRUModel adapter
│   │   ├── gradient_boost.py  GradBoostModel (HistGradientBoostingRegressor)
│   │   ├── conformal.py       Split-conformal prediction intervals
│   │   ├── metrics.py         MAE / RMSE / R² / NSE / Pearson corr
│   │   └── prototypes.py      TinyTide, HarmonicNet, WaveGRU, SurgeNet, TsunamiSentinel (pure-Python)
│   ├── alerts.py              High-water alert detection (std / absolute / percentile thresholds)
│   └── reporting/
│       └── report.py          HTML report generation
├── scripts/
│   ├── prepare_demo_data.py   Generate synthetic demo CSV
│   ├── train_baseline.py      Train all four pipeline models, save metrics JSON
│   ├── evaluate_horizons.py   Multi-horizon evaluation (1-step, 6h, 12h, 24h)
│   ├── generate_report.py     Build HTML reports from data + metrics
│   └── run_benchmark.py       Benchmark prototype models on tidecast data
├── app.py                     Streamlit dashboard (tabbed: Overview / Forecasts / Comparison / Alerts / Uncertainty / Benchmark)
├── data/demo/
│   ├── demo_water_levels.csv  Synthetic water-level data (safe to commit)
│   ├── README.md              Data provenance notes
│   └── tidecast/              NOAA-derived tidal predictions for 10 station locations
├── reports/
│   ├── model_metrics.json     Pipeline model metrics (synthetic data)
│   ├── horizon_metrics.json   Multi-horizon evaluation results
│   ├── benchmark_results.md   Prototype model RMSE table (tidecast data)
│   └── report_*.html          Per-station HTML reports
├── tests/                     pytest test suite (110 tests)
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

## Wave Model Names

Each model carries a Hawaiian wave name alongside its English label. Internal
Python class names (e.g. `TinyTidePrototype`, `HarmonicRidgeModel`) remain
unchanged for code stability. The display names below are used in reports, the
dashboard, and documentation. The single source of truth is
`src/models/branding.py`.

| Internal Name | Display Name | English Label | Category | Role |
|---------------|-------------|---------------|----------|------|
| TinyTidePrototype | ʻAle Iki (Ripple) | Ripple | Prototype | Smallest/simple benchmark or sanity-check model |
| HarmonicNetPrototype | Nalu Hoʻokani (Harmonic Wave) | Harmonic Wave | Prototype | Harmonic/periodic tide model using tidal rhythm |
| WaveGRUPrototype | Nalu Holo (Fast Wave) | Fast Wave | Prototype | Sequential smoothing/time-series movement model |
| SurgeNetPrototype | ʻAle Piʻi (Rising Wave) | Rising Wave | Prototype | Surge/high-water residual model |
| TsunamiSentinelPrototype | Kai Eʻe (Tsunami) | Tsunami | Prototype | Strongest anomaly/sentinel model — tsunami and rapid-spike detection |
| PersistenceModel | ʻAle Kūpaʻa (Steady Wave) | Steady Wave | Pipeline | Naive last-value baseline and stability floor |
| HarmonicRidgeModel | Nalu Hoʻokani Ridge (Harmonic Wave Ridge) | Harmonic Wave Ridge | Pipeline | Ridge regression over harmonic tide features |
| WaveGRUModel | Nalu Holo Adapter (Fast Wave Adapter) | Fast Wave Adapter | Pipeline | DataFrame adapter around WaveGRUPrototype |

---

## Known Limitations

- All pipeline results are on **synthetic data** and cannot be compared to
  published operational benchmarks.
- Tidecast benchmark metrics are on **NOAA-derived tidal predictions** (smooth,
  deterministic signal) — results will be substantially better than real-world
  performance on noisy sensor data.
- Prototype models are research baselines implemented in pure Python (stdlib
  math only); they are not production neural networks despite their names.
- No meteorological forcing (wind, pressure) is incorporated; storm surge
  cannot be predicted from first principles.
- NOAA API requests are limited to 31-day windows per call; longer time series
  require multiple requests.
- No hyperparameter tuning is performed; default parameters are reasonable
  starting points, not optimised values.
- Conformal intervals assume exchangeability; tidal series are non-stationary,
  so empirical coverage may fall below the nominal level.
- IDW spatial interpolation is a simple deterministic method with no
  uncertainty quantification.
- Advanced deep learning (LSTM, Transformer) and meteorological surge modeling
  (NWS/GFS covariates) are intentionally excluded to keep the repo lightweight
  and honest about scope.

See [`docs/metrics_interpretation.md`](docs/metrics_interpretation.md) for
a full discussion of how to interpret reported numbers.

---

## Resume Bullet

> Built **Wai**, a reproducible coastal water-level forecasting pipeline;
> implemented 8-constituent tidal harmonic feature engineering over NOAA
> CO-OPS data, trained four models (Persistence, HarmonicRidge, GradBoost,
> WaveGRU) with multi-horizon evaluation (6 min–24 h), added split-conformal
> prediction intervals, high-water alert detection, and inverse-distance
> spatial interpolation — all backed by 110 passing tests, an interactive
> Streamlit dashboard, automated HTML reports, and CI, with zero private data
> exposure.

---

## License

MIT — see [LICENSE](LICENSE).
