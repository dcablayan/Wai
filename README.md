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

## Features

| Feature | Details |
|---------|---------|
| Demo data loader | Reproducible synthetic dataset (2 stations, 90 days, 6-min cadence) |
| NOAA CO-OPS loader | Live ingestion from the public API (no key required) |
| Data validation | NaN, gap, duplicate, out-of-range, and timezone checks |
| Tidal feature engineering | M2/S2/K1/O1/N2 harmonic sin/cos + lags + rolling windows |
| Persistence baseline | Last-value naive baseline |
| Harmonic Ridge model | Linear tidal regression, honest R²/RMSE reporting |
| Metrics | MAE, RMSE, R², NSE, Pearson correlation |
| Streamlit dashboard | Time series, forecast overlay, anomaly markers, station map |
| HTML report generation | Per-station report with stats, metrics, anomaly table |
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

# 4. Generate HTML station reports
python -m scripts.generate_report

# 5. Launch the interactive dashboard
streamlit run app.py
```

Open `http://localhost:8501` to explore the dashboard.

---

## Demo Data Disclaimer

> **All data in `data/demo/` is synthetically generated and does not represent
> real sensor measurements.** The water-level signal is constructed from
> realistic tidal constituents (M2, S2, K1, O1) with Gaussian noise, a
> synthetic storm-surge event, and a king-tide pulse. No private Hohonu data,
> API keys, credentials, or proprietary information is present anywhere in
> this repository.
>
> NOAA data, when loaded via `load_noaa_data()`, is fetched from the public
> NOAA CO-OPS API in real time and is not stored in this repo.

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

The current implementation uses two models:

**1. Persistence baseline** — forecasts the last observed value for all
future steps. Provides a performance floor that any useful model must beat.

**2. Harmonic Ridge** — fits a Ridge-regularised linear regression over
tidal constituent sin/cos features (M2, S2, K1, O1, N2), lagged observations
at 6–240 minute windows, and rolling mean/std features. Captures the dominant
predictable tidal signal without deep learning infrastructure.

Both models are evaluated on a temporal hold-out (last 25% of data per
station) to prevent data leakage. Metrics are saved to
`reports/model_metrics.json`.

See [`docs/modeling.md`](docs/modeling.md) for full details and the roadmap
to LSTM/Transformer extensions.

---

## Sample Outputs

After running the quickstart commands:

- `reports/model_metrics.json` — MAE/RMSE/R² for each model and station
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
│   │   └── validation.py      Schema and quality validation
│   ├── features/
│   │   └── engineering.py     Tidal harmonics, lags, rolling windows
│   ├── models/
│   │   ├── baseline.py        Persistence + HarmonicRidge models
│   │   └── metrics.py         MAE / RMSE / R² / NSE
│   └── reporting/
│       └── report.py          HTML report generation
├── scripts/
│   ├── prepare_demo_data.py   Generate synthetic demo CSV
│   ├── train_baseline.py      Train models, save metrics JSON
│   └── generate_report.py     Build HTML reports
├── app.py                     Streamlit dashboard
├── data/demo/                 Synthetic demo data (safe to commit)
├── reports/                   Generated outputs
├── tests/                     pytest test suite
├── docs/
│   ├── architecture.md        Pipeline diagram and design decisions
│   └── modeling.md            Model details, metrics, next steps
├── .github/workflows/ci.yml   GitHub Actions CI
├── requirements.txt
└── pyproject.toml
```

---

## Limitations

- Demo results are on **synthetic data** and cannot be compared to published
  benchmarks.
- The harmonic model does not incorporate meteorological forcing (wind, pressure),
  so it cannot predict storm surge from first principles.
- NOAA API requests are limited to 31-day windows per call; longer time series
  require multiple requests.
- No hyperparameter tuning is performed; Ridge alpha=1.0 is a reasonable
  default, not an optimised one.

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

> Built **Wai**, a reproducible coastal water-level forecasting pipeline
> in Python; implemented tidal harmonic feature engineering over NOAA CO-OPS
> observations, trained Ridge and persistence baseline models (reporting
> MAE/RMSE/R² on held-out test sets), and shipped an interactive Streamlit
> dashboard with anomaly detection, station maps, and automated HTML reports
> — all with a CI-tested synthetic demo dataset and zero private data exposure.

---

## License

MIT — see [LICENSE](LICENSE).
