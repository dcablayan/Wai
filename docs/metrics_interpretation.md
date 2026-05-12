# Metrics Interpretation Guide

This document explains how to read the numbers produced by Wai honestly.
All results in this repository come from one of two data sources, each with
different characteristics and different implications for what the numbers mean.

---

## Two data sources, two different signals

### 1. Synthetic demo data (`data/demo/demo_water_levels.csv`)

Generated deterministically by `scripts/prepare_demo_data.py` using four tidal
constituents (M2, S2, K1, O1) plus Gaussian noise, a synthetic storm-surge
event, and a king-tide pulse.

**What the metrics mean:**
- Pipeline model metrics (`reports/model_metrics.json`) reflect how well
  Persistence and HarmonicRidge predict this *known, low-noise* synthetic
  signal on a temporal hold-out (last 25% of data).
- Because the signal is constructed from the same constituents the model uses
  as features, R² and RMSE will look better than real-world performance.
- **These numbers cannot be compared to published coastal forecasting benchmarks.**

### 2. NOAA-derived tidecast predictions (`data/demo/tidecast/*.csv`)

NOAA harmonic tidal predictions computed from official tidal constants for
10 coastal station locations in Hawaii. Originally published in
[dcablayan/tideformer](https://github.com/dcablayan/tideformer).

**What the metrics mean:**
- Prototype model metrics (`reports/benchmark_results.md`) reflect how well
  TinyTide, HarmonicNet, WaveGRU, and SurgeNet predict the *next 6-minute
  tidal value* from 144 minutes of context.
- Tidal predictions are **smooth and deterministic** — no sensor noise,
  no storm surge, no missing data. This makes them substantially easier to
  forecast than real sensor readings.
- **Tidecast RMSE will be lower than what you would achieve on real noisy
  sensor data at the same stations.**

---

## Unit differences

| Data source | Water-level units | RMSE units |
|-------------|-------------------|------------|
| Synthetic demo | meters (m) | meters |
| Tidecast predictions | feet (ft) | feet |

Do not compare RMSE values across these two datasets without unit conversion.
A tidecast RMSE of 0.222 ft ≈ 0.068 m.

---

## What benchmark results validate

`reports/benchmark_results.md` shows that:

1. **ʻAle Iki (Ripple)** (`TinyTidePrototype`, mean RMSE 0.222 ft) is the
   strongest prototype on smooth tidal predictions, outperforming naive
   persistence on a clean signal.
2. **Nalu Holo (Fast Wave)** (`WaveGRUPrototype`, mean RMSE 0.911 ft) is a
   useful complementary baseline using only smoothing — no harmonic knowledge
   required.
3. **Nalu Hoʻokani (Harmonic Wave)** and **ʻAle Piʻi (Rising Wave)**
   (`HarmonicNetPrototype` / `SurgeNetPrototype`, mean RMSE ~9 / ~8.5 ft)
   underperform on tidecast data because their harmonic fitting converges
   slowly on short windows and their residual heads add noise.

These results **validate that the prototype implementations are functional and
produce differentiated behaviour**. They do **not** claim operational readiness
or real-world deployment performance.

---

## What the numbers do NOT tell you

- **Operational skill** — real coastal forecasting requires meteorological
  forcing (wind, pressure, surge), multi-day horizons, and evaluation on
  real (noisy, gap-filled) sensor data.
- **Generalisation** — all results are on data geographically limited to
  Hawaii. Performance at Atlantic or Gulf coast stations may differ.
- **Production robustness** — prototype models are pure-Python research
  baselines (stdlib math only), not production-grade neural networks.
- **Multi-step skill** — all benchmarks use a one-step-ahead (horizon=1)
  setting. Multi-hour forecasting performance would be substantially lower.

---

## Interpreting the dashboard

Metrics shown in the Streamlit dashboard (`Model Performance Metrics` table)
are computed on **synthetic demo data** on a temporal hold-out test set.
All values shown are labelled `(m)` in the column headers.

High R² and low RMSE on this data reflect how well the models recover the
known synthetic signal — not how they would perform on live sensor streams.
