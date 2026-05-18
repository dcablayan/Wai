# Wai Portfolio Case Study

## Problem

Water-level forecasts need to beat serious baselines, not just look smooth on
a chart. The research question was whether tide-aware statistical modeling
could improve prediction against rolling persistence and NOAA tidal prediction
baselines while staying honest about scope.

## What I Built

I built a reproducible research demo with separated evidence tracks:

- Synthetic sanity checks with leakage guards, horizon evaluation, rolling
  origin folds, event metrics, and conformal intervals.
- A tidecast prototype benchmark with a persistence comparator.
- NOAA mock and live evaluation paths that keep mock artifacts separate from
  real NOAA output.
- A Streamlit dashboard for inspecting forecasts, uncertainty, alerts, and
  benchmark tables.
- Generated research figures and summary docs for portfolio review.

## Result

On the synthetic direct-horizon benchmark, HarmonicRidge reduced average MAE
against rolling persistence at every horizon:

| Horizon | Persistence MAE | HarmonicRidge MAE | Direction |
| --- | ---: | ---: | --- |
| 1-step / 6 min | 0.0262 m | 0.0179 m | Improved |
| 6 h | 0.5329 m | 0.0280 m | Improved |
| 12 h | 0.4050 m | 0.0297 m | Improved |
| 24 h | 0.1316 m | 0.0299 m | Improved |

The NOAA mock track did not show a hybrid win over NOAA tidal prediction,
which is an important honest result: the project reports losses when the
baseline wins.

## Why It Matters

The project demonstrates research discipline: temporal splits, baseline
comparisons, separated evidence types, generated reports, and tests for leakage
and NOAA mock integrity. The strongest claim is narrow and defensible:
tide-aware features improved the synthetic benchmark against rolling
persistence. The repo does not claim operational forecasting skill.

## Tech Stack

Python, pandas, NumPy, scikit-learn, statsmodels, Streamlit, Plotly, pytest,
pytest-cov, Make, SVG report generation, NOAA CO-OPS public API integration.

## Honest Limitations

- Synthetic and mock metrics are not operational NOAA proof.
- No meteorological forcing is included, so storm surge is not modeled from
  physical drivers.
- No live NOAA artifact is checked into this snapshot.
- The model set is intentionally lightweight; no deep learning is claimed.
- Conformal interval coverage degraded on event samples.
- The station count is too small for claims about broad coastal
  generalization.
