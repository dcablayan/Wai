# Wai

Wai is a research dashboard for exploring coastal water levels. It brings
public NOAA data, model estimates, uncertainty, and tide movement into one
control panel that is easier to inspect and compare.

> **Wai is a research tool, not an emergency or navigation system.** Do not use
> it for flood warnings, evacuation decisions, vessel operations, insurance,
> or infrastructure safety.

![CI](https://github.com/dcablayan/Wai/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.13%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## What you can do

- Browse active NOAA CO-OPS water-level stations across the United States.
- Compare measured water levels with NOAA astronomical tide predictions.
- View available NOAA Operational Forecast System guidance on the same UTC
  timeline.
- Follow an animated shoreline cross-section as the selected time changes.
- Replay held-out model estimates with uncertainty, error, and baseline
  comparisons.
- Connect a CSV export or another tide-gauge provider to the same pipeline.

Version 2 adds nationwide station discovery, live NOAA data layers, a clearer
dashboard, beginner onboarding, and stronger evidence and artifact checks.
The full change list is in [CHANGELOG.md](CHANGELOG.md).

## Start in a few minutes

Wai uses Python 3.13 or newer and [uv](docs/getting_started.md) for setup.

```bash
git clone https://github.com/dcablayan/Wai.git
cd Wai
uv sync --locked --all-extras
make dashboard
```

Streamlit prints the local address when the dashboard is ready. Open that
address in a browser, then choose a data source from the sidebar.

If this is your first Python project, follow
[docs/getting_started.md](docs/getting_started.md). It walks through installing
the tools, starting the dashboard, choosing a station, understanding datums,
and fixing common setup problems.

## Choose the right view

Wai keeps live public data separate from model backtests so the numbers are not
easy to misread.

| View | What it shows | What it means |
| --- | --- | --- |
| Live NOAA CO-OPS | Recent public observations, NOAA tide predictions, and available NOAA guidance | A live data monitor; NOAA predictions are not Wai machine-learning forecasts |
| Synthetic backtest | Held-out model estimates, observed outcomes, uncertainty, and error | A repeatable research test using generated demo data, not a live sensor feed |

In live mode, select a region or search by station name or ID. Wai downloads a
bounded time window only for the station you choose. No NOAA API key is needed.
Stations without published tide predictions still work as observation-only
monitors, including Great Lakes gauges.

The bundled station catalog makes the selector useful when NOAA discovery is
temporarily unavailable. Refresh it with:

```bash
make noaa-stations
```

## How the research pipeline works

The main modeling idea is simple:

1. Start with known tidal structure, such as NOAA tide predictions or harmonic
   time features.
2. Model the remaining difference with small statistical or machine-learning
   methods.
3. Compare the result with strong baselines, including persistence and the NOAA
   tide prediction itself.
4. Evaluate only on future, held-out time windows and report uncertainty and
   failure cases.

Run the complete offline evidence pipeline with:

```bash
make demo
make test
make coverage
```

Start with [reports/summary.json](reports/summary.json) for the machine-readable
index or [docs/research-report.md](docs/research-report.md) for the written
research summary.

### How to read the evidence

| Evidence | Use it for | Do not treat it as |
| --- | --- | --- |
| Synthetic demo | Checking code, splits, uncertainty, and model comparisons | Real-world forecast skill |
| NOAA-derived tidecast benchmark | Comparing simple methods on a smooth tidal signal | Noisy observed water levels |
| NOAA mock evaluation | Testing the offline API and evaluation path | Live NOAA performance |
| NOAA live evaluation | Comparing models and NOAA baselines on a short public-data window | Operational or seasonal validation |

Mock and live NOAA results are written to different files. The scientific
evidence audit checks that they remain separate.

## Use your own tide gauge

Wai can read a CSV without changing the forecasting code:

```bash
uv run python -m scripts.run_gauge_forecast \
  --csv my_gauge.csv \
  --station-id MY-GAUGE-01 \
  --timestamp-col time \
  --water-level-col level_ft \
  --units ft
```

For a reusable station catalog or a new provider adapter, see
[docs/onboarding_new_gauge.md](docs/onboarding_new_gauge.md).

## Advanced forecasting modes

Wai includes three numerical orchestration modes:

- `mini` is the fast default path.
- `ultra` can coordinate several numerical experts and verification steps.
- `legacy` keeps the original router available for regression testing.

No language model generates the water-level values. Numerical experts,
statistical combination, physical checks, and explicit fallback rules produce
the estimates.

```bash
uv run python -m scripts.run_orchestrated_forecast --mode mini --horizon-minutes 360
uv run python -m scripts.run_orchestrated_forecast --mode ultra --horizon-minutes 360
```

Technical details are in
[docs/forecast_orchestrator.md](docs/forecast_orchestrator.md).

## Important limits

- The checked-in demo metrics come from synthetic data and are sanity checks.
- Short NOAA API windows do not prove seasonal or extreme-event performance.
- The current checked-in reports do not include real wind, pressure, rainfall,
  or wave forcing.
- Split-conformal intervals are measured empirically because tidal time series
  do not guarantee the assumptions behind ideal coverage.
- Terrain in the tide animation is illustrative, not surveyed bathymetry.
- Wai does not provide validated storm-surge, tsunami, flood, or safety alerts.

Read [docs/model_card.md](docs/model_card.md) for intended use, failure modes,
and the exact evidence boundaries.

## Project map

```text
app.py        Streamlit control panel
src/          data adapters, models, forecasting, and verification
scripts/      repeatable evaluations and report builders
tests/        unit, integration, security, and evidence tests
data/         synthetic demo inputs and the NOAA station catalog
reports/      generated metrics and research reports
docs/         onboarding, model card, methods, and deeper explanations
```

## Useful commands

```bash
make dashboard       # start the control panel
make demo            # regenerate offline research evidence
make test            # run the test suite
make coverage        # enforce coverage checks
make noaa-stations   # refresh the nationwide station catalog
```

The locked environment is stored in `uv.lock`. A pip-compatible
`requirements.txt` is included for environments that cannot use uv.

## License

MIT — see [LICENSE](LICENSE).
