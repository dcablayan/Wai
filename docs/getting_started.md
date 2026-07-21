# Getting Started with Wai

This guide takes you from a new checkout to reading live nationwide NOAA water
levels. You do not need coastal-science, machine-learning, or NOAA API
experience.

## What Wai is

Wai is a research control panel with two deliberately separate modes:

- **Synthetic backtest** teaches and audits how the research models behave on
  held-out synthetic data.
- **Live NOAA CO-OPS** displays current public station observations and NOAA
  guidance without pretending those values are Wai machine-learning forecasts.

Wai is not an emergency-warning system. Do not use it for navigation, flood
evacuation, or life-safety decisions.

## 1. Install the project

You need Git and Python 3.13 or newer. The recommended package manager is
[`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/dcablayan/Wai.git
cd Wai
uv sync --locked --all-extras
```

The lock file pins the reproducible environment. There are no API keys or
secrets to configure for NOAA public data.

## 2. Build and verify the research demo

```bash
make demo
make test
```

`make demo` regenerates the synthetic data, models, reports, evidence audit,
and summary. `make test` verifies the data contracts and dashboard paths.

## 3. Start the control panel

```bash
make dashboard
```

Open the local address printed in the terminal, normally
`http://127.0.0.1:8501`. If another Wai server is already running, Streamlit
may choose the next available port.

## 4. Use the nationwide NOAA monitor

In the sidebar:

1. Set **Data source** to `Live NOAA CO-OPS`.
2. Use **Region** to reduce the nationwide catalog, or leave it at
   `All regions`.
3. Open **NOAA station** and type a station name or seven-digit ID.
4. Start with the default **Monitor window** and **Vertical datum**.
5. Choose the available **Guidance layer**.

The app discovers all active water-level stations from NOAA's Metadata API.
It does not download every station's time series on each page load. Instead,
it fetches the selected station's bounded time window, which keeps the UI fast
and respects a public service.

The checked-in `data/noaa_active_stations.json` file is a NOAA-sourced snapshot
used when live station discovery is unavailable. Refresh it with:

```bash
make noaa-stations
```

The command validates advertised row counts, station-ID uniqueness, required
names and regions, coordinate ranges, tide-prediction capability, and Great
Lakes classification before saving the snapshot.

## What the controls mean

### Monitor window

The recent history to request and display. Start with 72 hours. Seven days is
useful for context but transfers more data.

### Vertical datum

A datum is the zero reference for a water-level measurement. Values on
different datums must not be compared directly.

- `MLLW` is the default for tidal coastal stations.
- `IGLD` is the default for Great Lakes stations.
- `STND` is the station datum and is the safest general fallback for
  non-tidal gauges.
- `MSL`, `MHHW`, `NAVD`, and `LWD` appear only in station-type-appropriate
  option lists; NOAA may still reject a datum that is not published for an
  individual station.

Keep the first option unless you have a specific scientific reason to change
the reference.

### Guidance layer

- **Observations only** is live measured water level with no comparison
  baseline. It appears for active stations that do not publish astronomical
  tide predictions.
- **Astronomical tide** is NOAA's deterministic harmonic tide prediction. It
  does not include weather-driven surge.
- **NOAA OFS guidance** is hydrodynamic model guidance when NOAA offers the
  product at that station. Availability is checked by the real Data API; the
  app does not fabricate a fallback.

## How to read the cards and graphs

- **Latest observed level** is the newest station measurement in meters on the
  selected datum.
- **Observation age** is the difference between the current UTC time and the
  newest observation. A large value can indicate provider or station delay.
- **Observed minus prediction/guidance** is positive when measured water is
  higher than the selected baseline and negative when it is lower.
- **MAE** is mean absolute error over aligned samples. Lower is better, but it
  is a descriptive comparison here—not proof of future forecasting skill.
- **Aligned data quality** is the share of observed timestamps that found an
  exact six-minute baseline match.
- The nationwide map shows all active catalog stations; the selected station
  is highlighted.

## Common problems

### “Using the bundled NOAA station snapshot”

The Metadata API could not be reached. You can still browse the last bundled
NOAA catalog. Check your network, then refresh the browser or run
`make noaa-stations`.

### “NOAA CO-OPS could not be reached”

The selected live time series failed. Wai intentionally shows no value rather
than substituting demo data. Retry after a minute, use a shorter window, or
confirm the station on NOAA Tides & Currents.

### “OFS guidance is unavailable”

OFS is station-dependent. Switch back to astronomical tide or observations
only. This is not an installation failure.

### The selected datum is rejected

Return to the first datum option. NOAA station datum coverage differs even
among stations of the same broad type.

### The dashboard says evidence is stale

Source code changed after the reports were generated. Rebuild them:

```bash
make demo
uv run python -m scripts.check_report_freshness
```

### The normal port is busy

Stop the other Streamlit process or run:

```bash
uv run streamlit run app.py --server.port 8503
```

## Beginner-safe development loop

```bash
make noaa-stations       # optional: refresh nationwide metadata
make demo                # rebuild research evidence
make test                # run the test suite
make dashboard           # inspect the UI
```

Useful code locations:

- `app.py`: Streamlit controls and visualizations.
- `src/data/noaa_catalog.py`: nationwide metadata discovery and validation.
- `src/data/noaa_live.py`: bounded selected-station observation/guidance joins.
- `src/data/noaa.py`: canonical NOAA Data API adapter.
- `scripts/sync_noaa_stations.py`: reproducible station-catalog refresh.
- `tests/test_noaa_catalog.py`: catalog quality and fallback tests.

## Data and privacy

NOAA observations and metadata are public. Wai sends the selected NOAA station
ID, date window, product, datum, metric units, and GMT timezone to NOAA's public
API. It does not require an account, store credentials, or upload local files
when using the live monitor.

For scientific limitations and evidence scope, continue with
[`model_card.md`](model_card.md) and
[`metrics_interpretation.md`](metrics_interpretation.md).
