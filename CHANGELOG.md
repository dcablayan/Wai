# Changelog

This file records user-visible changes to Wai. Dates use the project release
date in Hawaii Standard Time.

## 2.0.0 - 2026-07-21

### Added

- Nationwide discovery of active NOAA CO-OPS water-level stations with a
  checked-in, offline-safe catalog and an explicit refresh command.
- Region filtering, station search, station metadata, and a nationwide map in
  the live control panel.
- On-demand NOAA observations, astronomical tide predictions, and available
  NOAA Operational Forecast System guidance on one UTC timeline.
- Observation-only support for stations where NOAA does not publish tidal
  predictions, including Great Lakes gauges.
- Beginner onboarding for Python setup, dashboard use, coastal datums, live
  data layers, station selection, and common recovery steps.
- Integrity-checked JSON model artifacts, report freshness checks, critical
  coverage enforcement, and expanded API and legacy-stack tests.

### Changed

- Reworked the Streamlit interface as a responsive control panel with clearer
  typography, fewer overlapping controls, denser metric cards, and reduced
  header dead space.
- Synchronized the tide-motion cross-section with the selected time-series
  point while reducing the land footprint and keeping annotations legible.
- Improved live-data caching, explicit refresh behavior, missing-value
  handling, station-aware datum defaults, and error messages.
- Updated the supported runtime to Python 3.13 with CI coverage for Python
  3.14 and a locked uv environment.
- Tightened evidence labeling so synthetic, mock, and live NOAA results remain
  visibly separate.

### Security

- Replaced executable router serialization with validated, integrity-checked
  JSON and rejected pickle/joblib model artifacts.
- Restricted API inputs, bounded remote fetches, removed unsafe archive
  extraction paths, and hardened timestamp and payload validation.

### Limitations

- Wai remains a research and product demonstration, not an operational flood,
  storm-surge, tsunami, navigation, or emergency-alert system.
- NOAA observations and deterministic tide predictions are public data, not
  Wai machine-learning forecasts. NOAA guidance availability varies by
  station and forecast system.
- Nationwide support means nationwide station discovery and on-demand fetches;
  it does not bulk-download every station's time series.
