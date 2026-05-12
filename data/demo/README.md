# Demo Data

> **This is entirely synthetic data. No real sensor measurements, private
> Hohonu data, or proprietary information is present.**

## How it was generated

Run `python -m scripts.prepare_demo_data` to regenerate from scratch.

The signal is composed of four tidal constituents (M2, S2, K1, O1) with
amplitudes tuned to be regionally plausible, plus:
- Gaussian measurement noise (σ ≈ 2 cm)
- A synthetic storm-surge event (peak +0.45 m, around day 20)
- A king-tide pulse (peak +0.25 m, around day 10)

## Schema

| Column       | Type    | Notes                          |
|--------------|---------|--------------------------------|
| timestamp    | str UTC | ISO 8601, UTC timezone         |
| station_id   | str     | DEMO-HNL or DEMO-SFO           |
| water_level  | float   | meters, MLLW datum             |
| datum        | str     | MLLW                           |
| units        | str     | m                              |
| lat          | float   | decimal degrees                |
| lon          | float   | decimal degrees                |
| source       | str     | DEMO_SYNTHETIC                 |

## Stations

| Station ID | Lat     | Lon       | Notes               |
|-----------|---------|-----------|---------------------|
| DEMO-HNL  | 21.3069 | -157.8583 | Hawaii-style diurnal tides |
| DEMO-SFO  | 37.7749 | -122.4194 | Pacific mixed semi-diurnal |
