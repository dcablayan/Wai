# Tidecast Reference Data

> Tidal prediction data for 10 Hohonu coastal monitoring station locations
> around Hawaii. Originally published in `dcablayan/tideformer` (public repo).

## About

These files contain NOAA-derived harmonic tidal **predictions** (not raw sensor
readings) at 6-minute intervals for July–October 2024.  The `prediction` column
holds water-level forecasts computed from NOAA harmonic tidal constituents.

These are **not** proprietary Hohonu sensor data.  They are equivalent to what
you would obtain by calling the NOAA CO-OPS `predictions` product for the
corresponding station locations.

## Schema

| Column     | Type       | Notes                             |
|------------|------------|-----------------------------------|
| dt         | str (UTC)  | ISO 8601 timestamp                |
| prediction | float      | Water level (feet, MLLW datum)    |

## Stations

| File                      | Station ID |
|---------------------------|------------|
| hohonu-12_tidecast.csv    | hohonu-12  |
| hohonu-16_tidecast.csv    | hohonu-16  |
| hohonu-37_tidecast.csv    | hohonu-37  |
| hohonu-74_tidecast.csv    | hohonu-74  |
| hohonu-142_tidecast.csv   | hohonu-142 |
| hohonu-160_tidecast.csv   | hohonu-160 |
| hohonu-168_tidecast.csv   | hohonu-168 |
| hohonu-186_tidecast.csv   | hohonu-186 |
| hohonu-191_tidecast.csv   | hohonu-191 |
| hohonu-207_tidecast.csv   | hohonu-207 |

## Usage

```python
from src.data.windowing import load_tidecast_dataframe, load_tidecast_series

# Wai-schema DataFrame
df = load_tidecast_dataframe("data/demo/tidecast/hohonu-37_tidecast.csv")

# Raw (times_hours, values) lists for prototype models
times, values = load_tidecast_series("data/demo/tidecast/hohonu-37_tidecast.csv")
```

## Benchmark results

See `reports/benchmark_results.md` after running:

    python -m scripts.run_benchmark
