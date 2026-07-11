# Onboarding a New Tide Gauge

Wai accepts observations from any gauge provider. This guide covers the three
onboarding paths, from a one-off CSV to a fully registered provider.

## Concepts

- **Canonical schema** (`src/data/canonicalize.py`): every source is
  translated to one vocabulary (`timestamp_utc`, `water_level_m` in meters,
  `datum`, QC fields) before anything downstream sees it.
- **Roles, not vendors**: the orchestrator wants a *local* gauge (the station
  being forecast) and optionally a *regional reference* (a nearby
  well-instrumented station, e.g. NOAA) plus its tide predictions. Any
  provider can fill either role.
- **Regularization** (`src/data/regularize.py`): observations are snapped to
  a regular grid at the gauge's native cadence, despiked (rolling-median MAD
  test), and small gaps are interpolated. Long outages stay NaN so models
  never train across invented data.
- **No tide product needed**: without tide predictions, the
  `harmonic_fallback` expert fits tidal constituents to the gauge's own
  history (needs ≥48 h of data), so any gauge beats plain persistence.

## Path 1: one-off CSV export

Any delimited export works; map the column names on the command line:

```bash
python -m scripts.run_gauge_forecast \
    --csv my_gauge.csv \
    --station-id MY-GAUGE-01 \
    --timestamp-col time --water-level-col level_ft \
    --units ft --datum MLLW --lat 21.30 --lon -157.86 \
    --horizon-minutes 180
```

Add a reference station and tide predictions when available:

```bash
python -m scripts.run_gauge_forecast \
    --csv my_gauge.csv --station-id MY-GAUGE-01 \
    --reference-csv noaa_water_level.csv --reference-station-id 1612340 \
    --tide-csv noaa_predictions.csv
```

## Path 2: register the station in a catalog

Describe the station once in `data/stations.json`:

```json
{
  "stations": [
    {
      "station_id": "MY-GAUGE-01",
      "name": "Harbor east dock",
      "source": "csv",
      "source_config": {"path": "data/my_gauge.csv",
                        "column_map": {"timestamp": "time",
                                       "water_level": "level_ft"},
                        "units": "ft", "station_id": "MY-GAUGE-01"},
      "latitude": 21.30,
      "longitude": -157.86,
      "datum": "NAVD88",
      "cadence_minutes": 5,
      "reference_station_id": "1612340",
      "reference_source": "noaa_coops",
      "datum_offsets": {"NAVD88->MLLW": 0.482}
    }
  ]
}
```

Then:

```bash
python -m scripts.run_gauge_forecast --catalog data/stations.json \
    --station-id MY-GAUGE-01
```

Notes:

- `cadence_minutes` is the native sampling interval; omit it to infer from
  the data.
- `datum_offsets` are additive offsets in meters (`level_to = level_from +
  offset`). When the gauge and its reference are on different datums, Wai
  converts using these offsets; without them, mixing datums fails closed by
  design. NOAA publishes per-station datum tables at
  https://tidesandcurrents.noaa.gov/datums.html.

## Path 3: implement a provider adapter

For a vendor API, subclass `DataSource` and register it
(`src/data/sources.py`):

```python
from src.data.canonicalize import canonicalize_frame
from src.data.sources import DataSource, register_source

@register_source("my_vendor")
class MyVendorSource(DataSource):
    def __init__(self, base_url: str, api_key_env: str = "MY_VENDOR_KEY"):
        ...

    def fetch_observations(self, station_id, start, end, **kwargs):
        payload = ...  # provider-specific HTTP call
        frame = ...    # DataFrame with timestamp/water_level/lat/lon/datum
        return canonicalize_frame(frame, source="MY_VENDOR",
                                  record_type="observation", units="m")
```

The adapter's only contract is returning canonical frames. Once registered,
catalog stations can use `"source": "my_vendor"` with constructor arguments
under `source_config`, and everything downstream (regularization, routing,
experts, verifier) works unchanged. `fetch_tide_predictions` is optional —
providers without a tide product simply rely on the harmonic fallback.

## What quality to expect

| Data available | Experts engaged |
| --- | --- |
| Local observations only, <48 h | persistence only |
| Local observations only, ≥48 h | persistence + harmonic fallback |
| + reference observations & tide predictions | + NOAA-residual and regional-residual transfer |
| + `residual_scale`/`lag_minutes` tuned in the catalog | best regional transfer quality |

Tabular research models (`HarmonicRidge`, gradient boosting) are also fully
provider-agnostic: convert canonical frames with
`src.data.schema.to_model_frame`, regularize, and pass
`cadence_minutes` to `build_feature_matrix` so lag/rolling windows keep their
physical durations on any grid.
