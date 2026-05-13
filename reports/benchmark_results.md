# Wai Benchmark Results

Models from `src/models/prototypes.py` (ported from dcablayan/tideformer) plus a last-value persistence comparator.
Evaluated on tidecast tidal-prediction data (NOAA-derived, Hawaiian stations).
WaveGRUPrototype is a smoothing heuristic, not a real GRU. SurgeNetPrototype is a residual heuristic, not meteorological surge modeling.
Lookback: 24 steps (144 min at 6-min cadence) · Max windows: 2000

| Station | Persistence (last value) RMSE | ʻAle Iki (Ripple) RMSE | Nalu Hoʻokani (Harmonic Wave) RMSE | Nalu Holo (Fast Wave) RMSE | ʻAle Piʻi (Rising Wave) RMSE |
| --- | --- | --- | --- | --- | --- |
| hohonu-12_tidecast | 0.493 | 0.493 | 8.275 | 0.682 | 10.159 |
| hohonu-142_tidecast | 0.169 | 0.169 | 2.062 | 0.245 | 2.393 |
| hohonu-160_tidecast | 0.101 | 0.101 | 4.761 | 1.188 | 4.966 |
| hohonu-168_tidecast | 0.127 | 0.127 | 1.085 | 0.296 | 1.105 |
| hohonu-16_tidecast | 0.210 | 0.210 | 8.123 | 1.156 | 7.409 |
| hohonu-186_tidecast | 0.609 | 0.609 | 13.935 | 2.265 | 12.929 |
| hohonu-191_tidecast | 0.267 | 0.267 | 9.063 | 1.799 | 8.350 |
| hohonu-207_tidecast | 0.015 | 0.015 | 9.591 | 0.094 | 8.770 |
| hohonu-37_tidecast | 0.083 | 0.083 | 12.313 | 0.966 | 10.517 |
| hohonu-74_tidecast | 0.151 | 0.151 | 2.611 | 0.416 | 2.384 |

**Averages**
| Model | Mean RMSE |
| --- | --- |
| Persistence (last value) | 0.222 |
| ʻAle Iki (Ripple) | 0.222 |
| Nalu Hoʻokani (Harmonic Wave) | 7.182 |
| Nalu Holo (Fast Wave) | 0.911 |
| ʻAle Piʻi (Rising Wave) | 6.898 |
