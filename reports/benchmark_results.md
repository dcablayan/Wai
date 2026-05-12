# Wai Benchmark Results

Models from `src/models/prototypes.py` (ported from dcablayan/tideformer).
Evaluated on tidecast tidal-prediction data (NOAA-derived, Hawaiian stations).
Lookback: 24 steps (144 min at 6-min cadence) · Max windows: 2000

| Station | TinyTide RMSE | HarmonicNet RMSE | WaveGRU RMSE | SurgeNet RMSE |
| --- | --- | --- | --- | --- |
| hohonu-12_tidecast | 0.493 | 4.976 | 0.682 | 4.384 |
| hohonu-142_tidecast | 0.169 | 1.496 | 0.245 | 1.668 |
| hohonu-160_tidecast | 0.101 | 12.590 | 1.188 | 11.066 |
| hohonu-168_tidecast | 0.127 | 1.472 | 0.296 | 1.334 |
| hohonu-16_tidecast | 0.210 | 7.738 | 1.156 | 6.969 |
| hohonu-186_tidecast | 0.609 | 7.373 | 2.265 | 8.807 |
| hohonu-191_tidecast | 0.267 | 12.101 | 1.799 | 11.669 |
| hohonu-207_tidecast | 0.015 | 5.210 | 0.094 | 4.808 |
| hohonu-37_tidecast | 0.083 | 6.886 | 0.966 | 6.306 |
| hohonu-74_tidecast | 0.151 | 7.029 | 0.416 | 6.273 |

**Averages**
| Model | Mean RMSE |
| --- | --- |
| TinyTide | 0.222 |
| HarmonicNet | 6.687 |
| WaveGRU | 0.911 |
| SurgeNet | 6.328 |
