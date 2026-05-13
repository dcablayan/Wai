# Model Branding Notes

Internal class names are kept for compatibility with existing imports and
saved artifacts. They are not capability claims.

| Internal name | Public interpretation |
| --- | --- |
| `TinyTidePrototype` | Lightweight next-step prototype with simple tidal covariates and a skip connection |
| `HarmonicNetPrototype` | Harmonic projection baseline over tidal periods |
| `WaveGRUPrototype` | Smoothing heuristic; not a real GRU and not a deep-learning model |
| `SurgeNetPrototype` | Residual heuristic; not meteorological surge modeling |
| `TsunamiSentinelPrototype` | Anomaly toy; not a validated tsunami detector |
| `HarmonicRidgeModel` | Ridge regression over tidal, lag, and rolling features |
| `GradBoostModel` | scikit-learn gradient boosting over the same feature matrix |

Hawaiian display names may appear in the dashboard and older reports as
portfolio branding. The scientific documentation should use the plain
interpretations above when describing model capability.
