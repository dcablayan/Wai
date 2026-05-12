"""Generate synthetic demo water-level data for Wai.

This script creates a realistic-looking (but entirely fabricated) coastal
water-level dataset for two fictional demo stations.  The signal is built
from the four dominant tidal constituents (M2, S2, K1, O1) plus Gaussian
noise, a synthetic storm-surge event, and a king-tide pulse.

Usage
-----
    python -m scripts.prepare_demo_data

Output
------
    data/demo/demo_water_levels.csv   (schema-conformant Wai CSV)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from repo root whether run as a module or directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

DEMO_STATIONS = [
    {
        "station_id": "DEMO-HNL",
        "lat": 21.3069,
        "lon": -157.8583,
        "description": "Honolulu demo station (synthetic)",
        # Hawaii is predominantly diurnal — weaker M2, stronger K1/O1
        "amplitudes": {"M2": 0.18, "S2": 0.05, "K1": 0.22, "O1": 0.18},
    },
    {
        "station_id": "DEMO-SFO",
        "lat": 37.7749,
        "lon": -122.4194,
        "description": "San Francisco demo station (synthetic)",
        # Pacific coast US — mixed semi-diurnal
        "amplitudes": {"M2": 0.55, "S2": 0.12, "K1": 0.35, "O1": 0.30},
    },
]

CONSTITUENT_PERIODS = {
    "M2": 12.4206,
    "S2": 12.0000,
    "K1": 23.9345,
    "O1": 25.8193,
}


def _tidal_signal(
    t_hours: np.ndarray,
    amplitudes: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compose a synthetic tidal water-level signal."""
    signal = np.zeros(len(t_hours))
    for name, amp in amplitudes.items():
        period = CONSTITUENT_PERIODS[name]
        phase = rng.uniform(0, 2 * np.pi)
        signal += amp * np.sin(2 * np.pi * t_hours / period + phase)

    # Gaussian measurement noise (~2 cm std dev)
    signal += rng.normal(0, 0.02, len(t_hours))

    # Storm-surge event around day 20 (4-hour half-width)
    surge_center = 20 * 24.0
    signal += 0.45 * np.exp(-((t_hours - surge_center) ** 2) / (2 * 4.0**2))

    # King-tide pulse around day 10 (2-hour half-width)
    kt_center = 10 * 24.0
    signal += 0.25 * np.exp(-((t_hours - kt_center) ** 2) / (2 * 2.0**2))

    return signal


def generate_demo_data(
    days: int = 90,
    freq_minutes: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a DataFrame of synthetic water-level observations."""
    rng = np.random.default_rng(seed)
    n_steps = int(days * 24 * 60 / freq_minutes)
    start = pd.Timestamp("2024-01-01", tz="UTC")
    timestamps = pd.date_range(start, periods=n_steps, freq=f"{freq_minutes}min")
    t_hours = np.arange(n_steps) * freq_minutes / 60.0

    rows = []
    for station in DEMO_STATIONS:
        wl = _tidal_signal(t_hours, station["amplitudes"], rng)
        for i, ts in enumerate(timestamps):
            rows.append({
                "timestamp": ts,
                "station_id": station["station_id"],
                "water_level": round(float(wl[i]), 4),
                "datum": "MLLW",
                "units": "m",
                "lat": station["lat"],
                "lon": station["lon"],
                "source": "DEMO_SYNTHETIC",
            })

    return pd.DataFrame(rows)


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "data" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "demo_water_levels.csv"

    print("Generating synthetic demo data …")
    df = generate_demo_data(days=90)
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df):,} rows to {out_path}")
    for sid in df["station_id"].unique():
        sub = df[df["station_id"] == sid]["water_level"]
        print(
            f"  {sid}: {len(sub):,} obs | "
            f"range [{sub.min():.3f}, {sub.max():.3f}] m | "
            f"mean {sub.mean():.3f} m"
        )
    print()
    print("NOTE: This is SYNTHETIC demo data only — not real sensor measurements.")
    print("      Clearly labeled as source=DEMO_SYNTHETIC in the CSV.")


if __name__ == "__main__":
    main()
