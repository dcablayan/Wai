"""Data validation for Wai water-level DataFrames."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd

REQUIRED_COLUMNS = [
    "timestamp", "station_id", "water_level",
    "datum", "units", "lat", "lon", "source",
]

# Physical bounds covering all realistic coastal stations worldwide (meters)
WATER_LEVEL_RANGE = (-15.0, 30.0)


@dataclass
class ValidationReport:
    missing_timestamps: int = 0
    duplicate_timestamps: int = 0
    out_of_range_values: int = 0
    nan_values: int = 0
    timezone_issues: int = 0
    warnings: List[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not any([
            self.missing_timestamps,
            self.duplicate_timestamps,
            self.out_of_range_values,
            self.nan_values,
            self.timezone_issues,
        ])

    def __str__(self) -> str:
        lines = [
            f"ValidationReport(clean={self.is_clean})",
            f"  nan_values          : {self.nan_values}",
            f"  missing_timestamps  : {self.missing_timestamps}",
            f"  duplicate_timestamps: {self.duplicate_timestamps}",
            f"  out_of_range_values : {self.out_of_range_values}",
            f"  timezone_issues     : {self.timezone_issues}",
        ]
        if self.warnings:
            lines.append("  warnings:")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return "\n".join(lines)


def validate(df: pd.DataFrame) -> ValidationReport:
    """Run all validation checks and return a ValidationReport.

    Checks:
    - Required columns present
    - NaN water_level values
    - Timezone-aware timestamps
    - Per-station timestamp gaps (> 1.5× modal interval)
    - Duplicate (station_id, timestamp) pairs
    - Water level values outside physical plausible range
    """
    report = ValidationReport()

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    nan_mask = df["water_level"].isna()
    report.nan_values = int(nan_mask.sum())
    if report.nan_values:
        report.warnings.append(
            f"{report.nan_values} rows have NaN water_level"
        )

    ts = df["timestamp"]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        report.timezone_issues = len(df)
        report.warnings.append("timestamp column is not datetime dtype")
    else:
        if ts.dt.tz is None:
            report.timezone_issues = len(df)
            report.warnings.append(
                "Timestamps are timezone-naive; expected UTC-aware timestamps"
            )
        else:
            for station in df["station_id"].unique():
                sub = df[df["station_id"] == station].sort_values("timestamp")
                diffs = sub["timestamp"].diff().dropna()
                if diffs.empty:
                    continue
                mode_vals = diffs.mode()
                if mode_vals.empty:
                    continue
                expected = mode_vals.iloc[0]
                gaps = diffs[diffs > expected * 1.5]
                if not gaps.empty:
                    report.missing_timestamps += len(gaps)
                    report.warnings.append(
                        f"Station {station}: {len(gaps)} timestamp gaps detected"
                    )

    dup_mask = df.duplicated(subset=["station_id", "timestamp"])
    report.duplicate_timestamps = int(dup_mask.sum())
    if report.duplicate_timestamps:
        report.warnings.append(
            f"{report.duplicate_timestamps} duplicate (station_id, timestamp) pairs"
        )

    lo, hi = WATER_LEVEL_RANGE
    oor_mask = (~nan_mask) & ((df["water_level"] < lo) | (df["water_level"] > hi))
    report.out_of_range_values = int(oor_mask.sum())
    if report.out_of_range_values:
        report.warnings.append(
            f"{report.out_of_range_values} water_level values outside [{lo}, {hi}] m"
        )

    return report
