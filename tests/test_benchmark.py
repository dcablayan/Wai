"""Tests for scripts/run_benchmark.py output creation."""

import csv
import math
import tempfile
from pathlib import Path

import pytest

from scripts.run_benchmark import benchmark_station


def _write_tidecast_csv(path: Path, n: int = 300) -> None:
    """Write a minimal synthetic tidecast CSV in the hohonu format."""
    import math as m
    base = "2024-07-01T00:00:00+00:00"
    rows = []
    for i in range(n):
        minutes = i * 6
        hours = minutes / 60
        value = 2.0 * m.sin(2 * m.pi * hours / 12.42) + 0.5
        ts_minutes = minutes
        h, mi = divmod(ts_minutes, 60)
        day, h = divmod(h, 24)
        rows.append({
            "dt": f"2024-07-{1 + day:02d}T{h:02d}:{mi:02d}:00+00:00",
            "prediction": f"{value:.4f}",
        })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dt", "prediction"])
        writer.writeheader()
        writer.writerows(rows)


def test_benchmark_station_returns_expected_models():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "demo-station_tidecast.csv"
        _write_tidecast_csv(csv_path, n=300)
        results = benchmark_station(csv_path)
    assert set(results.keys()) == {
        "Persistence", "TinyTide", "HarmonicNet", "WaveGRU", "SurgeNet"
    }


def test_benchmark_station_rmse_are_finite():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "demo-station_tidecast.csv"
        _write_tidecast_csv(csv_path, n=300)
        results = benchmark_station(csv_path)
    for name, score in results.items():
        assert not math.isnan(score), f"{name} returned NaN RMSE"
        assert score >= 0.0, f"{name} returned negative RMSE"


def test_benchmark_station_too_short_returns_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "short_tidecast.csv"
        _write_tidecast_csv(csv_path, n=5)
        results = benchmark_station(csv_path)
    assert results == {}


def test_benchmark_writes_report_file(tmp_path, monkeypatch):
    """main() writes benchmark_results.md when tidecast CSVs are present."""
    tidecast_dir = tmp_path / "tidecast"
    tidecast_dir.mkdir()
    _write_tidecast_csv(tidecast_dir / "station-a_tidecast.csv", n=300)

    reports_dir = tmp_path / "reports"

    import scripts.run_benchmark as bm
    monkeypatch.setattr(bm, "TIDECAST_DIR", tidecast_dir)
    monkeypatch.setattr(bm, "REPORTS_DIR", reports_dir)

    bm.main()

    out = reports_dir / "benchmark_results.md"
    assert out.exists(), "benchmark_results.md was not created"
    content = out.read_text()
    assert "station-a_tidecast" in content
    assert "Persistence" in content
    assert "ʻAle Iki" in content
