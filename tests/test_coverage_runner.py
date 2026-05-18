"""Tests for the coverage fallback runner."""

from __future__ import annotations

from pathlib import Path

from scripts import run_coverage


def test_format_ranges_compacts_consecutive_lines():
    assert run_coverage._format_ranges([1, 2, 3, 7, 9, 10]) == "1-3, 7, 9-10"


def test_coverage_files_excludes_runner_itself():
    files = {path.resolve() for path in run_coverage._coverage_files()}
    assert Path(run_coverage.__file__).resolve() not in files
