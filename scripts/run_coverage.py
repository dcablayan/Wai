"""Run coverage for Wai with a pytest-cov fallback.

The preferred path uses pytest-cov when it is installed. Some lightweight
portfolio/demo environments have pytest but not pytest-cov, so this script
falls back to Python's stdlib ``trace`` module and prints a compact
term-missing style report for ``src`` and ``scripts``.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import trace
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
COVERAGE_ROOTS = (ROOT / "src", ROOT / "scripts")
FALLBACK_EXCLUDES = {Path(__file__).resolve()}


def _run_pytest_cov() -> int:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "-v",
        "--cov=src",
        "--cov=scripts",
        "--cov-report=term-missing",
    ]
    return subprocess.call(cmd, cwd=ROOT)


def _format_ranges(lines: list[int]) -> str:
    if not lines:
        return ""
    ranges: list[str] = []
    start = prev = lines[0]
    for line in lines[1:]:
        if line == prev + 1:
            prev = line
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = line
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ", ".join(ranges)


def _coverage_files() -> list[Path]:
    files: list[Path] = []
    for root in COVERAGE_ROOTS:
        files.extend(
            path
            for path in root.rglob("*.py")
            if "__pycache__" not in path.parts and path.resolve() not in FALLBACK_EXCLUDES
        )
    return sorted(files)


def _run_stdlib_trace() -> int:
    import pytest

    print(
        "pytest-cov is not installed; using stdlib trace fallback for "
        "src/ and scripts/."
    )
    tracer = trace.Trace(
        count=True,
        trace=False,
        # Do not ignore pytest's install directory: pytest calls into the test
        # functions, and those tests call project code. Ignoring pytest frames
        # prevents stdlib trace from following that call chain. The final
        # report is filtered back to src/ and scripts/.
        ignoredirs=[],
    )
    pytest_code = tracer.runfunc(pytest.main, ["tests/", "-v"])
    results = tracer.results()
    counts = {
        (str(Path(filename).resolve()), lineno): count
        for (filename, lineno), count in results.counts.items()
    }

    rows: list[tuple[str, int, int, int, str]] = []
    total_statements = 0
    total_missing = 0

    for path in _coverage_files():
        executable = sorted(
            line
            for line in trace._find_executable_linenos(str(path))  # type: ignore[attr-defined]
            if isinstance(line, int) and line > 0
        )
        if not executable:
            continue
        filename = str(path.resolve())
        covered = {lineno for (fname, lineno), count in counts.items() if fname == filename and count > 0}
        missing = [line for line in executable if line not in covered]
        statements = len(executable)
        missed = len(missing)
        coverage_pct = round(100 * (statements - missed) / statements)
        rows.append((
            str(path.relative_to(ROOT)),
            statements,
            missed,
            coverage_pct,
            _format_ranges(missing),
        ))
        total_statements += statements
        total_missing += missed

    total_pct = (
        round(100 * (total_statements - total_missing) / total_statements)
        if total_statements
        else 100
    )

    name_width = max([len("Name"), *(len(row[0]) for row in rows)], default=4)
    print()
    print(
        f"{'Name':<{name_width}}  {'Stmts':>5}  {'Miss':>5}  "
        f"{'Cover':>5}  Missing"
    )
    print(
        f"{'-' * name_width}  {'-' * 5:>5}  {'-' * 5:>5}  "
        f"{'-' * 5:>5}  {'-' * 7}"
    )
    for name, statements, missed, coverage_pct, missing in rows:
        print(
            f"{name:<{name_width}}  {statements:>5}  {missed:>5}  "
            f"{coverage_pct:>4}%  {missing}"
        )
    print(
        f"{'TOTAL':<{name_width}}  {total_statements:>5}  {total_missing:>5}  "
        f"{total_pct:>4}%"
    )
    return int(pytest_code)


def main() -> None:
    if importlib.util.find_spec("pytest_cov") is not None:
        raise SystemExit(_run_pytest_cov())
    raise SystemExit(_run_stdlib_trace())


if __name__ == "__main__":
    main()
