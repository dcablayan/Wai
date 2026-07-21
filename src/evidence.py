"""Content-based provenance for generated scientific evidence.

A Git commit cannot contain its own SHA, so embedding ``HEAD`` in a committed
report creates an unavoidable one-commit lag.  Wai instead fingerprints the
source, evaluation scripts, configuration, and demo inputs that determine the
reports while excluding the generated reports themselves.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable


EVIDENCE_DIRECTORIES = ("src", "scripts", "Hohonu-1", "data/demo")
EVIDENCE_FILES = (
    ".python-version",
    "app.py",
    "Makefile",
    "pyproject.toml",
    "requirements.txt",
    "uv.lock",
)
IGNORED_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}


def source_fingerprint(repo_root: str | Path) -> str:
    """Return a stable SHA-256 fingerprint of report-determining inputs."""

    root = Path(repo_root).resolve()
    digest = hashlib.sha256()
    for path in _evidence_paths(root):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def evidence_freshness(run_metadata: dict[str, Any] | None, repo_root: str | Path) -> dict[str, Any]:
    """Compare recorded and current source fingerprints at verification time."""

    current = source_fingerprint(repo_root)
    recorded = None if not run_metadata else run_metadata.get("source_fingerprint")
    return {
        "verification_method": "source_content_sha256_v1",
        "current_source_fingerprint": current,
        "recorded_source_fingerprint": recorded,
        "fresh_at_verification": bool(recorded and recorded == current),
        "legacy_git_sha": None if not run_metadata else run_metadata.get("git_sha"),
    }


def _evidence_paths(root: Path) -> Iterable[Path]:
    paths: set[Path] = set()
    for relative in EVIDENCE_FILES:
        path = root / relative
        if path.is_file():
            paths.add(path)
    for relative in EVIDENCE_DIRECTORIES:
        directory = root / relative
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            if IGNORED_PARTS.intersection(path.parts) or path.suffix in IGNORED_SUFFIXES:
                continue
            paths.add(path)
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())
