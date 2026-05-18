"""Start the Wai Streamlit dashboard, with a sandbox-safe smoke fallback."""

from __future__ import annotations

import socket
import subprocess
import sys
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _can_bind_localhost() -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return True
    except PermissionError:
        return False
    finally:
        sock.close()


def _smoke_check() -> None:
    sys.path.insert(0, str(ROOT))
    from app import run_forecast

    forecast = run_forecast("DEMO-HNL")
    required = [
        "timestamps",
        "actual",
        "persistence_pred",
        "harmonic_pred",
        "gradboost_pred",
        "harmonic_lower",
        "harmonic_upper",
    ]
    lengths = {key: len(forecast[key]) for key in required}
    if len(set(lengths.values())) != 1 or not next(iter(lengths.values()), 0):
        raise RuntimeError(f"Dashboard forecast arrays are not aligned: {lengths}")


def main() -> None:
    if _can_bind_localhost():
        raise SystemExit(
            subprocess.call(["streamlit", "run", "app.py"], cwd=ROOT)
        )

    print(
        "Local port binding is not permitted in this environment; "
        "running dashboard smoke check instead.",
        flush=True,
    )
    with redirect_stderr(StringIO()):
        _smoke_check()
    print("Dashboard smoke check passed. Run `streamlit run app.py` locally to view the UI.")


if __name__ == "__main__":
    main()
