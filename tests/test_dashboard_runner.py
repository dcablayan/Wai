"""Tests for the dashboard runner fallback."""

from __future__ import annotations

import socket

from streamlit.testing.v1 import AppTest

from scripts import run_dashboard


def test_can_bind_localhost_handles_permission_error(monkeypatch):
    class DeniedSocket:
        def bind(self, address):
            raise PermissionError("blocked")

        def close(self):
            pass

    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: DeniedSocket())

    assert run_dashboard._can_bind_localhost() is False


def test_control_panel_renders_without_frontend_exceptions(monkeypatch):
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "2")
    dashboard = AppTest.from_file("app.py").run(timeout=30)

    assert not list(dashboard.exception)
    assert [tab.label for tab in dashboard.tabs][0] == "Control Center"
    assert len(dashboard.tabs) == 7
    assert len(dashboard.get("plotly_chart")) >= 10
    assert len(dashboard.dataframe) >= 5
