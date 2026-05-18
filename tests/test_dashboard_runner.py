"""Tests for the dashboard runner fallback."""

from __future__ import annotations

import socket

from scripts import run_dashboard


def test_can_bind_localhost_handles_permission_error(monkeypatch):
    class DeniedSocket:
        def bind(self, address):
            raise PermissionError("blocked")

        def close(self):
            pass

    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: DeniedSocket())

    assert run_dashboard._can_bind_localhost() is False
