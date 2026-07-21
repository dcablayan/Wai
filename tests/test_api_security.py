"""Real-interface security tests for the bounded Hohonu API."""

from __future__ import annotations

import importlib
import asyncio
from pathlib import Path
import sys

import pytest


pytest.importorskip("fastapi")
pytest.importorskip("httpx2")
from fastapi.testclient import TestClient


LEGACY_DIR = Path(__file__).resolve().parent.parent / "Hohonu-1"
sys.path.insert(0, str(LEGACY_DIR))
api = importlib.import_module("api_server")
sys.path.remove(str(LEGACY_DIR))


@pytest.fixture(autouse=True)
def configured_api(monkeypatch):
    monkeypatch.setenv("WAI_API_KEY", "test-secret")
    api._RATE_LIMITER.clear()
    monkeypatch.setattr(
        api,
        "run_pipeline",
        lambda **kwargs: {"forecast": [0.1], "node_name": kwargs["node_name"]},
    )


def test_health_is_public_but_prediction_requires_api_key():
    client = TestClient(api.app)
    assert client.get("/health").status_code == 200
    assert client.post("/predict", json={"node_name": "safe-node"}).status_code == 401
    response = client.post(
        "/predict",
        headers={"X-API-Key": "test-secret"},
        json={"node_name": "safe-node", "steps": 24},
    )
    assert response.status_code == 200
    assert response.json()["node_name"] == "safe-node"


@pytest.mark.parametrize(
    "payload",
    [
        {"node_name": "../secret"},
        {"node_name": "safe", "steps": 100_000},
        {"node_name": "safe", "include_lstm": True},
        {"node_name": "safe", "candidate_profile": "broad"},
        {"node_name": "safe", "unknown_option": True},
    ],
)
def test_prediction_rejects_unbounded_or_unsafe_inputs(payload):
    response = TestClient(api.app).post(
        "/predict", headers={"X-API-Key": "test-secret"}, json=payload
    )
    assert response.status_code == 422


def test_batch_requires_nodes_and_caps_batch_size():
    client = TestClient(api.app)
    headers = {"X-API-Key": "test-secret"}
    assert client.post("/batch-predict", headers=headers, json={"node_names": []}).status_code == 422
    assert client.post(
        "/batch-predict",
        headers=headers,
        json={"node_names": [f"node-{idx}" for idx in range(5)]},
    ).status_code == 422
    response = client.post(
        "/batch-predict",
        headers=headers,
        json={"node_names": ["node-1", "node-2"]},
    )
    assert response.status_code == 200
    assert sorted(response.json()["predictions"]) == ["node-1", "node-2"]


def test_prediction_fails_closed_when_api_key_is_not_configured(monkeypatch):
    monkeypatch.delenv("WAI_API_KEY")
    response = TestClient(api.app).post(
        "/predict", headers={"X-API-Key": "anything"}, json={"node_name": "safe"}
    )
    assert response.status_code == 503


def test_oversized_body_is_rejected_before_prediction():
    response = TestClient(api.app).post(
        "/predict",
        headers={
            "X-API-Key": "test-secret",
            "Content-Length": str(api.MAX_REQUEST_BYTES + 1),
        },
        content=b"{}",
    )
    assert response.status_code == 413


def test_chunked_oversized_body_is_rejected_without_content_length():
    async def consume_body(scope, receive, send):
        while True:
            message = await receive()
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    messages = iter(
        [
            {"type": "http.request", "body": b"123", "more_body": True},
            {"type": "http.request", "body": b"45", "more_body": False},
        ]
    )
    sent = []

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message)

    middleware = api._RequestSizeLimit(consume_body, max_bytes=4)
    asyncio.run(
        middleware(
            {"type": "http", "headers": []},
            receive,
            send,
        )
    )
    assert sent[0]["status"] == 413
