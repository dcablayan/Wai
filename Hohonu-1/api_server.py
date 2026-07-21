"""Bounded HTTP inference wrapper for the Hohonu tide prediction pipeline.

The service is intentionally conservative: health is public, prediction
routes require ``X-API-Key``, request dimensions are capped, neural candidates
are unavailable through HTTP, and only a small number of expensive forecasts
may execute concurrently.  Broader training and benchmarking remain offline
CLI workflows.
"""

from __future__ import annotations

from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import logging
import os
import secrets
import threading
import time
from typing import Annotated, Dict, List, Literal, Optional

from hohonu_driver_script import run_pipeline

try:
    from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
    from pydantic import BaseModel, ConfigDict, Field
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False


if not _FASTAPI_AVAILABLE:
    raise RuntimeError(
        "The API extra is not installed. Install the project with "
        "`pip install -e '.[api]'`."
    )


LOGGER = logging.getLogger(__name__)
API_KEY_ENV = "WAI_API_KEY"
MAX_REQUEST_BYTES = 64 * 1024
MAX_CONCURRENT_FORECASTS = max(
    1, min(4, int(os.getenv("WAI_API_MAX_CONCURRENCY", "2")))
)
REQUESTS_PER_MINUTE = max(
    1, min(120, int(os.getenv("WAI_API_REQUESTS_PER_MINUTE", "20")))
)
NODE_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"
MODEL_FAMILY = Literal[
    "ridge", "lasso", "elastic", "knn", "svr", "hgb", "rf", "gbr", "extra"
]


class ForecastOptions(BaseModel):
    """Strict, bounded options shared by single and batch requests."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    strategy: Literal["var", "auto", "ensemble"] = "auto"
    steps: int = Field(default=240, ge=1, le=1440)
    use_digital_twin: bool = True
    ensemble_size: int = Field(default=3, ge=1, le=6)
    include_lstm: Literal[False] = False
    include_pinn: Literal[False] = False
    candidate_profile: Literal["compact"] = "compact"
    candidate_model_families: Optional[
        Annotated[List[MODEL_FAMILY], Field(min_length=1, max_length=8)]
    ] = None
    meta_top_k: int = Field(default=4, ge=1, le=8)
    meta_holdout_ratio: float = Field(default=0.2, gt=0.0, le=0.5)
    candidate_mix_max_size: int = Field(default=4, ge=2, le=6)


class PredictRequest(ForecastOptions):
    node_name: str = Field(min_length=1, max_length=64, pattern=NODE_PATTERN)


class BatchPredictRequest(ForecastOptions):
    node_names: Annotated[
        List[Annotated[str, Field(min_length=1, max_length=64, pattern=NODE_PATTERN)]],
        Field(min_length=1, max_length=4),
    ]


class _FixedWindowRateLimiter:
    """Small in-process limiter; production deployments should enforce an edge limit too."""

    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        self.limit = int(limit)
        self.window_seconds = float(window_seconds)
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, identity: str) -> bool:
        now = time.monotonic()
        with self._lock:
            events = self._events[identity]
            cutoff = now - self.window_seconds
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                return False
            events.append(now)
            return True

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


_RATE_LIMITER = _FixedWindowRateLimiter(REQUESTS_PER_MINUTE)
_FORECAST_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_FORECASTS)


class _RequestBodyTooLarge(Exception):
    """Internal control flow used before request parsing reaches FastAPI."""


class _RequestSizeLimit:
    """Enforce a body limit even when a client uses chunked transfer encoding."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = int(max_bytes)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_bytes:
                    await Response(status_code=status.HTTP_413_CONTENT_TOO_LARGE)(
                        scope, receive, send
                    )
                    return
            except ValueError:
                await Response(status_code=status.HTTP_400_BAD_REQUEST)(scope, receive, send)
                return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _RequestBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _RequestBodyTooLarge:
            await Response(status_code=status.HTTP_413_CONTENT_TOO_LARGE)(scope, receive, send)


def _coerce_model_families(values: Optional[List[str]]) -> Optional[List[str]]:
    if values is None:
        return None
    return [str(value).strip().lower() for value in values]


def _to_python_scalar(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def _to_serializable(payload: Dict) -> Dict:
    serialized = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            serialized[key] = _to_serializable(value)
        elif isinstance(value, (list, tuple)):
            serialized[key] = [
                _to_serializable(item) if isinstance(item, dict) else _to_python_scalar(item)
                for item in value
            ]
        else:
            serialized[key] = _to_python_scalar(value)
    return serialized


def _require_api_key(
    supplied: Annotated[Optional[str], Header(alias="X-API-Key")] = None,
) -> str:
    expected = os.getenv(API_KEY_ENV)
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Prediction API is disabled until {API_KEY_ENV} is configured",
        )
    if supplied is None or not secrets.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return supplied


def _authorize_and_limit(api_key: str = Depends(_require_api_key)) -> None:
    identity = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    if not _RATE_LIMITER.check(identity):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Prediction request rate exceeded",
            headers={"Retry-After": "60"},
        )


@contextmanager
def _forecast_capacity():
    if not _FORECAST_SLOTS.acquire(blocking=False):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="All forecast workers are busy; retry later",
            headers={"Retry-After": "5"},
        )
    try:
        yield
    finally:
        _FORECAST_SLOTS.release()


def _run_forecast(node_name: str, req: ForecastOptions) -> dict:
    try:
        return run_pipeline(
            node_name=node_name,
            model_strategy=req.strategy,
            steps=req.steps,
            use_digital_twin=req.use_digital_twin,
            ensemble_size=req.ensemble_size,
            include_lstm=False,
            include_pinn=False,
            candidate_profile="compact",
            candidate_model_families=_coerce_model_families(
                req.candidate_model_families
            ),
            meta_top_k=req.meta_top_k,
            meta_holdout_ratio=req.meta_holdout_ratio,
            candidate_mix_max_size=req.candidate_mix_max_size,
            return_metadata=True,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Requested node data was not found") from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except HTTPException:
        raise
    except Exception as error:
        LOGGER.exception("Forecast pipeline failed for node %s", node_name)
        raise HTTPException(status_code=500, detail="Forecast pipeline failed") from error


app = FastAPI(title="Hohonu Tide Prediction API", version="0.2.0")
app.add_middleware(_RequestSizeLimit, max_bytes=MAX_REQUEST_BYTES)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "prediction_api_configured": bool(os.getenv(API_KEY_ENV)),
        "utc": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict", dependencies=[Depends(_authorize_and_limit)])
def predict(req: PredictRequest):
    with _forecast_capacity():
        payload = _run_forecast(req.node_name, req)
    payload["candidate_profile"] = req.candidate_profile
    payload["candidate_model_families"] = _coerce_model_families(
        req.candidate_model_families
    )
    payload["candidate_mix_max_size"] = req.candidate_mix_max_size
    serialized = _to_serializable(payload)
    serialized["generated_utc"] = datetime.now(timezone.utc).isoformat()
    return serialized


@app.post("/batch-predict", dependencies=[Depends(_authorize_and_limit)])
def batch_predict(req: BatchPredictRequest):
    response = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "predictions": {},
    }
    with _forecast_capacity():
        for node_name in req.node_names:
            response["predictions"][node_name] = _to_serializable(
                _run_forecast(node_name, req)
            )
    return response


if __name__ == "__main__":
    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError("Install the project with `pip install -e '.[api]'`") from error

    uvicorn.run(
        app,
        host=os.getenv("WAI_API_HOST", "127.0.0.1"),
        port=int(os.getenv("WAI_API_PORT", "8000")),
    )
