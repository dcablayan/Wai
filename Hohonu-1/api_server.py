"""HTTP service wrapper for the Hohonu tide prediction pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from hohonu_driver_script import run_pipeline

try:
    from fastapi import FastAPI
    from pydantic import BaseModel, Field

    _FASTAPI_AVAILABLE = True
except Exception:
    FastAPI = None  # type: ignore[assignment]
    BaseModel = None  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]
    _FASTAPI_AVAILABLE = False


def _coerce_model_families(values: Optional[List[str]]) -> Optional[List[str]]:
    if values is None:
        return None

    tokens = []
    for item in values:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        for token in text.replace(" ", ",").split(","):
            token = token.strip().lower()
            if token:
                tokens.append(token)
    return tokens if tokens else None


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


if not _FASTAPI_AVAILABLE:
    raise RuntimeError(
        "FastAPI is not installed in this environment. "
        "Install it with `pip install fastapi uvicorn`."
    )


class PredictRequest(BaseModel):  # type: ignore[misc]
    node_name: str
    strategy: str = "auto"
    steps: int = 960
    use_digital_twin: bool = True
    ensemble_size: int = 3
    include_lstm: bool = False
    include_pinn: bool = False
    candidate_profile: str = "compact"
    candidate_model_families: Optional[List[str]] = Field(default=None)
    meta_top_k: int = 4
    meta_holdout_ratio: float = 0.2
    candidate_mix_max_size: int = 4


class BatchPredictRequest(PredictRequest):  # type: ignore[misc, valid-type]
    node_names: List[str]


app = FastAPI(title="Hohonu Tide Prediction API", version="0.1.0")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "utc": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    payload = run_pipeline(
        node_name=req.node_name,
        model_strategy=str(req.strategy).lower(),
        steps=req.steps,
        use_digital_twin=req.use_digital_twin,
        ensemble_size=req.ensemble_size,
        include_lstm=req.include_lstm,
        include_pinn=req.include_pinn,
        candidate_profile=req.candidate_profile,
        candidate_model_families=_coerce_model_families(req.candidate_model_families),
        meta_top_k=req.meta_top_k,
        meta_holdout_ratio=req.meta_holdout_ratio,
        candidate_mix_max_size=req.candidate_mix_max_size,
        return_metadata=True,
    )

    payload["candidate_profile"] = req.candidate_profile
    payload["candidate_model_families"] = _coerce_model_families(
        req.candidate_model_families
    )
    payload["candidate_mix_max_size"] = req.candidate_mix_max_size
    payload = _to_serializable(payload)
    payload["generated_utc"] = datetime.now(timezone.utc).isoformat()
    return payload


@app.post("/batch-predict")
def batch_predict(req: BatchPredictRequest):
    response = {"generated_utc": datetime.now(timezone.utc).isoformat(), "predictions": {}}
    for node_name in req.node_names:
        strategy = str(req.strategy).lower()
        response["predictions"][node_name] = _to_serializable(
            run_pipeline(
                node_name=node_name,
                model_strategy=strategy,
                steps=req.steps,
                use_digital_twin=req.use_digital_twin,
                ensemble_size=req.ensemble_size,
                include_lstm=req.include_lstm,
                include_pinn=req.include_pinn,
                candidate_profile=req.candidate_profile,
                candidate_model_families=_coerce_model_families(req.candidate_model_families),
                meta_top_k=req.meta_top_k,
                meta_holdout_ratio=req.meta_holdout_ratio,
                candidate_mix_max_size=req.candidate_mix_max_size,
                return_metadata=True,
            )
        )

    return response


if __name__ == "__main__":
    try:
        import uvicorn
    except Exception as err:
        raise RuntimeError("uvicorn is required to run this API: pip install uvicorn") from err

    uvicorn.run(app, host="0.0.0.0", port=8000)
