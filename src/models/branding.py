"""Single source of truth for Wai wave model display names.

Internal Python class names (TinyTidePrototype, HarmonicRidgeModel, etc.)
remain unchanged for import stability.  Public-facing labels use Wai's
Hawaiian wave branding formatted as: Hawaiian Name (English Label).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ModelBrand:
    internal_name: str   # Python class name
    internal_key: str    # short ID used in dicts / JSON keys
    hawaiian_name: str
    english_name: str
    display_name: str    # always "Hawaiian Name (English Label)"
    category: str        # "prototype" or "pipeline"
    role: str
    description: str
    order: int


WAVE_MODELS: List[ModelBrand] = [
    ModelBrand(
        internal_name="TinyTidePrototype",
        internal_key="TinyTide",
        hawaiian_name="ʻAle Iki",
        english_name="Ripple",
        display_name="ʻAle Iki (Ripple)",
        category="prototype",
        role="Smallest/simple benchmark or sanity-check model",
        description="Two-layer MLP-style forecaster with tidal covariates and skip connection.",
        order=1,
    ),
    ModelBrand(
        internal_name="HarmonicNetPrototype",
        internal_key="HarmonicNet",
        hawaiian_name="Nalu Hoʻokani",
        english_name="Harmonic Wave",
        display_name="Nalu Hoʻokani (Harmonic Wave)",
        category="prototype",
        role="Harmonic/periodic tide model using tidal rhythm",
        description="Physics-informed projection over 8 tidal constituents.",
        order=2,
    ),
    ModelBrand(
        internal_name="WaveGRUPrototype",
        internal_key="WaveGRU",
        hawaiian_name="Nalu Holo",
        english_name="Fast Wave",
        display_name="Nalu Holo (Fast Wave)",
        category="prototype",
        role="Sequential smoothing/time-series movement model",
        description="Bidirectional double-exponential smoothing with attention-like weighting.",
        order=3,
    ),
    ModelBrand(
        internal_name="SurgeNetPrototype",
        internal_key="SurgeNet",
        hawaiian_name="ʻAle Piʻi",
        english_name="Rising Wave",
        display_name="ʻAle Piʻi (Rising Wave)",
        category="prototype",
        role="Surge/high-water residual model",
        description="Residual high-water heuristic; not meteorological surge modeling.",
        order=4,
    ),
    ModelBrand(
        internal_name="TsunamiSentinelPrototype",
        internal_key="TsunamiSentinel",
        hawaiian_name="Kai Eʻe",
        english_name="Tsunami",
        display_name="Kai Eʻe (Tsunami)",
        category="prototype",
        role="Strongest anomaly/sentinel model — tsunami and rapid-spike detection",
        description="High-pass multi-scale anomaly detector for rapid water-level spikes.",
        order=5,
    ),
    ModelBrand(
        internal_name="PersistenceModel",
        internal_key="persistence",
        hawaiian_name="ʻAle Kūpaʻa",
        english_name="Steady Wave (rolling 1-step)",
        display_name="ʻAle Kūpaʻa (Steady Wave — rolling 1-step)",
        category="pipeline",
        role="Rolling 1-step naive baseline: pred[t] = obs[t-1]",
        description="Rolling 1-step persistence: the correct 6-min naive baseline. pred[t]=obs[t-1].",
        order=6,
    ),
    ModelBrand(
        internal_name="PersistenceModel",
        internal_key="persistence_constant",
        hawaiian_name="ʻAle Kūpaʻa Paʻa",
        english_name="Steady Wave (constant holdout)",
        display_name="ʻAle Kūpaʻa Paʻa (Steady Wave — constant holdout)",
        category="pipeline",
        role="Constant persistence: every test step predicted as last training value",
        description="Constant persistence reference: pred[t] = train[-1] for all t. Harder baseline.",
        order=7,
    ),
    ModelBrand(
        internal_name="HarmonicRidgeModel",
        internal_key="harmonic_ridge",
        hawaiian_name="Nalu Hoʻokani Ridge",
        english_name="Harmonic Wave Ridge",
        display_name="Nalu Hoʻokani Ridge (Harmonic Wave Ridge)",
        category="pipeline",
        role="Ridge regression over harmonic tide features",
        description="Harmonic regression over eight tidal constituents with Ridge regularisation.",
        order=8,
    ),
    ModelBrand(
        internal_name="WaveGRUModel",
        internal_key="wave_gru",
        hawaiian_name="Nalu Holo Adapter",
        english_name="Fast Wave Adapter",
        display_name="Nalu Holo Adapter (Fast Wave Adapter)",
        category="pipeline",
        role="DataFrame adapter around WaveGRUPrototype",
        description="DataFrame adapter wrapping WaveGRUPrototype with bidirectional smoothing.",
        order=9,
    ),
]

PROTOTYPE_MODELS: List[ModelBrand] = [m for m in WAVE_MODELS if m.category == "prototype"]
PIPELINE_MODELS: List[ModelBrand] = [m for m in WAVE_MODELS if m.category == "pipeline"]

# Fast lookups: internal_key / internal_name → display_name
DISPLAY_BY_KEY: Dict[str, str] = {m.internal_key: m.display_name for m in WAVE_MODELS}
DISPLAY_BY_NAME: Dict[str, str] = {m.internal_name: m.display_name for m in WAVE_MODELS}
