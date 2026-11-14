"""Planner-facing TRAIN-only decision scoring adapter (WP-E)."""
from __future__ import annotations

from typing import Any, Mapping

from ..core.errors import not_yet


def predict_states(state: Mapping[str, Any], plan: Mapping[str, Any], horizon_s: float, seed: int) -> Any:
    """Predict frames for a candidate plan (WP-E)."""
    raise not_yet("WP-E", "wms.eval.decide.predict_states")
