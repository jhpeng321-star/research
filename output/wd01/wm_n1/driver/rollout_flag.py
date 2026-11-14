"""Prepare the optional per-slot desired-speed array for the rollout IDM.

``rollout2.Rollout2Config.desired_speed_mps`` (default ``None`` = OFF) holds a
per-slot desired speed on the rollout's global slot axis.  This helper turns an
estimator output into that array: non-finite / missing slots fall back to the
caller's default (the segment limit), and every value is floored to a positive
speed.  With ``v0_mps=None`` it returns ``None`` so the engine keeps the frozen
byte-identical path.
"""
from __future__ import annotations

import numpy as np

MIN_SPEED_MPS = 1e-6


def apply_desired_speed_v0(v0_mps, fallback_mps) -> np.ndarray | None:
    if v0_mps is None:
        return None
    v0 = np.asarray(v0_mps, dtype=np.float64)
    fb = np.asarray(fallback_mps, dtype=np.float64)
    if fb.ndim == 0:
        fb = np.full(v0.shape, float(fb))
    if fb.shape != v0.shape:
        raise ValueError(f"fallback shape {fb.shape} != v0 shape {v0.shape}")
    out = np.where(np.isfinite(v0), v0, fb)
    return np.maximum(out, MIN_SPEED_MPS)
