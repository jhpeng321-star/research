"""RSU-observable per-driver desired-speed estimation (WM-S T3, P1).

From each vehicle's observed recent motion before ``t0`` -- the last
``window_s`` seconds of speed, together with the same-lane bumper gap that says
whether the vehicle was actually unconstrained -- estimate a desired speed
``v0_i``.  A free-flow vehicle's speed is a noisy lower bound on its desired
speed, so the raw estimate is a high quantile of the free-flow speeds in the
window; when few free-flow observations exist the estimate is shrunk toward a
population/vType prior.  An optional time headway ``T_i`` is estimated from
car-following episodes (small gap, moving) as ``gap / v``.

Inputs are plain NumPy arrays so the same function serves the offline pack
history and the online rolling buffer:

``speeds``/``present``/``gap`` are ``[T, K]`` (oldest first, ``t0`` last);
``gap`` is the same-lane bumper gap to the leader (``inf`` when no leader).
``prior_v0_mps`` is a scalar or a per-slot ``[K]`` array (e.g. the segment speed
limit times the population speed factor); it is a fixed infrastructure/population
constant, never a hidden driver attribute.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DT_S = 0.5
#: A vehicle with this bumper gap or more is treated as unconstrained.
FREE_GAP_M = 100.0
MIN_SPEED_MPS = 1.0
#: Probability-weighted SUMO ``hdv_mix`` speed factor (baseline.sumo_speed_factor_mean).
POP_SPEED_FACTOR = 0.992
#: Observations (ticks) above which the raw estimate is trusted unshrunk.
SHRINK_OBS = 10.0
HEADWAY_PRIOR_S = 1.1
HEADWAY_MIN_S = 0.2
HEADWAY_MAX_S = 5.0
V0_MAX_MPS = 60.0


@dataclass
class DesiredSpeedEstimate:
    """Per-slot desired-speed/headway estimate (``K`` = slot axis)."""

    v0_mps: np.ndarray        # [K] shrunk desired speed
    v0_raw_mps: np.ndarray    # [K] pre-shrink free-flow quantile
    n_free: np.ndarray        # [K] free-flow observations used
    T_s: np.ndarray           # [K] shrunk time headway
    shrunk: np.ndarray        # [K] bool, True iff n_free < shrink_obs


def _prior_vec(prior_v0_mps, k: int) -> np.ndarray:
    arr = np.asarray(prior_v0_mps, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(k, float(arr), dtype=np.float64)
    if arr.shape != (k,):
        raise ValueError(f"prior_v0_mps shape {arr.shape} != ({k},)")
    return arr.astype(np.float64)


def estimate_desired_speed(speeds, present, gap, *, prior_v0_mps,
                           dt_s: float = DT_S, free_gap_m: float = FREE_GAP_M,
                           window_s: float = 30.0,
                           min_speed_mps: float = MIN_SPEED_MPS,
                           shrink_obs: float = SHRINK_OBS,
                           headway_prior_s: float = HEADWAY_PRIOR_S,
                           speed_max_mps: float = V0_MAX_MPS,
                           quantile: float = 90.0) -> DesiredSpeedEstimate:
    """Estimate ``v0_i`` (and ``T_i``) from the trailing ``window_s`` window."""
    speeds = np.asarray(speeds, dtype=np.float64)
    present = np.asarray(present, dtype=bool)
    gap = np.asarray(gap, dtype=np.float64)
    if speeds.ndim != 2:
        raise ValueError("speeds must be [T, K]")
    if present.shape != speeds.shape or gap.shape != speeds.shape:
        raise ValueError("speeds/present/gap must share shape [T, K]")
    t, k = speeds.shape
    prior = _prior_vec(prior_v0_mps, k)

    n_win = max(1, int(round(float(window_s) / float(dt_s))))
    s = speeds[-n_win:]
    p = present[-n_win:]
    g = gap[-n_win:]

    free = p & (g >= float(free_gap_m)) & (s > float(min_speed_mps))
    n_free = free.sum(axis=0).astype(np.float64)

    v0_raw = prior.copy()
    for j in range(k):
        vals = s[free[:, j], j]
        if vals.size:
            v0_raw[j] = float(np.percentile(vals, float(quantile)))

    w = n_free
    v0 = (w * v0_raw + float(shrink_obs) * prior) / (w + float(shrink_obs))
    v0 = np.clip(v0, float(min_speed_mps), float(speed_max_mps))

    follow = p & (g >= 0.0) & (g < float(free_gap_m)) & (s > float(min_speed_mps))
    n_follow = follow.sum(axis=0).astype(np.float64)
    t_s = np.full(k, float(headway_prior_s), dtype=np.float64)
    for j in range(k):
        m = follow[:, j]
        if m.any():
            t_s[j] = float(np.median(g[m, j] / np.maximum(s[m, j], 1e-6)))
    t_s = (n_follow * t_s + float(shrink_obs) * float(headway_prior_s)) \
        / (n_follow + float(shrink_obs))
    t_s = np.clip(t_s, HEADWAY_MIN_S, HEADWAY_MAX_S)

    return DesiredSpeedEstimate(
        v0_mps=v0, v0_raw_mps=v0_raw, n_free=n_free, T_s=t_s,
        shrunk=(n_free < float(shrink_obs)))
