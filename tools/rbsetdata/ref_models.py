"""Reference physics model functions for the setdata likelihood harness.

These are intentionally SMALL, self-contained implementations of SUMO 1.27.1
models so the likelihood script has a default; the authoritative ports for the
rebuild live with the spec helpers (rb-shared/idmspec/ref_idm.py,
rb-shared/lcspec/).  Every constant below cites its SUMO source line (local
copy: rb-shared/idmspec/_sumo_src/, downloaded 2026-10-31, not executed).

IDM (MSCFModel_IDM.cpp, `_v()` lines 225-269):
    acc = a * (1 - (v/desSpeed)^delta - (s*/gap)^2)
    s*  = max(0, v*T + v*(v - v_lead)/(2*sqrt(a*b))) + minGap   (respectMinGap)
    stepping: `myIterations = max(1, round(TS/stepping))` with default
    stepping 0.25 s (line 44), TS = 0.5 s here (recorded launch command
    `--step-length 0.5`), so 2 sub-iterations per tick; per sub-iteration the
    gap is shrunk by the travelled relative distance (line 266).
    delta = 4 (line 41).  gap is floored at NUMERICAL_EPS = 0.001
    (src/config.h.cmake line 236; floored once before the loop, line 252).
    The caller passes the NET bumper gap (the C++ `gap2pred` "comes with
    minGap already subtracted", re-added inside `_v`, lines 237-251), i.e.
    gap = s_leader - s_ego - length_leader, the same quantity the extractor
    records as lead_gap_m.

desSpeed: `veh->getLane()->getVehicleMaxSpeed(veh)` = min(vType maxSpeed
    x speedFactor, lane speed limit).  We expose it as the free parameter v0
    capped at the road limit 31.29 m/s (tools/wmfixmerge/data.py-cited
    constant SPEED_LIMIT_MPS = 31.29 in wms/data/c4_episodes.py:69).

LC2013 scaffold (MSLCM2013): the docs describe the change decision as a
weighted sum of strategic, keep-right and speed-gain "bias" terms compared
against lcSigma-style logistic noise
(https://eclipse.dev/sumo/docs/docsum/LaneChangingModel.html).  Our scaffold
scores a target lane with those three documented terms; the weights ARE the
fitted parameters (no hand-set behavioural thresholds).  Replace with the
authoritative MSLCM2013 port from rb-shared/lcspec when it lands.
"""
from __future__ import annotations

import math

import numpy as np

NUMERICAL_EPS = 0.001      # src/config.h.cmake:236
SPEED_LIMIT_MPS = 31.29    # wms/data/c4_episodes.py:69 (documented road limit)
IDM_DELTA = 4.0            # MSCFModel_IDM.cpp:41 (default delta)
IDM_STEPPING = 0.25        # MSCFModel_IDM.cpp:44 (default stepping)

IDM_BOUNDS = {             # MLE search bounds (SUMO-documented parameter ranges)
    "v0": (1.0, 36.0),     # desired speed m/s (docs: maxSpeed x speedFactor)
    "T": (0.3, 3.0),       # headway time s (docs: tau, ~0.3..3)
    "a": (0.1, 3.0),       # max accel m/s^2
    "b": (0.1, 5.0),       # comfortable decel m/s^2
    "s0": (0.5, 6.0),      # minGap m (docs: 0..)
}
IDM_DEFAULTS = {"v0": 30.0, "T": 1.2, "a": 1.0, "b": 1.5, "s0": 2.5}


def idm_sumo_step(v, gap, lead_v, v0: float, T: float, a: float, b: float,
                  s0: float, dt: float = 0.5,
                  v_des_cap: float = SPEED_LIMIT_MPS):
    """One SUMO IDM tick: predicted v_next (m/s); numpy elementwise or scalar.

    Matches MSCFModel_IDM::_v with respectMinGap=True, default delta=4 and
    stepping 0.25 s.  ``gap`` is the net bumper gap to the leader.
    """
    des_speed = max(NUMERICAL_EPS, min(v0, v_des_cap))
    two_sqrt_ad = 2.0 * math.sqrt(a * b)
    g = np.maximum(NUMERICAL_EPS, np.asarray(gap, np.float64))
    new_speed = np.asarray(v, np.float64).copy()
    iterations = max(1, int(dt / IDM_STEPPING + 0.5))
    for _ in range(iterations):
        delta_v = new_speed - lead_v
        s_star = np.maximum(0.0, new_speed * T + new_speed * delta_v
                            / two_sqrt_ad) + s0
        acc = a * (1.0
                   - (new_speed / max(NUMERICAL_EPS, des_speed)) ** IDM_DELTA
                   - (s_star * s_star) / (g * g))
        new_speed = np.maximum(0.0, new_speed + acc * dt / iterations)
        g = g - np.maximum(0.0, (new_speed - lead_v) * dt / iterations)
    return np.maximum(0.0, new_speed)


def idm_default_fn(v, gap, lead_v, dt=0.5, v0: float = 30.0, T: float = 1.2,
                   a: float = 1.0, b: float = 1.5, s0: float = 2.5):
    """Model-fn contract wrapper: ``fn(v, gap, lead_v, dt, **params) -> v_next``."""
    return idm_sumo_step(v, gap, lead_v, v0, T, a, b, s0, dt=dt)


def make_idm_fn(params: dict):
    """`fn(v, gap, lead_v, dt) -> v_next` with fixed params (model-fn contract)."""
    p = {**IDM_DEFAULTS, **params}
    return lambda v, gap, lead_v, dt=0.5: idm_sumo_step(
        v, gap, lead_v, p["v0"], p["T"], p["a"], p["b"], p["s0"], dt=dt)


LC_BOUNDS = {              # scaffold logistic weights (fit, not thresholds)
    "w_keep_right": (-4.0, 4.0),
    "w_speed_gain": (-4.0, 4.0),
    "w_gap": (-4.0, 4.0),
    "bias": (-4.0, 4.0),
}
LC_DEFAULTS = {"w_keep_right": 0.5, "w_speed_gain": 0.5, "w_gap": 0.5, "bias": 0.0}


def lc2013_scaffold_logit(dv_target_lead: float, gap_target_lead: float,
                          dv_target_follow: float, gap_target_follow: float,
                          to_right: int, params: dict) -> float:
    """Logit of changing into the target lane (scaffold; see module docstring).

    Terms follow the documented LC2013 incentives: keep-right (to_right=1 when
    the target lane is the more right-hand/mainline lane), speed gain (the
    speed advantage the new lane offers) and gap safety (free space in the
    target lane).  Weights are the fitted parameters.
    """
    p = {**LC_DEFAULTS, **params}
    speed_gain = (dv_target_lead + dv_target_follow) * 0.5
    gap_term = (min(gap_target_lead, 100.0) + min(gap_target_follow, 100.0)) * 0.05 - 2.5
    return (p["w_keep_right"] * float(to_right)
            + p["w_speed_gain"] * speed_gain
            + p["w_gap"] * gap_term
            + p["bias"])
