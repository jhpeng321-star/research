"""WD-01 part 3d: STRUCTURED SUMO-emulating native Part A (B1-B4).

B1  exact SUMO IDM (delta=4, single-step default update, position by the
    measured next-speed form) + measured per-vType residual noise
B2  vType posterior from <=10 s observed (s,v,a,leader) history
    (supervision: type_label; never a rollout input)
B3  per-vehicle lane-change hazard (logistic), class-balanced, validated
    reliability, stochastic per member with CRN seeds
B4  ensemble M>=5 members sampling the posterior + hazard/noise streams

Zero simulation.  Only the 17 train + 4 validation roots are used.
"""
from __future__ import annotations

import numpy as np

# vTypes from the regenerated requests.rou.xml (SUMO IDM-relevant params)
VTYPES = {
    0: dict(name="aggressive", prior=0.10, tau=0.95, a_max=3.0, b=4.5,
            s0=1.5, sf=1.08),
    1: dict(name="normal", prior=0.70, tau=1.1, a_max=2.6, b=4.5,
            s0=2.5, sf=1.00),
    2: dict(name="cautious", prior=0.20, tau=1.5, a_max=2.0, b=3.0,
            s0=3.5, sf=0.92),
}
PRIOR = np.array([0.10, 0.70, 0.20])
LANE_LIMIT = (31.29, 31.29, 31.29)     # network limits by lane
ACCEL_MERGE_S0 = 378.66                # accel merge segment: 31.29 m/s
ACCEL_RAMP_LIMIT = 22.22
OFFSETS = (2645.67, 2644.22, 378.66)
GAP_MAX = 250.0
MERGE_ZONE = (378.66, 454.10)


def lane_limit(lane, s):
    """Network speed limit at (lane, s): main 31.29; accel ramp 22.22,
    merge segment 31.29."""
    if lane == 2 and s < ACCEL_MERGE_S0:
        return ACCEL_RAMP_LIMIT
    return 31.29


def lane_limits_vec(lane, s):
    out = np.empty(len(lane), dtype=np.float64)
    for i in range(len(lane)):
        out[i] = lane_limit(lane[i], s[i])
    return out


def sumo_idm_acc(v, gap, dv, tau, a_max, b, s0, v0):
    """SUMO MSCFModel_IDM, delta=4, single step (idmStepping off):
    a = a_max * (1 - (v/v0)^4 - (s*/gap)^2),
    s* = s0 + max(0, v*tau + v*dv / (2*sqrt(a_max*b))),
    free driving (gap > 250): interaction dropped."""
    v0 = np.maximum(v0, 0.1)
    ratio = np.minimum(v / v0, 4.0)
    free = gap > GAP_MAX
    g = np.where(free, np.inf, np.maximum(gap, 0.1))
    s_star = s0 + np.maximum(0.0, v * tau + v * dv /
                             (2.0 * np.sqrt(a_max * b)))
    interaction = np.where(free, 0.0, (s_star / g) ** 2)
    return a_max * (1.0 - ratio ** 4 - interaction)


def leaders_within_lane(lane, s):
    """Per vehicle: index of the same-lane leader (next larger s) or -1,
    plus the gap (s_leader - s_follower)."""
    n = len(lane)
    leader = np.full(n, -1, dtype=np.int64)
    gap = np.full(n, np.inf)
    for li in (0, 1, 2):
        ids = np.where(lane == li)[0]
        if len(ids) < 2:
            continue
        o = ids[np.argsort(s[ids], kind="stable")]   # front (large s) first
        for j in range(len(o) - 1):
            leader[o[j]] = o[j + 1]
            gap[o[j]] = s[o[j + 1]] - s[o[j]]
    return leader, gap


def sumo_step(s, v, lane, acc, dt=0.5):
    """One deterministic step: v' = clip(v + acc*dt, 0, 1.2*limit);
    s' = s + v'*dt (the MEASURED SUMO form)."""
    lim = lane_limits_vec(lane, s)
    v_new = np.clip(v + acc * dt, 0.0, 1.2 * lim)
    return s + v_new * dt, v_new
