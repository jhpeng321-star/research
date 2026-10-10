"""Tests for the part-3b fixes (D1 slots cutoff, D2 full-state phase-1
slots) plus the D5 scorer sanity check."""
import numpy as np
import pytest
import torch

import p3lib as P


def test_slots_presence_cutoff_at_250():
    """D1 fix: a leader whose true gap exceeds 250 m must be FREE
    (present = 0), not present with a capped gap."""
    lane = np.array([1, 1])
    s = np.array([100.0, 400.0])              # leader 300 m ahead
    v = np.array([30.0, 28.0])
    cav = np.zeros(2)
    sl = P.neighbour_slots(lane, s, v, cav)
    assert sl[0, 0, 0] == 0.0                 # beyond 250 -> absent
    s2 = np.array([100.0, 349.0])             # leader 249 m ahead
    sl2 = P.neighbour_slots(lane, s2, v, cav)
    assert sl2[0, 0, 0] == 1.0
    assert abs(sl2[0, 0, 1] - 249.0) < 1e-6


def test_phase1_slots_carry_neighbours():
    """D2 regression: before the fix every phase-1 sample saw an empty
    scene (own_leader present == 0 for ALL samples) because slots were
    built from a single-row state.  After the fix the sampled slot rate
    must match the full-state neighbour rate for the same ticks."""
    try:
        ra = P.root_arrays("cv17")
    except Exception:
        pytest.skip("p2C data not built")
    import engine as E
    d = E.phase1_samples(["cv17"], max_per_root=60)
    n = len(d["y_v"])
    assert n >= 30
    own_present = d["slots"][:, 0, 0]
    rate = float((own_present > 0).mean())
    # at these densities almost every sampled vehicle has an own-lane
    # leader within 250 m; the defect produced exactly 0.0
    assert rate > 0.5, rate


def test_scoring_synthetic_known_answer():
    """D5: perfect play scores 0; a +1 m/s bias scores RMSE exactly 1."""
    import engine as E
    try:
        ra = P.root_arrays("cv17")
    except Exception:
        pytest.skip("p2C data not built")
    k0 = 600
    snaps = {}
    for j in range(0, 121):
        ii = ra.tick_rows(k0 + j)
        snaps[j] = dict(vid=ra.vid[ii].copy(), lane=ra.lane_m[ii].copy(),
                        s=ra.s[ii].copy(), v=ra.v[ii].copy(),
                        a=ra.a[ii].copy(), collisions=0, capviol=0, lc=0,
                        n_birthed=0)
    m = E.window_metrics(snaps, ra, k0, 60.0)
    assert m["speed_rmse"] == 0.0 and m["pos_rmse"] == 0.0
    assert m["lane_frac"] == 1.0
    for j in range(0, 121):
        snaps[j]["v"] = snaps[j]["v"] + 1.0
    m2 = E.window_metrics(snaps, ra, k0, 60.0)
    assert abs(m2["speed_rmse"] - 1.0) < 1e-9
