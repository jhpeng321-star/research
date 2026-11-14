"""Tests for ``driver.desired_speed`` (shapes, shrinkage, free-flow mask)."""
from __future__ import annotations

import numpy as np

from driver.desired_speed import (DesiredSpeedEstimate, estimate_desired_speed)


def _grid(t, k):
    return (np.zeros((t, k)), np.ones((t, k), bool), np.full((t, k), np.inf))


def test_shrinks_to_prior_with_no_free_flow():
    t, k = 60, 2
    speeds, present, gap = _grid(t, k)
    speeds[:] = 8.0                                  # all constrained (small gap)
    gap[:] = 6.0
    est = estimate_desired_speed(speeds, present, gap, prior_v0_mps=28.0)
    assert isinstance(est, DesiredSpeedEstimate)
    assert est.v0_mps.shape == (k,)
    assert np.allclose(est.v0_mps, 28.0)             # fully shrunk to the prior
    assert np.all(est.n_free == 0)
    assert np.all(est.shrunk)


def test_free_flow_quantile_beats_prior_with_data():
    t, k = 60, 1
    speeds, present, gap = _grid(t, k)
    speeds[:] = 30.0                                 # unconstrained, fast driver
    est = estimate_desired_speed(speeds, present, gap, prior_v0_mps=28.0)
    assert est.n_free[0] == t
    assert not est.shrunk[0]
    assert abs(est.v0_mps[0] - 30.0) < 0.5           # raw dominates with 60 obs


def test_partial_data_shrinks_between_raw_and_prior():
    t, k = 60, 1
    speeds, present, gap = _grid(t, k)
    speeds[:] = 8.0
    speeds[:5] = 20.0                                # 5 free-flow ticks only
    gap[5:] = 6.0
    est = estimate_desired_speed(speeds, present, gap, prior_v0_mps=10.0,
                                 shrink_obs=10.0)
    assert est.n_free[0] == 5
    assert 10.0 < est.v0_mps[0] < 20.0               # between prior and raw
    assert est.shrunk[0]


def test_headway_from_car_following_episode():
    t, k = 60, 1
    speeds, present, gap = _grid(t, k)
    speeds[:] = 10.0
    gap[:] = 13.0                                    # T = gap/v = 1.3 s
    est = estimate_desired_speed(speeds, present, gap, prior_v0_mps=28.0)
    assert abs(est.T_s[0] - 1.3) < 0.05


def test_slot_axis_independent():
    t, k = 40, 3
    speeds, present, gap = _grid(t, k)
    speeds[:, 0] = 30.0                              # free driver
    speeds[:, 1] = 6.0
    gap[:, 0] = np.inf
    gap[:, 1] = 4.0                                  # blocked
    present[:, 2] = False                            # absent slot
    est = estimate_desired_speed(speeds, present, gap, prior_v0_mps=25.0)
    assert est.v0_mps[0] > est.v0_mps[1]
    assert est.n_free[0] == t and est.n_free[1] == 0 and est.n_free[2] == 0
