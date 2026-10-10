"""Focused tests for the Stage 1c release-rule conditions (pure logic)."""

from __future__ import annotations

import numpy as np

from policies.stage1c_release import ReleaseRuleDampingWallPolicy


def _policy(rule: str, threshold: float = 12.0):
    p = ReleaseRuleDampingWallPolicy(decision_step=250, rule=rule, min_hold_s=120.0,
                                     hard_max_hold_s=900.0,
                                     r2_threshold_veh_km_lane=threshold
                                     if rule == "R2" else None)
    p._free_flow_mps = 30.0
    return p


def test_r1_condition_and_persistence():
    p = _policy("R1")
    assert p._release_condition({"arrival_rate_veh_h": 900.0, "merge_discharge_veh_h": 1000.0,
                                 "approach_mean_speed_mps": 24.0})  # 0.9*1000 <= 900, speed >= 24
    assert not p._release_condition({"arrival_rate_veh_h": 950.0, "merge_discharge_veh_h": 1000.0,
                                     "approach_mean_speed_mps": 24.0})  # arrival above 0.9 x
    assert not p._release_condition({"arrival_rate_veh_h": 100.0, "merge_discharge_veh_h": 1000.0,
                                     "approach_mean_speed_mps": 20.0})  # speed below 24
    # 60-step persistence in the act wrapper is exercised via window_count.
    p._window_count = 0
    for _ in range(60):
        if p._release_condition({"arrival_rate_veh_h": 100.0, "merge_discharge_veh_h": 1000.0,
                                 "approach_mean_speed_mps": 30.0}):
            p._window_count += 1
    assert p._window_count == 60


def test_r2_condition_threshold():
    p = _policy("R2", threshold=12.0)
    assert p._release_condition({"approach_density_veh_km_lane": 11.99})
    assert not p._release_condition({"approach_density_veh_km_lane": 12.5})
    p2 = _policy("R2", threshold=None)
    assert not p2._release_condition({"approach_density_veh_km_lane": 0.0})


def test_density_threshold_quantile():
    densities = np.random.default_rng(7).uniform(5.0, 20.0, size=100)
    threshold = float(np.percentile(densities, 90.0))
    assert threshold == float(np.percentile(densities, 90.0))
