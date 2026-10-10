"""Focused tests for the Sustained-wall Stage 1a pilot (pure logic only)."""

from __future__ import annotations

import math

import numpy as np

from experiments.idm import sustained_wall_v1_contract as C
from experiments.idm.analyze_sustained_wall_v1_stage1a import _bootstrap_intervals, _quantiles
from policies.noop_base import NoOpNativeIDMPolicy
from policies.sustained_wall import (
    PerturbationPolicy,
    SustainedWallPolicy,
    band_distance_in_m,
    select_nudge_candidate,
)


class _FakeEnv:
    """Minimal env stub exposing the health accessor the policies use."""

    def __init__(self, health_sequence):
        self._health_sequence = list(health_sequence)
        self.calls = 0

    def metastable_health_detail(self):
        self.calls += 1
        if not self._health_sequence:
            return (1.0, 0.0, 25.0)
        value = self._health_sequence.pop(0) if len(self._health_sequence) > 1 else self._health_sequence[0]
        return (float(value), 0.0, 25.0)


# -- landmark detection ----------------------------------------------------
def test_find_landmark_detects_first_valid_step():
    h = [1.0] * 250
    h[200] = 0.6
    h[201] = 0.55
    h[202] = 0.4  # <= 0.45 and h[202] - h[172] <= 0
    h[203] = 0.42  # decay relative to 173 is positive-ish? h[203] > h[202] -> -0.02
    assert C.find_landmark(h) == 202


def test_find_landmark_requires_decay_vs_30s_ago():
    h = [1.0] * 250
    h[200] = 0.3
    # h[200] - h[170] = 0.3 - 1.0 <= 0 -> should fire at 200
    assert C.find_landmark(h) == 200


def test_find_landmark_ignores_rising_vs_30s_ago():
    h = [1.0] * 250
    h[130:200] = [0.2] * 70  # deeply unhealthy 30 s before -> decay condition positive
    h[200] = 0.3
    # h[200] - h[170] = 0.3 - 0.2 = +0.1 > 0 -> step 200 not a landmark
    h[201] = 0.28
    # h[201] - h[171] = 0.28 - 0.2 = +0.08 > 0 -> still no
    assert C.find_landmark(h) is None


def test_find_landmark_no_hit_before_last_bound():
    h = [1.0] * 720
    h[720:] = [0.4] * 20
    assert C.find_landmark(h) is None  # landmark at 721 > 700 -> no landmark
    h[700] = 0.4  # now 700 itself qualifies
    assert C.find_landmark(h) == 700


# -- collapse labels -------------------------------------------------------
def test_health_collapse_statistics_incidence_and_runs():
    h = [1.0] * 300
    # 60 consecutive unhealthy seconds starting at step 150, preceded by
    # 10 unhealthy seconds at step 60 (before t=100, ignored and too short).
    h[60:70] = [0.1] * 10
    h[150:210] = [0.1] * 60
    stats = C.health_collapse_statistics(h)
    assert stats.incidence
    assert stats.first_onset_step == 150
    assert stats.longest_run_s == 60.0
    assert stats.total_duration_s == 60.0


def test_health_collapse_statistics_ignores_pre_100_and_short_runs():
    h = [1.0] * 250
    h[40:100] = [0.1] * 60  # before t=100: not counted
    h[110:169] = [0.1] * 59  # one step short of 60
    stats = C.health_collapse_statistics(h)
    assert not stats.incidence
    assert stats.total_duration_s == 59.0  # the qualifying run is absent


def test_physical_collapse_statistics():
    metrics = []
    # 60 steps with >= 5 slow vehicles and mean <= 15 m/s
    for _ in range(60):
        metrics.append({"queue_count": 6, "pre_merge_mean_speed_mps": 12.0})
    for _ in range(30):
        metrics.append({"queue_count": 2, "pre_merge_mean_speed_mps": 12.0})
    stats = C.physical_collapse_statistics(metrics)
    assert stats.incidence
    assert stats.first_onset_step == 0
    assert stats.longest_run_s == 60.0
    assert stats.total_duration_s == 60.0


def test_physical_collapse_statistics_short_run():
    metrics = [
        {"queue_count": 5, "pre_merge_mean_speed_mps": 15.0}
        for _ in range(59)
    ]
    stats = C.physical_collapse_statistics(metrics)
    assert not stats.incidence
    assert stats.longest_run_s == 59.0


# -- band passage and nudge selection --------------------------------------
def test_band_distance_in_m_restricts_to_mainline():
    assert band_distance_in_m(("1191554636", "1191554636_0", 300.0))
    assert not band_distance_in_m(("1191554636", "1191554636_0", 199.9))
    assert not band_distance_in_m(("1191554636", "1191554636_0", 400.1))
    assert not band_distance_in_m(("11553048#1", "11553048#1_0", 300.0))  # ramp


def test_select_nudge_candidate_nearest_and_tie_break():
    candidates = [
        ("veh_a", 250.0),
        ("veh_b", 310.0),
        ("veh_c", 300.0),
        ("veh_d", 295.0),
    ]
    assert select_nudge_candidate(candidates, 300.0) == "veh_c"
    assert select_nudge_candidate([("veh_a", 300.0), ("veh_b", 300.0)], 300.0) == "veh_a"
    assert select_nudge_candidate([], 300.0) is None


# -- no-op base policy -----------------------------------------------------
def test_noop_policy_issues_no_actions_and_keeps_health_discipline():
    env = _FakeEnv([0.5] * 10)
    policy = NoOpNativeIDMPolicy()
    policy.reset(env, seed=7)
    for _ in range(3):
        assert policy.act({}) == {}
    assert env.calls == 3


# -- sustained wall state machine (no live CAVs) ---------------------------
def test_sustained_wall_engagement_and_health_release_timeline():
    env = _FakeEnv([0.3] * 12 + [0.6] * 60)
    policy = SustainedWallPolicy(
        decision_step=10,
        wall_params=dict(C.WALL_PARAMS),
        recovery_h=C.RECOVERY_H,
        recovery_persist_s=C.RECOVERY_PERSIST_S,
        max_wall_duration_s=C.MAX_WALL_DURATION_S,
        wall_cap_s=C.WALL_CAP_S,
        taper_duration_s=C.TAPER_DURATION_S,
    )
    policy.reset(env, seed=7)
    states = []
    wall_steps = []
    for step in range(1, 70):
        policy.act({})
        states.append((step, policy.last_audit["treatment_state"]))
        if policy.last_audit["wall_active"]:
            wall_steps.append(step)
    assert states[10][1] == "HOLDING"  # step 11 = t_d + 1
    assert all(state == "DAMPING" for step, state in states[:10])
    # Health 0.3 for the 12 pre-step readings of steps 1-12, then 0.6 from
    # step 13: the 20th consecutive healthy pre-step reading lands at step 32
    # -> release there (no members -> taper collapses instantly).
    release_step = None
    for step, state in states:
        if release_step is None and step > 11 and state != "HOLDING":
            release_step = step
    assert release_step == 32
    assert policy._release_reason == "health_recovery"
    assert policy._release_step == 32
    assert policy._state == "RELEASED"
    # Wall applied on steps 11..31.
    assert wall_steps == list(range(11, 32))


def test_sustained_wall_max_duration_cap():
    env = _FakeEnv([0.1] * 400)
    policy = SustainedWallPolicy(
        decision_step=50,
        wall_params=dict(C.WALL_PARAMS),
        recovery_h=C.RECOVERY_H,
        recovery_persist_s=C.RECOVERY_PERSIST_S,
        max_wall_duration_s=180.0,
        wall_cap_s=900.0,
        taper_duration_s=5.0,
    )
    policy.reset(env, seed=7)
    wall_steps = []
    for step in range(1, 400):
        policy.act({})
        if policy.last_audit["wall_active"]:
            wall_steps.append(step)
    assert wall_steps[:2] == [51, 52]
    assert policy._release_reason == "max_duration"
    assert policy._release_step == 231  # 180 s after engagement step 51
    assert policy.last_audit["treatment_state"] == "RELEASED"
    # Exactly 180 wall-application steps (51..230).
    assert wall_steps == list(range(51, 231))


# -- perturbation policy timeline ------------------------------------------
def test_perturbation_policy_nudge_window_and_release():
    env = _FakeEnv([0.5] * 10)
    policy = PerturbationPolicy(decision_step=5, slow_down_mps=0.5, landmark_distance_m=300.0)
    policy.reset(env, seed=7)
    for step in range(1, 3):
        policy.act({})
        assert policy.last_audit["nudge_vehicle_id"] is None  # no CAVs -> no nudge
        assert policy.last_audit["nudge_applied"] is False


# -- analysis helpers ------------------------------------------------------
def test_quantiles_and_bootstrap_intervals_deterministic():
    values = [float(value) for value in range(1, 31)]
    a = _quantiles(values)
    assert a["median"] == 15.5
    b = _bootstrap_intervals(values)
    assert b["mean"]["ci90"]["lo"] <= b["mean"]["ci90"]["hi"]
    assert b["median"]["ci95"]["lo"] <= b["median"]["ci95"]["hi"]
    again = _bootstrap_intervals(values)
    assert b == again  # seeded RNG -> reproducible
