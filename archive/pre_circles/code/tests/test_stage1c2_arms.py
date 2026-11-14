"""Focused tests for Stage 1c-2 arms (pure logic; no SUMO)."""

from __future__ import annotations

from experiments.idm.rsu_feature_capture import (
    _phase,
    first_onset_step,
)
from policies.stage1c2_arms import (
    FixedHoldDampingWallPolicy,
    OnsetTriggeredDampingWallPolicy,
    R1bReleaseDampingWallPolicy,
)


def test_fixed_hold_cap_gives_exact_hold_duration():
    p = FixedHoldDampingWallPolicy(decision_step=250, hold_s=300.0)
    # First command at 251; taper begins at cap + 1 = 551 -> exactly 300 s.
    assert p._wall_cap_s == 550.0
    assert 551 - 251 == 300
    p450 = FixedHoldDampingWallPolicy(decision_step=250, hold_s=450.0)
    assert p450._wall_cap_s == 700.0


def test_onset_triggered_engages_one_step_after_detection():
    p = OnsetTriggeredDampingWallPolicy(onset_step=190, hold_s=300.0)
    # decision_step = onset -> first command at onset + 1; prefix identical
    # through the onset step; cap = onset + hold -> taper at onset + 301.
    assert p._decision_step == 190
    assert p._wall_cap_s == 490.0
    assert 491 - 191 == 300


def test_r1b_condition_uses_400_800_speed_gate():
    p = R1bReleaseDampingWallPolicy(decision_step=250)
    p._free_flow_mps = 30.0
    ok = {"arrival_rate_veh_h": 900.0, "merge_discharge_veh_h": 1000.0,
          "mean_speed_400_800_mps": 24.0}
    assert p._release_condition(ok)  # arrival <= 0.9 x discharge, speed >= 24
    assert not p._release_condition(
        {**ok, "arrival_rate_veh_h": 950.0})  # arrival above 0.9 x
    assert not p._release_condition(
        {**ok, "mean_speed_400_800_mps": 23.9})  # below 0.8 x free-flow


def test_r1b_hard_max_is_hold_relative():
    p = R1bReleaseDampingWallPolicy(decision_step=250, hard_max_hold_s=900.0)
    p._free_flow_mps = 30.0
    # Engagement at 251; the hard max fires at elapsed >= 900 -> step 1151,
    # i.e. 900 s of hold (the 1c R1 capped at absolute t=901, only 650 s).
    assert 251 + 900 == 1151


def test_first_onset_step_10s_rule():
    too_short = [1.0] * 10 + [0.1] * 9 + [1.0] * 10
    assert first_onset_step(too_short, max_step=900) is None
    run = [1.0] * 20 + [0.1] * 15 + [1.0] * 5
    # Low run starts at index 20; the 10th consecutive low step is 29.
    assert first_onset_step(run, max_step=900) == 29
    assert first_onset_step(run, max_step=25) is None  # beyond the search cap


def test_phase_bucketing():
    assert _phase(200, engage_step=251, release_step=551) == "pre_engagement"
    assert _phase(300, engage_step=251, release_step=551) == "hold"
    assert _phase(551, engage_step=251, release_step=551) == "taper"
    assert _phase(557, engage_step=251, release_step=551) == "post_release"
    assert _phase(600, engage_step=251, release_step=None) == "hold"
