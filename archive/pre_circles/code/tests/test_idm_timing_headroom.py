"""Focused tests for the Path B timing headroom machinery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from idm.timing_headroom import (
    MINIMUM_DOSE_STEPS,
    TimingState,
    classify_timing,
    common_horizon,
    common_state_horizon,
    rate_utility,
    timing_gates,
    timing_masks,
)
from tools.analyze_idm_timing_oracle import analyze


def test_common_horizon_uses_latest_landmark_plus_60():
    assert common_horizon(100, 220, 340) == 400
    assert common_horizon(100, 220, None) == 280
    assert common_horizon(100, None, None) == 160
    assert common_state_horizon(100, 220, 340) == 300
    assert common_state_horizon(100, None, None) == 60


def test_timing_masks_follow_schedule_definitively():
    state = TimingState(
        state_id="s1", seed=4121, penetration=0.1, demand=2000,
        t1=100, t2=220, t3=None, schedule_hash="h",
        observations={"t1": {"admissible": True}, "t2": {"admissible": True}},
        schedule_identity=True,
    )
    masks = state.masks
    assert masks["abstain"] is True
    assert masks["activate_now"] is True
    assert masks["defer_r2"] is True
    assert masks["defer_r3"] is False
    assert timing_masks(100, 220, 340, {"t1": {"admissible": True}, "t2": {"admissible": False}, "t3": {"admissible": True}})[
        "defer_r2"
    ] is False


def test_rate_utility_normalizes_by_horizon_length():
    u = rate_utility(severe_delta=1.0, energy_delta=2.0, horizon_steps=100, severe_ref=0.1, energy_ref=10.0)
    assert u == pytest.approx(0.5 * (1.0 / 10.0) + 0.5 * (2.0 / 1000.0))
    # same delta over twice the horizon halves the rate contribution
    u_short = rate_utility(severe_delta=1.0, energy_delta=2.0, horizon_steps=50, severe_ref=0.1, energy_ref=10.0)
    assert u_short == pytest.approx(2.0 * u)


def test_classification_and_reward_fixed_deadband():
    assert classify_timing(0.05, {"a": True}) == "helpful"
    assert classify_timing(0.01, {"a": True}) == "neutral"
    assert classify_timing(-0.05, {"a": True}) == "harmful"
    assert classify_timing(0.05, {"a": False}) == "harmful"


def test_timing_gates_report_leave_one_seed_out():
    gates = timing_gates(
        oracle_mean=0.05,
        fixed_best_mean=0.01,
        per_seed_gain={4121: 0.03, 4127: 0.031, 4129: -0.05, 4130: 0.03},
        options_used={"defer_r3": 3, "defer_r2": 2},
        outcomes={"helpful": 3, "harmful": 4, "neutral": 5},
        constraint_failures=0,
    )
    assert gates["improves_best_fixed_timing_rule_by_0p02"] is True
    assert gates["meaningful_gain_in_at_least_three_seeds"] is True
    assert gates["more_than_one_activation_timing_selected"] is True
    assert gates["no_single_seed_dominating"] is True
    assert gates["hard_safety_service_compliance"] is True
    assert "effects_exceed_replay_noise" in gates


def _stage_dicts() -> tuple[dict, dict]:
    """Two synthetic stages: one with a dose-failed feasible action, one clean."""
    dose_failed = {
        "seed": 4121,
        "penetration": 0.10,
        "demand": 2000,
        "recurrence": 3,
        "actions": {
            "abstain": {"feasible": True, "reward": 0.0, "classification": "neutral", "constraints_pass": True, "dose_steps": 0},
            "activate_now": {"feasible": True, "reward": -2.0, "classification": "harmful", "constraints_pass": False, "dose_steps": 9},
            "defer_r2": {"feasible": True, "reward": 0.05, "classification": "helpful", "constraints_pass": True, "dose_steps": 20},
            "defer_r3": {"feasible": False, "reward": 0.0, "classification": "neutral", "constraints_pass": True, "dose_steps": 0},
        },
    }
    clean = {
        "seed": 4122,
        "penetration": 0.05,
        "demand": 1900,
        "recurrence": 1,
        "actions": {
            "abstain": {"feasible": True, "reward": 0.0, "classification": "neutral", "constraints_pass": True, "dose_steps": 0},
            "activate_now": {"feasible": True, "reward": -0.05, "classification": "harmful", "constraints_pass": True, "dose_steps": 20},
            "defer_r2": {"feasible": False, "reward": 0.0, "classification": "neutral", "constraints_pass": True, "dose_steps": 0},
            "defer_r3": {"feasible": True, "reward": 0.02, "classification": "helpful", "constraints_pass": True, "dose_steps": 20},
        },
    }
    return dose_failed, clean


def test_fixed_policy_value_retains_realized_penalties_at_every_scope():
    from tools.analyze_idm_timing_oracle import (
        _cohort_value,
        fixed_policy_value,
    )

    dose_failed, clean = _stage_dicts()
    states = [dose_failed, clean]
    # Dose-failed feasible attempt keeps the -2 penalty: mean over 2 states.
    expected_now = (-2.0 + -0.05) / 2.0
    assert fixed_policy_value(states, "activate_now", MINIMUM_DOSE_STEPS) == pytest.approx(expected_now)
    # Global, subgroup (per-seed) and subgroup-by-penetration all agree.
    subset = [dose_failed]
    assert fixed_policy_value(subset, "activate_now", MINIMUM_DOSE_STEPS) == pytest.approx(-2.0)
    assert _cohort_value(subset, "activate_now", MINIMUM_DOSE_STEPS) == pytest.approx(-2.0)
    assert _cohort_value(states, "activate_now", MINIMUM_DOSE_STEPS) == pytest.approx(expected_now)
    # Infeasible timing abstains instead of counting a penalty.
    assert fixed_policy_value(states, "defer_r2", MINIMUM_DOSE_STEPS) == pytest.approx((0.05 + 0.0) / 2.0)
    # The oracle is the only component that may exclude the inadmissible action.
    oracle_value = _cohort_value(states, "oracle", MINIMUM_DOSE_STEPS)
    assert oracle_value == pytest.approx((0.05 + 0.02) / 2.0)


def test_leave_one_seed_out_uses_identical_fixed_semantics():
    from tools.analyze_idm_timing_oracle import _leave_one_seed_out, _cohort_value

    dose_failed, clean = _stage_dicts()
    states = [dose_failed, clean]
    loo = _leave_one_seed_out(states, MINIMUM_DOSE_STEPS)
    # Removing seed 4121 leaves only the clean state: oracle must beat fixed.
    remaining = [clean]
    fixed_best = max(
        ("abstain",) + ("activate_now", "defer_r2", "defer_r3"),
        key=lambda name: _cohort_value(remaining, name, MINIMUM_DOSE_STEPS),
    )
    assert loo["4121"] == pytest.approx(
        _cohort_value(remaining, "oracle", MINIMUM_DOSE_STEPS)
        - _cohort_value(remaining, fixed_best, MINIMUM_DOSE_STEPS)
    )
    # The per-seed subgroup of seed 4121 must retain the -2 penalty too.
    assert _cohort_value([dose_failed], fixed_best, MINIMUM_DOSE_STEPS) == pytest.approx(
        _cohort_value([dose_failed], fixed_best, MINIMUM_DOSE_STEPS)
    )


def _branch(state_id, arm, L, B, E, *, dose=20, r2=None, r3=None, unexpected=0, lowttc=0, **overrides):
    row = {
        "schema": "idm_timing_headroom_pilot/1",
        "state_id": state_id,
        "arm": arm,
        "demand": 2000,
        "seed": 4121,
        "cav_penetration": 0.10,
        "t1": 100,
        "t2": r2,
        "t3": r3,
        "L": L,
        "H": 100 + L,
        "fired": arm != "abstain",
        "completed_horizon_steps": L,
        "target_command_step_count": dose,
        "unexpected_command_count": unexpected,
        "severe_braking_count_sum": B,
        "acceleration_energy_sum": E,
        "ttc_count_sum": 0,
        "low_ttc_exposure_steps": lowttc,
        "mean_corridor_delay_rate": 0.5,
        "arrived_vehicles": 10,
        "merge_discharge_count": 4,
        "collision_observations": 0,
        "teleport_observations": 0,
        "member_changes": [],
        "release_rows": [],
        "identity": {"t1_identity": {"ok": True}},
        "runtime_s": 1.0,
    }
    row.update(overrides)
    return row


def test_analyze_computes_oracle_and_fixed_rules(tmp_path: Path):
    refs = tmp_path / "refs.json"
    refs.write_text(
        json.dumps(
            {
                "refs": {
                    "osm.dense.idm_m2000r400.sumocfg": {
                        "corridor_severe_braking_count_ref_per_s": 0.1,
                        "corridor_acceleration_energy_sum_ref_m2ps4": 10.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    pilot = tmp_path / "pilot"
    (pilot / "branches").mkdir(parents=True)
    # state A: t1=100, t2=180, t3=260 -> L = 220; defer_r3 has big effect
    state_a = "p0.10_m2000_seed4121"
    (pilot / "branches" / f"{state_a}.abstain.json").write_text(
        json.dumps(_branch(state_a, "abstain", 220, B=12.0, E=100.0, r2=180, r3=260)), encoding="utf-8"
    )
    (pilot / "branches" / f"{state_a}.abstain_replay.json").write_text(
        json.dumps(_branch(state_a, "abstain_replay", 220, B=12.0, E=100.0, r2=180, r3=260)), encoding="utf-8"
    )
    (pilot / "branches" / f"{state_a}.activate_now.json").write_text(
        json.dumps(_branch(state_a, "activate_now", 220, B=12.2, E=101.0, r2=180, r3=260)), encoding="utf-8"
    )
    (pilot / "branches" / f"{state_a}.defer_r2.json").write_text(
        json.dumps(_branch(state_a, "defer_r2", 220, B=11.0, E=98.0, r2=180, r3=260)), encoding="utf-8"
    )
    (pilot / "branches" / f"{state_a}.defer_r3.json").write_text(
        json.dumps(_branch(state_a, "defer_r3", 220, B=6.0, E=50.0, r2=180, r3=260)), encoding="utf-8"
    )
    # state B: t1=100, t2=180 only -> L = 140; all activation arms harmful
    state_b = "p0.10_m2000_seed4122"
    for arm, B, E in (("abstain", 8.0, 70.0), ("abstain_replay", 8.0, 70.0), ("activate_now", 9.0, 80.0), ("defer_r2", 9.5, 85.0)):
        row = _branch(state_b, arm, 140, B=B, E=E, r2=180, r3=None, seed=4122)
        row["seed"] = 4122
        (pilot / "branches" / f"{state_b}.{arm}.json").write_text(json.dumps(row), encoding="utf-8")

    result = analyze(pilot, refs)
    assert result["state_count"] == 2
    assert result["excluded_states"] == []
    assert result["oracle_passes"] in (True, False)
    assert result["oracle"]["mean_policy_reward"] > 0.0
    assert "defer_r3" in result["oracle"]["options_used"]
    assert result["best_fixed_deployable_timing_rule"] in ("abstain", "activate_now", "defer_r2", "defer_r3")
    assert "gains_by_demand" in result["oracle"]
