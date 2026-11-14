"""Focused tests for the RL timing/candidate option library and oracle headroom."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import tools.analyze_idm_rl_timing_oracle as oracle
from idm.rl_option_library import (
    ACTION_BY_SLOT,
    ActionOutcome,
    RLOptionRow,
    load_rl_library,
    rl_rows_from_raw_records,
    write_rl_library,
)


def _raw(seed: int, pen: float, demand: int, recurrence: int, step: int, slot: int, reward: float, cls: str = "neutral", dose: int = 20):
    return SimpleNamespace(
        sample_id=f"m{demand}.p{pen:.3f}.seed{seed}.d{recurrence}.slot{slot}",
        source="runs/eval/fake",
        seed=seed,
        penetration=pen,
        demand=demand,
        decision_index=recurrence,
        decision_step=step,
        vehicle_id=f"mainline.{slot}",
        observation=tuple([0.0] * 134),
        activate_reward=reward,
        physical_utility=reward,
        constraints_pass=True,
        classification=cls,
        dose_steps=dose,
    )


def test_grouping_builds_one_row_per_state_with_both_candidate_actions(monkeypatch):
    monkeypatch.setattr(
        "idm.rl_option_library.features_from_source",
        lambda source, slot, record: ({"slot": slot, "runway_m": float(slot * 10.0)}, {"idm_wave_energy_level": 1.0}),
    )
    raws = [
        _raw(4121, 0.05, 1900, 1, 100, 1, 0.10, "helpful"),
        _raw(4121, 0.05, 1900, 1, 100, 2, -0.05, "harmful"),
        _raw(4121, 0.05, 1900, 2, 220, 1, 0.0, "neutral"),
    ]
    rows = rl_rows_from_raw_records(raws, {})
    assert len(rows) == 2
    state = next(row for row in rows if row.recurrence == 1)
    assert state.actions["candidate_1"].reward == 0.10
    assert state.actions["candidate_2"].reward == -0.05
    assert state.actions["abstain"].reward == 0.0
    assert state.candidate_features["candidate_1"]["runway_m"] == 10.0
    single = next(row for row in rows if row.recurrence == 2)
    assert "candidate_2" not in single.candidate_features
    assert single.actions["candidate_2"].feasible is False


def _row(seed: int, demand: int, pen: float, recurrence: int, c1: float, c2: float | None, c1_cls: str = "neutral", c2_cls: str = "neutral") -> RLOptionRow:
    actions = {
        "abstain": ActionOutcome("abstain", True, 0.0, 0.0, True, "neutral", 0, None),
        "candidate_1": ActionOutcome("candidate_1", True, c1, c1, True, c1_cls, 20, "v1"),
    }
    if c2 is not None:
        actions["candidate_2"] = ActionOutcome("candidate_2", True, c2, c2, True, c2_cls, 20, "v2")
    else:
        actions["candidate_2"] = ActionOutcome("candidate_2", False, 0.0, 0.0, True, "neutral", 0, None)
    return RLOptionRow(
        sample_id=f"m{demand}.p{pen:.3f}.seed{seed}.r{recurrence}",
        source="runs/eval/fake",
        seed=seed,
        penetration=pen,
        demand=demand,
        recurrence=recurrence,
        decision_step=100 + recurrence,
        observation=tuple([0.0] * 134),
        candidate_features={},
        pattern_context={},
        actions=actions,
    )


def test_oracle_quantifies_gain_over_fixed_rule(monkeypatch):
    monkeypatch.setattr(oracle, "_replay_noise_ok", lambda *args: True)
    rows = [
        _row(4121, 1900, 0.05, 1, c1=0.01, c2=0.20, c2_cls="helpful"),
        _row(4127, 1900, 0.05, 2, c1=0.19, c2=None, c1_cls="helpful"),
        _row(4129, 1900, 0.10, 1, c1=0.00, c2=-0.10, c2_cls="harmful"),
        _row(4132, 1900, 0.10, 2, c1=0.05, c2=None, c1_cls="helpful"),
    ]
    result = oracle.analyze(rows, {4121, 4127}, {4129, 4132})
    assert result["validation_row_count"] == 2
    constrained = result["oracles"]["constrained"]
    assert constrained["mean_policy_reward"] > 0.0
    assert "gain_over_best_fixed_rule" in constrained
    assert "gates" in constrained
    # row 4129: c2 harmful; oracle abstains (0) vs best fixed rule (c1 0.00) -> no loss
    assert result["headroom_passes"] in (True, False)


def test_library_write_load_roundtrip(tmp_path):
    rows = [_row(4129, 1900, 0.10, 1, c1=0.02, c2=0.05)]
    manifest = write_rl_library(
        tmp_path / "lib", rows, {"reward_refs_path": "x"}, train_seeds=[4121], validation_seeds=[4129]
    )
    loaded, loaded_manifest = load_rl_library(tmp_path / "lib")
    assert loaded_manifest["record_count"] == 1
    assert loaded[0].actions["candidate_2"].reward == 0.05
    assert loaded[0].observation[0] == 0.0
    assert manifest["dataset_sha256"]


def test_each_slot_maps_to_expected_action_name():
    assert ACTION_BY_SLOT == {1: "candidate_1", 2: "candidate_2"}
