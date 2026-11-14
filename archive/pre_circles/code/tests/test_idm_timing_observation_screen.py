"""Focused tests: recursive targets, provenance, grouped folds, sequential replay."""

from __future__ import annotations

import numpy as np
import pytest

from idm.timing_observation import (
    FORBIDDEN_FEATURES,
    FEATURE_NAMES,
    StageNode,
    build_stage_graph,
    grouped_folds,
    sequential_replay,
)


def _outcome(reward: float, *, constraints_pass: bool = True, dose: int = 20):
    return {
        "reward": reward,
        "constraints_pass": constraints_pass,
        "dose_steps": dose,
        "classification": "helpful" if reward > 0.02 else "harmful" if reward < -0.02 else "neutral",
        "utility": reward,
    }


def _occurrence(step: int, lane="1191554636_0"):
    return {
        "step": step,
        "landmark": "wave_onset",
        "persistence_steps": 3,
        "lane_id": lane,
        "runway_m": 500.0 + step,
        "predicted_dose_s": 20.0,
        "admissible": True,
    }


def _state(seed: int, t2: int | None, t3: int | None, now_reward: float, r2_reward: float | None, r3_reward: float | None, r2_admissible: bool = True, r3_admissible: bool = True):
    occurrences = [_occurrence(100)]
    if t2 is not None:
        occurrences.append(_occurrence(t2))
    if t3 is not None:
        occurrences.append(_occurrence(t3))
    outcomes = {"activate_now": _outcome(now_reward)}
    if r2_reward is not None:
        outcomes["defer_r2"] = _outcome(r2_reward, constraints_pass=r2_admissible, dose=20 if r2_admissible else 9)
    if r3_reward is not None:
        outcomes["defer_r3"] = _outcome(r3_reward, constraints_pass=r3_admissible, dose=20 if r3_admissible else 9)
    return {
        "state_id": f"s{seed}",
        "seed": seed,
        "cav_penetration": 0.10,
        "demand": 2000,
        "occurrences": occurrences,
        "outcomes": outcomes,
    }


def test_recursive_wait_values_chain_and_missing_landmarks():
    graph = build_stage_graph([_state(4121, 220, 340, now_reward=-2.0, r2_reward=0.05, r3_reward=0.03)], {"severe": 1.0, "energy": 1.0})
    nodes = {node.stage: node for node in graph[0]["nodes"]}
    assert nodes[3].wait_value == 0.0
    assert nodes[2].wait_value == pytest.approx(0.03)
    assert nodes[1].wait_value == pytest.approx(0.05)
    assert nodes[1].target == pytest.approx(-2.05)
    assert nodes[2].target == pytest.approx(0.05 - 0.03)
    # missing t3 collapses the wait chain
    graph2 = build_stage_graph([_state(4121, 220, None, -2.0, 0.05, None)], {"severe": 1.0, "energy": 1.0})
    nodes2 = {node.stage: node for node in graph2[0]["nodes"]}
    assert nodes2[2].wait_value == 0.0
    assert nodes2[1].wait_value == pytest.approx(0.05)
    # inadmissible r2 is excluded from the wait value
    graph3 = build_stage_graph([_state(4121, 220, None, -2.0, 0.05, None, r2_admissible=False)], {"severe": 1.0, "energy": 1.0})
    nodes3 = {node.stage: node for node in graph3[0]["nodes"]}
    assert nodes3[1].wait_value == 0.0
    assert nodes3[2].target == pytest.approx(0.05)


def test_feature_provenance_excludes_future_and_outcomes():
    assert FEATURE_NAMES
    forbidden_in_features = set(FORBIDDEN_FEATURES) & set(FEATURE_NAMES)
    assert forbidden_in_features == set()
    nodes = build_stage_graph([_state(4121, 220, 340, -2.0, 0.05, 0.03)], {"severe": 1.0, "energy": 1.0})[0]["nodes"]
    for node in nodes:
        assert len(node.features) == len(FEATURE_NAMES)
        assert node.features[0] == float(node.stage)
        assert 0.0 <= node.features[6]  # current instant, causal present time


def test_grouped_folds_keep_seed_together():
    states = [_state(4121, 220, 340, -2.0, 0.05, 0.03), _state(4121, None, None, -0.1, None, None), _state(4127, 220, None, -2.0, 0.02, None)]
    nodes = []
    for state in states:
        nodes.extend(build_stage_graph([state], {"severe": 1.0, "energy": 1.0})[0]["nodes"])
    folds = grouped_folds(nodes)
    assert set(folds) == {4121, 4127}
    assert all(node.seed == 4121 for node in folds[4121])
    assert all(node.seed == 4127 for node in folds[4127])


def test_sequential_replay_retains_attempted_penalties_and_waits():
    state = _state(4121, 220, 340, -2.0, 0.05, 0.03)
    nodes = build_stage_graph([state], {"severe": 1.0, "energy": 1.0})[0]["nodes"]
    # policy always activates at stage 1 -> realized reward -2 retained, not abstained
    always_now = sequential_replay(nodes, {1: lambda n: 1.0, 2: lambda n: -1.0, 3: lambda n: -1.0})
    assert always_now["activated_stage"] == 1
    assert always_now["reward"] == pytest.approx(-2.0)
    assert always_now["dose_steps"] == 20
    # wait until stage 2, activate -> r2 reward
    wait_then_r2 = sequential_replay(nodes, {1: lambda n: -1.0, 2: lambda n: 1.0, 3: lambda n: -1.0})
    assert wait_then_r2["activated_stage"] == 2
    assert wait_then_r2["reward"] == pytest.approx(0.05)
    # always waiting -> abstain 0
    never = sequential_replay(nodes, {1: lambda n: -1.0, 2: lambda n: -1.0, 3: lambda n: -1.0})
    assert never["activated_stage"] is None
    assert never["reward"] == 0.0


def test_lambda_policies_never_leak_held_seed():
    # feature vector must not contain the seed under any encoding of the tuple
    nodes = build_stage_graph([_state(4121, 220, 340, -2.0, 0.05, 0.03)], {"severe": 1.0, "energy": 1.0})[0]["nodes"]
    for node in nodes:
        assert 4121 not in node.features
        assert node.state_id not in node.features
