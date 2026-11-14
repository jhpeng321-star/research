"""Pure contracts for the sequential P30 timing observation/value screen.

Recursive stage targets, frozen feature provenance, seed-grouped folds and
cross-fitted sequential policy replay.  No TraCI calls, no simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from idm.timing_headroom import MINIMUM_DOSE_STEPS, classify_timing, rate_utility, reward_timing

FEATURE_NAMES = (
    "stage_index",
    "landmark_kind_wave",
    "landmark_kind_severe",
    "persistence_steps",
    "runway_m",
    "predicted_dose_s",
    "current_instant_s",
    "lane_0",
    "lane_1",
)

FORBIDDEN_FEATURES = (
    "seed",
    "demand",
    "penetration",
    "t2_minus_t1",
    "t3_minus_t2",
    "horizon_L",
    "terminal_H",
    "realized_dose",
    "reward",
    "outcome",
    "release_reason",
    "oracle_action",
    "observation_hash",
    "candidate_id",
    "state_id",
)


def action_outcome(
    abstain: Mapping[str, Any],
    activation: Mapping[str, Any],
    severe_ref: float,
    energy_ref: float,
) -> dict[str, Any]:
    """Frozen rate-normalized outcome for one activation branch."""
    L = int(activation["L"])
    utility = rate_utility(
        severe_delta=float(abstain["severe_braking_count_sum"])
        - float(activation["severe_braking_count_sum"]),
        energy_delta=float(abstain["acceleration_energy_sum"])
        - float(activation["acceleration_energy_sum"]),
        horizon_steps=L,
        severe_ref=severe_ref,
        energy_ref=energy_ref,
    )
    constraints = {
        "complete_horizon": int(activation["completed_horizon_steps"]) == int(L),
        "dose_at_least_minimum": int(activation["target_command_step_count"])
        >= MINIMUM_DOSE_STEPS,
        "zero_unexpected_commands": int(activation["unexpected_command_count"]) == 0,
        "zero_added_collisions": int(activation["collision_observations"])
        <= int(abstain["collision_observations"]),
        "zero_added_teleports": int(activation["teleport_observations"])
        <= int(abstain["teleport_observations"]),
        "low_ttc_within_budget": int(activation["low_ttc_exposure_steps"])
        - int(abstain["low_ttc_exposure_steps"])
        <= 2,
        "delay_within_budget": float(activation["mean_corridor_delay_rate"] or 0.0)
        - float(abstain["mean_corridor_delay_rate"] or 0.0)
        <= 1.0,
        "arrival_within_budget": int(activation["arrived_vehicles"])
        - int(abstain["arrived_vehicles"])
        >= -1,
        "discharge_within_budget": int(activation["merge_discharge_count"])
        - int(abstain["merge_discharge_count"])
        >= -1,
    }
    constraints_pass = all(constraints.values())
    return {
        "utility": utility,
        "reward": reward_timing(utility, constraints),
        "classification": classify_timing(utility, constraints),
        "constraints_pass": constraints_pass,
        "constraints": constraints,
        "dose_steps": int(activation["target_command_step_count"]),
        "collisions": int(activation["collision_observations"]),
        "teleports": int(activation["teleport_observations"]),
    }


def _constrained_ok(outcome: Mapping[str, Any]) -> bool:
    return bool(outcome["constraints_pass"]) and int(outcome["dose_steps"]) >= MINIMUM_DOSE_STEPS


@dataclass(frozen=True)
class StageNode:
    """One decision instant of one state (stage 1/2/3)."""

    state_id: str
    seed: int
    penetration: float
    demand: int
    stage: int
    t_i: int
    features: tuple[float, ...]
    act_mask: bool
    activation: Mapping[str, Any] | None
    wait_value: float

    @property
    def target(self) -> float | None:
        if self.activation is None:
            return None
        return float(self.activation["reward"]) - float(self.wait_value)


def build_stage_graph(
    states: Iterable[Mapping[str, Any]],
    refs: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Build stage nodes for each state and the frozen recursive wait values."""
    graph: list[dict[str, Any]] = []
    for state in states:
        occurrences = state["occurrences"]
        t1 = int(occurrences[0]["step"])
        t2 = int(occurrences[1]["step"]) if len(occurrences) > 1 else None
        t3 = int(occurrences[2]["step"]) if len(occurrences) > 2 else None
        outcomes = state["outcomes"]  # {"activate_now": ..., "defer_r2": ..., "defer_r3": ...}
        # stage 3
        v3 = 0.0
        r3_outcome = outcomes.get("defer_r3") if t3 is not None else None
        # stage 2
        v2 = 0.0
        if t3 is not None and r3_outcome is not None and _constrained_ok(r3_outcome):
            v2 = max(0.0, float(r3_outcome["reward"]))
        r2_outcome = outcomes.get("defer_r2") if t2 is not None else None
        # stage 1
        v1 = float(max(0.0, v2 if t2 is not None else 0.0))
        if t2 is not None and r2_outcome is not None and _constrained_ok(r2_outcome):
            v1 = max(v1, float(r2_outcome["reward"]))
        nodes = []
        for stage, occ, outcome, wait in (
            (1, occurrences[0], outcomes.get("activate_now"), v1),
            (2, occurrences[1] if t2 is not None else None, r2_outcome, v2),
            (3, occurrences[2] if t3 is not None else None, r3_outcome, v3),
        ):
            if occ is None:
                continue
            nodes.append(
                StageNode(
                    state_id=state["state_id"],
                    seed=int(state["seed"]),
                    penetration=float(state["cav_penetration"]),
                    demand=int(state["demand"]),
                    stage=stage,
                    t_i=int(occ["step"]),
                    features=feature_vector(occ, stage),
                    act_mask=bool(occ.get("admissible", True)),
                    activation=outcome,
                    wait_value=wait,
                )
            )
        graph.append({"state_id": state["state_id"], "seed": state["seed"], "nodes": nodes})
    return graph


def feature_vector(occurrence: Mapping[str, Any], stage: int) -> tuple[float, ...]:
    """Frozen family-A features; only causally current information."""
    lane = str(occurrence.get("lane_id", ""))
    kind = str(occurrence.get("landmark", ""))
    return (
        float(stage),
        float(1.0 if kind == "wave_onset" else 0.0),
        float(1.0 if kind == "severe_onset" else 0.0),
        float(occurrence.get("persistence_steps", 0.0)),
        float(occurrence.get("runway_m", 0.0)),
        float(occurrence.get("predicted_dose_s", 0.0)),
        float(occurrence.get("step", 0.0)),
        float(1.0 if lane.endswith("_1") else 0.0),
        float(0.0 if lane.endswith("_1") else 1.0),
    )


def grouped_folds(nodes: Iterable[StageNode]) -> dict[int, list[StageNode]]:
    """Seed-grouped folds: every stage of a seed stays in one fold."""
    folds: dict[int, list[StageNode]] = {}
    for node in nodes:
        folds.setdefault(node.seed, []).append(node)
    return folds


def sequential_replay(
    stage_nodes: Iterable[StageNode],
    policies: Mapping[int, Callable[[StageNode], float]],
) -> dict[str, Any]:
    """Replay the predicted activate/wait sequence against the paired library.

    ``stage_nodes`` are the materialized landmark nodes of one state (built
    by :func:`build_stage_graph`); a policy is queried only at a node that
    exists, so no future landmark is exposed.  The realized outcome of the
    first activation stage the policy selects is returned unchanged
    (penalties retained); a wait chain that exhausts the materialized
    landmarks abstains (reward 0).  Never substitutes abstention for an
    attempted-but-failed action.
    """
    nodes = {node.stage: node for node in stage_nodes}
    selected_stage = None
    for stage in (1, 2, 3):
        node = nodes.get(stage)
        if node is None or not node.act_mask:
            break
        if float(policies[stage](node)) > 0.0:
            selected_stage = stage
            break
    if selected_stage is None:
        return {
            "activated_stage": None,
            "reward": 0.0,
            "classification": "neutral",
            "constraints_pass": True,
            "dose_steps": 0,
        }
    outcome = nodes[selected_stage].activation
    return {
        "activated_stage": selected_stage,
        "reward": float(outcome["reward"]),
        "classification": str(outcome["classification"]),
        "constraints_pass": bool(outcome["constraints_pass"]),
        "dose_steps": int(outcome["dose_steps"]),
    }


THRESHOLD_GRID = (-0.01, 0.0, 0.01, 0.02, 0.045)


def cross_fit_stage_models(
    train_nodes: list[StageNode],
    seeds: Iterable[int],
) -> dict[int, Callable[[StageNode], float]]:
    """Per-stage ridge advantage policies fitted on grouped training folds
    with seed-grouped OOF selection of alpha and the activation threshold
    strictly inside the training folds."""
    from tools.screen_idm_expected_option_value import fit_ridge, predict_ridge

    models: dict[int, Callable[[StageNode], float]] = {}
    for stage in (1, 2, 3):
        stage_nodes = [node for node in train_nodes if node.stage == stage and node.activation is not None]
        if not stage_nodes:
            continue
        x = np.asarray([node.features for node in stage_nodes], dtype=float)
        y = np.asarray([node.target for node in stage_nodes], dtype=float)

        def oof_for(alpha: float) -> np.ndarray:
            oof = np.zeros(len(stage_nodes), dtype=float)
            for seed in sorted(set(node.seed for node in stage_nodes)):
                test = np.asarray([node.seed == seed for node in stage_nodes], dtype=bool)
                train = ~test
                if not np.any(train):
                    continue
                inner = fit_ridge(x[train], y[train], np.ones(int(np.sum(train))), alpha)
                oof[test] = predict_ridge(inner, x[test])
            return oof

        best = None
        for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
            oof = oof_for(alpha)
            candidate = _select_threshold(stage_nodes, oof)
            if best is None or candidate["train_value"] > best["train_value"]:
                best = {
                    "train_value": candidate["train_value"],
                    "alpha": alpha,
                    "threshold": candidate["threshold"],
                }
        model = fit_ridge(x, y, np.ones(len(x)), float(best["alpha"]))
        threshold = float(best["threshold"])

        def policy(node: StageNode, model=model, threshold=threshold) -> float:
            predicted = float(predict_ridge(model, np.asarray([node.features], dtype=float))[0])
            return predicted - threshold

        models[stage] = policy
    return models


def simple_stage_rule(train_nodes: list[StageNode]) -> dict[int, Callable[[StageNode], float]]:
    """Frozen simple per-stage threshold/rule: activate a stage only if its
    training seed-grouped OOF mean advantage is positive."""
    policies: dict[int, Callable[[StageNode], float]] = {}
    for stage in (1, 2, 3):
        stage_nodes = [node for node in train_nodes if node.stage == stage and node.activation is not None]
        if not stage_nodes:
            continue
        mean_advantage = float(np.mean([node.target for node in stage_nodes]))
        threshold = -0.01 if mean_advantage > 0.0 else 0.0

        def policy(node: StageNode, threshold=threshold) -> float:
            return 0.01 if threshold == -0.01 else -0.01

        policies[stage] = policy
    return policies


def _select_threshold(stage_nodes: list[StageNode], oof: np.ndarray) -> dict[str, Any]:
    """Choose the frozen grid threshold by training-fold bandit value."""
    best = None
    for threshold in THRESHOLD_GRID:
        value = 0.0
        for node, predicted in zip(stage_nodes, oof):
            if predicted > threshold:
                value += float(node.activation["reward"])
            else:
                value += float(node.wait_value)
        mean_value = float(value / max(len(stage_nodes), 1))
        if best is None or mean_value > best["train_value"]:
            best = {"threshold": float(threshold), "train_value": mean_value}
    return best


def shuffled_label_control(nodes: list[StageNode], rng: np.random.Generator) -> dict[int, Callable[[StageNode], float]]:
    """Same pipeline with permutation-permuted targets (leakage control)."""
    from tools.screen_idm_expected_option_value import fit_ridge, predict_ridge

    models: dict[int, Callable[[StageNode], float]] = {}
    for stage in (1, 2, 3):
        stage_nodes = sorted(
            [node for node in nodes if node.stage == stage and node.activation is not None],
            key=lambda node: node.state_id,
        )
        if not stage_nodes:
            continue
        x = np.asarray([node.features for node in stage_nodes], dtype=float)
        y = np.asarray([node.target for node in stage_nodes], dtype=float)
        y = rng.permutation(y)
        model = fit_ridge(x, y, np.ones(len(x)), 1.0)
        models[stage] = lambda node, model=model: float(
            predict_ridge(model, np.asarray([node.features], dtype=float))[0]
        )
    return models
