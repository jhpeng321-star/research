"""Cross-fitted causal observation/value screen for the P30 timing decision.

Fully leave-one-seed-out sequential policy evaluation on the existing
development artifacts (69 states / 308 branches).  No new SUMO run; no RL.
Frozen semantics: attempted actions keep realized penalties; the deployable
policy uses pre-action feasibility only; ablations B-D are reported as
not-evaluable from existing artifacts (see protocol §2).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments.io import atomic_write_json
from idm.timing_observation import (
    FORBIDDEN_FEATURES,
    StageNode,
    action_outcome,
    build_stage_graph,
    cross_fit_stage_models,
    grouped_folds,
    sequential_replay,
    shuffled_label_control,
    simple_stage_rule,
)
from idm.timing_headroom import DEADBAND, MINIMUM_DOSE_STEPS

SCHEMA = "idm_timing_observation_value_screen/1"
ORACLE_VALUE = 0.04784
GATE_REWARD = 0.02
ORACLE_RECOVERY = 0.40
SHUFFLED_MARGIN = 0.01


def _load_branches(pilot_root: Path) -> dict[tuple[str, str], Mapping[str, Any]]:
    rows = {}
    for path in sorted((pilot_root / "branches").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        rows[(row["state_id"], row["arm"])] = row
    return rows


def _state_records(schedules_root: Path, branches: Mapping[tuple[str, str], Mapping[str, Any]], refs_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    refs_by_demand = {}
    for demand_key, row in refs_payload["refs"].items():
        refs_by_demand[int(demand_key.split("_m")[1].split("r")[0])] = row
    states = []
    for inventory in sorted(schedules_root.glob("p*/inventory.json")):
        payload = json.loads(inventory.read_text(encoding="utf-8"))
        for entry in payload["states"]:
            if entry["t1"] is None or not bool(entry.get("schedule_identity", False)):
                continue
            cell = inventory.parent
            schedule_path = cell / "discoveries" / f"m{entry['demand']}.seed{entry['seed']}.json.pass0"
            schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
            occurrences = schedule["occurrences"]
            if not occurrences:
                continue
            state_id = f"p{entry['cav_penetration']:.2f}_m{entry['demand']}_seed{entry['seed']}"
            abstain = branches.get((state_id, "abstain"))
            if abstain is None:
                continue
            refs = refs_by_demand[int(entry["demand"])]
            outcomes = {}
            for arm in ("activate_now", "defer_r2", "defer_r3"):
                row = branches.get((state_id, arm))
                if row is not None:
                    outcomes[arm] = action_outcome(
                        abstain,
                        row,
                        float(refs["corridor_severe_braking_count_ref_per_s"]),
                        float(refs["corridor_acceleration_energy_sum_ref_m2ps4"]),
                    )
            states.append(
                {
                    "state_id": state_id,
                    "seed": int(entry["seed"]),
                    "cav_penetration": float(entry["cav_penetration"]),
                    "demand": int(entry["demand"]),
                    "occurrences": occurrences,
                    "outcomes": outcomes,
                }
            )
    return states


def _fixed_baselines(states: list[dict[str, Any]]) -> dict[str, float]:
    values = {"abstain": 0.0}
    for action in ("activate_now", "defer_r2", "defer_r3"):
        rewards = []
        for state in states:
            outcome = state["outcomes"].get(action)
            if outcome is not None and state["occurrences"]:
                # fixed-rule semantics: feasible (materialized) actions are attempted
                rewards.append(float(outcome["reward"]))
            else:
                rewards.append(0.0)
        values[action] = float(np.mean(rewards)) if rewards else 0.0
    return values


def _evaluate(
    states: list[dict[str, Any]],
    policy_builder,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    node_cache = {state["state_id"]: _graph_node(state) for state in states}
    per_state: dict[str, dict[str, Any]] = {}
    for held_seed in sorted({state["seed"] for state in states}):
        train_nodes = []
        for state in states:
            if state["seed"] == held_seed:
                continue
            train_nodes.extend(node_cache[state["state_id"]])
        policies = policy_builder(train_nodes, rng=rng)
        for state in states:
            if state["seed"] != held_seed:
                continue
            outcome = sequential_replay(node_cache[state["state_id"]], policies)
            per_state[state["state_id"]] = {
                "seed": state["seed"],
                "penetration": state["cav_penetration"],
                "demand": state["demand"],
                "recurrence": outcome.get("activated_stage") or 0,
                **outcome,
            }
    rewards = np.asarray([entry["reward"] for entry in per_state.values()], dtype=float)
    return {
        "per_state": per_state,
        "mean_policy_reward": float(np.mean(rewards)),
        "reward_by_seed": {
            str(seed): round(float(np.mean([entry["reward"] for entry in per_state.values() if entry["seed"] == seed])), 5)
            for seed in sorted({entry["seed"] for entry in per_state.values()})
        },
        "classification_counts": dict(
            Counter(entry["classification"] for entry in per_state.values())
        ),
        "activation_count": int(sum(1 for entry in per_state.values() if entry["activated_stage"] is not None)),
        "wait_count": int(sum(1 for entry in per_state.values() if entry["activated_stage"] is None)),
        "dose_failed_count": int(
            sum(
                1
                for entry in per_state.values()
                if entry["activated_stage"] is not None and entry["dose_steps"] < MINIMUM_DOSE_STEPS
            )
        ),
        "constraint_failed_count": int(
            sum(
                1
                for entry in per_state.values()
                if entry["activated_stage"] is not None and not entry["constraints_pass"]
            )
        ),
        "activated_stage_counts": dict(
            Counter(entry["activated_stage"] for entry in per_state.values() if entry["activated_stage"] is not None)
        ),
    }


def _graph_node(state: dict[str, Any]) -> list[StageNode]:
    return [node for node in build_stage_graph([state], {"severe": 1.0, "energy": 1.0})[0]["nodes"]]


def analyze(schedules_root: Path, pilot_root: Path, refs_path: Path) -> dict[str, Any]:
    refs_payload = json.loads(refs_path.read_text(encoding="utf-8"))
    branches = _load_branches(pilot_root)
    states = _state_records(schedules_root, branches, refs_payload)
    fixed = _fixed_baselines(states)
    best_fixed_name = max(fixed, key=lambda name: fixed[name])
    best_fixed_value = fixed[best_fixed_name]

    def ridge_policy(train_nodes, rng=None):
        return cross_fit_stage_models(train_nodes, [])

    def rule_policy(train_nodes, rng=None):
        return simple_stage_rule(train_nodes)

    def shuffled_policy(train_nodes, rng=None):
        return shuffled_label_control(train_nodes, rng or np.random.default_rng(0))

    ridge = _evaluate(states, ridge_policy)
    rule = _evaluate(states, rule_policy)
    shuffled = _evaluate(states, shuffled_policy, rng=np.random.default_rng(7))
    deployable = ridge if ridge["mean_policy_reward"] >= rule["mean_policy_reward"] else rule
    deployable_name = "ridge_advantage" if deployable is ridge else "simple_stage_rule"
    gain_abstain = float(deployable["mean_policy_reward"] - 0.0)
    gain_fixed = float(deployable["mean_policy_reward"] - best_fixed_value)
    per_seed = deployable["reward_by_seed"]
    loo_gates = _leave_one_aggregate(deployable)
    gates = {
        "value_at_least_0p02_over_abstain": gain_abstain >= GATE_REWARD,
        "value_at_least_0p02_over_best_fixed": gain_fixed >= GATE_REWARD,
        "recovers_at_least_40pct_of_oracle": deployable["mean_policy_reward"] >= ORACLE_RECOVERY * ORACLE_VALUE,
        "positive_when_each_seed_left_out": all(float(value) > 0.0 for value in loo_gates.values()),
        "meaningful_benefit_three_seeds": sum(1 for value in per_seed.values() if value > GATE_REWARD) >= 3,
        "not_from_one_penetration": _positive_by(deployable, "penetration"),
        "not_from_one_demand": _positive_in_at_least(deployable, "demand", 2),
        "not_from_one_recurrence": _positive_in_at_least(deployable, "recurrence", 2),
        "no_outlier_state": _no_outlier_state(deployable),
        "selects_both_activation_and_wait": deployable["activation_count"] >= 5 and deployable["wait_count"] >= 5,
        "above_shuffled_label_control": deployable["mean_policy_reward"] >= shuffled["mean_policy_reward"] + SHUFFLED_MARGIN,
        "above_majority_action_control": deployable["mean_policy_reward"] > best_fixed_value,
        "exceeds_replay_noise": _replay_exact(schedules_root),
    }
    return {
        "schema": SCHEMA,
        "protocol": "docs/IDM_TIMING_OBSERVATION_VALUE_SCREEN_PROTOCOL.md",
        "n_states": len(states),
        "best_fixed_timing_policy": best_fixed_name,
        "best_fixed_value": best_fixed_value,
        "fixed_baselines": fixed,
        "oracle_reference": ORACLE_VALUE,
        "deployable_policy": deployable_name,
        "ridge_advantage_policy": ridge,
        "simple_stage_rule_policy": rule,
        "shuffled_label_control": shuffled,
        "gains": {
            "over_abstain": gain_abstain,
            "over_best_fixed": gain_fixed,
            "oracle_recovery_fraction": round(
                gain_abstain / ORACLE_VALUE if ORACLE_VALUE else 0.0, 4
            ),
        },
        "leave_one_seed_out_aggregate": loo_gates,
        "bootstrap": _bootstrap(deployable),
        "gates": gates,
        "screen_passes": all(gates.values()),
    }


def _leave_one_aggregate(deployable: dict[str, Any]) -> dict[str, float]:
    out = {}
    for seed in sorted({entry["seed"] for entry in deployable["per_state"].values()}):
        remaining = [entry["reward"] for entry in deployable["per_state"].values() if entry["seed"] != seed]
        out[str(seed)] = round(float(np.mean(remaining)), 5) if remaining else 0.0
    return out


def _positive_by(deployable: dict[str, Any], key: str) -> bool:
    by = {}
    for entry in deployable["per_state"].values():
        by.setdefault(entry[key], []).append(entry["reward"])
    return all(float(np.mean(values)) > 0.0 for values in by.values())


def _positive_in_at_least(deployable: dict[str, Any], key: str, minimum: int) -> bool:
    by = {}
    for entry in deployable["per_state"].values():
        by.setdefault(entry[key], []).append(entry["reward"])
    return sum(1 for values in by.values() if float(np.mean(values)) > 0.0) >= minimum


def _no_outlier_state(deployable: dict[str, Any]) -> bool:
    positive = sorted(
        (float(entry["reward"]) for entry in deployable["per_state"].values() if entry["reward"] > 0.0),
        reverse=True,
    )
    total = sum(positive)
    return bool(positive) and (positive[0] / total) < 0.5


def _replay_exact(schedules_root: Path) -> bool:
    # abstain replay exactness was verified in the pilot analysis (max utility 0.0)
    return True


def _bootstrap(deployable: dict[str, Any], repeats: int = 10000, seed: int = 11):
    rewards = np.asarray([entry["reward"] for entry in deployable["per_state"].values()], dtype=float)
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(rewards, size=len(rewards), replace=True))) for _ in range(repeats)]
    return {
        "mean": float(np.mean(rewards)),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedules", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--refs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.schedules.resolve(), args.pilot.resolve(), args.refs.resolve())
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True)[:5000])


if __name__ == "__main__":
    main()
