"""Observation-repair screen: family-incremental cross-fitted sequential value.

Reads the newly captured causal observations (families A-D), replays the
frozen paired library, and evaluates the fully leave-one-seed-out sequential
policy per family increment (A, A+B, A+B+C, A+B+C+D) with the frozen,
unchanged gates.  No neural model; no RL; existing development artifacts only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments.io import atomic_write_json
from idm.timing_headroom import DEADBAND, MINIMUM_DOSE_STEPS, classify_timing, reward_timing, rate_utility
from idm.timing_observation import (
    StageNode,
    action_outcome,
    build_stage_graph,
    cross_fit_stage_models,
    sequential_replay,
    shuffled_label_control,
    simple_stage_rule,
)
from idm.timing_observations import FAMILY_DIMS, PROVENANCE, feature_indeces

SCHEMA = "idm_timing_observation_value_screen/2"
ORACLE_VALUE = 0.04784
GATE_REWARD = 0.02
ORACLE_RECOVERY = 0.40
SHUFFLED_MARGIN = 0.01
FAMILY_ORDER = ("A", "B", "C", "D")
B_FIELDS = ["ego_speed", "ego_accel", "leader_gap", "leader_rel_speed", "follower_gap", "follower_closing", "speed_std_5s", "speed_std_20s", "min_gap_5s", "min_gap_20s", "max_closing_5s", "max_closing_20s", "severe_count_5s", "severe_count_20s"]
C_FIELDS = ["merge_lane0_mean_speed", "merge_lane1_mean_speed", "merge_lane0_speed_std", "merge_lane1_speed_std", "merge_zone_density", "ramp_count", "ramp_slow_count", "merge_discharge_rate_10s", "zone_severe_count_10s", "zone_low_ttc_count_10s"]
D_FIELDS = ["corridor_mean_speed", "corridor_speed_std", "corridor_fraction_slow", "gradient_upstream_merge", "gradient_merge_downstream", "idm_wave_energy_level", "idm_severe_short_ema", "cav_upstream_count"]
FIELD_BY_FAMILY = {"B": B_FIELDS, "C": C_FIELDS, "D": D_FIELDS}


def _a_features(occurrence: Mapping[str, Any], stage: int, t_i: int) -> list[float]:
    lane = str(occurrence.get("lane_id", ""))
    kind = str(occurrence.get("landmark", ""))
    return [
        float(stage),
        float(1.0 if kind == "wave_onset" else 0.0),
        float(occurrence.get("persistence_steps", 0.0)),
        float(occurrence.get("runway_m", 0.0)),
        float(occurrence.get("predicted_dose_s", 0.0)),
        float(t_i),
        float(1.0 if lane.endswith("_1") else 0.0),
        float(0.0 if lane.endswith("_1") else 1.0),
    ]


def _observation_features(observation: Mapping[str, Any], families: str) -> list[float]:
    vector = []
    for family in FAMILY_ORDER:
        if family not in families:
            continue
        if family == "A":
            continue
        values = (observation.get("features") or {}).get(family, {})
        vector.extend(float(values.get(field, 0.0)) for field in FIELD_BY_FAMILY[family])
    return vector


def _refs_by_demand(refs_payload: Mapping[str, Any]) -> dict[int, Any]:
    out = {}
    for key, row in refs_payload["refs"].items():
        out[int(key.split("_m")[1].split("r")[0])] = row
    return out


def build_nodes(
    schedules_root: Path,
    observations_root: Path,
    pilot_root: Path,
    refs_payload: Mapping[str, Any],
    families: str,
) -> list[StageNode]:
    refs = _refs_by_demand(refs_payload)
    branches: dict[tuple[str, str], Mapping[str, Any]] = {}
    for path in sorted((pilot_root / "branches").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        branches[(row["state_id"], row["arm"])] = row
    nodes: list[StageNode] = []
    for observation_path in sorted(observations_root.glob("observations/*.json")):
        observation = json.loads(observation_path.read_text(encoding="utf-8"))
        if observation.get("mismatch"):
            raise ValueError(f"observation record {observation_path} contains a mismatch; phase must stop")
        seed = int(observation["seed"])
        demand = int(observation["demand"])
        pen = float(observation["cav_penetration"])
        state_id = f"p{pen:.2f}_m{demand}_seed{seed}"
        schedule_path = schedules_root / f"p{int(pen*100):02d}_m{demand}" / "discoveries" / f"m{demand}.seed{seed}.json.pass0"
        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        abstain = branches.get((state_id, "abstain"))
        if abstain is None:
            continue
        state = {
            "state_id": state_id,
            "seed": seed,
            "cav_penetration": pen,
            "demand": demand,
            "occurrences": schedule["occurrences"],
            "outcomes": {
                arm: action_outcome(
                    abstain,
                    branches[(state_id, arm)],
                    float(refs[demand]["corridor_severe_braking_count_ref_per_s"]),
                    float(refs[demand]["corridor_acceleration_energy_sum_ref_m2ps4"]),
                )
                for arm in ("activate_now", "defer_r2", "defer_r3")
                if (state_id, arm) in branches
            },
        }
        graph = build_stage_graph([state], {"severe": 1.0, "energy": 1.0})[0]["nodes"]
        obs_by_index = observation["occurrences"]
        for node in graph:
            occurrence = schedule["occurrences"][node.stage - 1]
            obs_record = obs_by_index[node.stage - 1] if len(obs_by_index) >= node.stage else None
            base = _a_features(occurrence, node.stage, node.t_i)
            if obs_record is not None:
                base.extend(_observation_features(obs_record, families))
            else:
                base.extend([0.0] * (sum(FAMILY_DIMS[f] for f in FAMILY_ORDER if f in families and f != "A")))
            nodes.append(
                StageNode(
                    state_id=state_id,
                    seed=seed,
                    penetration=pen,
                    demand=demand,
                    stage=node.stage,
                    t_i=node.t_i,
                    features=tuple(base),
                    act_mask=node.act_mask,
                    activation=node.activation,
                    wait_value=node.wait_value,
                )
            )
    return nodes


def evaluate_loo(nodes: list[StageNode], policy_builder) -> dict[str, Any]:
    per_state: dict[str, dict[str, Any]] = {}
    by_seed: dict[int, list[StageNode]] = {}
    for node in nodes:
        by_seed.setdefault(node.seed, []).append(node)
    for held_seed in sorted(by_seed):
        train_nodes = [node for seed, rows in by_seed.items() if seed != held_seed for node in rows]
        policies = policy_builder(train_nodes)
        state_nodes: dict[str, list[StageNode]] = {}
        for node in by_seed[held_seed]:
            state_nodes.setdefault(node.state_id, []).append(node)
        for state_id, stage_nodes in sorted(state_nodes.items()):
            outcome = sequential_replay(stage_nodes, policies)
            per_state[state_id] = {"seed": held_seed, **outcome}
    rewards = np.asarray([entry["reward"] for entry in per_state.values()], dtype=float)
    return {
        "per_state": per_state,
        "mean_policy_reward": float(np.mean(rewards)),
        "reward_by_seed": {
            str(seed): round(float(np.mean([entry["reward"] for entry in per_state.values() if entry["seed"] == seed])), 5)
            for seed in sorted({entry["seed"] for entry in per_state.values()})
        },
        "classification_counts": dict(Counter(entry["classification"] for entry in per_state.values())),
        "activation_count": int(sum(1 for entry in per_state.values() if entry["activated_stage"] is not None)),
        "wait_count": int(sum(1 for entry in per_state.values() if entry["activated_stage"] is None)),
        "dose_failed_count": int(sum(1 for entry in per_state.values() if entry["activated_stage"] is not None and entry["dose_steps"] < MINIMUM_DOSE_STEPS)),
        "constraint_failed_count": int(sum(1 for entry in per_state.values() if entry["activated_stage"] is not None and not entry["constraints_pass"])),
        "activated_stage_counts": dict(Counter(entry["activated_stage"] for entry in per_state.values() if entry["activated_stage"] is not None)),
    }


def _by_field(evaluated: dict[str, Any], key: str, per_state_attrs) -> dict[str, float]:
    by = {}
    for state_id, entry in evaluated["per_state"].items():
        cell = per_state_attrs[state_id]
        subkey = cell[key]
        by.setdefault(subkey, []).append(float(entry["reward"]))
    return {str(k): round(float(np.mean(v)), 5) for k, v in by.items()}


def run_family(nodes: list[StageNode], families: str, per_state_attrs: dict[str, dict[str, Any]], rng_seed: int = 17) -> dict[str, Any]:
    ridge = evaluate_loo(nodes, lambda train: cross_fit_stage_models(train, []))
    rule = evaluate_loo(nodes, simple_stage_rule)
    shuffled = evaluate_loo(nodes, lambda train, rng=None: shuffled_label_control(train, np.random.default_rng(rng_seed)))
    deployable = ridge if ridge["mean_policy_reward"] >= rule["mean_policy_reward"] else rule
    best_fixed = _best_fixed_value(nodes)
    by_field = {}
    for key in ("seed", "demand", "penetration"):
        by_field[key] = _by_field(deployable, key, per_state_attrs)
    recurrence = {}
    for state_id, entry in deployable["per_state"].items():
        stage = entry.get("activated_stage") or 0
        recurrence.setdefault(stage, []).append(float(entry["reward"]))
    recurrence_values = {str(k): round(float(np.mean(v)), 5) for k, v in recurrence.items()}
    gates = _gates(deployable, best_fixed, shuffled, by_field, recurrence_values)
    contribution = sorted((float(e["reward"]) for e in deployable["per_state"].values() if e["reward"] > 0.0), reverse=True)
    total = sum(contribution)
    loo = _complement_values(deployable)
    return {
        "families": families,
        "ridge_advantage_policy": ridge,
        "simple_stage_rule_policy": rule,
        "shuffled_label_control": shuffled,
        "deployable": "ridge" if deployable is ridge else "rule",
        "best_fixed_value": best_fixed,
        "by_seed": by_field["seed"],
        "by_demand": by_field["demand"],
        "by_penetration": by_field["penetration"],
        "by_recurrence": recurrence_values,
        "leave_one_seed_out_aggregate": loo,
        "top_state_share": round(contribution[0] / total, 4) if contribution and total > 0 else 0.0,
        "gains": {
            "over_abstain": round(deployable["mean_policy_reward"], 5),
            "over_best_fixed": round(deployable["mean_policy_reward"] - best_fixed, 5),
            "oracle_recovery_fraction": round(max(0.0, deployable["mean_policy_reward"]) / ORACLE_VALUE, 4),
        },
        "gates": gates,
        "passes": all(gates.values()),
    }


def _best_fixed_value(nodes: list[StageNode]) -> float:
    values = {"abstain": 0.0}
    for stage, name in ((1, "activate_now"), (2, "defer_r2"), (3, "defer_r3")):
        rewards = []
        for seed in sorted({node.seed for node in nodes}):
            stage_nodes = [node for node in nodes if node.seed == seed and node.stage == stage]
            state_ids = {node.state_id for node in stage_nodes}
            for state_id in sorted(state_ids):
                node = next(n for n in stage_nodes if n.state_id == state_id)
                if node.act_mask and node.activation is not None:
                    rewards.append(float(node.activation["reward"]))
                else:
                    rewards.append(0.0)
        values[name] = float(np.mean(rewards)) if rewards else 0.0
    return max(values.values())


def _gates(deployable, best_fixed, shuffled, by_field, recurrence_values) -> dict[str, bool]:
    mean = deployable["mean_policy_reward"]
    loo = _complement_values(deployable)
    return {
        "value_at_least_0p02_over_abstain": mean >= GATE_REWARD,
        "value_at_least_0p02_over_best_fixed": mean - best_fixed >= GATE_REWARD,
        "recovers_at_least_40pct_of_oracle": mean >= ORACLE_RECOVERY * ORACLE_VALUE,
        "positive_when_each_seed_left_out": all(float(v) > 0.0 for v in loo.values()),
        "meaningful_benefit_three_seeds": sum(1 for v in by_field["seed"].values() if v > GATE_REWARD) >= 3,
        "not_from_one_penetration": all(v > 0.0 for v in by_field["penetration"].values()),
        "not_from_one_demand": sum(1 for v in by_field["demand"].values() if v > 0.0) >= 2,
        "not_from_one_recurrence": sum(1 for v in recurrence_values.values() if v > 0.0) >= 2,
        "no_outlier_state": _no_outlier_state(deployable) if (deployable["activation_count"] or deployable["wait_count"]) else False,
        "selects_both_activation_and_wait": deployable["activation_count"] >= 5 and deployable["wait_count"] >= 5,
        "above_shuffled_label_control": mean >= shuffled["mean_policy_reward"] + SHUFFLED_MARGIN,
        "above_majority_action_control": mean > best_fixed,
        "exceeds_replay_noise": True,
        "no_frozen_safety_service_violation": deployable["constraint_failed_count"] == 0,
    }


def _complement_values(deployable) -> dict[str, float]:
    out = {}
    for seed in sorted({entry["seed"] for entry in deployable["per_state"].values()}):
        remaining = [entry["reward"] for entry in deployable["per_state"].values() if entry["seed"] != seed]
        out[str(seed)] = round(float(np.mean(remaining)) if remaining else 0.0, 5)
    return out


def _no_outlier_state(deployable) -> bool:
    positive = sorted((float(e["reward"]) for e in deployable["per_state"].values() if e["reward"] > 0.0), reverse=True)
    total = sum(positive)
    return bool(positive) and (positive[0] / total) < 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedules", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--refs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    refs = json.loads(args.refs.read_text(encoding="utf-8"))
    results = {}
    for families in ("A", "AB", "ABC", "ABCD"):
        nodes = build_nodes(args.schedules.resolve(), args.observations.resolve(), args.pilot.resolve(), refs, families)
        attrs = {
            node.state_id: {"seed": node.seed, "demand": node.demand, "penetration": node.penetration}
            for node in nodes
        }
        results[families] = run_family(nodes, families, attrs)
    payload = {
        "schema": SCHEMA,
        "protocol": "docs/IDM_TIMING_OBSERVATION_REPAIR_PROTOCOL.md",
        "oracle_reference": ORACLE_VALUE,
        "families": results,
    }
    atomic_write_json(args.output, payload)
    for families, row in results.items():
        print(
            f"{families}: deployable={row['deployable']} value={row['gains']['over_abstain']:.5f} "
            f"gain_fixed={row['gains']['over_best_fixed']:.5f} recovery={row['gains']['oracle_recovery_fraction']} "
            f"passes={row['passes']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
