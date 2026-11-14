"""Retrospective timing-oracle analysis for the Path B headroom pilot.

Applies the frozen rate-normalized utility and constraints, computes fixed
timing baselines (never / now / defer-r2 / defer-r3 / best fixed deployable
timing rule) and the constrained retrospective oracle with leave-one-seed-out
gain, by-cell breakdowns, feasibility fractions and per-action value
attribution.  The oracle is unattainable evidence only and never authorizes
training on its own.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments.io import atomic_write_json
from idm.timing_headroom import (
    DEADBAND,
    MINIMUM_DOSE_STEPS,
    classify_timing,
    common_state_horizon,
    rate_utility,
    reward_timing,
    timing_gates,
)

SCHEMA = "idm_timing_oracle/1"
ACTION_ORDER = ("activate_now", "defer_r2", "defer_r3")


def _refs(payload: Mapping[str, Any], demand: int) -> Mapping[str, float]:
    row = payload["refs"][f"osm.dense.idm_m{demand}r400.sumocfg"]
    return {"severe": row["corridor_severe_braking_count_ref_per_s"], "energy": row["corridor_acceleration_energy_sum_ref_m2ps4"]}


def analyze(pilot_root: Path, refs_path: Path) -> dict[str, Any]:
    refs_payload = json.loads(refs_path.read_text(encoding="utf-8"))
    branches: list[dict[str, Any]] = []
    for path in sorted((pilot_root / "branches").glob("*.json")):
        branches.append(json.loads(path.read_text(encoding="utf-8")))
    by_state: dict[str, list[dict[str, Any]]] = {}
    for row in branches:
        by_state.setdefault(row["state_id"], []).append(row)

    states: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    replay_utilities: list[float] = []
    for state_id, rows in sorted(by_state.items()):
        indexed = {row["arm"]: row for row in rows}
        abstain = indexed.get("abstain")
        replay = indexed.get("abstain_replay")
        if abstain is None or replay is None:
            excluded.append({"state_id": state_id, "reason": "missing_abstain_or_replay"})
            continue
        first = rows[0]
        L = int(first["L"])
        refs = _refs(refs_payload, int(first["demand"]))
        replay_utility = rate_utility(
            severe_delta=float(abstain["severe_braking_count_sum"]) - float(replay["severe_braking_count_sum"]),
            energy_delta=float(abstain["acceleration_energy_sum"]) - float(replay["acceleration_energy_sum"]),
            horizon_steps=L,
            severe_ref=refs["severe"],
            energy_ref=refs["energy"],
        )
        if abs(replay_utility) > 1e-12:
            excluded.append({"state_id": state_id, "reason": "abstain_replay_mismatch", "utility": replay_utility})
            continue
        replay_utilities.append(abs(replay_utility))
        actions: dict[str, dict[str, Any]] = {
            "abstain": {
                "feasible": True,
                "reward": 0.0,
                "utility": 0.0,
                "classification": "neutral",
                "constraints_pass": True,
                "dose_steps": 0,
            }
        }
        for action in ACTION_ORDER:
            row = indexed.get(action)
            if row is None:
                actions[action] = {
                    "feasible": False,
                    "reward": 0.0,
                    "utility": 0.0,
                    "classification": "neutral",
                    "constraints_pass": True,
                    "dose_steps": 0,
                }
                continue
            utility = rate_utility(
                severe_delta=float(abstain["severe_braking_count_sum"]) - float(row["severe_braking_count_sum"]),
                energy_delta=float(abstain["acceleration_energy_sum"]) - float(row["acceleration_energy_sum"]),
                horizon_steps=L,
                severe_ref=refs["severe"],
                energy_ref=refs["energy"],
            )
            constraints = {
                "complete_horizon": int(row["completed_horizon_steps"]) == int(L),
                "dose_at_least_minimum": int(row["target_command_step_count"]) >= MINIMUM_DOSE_STEPS,
                "zero_unexpected_commands": int(row["unexpected_command_count"]) == 0,
                "zero_added_collisions": int(row["collision_observations"]) <= int(abstain["collision_observations"]),
                "zero_added_teleports": int(row["teleport_observations"]) <= int(abstain["teleport_observations"]),
                "low_ttc_within_budget": int(row["low_ttc_exposure_steps"]) - int(abstain["low_ttc_exposure_steps"]) <= 2,
                "delay_within_budget": float(row["mean_corridor_delay_rate"] or 0.0) - float(abstain["mean_corridor_delay_rate"] or 0.0) <= 1.0,
                "arrival_within_budget": int(row["arrived_vehicles"]) - int(abstain["arrived_vehicles"]) >= -1,
                "discharge_within_budget": int(row["merge_discharge_count"]) - int(abstain["merge_discharge_count"]) >= -1,
            }
            classification = classify_timing(utility, constraints)
            actions[action] = {
                "feasible": True,
                "reward": reward_timing(utility, constraints),
                "utility": utility,
                "classification": classification,
                "constraints_pass": all(constraints.values()),
                "dose_steps": int(row["target_command_step_count"]),
                "constraints": constraints,
                "identity": row["identity"],
            }
        states.append(
            {
                "state_id": state_id,
                "seed": int(first["seed"]),
                "penetration": float(first["cav_penetration"]),
                "demand": int(first["demand"]),
                "t1": int(first["t1"]),
                "t2": first["t2"],
                "t3": first["t3"],
                "recurrence": 3 if first["t3"] is not None else 2 if first["t2"] is not None else 1,
                "L": L,
                "H": int(first["H"]),
                "actions": actions,
            }
        )

    state_by_seed = {}
    for row in states:
        state_by_seed.setdefault(row["seed"], []).append(row)

    def policy_report(selector: str) -> tuple[float, Counter, int]:
        """Counts and failures for one selector under frozen semantics."""
        rewards = []
        classes = Counter()
        failures = 0
        for row in states:
            if selector == "oracle":
                best, best_reward = "abstain", 0.0
                for action in ACTION_ORDER:
                    entry = row["actions"][action]
                    if not entry["feasible"]:
                        continue
                    if not entry["constraints_pass"] or entry["dose_steps"] < MINIMUM_DOSE_STEPS:
                        continue
                    if entry["reward"] > best_reward:
                        best, best_reward = action, entry["reward"]
                chosen = row["actions"][best]
            else:
                entry = row["actions"].get(selector) or row["actions"]["abstain"]
                chosen = entry if entry["feasible"] else row["actions"]["abstain"]
            rewards.append(float(chosen["reward"]))
            classes[chosen["classification"]] += 1
            if chosen is not row["actions"]["abstain"] and not chosen["constraints_pass"]:
                failures += 1
        return float(np.mean(rewards)), classes, failures

    fixed_policy_values = {}
    for name in ("abstain",) + ACTION_ORDER:
        fixed_policy_values[name] = (
            fixed_policy_value(states, name, MINIMUM_DOSE_STEPS),
            *policy_report(name)[1:],
        )
    baseline: dict[str, dict[str, Any]] = {}
    for name, (mean, classes, fails) in fixed_policy_values.items():
        baseline[name] = {
            "mean_policy_reward": mean,
            "classification_counts": dict(classes),
            "constraint_failures": fails,
        }
    current_rule = "defer_r3"
    best_fixed_name = max(
        ("abstain",) + ACTION_ORDER,
        key=lambda name: fixed_policy_values[name][0],
    )
    best_fixed_mean = fixed_policy_values[best_fixed_name][0]

    oracle_mean, oracle_classes, oracle_failures = policy_report("oracle")
    per_seed_gain: dict[int, float] = {}
    seeds = sorted({row["seed"] for row in states})
    for seed in seeds:
        seed_states = [row for row in states if row["seed"] == seed]
        oracle_best_fixed = _cohort_value(seed_states, "oracle", MINIMUM_DOSE_STEPS)
        fixed_best = max(
            ("abstain",) + ACTION_ORDER,
            key=lambda name: _cohort_value(seed_states, name, MINIMUM_DOSE_STEPS),
        )
        per_seed_gain[seed] = oracle_best_fixed - _cohort_value(seed_states, fixed_best, MINIMUM_DOSE_STEPS)
    per_seed_gain_values = {
        str(seed): round(value, 5) for seed, value in per_seed_gain.items()
    }
    top_seed = max(per_seed_gain, key=per_seed_gain.get) if per_seed_gain else None
    gain = float(oracle_mean - best_fixed_mean)
    loo_gains = _leave_one_seed_out(states, MINIMUM_DOSE_STEPS)
    gain_without_top = loo_gains.get(str(top_seed)) if top_seed is not None else None
    options_used = _oracle_options(states, MINIMUM_DOSE_STEPS)
    outcome_counts = _oracle_outcomes(states, MINIMUM_DOSE_STEPS)
    cohort_diversity = _cohort_outcome_diversity(states)
    replay_noise = _replay_noise_ok(states, max(replay_utilities, default=0.0))
    dose_integrity = _dose_integrity(states)
    gates = timing_gates(
        oracle_mean=oracle_mean,
        fixed_best_mean=best_fixed_mean,
        per_seed_gain=per_seed_gain,
        options_used=options_used,
        outcomes=outcome_counts,
        constraint_failures=oracle_failures,
        loo_gains=loo_gains,
        cohort_diversity=cohort_diversity,
        replay_noise_ok=replay_noise,
    )
    return {
        "schema": SCHEMA,
        "state_count": len(states),
        "excluded_states": excluded,
        "fixed_baselines": baseline,
        "current_rule": current_rule,
        "best_fixed_deployable_timing_rule": best_fixed_name,
        "best_fixed_mean_reward": best_fixed_mean,
        "oracle": {
            "mean_policy_reward": oracle_mean,
            "gain_over_best_fixed_rule": gain,
            "classification_counts": dict(oracle_classes),
            "constraint_failures": oracle_failures,
            "options_used": dict(options_used),
            "per_seed_gain": per_seed_gain_values,
            "leave_one_seed_out_gain": loo_gains,
            "gain_without_top_seed": gain_without_top,
            "top_contributing_seed": top_seed,
            "gains_by_penetration": _gain_by(states, "penetration", MINIMUM_DOSE_STEPS),
            "gains_by_demand": _gain_by(states, "demand", MINIMUM_DOSE_STEPS),
            "gains_by_recurrence": _gain_by(states, "recurrence", MINIMUM_DOSE_STEPS),
            "feasibility_fractions": _feasibility_fractions(states),
            "value_attribution": _value_attribution(states, MINIMUM_DOSE_STEPS),
            "cohort_outcome_diversity": cohort_diversity,
            "dose_integrity": dose_integrity,
        },
        "gates": gates,
        "replay_utility_max": float(max(replay_utilities, default=0.0)),
        "oracle_passes": all(gates.values()),
    }


def fixed_policy_value(states: list[dict[str, Any]], action: str, minimum_dose: int) -> float:
    """Frozen fixed-policy semantics (identical at every scope).

    A fixed timing policy attempts every scheduled feasible action: if the
    realized dose falls below 16 steps or any other post-activation
    constraint fails, the frozen penalty (reward -2, classification
    harmful) is retained in the policy value.  Only a missing or
    pre-action-infeasible timing abstains.  The constrained retrospective
    oracle -- and only the oracle -- may exclude outcome-inadmissible
    actions, because it is an explicitly unattainable upper bound.  Callers
    must never substitute abstention for an attempted-but-failed dose.
    """
    rewards = []
    for row in states:
        entry = row["actions"].get(action) or row["actions"]["abstain"]
        chosen = entry if entry["feasible"] else row["actions"]["abstain"]
        rewards.append(float(chosen["reward"]))
    return float(np.mean(rewards)) if rewards else 0.0


def _replay_noise_ok(states: list[dict[str, Any]], replay_utility_max: float) -> bool:
    """Abstain replay is numerically exact (every state retained by analyze
    passed the 1e-12 abstain-vs-abstain-replay utility check) and realized
    treatment effects exceed the frozen 0.02 deadband in at least one
    outcome."""
    max_utility = 0.0
    for row in states:
        for action in ACTION_ORDER:
            entry = row["actions"][action]
            if entry["feasible"]:
                max_utility = max(max_utility, abs(float(entry["utility"])))
    return bool(replay_utility_max < 1e-9 and max_utility > DEADBAND)


def _dose_integrity(states: list[dict[str, Any]]) -> dict[str, Any]:
    """Attempted activations, dose success and retained-vs-excluded results."""
    attempted = 0
    achieved = 0
    by_action: dict[str, dict[str, Any]] = {}
    for action in ACTION_ORDER:
        count = 0
        ok = 0
        for row in states:
            entry = row["actions"][action]
            if not entry["feasible"]:
                continue
            count += 1
            if entry["dose_steps"] >= MINIMUM_DOSE_STEPS:
                ok += 1
        by_action[action] = {"attempted": count, "achieved_at_least_16": ok}
        attempted += count
        achieved += ok
    return {
        "attempted_activations": attempted,
        "achieved_at_least_16_steps": achieved,
        "fraction_achieving_16_of_20": round(achieved / max(attempted, 1), 4),
        "by_action": by_action,
        "fixed_policy_values_retaining_penalties": {
            action: round(fixed_policy_value(states, action, MINIMUM_DOSE_STEPS), 5)
            for action in ACTION_ORDER
        },
        "constrained_oracle_excludes_inadequate": True,
    }


def _leave_one_seed_out(states: list[dict[str, Any]], minimum_dose: int) -> dict[str, float]:
    """Oracle gain recomputed on the complement of each seed (frozen gate)."""
    seeds = sorted({row["seed"] for row in states})
    out = {}
    for seed in seeds:
        remaining = [row for row in states if row["seed"] != seed]
        oracle = _cohort_value(remaining, "oracle", minimum_dose)
        fixed = max(
            ("abstain",) + ACTION_ORDER,
            key=lambda name: _cohort_value(remaining, name, minimum_dose),
        )
        out[str(seed)] = round(oracle - _cohort_value(remaining, fixed, minimum_dose), 5)
    return out


def _cohort_outcome_diversity(states: list[dict[str, Any]]) -> dict[str, int]:
    """Beneficial/nonbeneficial timing realizations across all activation arms."""
    counts = Counter()
    for row in states:
        for action in ACTION_ORDER:
            entry = row["actions"][action]
            if entry["feasible"]:
                counts[entry["classification"]] += 1
    return dict(counts)


def _cohort_value(rows: list[dict[str, Any]], selector: str, minimum_dose: int) -> float:
    """Single value function used by the global baseline, per-seed and
    leave-one-seed-out evaluations: fixed selectors use frozen fixed-policy
    semantics (penalties retained); only 'oracle' uses the retrospective
    upper-bound semantics."""
    if selector != "oracle":
        return fixed_policy_value(rows, selector, minimum_dose)
    rewards = []
    for row in rows:
        best, best_reward = "abstain", 0.0
        for action in ACTION_ORDER:
            entry = row["actions"][action]
            if not entry["feasible"] or not entry["constraints_pass"] or entry["dose_steps"] < minimum_dose:
                continue
            if entry["reward"] > best_reward:
                best, best_reward = action, entry["reward"]
        rewards.append(float(row["actions"][best]["reward"]))
    return float(np.mean(rewards)) if rewards else 0.0


def _oracle_options(states: list[dict[str, Any]], minimum_dose: int) -> Counter:
    counts = Counter()
    for row in states:
        best, best_reward = "abstain", 0.0
        for action in ACTION_ORDER:
            entry = row["actions"][action]
            if not entry["feasible"] or not entry["constraints_pass"] or entry["dose_steps"] < minimum_dose:
                continue
            if entry["reward"] > best_reward:
                best, best_reward = action, entry["reward"]
        counts[best] += 1
    return Counter({key: value for key, value in counts.items() if key != "abstain" and value})


def _oracle_outcomes(states: list[dict[str, Any]], minimum_dose: int) -> Counter:
    counts = Counter()
    for row in states:
        best, best_reward = "abstain", 0.0
        for action in ACTION_ORDER:
            entry = row["actions"][action]
            if not entry["feasible"] or not entry["constraints_pass"] or entry["dose_steps"] < minimum_dose:
                continue
            if entry["reward"] > best_reward:
                best, best_reward = action, entry["reward"]
        counts[row["actions"][best]["classification"]] += 1
    return counts


def _gain_by(states: list[dict[str, Any]], key: str, minimum_dose: int) -> dict[str, float]:
    out = {}
    for group in sorted({str(row[key]) for row in states}):
        rows = [row for row in states if str(row[key]) == group]
        oracle = _cohort_value(rows, "oracle", minimum_dose)
        fixed = max(
            ("abstain",) + ACTION_ORDER,
            key=lambda name: _cohort_value(rows, name, minimum_dose),
        )
        out[group] = round(oracle - _cohort_value(rows, fixed, minimum_dose), 5)
    return out


def _feasibility_fractions(states: list[dict[str, Any]]) -> dict[str, float]:
    total = len(states)
    return {
        "activate_now": round(sum(1 for row in states if row["actions"]["activate_now"]["feasible"]) / max(total, 1), 3),
        "defer_r2": round(sum(1 for row in states if row["actions"]["defer_r2"]["feasible"]) / max(total, 1), 3),
        "defer_r3": round(sum(1 for row in states if row["actions"]["defer_r3"]["feasible"]) / max(total, 1), 3),
    }


def _value_attribution(states: list[dict[str, Any]], minimum_dose: int) -> dict[str, float]:
    total = 0.0
    per = Counter()
    for row in states:
        best, best_reward = "abstain", 0.0
        for action in ACTION_ORDER:
            entry = row["actions"][action]
            if not entry["feasible"] or not entry["constraints_pass"] or entry["dose_steps"] < minimum_dose:
                continue
            if entry["reward"] > best_reward:
                best, best_reward = action, entry["reward"]
        if best != "abstain":
            total += float(row["actions"][best]["reward"])
            per[best] += float(row["actions"][best]["reward"])
    total = max(total, 1e-9)
    return {action: round(per[action] / total, 4) for action in ACTION_ORDER if per[action] > 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--refs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.pilot.resolve(), args.refs.resolve())
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True)[:4500])


if __name__ == "__main__":
    main()
