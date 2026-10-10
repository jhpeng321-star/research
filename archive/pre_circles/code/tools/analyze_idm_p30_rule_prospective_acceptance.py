"""Frozen prospective acceptance analysis for the rule-based P30 arm.

The analysis treats every requested seed/cell as one deployable opportunity.
If the frozen third-risk-onset decision is absent, the rule abstains and its
cell value is exactly zero.  Attempted actions retain every physical,
safety, service, and dose failure.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments.io import atomic_write_json
from idm.contract import DAMPING_ACCEPTANCE_SEEDS
from idm.option_reward import IDM_BINARY_OPTION_REWARD_V1


SCHEMA = "idm_p30_rule_prospective_acceptance/1"
DEADBAND = 0.02
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20260830
DEMANDS = (1900, 2000, 2100)
PENETRATIONS = (0.05, 0.10)
REWARD_REFS_SHA256 = "737b9ce4bb1945f7681b807941c86509ae3143128bd0729d2cb2aee5237c5b58"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _nested(row: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
    value: Any = row
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return float(default)
        value = value[key]
    return float(value)


def _replay_error(control: Mapping[str, Any], replay: Mapping[str, Any]) -> float:
    scalar_keys = (
        "severe_braking_count_sum",
        "acceleration_energy_sum",
        "low_ttc_exposure_steps",
        "mean_corridor_delay_rate",
        "arrived_vehicles",
        "merge_discharge_count",
        "collision_observations",
        "teleport_observations",
        "completed_horizon_steps",
    )
    errors = [abs(float(control[key]) - float(replay[key])) for key in scalar_keys]
    for region in ("corridor", "immediate_follower", "influence_region", "lane"):
        for metric in (
            "acceleration_energy_m2ps4",
            "integrated_negative_acceleration_mps",
            "low_ttc_count",
            "severe_braking_count",
        ):
            errors.append(
                abs(
                    _nested(control, "spatial_totals", region, metric)
                    - _nested(replay, "spatial_totals", region, metric)
                )
            )
    return max(errors, default=0.0)


def _constraints(
    control: Mapping[str, Any], treatment: Mapping[str, Any]
) -> dict[str, bool]:
    contract = IDM_BINARY_OPTION_REWARD_V1
    return {
        "complete_horizon": int(treatment["completed_horizon_steps"])
        == contract.horizon_steps,
        "dose_at_least_minimum": int(treatment["target_command_step_count"])
        >= contract.minimum_dose_steps,
        "zero_unexpected_commands": int(treatment["unexpected_command_count"]) == 0,
        "zero_added_collisions": int(treatment["collision_observations"])
        <= int(control["collision_observations"]),
        "zero_added_teleports": int(treatment["teleport_observations"])
        <= int(control["teleport_observations"]),
        "low_ttc_within_budget": int(treatment["low_ttc_exposure_steps"])
        - int(control["low_ttc_exposure_steps"])
        <= contract.low_ttc_excess_budget_steps,
        "immediate_follower_braking_noninferior": _nested(
            treatment,
            "spatial_totals",
            "immediate_follower",
            "severe_braking_count",
        )
        - _nested(
            control,
            "spatial_totals",
            "immediate_follower",
            "severe_braking_count",
        )
        <= 0.0,
        "delay_within_budget": float(treatment["mean_corridor_delay_rate"])
        - float(control["mean_corridor_delay_rate"])
        <= contract.maximum_delay_rate_regression,
        "arrival_within_budget": int(treatment["arrived_vehicles"])
        - int(control["arrived_vehicles"])
        >= -contract.maximum_arrival_loss,
        "discharge_within_budget": int(treatment["merge_discharge_count"])
        - int(control["merge_discharge_count"])
        >= -contract.maximum_discharge_loss,
        "zero_treatment_safety_aborts": not bool(
            treatment.get("treatment_safety_abort_rows")
        ),
    }


def _acceptance_reward(physical: float, constraints: Mapping[str, bool]) -> float:
    return (
        math.tanh(float(physical))
        if all(bool(value) for value in constraints.values())
        else IDM_BINARY_OPTION_REWARD_V1.hard_failure_reward
    )


def _cell_directory(root: Path, demand: int, penetration: float) -> Path:
    slug = f"p{int(round(100 * penetration)):02d}"
    return root / f"idm_paired_option_grid_{slug}_m{demand}_l3_v1"


def _validate_protocol(
    protocol: Mapping[str, Any], *, demand: int, penetration: float
) -> None:
    expected = {
        "development_only": False,
        "evidence_role": "damping_acceptance",
        "demand": demand,
        "seeds": list(DAMPING_ACCEPTANCE_SEEDS),
        "action_set": "activation_speed_ceiling_v1",
        "terminal_reference_percentile": "p30",
        "option_steps": 20,
        "minimum_dose_steps": 16,
        "cav_penetration": penetration,
        "decision_index": 3,
        "candidate_ranking": "runway",
        "candidate_count": 1,
        "safety_horizon_steps": 60,
        "service_horizon_steps": 60,
        "max_predecision_steps": 900,
        "minimum_decision_step": 0,
        "landmark": "risk_onset",
        "maximum_landmarks_per_cell": 3,
        "discover_only": False,
        "decision_source_path": None,
        "reward_refs_sha256": REWARD_REFS_SHA256,
    }
    mismatches = [
        key for key, value in expected.items() if protocol.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "prospective cell protocol mismatch: " + ", ".join(sorted(mismatches))
        )


def _seed_means(rows: list[dict[str, Any]], key: str) -> dict[int, float]:
    return {
        seed: float(np.mean([float(row[key]) for row in rows if row["seed"] == seed]))
        for seed in sorted({int(row["seed"]) for row in rows})
    }


def _bootstrap_interval(values: list[float]) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    array = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_RESAMPLES, len(array)))
    means = np.mean(array[indices], axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def analyze(root: Path) -> dict[str, Any]:
    root = root.resolve()
    expected_seeds = list(DAMPING_ACCEPTANCE_SEEDS)
    rows: list[dict[str, Any]] = []
    protocol_hashes: list[str] = []
    replay_errors: list[float] = []
    integrity_failures: list[str] = []

    for penetration in PENETRATIONS:
        for demand in DEMANDS:
            directory = _cell_directory(root, demand, penetration)
            summary = _load(directory / "summary.json")
            protocol = summary["protocol"]
            _validate_protocol(protocol, demand=demand, penetration=penetration)
            progress = _load(directory / "progress.json")
            if int(progress.get("pending", -1)) != 0:
                raise ValueError(f"{directory} is incomplete")
            branch_files = sorted((directory / "branches").glob("*.json"))
            if int(progress.get("complete", -1)) != len(branch_files):
                raise ValueError(f"{directory} progress does not match branch files")
            if int(summary.get("branch_count", -1)) != len(branch_files):
                raise ValueError(f"{directory} summary does not match branch files")
            protocol_hashes.append(str(summary["protocol_hash"]))
            cases = {int(case["seed"]): case for case in summary.get("cases", ())}
            if len(cases) != len(summary.get("cases", ())):
                raise ValueError(f"{directory} contains duplicate case seeds")
            discoveries: dict[int, dict[str, Any]] = {}
            for seed in expected_seeds:
                discovery_path = directory / "discoveries" / f"m{demand}.seed{seed}.json"
                if not discovery_path.exists():
                    raise ValueError(f"{directory} lacks discovery seed {seed}")
                discoveries[seed] = _load(discovery_path)
            decision_seeds = {
                seed for seed, discovery in discoveries.items() if discovery.get("decision") is not None
            }
            if set(cases) != decision_seeds:
                raise ValueError(
                    f"{directory} discovery/case mismatch: decisions={sorted(decision_seeds)} "
                    f"cases={sorted(cases)}"
                )
            for seed in expected_seeds:
                base = {
                    "seed": seed,
                    "demand": demand,
                    "penetration": penetration,
                    "decision_found": seed in cases,
                    "physical_effect": 0.0,
                    "reward": 0.0,
                    "classification": "no_decision",
                    "constraints_pass": True,
                    "constraint_failures": [],
                    "replay_error": 0.0,
                }
                if discoveries[seed].get("decision") is None:
                    if any(
                        path.name.startswith(f"m{demand}.seed{seed}.")
                        for path in branch_files
                    ):
                        raise ValueError(
                            f"{directory} no-decision seed {seed} has branch artifacts"
                        )
                    rows.append(base)
                    continue
                case = cases[seed]
                if len(case.get("effects", ())) != 1:
                    raise ValueError(f"{case.get('case_id')} does not have one P30 effect")
                effect = case["effects"][0]
                case_id = str(case["case_id"])
                branches = directory / "branches"
                expected_paths = (
                    branches / f"{case_id}.abstain.json",
                    branches / f"{case_id}.abstain_replay.json",
                    branches / f"{case_id}.{effect['arm']}.json",
                )
                if not all(path.exists() for path in expected_paths):
                    raise ValueError(f"{case_id} lacks its exact three branches")
                if len([path for path in branch_files if path.name.startswith(case_id + ".")]) != 3:
                    raise ValueError(f"{case_id} does not contain exactly three branches")
                control, replay, treatment = (_load(path) for path in expected_paths)
                if any(
                    int(branch.get("completed_horizon_steps", -1)) != 60
                    for branch in (control, replay, treatment)
                ):
                    integrity_failures.append(f"{case_id}:incomplete_branch_horizon")
                replay_error = _replay_error(control, replay)
                replay_errors.append(replay_error)
                constraints = _constraints(control, treatment)
                constraints_pass = all(constraints.values())
                physical = float(effect["safety_composite_reference_units"])
                reward = _acceptance_reward(physical, constraints)
                classification = (
                    "harmful"
                    if not constraints_pass or physical < -DEADBAND
                    else "helpful"
                    if physical > DEADBAND
                    else "neutral"
                )
                if not bool(case.get("all_discovery_identity", False)):
                    integrity_failures.append(f"{case_id}:discovery_identity")
                if not bool(case.get("all_branch_decision_identity", False)):
                    integrity_failures.append(f"{case_id}:branch_identity")
                rows.append(
                    {
                        **base,
                        "case_id": case_id,
                        "physical_effect": physical,
                        "reward": reward,
                        "classification": classification,
                        "constraints_pass": constraints_pass,
                        "constraints": constraints,
                        "constraint_failures": [
                            name for name, passed in constraints.items() if not passed
                        ],
                        "replay_error": replay_error,
                        "target_command_steps": int(
                            treatment["target_command_step_count"]
                        ),
                    }
                )

    expected_cell_count = len(expected_seeds) * len(DEMANDS) * len(PENETRATIONS)
    if len(rows) != expected_cell_count:
        raise RuntimeError("prospective cell ledger is incomplete")
    activations = [row for row in rows if row["decision_found"]]
    classes = Counter(row["classification"] for row in activations)
    seed_reward = _seed_means(rows, "reward")
    seed_physical = _seed_means(rows, "physical_effect")
    mean_reward = float(np.mean(list(seed_reward.values())))
    mean_physical = float(np.mean(list(seed_physical.values())))
    ci_low, ci_high = _bootstrap_interval(list(seed_reward.values()))
    loo = {
        str(seed): float(
            np.mean([value for other, value in seed_reward.items() if other != seed])
        )
        for seed in expected_seeds
    }
    max_replay = max(replay_errors, default=0.0)
    qualified_nonneutral = [
        row
        for row in activations
        if row["constraints_pass"]
        and row["classification"] in {"helpful", "harmful"}
    ]
    activation_count = len(activations)
    helpful = int(classes["helpful"])
    harmful = int(classes["harmful"])
    neutral = int(classes["neutral"])
    harmful_fraction = harmful / activation_count if activation_count else 1.0
    helpful_neutral_fraction = (
        (helpful + neutral) / activation_count if activation_count else 0.0
    )
    all_constraints = all(row["constraints_pass"] for row in activations)
    constraint_names = sorted(
        {name for row in activations for name in row.get("constraints", {})}
    )
    constraint_pass = {
        name: all(row.get("constraints", {}).get(name, True) for row in activations)
        for name in constraint_names
    }
    gates = {
        "positive_mean_physical_effect": mean_physical > 0.0,
        "positive_mean_reward": mean_reward > 0.0,
        "helpful_more_frequent_than_harmful": helpful > harmful,
        "helpful_or_neutral_fraction_at_least_70pct": helpful_neutral_fraction
        >= 0.70,
        "harmful_fraction_at_most_25pct": harmful_fraction <= 0.25,
        "zero_added_collisions_and_teleports": constraint_pass.get(
            "zero_added_collisions", True
        )
        and constraint_pass.get("zero_added_teleports", True),
        "safety_noninferiority": constraint_pass.get("low_ttc_within_budget", True)
        and constraint_pass.get("immediate_follower_braking_noninferior", True)
        and constraint_pass.get("zero_treatment_safety_aborts", True),
        "service_noninferiority": constraint_pass.get("delay_within_budget", True)
        and constraint_pass.get("arrival_within_budget", True)
        and constraint_pass.get("discharge_within_budget", True),
        "dose_and_command_integrity": constraint_pass.get(
            "complete_horizon", True
        )
        and constraint_pass.get("dose_at_least_minimum", True)
        and constraint_pass.get("zero_unexpected_commands", True),
        "effects_exceed_replay_noise": max_replay <= 1e-12
        and all(
            abs(float(row["physical_effect"])) > DEADBAND
            for row in qualified_nonneutral
        ),
        "all_branch_and_discovery_identity": not integrity_failures,
        "leave_one_seed_out_mean_remains_positive": all(
            value > 0.0 for value in loo.values()
        ),
        "all_exact_constraints_pass": all_constraints,
        "all_requested_cells_reported": len(rows) == expected_cell_count,
    }

    return {
        "schema": SCHEMA,
        "protocol": "docs/IDM_P30_RULE_ARM_PROSPECTIVE_ACCEPTANCE_PROTOCOL.md",
        "evidence_role": "damping_acceptance",
        "grid_root": str(root),
        "seeds": expected_seeds,
        "demands": list(DEMANDS),
        "penetrations": list(PENETRATIONS),
        "protocol_hashes": protocol_hashes,
        "cell_count": len(rows),
        "activation_count": activation_count,
        "no_decision_count": len(rows) - activation_count,
        "classification_counts": {
            "helpful": helpful,
            "harmful": harmful,
            "neutral": neutral,
            "no_decision": len(rows) - activation_count,
        },
        "mean_policy_physical_effect": mean_physical,
        "mean_policy_reward": mean_reward,
        "activation_only_mean_physical_effect": float(
            np.mean([row["physical_effect"] for row in activations])
        )
        if activations
        else 0.0,
        "activation_only_mean_reward": float(
            np.mean([row["reward"] for row in activations])
        )
        if activations
        else 0.0,
        "helpful_or_neutral_fraction": helpful_neutral_fraction,
        "harmful_fraction": harmful_fraction,
        "seed_level_reward": {str(key): value for key, value in seed_reward.items()},
        "seed_level_physical_effect": {
            str(key): value for key, value in seed_physical.items()
        },
        "seed_cluster_bootstrap_95pct": {
            "low": ci_low,
            "high": ci_high,
            "resamples": BOOTSTRAP_RESAMPLES,
            "rng_seed": BOOTSTRAP_SEED,
        },
        "leave_one_seed_out_mean_reward": loo,
        "maximum_abstain_replay_error": max_replay,
        "constraint_pass": constraint_pass,
        "constraint_failure_count": sum(
            not row["constraints_pass"] for row in activations
        ),
        "integrity_failures": integrity_failures,
        "gates": gates,
        "acceptance_passes": all(gates.values()),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.grid_root)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output / "summary.json", result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "activation_count",
                    "no_decision_count",
                    "classification_counts",
                    "mean_policy_physical_effect",
                    "mean_policy_reward",
                    "harmful_fraction",
                    "gates",
                    "acceptance_passes",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
