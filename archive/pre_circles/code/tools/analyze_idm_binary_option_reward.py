"""Verify the frozen paired binary IDM activation reward on stored branches."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from experiments.io import atomic_write_json
from idm.option_reward import (
    IDM_BINARY_OPTION_REWARD_V1,
    binary_option_reward,
    paired_physical_utility,
)


EXPECTED_ACTION_SET = "activation_speed_ceiling_v1"
EXPECTED_CELLS = {(demand, recurrence) for demand in (1900, 2000, 2100) for recurrence in (1, 2, 3)}
EXPECTED_SEEDS = set(range(4121, 4133))
EXPECTED_REFS_SHA256 = "737b9ce4bb1945f7681b807941c86509ae3143128bd0729d2cb2aee5237c5b58"
FORBIDDEN_API_PARAMETERS = {
    "selected_cav_lifetime",
    "agent_count",
    "queue_area",
    "seed",
    "decision_step",
    "recurrence",
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        value = 0.5 * (cursor + 1 + end)
        for index in order[cursor:end]:
            result[index] = value
        cursor = end
    return result


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return math.nan
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator > 0.0 else math.nan


def _reference_by_demand(payload: Mapping[str, Any]) -> dict[int, dict[str, float]]:
    result = {}
    for row in payload["refs"].values():
        demand = int(row["demand_mainline_vph"])
        result[demand] = {
            "severe": float(row["corridor_severe_braking_count_ref_per_s"]),
            "energy": float(row["corridor_acceleration_energy_sum_ref_m2ps4"]),
        }
    return result


def _case_branch(root: Path, case_id: str, arm: str) -> dict[str, Any]:
    path = root / "branches" / f"{case_id}.{arm}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return _read(path)


def _effect_deltas(computed: Mapping[str, Any], stored: Mapping[str, Any]) -> dict[str, float]:
    return {
        "severe": abs(
            float(computed["severe_braking_effect_reference_units"])
            - float(stored["severe_braking_effect_reference_units"])
        ),
        "energy": abs(
            float(computed["energy_effect_reference_units"])
            - float(stored["wave_effect_reference_units"])
        ),
        "composite": abs(
            float(computed["physical_utility"])
            - float(stored["safety_composite_reference_units"])
        ),
    }


def analyze(inputs: Sequence[Path], refs_path: Path) -> dict[str, Any]:
    refs_path = refs_path.resolve()
    refs_sha256 = _sha256(refs_path)
    refs_payload = _read(refs_path)
    references = _reference_by_demand(refs_payload)
    records: list[dict[str, Any]] = []
    source_rows = []
    replay_rewards = []
    observed_cells = set()
    observed_seeds = set()
    retained_summary_case_count = 0

    for input_path in inputs:
        root = input_path.resolve()
        summary = _read(root / "summary.json")
        protocol = dict(summary["protocol"])
        demand = int(protocol["demand"])
        recurrence = int(protocol["decision_index"])
        observed_cells.add((demand, recurrence))
        source_identity = {
            "root": str(root),
            "action_set_exact": protocol.get("action_set") == EXPECTED_ACTION_SET,
            "option_steps_exact": int(protocol.get("option_steps", -1))
            == IDM_BINARY_OPTION_REWARD_V1.option_steps,
            "safety_horizon_exact": int(protocol.get("safety_horizon_steps", -1))
            == IDM_BINARY_OPTION_REWARD_V1.horizon_steps,
            "service_horizon_exact": int(protocol.get("service_horizon_steps", -1))
            == IDM_BINARY_OPTION_REWARD_V1.horizon_steps,
            "reward_reference_exact": protocol.get("reward_refs_sha256")
            == EXPECTED_REFS_SHA256,
            "all_branch_decision_identity": all(
                bool(case.get("all_branch_decision_identity"))
                for case in summary["cases"]
            ),
            "all_discovery_identity": all(
                bool(case.get("all_discovery_identity"))
                for case in summary["cases"]
            ),
            "case_count": len(summary["cases"]),
            "demand": demand,
            "recurrence": recurrence,
        }
        source_rows.append(source_identity)
        retained_summary_case_count += len(summary["cases"])
        refs = references[demand]

        for case in summary["cases"]:
            case_id = str(case["case_id"])
            seed = int(case["seed"])
            observed_seeds.add(seed)
            control = _case_branch(root, case_id, "abstain")
            replay = _case_branch(root, case_id, "abstain_replay")
            replay_physical = paired_physical_utility(
                control,
                replay,
                severe_reference_per_step=refs["severe"],
                energy_reference_per_step=refs["energy"],
            )
            replay_rewards.append(math.tanh(replay_physical["physical_utility"]))
            for stored in case["effects"]:
                action = _case_branch(root, case_id, str(stored["arm"]))
                computed = binary_option_reward(
                    control,
                    action,
                    severe_reference_per_step=refs["severe"],
                    energy_reference_per_step=refs["energy"],
                )
                effect_deltas = _effect_deltas(computed, stored)
                constraints_exact = computed["constraints"] == stored["constraints"]
                reward = float(computed["activate_reward"])
                records.append(
                    {
                        "case_id": case_id,
                        "demand": demand,
                        "recurrence": recurrence,
                        "seed": seed,
                        "arm": str(stored["arm"]),
                        "stored_classification": str(stored["classification"]),
                        "stored_constraints_pass": bool(stored["constraints_pass"]),
                        "stored_effect": float(
                            stored["safety_composite_reference_units"]
                        ),
                        "computed": computed,
                        "effect_absolute_deltas": effect_deltas,
                        "constraints_exact": constraints_exact,
                        "reward_finite": math.isfinite(reward),
                        "reward_in_range": (
                            IDM_BINARY_OPTION_REWARD_V1.hard_failure_reward
                            <= reward
                            < 1.0
                        ),
                    }
                )

    qualified = [row for row in records if row["computed"]["constraints_pass"]]
    failed = [row for row in records if not row["computed"]["constraints_pass"]]
    effects = [float(row["stored_effect"]) for row in qualified]
    rewards = [float(row["computed"]["activate_reward"]) for row in qualified]
    nonneutral = [
        row
        for row in qualified
        if abs(float(row["stored_effect"]))
        > IDM_BINARY_OPTION_REWARD_V1.deadband + 1e-12
    ]
    neutral = [
        row
        for row in qualified
        if abs(float(row["stored_effect"]))
        <= IDM_BINARY_OPTION_REWARD_V1.deadband + 1e-12
    ]
    substantial = [
        row
        for row in qualified
        if abs(float(row["stored_effect"])) + 1e-12
        >= IDM_BINARY_OPTION_REWARD_V1.substantial_effect
    ]
    sign_match = lambda row: (
        float(row["stored_effect"])
        * float(row["computed"]["activate_reward"])
        > 0.0
    )
    severe_mass = sum(
        abs(float(row["computed"]["severe_contribution"])) for row in qualified
    )
    energy_mass = sum(
        abs(float(row["computed"]["energy_contribution"])) for row in qualified
    )
    component_mass = severe_mass + energy_mass
    component_shares = {
        "severe_braking": severe_mass / component_mass if component_mass else 0.0,
        "acceleration_energy": energy_mass / component_mass if component_mass else 0.0,
    }
    max_effect_delta = max(
        (
            max(row["effect_absolute_deltas"].values())
            for row in records
        ),
        default=math.inf,
    )
    pearson = _correlation(effects, rewards)
    spearman = _correlation(_rank(effects), _rank(rewards))
    neutral_max = max(
        (abs(float(row["computed"]["activate_reward"])) for row in neutral),
        default=0.0,
    )
    api_parameters = set(inspect.signature(binary_option_reward).parameters)
    source_identity_pass = bool(source_rows) and all(
        all(
            bool(row[key])
            for key in (
                "action_set_exact",
                "option_steps_exact",
                "safety_horizon_exact",
                "service_horizon_exact",
                "reward_reference_exact",
                "all_branch_decision_identity",
                "all_discovery_identity",
            )
        )
        for row in source_rows
    )
    gates = {
        "exact_nine_development_cells": observed_cells == EXPECTED_CELLS
        and len(source_rows) == 9,
        "exact_twelve_development_seeds": observed_seeds == EXPECTED_SEEDS,
        "reference_file_identity": refs_sha256 == EXPECTED_REFS_SHA256,
        "all_source_protocol_and_case_identity": source_identity_pass,
        "every_summary_case_retained": len(records) == retained_summary_case_count,
        "paired_constraints_reproduce_stored_contract": all(
            row["constraints_exact"] for row in records
        ),
        "paired_effect_components_reproduce_stored_contract": max_effect_delta
        <= 1e-12,
        "abstain_replay_reward_exact": max(
            (abs(value) for value in replay_rewards), default=math.inf
        )
        <= 1e-12,
        "qualified_helpful_and_harmful_present": (
            any(float(row["stored_effect"]) > IDM_BINARY_OPTION_REWARD_V1.deadband for row in qualified)
            and any(float(row["stored_effect"]) < -IDM_BINARY_OPTION_REWARD_V1.deadband for row in qualified)
        ),
        "all_qualified_nonneutral_signs_agree": bool(nonneutral)
        and all(sign_match(row) for row in nonneutral),
        "all_substantial_pairs_ordered": bool(substantial)
        and all(sign_match(row) for row in substantial),
        "neutral_reward_within_deadband": neutral_max
        <= IDM_BINARY_OPTION_REWARD_V1.deadband + 1e-12,
        "pearson_at_least_0p95": math.isfinite(pearson) and pearson >= 0.95,
        "spearman_at_least_0p99": math.isfinite(spearman) and spearman >= 0.99,
        "component_shares_at_most_80pct": bool(component_shares)
        and max(component_shares.values()) <= 0.80,
        "actual_constraint_failures_receive_minus_two": bool(failed)
        and all(
            float(row["computed"]["activate_reward"])
            == IDM_BINARY_OPTION_REWARD_V1.hard_failure_reward
            for row in failed
        ),
        "hard_failures_rank_below_all_qualified": bool(failed)
        and bool(qualified)
        and max(
            float(row["computed"]["activate_reward"]) for row in failed
        )
        < min(float(row["computed"]["activate_reward"]) for row in qualified),
        "all_rewards_finite_and_bounded": all(
            row["reward_finite"] and row["reward_in_range"] for row in records
        ),
        "reward_api_excludes_forbidden_credit_inputs": api_parameters.isdisjoint(
            FORBIDDEN_API_PARAMETERS
        ),
    }
    valid = all(gates.values())
    return {
        "schema": "idm_binary_option_reward_audit/1",
        "development_only": True,
        "reward_contract": IDM_BINARY_OPTION_REWARD_V1.to_dict(),
        "inputs": [str(path.resolve()) for path in inputs],
        "reference_path": str(refs_path),
        "reference_sha256": refs_sha256,
        "source_rows": source_rows,
        "record_count": len(records),
        "qualified_count": len(qualified),
        "constraint_failed_count": len(failed),
        "qualified_classification_counts": dict(
            Counter(row["stored_classification"] for row in qualified)
        ),
        "qualified_nonneutral_count": len(nonneutral),
        "substantial_pair_count": len(substantial),
        "maximum_effect_reproduction_error": max_effect_delta,
        "maximum_absolute_replay_reward": max(
            (abs(value) for value in replay_rewards), default=None
        ),
        "maximum_absolute_neutral_reward": neutral_max,
        "pearson_reward_vs_effect": pearson,
        "spearman_reward_vs_effect": spearman,
        "component_absolute_mass_shares": component_shares,
        "gates": gates,
        "valid_paired_binary_option_reward": valid,
        "paired_option_wrapper_authorized": valid,
        "ordinary_multiagent_reward_authorized": False,
        "rl_training_authorized": False,
        "records": records,
    }


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    failed = [name for name, value in summary["gates"].items() if not value]
    lines = [
        "# Binary option-reward audit",
        "",
        f"- Valid paired reward: `{summary['valid_paired_binary_option_reward']}`",
        f"- Records: {summary['record_count']} ({summary['qualified_count']} qualified, "
        f"{summary['constraint_failed_count']} constraint-failed)",
        f"- Pearson reward/effect: {summary['pearson_reward_vs_effect']:.6f}",
        f"- Spearman reward/effect: {summary['spearman_reward_vs_effect']:.6f}",
        f"- Maximum neutral |reward|: {summary['maximum_absolute_neutral_reward']:.6f}",
        f"- Component shares: {summary['component_absolute_mass_shares']}",
        f"- Failed gates: {failed or 'none'}",
        "- Ordinary multi-agent reward authorized: `False`",
        "- RL training authorized: `False`",
        "",
    ]
    return "\n".join(lines)


def write_outputs(summary: Mapping[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    records = list(summary["records"])
    compact = {key: value for key, value in summary.items() if key != "records"}
    atomic_write_json(output / "summary.json", compact)
    (output / "records.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    (output / "summary.md").write_text(
        _summary_markdown(compact), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--refs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(args.inputs, args.refs)
    write_outputs(summary, args.output.resolve())
    compact = {key: value for key, value in summary.items() if key != "records"}
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
