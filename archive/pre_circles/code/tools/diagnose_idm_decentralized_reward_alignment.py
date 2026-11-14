"""Reweight stored paired reward components without rerunning SUMO.

This development diagnostic operates only on counterfactual branches generated
by ``run_idm_decentralized_authority_verification``.  It identifies whether
online cooperative reward terms agree with the frozen physical effect label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.io import atomic_write_json


CURRENT_WEIGHTS = {
    "efficiency": 0.10,
    "delay_budget": 0.10,
    "corridor_ttc": 1.0,
    "corridor_severe_braking": 1.0,
    "corridor_wave_energy": 1.0,
}

CANDIDATES = {
    "current": dict(CURRENT_WEIGHTS),
    "safety_only": {
        "efficiency": 0.0,
        "delay_budget": 0.0,
        "corridor_ttc": 0.0,
        "corridor_severe_braking": 1.0,
        "corridor_wave_energy": 1.0,
    },
    "safety_ttc_0p10": {
        "efficiency": 0.0,
        "delay_budget": 0.0,
        "corridor_ttc": 0.10,
        "corridor_severe_braking": 1.0,
        "corridor_wave_energy": 1.0,
    },
    "safety_ttc_service_small": {
        "efficiency": 0.02,
        "delay_budget": 0.02,
        "corridor_ttc": 0.10,
        "corridor_severe_braking": 1.0,
        "corridor_wave_energy": 1.0,
    },
}


def _raw_component_delta(control: dict, action: dict, name: str) -> float:
    weighted = float(action["role_reward_component_sums"].get(name, 0.0)) - float(
        control["role_reward_component_sums"].get(name, 0.0)
    )
    original_weight = CURRENT_WEIGHTS[name]
    return weighted / original_weight if original_weight else 0.0


def load_records(inputs: list[Path]) -> list[dict]:
    records = []
    for directory in inputs:
        root = directory.resolve()
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        demand = int(summary["protocol"]["demand"])
        minimum_dose = int(summary["protocol"]["minimum_dose_steps"])
        for case in summary["cases"]:
            branches = {
                row["arm"]: row
                for path in (root / "branches").glob(f"{case['case_id']}.*.json")
                for row in [json.loads(path.read_text(encoding="utf-8"))]
            }
            control = branches["abstain"]
            for effect in case["effects"]:
                action = next(
                    row
                    for row in branches.values()
                    if not row["is_control"]
                    and int(row["candidate_slot"]) == int(effect["candidate_slot"])
                    and int(row["mode_index"]) == int(effect["mode_index"])
                )
                constraints = dict(effect.get("constraints") or {})
                admissible_except_dose = all(
                    value
                    for key, value in constraints.items()
                    if key != "dose_at_least_minimum"
                )
                qualified = (
                    int(effect["target_command_step_count"]) >= minimum_dose
                    and admissible_except_dose
                )
                raw = {
                    name: _raw_component_delta(control, action, name)
                    for name in CURRENT_WEIGHTS
                }
                records.append(
                    {
                        "demand": demand,
                        "seed": int(case["seed"]),
                        "decision_index": int(case.get("decision_index", 1)),
                        "classification": str(effect["classification"]),
                        "effect": float(effect["safety_composite_reference_units"]),
                        "qualified": qualified,
                        "raw_component_deltas": raw,
                        "trajectory_return_deltas": {
                            "selected_role": float(action["role_reward_sum"])
                            - float(control["role_reward_sum"]),
                            "selected_role_safety_horizon": float(
                                action["role_reward_safety_horizon_sum"]
                            )
                            - float(control["role_reward_safety_horizon_sum"]),
                            "all_active_agents": float(action["all_agent_reward_sum"])
                            - float(control["all_agent_reward_sum"]),
                        },
                    }
                )
    return records


def _metrics(records: list[dict], weights: dict[str, float]) -> dict:
    values = []
    for row in records:
        reward = sum(
            weights[name] * row["raw_component_deltas"][name]
            for name in CURRENT_WEIGHTS
        )
        values.append((row, float(reward)))
    nonneutral = [item for item in values if item[0]["classification"] != "neutral"]
    neutral = [abs(reward) for row, reward in values if row["classification"] == "neutral"]
    effects = np.asarray([row["effect"] for row, _reward in values], dtype=float)
    rewards = np.asarray([reward for _row, reward in values], dtype=float)
    sign_matches = [
        (row["classification"] == "helpful" and reward > 0.0)
        or (row["classification"] == "harmful" and reward < 0.0)
        for row, reward in nonneutral
    ]
    return {
        "case_count": len(values),
        "nonneutral_count": len(nonneutral),
        "reward_effect_correlation": (
            float(np.corrcoef(effects, rewards)[0, 1]) if len(values) >= 2 else float("nan")
        ),
        "nonneutral_sign_match_fraction": (
            float(np.mean(sign_matches)) if sign_matches else float("nan")
        ),
        "neutral_max_absolute_reward_delta": max(neutral, default=0.0),
        "neutral_p90_absolute_reward_delta": (
            float(np.percentile(neutral, 90.0)) if neutral else 0.0
        ),
    }


def _trajectory_metrics(records: list[dict], name: str) -> dict:
    values = [
        (row, float(row["trajectory_return_deltas"][name])) for row in records
    ]
    nonneutral = [item for item in values if item[0]["classification"] != "neutral"]
    neutral = [abs(reward) for row, reward in values if row["classification"] == "neutral"]
    effects = np.asarray([row["effect"] for row, _reward in values], dtype=float)
    rewards = np.asarray([reward for _row, reward in values], dtype=float)
    matches = [
        (row["classification"] == "helpful" and reward > 0.0)
        or (row["classification"] == "harmful" and reward < 0.0)
        for row, reward in nonneutral
    ]
    return {
        "case_count": len(values),
        "nonneutral_count": len(nonneutral),
        "reward_effect_correlation": (
            float(np.corrcoef(effects, rewards)[0, 1]) if len(values) >= 2 else float("nan")
        ),
        "nonneutral_sign_match_fraction": (
            float(np.mean(matches)) if matches else float("nan")
        ),
        "neutral_max_absolute_reward_delta": max(neutral, default=0.0),
        "neutral_p90_absolute_reward_delta": (
            float(np.percentile(neutral, 90.0)) if neutral else 0.0
        ),
    }


def diagnose(inputs: list[Path]) -> dict:
    records = [row for row in load_records(inputs) if row["qualified"]]
    for row in records:
        row["candidate_reward_deltas"] = {
            name: float(
                sum(
                    weights[key] * row["raw_component_deltas"][key]
                    for key in CURRENT_WEIGHTS
                )
            )
            for name, weights in CANDIDATES.items()
        }
        row["full_horizon_safety_reward_delta_expected"] = float(
            2.0 * 60.0 * row["effect"]
        )
    return {
        "schema": "idm_decentralized_reward_alignment_diagnosis/1",
        "development_only": True,
        "inputs": [str(path.resolve()) for path in inputs],
        "qualified_case_count": len(records),
        "candidates": {
            name: {"weights": weights, "metrics": _metrics(records, weights)}
            for name, weights in CANDIDATES.items()
        },
        "trajectory_return_candidates": {
            name: _trajectory_metrics(records, name)
            for name in (
                "selected_role",
                "selected_role_safety_horizon",
                "all_active_agents",
            )
        },
        "records": records,
        "interpretation": (
            "Offline reweighting is a screening diagnostic. A selected reward must be "
            "implemented, unit-tested and reverified on paired branches before training."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = diagnose(args.inputs)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
