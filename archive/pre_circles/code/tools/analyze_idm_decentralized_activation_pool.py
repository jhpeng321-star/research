"""Pool the two frozen development landmarks for the IDM activation gate."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.io import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summaries = []
    records = []
    observation_lengths = []
    observation_hashes = set()
    for directory in args.inputs:
        root = directory.resolve()
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        summaries.append(summary)
        for case in summary["cases"]:
            seed, step = int(case["seed"]), int(case["decision_step"])
            discovery = json.loads(
                (root / "discoveries" / f"m{summary['protocol']['demand']}.seed{seed}.json")
                .read_text(encoding="utf-8")
            )
            decision = discovery["decision"]
            if decision is None or int(decision["decision_step"]) != step:
                raise RuntimeError(f"missing matching discovery for seed {seed} step {step}")
            vehicle_id = decision["candidates"][0]["vehicle_id"]
            observation = decision["candidate_observations"][vehicle_id]
            observation_lengths.append(len(observation))
            observation_hashes.add(json.dumps(observation, separators=(",", ":")))

            branches = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in (root / "branches").glob(f"{case['case_id']}.*.json")
            ]
            control = next(row for row in branches if row["arm"] == "abstain")
            action = next(row for row in branches if not row["is_control"])
            effect = case["effects"][0]
            records.append(
                {
                    "seed": seed,
                    "decision_step": step,
                    "decision_index": int(decision.get("decision_index", 1)),
                    "classification": effect["classification"],
                    "effect": float(effect["safety_composite_reference_units"]),
                    "constraints_pass": bool(effect["constraints_pass"]),
                    "dose_steps": int(effect["target_command_step_count"]),
                    "reward_delta": float(action["role_reward_sum"])
                    - float(control["role_reward_sum"]),
                    "all_agent_reward_delta": float(action["all_agent_reward_sum"])
                    - float(control["all_agent_reward_sum"]),
                }
            )

    invariant_keys = (
        "schema",
        "demand",
        "action_set",
        "candidate_slots",
        "candidate_count",
        "option_steps",
        "minimum_dose_steps",
        "safety_horizon_steps",
        "service_horizon_steps",
        "landmark",
        "landmark_contract",
        "cav_penetration",
        "candidate_ranking",
    )
    reference = summaries[0]["protocol"]
    protocol_identity = all(
        all(summary["protocol"].get(key) == reference.get(key) for key in invariant_keys)
        for summary in summaries[1:]
    )
    by_seed = defaultdict(set)
    for row in records:
        by_seed[row["seed"]].add(row["classification"])
    helpful_seeds = sorted(
        seed for seed, values in by_seed.items() if "helpful" in values
    )
    nonbeneficial_seeds = sorted(
        seed for seed, values in by_seed.items() if values - {"helpful"}
    )
    within_seed_diversity = sorted(seed for seed, values in by_seed.items() if len(values) > 1)
    nonneutral = [row for row in records if row["classification"] != "neutral"]
    reward_sign_matches = [
        (row["classification"] == "helpful" and row["reward_delta"] > 0.0)
        or (row["classification"] == "harmful" and row["reward_delta"] < 0.0)
        for row in nonneutral
    ]
    neutral_reward_bounded = all(
        abs(row["reward_delta"]) <= 5.0
        for row in records
        if row["classification"] == "neutral"
    )
    effects = np.asarray([row["effect"] for row in records], dtype=float)
    reward_deltas = np.asarray([row["reward_delta"] for row in records], dtype=float)
    reward_effect_correlation = float(np.corrcoef(effects, reward_deltas)[0, 1])
    shared_integrity = all(
        all(
            value
            for key, value in summary["pilot_gates"].items()
            if key
            in {
                "abstain_replay_exact",
                "all_branch_decision_identity",
                "all_discovery_identity",
                "all_horizons_complete",
                "control_issues_zero_commands",
                "zero_collisions",
                "zero_teleports",
            }
        )
        for summary in summaries
    )
    response_fraction = float(np.mean(np.abs(effects) > float(summaries[0]["frozen_deadband"])))
    gates = {
        "protocol_identity": protocol_identity,
        "paired_landmarks_at_least_8": len(records) >= 8,
        "seed_coverage_at_least_5": len(by_seed) >= 5,
        "shared_integrity": shared_integrity,
        "all_treatments_full_dose": all(
            row["dose_steps"] >= int(reference["minimum_dose_steps"]) for row in records
        ),
        "all_treatments_safe_and_service_admissible": all(
            row["constraints_pass"] for row in records
        ),
        "response_fraction_at_least_30pct": response_fraction >= 0.30,
        "helpful_in_at_least_3_seeds": len(helpful_seeds) >= 3,
        "nonbeneficial_in_at_least_3_seeds": len(nonbeneficial_seeds) >= 3,
        "within_seed_outcome_diversity": bool(within_seed_diversity),
        "observation_dimension_is_136": bool(observation_lengths)
        and set(observation_lengths) == {136},
        "decision_observations_are_not_duplicates": len(observation_hashes) == len(records),
        "reward_sign_matches_all_nonneutral_effects": all(reward_sign_matches),
        "neutral_reward_deltas_are_bounded": neutral_reward_bounded,
        "reward_effect_correlation_at_least_0_8": reward_effect_correlation >= 0.80,
    }
    payload = {
        "schema": "idm_decentralized_activation_pool/1",
        "development_only": True,
        "inputs": [str(path.resolve()) for path in args.inputs],
        "case_count": len(records),
        "seed_count": len(by_seed),
        "classification_counts": dict(Counter(row["classification"] for row in records)),
        "helpful_seeds": helpful_seeds,
        "nonbeneficial_seeds": nonbeneficial_seeds,
        "within_seed_diversity": within_seed_diversity,
        "response_fraction": response_fraction,
        "reward_effect_correlation": reward_effect_correlation,
        "gates": gates,
        "training_authorized": all(gates.values()),
        "records": sorted(records, key=lambda row: (row["seed"], row["decision_step"])),
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "summary.json", payload)
    report = [
        "# IDM Decentralized Activation Pretraining Gate",
        "",
        "Status: **PASS**" if payload["training_authorized"] else "Status: **FAIL**",
        "",
        "Development-only causal authority evidence; this does not establish independent-seed performance.",
        "",
        f"- Paired landmarks: {len(records)} across {len(by_seed)} seeds",
        f"- Outcomes: {payload['classification_counts']}",
        f"- Helpful seeds: {helpful_seeds}",
        f"- Nonbeneficial seeds: {nonbeneficial_seeds}",
        f"- Reward/effect correlation: {reward_effect_correlation:.3f}",
        "",
        "## Gates",
        "",
        *[f"- {'PASS' if value else 'FAIL'} — `{key}`" for key, value in gates.items()],
        "",
        "Passing authorizes only a small resumable PPO smoke. It does not prove the learned policy works.",
    ]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
