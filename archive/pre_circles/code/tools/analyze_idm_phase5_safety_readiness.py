"""Apply the frozen Phase-5 safety-first IDM training-readiness gates."""

from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from tools.analyze_idm_phase5_reward_audit import correlation, ranks
from experiments.idm.run_idm_residual_checkpoint_eval import REWARD_COMPONENT_KEYS

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit", type=Path)
    parser.add_argument("--discrimination", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((args.audit / "cases").glob("*.json"))
    ]
    indexed = {(row["arm"], int(row["demand_mainline_vph"]), int(row["seed"])): row for row in rows}
    effects = []
    for (arm, demand, seed), row in indexed.items():
        if arm == "zero":
            continue
        zero = indexed[("zero", demand, seed)]
        wave0 = float(zero["trajectory"]["wave"]["oscillation_energy_m2ps3"])
        brake0 = float(zero["trajectory"]["safety"]["severe_braking_events"])
        wave_reduction = (
            (wave0 - float(row["trajectory"]["wave"]["oscillation_energy_m2ps3"])) / wave0
            if wave0
            else 0.0
        )
        brake_reduction = (
            (brake0 - float(row["trajectory"]["safety"]["severe_braking_events"])) / brake0
            if brake0
            else 0.0
        )
        delay_delta = float(row["metrics"]["mean_time_loss"]) - float(
            zero["metrics"]["mean_time_loss"]
        )
        arrived_delta = int(row["metrics"]["arrived"]) - int(zero["metrics"]["arrived"])
        collision_delta = int(row["metrics"]["collisions"]) - int(zero["metrics"]["collisions"])
        utility = (
            0.5 * (wave_reduction + brake_reduction)
            - 0.01 * max(delay_delta, 0.0)
            - 0.25 * max(-arrived_delta - 1, 0)
            - max(collision_delta, 0)
        )
        reward_delta = float(row["learning_reward"]["mean_per_selected_agent_step"]) - float(
            zero["learning_reward"]["mean_per_selected_agent_step"]
        )
        effects.append(
            {
                "arm": arm,
                "demand": demand,
                "seed": seed,
                "wave_reduction": wave_reduction,
                "severe_braking_reduction": brake_reduction,
                "delay_delta_s_per_vehicle": delay_delta,
                "arrived_delta": arrived_delta,
                "collision_delta": collision_delta,
                "physical_utility": utility,
                "reward_delta": reward_delta,
            }
        )

    grouped = {}
    for arm in ("n010", "p010", "p020"):
        for demand in (2200, 2700):
            group = [row for row in effects if row["arm"] == arm and row["demand"] == demand]
            grouped[f"{arm}_m{demand}"] = {
                "mean_physical_utility": mean(row["physical_utility"] for row in group),
                "mean_reward_delta": mean(row["reward_delta"] for row in group),
                "mean_wave_reduction": mean(row["wave_reduction"] for row in group),
                "mean_severe_braking_reduction": mean(
                    row["severe_braking_reduction"] for row in group
                ),
                "mean_delay_delta_s_per_vehicle": mean(
                    row["delay_delta_s_per_vehicle"] for row in group
                ),
                "positive_physical_seed_count": sum(row["physical_utility"] > 0 for row in group),
                "positive_reward_seed_count": sum(row["reward_delta"] > 0 for row in group),
            }

    harmful_2700 = [
        row
        for row in effects
        if row["demand"] == 2700
        and row["wave_reduction"] < 0
        and row["severe_braking_reduction"] < 0
    ]
    harmful_aligned = [row for row in harmful_2700 if row["reward_delta"] <= 0]
    absolute = {key: 0.0 for key in REWARD_COMPONENT_KEYS}
    selected_total = 0
    maximum_error = 0.0
    missing_diagnostics = 0
    for row in rows:
        learning = row.get("learning_reward", {})
        selected = int(learning.get("selected_agent_steps", 0))
        selected_total += selected
        maximum_error = max(
            maximum_error, float(learning.get("max_step_accounting_error", float("inf")))
        )
        component_abs = learning.get("component_mean_absolute")
        if not component_abs:
            missing_diagnostics += 1
            continue
        for key in REWARD_COMPONENT_KEYS:
            absolute[key] += float(component_abs[key]) * selected
    absolute_sum = sum(absolute.values())
    shares = {key: value / absolute_sum if absolute_sum else 0.0 for key, value in absolute.items()}

    reward_winners = []
    for demand in (2200, 2700):
        for seed in range(4107, 4113):
            candidates = [indexed[(arm, demand, seed)] for arm in ("zero", "n010", "p010", "p020")]
            reward_winners.append(
                max(
                    candidates,
                    key=lambda row: float(row["learning_reward"]["mean_per_selected_agent_step"]),
                )["arm"]
            )

    discrimination_summary = json.loads(
        (args.discrimination / "summary.json").read_text(encoding="utf-8")
    )
    discrimination = {int(row["seed"]): row for row in discrimination_summary["paired_effects"]}
    p010_2200 = grouped["p010_m2200"]
    preservation = all(
        grouped[f"{arm}_m2700"]["mean_physical_utility"] <= 0
        and grouped[f"{arm}_m2700"]["mean_reward_delta"] <= 0
        for arm in ("n010", "p010", "p020")
    )
    rho = correlation(
        ranks([row["physical_utility"] for row in effects]),
        ranks([row["reward_delta"] for row in effects]),
    )
    gates = {
        "complete_48_cases": len(rows) == 48 and len(effects) == 36,
        "diagnostics_complete": missing_diagnostics == 0,
        "reward_accounting_exact": maximum_error <= 1e-12,
        "no_added_collisions": all(row["collision_delta"] <= 0 for row in effects),
        "no_material_arrival_loss": all(row["arrived_delta"] >= -1 for row in effects),
        "p010_2200_actionable_headroom": p010_2200["mean_physical_utility"] > 0
        and p010_2200["mean_reward_delta"] > 0
        and p010_2200["positive_physical_seed_count"] >= 3,
        "2700_zero_preservation": preservation,
        "harmful_2700_directionality_80pct": len(harmful_aligned) / len(harmful_2700) >= 0.80
        if harmful_2700
        else False,
        "reward_physical_spearman_at_least_0p50": rho >= 0.50,
        "no_component_over_80pct": max(shares.values()) <= 0.80,
        "reward_action_diversity": len(set(reward_winners)) >= 2,
        "locked_seed4108_negative": float(discrimination[4108]["learning_reward_delta"]) < 0,
        "locked_seed4112_positive": float(discrimination[4112]["learning_reward_delta"]) > 0,
    }
    result = {
        "schema": "idm_phase5_safety_readiness/1",
        "audit": str(args.audit.resolve()),
        "discrimination": str(args.discrimination.resolve()),
        "case_count": len(rows),
        "paired_effect_count": len(effects),
        "spearman_reward_vs_safety_first_utility": rho,
        "harmful_2700_case_count": len(harmful_2700),
        "harmful_2700_aligned_count": len(harmful_aligned),
        "component_absolute_shares": shares,
        "dominant_component": max(shares, key=shares.get),
        "dominant_component_share": max(shares.values()),
        "reward_winner_counts": dict(Counter(reward_winners)),
        "by_arm_demand": grouped,
        "gates": gates,
        "scientific_readiness_pass": all(gates.values()),
    }
    output = args.audit / "safety_readiness.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
