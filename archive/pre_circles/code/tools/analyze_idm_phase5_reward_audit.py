"""Apply the frozen IDM Phase-5 readiness gates to a fixed-action audit."""

from __future__ import annotations
import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean

COMPONENTS = (
    "local",
    "efficiency",
    "delay_budget",
    "corridor_ttc",
    "corridor_severe_braking",
    "corridor_wave_energy",
    "wave",
    "residual_effort",
)


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for index in order[cursor:end]:
            result[index] = rank
        cursor = end
    return result


def correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return math.nan
    lmean, rmean = mean(left), mean(right)
    numerator = sum((x - lmean) * (y - rmean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - lmean) ** 2 for x in left) * sum((y - rmean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit", type=Path)
    args = parser.parse_args()
    case_paths = sorted((args.audit / "cases").glob("*.json"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in case_paths]
    indexed = {(row["arm"], int(row["demand_mainline_vph"]), int(row["seed"])): row for row in rows}
    effects = []
    for (arm, demand, seed), row in indexed.items():
        if arm == "zero":
            continue
        zero = indexed[("zero", demand, seed)]
        wave0 = float(zero["trajectory"]["wave"]["oscillation_energy_m2ps3"])
        brake0 = float(zero["trajectory"]["safety"]["severe_braking_events"])
        wave = float(row["trajectory"]["wave"]["oscillation_energy_m2ps3"])
        brake = float(row["trajectory"]["safety"]["severe_braking_events"])
        delay_delta = float(row["metrics"]["mean_time_loss"]) - float(
            zero["metrics"]["mean_time_loss"]
        )
        arrived_delta = int(row["metrics"]["arrived"]) - int(zero["metrics"]["arrived"])
        collision_delta = int(row["metrics"]["collisions"]) - int(zero["metrics"]["collisions"])
        wave_reduction = (wave0 - wave) / wave0 if wave0 else 0.0
        brake_reduction = (brake0 - brake) / brake0 if brake0 else 0.0
        physical_utility = (
            0.5 * (wave_reduction + brake_reduction)
            - 0.05 * max(delay_delta, 0.0)
            - 0.25 * max(-arrived_delta - 1, 0)
            - 1.0 * max(collision_delta, 0)
        )
        reward_delta = float(row["learning_reward"]["mean_per_selected_agent_step"]) - float(
            zero["learning_reward"]["mean_per_selected_agent_step"]
        )
        component_deltas = {
            key: float(row["learning_reward"]["component_means"][key])
            - float(zero["learning_reward"]["component_means"][key])
            for key in COMPONENTS
        }
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
                "physical_utility": physical_utility,
                "reward_delta": reward_delta,
                "component_deltas": component_deltas,
                "executed_residual_mean_mps2": float(row["residual"]["mean_mps2"]),
            }
        )

    rho = correlation(
        ranks([row["physical_utility"] for row in effects]),
        ranks([row["reward_delta"] for row in effects]),
    )
    beneficial = [row for row in effects if row["physical_utility"] > 0.0]
    effort_reversals = [
        row
        for row in beneficial
        if row["reward_delta"] <= 0.0
        and row["reward_delta"] - row["component_deltas"]["residual_effort"] > 0.0
    ]

    absolute_totals = {key: 0.0 for key in COMPONENTS}
    selected_total = 0
    max_accounting_error = 0.0
    missing_diagnostics = 0
    for row in rows:
        reward = row.get("learning_reward", {})
        selected = int(reward.get("selected_agent_steps", 0))
        selected_total += selected
        if not reward or not reward.get("component_mean_absolute"):
            missing_diagnostics += 1
            continue
        max_accounting_error = max(max_accounting_error, float(reward["max_step_accounting_error"]))
        for key in COMPONENTS:
            absolute_totals[key] += float(reward["component_mean_absolute"][key]) * selected
    magnitude_sum = sum(absolute_totals.values())
    component_shares = {
        key: value / magnitude_sum if magnitude_sum else 0.0
        for key, value in absolute_totals.items()
    }

    reward_winners, physical_winners = [], []
    for demand in sorted({key[1] for key in indexed}):
        for seed in sorted({key[2] for key in indexed if key[1] == demand}):
            candidates = [indexed[(arm, demand, seed)] for arm in ("zero", "n010", "p010", "p020")]
            reward_winners.append(
                max(
                    candidates,
                    key=lambda row: float(row["learning_reward"]["mean_per_selected_agent_step"]),
                )["arm"]
            )
            cell_effects = [
                row for row in effects if row["demand"] == demand and row["seed"] == seed
            ]
            physical_winners.append(
                max(
                    [("zero", 0.0)]
                    + [(row["arm"], row["physical_utility"]) for row in cell_effects],
                    key=lambda pair: pair[1],
                )[0]
            )

    by_arm_demand = {}
    for arm in ("n010", "p010", "p020"):
        for demand in (2200, 2700):
            group = [row for row in effects if row["arm"] == arm and row["demand"] == demand]
            by_arm_demand[f"{arm}_m{demand}"] = {
                "count": len(group),
                "mean_reward_delta": mean(row["reward_delta"] for row in group),
                "mean_physical_utility": mean(row["physical_utility"] for row in group),
                "mean_wave_reduction": mean(row["wave_reduction"] for row in group),
                "mean_severe_braking_reduction": mean(
                    row["severe_braking_reduction"] for row in group
                ),
                "mean_delay_delta_s_per_vehicle": mean(
                    row["delay_delta_s_per_vehicle"] for row in group
                ),
                "maximum_delay_delta_s_per_vehicle": max(
                    row["delay_delta_s_per_vehicle"] for row in group
                ),
                "minimum_arrived_delta": min(row["arrived_delta"] for row in group),
                "maximum_collision_delta": max(row["collision_delta"] for row in group),
                "headroom_seed_count": sum(
                    row["wave_reduction"] >= 0.10 or row["severe_braking_reduction"] >= 0.10
                    for row in group
                ),
                "positive_utility_seed_count": sum(row["physical_utility"] > 0 for row in group),
                "positive_reward_seed_count": sum(row["reward_delta"] > 0 for row in group),
            }

    p010_positive_demand = any(
        summary["mean_physical_utility"] > 0 and summary["mean_reward_delta"] > 0
        for key, summary in by_arm_demand.items()
        if key.startswith("p010_")
    )
    result = {
        "schema": "idm_phase5_reward_readiness/1",
        "audit": str(args.audit.resolve()),
        "case_count": len(rows),
        "paired_effect_count": len(effects),
        "spearman_reward_vs_physical_utility": rho,
        "beneficial_case_count": len(beneficial),
        "effort_penalty_reversal_count": len(effort_reversals),
        "effort_penalty_reversals": effort_reversals,
        "component_absolute_shares": component_shares,
        "dominant_component": max(component_shares, key=component_shares.get),
        "dominant_component_share": max(component_shares.values()),
        "reward_winner_counts": dict(Counter(reward_winners)),
        "physical_winner_counts": dict(Counter(physical_winners)),
        "reward_physical_winner_match_fraction": mean(
            left == right for left, right in zip(reward_winners, physical_winners)
        ),
        "max_step_accounting_error": max_accounting_error,
        "missing_diagnostic_cases": missing_diagnostics,
        "by_arm_demand": by_arm_demand,
        "gates": {
            "complete_48_cases": len(rows) == 48 and len(effects) == 36,
            "reward_accounting_exact": max_accounting_error <= 1e-12,
            "diagnostics_complete": missing_diagnostics == 0,
            "reward_physical_spearman_at_least_0p50": rho >= 0.50,
            "no_effort_penalty_reversals": not effort_reversals,
            "p010_positive_reward_where_mean_utility_positive": p010_positive_demand,
            "reward_action_diversity": (
                len(set(reward_winners)) >= 2
                or (len(set(physical_winners)) == 1 and reward_winners == physical_winners)
            ),
            "no_unjustified_component_over_80pct": max(component_shares.values()) <= 0.80,
            "no_added_collisions": all(row["collision_delta"] <= 0 for row in effects),
            "no_material_discharge_loss": all(row["arrived_delta"] >= -1 for row in effects),
            "2700_headroom": any(
                summary["headroom_seed_count"] >= 2
                for key, summary in by_arm_demand.items()
                if key.endswith("m2700")
            ),
            "nonzero_arms_execute": all(
                abs(row["executed_residual_mean_mps2"]) > 1e-6 for row in effects
            ),
        },
    }
    result["training_readiness_pass"] = all(result["gates"].values())
    output = args.audit / "phase5_readiness.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
