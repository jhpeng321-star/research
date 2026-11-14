"""Re-score stored Phase-5 audit trajectories for residual-effort weights."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from statistics import mean
from analyze_idm_phase5_reward_audit import correlation, ranks

CURRENT_WEIGHT = 0.15
WEIGHTS = (0.0, 0.005, 0.01, 0.025, 0.05, 0.10, 0.15)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.audit / "summary.json").read_text(encoding="utf-8"))
    effects = summary["paired_effects"]
    rows = []
    for effect in effects:
        physical = (
            0.5 * (effect["wave_reduction"] + effect["severe_braking_reduction"])
            - 0.05 * max(effect["delay_delta_s_per_vehicle"], 0.0)
            - 0.25 * max(-effect["arrived_delta"] - 1, 0)
            - 1.0 * max(effect["collision_delta"], 0)
        )
        rows.append(effect | {"physical_utility": physical})
    output = []
    for weight in WEIGHTS:
        rescored = []
        for row in rows:
            effort = row["reward_component_deltas"]["residual_effort"]
            reward = row["learning_reward_delta"] - effort + effort * weight / CURRENT_WEIGHT
            rescored.append(row | {"rescored_reward_delta": reward})
        beneficial = [row for row in rescored if row["physical_utility"] > 0]
        reversals = [row for row in beneficial if row["rescored_reward_delta"] <= 0]
        output.append(
            {
                "residual_effort_weight": weight,
                "spearman": correlation(
                    ranks([row["physical_utility"] for row in rescored]),
                    ranks([row["rescored_reward_delta"] for row in rescored]),
                ),
                "beneficial_case_count": len(beneficial),
                "beneficial_reward_reversal_count": len(reversals),
                "p010_m2200_mean_reward_delta": mean(
                    row["rescored_reward_delta"]
                    for row in rescored
                    if row["arm"] == "p010" and row["demand"] == 2200
                ),
                "p010_m2700_mean_reward_delta": mean(
                    row["rescored_reward_delta"]
                    for row in rescored
                    if row["arm"] == "p010" and row["demand"] == 2700
                ),
                "reversal_cells": [
                    {
                        "arm": row["arm"],
                        "demand": row["demand"],
                        "seed": row["seed"],
                        "physical_utility": row["physical_utility"],
                        "reward_delta": row["rescored_reward_delta"],
                    }
                    for row in reversals
                ],
            }
        )
    path = args.audit / "effort_weight_sensitivity.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
