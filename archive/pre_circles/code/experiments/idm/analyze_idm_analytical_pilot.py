"""Generate the frozen Phase-3 IDM actuator diagnosis from case JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def load_cases(root: Path) -> dict:
    rows = {}
    for path in (root / "cases").glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        rows[(row["arm"], int(row["demand_mainline_vph"]), int(row["seed"]))] = row
    return rows


def value(row: dict, *keys):
    for key in keys:
        row = row[key]
    return float(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = load_cases(root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    analytical_arm = summary.get("evaluated_analytical_arm", "analytical")
    cells = {}
    seed_rows = []
    demands = sorted({key[1] for key in rows if key[0] == "noop"})
    for demand in demands:
        effects = []
        seeds = sorted(key[2] for key in rows if key[0] == "noop" and key[1] == demand)
        for seed in seeds:
            noop = rows[("noop", demand, seed)]
            arm = rows[(analytical_arm, demand, seed)]
            wave0 = value(noop, "trajectory", "wave", "oscillation_energy_m2ps3")
            wave1 = value(arm, "trajectory", "wave", "oscillation_energy_m2ps3")
            brake0 = value(noop, "trajectory", "safety", "severe_braking_events")
            brake1 = value(arm, "trajectory", "safety", "severe_braking_events")
            item = {
                "demand": demand,
                "seed": seed,
                "wave_reduction": (wave0 - wave1) / wave0,
                "severe_braking_reduction": (brake0 - brake1) / brake0,
                "delay_delta_s_per_vehicle": (
                    value(arm, "metrics", "mean_time_loss")
                    - value(noop, "metrics", "mean_time_loss")
                ),
                "ttc_event_delta": (
                    int(value(arm, "metrics", "ttc", "total_events"))
                    - int(value(noop, "metrics", "ttc", "total_events"))
                ),
                "side_ttc_event_delta": (
                    int(value(arm, "metrics", "ttc_side", "total_events"))
                    - int(value(noop, "metrics", "ttc_side", "total_events"))
                ),
                "hard_braking_reduction": (
                    value(noop, "trajectory", "safety", "hard_braking_events_recomputed")
                    - value(arm, "trajectory", "safety", "hard_braking_events_recomputed")
                )
                / max(
                    value(noop, "trajectory", "safety", "hard_braking_events_recomputed"),
                    1.0,
                ),
                "jerk_reduction": (
                    value(noop, "trajectory", "safety", "integrated_absolute_jerk_mps2")
                    - value(arm, "trajectory", "safety", "integrated_absolute_jerk_mps2")
                )
                / value(noop, "trajectory", "safety", "integrated_absolute_jerk_mps2"),
                "follower_response_samples": int(
                    value(arm, "trajectory", "safety", "follower_response_samples")
                ),
                "follower_severe_response_events": int(
                    value(arm, "trajectory", "safety", "follower_severe_response_events")
                ),
                "follower_min_acceleration_mps2": value(
                    arm, "trajectory", "safety", "follower_min_acceleration_mps2"
                ),
                "collisions": int(value(arm, "metrics", "collisions")),
                "arrived_delta": (
                    int(value(arm, "metrics", "arrived")) - int(value(noop, "metrics", "arrived"))
                ),
                "controlled_cav_seconds": value(arm, "control", "controlled_cav_seconds"),
                "projection_frequency": value(arm, "control", "projection_constraint_frequency"),
                "low_speed_wave_duration_noop_s": value(
                    noop, "trajectory", "wave", "low_speed_wave_duration_s"
                ),
            }
            effects.append(item)
            seed_rows.append(item)
        cells[str(demand)] = {
            "mean_wave_reduction": mean(x["wave_reduction"] for x in effects),
            "mean_severe_braking_reduction": mean(x["severe_braking_reduction"] for x in effects),
            "wave_wins_at_10pct": sum(x["wave_reduction"] >= 0.10 for x in effects),
            "braking_wins_at_10pct": sum(x["severe_braking_reduction"] >= 0.10 for x in effects),
            "mean_delay_delta_s_per_vehicle": mean(x["delay_delta_s_per_vehicle"] for x in effects),
            "max_delay_delta_s_per_vehicle": max(x["delay_delta_s_per_vehicle"] for x in effects),
            "zero_collisions": all(x["collisions"] == 0 for x in effects),
            "no_discharge_loss": all(x["arrived_delta"] >= -1 for x in effects),
            "mean_controlled_cav_seconds": mean(x["controlled_cav_seconds"] for x in effects),
            "mean_projection_frequency": mean(x["projection_frequency"] for x in effects),
            "collapse_headroom_cases": sum(
                x["low_speed_wave_duration_noop_s"] > 0 for x in effects
            ),
        }
    at_2200 = [row for row in seed_rows if row["demand"] == 2200]
    best = max(at_2200, key=lambda row: row["wave_reduction"])
    worst = min(at_2200, key=lambda row: row["wave_reduction"])
    diagnosis = {
        "schema": "idm_analytical_pilot_diagnosis/1",
        "record_count": len(rows),
        "identity_all_pass": bool(summary["identity_all_pass"]),
        "phase3_pass": bool(summary["phase3_pass"]),
        "evaluated_analytical_arm": analytical_arm,
        "cells": cells,
        "seed_effects": seed_rows,
        "best_2200_seed": best,
        "worst_2200_seed": worst,
        "conclusion": (
            "The software and zero-residual identity contracts pass, but the "
            "selected-absorber actuator does not meet the four-of-six efficacy "
            "gate and the frozen cells contain no sustained-collapse headroom. "
            "Diagnose projection/reference/geometry on the best and worst 2200 "
            "development seeds before changing the treatment or training RL."
        ),
    }
    (root / "diagnosis.json").write_text(
        json.dumps(diagnosis, indent=2, sort_keys=True), encoding="utf-8"
    )
    lines = [
        "# IDM analytical actuator Phase-3 diagnosis",
        "",
        f"- Records: {len(rows)}",
        f"- Analytical/zero-residual identity: {summary['identity_all_pass']}",
        f"- Frozen Phase-3 gate: {summary['phase3_pass']}",
        "- Sustained-collapse headroom: 0/24 no-op cases",
        "",
        "## Demand-cell results",
        "",
        "| Mainline demand | Mean wave reduction | Mean severe-brake reduction | "
        "10% wins (wave/brake) | Mean delay delta | Mean dose |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for demand, cell in cells.items():
        lines.append(
            f"| {demand} | {100 * cell['mean_wave_reduction']:.1f}% | "
            f"{100 * cell['mean_severe_braking_reduction']:.1f}% | "
            f"{cell['wave_wins_at_10pct']}/"
            f"{cell['braking_wins_at_10pct']} of 6 | "
            f"{cell['mean_delay_delta_s_per_vehicle']:+.2f} s/veh | "
            f"{cell['mean_controlled_cav_seconds']:.0f} CAV-s |"
        )
    lines += [
        "",
        "## Targeted diagnosis seeds",
        "",
        f"- Best 2200 seed: {best['seed']} ({100 * best['wave_reduction']:.1f}% wave reduction).",
        f"- Worst 2200 seed: {worst['seed']} "
        f"({100 * worst['wave_reduction']:.1f}% wave reduction).",
        "",
        "## Decision",
        "",
        diagnosis["conclusion"],
    ]
    (root / "DIAGNOSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "diagnosis": str(root / "diagnosis.json"),
                "report": str(root / "DIAGNOSIS.md"),
                "best_2200_seed": best["seed"],
                "worst_2200_seed": worst["seed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
