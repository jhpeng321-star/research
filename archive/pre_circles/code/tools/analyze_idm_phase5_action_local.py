"""Analyze action-local pulse effects without fitting an underpowered model."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from experiments.io import atomic_write_json


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    run_dir = args.summary.resolve().parent
    branches = {
        row["branch_id"]: row
        for path in (run_dir / "branches").glob("*.json")
        for row in [json.loads(path.read_text(encoding="utf-8"))]
    }
    rows = []
    by_cell: dict[tuple[int, str], list[dict]] = defaultdict(list)
    component_abs = Counter()
    accounting_errors = []
    for branch in branches.values():
        accounting_errors.append(
            abs(float(branch["reward_sum"]) - sum(branch["reward_component_totals"].values()))
        )
    for case in summary["cases"]:
        zero = branches[f"{case['case_id']}.zero"]
        for effect in case["effects"]:
            action = branches[f"{case['case_id']}.{effect['arm']}"]
            component_deltas = {
                key: float(action["reward_component_totals"][key])
                - float(zero["reward_component_totals"][key])
                for key in zero["reward_component_totals"]
            }
            for key, value in component_deltas.items():
                component_abs[key] += abs(value)
            pulse_commands = [
                command
                for step in action["command_trace"]
                if int(step["relative_step"]) <= 10
                for command in step["commands"]
                if command.get("treatment_role", False)
            ]
            row = {
                "case_id": case["case_id"],
                "demand": int(case["demand"]),
                "seed": int(case["seed"]),
                "landmark": case["landmark"],
                **effect,
                "pulse_role_command_steps": len(pulse_commands),
                "pulse_mean_abs_executed_residual_mps2": _mean(
                    [abs(float(command["executed_residual_mps2"])) for command in pulse_commands]
                ),
                "pulse_max_abs_executed_residual_mps2": (
                    max(abs(float(command["executed_residual_mps2"])) for command in pulse_commands)
                    if pulse_commands
                    else 0.0
                ),
                "reward_component_deltas": component_deltas,
            }
            rows.append(row)
            by_cell[(row["demand"], row["arm"])].append(row)
    cell_summaries = {}
    for (demand, arm), cell in sorted(by_cell.items()):
        cell_summaries[f"m{demand}.{arm}"] = {
            "case_count": len(cell),
            "classification_counts": dict(Counter(row["classification"] for row in cell)),
            "helpful_seed_count": len(
                {row["seed"] for row in cell if row["classification"] == "helpful"}
            ),
            "harmful_seed_count": len(
                {row["seed"] for row in cell if row["classification"] == "harmful"}
            ),
            "mean_safety_composite": _mean(
                [float(row["safety_composite_reference_units"]) for row in cell]
            ),
            "mean_pulse_executed_magnitude_mps2": _mean(
                [float(row["pulse_mean_abs_executed_residual_mps2"] or 0.0) for row in cell]
            ),
        }
    utility = np.asarray([row["safety_composite_reference_units"] for row in rows], dtype=float)
    reward = np.asarray([row["reward_delta"] for row in rows], dtype=float)
    rho = float(spearmanr(utility, reward).statistic)
    harmful = [row for row in rows if row["classification"] == "harmful"]
    harmful_aligned = [row for row in harmful if float(row["reward_delta"]) <= 0.0]
    delay_only_wins = [
        row
        for row in rows
        if row["severe_braking_effect_reference_units"] < 0
        and row["wave_effect_reference_units"] < 0
        and row["reward_delta"] > 0
    ]
    total_component_abs = sum(component_abs.values())
    component_shares = {
        key: float(value / total_component_abs) if total_component_abs else 0.0
        for key, value in sorted(component_abs.items())
    }
    helpful_2200 = [
        row for row in rows if row["demand"] == 2200 and row["classification"] == "helpful"
    ]
    helpful_seeds = {row["seed"] for row in rows if row["classification"] == "helpful"}
    harmful_seeds = {row["seed"] for row in rows if row["classification"] == "harmful"}
    oracle_counts = Counter(case["oracle_action"] for case in summary["cases"])
    reward_winners = Counter()
    for case in summary["cases"]:
        candidates = [("zero", 0.0)] + [
            (effect["arm"], float(effect["reward_delta"])) for effect in case["effects"]
        ]
        reward_winners[max(candidates, key=lambda item: (item[1], item[0] == "zero"))[0]] += 1
    gates = {
        "action_execution_measurable": all(
            float(row["pulse_mean_abs_executed_residual_mps2"] or 0.0) > 1e-6 for row in rows
        ),
        "helpful_and_harmful_span_three_seeds": len(helpful_seeds) >= 3 and len(harmful_seeds) >= 3,
        "helpful_2200_spans_three_seeds": len({row["seed"] for row in helpful_2200}) >= 3,
        "meaningful_oracle_action_diversity": sum(count > 0 for count in oracle_counts.values())
        >= 2,
        "reward_utility_spearman_at_least_0p50": rho >= 0.50,
        "harmful_reward_direction_at_least_80pct": (
            len(harmful_aligned) / len(harmful) >= 0.80 if harmful else False
        ),
        "no_delay_only_safety_worsening_reward_wins": not delay_only_wins,
        "no_reward_component_above_80pct": max(component_shares.values(), default=0.0) <= 0.80,
        "reward_winner_action_diversity": sum(count > 0 for count in reward_winners.values()) >= 2,
        "exact_reward_component_accounting": max(accounting_errors, default=0.0) <= 1e-9,
    }
    model_status = {
        "fit_performed": False,
        "reason": (
            "No helpful action-state pairs occurred at 2200 veh/h; the primary within-regime "
            "classification target has one class, so AUROC and balanced accuracy are undefined."
        ),
        "pooled_model_not_used": (
            "All helpful pairs occurred at 2700 veh/h, so a pooled model could be driven by "
            "demand-regime separation rather than metastable action value."
        ),
    }
    report = {
        "schema": "idm_phase5_action_local_analysis/1",
        "source": str(args.summary.resolve()),
        "source_protocol_hash": summary["protocol_hash"],
        "case_count": len(summary["cases"]),
        "action_state_pair_count": len(rows),
        "cell_summaries": cell_summaries,
        "classification_counts": dict(Counter(row["classification"] for row in rows)),
        "helpful_seed_count": len(helpful_seeds),
        "harmful_seed_count": len(harmful_seeds),
        "oracle_action_counts": dict(oracle_counts),
        "reward_winner_counts": dict(reward_winners),
        "reward_alignment": {
            "spearman_rho": rho,
            "harmful_count": len(harmful),
            "harmful_nonpositive_reward_count": len(harmful_aligned),
            "harmful_direction_fraction": (
                len(harmful_aligned) / len(harmful) if harmful else None
            ),
            "delay_only_safety_worsening_reward_win_count": len(delay_only_wins),
            "delay_only_safety_worsening_reward_wins": delay_only_wins,
            "component_absolute_shares": component_shares,
            "maximum_component_accounting_error": max(accounting_errors, default=0.0),
        },
        "model_status": model_status,
        "gates": gates,
        "diagnostic_pass": all(gates.values()),
        "training_authorized": False,
        "rows": rows,
    }
    atomic_write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
