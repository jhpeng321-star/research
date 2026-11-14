"""Run the frozen-development signal test for the v2 escalation gate."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from counterfactual.escalation_v2_model import (
    FEATURE_SETS,
    feature_matrix,
    grouped_oof_probabilities,
    prepare_rows,
    probability_metrics,
    rule_policy_metrics,
    threshold_sweep,
)
from experiments.io import atomic_write_json as atomic_json


MODEL_NAMES = ("regularized_logistic", "shallow_gradient_boosting")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=20260825)
    return parser.parse_args()


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        )
    case_ids = [str(row["case_id"]) for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("pooled v2 inputs contain duplicate case IDs")
    return rows


def write_predictions(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def baselines(prepared) -> dict[str, Any]:
    rules = {
        "never_wall": lambda _row: False,
        "always_command_wall": lambda _row: True,
        "late_step_180_or_240": lambda row: int(row["decision_step"]) >= 180,
        "step_240_only": lambda row: int(row["decision_step"]) == 240,
        "p30_late": lambda row: (
            abs(float(row["scenario"]["cav_penetration"]) - 0.30) < 1e-9
            and int(row["decision_step"]) >= 180
        ),
    }
    reports = {name: rule_policy_metrics(prepared, rule, name=name) for name, rule in rules.items()}
    health_values = sorted({float(row["features"]["health"]) for row in prepared.rows})
    health_rules = []
    for threshold in health_values:
        for direction in ("at_or_above", "at_or_below"):
            if direction == "at_or_above":

                def rule(row, value=threshold):
                    return float(row["features"]["health"]) >= value
            else:

                def rule(row, value=threshold):
                    return float(row["features"]["health"]) <= value

            report = rule_policy_metrics(
                prepared,
                rule,
                name=f"health_{direction}_{threshold:.6g}",
            )
            report["health_threshold"] = threshold
            report["direction"] = direction
            health_rules.append(report)
    reports["health_threshold_diagnostic_frontier"] = health_rules
    return reports


def evaluate_cohort(prepared, *, folds: int, random_state: int):
    report: dict[str, Any] = {
        "case_count": len(prepared.rows),
        "group_count": int(len(np.unique(prepared.groups))),
        "selected_case_count": int(prepared.selected.sum()),
        "engaged_case_count": int(prepared.engaged.sum()),
        "unsafe_engagement_case_count": int(prepared.unsafe.sum()),
        "baselines": baselines(prepared),
        "models": {},
    }
    predictions: dict[str, np.ndarray] = {}
    for feature_set, feature_names in FEATURE_SETS.items():
        matrix = feature_matrix(prepared.rows, feature_names)
        unique_counts = {
            name: int(len(np.unique(matrix[:, index]))) for index, name in enumerate(feature_names)
        }
        for model_name in MODEL_NAMES:
            key = f"{model_name}__{feature_set}"
            probabilities, fold_details = grouped_oof_probabilities(
                matrix,
                prepared.selected,
                prepared.groups,
                model_name=model_name,
                n_splits=folds,
                random_state=random_state,
            )
            predictions[key] = probabilities
            report["models"][key] = {
                "model_name": model_name,
                "feature_set": feature_set,
                "feature_names": list(feature_names),
                "feature_unique_value_counts": unique_counts,
                "probability_metrics": probability_metrics(prepared.selected, probabilities),
                "threshold_sweep_diagnostic_only": threshold_sweep(prepared, probabilities),
                "folds": fold_details,
            }
    return report, predictions


def main() -> int:
    args = parse_args()
    input_paths = [path.resolve() for path in args.input]
    rows = read_rows(input_paths)
    primary = prepare_rows(rows, engaged_only=False)
    conditional = prepare_rows(rows, engaged_only=True)
    primary_report, primary_predictions = evaluate_cohort(
        primary, folds=args.folds, random_state=args.random_state
    )
    conditional_report, _ = evaluate_cohort(
        conditional, folds=args.folds, random_state=args.random_state
    )

    prediction_records: list[dict[str, Any]] = []
    for index, (row, oracle) in enumerate(zip(primary.rows, primary.oracle)):
        record: dict[str, Any] = {
            "case_id": row["case_id"],
            "group_id": primary.groups[index],
            "demand": row["scenario"]["demand"],
            "cav_penetration": row["scenario"]["cav_penetration"],
            "seed": row["seed"],
            "decision_step": row["decision_step"],
            "wall_engaged": int(oracle["wall_engaged"]),
            "safety_admissible": int(oracle["safety_admissible"]),
            "meaningful_benefit": int(oracle["meaningful_benefit"]),
            "wall_selected": int(oracle["wall_selected"]),
        }
        for key, probabilities in primary_predictions.items():
            record[f"p__{key}"] = float(probabilities[index])
        prediction_records.append(record)

    report = {
        "schema_version": "counterfactual_escalation_v2_model_signal/1",
        "status": "development_oof_signal_test_complete",
        "input_paths": [str(path) for path in input_paths],
        "evaluation_protocol": {
            "primary_cohort": "all eligible decision records",
            "outer_evaluation": (f"{args.folds}-fold stratified grouped out-of-fold prediction"),
            "group": "demand|penetration|HDV model|traffic/CAV seed",
            "random_state": args.random_state,
            "future_outcomes_in_features": False,
            "model_hyperparameters": "fixed before inspecting locked test data",
            "threshold_sweeps": (
                "development diagnostics only; no deployment threshold is frozen by this run"
            ),
        },
        "primary_all_eligible": primary_report,
        "engaged_only_post_decision_diagnostic": {
            "warning": (
                "This cohort conditions on eventual post-decision wall engagement and "
                "must not be presented as deployable gate performance."
            ),
            **conditional_report,
        },
        "predictions_output": str(args.predictions_output.resolve()),
        "locked_test_opened": False,
        "claim_status": "not_yet_verified",
    }
    atomic_json(args.output.resolve(), report)
    write_predictions(args.predictions_output.resolve(), prediction_records)

    ranked = sorted(
        (
            (
                model["probability_metrics"]["average_precision"],
                name,
                model["probability_metrics"]["roc_auc"],
            )
            for name, model in primary_report["models"].items()
        ),
        reverse=True,
    )
    print(
        json.dumps(
            {
                "eligible_cases": len(primary.rows),
                "groups": len(np.unique(primary.groups)),
                "selected": int(primary.selected.sum()),
                "engaged": int(primary.engaged.sum()),
                "unsafe": int(primary.unsafe.sum()),
                "top_models_by_oof_average_precision": [
                    {"model": name, "average_precision": ap, "roc_auc": auc}
                    for ap, name, auc in ranked
                ],
            },
            indent=2,
        )
    )
    print(f"wrote {args.output.resolve()}")
    print(f"wrote {args.predictions_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
