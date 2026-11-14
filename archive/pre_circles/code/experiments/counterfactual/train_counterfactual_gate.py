"""Train and evaluate the preliminary placement-aware counterfactual gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from counterfactual.gate_model import (
    binary_metrics,
    choose_l2_logo,
    fit_logistic,
    nested_logo_predictions,
)


from experiments import REPO_ROOT

PROJECT = REPO_ROOT
DEFAULT_DATASET = (
    PROJECT / "runs" / "probe" / "counterfactual_tau0_hyper_n30_s101_110_steps120_180_240.jsonl"
)
DEFAULT_READINESS = (
    PROJECT
    / "runs"
    / "probe"
    / "counterfactual_tau0_hyper_n30_s101_110_steps120_180_240.readiness.json"
)

HEALTH_FEATURES = (
    "health",
    "health_delta_1",
    "health_change_30",
)
TRAFFIC_FEATURES = HEALTH_FEATURES + (
    "decision_queue_count",
    "decision_pre_merge_mean_speed_mps",
    "pattern60_queue_count_change",
    "pattern60_pre_merge_speed_mps_change",
)
PLACEMENT_PATTERN_FEATURES = TRAFFIC_FEATURES + (
    "placement_pre_merge_cav_fraction",
    "placement_soft_band_cav_count",
    "placement_soft_band_lane_coverage_fraction",
)
FEATURE_SETS = {
    "health_only": HEALTH_FEATURES,
    "traffic_pattern": TRAFFIC_FEATURES,
    "placement_pattern": PLACEMENT_PATTERN_FEATURES,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train leakage-safe preliminary counterfactual gate models."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--l2-candidates", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "runs" / "probe" / "counterfactual_gate_n30.report.json",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=PROJECT / "runs" / "probe" / "counterfactual_gate_n30.predictions.csv",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _feature_matrix(rows: Sequence[Mapping[str, Any]], names: Sequence[str]) -> np.ndarray:
    values = []
    for row in rows:
        vector = []
        for name in names:
            value = row["features"].get(name)
            if value is None:
                raise ValueError(f"{row['case_id']} has missing feature {name}")
            vector.append(float(value))
        values.append(vector)
    matrix = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("selected feature matrix contains non-finite values")
    return matrix


def _policy_summary(
    rows: Sequence[Mapping[str, Any]],
    readiness_cases: Sequence[Mapping[str, Any]],
    actions: Sequence[int],
) -> Dict[str, Any]:
    outcomes = [row["arms"][str(action)]["outcomes"] for row, action in zip(rows, actions)]
    unsafe_activations = sum(
        action == 1 and not bool(case["actions"]["1"]["safety_admissible"])
        for case, action in zip(readiness_cases, actions)
    )
    return {
        "case_count": len(outcomes),
        "treatment_count": sum(action == 1 for action in actions),
        "unsafe_activation_count": unsafe_activations,
        "sustained_congestion_count": sum(
            bool(value["sustained_congestion"]) for value in outcomes
        ),
        "mean_queue_area_vehicle_s": mean(
            float(value["queue_area_vehicle_s"]) for value in outcomes
        ),
        "mean_incremental_delay_s": mean(float(value["incremental_delay_s"]) for value in outcomes),
        "mean_merge_discharge_count": mean(
            float(value["merge_discharge_count"]) for value in outcomes
        ),
        "mean_hard_brake_vehicle_steps": mean(
            float(value["hard_brake_vehicle_steps"]) for value in outcomes
        ),
        "mean_low_ttc_exposure_steps": mean(
            float(value["low_ttc_exposure_steps"]) for value in outcomes
        ),
        "mean_duty_fraction": mean(float(value["duty_fraction"]) for value in outcomes),
    }


def _write_predictions(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    args = parse_args()
    rows = _read_jsonl(args.input.resolve())
    readiness = json.loads(args.readiness.resolve().read_text(encoding="utf-8"))
    ready_by_id = {str(case["case_id"]): case for case in readiness["cases"]}
    readiness_cases = [ready_by_id[str(row["case_id"])] for row in rows]
    labels = np.asarray([int(case["selected_action"] == 1) for case in readiness_cases])
    groups = np.asarray([int(row["seed"]) for row in rows])
    threshold = float(args.threshold)

    print(
        f"training {len(rows)} cases across {len(np.unique(groups))} seed groups; "
        f"labels off={int((labels == 0).sum())}, soft={int(labels.sum())}",
        flush=True,
    )
    models: Dict[str, Any] = {}
    prediction_records = [
        {
            "case_id": str(row["case_id"]),
            "seed": int(row["seed"]),
            "decision_step": int(row["decision_step"]),
            "oracle_label": int(label),
            "soft_safety_admissible": bool(case["actions"]["1"]["safety_admissible"]),
            "soft_traffic_beneficial": bool(case["actions"]["1"]["traffic_beneficial"]),
        }
        for row, case, label in zip(rows, readiness_cases, labels)
    ]

    policies = {
        "always_off": _policy_summary(rows, readiness_cases, [0] * len(rows)),
        "always_soft": _policy_summary(rows, readiness_cases, [1] * len(rows)),
        "deployable_label_oracle": _policy_summary(
            rows, readiness_cases, [int(label) for label in labels]
        ),
    }

    for model_name, feature_names in FEATURE_SETS.items():
        matrix = _feature_matrix(rows, feature_names)
        variation = {
            name: int(len(np.unique(matrix[:, index]))) for index, name in enumerate(feature_names)
        }
        result = nested_logo_predictions(
            matrix,
            labels,
            groups,
            feature_names=feature_names,
            l2_candidates=tuple(args.l2_candidates),
        )
        probabilities = result["probabilities"]
        predictions = (probabilities >= threshold).astype(int)
        metrics = binary_metrics(labels, probabilities, threshold=threshold)
        selected_l2, all_data_scores = choose_l2_logo(
            matrix,
            labels,
            groups,
            feature_names=feature_names,
            candidates=args.l2_candidates,
        )
        final_model = fit_logistic(
            matrix,
            labels,
            feature_names=feature_names,
            l2=selected_l2,
        )
        unsafe_false_activations = int(
            sum(
                prediction == 1 and not bool(case["actions"]["1"]["safety_admissible"])
                for prediction, case in zip(predictions, readiness_cases)
            )
        )
        metrics["unsafe_false_activation_count"] = unsafe_false_activations
        policy = _policy_summary(rows, readiness_cases, predictions.tolist())
        threshold_sensitivity = {}
        for diagnostic_threshold in (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5):
            diagnostic_predictions = (probabilities >= diagnostic_threshold).astype(int)
            threshold_sensitivity[str(diagnostic_threshold)] = {
                "metrics": binary_metrics(
                    labels,
                    probabilities,
                    threshold=diagnostic_threshold,
                ),
                "policy": _policy_summary(
                    rows,
                    readiness_cases,
                    diagnostic_predictions.tolist(),
                ),
            }
        policies[model_name] = policy
        models[model_name] = {
            "feature_names": list(feature_names),
            "feature_unique_value_counts": variation,
            "evaluation": "nested leave-one-seed-out; fixed threshold",
            "metrics": metrics,
            "policy": policy,
            "threshold_sensitivity_diagnostic_only": threshold_sensitivity,
            "folds": result["folds"],
            "final_all_data_l2_selection_log_loss": all_data_scores,
            "final_model": final_model.to_dict(),
        }
        for record, probability, prediction in zip(prediction_records, probabilities, predictions):
            record[f"{model_name}_probability_soft"] = float(probability)
            record[f"{model_name}_prediction"] = int(prediction)
        print(
            f"{model_name}: balanced_accuracy={metrics['balanced_accuracy']:.3f}, "
            f"precision={metrics['precision_soft']:.3f}, recall={metrics['recall_soft']:.3f}, "
            f"MCC={metrics['matthews_correlation']:.3f}, "
            f"unsafe_activations={policy['unsafe_activation_count']}",
            flush=True,
        )
        for fold in result["folds"]:
            print(
                f"  held-out seed {fold['held_out_group']}: "
                f"test_positive={fold['test_positive_count']}, "
                f"l2={fold['selected_l2']}, "
                f"p={[round(value, 3) for value in fold['test_probabilities']]}",
                flush=True,
            )

    primary_metrics = models["placement_pattern"]["metrics"]
    primary_status = (
        "promising_preliminary_signal"
        if primary_metrics["balanced_accuracy"] > 0.5
        and primary_metrics["recall_soft"] > 0.0
        and primary_metrics["unsafe_false_activation_count"] == 0
        else "insufficient_held_out_seed_signal"
    )
    report = {
        "schema_version": 1,
        "status": "preliminary_grouped_evaluation_complete",
        "dataset": str(args.input.resolve()),
        "readiness_report": str(args.readiness.resolve()),
        "case_count": len(rows),
        "seed_group_count": int(len(np.unique(groups))),
        "label_counts": {"off": int((labels == 0).sum()), "soft": int(labels.sum())},
        "evaluation_protocol": {
            "outer_split": "leave one seed out",
            "inner_l2_selection": "leave one seed out on outer-training seeds",
            "l2_candidates": [float(value) for value in args.l2_candidates],
            "activation_threshold": threshold,
            "future_outcomes_in_features": False,
            "note": "All windows from a seed remain together in each held-out fold.",
        },
        "feature_sets": {name: list(values) for name, values in FEATURE_SETS.items()},
        "models": models,
        "policies": policies,
        "primary_model": "placement_pattern",
        "primary_model_status": primary_status,
        "prediction_file": str(args.predictions_output.resolve()),
        "limitation": (
            "Preliminary evaluation on 10 seed groups; a locked unseen-seed test "
            "is required before claiming predictor generalization."
        ),
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_predictions(args.predictions_output.resolve(), prediction_records)
    print(f"wrote {args.output.resolve()}", flush=True)
    print(f"wrote {args.predictions_output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
