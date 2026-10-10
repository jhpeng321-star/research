"""Train a post-hoc merge-service counterfactual wall gate.

The frozen four-traffic-head experiment remains unchanged. This separate
trainer predicts only paired merge-speed gain and merge-discharge gain, plus
the same two absolute safety risks. Upstream queue and SUMO timeLoss are
reported policy outcomes but do not enter the activation rule.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from counterfactual.gate_model import (
    choose_l2_logo_regression,
    fit_ridge,
    nested_logo_regression_predictions,
    regression_metrics,
)
from counterfactual.win_sensitivity import (
    is_traffic_win,
    policy_summary,
    treatment_effects,
)
from experiments.counterfactual.train_counterfactual_two_head_gate import (
    FEATURE_SETS,
    _choose_feature_set,
    _feature_matrix,
    _read_jsonl,
)


from experiments import REPO_ROOT

PROJECT = REPO_ROOT
DEFAULT_DEV_DATA = (
    PROJECT / "runs/probe/counterfactual_local_v2_dev_combined_s101_130_steps120_180_240.jsonl"
)
DEFAULT_DEV_READINESS = (
    PROJECT / "runs/probe/counterfactual_local_v2_dev_s101_130_steps120_180_240.readiness.json"
)
DEFAULT_TEST_DATA = (
    PROJECT / "runs/probe/counterfactual_local_v2_locked_s201_210_steps120_180_240.jsonl"
)
DEFAULT_TEST_READINESS = (
    PROJECT / "runs/probe/counterfactual_local_v2_locked_s201_210_steps120_180_240.readiness.json"
)

TRAFFIC_TARGETS = ("merge_speed_gain_mps", "merge_discharge_gain")
SAFETY_TARGETS = (
    "soft_hard_brake_rate_vehicle_steps_per_s",
    "soft_low_ttc_exposure_fraction",
)
ALL_TARGETS = TRAFFIC_TARGETS + SAFETY_TARGETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DEV_DATA)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_DEV_READINESS)
    parser.add_argument("--test-input", type=Path, default=DEFAULT_TEST_DATA)
    parser.add_argument("--test-readiness", type=Path, default=DEFAULT_TEST_READINESS)
    parser.add_argument("--feature-set", choices=["auto", *FEATURE_SETS], default="auto")
    parser.add_argument(
        "--l2-candidates",
        type=float,
        nargs="+",
        default=[0.1, 1.0, 10.0, 100.0],
    )
    parser.add_argument("--safety-margin", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "runs/probe/counterfactual_merge_service_gate_local_v2.report.json",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=PROJECT / "runs/probe/counterfactual_merge_service_gate_local_v2.predictions.csv",
    )
    parser.add_argument(
        "--test-predictions-output",
        type=Path,
        default=PROJECT
        / "runs/probe/counterfactual_merge_service_gate_local_v2.retrospective_test_predictions.csv",
    )
    return parser.parse_args()


def _ready_by_id(readiness: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {str(case["case_id"]): case for case in readiness["cases"]}


def _targets(row: Mapping[str, Any], readiness_case: Mapping[str, Any]) -> Dict[str, float]:
    effect = treatment_effects(row)
    safety = readiness_case["actions"]["1"]
    return {
        "merge_speed_gain_mps": float(effect["merge_speed_gain_mps"]),
        "merge_discharge_gain": float(effect["merge_discharge_gain"]),
        "soft_hard_brake_rate_vehicle_steps_per_s": float(
            safety["hard_brake_rate_vehicle_steps_per_s"]
        ),
        "soft_low_ttc_exposure_fraction": float(safety["low_ttc_exposure_fraction"]),
    }


def merge_service_gate_actions(
    predictions: Mapping[str, np.ndarray],
    *,
    hard_brake_limit: float,
    low_ttc_limit: float,
    safety_margin: float = 1.0,
) -> np.ndarray:
    """Activate only for predicted exact-throughput merge-service benefit."""
    speed = np.asarray(predictions["merge_speed_gain_mps"], dtype=float)
    discharge = np.asarray(predictions["merge_discharge_gain"], dtype=float)
    traffic_effective = (speed >= 0.0) & (discharge >= 0.0) & ((speed > 0.0) | (discharge > 0.0))
    predicted_safe = (
        np.maximum(
            0.0,
            np.asarray(
                predictions["soft_hard_brake_rate_vehicle_steps_per_s"],
                dtype=float,
            ),
        )
        <= float(hard_brake_limit) * float(safety_margin)
    ) & (
        np.maximum(
            0.0,
            np.asarray(predictions["soft_low_ttc_exposure_fraction"], dtype=float),
        )
        <= float(low_ttc_limit) * float(safety_margin)
    )
    return (traffic_effective & predicted_safe).astype(int)


def _classification(
    rows: Sequence[Mapping[str, Any]],
    readiness_cases: Sequence[Mapping[str, Any]],
    actions: Sequence[int],
) -> Dict[str, Any]:
    actual = np.asarray(
        [
            int(
                is_traffic_win(row, "bottleneck_service_exact")
                and bool(case["actions"]["1"]["safety_admissible"])
            )
            for row, case in zip(rows, readiness_cases)
        ],
        dtype=int,
    )
    predicted = np.asarray(actions, dtype=int)
    tp = int(np.sum((actual == 1) & (predicted == 1)))
    fp = int(np.sum((actual == 0) & (predicted == 1)))
    fn = int(np.sum((actual == 1) & (predicted == 0)))
    tn = int(np.sum((actual == 0) & (predicted == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "actual_safe_win_count": int(np.sum(actual)),
        "predicted_soft_count": int(np.sum(predicted)),
    }


def _write_predictions(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _records(
    rows: Sequence[Mapping[str, Any]],
    readiness_cases: Sequence[Mapping[str, Any]],
    targets: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
    actions: Sequence[int],
) -> list[Dict[str, Any]]:
    records = []
    for index, (row, case, action) in enumerate(zip(rows, readiness_cases, actions)):
        record: Dict[str, Any] = {
            "case_id": str(row["case_id"]),
            "seed": int(row["seed"]),
            "decision_step": int(row["decision_step"]),
            "actual_safe_merge_service_win": int(
                is_traffic_win(row, "bottleneck_service_exact")
                and bool(case["actions"]["1"]["safety_admissible"])
            ),
            "predicted_action": int(action),
        }
        for target_name in ALL_TARGETS:
            record[f"actual_{target_name}"] = float(targets[target_name][index])
            record[f"predicted_{target_name}"] = float(predictions[target_name][index])
        records.append(record)
    return records


def _policy_set(
    rows: Sequence[Mapping[str, Any]],
    readiness: Mapping[str, Any],
    actions: Sequence[int],
) -> Dict[str, Any]:
    ready_map = _ready_by_id(readiness)
    oracle_actions = [
        int(
            is_traffic_win(row, "bottleneck_service_exact")
            and bool(ready_map[str(row["case_id"])]["actions"]["1"]["safety_admissible"])
        )
        for row in rows
    ]
    return {
        "always_off": policy_summary(rows, ready_map, [0] * len(rows)),
        "always_soft": policy_summary(rows, ready_map, [1] * len(rows)),
        "merge_service_label_oracle": policy_summary(rows, ready_map, oracle_actions),
        "predicted_gate": policy_summary(rows, ready_map, actions),
    }


def _policy_delta(policy: Mapping[str, Any], off: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "mean_merge_speed_gain_mps": float(policy["mean_merge_speed_mps"])
        - float(off["mean_merge_speed_mps"]),
        "mean_merge_discharge_gain": float(policy["mean_merge_discharge_count"])
        - float(off["mean_merge_discharge_count"]),
        "mean_delay_change_s": float(policy["mean_incremental_delay_s"])
        - float(off["mean_incremental_delay_s"]),
        "mean_queue_area_change_vehicle_s": float(policy["mean_queue_area_vehicle_s"])
        - float(off["mean_queue_area_vehicle_s"]),
    }


def main() -> int:
    args = parse_args()
    rows = [row for row in _read_jsonl(args.input.resolve()) if row.get("eligible")]
    readiness = json.loads(args.readiness.resolve().read_text(encoding="utf-8"))
    ready_map = _ready_by_id(readiness)
    readiness_cases = [ready_map[str(row["case_id"])] for row in rows]
    feature_set_name, feature_names = _choose_feature_set(rows, args.feature_set)
    matrix = _feature_matrix(rows, feature_names)
    groups = np.asarray([int(row["seed"]) for row in rows], dtype=int)
    target_rows = [_targets(row, case) for row, case in zip(rows, readiness_cases)]
    target_vectors = {
        name: np.asarray([target[name] for target in target_rows], dtype=float)
        for name in ALL_TARGETS
    }
    safety_budget = readiness["safety_budget"]
    hard_brake_limit = float(safety_budget["hard_brake_rate_vehicle_steps_per_s_max"])
    low_ttc_limit = float(safety_budget["low_ttc_exposure_fraction_max"])

    models: Dict[str, Any] = {}
    final_models: Dict[str, Any] = {}
    crossfit_predictions: Dict[str, np.ndarray] = {}
    print(
        f"training merge-service gate on {len(rows)} cases across {len(np.unique(groups))} seeds",
        flush=True,
    )
    for target_name in ALL_TARGETS:
        target = target_vectors[target_name]
        nested = nested_logo_regression_predictions(
            matrix,
            target,
            groups,
            feature_names=feature_names,
            l2_candidates=tuple(args.l2_candidates),
        )
        prediction = np.asarray(nested["predictions"], dtype=float)
        crossfit_predictions[target_name] = prediction
        selected_l2, scores = choose_l2_logo_regression(
            matrix,
            target,
            groups,
            feature_names=feature_names,
            candidates=args.l2_candidates,
        )
        final_model = fit_ridge(matrix, target, feature_names=feature_names, l2=selected_l2)
        final_models[target_name] = final_model
        metrics = regression_metrics(target, prediction)
        models[target_name] = {
            "head": "traffic_effect" if target_name in TRAFFIC_TARGETS else "safety_risk",
            "metrics": metrics,
            "folds": nested["folds"],
            "final_all_data_l2_selection_rmse": scores,
            "final_model": final_model.to_dict(),
        }
        print(
            f"{target_name}: RMSE={metrics['rmse']:.3f}, "
            f"R2={metrics['r2']:.3f}, corr={metrics['pearson_correlation']}",
            flush=True,
        )

    actions = merge_service_gate_actions(
        crossfit_predictions,
        hard_brake_limit=hard_brake_limit,
        low_ttc_limit=low_ttc_limit,
        safety_margin=float(args.safety_margin),
    )
    policies = _policy_set(rows, readiness, actions.tolist())
    dev_classification = _classification(rows, readiness_cases, actions.tolist())
    report: Dict[str, Any] = {
        "schema_version": 1,
        "status": "post_hoc_merge_service_gate_diagnostic",
        "frozen_four_traffic_head_experiment_modified": False,
        "dataset": str(args.input.resolve()),
        "readiness_report": str(args.readiness.resolve()),
        "case_count": len(rows),
        "seed_group_count": int(len(np.unique(groups))),
        "feature_set": feature_set_name,
        "feature_names": list(feature_names),
        "heads": {
            "traffic_effect": list(TRAFFIC_TARGETS),
            "safety_risk": list(SAFETY_TARGETS),
        },
        "decision_rule": {
            "merge_speed_gain_mps_min": 0.0,
            "merge_discharge_gain_min": 0.0,
            "requires_at_least_one_strict_traffic_improvement": True,
            "queue_and_delay_in_activation_rule": False,
            "hard_brake_rate_vehicle_steps_per_s_max": hard_brake_limit * float(args.safety_margin),
            "low_ttc_exposure_fraction_max": low_ttc_limit * float(args.safety_margin),
        },
        "evaluation_protocol": {
            "development": "nested leave-one-seed-out cross-fitting",
            "retrospective_test_warning": (
                "The prior locked split was inspected while choosing this post-hoc "
                "label and is no longer a confirmatory holdout."
            ),
        },
        "models": models,
        "development": {
            "classification": dev_classification,
            "policies": policies,
            "predicted_gate_delta_vs_off": _policy_delta(
                policies["predicted_gate"], policies["always_off"]
            ),
        },
        "prediction_file": str(args.predictions_output.resolve()),
    }
    _write_predictions(
        args.predictions_output.resolve(),
        _records(
            rows,
            readiness_cases,
            target_vectors,
            crossfit_predictions,
            actions.tolist(),
        ),
    )

    if args.test_input is not None or args.test_readiness is not None:
        if args.test_input is None or args.test_readiness is None:
            raise ValueError("--test-input and --test-readiness must be provided together")
        test_rows = [row for row in _read_jsonl(args.test_input.resolve()) if row.get("eligible")]
        test_readiness = json.loads(args.test_readiness.resolve().read_text(encoding="utf-8"))
        test_ready_map = _ready_by_id(test_readiness)
        test_cases = [test_ready_map[str(row["case_id"])] for row in test_rows]
        test_matrix = _feature_matrix(test_rows, feature_names)
        test_target_rows = [_targets(row, case) for row, case in zip(test_rows, test_cases)]
        test_targets = {
            name: np.asarray([target[name] for target in test_target_rows], dtype=float)
            for name in ALL_TARGETS
        }
        test_predictions = {name: final_models[name].predict(test_matrix) for name in ALL_TARGETS}
        test_actions = merge_service_gate_actions(
            test_predictions,
            hard_brake_limit=hard_brake_limit,
            low_ttc_limit=low_ttc_limit,
            safety_margin=float(args.safety_margin),
        )
        test_policies = _policy_set(test_rows, test_readiness, test_actions.tolist())
        report["retrospective_test"] = {
            "dataset": str(args.test_input.resolve()),
            "readiness_report": str(args.test_readiness.resolve()),
            "case_count": len(test_rows),
            "seed_group_count": len({int(row["seed"]) for row in test_rows}),
            "confirmatory_holdout": False,
            "target_metrics": {
                name: regression_metrics(test_targets[name], test_predictions[name])
                for name in ALL_TARGETS
            },
            "classification": _classification(test_rows, test_cases, test_actions.tolist()),
            "policies": test_policies,
            "predicted_gate_delta_vs_off": _policy_delta(
                test_policies["predicted_gate"], test_policies["always_off"]
            ),
            "prediction_file": str(args.test_predictions_output.resolve()),
        }
        _write_predictions(
            args.test_predictions_output.resolve(),
            _records(
                test_rows,
                test_cases,
                test_targets,
                test_predictions,
                test_actions.tolist(),
            ),
        )

    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"development: treatments={dev_classification['predicted_soft_count']}, "
        f"precision={dev_classification['precision']:.3f}, "
        f"recall={dev_classification['recall']:.3f}",
        flush=True,
    )
    if "retrospective_test" in report:
        test_classification = report["retrospective_test"]["classification"]
        print(
            f"retrospective_test: treatments={test_classification['predicted_soft_count']}, "
            f"precision={test_classification['precision']:.3f}, "
            f"recall={test_classification['recall']:.3f}",
            flush=True,
        )
    print(f"wrote {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
