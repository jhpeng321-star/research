"""Train a grouped, safety-constrained continuous counterfactual gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from counterfactual.gate_model import (
    choose_l2_logo_regression,
    fit_ridge,
    nested_logo_regression_predictions,
    regression_metrics,
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

EXISTING_PATTERN_FEATURES = (
    "decision_queue_count",
    "decision_pre_merge_vehicle_count",
    "decision_pre_merge_mean_speed_mps",
    "decision_merge_mean_speed_mps",
    "merge_density_ema_scaled",
    "queue_spillback",
    "pattern60_queue_count_change",
    "pattern60_queue_count_range",
    "pattern60_pre_merge_speed_mps_change",
    "pattern60_pre_merge_speed_mps_range",
    "pattern60_merge_speed_mps_change",
    "placement_pre_merge_cav_fraction",
    "placement_soft_band_cav_count",
    "placement_soft_band_lane_coverage_fraction",
    "placement_cav_distance_to_merge_mean_m",
    "placement_longitudinal_span_m",
)
LOCAL_V2_FEATURES = EXISTING_PATTERN_FEATURES + (
    "local_bin_0_100m_vehicle_count",
    "local_bin_0_100m_speed_std_mps",
    "local_bin_100_200m_vehicle_count",
    "local_bin_100_200m_speed_std_mps",
    "local_bin_200_400m_vehicle_count",
    "local_bin_200_400m_speed_std_mps",
    "local_downstream_minus_upstream_speed_mps",
    "decision_hard_brake_vehicle_count",
    "decision_low_ttc_pair_count",
    "pattern60_pre_merge_speed_std_mps_mean",
    "pattern60_hard_brake_active_fraction",
    "pattern60_hard_brake_onset_count",
    "pattern60_low_ttc_active_fraction",
    "formation_leader_gap_min_m",
    "formation_follower_gap_min_m",
    "formation_cav_minus_neighbor_hdv_speed_mps",
)
FEATURE_SETS = {
    "existing_pattern": EXISTING_PATTERN_FEATURES,
    "local_v2": LOCAL_V2_FEATURES,
}
TRAFFIC_TARGETS = (
    "sustained_congestion_reduction",
    "queue_area_reduction_vehicle_s",
    "delay_reduction_s",
    "additional_merge_discharge",
)
SAFETY_TARGETS = (
    "soft_hard_brake_rate_vehicle_steps_per_s",
    "soft_low_ttc_exposure_fraction",
)
ALL_TARGETS = TRAFFIC_TARGETS + SAFETY_TARGETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train continuous traffic-effect and safety-risk heads with nested "
            "leave-one-seed-out evaluation."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument(
        "--test-input",
        type=Path,
        default=None,
        help="Optional locked paired dataset; never used for model or rule selection.",
    )
    parser.add_argument(
        "--test-readiness",
        type=Path,
        default=None,
        help="Frozen readiness audit corresponding to --test-input.",
    )
    parser.add_argument(
        "--feature-set",
        choices=["auto", *FEATURE_SETS],
        default="auto",
    )
    parser.add_argument(
        "--l2-candidates",
        type=float,
        nargs="+",
        default=[0.1, 1.0, 10.0, 100.0],
    )
    parser.add_argument("--queue-reduction-min", type=float, default=0.0)
    parser.add_argument("--delay-reduction-min", type=float, default=0.0)
    parser.add_argument("--discharge-gain-min", type=float, default=0.0)
    parser.add_argument("--safety-margin", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "runs" / "probe" / "counterfactual_two_head_gate.report.json",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=PROJECT / "runs" / "probe" / "counterfactual_two_head_gate.predictions.csv",
    )
    parser.add_argument(
        "--test-predictions-output",
        type=Path,
        default=PROJECT
        / "runs"
        / "probe"
        / "counterfactual_two_head_gate.locked_test_predictions.csv",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _available(rows: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    return all(
        name in row["features"] and row["features"][name] is not None
        for row in rows
        for name in names
    )


def _choose_feature_set(
    rows: Sequence[Mapping[str, Any]], requested: str
) -> tuple[str, tuple[str, ...]]:
    if requested != "auto":
        names = FEATURE_SETS[requested]
        if not _available(rows, names):
            missing = sorted(
                {
                    name
                    for row in rows
                    for name in names
                    if name not in row["features"] or row["features"][name] is None
                }
            )
            raise ValueError(f"feature set {requested} is unavailable; missing {missing}")
        return requested, names
    if _available(rows, LOCAL_V2_FEATURES):
        return "local_v2", LOCAL_V2_FEATURES
    return "existing_pattern", EXISTING_PATTERN_FEATURES


def _feature_matrix(rows: Sequence[Mapping[str, Any]], names: Sequence[str]) -> np.ndarray:
    matrix = np.asarray(
        [[float(row["features"][name]) for name in names] for row in rows],
        dtype=float,
    )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("selected feature matrix contains non-finite values")
    return matrix


def _targets(row: Mapping[str, Any], readiness_case: Mapping[str, Any]) -> Dict[str, float]:
    off = row["arms"]["0"]["outcomes"]
    soft = row["arms"]["1"]["outcomes"]
    safety = readiness_case["actions"]["1"]
    return {
        "sustained_congestion_reduction": float(
            int(bool(off["sustained_congestion"])) - int(bool(soft["sustained_congestion"]))
        ),
        "queue_area_reduction_vehicle_s": (
            float(off["queue_area_vehicle_s"]) - float(soft["queue_area_vehicle_s"])
        ),
        "delay_reduction_s": (
            float(off["incremental_delay_s"]) - float(soft["incremental_delay_s"])
        ),
        "additional_merge_discharge": (
            float(soft["merge_discharge_count"]) - float(off["merge_discharge_count"])
        ),
        "soft_hard_brake_rate_vehicle_steps_per_s": float(
            safety["hard_brake_rate_vehicle_steps_per_s"]
        ),
        "soft_low_ttc_exposure_fraction": float(safety["low_ttc_exposure_fraction"]),
    }


def _gate_actions(
    predictions: Mapping[str, np.ndarray],
    *,
    hard_brake_limit: float,
    low_ttc_limit: float,
    queue_reduction_min: float,
    delay_reduction_min: float,
    discharge_gain_min: float,
    safety_margin: float,
) -> np.ndarray:
    traffic_nonworsening = (
        (predictions["sustained_congestion_reduction"] >= 0.0)
        & (predictions["queue_area_reduction_vehicle_s"] >= queue_reduction_min)
        & (predictions["delay_reduction_s"] >= delay_reduction_min)
        & (predictions["additional_merge_discharge"] >= discharge_gain_min)
    )
    predicted_strict_benefit = (
        (predictions["sustained_congestion_reduction"] > 0.0)
        | (predictions["queue_area_reduction_vehicle_s"] > queue_reduction_min)
        | (predictions["delay_reduction_s"] > delay_reduction_min)
        | (predictions["additional_merge_discharge"] > discharge_gain_min)
    )
    predicted_safe = (
        np.maximum(
            0.0,
            predictions["soft_hard_brake_rate_vehicle_steps_per_s"],
        )
        <= hard_brake_limit * safety_margin
    ) & (
        np.maximum(
            0.0,
            predictions["soft_low_ttc_exposure_fraction"],
        )
        <= low_ttc_limit * safety_margin
    )
    return (traffic_nonworsening & predicted_strict_benefit & predicted_safe).astype(int)


def _policy_summary(
    rows: Sequence[Mapping[str, Any]],
    readiness_cases: Sequence[Mapping[str, Any]],
    actions: Sequence[int],
) -> Dict[str, Any]:
    outcomes = [row["arms"][str(action)]["outcomes"] for row, action in zip(rows, actions)]
    treatment_count = sum(action == 1 for action in actions)
    unsafe_activations = sum(
        action == 1 and not bool(case["actions"]["1"]["safety_admissible"])
        for case, action in zip(readiness_cases, actions)
    )
    beneficial_activations = sum(
        action == 1 and bool(case["actions"]["1"]["traffic_beneficial"])
        for case, action in zip(readiness_cases, actions)
    )
    deployable_activations = sum(
        action == 1 and bool(case["actions"]["1"]["deployable"])
        for case, action in zip(readiness_cases, actions)
    )
    return {
        "case_count": len(outcomes),
        "treatment_count": treatment_count,
        "unsafe_activation_count": unsafe_activations,
        "traffic_beneficial_activation_count": beneficial_activations,
        "deployable_activation_count": deployable_activations,
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
        "mean_collision_unique_vehicle_count": mean(
            float(value.get("collision_unique_vehicle_count", 0)) for value in outcomes
        ),
        "mean_starting_teleport_unique_vehicle_count": mean(
            float(value.get("starting_teleport_unique_vehicle_count", 0)) for value in outcomes
        ),
    }


def _write_predictions(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    args = parse_args()
    if (args.test_input is None) != (args.test_readiness is None):
        raise ValueError("--test-input and --test-readiness must be provided together")
    rows = [row for row in _read_jsonl(args.input.resolve()) if row.get("eligible")]
    if not rows or any("1" not in row.get("arms", {}) for row in rows):
        raise ValueError("all eligible rows must contain paired off and soft arms")
    readiness = json.loads(args.readiness.resolve().read_text(encoding="utf-8"))
    ready_by_id = {str(case["case_id"]): case for case in readiness["cases"]}
    readiness_cases = [ready_by_id[str(row["case_id"])] for row in rows]
    feature_set_name, feature_names = _choose_feature_set(rows, args.feature_set)
    matrix = _feature_matrix(rows, feature_names)
    groups = np.asarray([int(row["seed"]) for row in rows])
    target_rows = [_targets(row, case) for row, case in zip(rows, readiness_cases)]
    target_vectors = {
        name: np.asarray([values[name] for values in target_rows], dtype=float)
        for name in ALL_TARGETS
    }
    safety_budget = readiness["safety_budget"]
    hard_brake_limit = float(safety_budget["hard_brake_rate_vehicle_steps_per_s_max"])
    low_ttc_limit = float(safety_budget["low_ttc_exposure_fraction_max"])

    print(
        f"training two-head gate on {len(rows)} cases across "
        f"{len(np.unique(groups))} seeds with {feature_set_name} "
        f"({len(feature_names)} features)",
        flush=True,
    )
    models: Dict[str, Any] = {}
    final_models: Dict[str, Any] = {}
    crossfit_predictions: Dict[str, np.ndarray] = {}
    for target_name in ALL_TARGETS:
        target = target_vectors[target_name]
        result = nested_logo_regression_predictions(
            matrix,
            target,
            groups,
            feature_names=feature_names,
            l2_candidates=tuple(args.l2_candidates),
        )
        prediction = result["predictions"]
        crossfit_predictions[target_name] = prediction
        selected_l2, selection_scores = choose_l2_logo_regression(
            matrix,
            target,
            groups,
            feature_names=feature_names,
            candidates=args.l2_candidates,
        )
        final_model = fit_ridge(
            matrix,
            target,
            feature_names=feature_names,
            l2=selected_l2,
        )
        final_models[target_name] = final_model
        metrics = regression_metrics(target, prediction)
        models[target_name] = {
            "head": "traffic_effect" if target_name in TRAFFIC_TARGETS else "safety_risk",
            "metrics": metrics,
            "folds": result["folds"],
            "final_all_data_l2_selection_rmse": selection_scores,
            "final_model": final_model.to_dict(),
        }
        print(
            f"{target_name}: RMSE={metrics['rmse']:.3f}, "
            f"R2={metrics['r2']:.3f}, "
            f"corr={metrics['pearson_correlation']}",
            flush=True,
        )

    gate_actions = _gate_actions(
        crossfit_predictions,
        hard_brake_limit=hard_brake_limit,
        low_ttc_limit=low_ttc_limit,
        queue_reduction_min=float(args.queue_reduction_min),
        delay_reduction_min=float(args.delay_reduction_min),
        discharge_gain_min=float(args.discharge_gain_min),
        safety_margin=float(args.safety_margin),
    )
    deployable_actions = np.asarray(
        [int(case["selected_action"] == 1) for case in readiness_cases],
        dtype=int,
    )
    policies = {
        "always_off": _policy_summary(rows, readiness_cases, [0] * len(rows)),
        "always_soft": _policy_summary(rows, readiness_cases, [1] * len(rows)),
        "deployable_label_oracle": _policy_summary(
            rows, readiness_cases, deployable_actions.tolist()
        ),
        "two_head_gate_crossfit": _policy_summary(rows, readiness_cases, gate_actions.tolist()),
    }
    diagnostics: Dict[str, Any] = {}
    for queue_min in (0.0, 100.0, 250.0):
        for margin in (0.8, 1.0):
            name = f"queue_min_{queue_min:g}_safety_margin_{margin:g}"
            actions = _gate_actions(
                crossfit_predictions,
                hard_brake_limit=hard_brake_limit,
                low_ttc_limit=low_ttc_limit,
                queue_reduction_min=queue_min,
                delay_reduction_min=float(args.delay_reduction_min),
                discharge_gain_min=float(args.discharge_gain_min),
                safety_margin=margin,
            )
            diagnostics[name] = _policy_summary(rows, readiness_cases, actions.tolist())

    gate_policy = policies["two_head_gate_crossfit"]
    off_policy = policies["always_off"]
    improves_off = (
        gate_policy["mean_queue_area_vehicle_s"] < off_policy["mean_queue_area_vehicle_s"]
        and gate_policy["mean_incremental_delay_s"] <= off_policy["mean_incremental_delay_s"]
        and gate_policy["mean_merge_discharge_count"] >= off_policy["mean_merge_discharge_count"]
    )
    status = (
        "promising_development_signal"
        if improves_off and gate_policy["unsafe_activation_count"] == 0
        else "insufficient_held_out_seed_policy_signal"
    )

    records = []
    for index, (row, case, action) in enumerate(zip(rows, readiness_cases, gate_actions)):
        record: Dict[str, Any] = {
            "case_id": str(row["case_id"]),
            "seed": int(row["seed"]),
            "decision_step": int(row["decision_step"]),
            "actual_deployable_soft": int(case["selected_action"] == 1),
            "predicted_action": int(action),
        }
        for target_name in ALL_TARGETS:
            record[f"actual_{target_name}"] = float(target_vectors[target_name][index])
            record[f"predicted_{target_name}"] = float(crossfit_predictions[target_name][index])
        records.append(record)

    report = {
        "schema_version": 1,
        "status": "preliminary_grouped_two_head_evaluation_complete",
        "primary_model_status": status,
        "dataset": str(args.input.resolve()),
        "readiness_report": str(args.readiness.resolve()),
        "case_count": len(rows),
        "seed_group_count": int(len(np.unique(groups))),
        "feature_set": feature_set_name,
        "feature_names": list(feature_names),
        "feature_unique_value_counts": {
            name: int(len(np.unique(matrix[:, index]))) for index, name in enumerate(feature_names)
        },
        "heads": {
            "traffic_effect": list(TRAFFIC_TARGETS),
            "safety_risk": list(SAFETY_TARGETS),
        },
        "decision_rule": {
            "action_space": {"0": "off", "1": "safe_soft_wall"},
            "queue_area_reduction_vehicle_s_min": float(args.queue_reduction_min),
            "delay_reduction_s_min": float(args.delay_reduction_min),
            "additional_merge_discharge_min": float(args.discharge_gain_min),
            "sustained_congestion_reduction_min": 0.0,
            "hard_brake_rate_vehicle_steps_per_s_max": (
                hard_brake_limit * float(args.safety_margin)
            ),
            "low_ttc_exposure_fraction_max": (low_ttc_limit * float(args.safety_margin)),
            "requires_at_least_one_strict_predicted_traffic_improvement": True,
        },
        "evaluation_protocol": {
            "outer_split": "leave one seed out",
            "inner_l2_selection": "leave one seed out on outer-training seeds",
            "l2_candidates": [float(value) for value in args.l2_candidates],
            "future_outcomes_in_features": False,
            "policy_actions_use_cross_fitted_predictions_only": True,
            "note": "All windows from a seed remain in the same held-out fold.",
        },
        "models": models,
        "policies": policies,
        "diagnostic_rules_not_selected": diagnostics,
        "prediction_file": str(args.predictions_output.resolve()),
        "limitations": [
            "Only 10 independent seed groups are available in this development audit.",
            (
                "The legacy 30-case dataset predates the local_v2 shock and "
                "formation features, so auto mode uses existing_pattern."
            )
            if feature_set_name != "local_v2"
            else "Local-v2 features are available.",
            (
                "Collisions and teleports are hard evaluation vetoes rather than "
                "learned outputs because all current off/soft cases are event-free."
            ),
            (
                "A locked unseen-seed test is required before a working-gate claim."
                if args.test_input is None
                else "The optional locked test was evaluated without model or rule selection."
            ),
        ],
    }
    if args.test_input is not None:
        test_rows = [row for row in _read_jsonl(args.test_input.resolve()) if row.get("eligible")]
        if not test_rows or any("1" not in row.get("arms", {}) for row in test_rows):
            raise ValueError("all locked-test rows must contain paired off and soft arms")
        test_groups = np.asarray([int(row["seed"]) for row in test_rows])
        overlap = sorted(set(groups.tolist()) & set(test_groups.tolist()))
        if overlap:
            raise ValueError(f"locked-test seeds overlap development seeds: {overlap}")
        if not _available(test_rows, feature_names):
            raise ValueError(
                "locked-test dataset does not contain the frozen development feature set"
            )
        test_readiness = json.loads(args.test_readiness.resolve().read_text(encoding="utf-8"))
        test_ready_by_id = {str(case["case_id"]): case for case in test_readiness["cases"]}
        test_readiness_cases = [test_ready_by_id[str(row["case_id"])] for row in test_rows]
        test_matrix = _feature_matrix(test_rows, feature_names)
        test_target_rows = [
            _targets(row, case) for row, case in zip(test_rows, test_readiness_cases)
        ]
        test_target_vectors = {
            name: np.asarray(
                [values[name] for values in test_target_rows],
                dtype=float,
            )
            for name in ALL_TARGETS
        }
        test_predictions = {name: final_models[name].predict(test_matrix) for name in ALL_TARGETS}
        test_actions = _gate_actions(
            test_predictions,
            hard_brake_limit=hard_brake_limit,
            low_ttc_limit=low_ttc_limit,
            queue_reduction_min=float(args.queue_reduction_min),
            delay_reduction_min=float(args.delay_reduction_min),
            discharge_gain_min=float(args.discharge_gain_min),
            safety_margin=float(args.safety_margin),
        )
        test_deployable_actions = [
            int(case["selected_action"] == 1) for case in test_readiness_cases
        ]
        test_policies = {
            "always_off": _policy_summary(
                test_rows,
                test_readiness_cases,
                [0] * len(test_rows),
            ),
            "always_soft": _policy_summary(
                test_rows,
                test_readiness_cases,
                [1] * len(test_rows),
            ),
            "deployable_label_oracle": _policy_summary(
                test_rows,
                test_readiness_cases,
                test_deployable_actions,
            ),
            "two_head_gate_locked": _policy_summary(
                test_rows,
                test_readiness_cases,
                test_actions.tolist(),
            ),
        }
        locked_gate = test_policies["two_head_gate_locked"]
        locked_off = test_policies["always_off"]
        locked_improves_off = (
            locked_gate["mean_queue_area_vehicle_s"] < locked_off["mean_queue_area_vehicle_s"]
            and locked_gate["mean_incremental_delay_s"] <= locked_off["mean_incremental_delay_s"]
            and locked_gate["mean_merge_discharge_count"]
            >= locked_off["mean_merge_discharge_count"]
        )
        locked_status = (
            "passes_prespecified_policy_criteria"
            if locked_improves_off and locked_gate["unsafe_activation_count"] == 0
            else "does_not_pass_prespecified_policy_criteria"
        )
        test_records = []
        for index, (row, case, action) in enumerate(
            zip(test_rows, test_readiness_cases, test_actions)
        ):
            record = {
                "case_id": str(row["case_id"]),
                "seed": int(row["seed"]),
                "decision_step": int(row["decision_step"]),
                "actual_deployable_soft": int(case["selected_action"] == 1),
                "predicted_action": int(action),
            }
            for target_name in ALL_TARGETS:
                record[f"actual_{target_name}"] = float(test_target_vectors[target_name][index])
                record[f"predicted_{target_name}"] = float(test_predictions[target_name][index])
            test_records.append(record)
        _write_predictions(args.test_predictions_output.resolve(), test_records)
        report["locked_test"] = {
            "status": locked_status,
            "dataset": str(args.test_input.resolve()),
            "readiness_report": str(args.test_readiness.resolve()),
            "case_count": len(test_rows),
            "seed_group_count": int(len(np.unique(test_groups))),
            "development_seed_overlap": overlap,
            "model_or_rule_selection_using_locked_test": False,
            "target_metrics": {
                name: regression_metrics(test_target_vectors[name], test_predictions[name])
                for name in ALL_TARGETS
            },
            "policies": test_policies,
            "prediction_file": str(args.test_predictions_output.resolve()),
        }
        print(
            f"locked_test: treatments={locked_gate['treatment_count']}, "
            f"beneficial={locked_gate['traffic_beneficial_activation_count']}, "
            f"unsafe={locked_gate['unsafe_activation_count']}, "
            f"status={locked_status}",
            flush=True,
        )
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_predictions(args.predictions_output.resolve(), records)
    print(
        f"two_head_gate: treatments={gate_policy['treatment_count']}, "
        f"beneficial={gate_policy['traffic_beneficial_activation_count']}, "
        f"unsafe={gate_policy['unsafe_activation_count']}, status={status}",
        flush=True,
    )
    print(f"wrote {args.output.resolve()}", flush=True)
    print(f"wrote {args.predictions_output.resolve()}", flush=True)
    if args.test_input is not None:
        print(
            f"wrote {args.test_predictions_output.resolve()}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
