"""No-retuning prospective evaluation for the frozen exploratory IDM Gate B."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from counterfactual.actionability import dose_qualified
from counterfactual.idm_predictor_contract import validate_decision_features
from experiments.idm.wall_v5_4_predictor_validation_contract import (
    BENEFIT_THRESHOLD,
    SAFETY_THRESHOLD,
    prospective_gate,
)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float]:
    if total <= 0:
        return {"lower": 0.0, "upper": 1.0}
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    return {"lower": max(0.0, center - half), "upper": min(1.0, center + half)}


def score_admitted_rows(
    rows: Sequence[Mapping[str, Any]], bundle: Mapping[str, Any]
) -> list[dict[str, Any]]:
    feature_names = tuple(str(value) for value in bundle["feature_names"])
    safety_model = bundle["safety_model"]
    benefit_model = bundle["conditional_benefit_model"]
    admitted = [
        row for row in rows
        if bool(row.get("predecision_supervisor", {}).get("admitted"))
    ]
    if not admitted:
        return []
    matrix = np.asarray(
        [validate_decision_features(row, feature_names) for row in admitted], dtype=float
    )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("prospective feature matrix is non-finite")
    safety_probability = safety_model.predict_proba(matrix)[:, 1]
    benefit_probability = benefit_model.predict_proba(matrix)[:, 1]
    if not np.all(np.isfinite(safety_probability)) or not np.all(
        np.isfinite(benefit_probability)
    ):
        raise ValueError("prospective probabilities are non-finite")
    if np.any((safety_probability < 0.0) | (safety_probability > 1.0)) or np.any(
        (benefit_probability < 0.0) | (benefit_probability > 1.0)
    ):
        raise ValueError("prospective probabilities are outside [0, 1]")
    records = []
    for row, safety, benefit in zip(admitted, safety_probability, benefit_probability):
        selected = bool(safety >= SAFETY_THRESHOLD and benefit >= BENEFIT_THRESHOLD)
        records.append(
            {
                "case_id": str(row["case_id"]),
                "seed": int(row["seed"]),
                "decision_step": int(row["decision_step"]),
                "safety_probability": float(safety),
                "conditional_joint_benefit_probability": float(benefit),
                "selected": selected,
            }
        )
    return records


def _effect_summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    if not rows:
        return {}
    names = tuple(sorted(rows[0]["safety_service_oracle"]["effects"]))
    return {
        name: {
            "count": len(rows),
            "sum": float(np.sum(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
        for name in names
        for values in [[
            float(row["safety_service_oracle"]["effects"][name]) for row in rows
        ]]
    }


def _rate(successes: int, total: int) -> dict[str, Any]:
    return {
        "numerator": successes,
        "denominator": total,
        "rate": successes / total if total else 0.0,
        "wilson_95": wilson_interval(successes, total),
    }


def probability_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    admitted = {
        str(row["case_id"]): row for row in rows
        if bool(row.get("predecision_supervisor", {}).get("admitted"))
    }

    def metrics(labels: list[int], probabilities: list[float]) -> dict[str, Any]:
        if len(set(labels)) < 2:
            return {
                "available": False,
                "reason": "validation target contains one class",
                "case_count": len(labels),
            }
        return {
            "available": True,
            "case_count": len(labels),
            "roc_auc": float(roc_auc_score(labels, probabilities)),
            "average_precision": float(average_precision_score(labels, probabilities)),
            "brier_score": float(brier_score_loss(labels, probabilities)),
            "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        }

    safety_labels = []
    safety_probability = []
    benefit_labels = []
    benefit_probability = []
    for prediction in predictions:
        row = admitted[str(prediction["case_id"])]
        safe = bool(row["safety_service_oracle"]["safety_admissible"])
        safety_labels.append(int(safe))
        safety_probability.append(float(prediction["safety_probability"]))
        if safe and dose_qualified(row):
            benefit_labels.append(int(row["safety_service_oracle"]["wall_selected"]))
            benefit_probability.append(
                float(prediction["conditional_joint_benefit_probability"])
            )
    return {
        "safety_all_gate_f_admissions": metrics(safety_labels, safety_probability),
        "conditional_benefit_safe_dose_qualified_admissions": metrics(
            benefit_labels, benefit_probability
        ),
        "guardrail": "Descriptive discrimination only; no calibration claim.",
    }


def evaluate_fixed_policy(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    integrity_passed: bool,
) -> dict[str, Any]:
    admitted = [
        row for row in rows
        if bool(row.get("predecision_supervisor", {}).get("admitted"))
    ]
    by_id = {str(row["case_id"]): row for row in admitted}
    if len(by_id) != len(admitted):
        raise ValueError("duplicate admitted case IDs")
    prediction_by_id = {str(row["case_id"]): row for row in predictions}
    if set(prediction_by_id) != set(by_id):
        raise ValueError("predictions must cover every Gate-F admission exactly once")

    selected = [
        by_id[case_id]
        for case_id, prediction in prediction_by_id.items()
        if bool(prediction["selected"])
    ]
    safe = lambda row: bool(row["safety_service_oracle"]["safety_admissible"])
    beneficial = lambda row: bool(
        dose_qualified(row) and row["safety_service_oracle"]["wall_selected"]
    )
    catastrophic = lambda row: bool(
        int(row["safety_service_oracle"]["effects"]["collision_excess_vehicle_count"]) > 0
        or int(row["safety_service_oracle"]["effects"]["teleport_excess_vehicle_count"]) > 0
    )

    admitted_count = len(admitted)
    selected_count = len(selected)
    baseline_safe_count = sum(safe(row) for row in admitted)
    baseline_benefit_count = sum(beneficial(row) for row in admitted)
    selected_safe_count = sum(safe(row) for row in selected)
    selected_benefit_count = sum(beneficial(row) for row in selected)
    selected_dose_count = sum(dose_qualified(row) for row in selected)
    selected_catastrophic_count = sum(catastrophic(row) for row in selected)
    traffic = lambda row: bool(row["safety_service_oracle"]["traffic_noninferior"])
    meaningful = lambda row: bool(row["safety_service_oracle"]["meaningful_benefit"])
    baseline_traffic_count = sum(traffic(row) for row in admitted)
    baseline_meaningful_count = sum(meaningful(row) for row in admitted)
    selected_traffic_count = sum(traffic(row) for row in selected)
    selected_meaningful_count = sum(meaningful(row) for row in selected)

    baseline_safety = baseline_safe_count / admitted_count if admitted_count else 0.0
    baseline_benefit = baseline_benefit_count / admitted_count if admitted_count else 0.0
    selected_safety = selected_safe_count / selected_count if selected_count else 0.0
    selected_benefit = selected_benefit_count / selected_count if selected_count else 0.0
    selected_dose = selected_dose_count / selected_count if selected_count else 0.0
    safety_enrichment = selected_safety - baseline_safety
    benefit_enrichment = selected_benefit - baseline_benefit
    thresholds = prospective_gate()

    gates = {
        "paired_integrity_passed": bool(integrity_passed),
        "minimum_command_cases": selected_count >= thresholds["minimum_command_cases"],
        "minimum_command_seeds": len({int(row["seed"]) for row in selected})
        >= thresholds["minimum_command_seeds"],
        "selected_safety_rate": selected_safety
        >= thresholds["minimum_selected_safety_rate"],
        "joint_benefit_precision": selected_benefit
        >= thresholds["minimum_joint_benefit_precision"],
        "safety_enrichment": safety_enrichment >= thresholds["minimum_safety_enrichment"],
        "joint_benefit_enrichment": benefit_enrichment
        >= thresholds["minimum_joint_benefit_enrichment"],
        "selected_dose_success": selected_dose
        >= thresholds["minimum_selected_dose_success_rate"],
        "zero_selected_excess_collision_or_teleport": selected_catastrophic_count
        <= thresholds["maximum_selected_excess_collision_or_teleport_cases"],
    }

    selected_failures = []
    for row in selected:
        if safe(row) and beneficial(row) and dose_qualified(row) and not catastrophic(row):
            continue
        dose = row["treatment_dose"]
        selected_failures.append(
            {
                "case_id": str(row["case_id"]),
                "seed": int(row["seed"]),
                "decision_step": int(row["decision_step"]),
                "dose_qualified": bool(dose_qualified(row)),
                "realized_duration_fraction": float(dose["realized_duration_fraction"]),
                "completion_reason": str(dose.get("completion_reason", "unknown")),
                "safety_admissible": safe(row),
                "joint_beneficial": beneficial(row),
                "traffic_noninferior": bool(
                    row["safety_service_oracle"]["traffic_noninferior"]
                ),
                "effects": dict(row["safety_service_oracle"]["effects"]),
            }
        )

    by_step = []
    for decision_step in sorted({int(row["decision_step"]) for row in admitted}):
        step_rows = [row for row in selected if int(row["decision_step"]) == decision_step]
        by_step.append(
            {
                "decision_step": decision_step,
                "command_case_count": len(step_rows),
                "command_seed_count": len({int(row["seed"]) for row in step_rows}),
                "safe_command_count": sum(safe(row) for row in step_rows),
                "joint_beneficial_command_count": sum(beneficial(row) for row in step_rows),
            }
        )

    policy_population_rows = []
    selected_ids = {str(row["case_id"]) for row in selected}
    effect_names = tuple(sorted(admitted[0]["safety_service_oracle"]["effects"])) if admitted else ()
    for row in admitted:
        if str(row["case_id"]) in selected_ids:
            policy_population_rows.append(row)
            continue
        clone = dict(row)
        oracle = dict(row["safety_service_oracle"])
        oracle["effects"] = {name: 0.0 for name in effect_names}
        clone["safety_service_oracle"] = oracle
        policy_population_rows.append(clone)

    return {
        "schema": "idm_v5_4_gate_b_exploratory_prospective_validation_report/1",
        "scientific_role": "metastability_tolerant_exploratory_validation",
        "policy_thresholds": {
            "safety_threshold": SAFETY_THRESHOLD,
            "benefit_threshold": BENEFIT_THRESHOLD,
        },
        "prospective_gate": thresholds,
        "population": {
            "gate_f_admitted_case_count": admitted_count,
            "gate_f_admitted_seed_count": len({int(row["seed"]) for row in admitted}),
            "dose_qualified_case_count": sum(dose_qualified(row) for row in admitted),
            "dose_failure_case_count": sum(not dose_qualified(row) for row in admitted),
        },
        "feasible_always_wall_baseline": {
            "safe_case_count": baseline_safe_count,
            "safety_rate": baseline_safety,
            "safety_rate_wilson_95": wilson_interval(baseline_safe_count, admitted_count),
            "joint_beneficial_case_count": baseline_benefit_count,
            "joint_benefit_rate": baseline_benefit,
            "joint_benefit_rate_wilson_95": wilson_interval(
                baseline_benefit_count, admitted_count
            ),
            "traffic_noninferior": _rate(baseline_traffic_count, admitted_count),
            "meaningful_benefit": _rate(baseline_meaningful_count, admitted_count),
            "effect_summaries": _effect_summaries(admitted),
        },
        "selected_policy": {
            "command_case_count": selected_count,
            "command_seed_count": len({int(row["seed"]) for row in selected}),
            "command_rate": selected_count / admitted_count if admitted_count else 0.0,
            "safe_command_count": selected_safe_count,
            "selected_safety_rate": selected_safety,
            "selected_safety_rate_wilson_95": wilson_interval(
                selected_safe_count, selected_count
            ),
            "joint_beneficial_command_count": selected_benefit_count,
            "joint_benefit_precision": selected_benefit,
            "joint_benefit_precision_wilson_95": wilson_interval(
                selected_benefit_count, selected_count
            ),
            "dose_qualified_command_count": selected_dose_count,
            "selected_dose_success_rate": selected_dose,
            "selected_excess_collision_or_teleport_case_count": selected_catastrophic_count,
            "traffic_noninferior": _rate(selected_traffic_count, selected_count),
            "meaningful_benefit": _rate(selected_meaningful_count, selected_count),
            "safety_enrichment": safety_enrichment,
            "joint_benefit_enrichment": benefit_enrichment,
            "traffic_noninferiority_enrichment": (
                selected_traffic_count / selected_count if selected_count else 0.0
            ) - (baseline_traffic_count / admitted_count if admitted_count else 0.0),
            "effect_summaries": _effect_summaries(selected),
            "policy_population_effect_summaries_zero_for_abstentions": _effect_summaries(
                policy_population_rows
            ),
            "by_decision_step": by_step,
            "selected_case_ids": [str(row["case_id"]) for row in selected],
        },
        "gates": gates,
        "validation_passed": all(gates.values()),
        "final_test_authorized": all(gates.values()),
        "selected_failure_or_nonbenefit_cases": selected_failures,
        "guardrail": (
            "Exploratory selective enrichment only; Wilson intervals are mandatory "
            "diagnostics and probabilities are not claimed calibrated."
        ),
    }


__all__ = [
    "evaluate_fixed_policy",
    "probability_diagnostics",
    "score_admitted_rows",
    "wilson_interval",
]
