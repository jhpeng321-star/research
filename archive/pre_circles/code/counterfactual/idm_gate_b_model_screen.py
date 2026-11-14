"""Bounded post-primary-failure diagnostics for the frozen IDM Gate-B data."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from counterfactual.actionability import dose_qualified
from counterfactual.escalation_v2_model import grouped_splits, probability_metrics
from counterfactual.idm_gate_labels import (
    IDMGateBMargins,
    classify_idm_gate_b_case,
)
from counterfactual.idm_predictor_contract import (
    FORBIDDEN_FEATURE_TOKENS,
    IDM_DECISION_FEATURES,
    IDM_PRIMARY_FEATURES,
    complete_group_id,
    validate_decision_features,
)
from counterfactual.idm_observation_v3_contract import (
    IDM_OBSERVATION_V3_FEATURES,
    feature_contract_manifest as observation_v3_contract_manifest,
    validate_observation_v3_features,
)
from counterfactual.idm_two_stage_gate import PreparedGateB, select_thresholds


SCREEN_CONTRACT_VERSION = "idm_v5_4_gate_b_observation_model_repair/1"
RANDOM_STATE = 20260831
EXPLORATORY_MIN_SAFETY_RATE = 0.80
EXPLORATORY_MIN_BENEFIT_PRECISION = 0.45
EXPLORATORY_MIN_ENRICHMENT = 0.15


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    feature_scope: str
    model_family: str

    @property
    def feature_names(self) -> tuple[str, ...]:
        if self.feature_scope == "primary":
            return IDM_PRIMARY_FEATURES
        if self.feature_scope == "expanded":
            return IDM_DECISION_FEATURES
        if self.feature_scope == "observation_v3":
            return IDM_OBSERVATION_V3_FEATURES
        raise ValueError(f"unknown feature scope {self.feature_scope!r}")


CANDIDATE_SPECS = (
    CandidateSpec("primary_logistic_reference", "primary", "regularized_logistic"),
    CandidateSpec("expanded_logistic", "expanded", "regularized_logistic"),
    CandidateSpec("primary_additive_spline", "primary", "additive_spline_logistic"),
    CandidateSpec("expanded_additive_spline", "expanded", "additive_spline_logistic"),
    CandidateSpec("primary_shallow_boosting", "primary", "shallow_gradient_boosting"),
    CandidateSpec("expanded_shallow_boosting", "expanded", "shallow_gradient_boosting"),
)


OBSERVATION_V3_CANDIDATE_SPECS = (
    CandidateSpec("v3_logistic", "observation_v3", "regularized_logistic"),
    CandidateSpec("v3_additive_spline", "observation_v3", "additive_spline_logistic"),
    CandidateSpec("v3_shallow_boosting", "observation_v3", "shallow_gradient_boosting"),
)


def screen_manifest() -> dict[str, Any]:
    return {
        "contract_version": SCREEN_CONTRACT_VERSION,
        "random_state": RANDOM_STATE,
        "grouped_fold_count": 5,
        "candidates": [
            {
                "name": spec.name,
                "feature_scope": spec.feature_scope,
                "feature_count": len(spec.feature_names),
                "model_family": spec.model_family,
            }
            for spec in CANDIDATE_SPECS
        ],
        "model_selection_role": "development_only_after_primary_failure",
        "labels_changed": False,
        "thresholds_changed": False,
        "metastability_tolerant_exploratory_gate": {
            "minimum_command_cases": 10,
            "minimum_command_seeds": 8,
            "minimum_safety_rate": EXPLORATORY_MIN_SAFETY_RATE,
            "minimum_joint_benefit_precision": EXPLORATORY_MIN_BENEFIT_PRECISION,
            "minimum_safety_enrichment": EXPLORATORY_MIN_ENRICHMENT,
            "minimum_benefit_enrichment": EXPLORATORY_MIN_ENRICHMENT,
            "maximum_excess_collision_or_teleport_cases": 0,
            "scientific_role": "exploratory_selective_enrichment_only",
        },
        "new_sumo_outcomes": False,
    }


def observation_v3_screen_manifest() -> dict[str, Any]:
    payload = screen_manifest()
    return {
        **payload,
        "contract_version": "idm_v5_4_gate_b_observation_v3_screen/1",
        "observation_contract": observation_v3_contract_manifest(),
        "candidates": [
            {
                "name": spec.name,
                "feature_scope": spec.feature_scope,
                "feature_count": len(spec.feature_names),
                "model_family": spec.model_family,
            }
            for spec in OBSERVATION_V3_CANDIDATE_SPECS
        ],
        "repair_iteration": 1,
        "further_observation_repairs_authorized": False,
    }


def prepare_gate_b_rows_with_features(
    rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str]
) -> PreparedGateB:
    qualified = [row for row in rows if dose_qualified(row)]
    if not qualified:
        raise ValueError("no dose-qualified Gate-B cases")
    names = tuple(feature_names)
    if names == IDM_OBSERVATION_V3_FEATURES:
        matrix = np.asarray(
            [validate_observation_v3_features(row) for row in qualified], dtype=float
        )
    else:
        matrix = np.asarray(
            [validate_decision_features(row, names) for row in qualified], dtype=float
        )
    labels = [row["safety_service_oracle"] for row in qualified]
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Gate-B feature matrix is non-finite")
    return PreparedGateB(
        rows=tuple(qualified),
        matrix=matrix,
        groups=np.asarray([complete_group_id(row) for row in qualified]),
        safety=np.asarray([int(value["safety_admissible"]) for value in labels], dtype=int),
        joint_benefit=np.asarray([int(value["wall_selected"]) for value in labels], dtype=int),
    )


def make_candidate_estimator(model_family: str, random_state: int = RANDOM_STATE):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer, StandardScaler

    if model_family == "regularized_logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1,
                class_weight="balanced",
                max_iter=5000,
                random_state=random_state,
                solver="liblinear",
            ),
        )
    if model_family == "additive_spline_logistic":
        return make_pipeline(
            VarianceThreshold(threshold=0.0),
            SplineTransformer(
                n_knots=3,
                degree=2,
                knots="quantile",
                include_bias=False,
            ),
            StandardScaler(),
            LogisticRegression(
                C=0.05,
                class_weight="balanced",
                max_iter=5000,
                random_state=random_state,
                solver="liblinear",
            ),
        )
    if model_family == "shallow_gradient_boosting":
        return make_pipeline(
            VarianceThreshold(threshold=0.0),
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=100,
                max_depth=2,
                min_samples_leaf=15,
                l2_regularization=1.0,
                class_weight="balanced",
                early_stopping=False,
                random_state=random_state,
            ),
        )
    raise ValueError(f"unknown Gate-B model family {model_family!r}")


def grouped_oof_candidate(
    prepared: PreparedGateB,
    estimator: Any,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Cross-fit both heads on one immutable grouped-fold assignment."""
    from sklearn.base import clone

    safety_prob = np.full(len(prepared.rows), np.nan, dtype=float)
    benefit_prob = np.full(len(prepared.rows), np.nan, dtype=float)
    details: list[dict[str, Any]] = []
    for fold_index, (train, test) in enumerate(folds):
        safety_model = clone(estimator)
        safety_model.fit(prepared.matrix[train], prepared.safety[train])
        safety_prob[test] = safety_model.predict_proba(prepared.matrix[test])[:, 1]

        safe_train = train[prepared.safety[train] == 1]
        benefit_labels = prepared.joint_benefit[safe_train]
        if len(np.unique(benefit_labels)) != 2:
            raise ValueError(f"fold {fold_index} safe training rows lack a benefit class")
        benefit_model = clone(estimator)
        benefit_model.fit(prepared.matrix[safe_train], benefit_labels)
        benefit_prob[test] = benefit_model.predict_proba(prepared.matrix[test])[:, 1]
        details.append(
            {
                "fold": fold_index,
                "train_case_count": int(len(train)),
                "test_case_count": int(len(test)),
                "train_seed_count": int(len(np.unique(prepared.groups[train]))),
                "test_seed_count": int(len(np.unique(prepared.groups[test]))),
                "safe_train_case_count": int(len(safe_train)),
                "test_groups": sorted(str(value) for value in np.unique(prepared.groups[test])),
            }
        )
    if not np.all(np.isfinite(safety_prob)) or not np.all(np.isfinite(benefit_prob)):
        raise AssertionError("incomplete grouped OOF candidate probabilities")
    return safety_prob, benefit_prob, details


def _fixed_policy_metrics(
    prepared: PreparedGateB,
    safety_prob: np.ndarray,
    benefit_prob: np.ndarray,
    thresholds: Mapping[str, Any],
    include: np.ndarray | None = None,
) -> dict[str, Any]:
    mask = np.ones(len(prepared.rows), dtype=bool) if include is None else include.astype(bool)
    command = mask & (safety_prob >= float(thresholds["safety_threshold"])) & (
        benefit_prob >= float(thresholds["benefit_threshold"])
    )
    count = int(command.sum())
    safe = int(prepared.safety[command].sum())
    benefit = int(prepared.joint_benefit[command].sum())
    groups = set(str(value) for value in prepared.groups[command])
    decision_steps = sorted({int(row["decision_step"]) for row, take in zip(prepared.rows, command) if take})
    return {
        "command_case_count": count,
        "command_seed_count": len(groups),
        "command_decision_steps": decision_steps,
        "safe_command_count": safe,
        "unsafe_command_count": count - safe,
        "joint_beneficial_command_count": benefit,
        "selected_safety_rate": safe / count if count else 0.0,
        "joint_benefit_precision": benefit / count if count else 0.0,
    }


def _candidate_report(
    spec: CandidateSpec,
    prepared: PreparedGateB,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    estimator = make_candidate_estimator(spec.model_family)
    safety_prob, benefit_prob, fold_details = grouped_oof_candidate(
        prepared, estimator, folds
    )
    selected, sweep = select_thresholds(prepared, safety_prob, benefit_prob)
    exploratory_selected, exploratory_sweep = select_exploratory_thresholds(
        prepared, safety_prob, benefit_prob, sweep
    )
    safe = prepared.safety == 1
    return (
        {
            "name": spec.name,
            "feature_scope": spec.feature_scope,
            "feature_count": len(spec.feature_names),
            "model_family": spec.model_family,
            "safety_probability_metrics": probability_metrics(prepared.safety, safety_prob),
            "conditional_benefit_probability_metrics": probability_metrics(
                prepared.joint_benefit[safe], benefit_prob[safe]
            ),
            "selected_thresholds": selected,
            "threshold_selection_passed": selected is not None,
            "threshold_sweep": sweep,
            "exploratory_selected_thresholds": exploratory_selected,
            "exploratory_threshold_selection_passed": exploratory_selected is not None,
            "exploratory_threshold_sweep": exploratory_sweep,
            "folds": fold_details,
        },
        safety_prob,
        benefit_prob,
    )


def _catastrophic_excess_count(
    prepared: PreparedGateB, command: np.ndarray
) -> int:
    count = 0
    for row, selected in zip(prepared.rows, command):
        if not selected:
            continue
        effects = (row.get("safety_service_oracle") or {}).get("effects") or {}
        collision = int(effects.get("collision_excess_vehicle_count", 0))
        teleport = int(effects.get("teleport_excess_vehicle_count", 0))
        count += int(collision > 0 or teleport > 0)
    return count


def select_exploratory_thresholds(
    prepared: PreparedGateB,
    safety_prob: np.ndarray,
    benefit_prob: np.ndarray,
    primary_sweep: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Apply the separately frozen metastability-tolerant enrichment gate."""
    rows = []
    base_safety = float(np.mean(prepared.safety))
    base_benefit = float(np.mean(prepared.joint_benefit))
    sweep = list(primary_sweep or select_thresholds(prepared, safety_prob, benefit_prob)[1])
    for original in sweep:
        row = dict(original)
        command = (safety_prob >= float(row["safety_threshold"])) & (
            benefit_prob >= float(row["benefit_threshold"])
        )
        catastrophic = _catastrophic_excess_count(prepared, command)
        safety_enrichment = float(row["selected_safety_rate"]) - base_safety
        benefit_enrichment = float(row["joint_benefit_precision"]) - base_benefit
        row.update(
            {
                "unconditional_safety_rate": base_safety,
                "unconditional_joint_benefit_rate": base_benefit,
                "safety_enrichment": safety_enrichment,
                "joint_benefit_enrichment": benefit_enrichment,
                "excess_collision_or_teleport_case_count": catastrophic,
                "passes_exploratory_metastability_gate": bool(
                    int(row["command_case_count"]) >= 10
                    and int(row["command_seed_count"]) >= 8
                    and float(row["selected_safety_rate"])
                    >= EXPLORATORY_MIN_SAFETY_RATE
                    and float(row["joint_benefit_precision"])
                    >= EXPLORATORY_MIN_BENEFIT_PRECISION
                    and safety_enrichment >= EXPLORATORY_MIN_ENRICHMENT
                    and benefit_enrichment >= EXPLORATORY_MIN_ENRICHMENT
                    and catastrophic == 0
                ),
            }
        )
        rows.append(row)
    passing = [row for row in rows if row["passes_exploratory_metastability_gate"]]
    if not passing:
        return None, rows
    passing.sort(
        key=lambda row: (
            row["joint_beneficial_command_count"],
            row["joint_benefit_precision"],
            row["selected_safety_rate"],
            row["command_seed_count"],
            row["safety_threshold"],
            row["benefit_threshold"],
        ),
        reverse=True,
    )
    return passing[0], rows


def _univariate_auc(values: np.ndarray, labels: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(values)) < 2 or len(np.unique(labels)) < 2:
        return None
    auc = float(roc_auc_score(labels, values))
    return max(auc, 1.0 - auc)


def feature_diagnostics(prepared: PreparedGateB, feature_names: Sequence[str]) -> dict[str, Any]:
    from scipy.stats import rankdata

    safe = prepared.safety == 1
    rows = []
    for index, name in enumerate(feature_names):
        values = prepared.matrix[:, index]
        rows.append(
            {
                "name": name,
                "unique_count": int(len(np.unique(values))),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "mean": float(np.mean(values)),
                "standard_deviation": float(np.std(values)),
                "safety_direction_invariant_auc": _univariate_auc(values, prepared.safety),
                "conditional_benefit_direction_invariant_auc": _univariate_auc(
                    values[safe], prepared.joint_benefit[safe]
                ),
            }
        )
    variable = [index for index in range(prepared.matrix.shape[1]) if len(np.unique(prepared.matrix[:, index])) > 1]
    redundancy = []
    if len(variable) > 1:
        ranked = np.column_stack([rankdata(prepared.matrix[:, index]) for index in variable])
        correlations = np.corrcoef(ranked, rowvar=False)
        for left in range(len(variable)):
            for right in range(left + 1, len(variable)):
                value = float(correlations[left, right])
                if math.isfinite(value) and abs(value) >= 0.98:
                    redundancy.append(
                        {
                            "left": str(feature_names[variable[left]]),
                            "right": str(feature_names[variable[right]]),
                            "spearman": value,
                        }
                    )
    return {
        "feature_count": len(feature_names),
        "finite_value_count": int(np.isfinite(prepared.matrix).sum()),
        "missing_or_nonfinite_value_count": int((~np.isfinite(prepared.matrix)).sum()),
        "constant_feature_count": sum(row["unique_count"] == 1 for row in rows),
        "forbidden_name_count": sum(
            any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)
            for name in feature_names
        ),
        "features": rows,
        "absolute_spearman_at_least_0_98": redundancy,
    }


def cohort_diagnostics(prepared: PreparedGateB) -> dict[str, Any]:
    by_step = []
    for step in sorted({int(row["decision_step"]) for row in prepared.rows}):
        mask = np.asarray([int(row["decision_step"]) == step for row in prepared.rows])
        by_step.append(
            {
                "decision_step": step,
                "case_count": int(mask.sum()),
                "seed_count": int(len(np.unique(prepared.groups[mask]))),
                "safety_rate": float(np.mean(prepared.safety[mask])),
                "joint_benefit_rate": float(np.mean(prepared.joint_benefit[mask])),
            }
        )
    mixed_safety = 0
    mixed_benefit = 0
    multiwindow = 0
    for group in np.unique(prepared.groups):
        mask = prepared.groups == group
        if int(mask.sum()) < 2:
            continue
        multiwindow += 1
        mixed_safety += int(len(np.unique(prepared.safety[mask])) > 1)
        mixed_benefit += int(len(np.unique(prepared.joint_benefit[mask])) > 1)
    mechanism_counts: dict[str, int] = {}
    emergency_count = 0
    for row in prepared.rows:
        dose = row.get("treatment_dose") or {}
        reason = str(dose.get("completion_reason", "unknown"))
        mechanism_counts[reason] = mechanism_counts.get(reason, 0) + 1
        emergency_count += int(bool(dose.get("emergency_taper_zero_handoff_steps")))
    return {
        "case_count": len(prepared.rows),
        "seed_count": int(len(np.unique(prepared.groups))),
        "safe_case_count": int(prepared.safety.sum()),
        "unsafe_case_count": int(len(prepared.rows) - prepared.safety.sum()),
        "joint_beneficial_case_count": int(prepared.joint_benefit.sum()),
        "joint_nonbeneficial_case_count": int(len(prepared.rows) - prepared.joint_benefit.sum()),
        "by_decision_step": by_step,
        "multiwindow_seed_count": multiwindow,
        "mixed_safety_multiwindow_seed_count": mixed_safety,
        "mixed_benefit_multiwindow_seed_count": mixed_benefit,
        "postdecision_mechanism_diagnostic_only": {
            "completion_reason_counts": mechanism_counts,
            "emergency_zero_handoff_case_count": emergency_count,
        },
    }


def selected_candidate_robustness(
    spec: CandidateSpec,
    prepared: PreparedGateB,
    safety_prob: np.ndarray,
    benefit_prob: np.ndarray,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    from sklearn.base import clone

    aggregate = _fixed_policy_metrics(prepared, safety_prob, benefit_prob, thresholds)
    command = (safety_prob >= float(thresholds["safety_threshold"])) & (
        benefit_prob >= float(thresholds["benefit_threshold"])
    )
    leave_one_seed = []
    for group in sorted(set(str(value) for value in prepared.groups[command])):
        include = np.asarray([str(value) != group for value in prepared.groups])
        leave_one_seed.append({"excluded_group": group, **_fixed_policy_metrics(
            prepared, safety_prob, benefit_prob, thresholds, include
        )})

    held_safety = np.full(len(prepared.rows), np.nan, dtype=float)
    held_benefit = np.full(len(prepared.rows), np.nan, dtype=float)
    held_step_details = []
    estimator = make_candidate_estimator(spec.model_family)
    decision_values = np.asarray([int(row["decision_step"]) for row in prepared.rows])
    for step in sorted(np.unique(decision_values)):
        test = np.flatnonzero(decision_values == step)
        train = np.flatnonzero(decision_values != step)
        safety_model = clone(estimator)
        safety_model.fit(prepared.matrix[train], prepared.safety[train])
        held_safety[test] = safety_model.predict_proba(prepared.matrix[test])[:, 1]
        safe_train = train[prepared.safety[train] == 1]
        labels = prepared.joint_benefit[safe_train]
        if len(np.unique(labels)) != 2:
            raise ValueError(f"decision holdout {step} lacks a benefit training class")
        benefit_model = clone(estimator)
        benefit_model.fit(prepared.matrix[safe_train], labels)
        held_benefit[test] = benefit_model.predict_proba(prepared.matrix[test])[:, 1]
        include = decision_values == step
        held_step_details.append(
            {"held_decision_step": int(step), **_fixed_policy_metrics(
                prepared, held_safety, held_benefit, thresholds, include
            )}
        )
    held_aggregate = _fixed_policy_metrics(
        prepared, held_safety, held_benefit, thresholds
    )

    stricter_safety = IDMGateBMargins(
        max_hard_brake_excess=1,
        max_hard_brake_excess_fraction=0.05,
        max_low_ttc_excess_steps=1,
    )
    stricter_benefit = IDMGateBMargins(
        min_hard_brake_onset_reduction=3,
        min_hard_brake_onset_reduction_fraction=0.15,
        min_hard_brake_exposure_reduction=5,
        min_hard_brake_exposure_reduction_fraction=0.15,
        min_low_ttc_reduction_steps=3,
        min_discharge_gain=4,
    )
    strict_safety_labels = np.asarray(
        [int(classify_idm_gate_b_case(row, stricter_safety)["safety_admissible"]) for row in prepared.rows]
    )
    strict_benefit_labels = np.asarray(
        [int(classify_idm_gate_b_case(row, stricter_benefit)["wall_selected"]) for row in prepared.rows]
    )
    command_count = int(command.sum())
    sensitivity = {
        "tighter_safety_rate": (
            float(strict_safety_labels[command].mean()) if command_count else 0.0
        ),
        "stricter_joint_benefit_precision": (
            float(strict_benefit_labels[command].mean()) if command_count else 0.0
        ),
    }
    seed_counts = [int(np.sum(prepared.groups[command] == group)) for group in np.unique(prepared.groups[command])]
    return {
        "aggregate_grouped_oof_fixed_thresholds": aggregate,
        "leave_one_selected_seed_out": leave_one_seed,
        "minimum_leave_one_seed_safety_rate": min(
            (row["selected_safety_rate"] for row in leave_one_seed), default=0.0
        ),
        "minimum_leave_one_seed_benefit_precision": min(
            (row["joint_benefit_precision"] for row in leave_one_seed), default=0.0
        ),
        "largest_single_seed_command_fraction": (
            max(seed_counts) / command_count if command_count and seed_counts else 0.0
        ),
        "leave_one_decision_time_out_fixed_thresholds": {
            "aggregate": held_aggregate,
            "by_held_decision_step": held_step_details,
        },
        "label_sensitivity_diagnostic_only": sensitivity,
    }


def fit_full_candidate(prepared: PreparedGateB, spec: CandidateSpec) -> tuple[Any, Any]:
    from sklearn.base import clone

    estimator = make_candidate_estimator(spec.model_family)
    safety_model = clone(estimator).fit(prepared.matrix, prepared.safety)
    safe = prepared.safety == 1
    benefit_model = clone(estimator).fit(
        prepared.matrix[safe], prepared.joint_benefit[safe]
    )
    return safety_model, benefit_model


def operational_admitted_population_audit(
    rows: Sequence[Mapping[str, Any]],
    spec: CandidateSpec,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate a fixed policy on all Gate-F admissions, retaining dose failures."""
    from sklearn.base import clone

    admitted = [row for row in rows if bool(row.get("predecision_supervisor", {}).get("admitted"))]
    qualified = prepare_gate_b_rows_with_features(admitted, spec.feature_names)

    def vector(row: Mapping[str, Any]) -> tuple[float, ...]:
        if spec.feature_scope == "observation_v3":
            return validate_observation_v3_features(row)
        return validate_decision_features(row, spec.feature_names)

    matrix = np.asarray([vector(row) for row in admitted], dtype=float)
    groups = np.asarray([complete_group_id(row) for row in admitted])
    folds = grouped_splits(
        qualified.safety,
        qualified.groups,
        n_splits=5,
        random_state=RANDOM_STATE,
    )
    safety_prob = np.full(len(admitted), np.nan, dtype=float)
    benefit_prob = np.full(len(admitted), np.nan, dtype=float)
    estimator = make_candidate_estimator(spec.model_family)
    for train, test in folds:
        test_groups = set(str(value) for value in qualified.groups[test])
        admitted_test = np.asarray(
            [index for index, group in enumerate(groups) if str(group) in test_groups],
            dtype=int,
        )
        safety_model = clone(estimator).fit(
            qualified.matrix[train], qualified.safety[train]
        )
        safety_prob[admitted_test] = safety_model.predict_proba(matrix[admitted_test])[:, 1]
        safe_train = train[qualified.safety[train] == 1]
        benefit_model = clone(estimator).fit(
            qualified.matrix[safe_train], qualified.joint_benefit[safe_train]
        )
        benefit_prob[admitted_test] = benefit_model.predict_proba(matrix[admitted_test])[:, 1]

    missing = np.flatnonzero(~np.isfinite(safety_prob))
    if len(missing):
        # These groups contain only failed-dose admissions and therefore occur
        # in no qualified fold. Fitting on all qualified rows is still group
        # disjoint from them and provides a leakage-safe development estimate.
        safety_model, benefit_model = fit_full_candidate(qualified, spec)
        safety_prob[missing] = safety_model.predict_proba(matrix[missing])[:, 1]
        benefit_prob[missing] = benefit_model.predict_proba(matrix[missing])[:, 1]
    if not np.all(np.isfinite(safety_prob)) or not np.all(np.isfinite(benefit_prob)):
        raise AssertionError("incomplete admitted-population probabilities")

    command = (safety_prob >= float(thresholds["safety_threshold"])) & (
        benefit_prob >= float(thresholds["benefit_threshold"])
    )
    safety = np.asarray(
        [int(row["safety_service_oracle"]["safety_admissible"]) for row in admitted]
    )
    benefit = np.asarray(
        [
            int(dose_qualified(row) and row["safety_service_oracle"]["wall_selected"])
            for row in admitted
        ]
    )
    failed_dose = np.asarray([not dose_qualified(row) for row in admitted])
    catastrophic = np.asarray(
        [
            int(
                int(row["safety_service_oracle"]["effects"]["collision_excess_vehicle_count"])
                > 0
                or int(
                    row["safety_service_oracle"]["effects"][
                        "teleport_excess_vehicle_count"
                    ]
                )
                > 0
            )
            for row in admitted
        ]
    )
    count = int(command.sum())
    return {
        "admitted_case_count": len(admitted),
        "dose_qualified_case_count": len(qualified.rows),
        "dose_failure_case_count": len(admitted) - len(qualified.rows),
        "dose_failure_only_case_count_predicted_by_all_qualified_model": int(len(missing)),
        "command_case_count": count,
        "command_seed_count": len(set(str(value) for value in groups[command])),
        "safe_command_count": int(safety[command].sum()),
        "selected_safety_rate": float(np.mean(safety[command])) if count else 0.0,
        "joint_beneficial_command_count": int(benefit[command].sum()),
        "joint_benefit_precision": float(np.mean(benefit[command])) if count else 0.0,
        "selected_failed_dose_case_count": int(np.sum(command & failed_dose)),
        "selected_excess_collision_or_teleport_case_count": int(
            catastrophic[command].sum()
        ),
        "command_case_ids": [
            str(row["case_id"]) for row, selected in zip(admitted, command) if selected
        ],
    }


def run_model_screen(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    prepared_by_scope = {
        "primary": prepare_gate_b_rows_with_features(rows, IDM_PRIMARY_FEATURES),
        "expanded": prepare_gate_b_rows_with_features(rows, IDM_DECISION_FEATURES),
    }
    primary = prepared_by_scope["primary"]
    expanded = prepared_by_scope["expanded"]
    primary_ids = [str(row["case_id"]) for row in primary.rows]
    if primary_ids != [str(row["case_id"]) for row in expanded.rows]:
        raise AssertionError("feature scopes do not contain identical cases")
    if not np.array_equal(primary.safety, expanded.safety) or not np.array_equal(
        primary.joint_benefit, expanded.joint_benefit
    ):
        raise AssertionError("feature scopes do not contain identical labels")
    folds = grouped_splits(
        primary.safety,
        primary.groups,
        n_splits=5,
        random_state=RANDOM_STATE,
    )
    reports = []
    predictions: dict[str, Any] = {}
    for order, spec in enumerate(CANDIDATE_SPECS):
        prepared = prepared_by_scope[spec.feature_scope]
        report, safety_prob, benefit_prob = _candidate_report(spec, prepared, folds)
        report["candidate_order"] = order
        reports.append(report)
        predictions[spec.name] = {
            "safety_probability_oof": safety_prob.tolist(),
            "conditional_benefit_probability_oof": benefit_prob.tolist(),
        }

    primary_passing = [report for report in reports if report["threshold_selection_passed"]]
    exploratory_passing = [
        report for report in reports if report["exploratory_threshold_selection_passed"]
    ]
    selection_role = None
    passing = primary_passing
    threshold_key = "selected_thresholds"
    if not passing:
        passing = exploratory_passing
        threshold_key = "exploratory_selected_thresholds"
        selection_role = "exploratory_selective_enrichment"
    elif passing:
        selection_role = "primary_development_policy_gate"
    selected_report = None
    selected_policy_thresholds = None
    selected_robustness = None
    if passing:
        selected_report = max(
            passing,
            key=lambda report: (
                report[threshold_key]["joint_beneficial_command_count"],
                report[threshold_key]["joint_benefit_precision"],
                report[threshold_key]["selected_safety_rate"],
                report[threshold_key]["command_seed_count"],
                -int(report["candidate_order"]),
            ),
        )
        selected_policy_thresholds = selected_report[threshold_key]
        spec = next(value for value in CANDIDATE_SPECS if value.name == selected_report["name"])
        prepared = prepared_by_scope[spec.feature_scope]
        selected_robustness = selected_candidate_robustness(
            spec,
            prepared,
            np.asarray(predictions[spec.name]["safety_probability_oof"]),
            np.asarray(predictions[spec.name]["conditional_benefit_probability_oof"]),
            selected_policy_thresholds,
        )
        operational_audit = operational_admitted_population_audit(
            rows, spec, selected_policy_thresholds
        )
    else:
        operational_audit = None

    report = {
        "schema": "idm_v5_4_gate_b_observation_model_repair_report/1",
        "screen_contract": screen_manifest(),
        "cohort": cohort_diagnostics(primary),
        "feature_diagnostics": {
            "primary": feature_diagnostics(primary, IDM_PRIMARY_FEATURES),
            "expanded": feature_diagnostics(expanded, IDM_DECISION_FEATURES),
        },
        "candidate_reports": reports,
        "selected_candidate": selected_report,
        "selected_candidate_role": selection_role,
        "selected_policy_thresholds": selected_policy_thresholds,
        "selected_candidate_robustness": selected_robustness,
        "deployment_intent_admitted_population_audit": operational_audit,
        "aggregate_policy_gate_passed": bool(primary_passing),
        "exploratory_policy_gate_passed": bool(exploratory_passing),
        "validation_protocol_freeze_candidate_available": selected_report is not None,
        "guardrail": (
            "Post-primary-failure grouped development model selection only; "
            "no calibration, validation or deployability claim."
        ),
    }
    prediction_payload = {
        "schema": "idm_v5_4_gate_b_observation_model_repair_oof/1",
        "case_ids": primary_ids,
        "seed_groups": [str(value) for value in primary.groups],
        "observed_safety": primary.safety.tolist(),
        "observed_joint_benefit": primary.joint_benefit.tolist(),
        "candidates": predictions,
    }
    return report, prediction_payload


def run_observation_v3_screen(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the single frozen safety-history observation repair."""
    prepared = prepare_gate_b_rows_with_features(rows, IDM_OBSERVATION_V3_FEATURES)
    folds = grouped_splits(
        prepared.safety,
        prepared.groups,
        n_splits=5,
        random_state=RANDOM_STATE,
    )
    reports = []
    predictions: dict[str, Any] = {}
    for order, spec in enumerate(OBSERVATION_V3_CANDIDATE_SPECS):
        report, safety_prob, benefit_prob = _candidate_report(spec, prepared, folds)
        report["candidate_order"] = order
        reports.append(report)
        predictions[spec.name] = {
            "safety_probability_oof": safety_prob.tolist(),
            "conditional_benefit_probability_oof": benefit_prob.tolist(),
        }

    primary_passing = [report for report in reports if report["threshold_selection_passed"]]
    exploratory_passing = [
        report for report in reports if report["exploratory_threshold_selection_passed"]
    ]
    passing = primary_passing or exploratory_passing
    threshold_key = (
        "selected_thresholds" if primary_passing else "exploratory_selected_thresholds"
    )
    selection_role = (
        "primary_development_policy_gate"
        if primary_passing
        else ("exploratory_selective_enrichment" if exploratory_passing else None)
    )
    selected_report = None
    selected_thresholds = None
    robustness = None
    if passing:
        selected_report = max(
            passing,
            key=lambda report: (
                report[threshold_key]["joint_beneficial_command_count"],
                report[threshold_key]["joint_benefit_precision"],
                report[threshold_key]["selected_safety_rate"],
                report[threshold_key]["command_seed_count"],
                -int(report["candidate_order"]),
            ),
        )
        selected_thresholds = selected_report[threshold_key]
        spec = next(
            value
            for value in OBSERVATION_V3_CANDIDATE_SPECS
            if value.name == selected_report["name"]
        )
        robustness = selected_candidate_robustness(
            spec,
            prepared,
            np.asarray(predictions[spec.name]["safety_probability_oof"]),
            np.asarray(
                predictions[spec.name]["conditional_benefit_probability_oof"]
            ),
            selected_thresholds,
        )
        operational_audit = operational_admitted_population_audit(
            rows, spec, selected_thresholds
        )
    else:
        operational_audit = None

    report = {
        "schema": "idm_v5_4_gate_b_observation_v3_repair_report/1",
        "screen_contract": observation_v3_screen_manifest(),
        "cohort": cohort_diagnostics(prepared),
        "feature_diagnostics": feature_diagnostics(
            prepared, IDM_OBSERVATION_V3_FEATURES
        ),
        "candidate_reports": reports,
        "selected_candidate": selected_report,
        "selected_candidate_role": selection_role,
        "selected_policy_thresholds": selected_thresholds,
        "selected_candidate_robustness": robustness,
        "deployment_intent_admitted_population_audit": operational_audit,
        "aggregate_policy_gate_passed": bool(primary_passing),
        "exploratory_policy_gate_passed": bool(exploratory_passing),
        "validation_protocol_freeze_candidate_available": selected_report is not None,
        "guardrail": (
            "Single post-primary-failure observation repair on grouped development "
            "data only; no validation or deployability claim."
        ),
    }
    prediction_payload = {
        "schema": "idm_v5_4_gate_b_observation_v3_repair_oof/1",
        "case_ids": [str(row["case_id"]) for row in prepared.rows],
        "seed_groups": [str(value) for value in prepared.groups],
        "observed_safety": prepared.safety.tolist(),
        "observed_joint_benefit": prepared.joint_benefit.tolist(),
        "candidates": predictions,
    }
    return report, prediction_payload


__all__ = [
    "CANDIDATE_SPECS",
    "OBSERVATION_V3_CANDIDATE_SPECS",
    "CandidateSpec",
    "SCREEN_CONTRACT_VERSION",
    "feature_diagnostics",
    "fit_full_candidate",
    "grouped_oof_candidate",
    "make_candidate_estimator",
    "prepare_gate_b_rows_with_features",
    "observation_v3_screen_manifest",
    "operational_admitted_population_audit",
    "run_observation_v3_screen",
    "run_model_screen",
    "screen_manifest",
    "select_exploratory_thresholds",
]
