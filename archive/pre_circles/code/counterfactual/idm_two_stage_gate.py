"""Interpretable grouped models for the frozen IDM Gate-B decision."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from counterfactual.actionability import dose_qualified
from counterfactual.escalation_v2_model import grouped_splits, probability_metrics
from counterfactual.idm_predictor_contract import (
    IDM_PRIMARY_FEATURES,
    complete_group_id,
    validate_decision_features,
)


MODEL_CONTRACT_VERSION = "idm_v5_4_gate_b_regularized_logistic/1"
SAFETY_THRESHOLDS = tuple(float(x) for x in np.arange(0.50, 0.951, 0.05))
BENEFIT_THRESHOLDS = tuple(float(x) for x in np.arange(0.30, 0.901, 0.05))


@dataclass(frozen=True)
class PreparedGateB:
    rows: tuple[Mapping[str, Any], ...]
    matrix: np.ndarray
    groups: np.ndarray
    safety: np.ndarray
    joint_benefit: np.ndarray


def prepare_gate_b_rows(rows: Sequence[Mapping[str, Any]]) -> PreparedGateB:
    qualified = [row for row in rows if dose_qualified(row)]
    if not qualified:
        raise ValueError("no dose-qualified Gate-B cases")
    matrix = np.asarray(
        [validate_decision_features(row, IDM_PRIMARY_FEATURES) for row in qualified],
        dtype=float,
    )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Gate-B feature matrix is non-finite")
    labels = [row["safety_service_oracle"] for row in qualified]
    return PreparedGateB(
        rows=tuple(qualified),
        matrix=matrix,
        groups=np.asarray([complete_group_id(row) for row in qualified]),
        safety=np.asarray([int(value["safety_admissible"]) for value in labels], dtype=int),
        joint_benefit=np.asarray([int(value["wall_selected"]) for value in labels], dtype=int),
    )


def make_logistic(random_state: int = 20260831):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

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


def grouped_oof_gate_b(
    prepared: PreparedGateB,
    *,
    n_splits: int = 5,
    random_state: int = 20260831,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Cross-fit safety and conditional joint-benefit probabilities by seed."""

    from sklearn.base import clone

    safety_prob = np.full(len(prepared.rows), np.nan, dtype=float)
    benefit_prob = np.full(len(prepared.rows), np.nan, dtype=float)
    details: list[dict[str, Any]] = []
    base = make_logistic(random_state)
    folds = grouped_splits(
        prepared.safety,
        prepared.groups,
        n_splits=n_splits,
        random_state=random_state,
    )
    for fold_index, (train, test) in enumerate(folds):
        safety_model = clone(base)
        safety_model.fit(prepared.matrix[train], prepared.safety[train])
        safety_prob[test] = safety_model.predict_proba(prepared.matrix[test])[:, 1]

        safe_train = train[prepared.safety[train] == 1]
        benefit_train_labels = prepared.joint_benefit[safe_train]
        if len(np.unique(benefit_train_labels)) != 2:
            raise ValueError(f"fold {fold_index} safe training rows lack a benefit class")
        benefit_model = clone(base)
        benefit_model.fit(prepared.matrix[safe_train], benefit_train_labels)
        benefit_prob[test] = benefit_model.predict_proba(prepared.matrix[test])[:, 1]
        details.append(
            {
                "fold": fold_index,
                "train_case_count": int(len(train)),
                "test_case_count": int(len(test)),
                "train_seed_count": int(len(np.unique(prepared.groups[train]))),
                "test_seed_count": int(len(np.unique(prepared.groups[test]))),
                "safe_train_case_count": int(len(safe_train)),
                "test_groups": sorted(str(x) for x in np.unique(prepared.groups[test])),
            }
        )
    if not np.all(np.isfinite(safety_prob)) or not np.all(np.isfinite(benefit_prob)):
        raise AssertionError("incomplete grouped OOF probabilities")
    return safety_prob, benefit_prob, details


def _policy_row(
    prepared: PreparedGateB,
    safety_prob: np.ndarray,
    benefit_prob: np.ndarray,
    safety_threshold: float,
    benefit_threshold: float,
) -> dict[str, Any]:
    command = (safety_prob >= safety_threshold) & (benefit_prob >= benefit_threshold)
    command_count = int(command.sum())
    command_seeds = len(set(prepared.groups[command]))
    safe_count = int(np.sum(prepared.safety[command]))
    beneficial_count = int(np.sum(prepared.joint_benefit[command]))
    safety_rate = safe_count / command_count if command_count else 0.0
    precision = beneficial_count / command_count if command_count else 0.0
    oracle_total = int(prepared.joint_benefit.sum())
    return {
        "safety_threshold": safety_threshold,
        "benefit_threshold": benefit_threshold,
        "command_case_count": command_count,
        "command_seed_count": command_seeds,
        "safe_command_count": safe_count,
        "unsafe_command_count": command_count - safe_count,
        "joint_beneficial_command_count": beneficial_count,
        "selected_safety_rate": safety_rate,
        "joint_benefit_precision": precision,
        "joint_benefit_recall": beneficial_count / oracle_total if oracle_total else 0.0,
        "passes_development_policy_gate": bool(
            command_count >= 10
            and command_seeds >= 8
            and safety_rate >= 0.90
            and precision >= 0.50
        ),
    }


def select_thresholds(
    prepared: PreparedGateB,
    safety_prob: np.ndarray,
    benefit_prob: np.ndarray,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    sweep = [
        _policy_row(prepared, safety_prob, benefit_prob, safe, benefit)
        for safe in SAFETY_THRESHOLDS
        for benefit in BENEFIT_THRESHOLDS
    ]
    passing = [row for row in sweep if row["passes_development_policy_gate"]]
    if not passing:
        return None, sweep
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
    return passing[0], sweep


def fit_full_gate_b(prepared: PreparedGateB, random_state: int = 20260831):
    safety_model = make_logistic(random_state)
    safety_model.fit(prepared.matrix, prepared.safety)
    safe = prepared.safety == 1
    if len(np.unique(prepared.joint_benefit[safe])) != 2:
        raise ValueError("safe Gate-B population lacks a joint-benefit class")
    benefit_model = make_logistic(random_state)
    benefit_model.fit(prepared.matrix[safe], prepared.joint_benefit[safe])
    return safety_model, benefit_model


def development_report(
    prepared: PreparedGateB,
    safety_prob: np.ndarray,
    benefit_prob: np.ndarray,
    folds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected, sweep = select_thresholds(prepared, safety_prob, benefit_prob)
    safe_mask = prepared.safety == 1
    return {
        "model_contract_version": MODEL_CONTRACT_VERSION,
        "case_count": len(prepared.rows),
        "independent_seed_count": len(np.unique(prepared.groups)),
        "safe_case_count": int(prepared.safety.sum()),
        "unsafe_case_count": int(len(prepared.safety) - prepared.safety.sum()),
        "joint_beneficial_case_count": int(prepared.joint_benefit.sum()),
        "joint_nonbeneficial_case_count": int(
            len(prepared.joint_benefit) - prepared.joint_benefit.sum()
        ),
        "safety_probability_metrics": probability_metrics(prepared.safety, safety_prob),
        "conditional_benefit_probability_metrics": probability_metrics(
            prepared.joint_benefit[safe_mask], benefit_prob[safe_mask]
        ),
        "folds": list(folds),
        "selected_thresholds": selected,
        "threshold_selection_passed": selected is not None,
        "threshold_sweep": sweep,
        "predictor_ready_for_untouched_validation": selected is not None,
        "guardrail": "Development OOF evidence only; no calibration or validation claim.",
    }


def prediction_record(
    row: Mapping[str, Any], safety_probability: float, benefit_probability: float
) -> dict[str, Any]:
    if not all(math.isfinite(x) for x in (safety_probability, benefit_probability)):
        raise ValueError("non-finite predictor probability")
    return {
        "case_id": str(row["case_id"]),
        "seed": int(row["seed"]),
        "decision_step": int(row["decision_step"]),
        "safety_probability_oof": float(safety_probability),
        "conditional_joint_benefit_probability_oof": float(benefit_probability),
        "observed_safety_admissible": bool(
            row["safety_service_oracle"]["safety_admissible"]
        ),
        "observed_joint_beneficial": bool(row["safety_service_oracle"]["wall_selected"]),
    }
