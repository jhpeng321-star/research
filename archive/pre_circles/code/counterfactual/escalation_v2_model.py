"""Leakage-safe model utilities for counterfactual escalation v2.

The primary cohort contains every eligible decision record.  Conditioning on
``wall_engaged`` is supported only as a diagnostic because engagement is
observed after the decision and is not a deployable input to the gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from counterfactual.safety_service_oracle import classify_safety_service_case


HEALTH_ONLY = (
    "health",
    "health_delta_1",
    "health_change_30",
    "health_lag_5",
    "health_lag_10",
    "health_path_mean",
    "health_path_range",
)

AGGREGATE_TRAFFIC = HEALTH_ONLY + (
    "context_decision_step",
    "context_is_hyper",
    "decision_queue_count",
    "decision_pre_merge_mean_speed_mps",
    "decision_merge_mean_speed_mps",
    "decision_pre_merge_vehicle_count",
    "decision_hard_brake_vehicle_count",
    "decision_low_ttc_pair_count",
    "pattern60_queue_count_change",
    "pattern60_queue_count_slope_per_step",
    "pattern60_pre_merge_speed_mps_change",
    "pattern60_pre_merge_speed_slope_mps_per_step",
    "pattern60_pre_merge_speed_std_mps_mean",
    "pattern60_hard_brake_onset_count",
    "pattern60_hard_brake_active_fraction",
    "pattern60_low_ttc_active_fraction",
    "pattern60_inflow_scaled_change",
)

TRAFFIC_PLACEMENT = AGGREGATE_TRAFFIC + (
    "context_cav_penetration",
    "decision_active_cav_count",
    "decision_active_cav_pre_merge_count",
    "decision_active_cav_pre_merge_lane_count",
    "placement_network_cav_fraction",
    "placement_pre_merge_cav_fraction",
    "placement_soft_band_cav_count",
    "placement_soft_band_lane_coverage_fraction",
    "placement_strong_band_cav_count",
    "placement_strong_band_lane_coverage_fraction",
    "placement_cav_distance_to_merge_mean_m",
    "placement_longitudinal_span_m",
)

TRAFFIC_PLACEMENT_LOCAL = TRAFFIC_PLACEMENT + (
    "local_bin_0_100m_vehicle_count",
    "local_bin_0_100m_mean_speed_mps",
    "local_bin_0_100m_speed_std_mps",
    "local_bin_0_100m_queue_count",
    "local_bin_0_100m_hard_brake_count",
    "local_bin_100_200m_vehicle_count",
    "local_bin_100_200m_mean_speed_mps",
    "local_bin_100_200m_speed_std_mps",
    "local_bin_100_200m_queue_count",
    "local_bin_100_200m_hard_brake_count",
    "local_bin_200_400m_vehicle_count",
    "local_bin_200_400m_mean_speed_mps",
    "local_bin_200_400m_speed_std_mps",
    "local_downstream_minus_upstream_speed_mps",
    "formation_follower_gap_min_m",
    "formation_leader_gap_min_m",
    "formation_cav_minus_neighbor_hdv_speed_mps",
    "formation_soft_cav_speed_std_mps",
    "traffic_lag_10_queue_count",
    "traffic_lag_10_pre_merge_mean_speed_mps",
    "traffic_lag_30_queue_count",
    "traffic_lag_30_pre_merge_mean_speed_mps",
)

FEATURE_SETS = {
    "health_only": HEALTH_ONLY,
    "aggregate_traffic": AGGREGATE_TRAFFIC,
    "traffic_placement": TRAFFIC_PLACEMENT,
    "traffic_placement_local": TRAFFIC_PLACEMENT_LOCAL,
}


@dataclass(frozen=True)
class PreparedRows:
    rows: tuple[Mapping[str, Any], ...]
    oracle: tuple[Mapping[str, Any], ...]
    groups: np.ndarray
    selected: np.ndarray
    unsafe: np.ndarray
    engaged: np.ndarray


def complete_group_id(row: Mapping[str, Any]) -> str:
    """Return the complete scenario-seed identifier used for every split."""
    scenario = row["scenario"]
    return "|".join(
        (
            str(scenario["demand"]),
            f"p{float(scenario['cav_penetration']):.6f}",
            str(scenario["hdv_model"]),
            f"seed{int(row['seed'])}",
        )
    )


def prepare_rows(rows: Sequence[Mapping[str, Any]], *, engaged_only: bool = False) -> PreparedRows:
    """Classify rows with the frozen oracle and construct grouped targets."""
    eligible = [row for row in rows if bool(row.get("eligible", False))]
    classified = [classify_safety_service_case(row) for row in eligible]
    if engaged_only:
        pairs = [
            (row, result)
            for row, result in zip(eligible, classified)
            if bool(result["wall_engaged"])
        ]
        eligible = [row for row, _ in pairs]
        classified = [result for _, result in pairs]
    return PreparedRows(
        rows=tuple(eligible),
        oracle=tuple(classified),
        groups=np.asarray([complete_group_id(row) for row in eligible]),
        selected=np.asarray([int(result["wall_selected"]) for result in classified], dtype=int),
        unsafe=np.asarray(
            [
                int(result["wall_engaged"] and not result["safety_admissible"])
                for result in classified
            ],
            dtype=int,
        ),
        engaged=np.asarray([int(result["wall_engaged"]) for result in classified], dtype=int),
    )


def feature_value(row: Mapping[str, Any], name: str) -> float:
    """Read one explicitly allow-listed, decision-time feature."""
    if name == "context_decision_step":
        value = row["decision_step"]
    elif name == "context_is_hyper":
        value = int(str(row["scenario"]["demand"]).lower() == "hyper")
    elif name == "context_cav_penetration":
        value = row["scenario"]["cav_penetration"]
    else:
        value = row["features"].get(name)
    if value is None:
        raise ValueError(f"{row.get('case_id')} has missing feature {name!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{row.get('case_id')} has non-finite feature {name!r}")
    return result


def feature_matrix(rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str]) -> np.ndarray:
    """Build a finite matrix using only the named pre-treatment fields."""
    return np.asarray(
        [[feature_value(row, name) for name in feature_names] for row in rows],
        dtype=float,
    )


def grouped_splits(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 20260825,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create stratified group folds and enforce zero group leakage."""
    target = np.asarray(labels, dtype=int)
    group_values = np.asarray(groups)
    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except ModuleNotFoundError:
        unique_groups = np.unique(group_values)
        if len(unique_groups) < n_splits:
            raise ValueError("fewer scenario-seed groups than requested folds")
        classes = np.unique(target)
        if len(classes) != 2:
            raise ValueError("grouped binary splits require exactly two classes")

        rng = np.random.default_rng(random_state)
        shuffled = list(unique_groups[rng.permutation(len(unique_groups))])
        group_counts = {
            group: np.asarray(
                [np.sum(target[group_values == group] == label) for label in classes],
                dtype=float,
            )
            for group in unique_groups
        }
        # Place the most class-skewed/largest groups first, using the seeded
        # shuffle only to break exact ties. Each placement minimizes the
        # across-fold class imbalance that would result.
        shuffled.sort(
            key=lambda group: (
                np.std(group_counts[group] / max(float(np.sum(group_counts[group])), 1.0)),
                float(np.sum(group_counts[group])),
            ),
            reverse=True,
        )
        total_counts = np.asarray([np.sum(target == label) for label in classes], dtype=float)
        fold_counts = np.zeros((n_splits, len(classes)), dtype=float)
        fold_groups: list[list[Any]] = [[] for _ in range(n_splits)]
        for group in shuffled:
            counts = group_counts[group]
            best_fold = None
            best_score = None
            for fold_index in range(n_splits):
                candidate = fold_counts.copy()
                candidate[fold_index] += counts
                fractions = candidate / np.maximum(total_counts, 1.0)
                imbalance = float(np.mean(np.std(fractions, axis=0)))
                group_balance = float(
                    np.std(
                        [
                            len(values) + int(index == fold_index)
                            for index, values in enumerate(fold_groups)
                        ]
                    )
                )
                score = (imbalance, group_balance, len(fold_groups[fold_index]))
                if best_score is None or score < best_score:
                    best_score = score
                    best_fold = fold_index
            assert best_fold is not None
            fold_counts[best_fold] += counts
            fold_groups[best_fold].append(group)

        all_indices = np.arange(len(target))
        folds = []
        for test_groups in fold_groups:
            test_mask = np.isin(group_values, test_groups)
            folds.append((all_indices[~test_mask], all_indices[test_mask]))
    else:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds = list(splitter.split(np.zeros((len(target), 1)), target, group_values))
    for train, test in folds:
        if set(group_values[train]) & set(group_values[test]):
            raise AssertionError("scenario-seed leakage across a grouped fold")
        if len(np.unique(target[train])) != 2:
            raise ValueError("a grouped training fold lacks one target class")
    return folds


def make_model(name: str, random_state: int = 20260825):
    """Construct one frozen, intentionally small classical classifier."""
    if name == "regularized_logistic":
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
    if name == "shallow_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=100,
            max_leaf_nodes=7,
            min_samples_leaf=8,
            l2_regularization=1.0,
            random_state=random_state,
        )
    raise ValueError(f"unknown model {name!r}")


def grouped_oof_probabilities(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    model_name: str,
    n_splits: int = 5,
    random_state: int = 20260825,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Generate one out-of-fold probability for every grouped case."""
    from sklearn.base import clone

    probabilities = np.full(len(labels), np.nan, dtype=float)
    details: list[dict[str, Any]] = []
    base_model = make_model(model_name, random_state=random_state)
    for fold_index, (train, test) in enumerate(
        grouped_splits(labels, groups, n_splits=n_splits, random_state=random_state)
    ):
        model = clone(base_model)
        fit_kwargs: dict[str, Any] = {}
        if model_name == "shallow_gradient_boosting":
            positives = max(int(labels[train].sum()), 1)
            negatives = max(int(len(train) - labels[train].sum()), 1)
            weights = np.where(
                labels[train] == 1,
                len(train) / (2.0 * positives),
                len(train) / (2.0 * negatives),
            )
            fit_kwargs["sample_weight"] = weights
        model.fit(matrix[train], labels[train], **fit_kwargs)
        probabilities[test] = model.predict_proba(matrix[test])[:, 1]
        details.append(
            {
                "fold": fold_index,
                "train_case_count": int(len(train)),
                "test_case_count": int(len(test)),
                "train_group_count": int(len(np.unique(groups[train]))),
                "test_group_count": int(len(np.unique(groups[test]))),
                "train_positive_count": int(labels[train].sum()),
                "test_positive_count": int(labels[test].sum()),
                "test_groups": sorted(str(value) for value in np.unique(groups[test])),
            }
        )
    if not np.all(np.isfinite(probabilities)):
        raise AssertionError("OOF prediction vector is incomplete")
    return probabilities, details


def probability_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Return threshold-free grouped-development diagnostics."""
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    return {
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
    }


def threshold_policy_metrics(
    prepared: PreparedRows, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Evaluate a candidate command policy against paired observed outcomes."""
    command = np.asarray(probabilities >= threshold, dtype=bool)
    selected = prepared.selected.astype(bool)
    engaged = prepared.engaged.astype(bool)
    unsafe = prepared.unsafe.astype(bool)
    true_positive = int(np.sum(command & selected))
    command_count = int(command.sum())
    engagement_count = int(np.sum(command & engaged))
    unsafe_count = int(np.sum(command & unsafe))
    oracle_total = int(selected.sum())
    return {
        "threshold": float(threshold),
        "command_count": command_count,
        "engagement_count": engagement_count,
        "true_beneficial_activation_count": true_positive,
        "false_command_count": command_count - true_positive,
        "unsafe_engagement_count": unsafe_count,
        "beneficial_precision_per_command": (
            true_positive / command_count if command_count else 0.0
        ),
        "beneficial_precision_among_engagements": (
            true_positive / engagement_count if engagement_count else 0.0
        ),
        "oracle_recall": true_positive / oracle_total if oracle_total else 0.0,
        "oracle_missed_beneficial_count": oracle_total - true_positive,
        "unsafe_rate_per_command": unsafe_count / command_count if command_count else 0.0,
        "unsafe_rate_among_engagements": (
            unsafe_count / engagement_count if engagement_count else 0.0
        ),
    }


def threshold_sweep(
    prepared: PreparedRows,
    probabilities: np.ndarray,
    thresholds: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate a fixed grid; this is diagnostic and does not freeze a gate."""
    values = thresholds or np.linspace(0.05, 0.95, 19).tolist()
    return [
        threshold_policy_metrics(prepared, probabilities, float(threshold)) for threshold in values
    ]


def rule_policy_metrics(
    prepared: PreparedRows,
    rule: Callable[[Mapping[str, Any]], bool],
    *,
    name: str,
) -> dict[str, Any]:
    probabilities = np.asarray([float(bool(rule(row))) for row in prepared.rows])
    report = threshold_policy_metrics(prepared, probabilities, 0.5)
    report["name"] = name
    report.pop("threshold", None)
    return report
