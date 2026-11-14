"""Development-only observation sufficiency diagnostics for IDM Gate B.

The module deliberately accepts already generated rows and never launches SUMO.
All model inputs are limited to fields produced from the common control prefix
or deterministic predecision Gate-F preflight.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from counterfactual.actionability import dose_qualified
from counterfactual.idm_predictor_contract import (
    FORBIDDEN_FEATURE_TOKENS,
    complete_group_id,
)


RANDOM_STATE = 20260831
MODEL_FAMILIES = ("regularized_logistic", "shallow_gradient_boosting")
SAFETY_THRESHOLD = 0.7000000000000002
BENEFIT_THRESHOLD = 0.7499999999999999


@dataclass(frozen=True)
class ObservationPopulation:
    rows: tuple[Mapping[str, Any], ...]
    groups: np.ndarray
    dose: np.ndarray
    compact_dose: np.ndarray
    safety: np.ndarray
    benefit: np.ndarray


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def feature_value(row: Mapping[str, Any], name: str) -> float:
    if name == "context_decision_step":
        value = row.get("decision_step")
    elif name == "context_cav_penetration":
        value = row.get("cav_penetration")
    else:
        value = (row.get("features") or {}).get(name)
    result = _number(value)
    if result is None:
        raise ValueError(f"{row.get('case_id')} lacks finite scalar feature {name!r}")
    return result


def prepare_population(rows: Sequence[Mapping[str, Any]]) -> ObservationPopulation:
    admitted = tuple(
        row for row in rows
        if bool(row.get("predecision_supervisor", {}).get("admitted"))
    )
    if not admitted:
        raise ValueError("no Gate-F admissions")
    duration = np.asarray(
        [
            int(float((row.get("treatment_dose") or {}).get("realized_duration_fraction", 0.0)) >= 0.80)
            for row in admitted
        ],
        dtype=int,
    )
    compact = np.asarray([int(dose_qualified(row)) for row in admitted], dtype=int)
    safety = np.asarray(
        [int(row["safety_service_oracle"]["safety_admissible"]) for row in admitted],
        dtype=int,
    )
    benefit = np.asarray(
        [int(row["safety_service_oracle"]["wall_selected"]) for row in admitted],
        dtype=int,
    )
    return ObservationPopulation(
        rows=admitted,
        groups=np.asarray([complete_group_id(row) for row in admitted]),
        dose=duration,
        compact_dose=compact,
        safety=safety,
        benefit=benefit,
    )


def population_integrity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    population = prepare_population(rows)
    qualified = population.compact_dose == 1
    safe = qualified & (population.safety == 1)
    unsafe = qualified & (population.safety == 0)
    beneficial = qualified & (population.benefit == 1)
    nonbeneficial = qualified & (population.benefit == 0)

    def support(mask: np.ndarray) -> dict[str, int]:
        return {
            "case_count": int(mask.sum()),
            "independent_seed_count": len(set(population.groups[mask])),
        }

    all_mask = np.ones(len(population.rows), dtype=bool)
    raw_groups = {complete_group_id(row) for row in rows}
    return {
        "declared_windows": {
            "case_count": len(rows),
            "independent_seed_count": len(raw_groups),
        },
        "gate_f_admissions": support(all_mask),
        "dose_at_least_80_percent": support(population.dose == 1),
        "compact_dose_qualified": support(qualified),
        "dose_failures": support(~qualified),
        "safe_dose_qualified": support(safe),
        "unsafe_dose_qualified": support(unsafe),
        "joint_beneficial_dose_qualified": support(beneficial),
        "joint_nonbeneficial_dose_qualified": support(nonbeneficial),
        "safe_beneficial": support(safe & beneficial),
        "safe_nonbeneficial": support(safe & ~beneficial),
        "frozen_count_checks": {
            "declared_windows_480": len(rows) == 480,
            "gate_f_admissions_199": len(population.rows) == 199,
            "dose_qualified_179": int(qualified.sum()) == 179,
            "safe_106": int(safe.sum()) == 106,
            "unsafe_73": int(unsafe.sum()) == 73,
            "joint_beneficial_53": int(beneficial.sum()) == 53,
            "joint_nonbeneficial_126": int(nonbeneficial.sum()) == 126,
        },
    }


def _feature_flags(name: str) -> dict[str, bool]:
    lower = name.lower()
    temporal = bool(
        lower.startswith("traffic_lag_")
        or lower.startswith("pattern60_")
        or "health_lag" in lower
        or "health_change" in lower
        or "health_path" in lower
        or "health_falling_segments" in lower
        or "health_rising_segments" in lower
        or "autocorr" in lower
        or "_slope" in lower
        or "_active_fraction" in lower
        or "discharge_rate_" in lower
        or "short_minus_long" in lower
        or lower.endswith("_20s")
        or lower.endswith("_60s")
    )
    return {
        "instantaneous": not temporal,
        "temporal": temporal,
        "lane_resolved": lower.startswith("local_lane_") or lower.startswith("placement_lane_"),
        "treatment_aware": lower.startswith("preflight_") or lower.startswith("obs_v2_dose_"),
        "geometry_related": lower.startswith(("placement_", "formation_", "preflight_")),
        "rear_safety_related": any(
            token in lower
            for token in ("follower", "rear", "ttc", "hard_brake", "acceleration")
        ),
    }


def _feature_source(name: str) -> str:
    if name.startswith("preflight_"):
        return "deterministic_gate_f_preflight_at_decision"
    if name.startswith("obs_v2_dose_"):
        return "frozen_wall_proposal_and_common_control_prefix"
    if name.startswith("obs_v2_"):
        return "common_control_prefix_vehicle_snapshots"
    if name.startswith("context_"):
        return "declared_scenario_or_decision_context"
    return "common_control_prefix_features_from_trace"


def audit_features(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    raw_names = sorted(set.union(*(set(row.get("features") or {}) for row in rows)))
    names = ["context_decision_step", "context_cav_penetration", *raw_names]
    records = []
    scalar_columns: dict[str, np.ndarray] = {}
    for name in names:
        raw_values = []
        finite_values = []
        non_scalar = 0
        missing = 0
        for row in rows:
            if name == "context_decision_step":
                value = row.get("decision_step")
            elif name == "context_cav_penetration":
                value = row.get("cav_penetration")
            else:
                value = (row.get("features") or {}).get(name)
            raw_values.append(value)
            number = _number(value)
            if number is None:
                missing += 1
                non_scalar += int(isinstance(value, (list, tuple, dict, set)))
            else:
                finite_values.append(number)
        forbidden = [token for token in FORBIDDEN_FEATURE_TOKENS if token in name.lower()]
        unique = len(set(finite_values))
        dominant = 0.0
        if finite_values:
            dominant = max(Counter(finite_values).values()) / len(finite_values)
        approved = missing == 0 and non_scalar == 0 and not forbidden
        if approved:
            scalar_columns[name] = np.asarray(finite_values, dtype=float)
        records.append(
            {
                "name": name,
                "source": _feature_source(name),
                "latest_timestamp": "declared_decision_inclusive",
                "future_samples": 0,
                "flags": _feature_flags(name),
                "case_count": len(rows),
                "finite_scalar_count": len(finite_values),
                "missing_or_nonfinite_count": missing,
                "non_scalar_count": non_scalar,
                "unique_count": unique,
                "constant": unique == 1 and missing == 0,
                "near_constant_95pct": dominant >= 0.95,
                "dominant_value_fraction": dominant,
                "forbidden_leakage_tokens": forbidden,
                "approved_for_development_analysis": approved,
            }
        )

    duplicates = []
    scalar_names = sorted(scalar_columns)
    signatures: dict[bytes, list[str]] = defaultdict(list)
    for name in scalar_names:
        signatures[scalar_columns[name].tobytes()].append(name)
    for values in signatures.values():
        if len(values) > 1:
            duplicates.append({"feature_names": values, "count": len(values)})

    variable = [name for name in scalar_names if len(np.unique(scalar_columns[name])) > 1]
    correlations = []
    if variable:
        from scipy.stats import rankdata

        ranked = np.column_stack([rankdata(scalar_columns[name]) for name in variable])
        matrix = np.corrcoef(ranked, rowvar=False)
        for left in range(len(variable)):
            for right in range(left + 1, len(variable)):
                value = float(matrix[left, right])
                if math.isfinite(value) and abs(value) >= 0.98:
                    correlations.append(
                        {
                            "left": variable[left],
                            "right": variable[right],
                            "spearman": value,
                        }
                    )
    prohibited_sources = {
        "treatment_dose": "postdecision realized dose and lifecycle",
        "safety_service_oracle": "postdecision outcome and counterfactual effects",
        "arms": "realized treatment/control branch outcomes",
        "control_arm": "realized control outcome",
        "treatment_arm": "realized treatment outcome",
        "completion_reason": "postdecision lifecycle outcome",
        "release_reason": "postdecision lifecycle outcome",
        "oracle": "counterfactual label or oracle output",
    }
    top_level_names = set.union(*(set(row) for row in rows))
    explicitly_rejected = [
        {
            "field": name,
            "reason": reason,
            "present_in_saved_rows": name in top_level_names,
            "used_as_model_input": False,
        }
        for name, reason in prohibited_sources.items()
    ]
    return {
        "raw_feature_count": len(raw_names),
        "audited_feature_count_including_context": len(names),
        "approved_finite_scalar_count": len(scalar_names),
        "rejected_feature_count": len(names) - len(scalar_names),
        "constant_feature_count": sum(row["constant"] for row in records),
        "near_constant_feature_count": sum(row["near_constant_95pct"] for row in records),
        "missing_or_nonfinite_feature_count": sum(
            row["missing_or_nonfinite_count"] > 0 for row in records
        ),
        "forbidden_leakage_feature_count": sum(
            bool(row["forbidden_leakage_tokens"]) for row in records
        ),
        "features": records,
        "exact_duplicate_groups": duplicates,
        "absolute_spearman_at_least_0_98": correlations,
        "explicitly_rejected_future_or_outcome_fields": explicitly_rejected,
        "guardrail": (
            "Provenance follows the frozen generator: row.features is computed from "
            "the common control prefix and deterministic preflight only. Postdecision "
            "dose, treatment contract, arms and oracle fields are explicitly excluded."
        ),
    }


def feature_families(audit: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    approved = {
        row["name"]
        for row in audit["features"]
        if row["approved_for_development_analysis"] and not row["constant"]
    }

    def choose(predicate) -> tuple[str, ...]:
        return tuple(sorted(name for name in approved if predicate(name, name.lower())))

    temporal = choose(
        lambda name, lower: _feature_flags(name)["temporal"]
    )
    lane = choose(
        lambda name, lower: lower.startswith(("local_lane_", "local_bin_"))
        or lower == "local_downstream_minus_upstream_speed_mps"
    )
    geometry = choose(
        lambda name, lower: lower.startswith(("placement_", "formation_"))
        or lower in {
            "wall_band_cav_count_scaled",
            "decision_active_cav_count",
            "decision_active_cav_pre_merge_count",
            "decision_active_cav_pre_merge_lane_count",
        }
    )
    dose_projection = choose(
        lambda name, lower: lower.startswith("preflight_")
        or lower.startswith("obs_v2_dose_")
    )
    rear_safety = choose(
        lambda name, lower: any(
            token in lower
            for token in (
                "follower",
                "rear",
                "ttc",
                "hard_brake",
                "min_acceleration",
                "leader_gap",
            )
        )
    )
    excluded_prefixes = (
        "traffic_lag_",
        "pattern60_",
        "local_lane_",
        "local_bin_",
        "placement_",
        "formation_",
        "preflight_",
    )
    # Keep the six physical families interpretable.  A feature can be a useful
    # member of the combined contract without also being smuggled into the
    # static baseline (for example a t0 rear-TTC or lane-bin measurement).
    static = choose(
        lambda name, lower: not lower.startswith(excluded_prefixes)
        and name not in temporal
        and name not in lane
        and name not in geometry
        and name not in dose_projection
        and name not in rear_safety
        and not lower.startswith("obs_v2_dose_")
    )
    combined = tuple(
        sorted(set(static) | set(temporal) | set(lane) | set(geometry) | set(dose_projection) | set(rear_safety))
    )
    families = {
        "A_static_traffic": static,
        "B_temporal_speed_gap_density": temporal,
        "C_lane_resolved_traffic": lane,
        "D_wall_geometry_placement_authority": geometry,
        "E_predecision_dose_assembly_projection": dose_projection,
        "F_follower_compression_rear_safety": rear_safety,
        "G_combined_physical": combined,
    }
    if any(not values for values in families.values()):
        empty = [name for name, values in families.items() if not values]
        raise ValueError(f"empty observation families: {empty}")
    return families


def matrix_for(population: ObservationPopulation, names: Sequence[str]) -> np.ndarray:
    matrix = np.asarray(
        [[feature_value(row, name) for name in names] for row in population.rows],
        dtype=float,
    )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("non-finite observation matrix")
    return matrix


def target_masks(population: ObservationPopulation) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    qualified = population.compact_dose == 1
    safe = qualified & (population.safety == 1)
    return {
        "dose_80pct": (np.ones(len(population.rows), dtype=bool), population.dose),
        "safety_admissible": (qualified, population.safety),
        "conditional_joint_benefit": (safe, population.benefit),
    }


def current_model_error_analysis(
    rows: Sequence[Mapping[str, Any]], prediction_payload: Mapping[str, Any]
) -> dict[str, Any]:
    population = prepare_population(rows)
    candidate = prediction_payload["candidates"]["primary_shallow_boosting"]
    case_ids = [str(value) for value in prediction_payload["case_ids"]]
    qualified_by_id = {
        str(row["case_id"]): row for row, take in zip(population.rows, population.compact_dose)
        if take
    }
    if set(case_ids) != set(qualified_by_id):
        raise ValueError("current OOF cases do not match the 179 qualified cases")
    safety_prob = np.asarray(candidate["safety_probability_oof"], dtype=float)
    benefit_prob = np.asarray(candidate["conditional_benefit_probability_oof"], dtype=float)
    labels_safe = np.asarray(prediction_payload["observed_safety"], dtype=int)
    labels_benefit = np.asarray(prediction_payload["observed_joint_benefit"], dtype=int)
    command = (safety_prob >= SAFETY_THRESHOLD) & (benefit_prob >= BENEFIT_THRESHOLD)
    safety_positive = safety_prob >= SAFETY_THRESHOLD
    safe_rows = labels_safe == 1
    conditional_benefit_positive = benefit_prob >= BENEFIT_THRESHOLD

    def confusion(predicted: np.ndarray, observed: np.ndarray) -> dict[str, int]:
        return {
            "true_positive": int(np.sum(predicted & (observed == 1))),
            "false_positive": int(np.sum(predicted & (observed == 0))),
            "false_negative": int(np.sum(~predicted & (observed == 1))),
            "true_negative": int(np.sum(~predicted & (observed == 0))),
        }

    error_cases = []
    for case_id, ps, pb, selected, observed_safe, observed_benefit in zip(
        case_ids, safety_prob, benefit_prob, command, labels_safe, labels_benefit
    ):
        category = (
            "true_positive" if selected and observed_benefit
            else "false_positive" if selected
            else "false_negative" if observed_benefit
            else "true_negative"
        )
        row = qualified_by_id[case_id]
        error_cases.append(
            {
                "case_id": case_id,
                "seed": int(row["seed"]),
                "decision_step": int(row["decision_step"]),
                "category": category,
                "safety_probability_oof": float(ps),
                "benefit_probability_oof": float(pb),
                "observed_safety": bool(observed_safe),
                "observed_joint_benefit": bool(observed_benefit),
                "effects": dict(row["safety_service_oracle"]["effects"]),
            }
        )
    return {
        "candidate": "primary_shallow_boosting",
        "thresholds": {
            "safety": SAFETY_THRESHOLD,
            "conditional_benefit": BENEFIT_THRESHOLD,
        },
        "safety_head_confusion": confusion(safety_positive, labels_safe),
        "conditional_benefit_head_confusion_on_observed_safe_cases": confusion(
            conditional_benefit_positive[safe_rows], labels_benefit[safe_rows]
        ),
        "joint_command_confusion": confusion(command, labels_benefit),
        "selected_case_count": int(command.sum()),
        "selected_seed_count": len(
            {int(qualified_by_id[case_id]["seed"]) for case_id, take in zip(case_ids, command) if take}
        ),
        "cases": error_cases,
    }


def _safe_auc(labels: np.ndarray, values: np.ndarray, *, direction_invariant: bool = False) -> float | None:
    from sklearn.metrics import roc_auc_score

    if len(labels) < 2 or len(np.unique(labels)) < 2 or len(np.unique(values)) < 2:
        return None
    auc = float(roc_auc_score(labels, values))
    return max(auc, 1.0 - auc) if direction_invariant else auc


def univariate_information(
    population: ObservationPopulation,
    audit: Mapping[str, Any],
    *,
    bootstrap_repeats: int = 200,
) -> dict[str, Any]:
    from scipy.stats import rankdata
    from sklearn.feature_selection import mutual_info_classif

    approved = [
        row["name"] for row in audit["features"]
        if row["approved_for_development_analysis"] and not row["constant"]
    ]
    targets = target_masks(population)
    rng = np.random.default_rng(RANDOM_STATE)
    report: dict[str, Any] = {}
    for target_name, (eligible, labels_all) in targets.items():
        labels = labels_all[eligible]
        groups = population.groups[eligible]
        unique_groups = np.unique(groups)
        values_matrix = matrix_for(population, approved)[eligible]

        def column_auc_and_spearman(
            values: np.ndarray, binary_labels: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            ranks = rankdata(values, axis=0, method="average")
            positive = binary_labels == 1
            n_positive = int(positive.sum())
            n_negative = len(binary_labels) - n_positive
            if not n_positive or not n_negative:
                return (
                    np.full(values.shape[1], np.nan),
                    np.full(values.shape[1], np.nan),
                )
            auc = (
                ranks[positive].sum(axis=0)
                - n_positive * (n_positive + 1) / 2.0
            ) / (n_positive * n_negative)
            label_ranks = rankdata(binary_labels, method="average")
            centered_x = ranks - ranks.mean(axis=0)
            centered_y = label_ranks - label_ranks.mean()
            denominator = np.sqrt(
                np.sum(centered_x * centered_x, axis=0)
                * np.sum(centered_y * centered_y)
            )
            correlation = np.divide(
                np.sum(centered_x * centered_y[:, None], axis=0),
                denominator,
                out=np.full(values.shape[1], np.nan),
                where=denominator > 0,
            )
            return auc, correlation

        raw_auc, raw_correlation = column_auc_and_spearman(values_matrix, labels)
        bootstrap_auc: list[list[float]] = [[] for _ in approved]
        bootstrap_correlation: list[list[float]] = [[] for _ in approved]
        for _ in range(bootstrap_repeats):
            sampled_groups = rng.choice(
                unique_groups, size=len(unique_groups), replace=True
            )
            indices = np.concatenate(
                [np.flatnonzero(groups == group) for group in sampled_groups]
            )
            auc_values, correlation_values = column_auc_and_spearman(
                values_matrix[indices], labels[indices]
            )
            for index, value in enumerate(auc_values):
                if math.isfinite(float(value)):
                    bootstrap_auc[index].append(
                        max(float(value), 1.0 - float(value))
                    )
            for index, value in enumerate(correlation_values):
                if math.isfinite(float(value)):
                    bootstrap_correlation[index].append(float(value))

        records = []
        for index, name in enumerate(approved):
            values = values_matrix[:, index]
            auc_value = float(raw_auc[index])
            auc = max(auc_value, 1.0 - auc_value)
            correlation = float(raw_correlation[index])
            unique_values = np.unique(values)
            discrete = (
                len(unique_values) <= min(20, max(2, len(values) // 5))
                and np.allclose(unique_values, np.round(unique_values))
            )
            mi = float(
                mutual_info_classif(
                    values.reshape(-1, 1),
                    labels,
                    discrete_features=discrete,
                    random_state=RANDOM_STATE,
                )[0]
            )
            boot_auc = bootstrap_auc[index]
            boot_corr = bootstrap_correlation[index]
            direction = "positive" if correlation > 0 else ("negative" if correlation < 0 else "none")
            stable_sign = (
                sum((value > 0) == (correlation > 0) for value in boot_corr) / len(boot_corr)
                if correlation and boot_corr else 0.0
            )
            records.append(
                {
                    "feature": name,
                    "direction_invariant_auc": auc,
                    "seed_clustered_auc_bootstrap_95": (
                        [float(np.quantile(boot_auc, 0.025)), float(np.quantile(boot_auc, 0.975))]
                        if boot_auc else None
                    ),
                    "spearman": correlation,
                    "direction": direction,
                    "bootstrap_same_direction_fraction": stable_sign,
                    "mutual_information": mi,
                }
            )
        records.sort(
            key=lambda row: (
                row["direction_invariant_auc"] or 0.0,
                row["mutual_information"],
            ),
            reverse=True,
        )
        report[target_name] = {
            "eligible_case_count": int(eligible.sum()),
            "independent_seed_count": len(unique_groups),
            "features": records,
            "top_25": records[:25],
        }
    return report


def _stratified_group_folds(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    folds = list(splitter.split(np.zeros((len(labels), 1)), labels, groups))
    for train, test in folds:
        if set(groups[train]) & set(groups[test]):
            raise AssertionError("seed leakage across grouped folds")
        if len(np.unique(labels[train])) < 2:
            raise ValueError("grouped training fold lacks a target class")
    return folds


def development_outer_folds(
    population: ObservationPopulation,
    *,
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    # Four physics-defined strata balance dose failure, unsafe treatment,
    # safe nonbenefit and safe benefit without exposing row identities.
    strata = np.zeros(len(population.rows), dtype=int)
    qualified = population.compact_dose == 1
    safe = qualified & (population.safety == 1)
    strata[qualified & ~safe] = 1
    strata[safe & (population.benefit == 0)] = 2
    strata[safe & (population.benefit == 1)] = 3
    return _stratified_group_folds(
        strata,
        population.groups,
        n_splits=n_splits,
        random_state=RANDOM_STATE,
    )


def _make_estimator(model_family: str):
    from counterfactual.idm_gate_b_model_screen import make_candidate_estimator

    return make_candidate_estimator(model_family, RANDOM_STATE)


def _probability_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    if len(np.unique(labels)) < 2:
        return {"available": False, "reason": "one_class", "case_count": len(labels)}
    return {
        "available": True,
        "case_count": len(labels),
        "positive_count": int(labels.sum()),
        "negative_count": int(len(labels) - labels.sum()),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
    }


def _select_model_inner(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    random_state: int,
) -> tuple[str, list[dict[str, Any]]]:
    from sklearn.base import clone

    folds = _stratified_group_folds(
        labels, groups, n_splits=4, random_state=random_state
    )
    reports = []
    for model_order, model_family in enumerate(MODEL_FAMILIES):
        probability = np.full(len(labels), np.nan, dtype=float)
        for train, test in folds:
            model = clone(_make_estimator(model_family))
            model.fit(matrix[train], labels[train])
            probability[test] = model.predict_proba(matrix[test])[:, 1]
        metrics = _probability_metrics(labels, probability)
        reports.append(
            {
                "model_family": model_family,
                "model_order": model_order,
                "metrics": metrics,
            }
        )
    selected = max(
        reports,
        key=lambda row: (
            float(row["metrics"].get("roc_auc", -1.0)),
            float(row["metrics"].get("average_precision", -1.0)),
            -int(row["model_order"]),
        ),
    )
    return str(selected["model_family"]), reports


def _leave_one_seed_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    records = []
    for group in np.unique(groups):
        keep = groups != group
        metrics = _probability_metrics(labels[keep], probabilities[keep])
        records.append({"excluded_seed_group": str(group), "metrics": metrics})
    aucs = [row["metrics"]["roc_auc"] for row in records if row["metrics"].get("available")]
    return {
        "records": records,
        "roc_auc_minimum": min(aucs) if aucs else None,
        "roc_auc_maximum": max(aucs) if aucs else None,
    }


def _target_oof(
    matrix: np.ndarray,
    feature_names: Sequence[str],
    labels: np.ndarray,
    eligible: np.ndarray,
    groups: np.ndarray,
    outer_folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    permutation_repeats: int = 3,
) -> dict[str, Any]:
    from sklearn.base import clone
    from sklearn.inspection import permutation_importance

    nested_probability = np.full(len(labels), np.nan, dtype=float)
    fixed_probability = {
        family: np.full(len(labels), np.nan, dtype=float) for family in MODEL_FAMILIES
    }
    fold_reports = []
    importances: dict[str, list[float]] = {name: [] for name in feature_names}
    for fold_index, (outer_train, outer_test) in enumerate(outer_folds):
        train = outer_train[eligible[outer_train]]
        test_eligible = outer_test[eligible[outer_test]]
        selected_model, inner_reports = _select_model_inner(
            matrix[train],
            labels[train],
            groups[train],
            random_state=RANDOM_STATE + fold_index,
        )
        model = clone(_make_estimator(selected_model))
        model.fit(matrix[train], labels[train])
        nested_probability[outer_test] = model.predict_proba(matrix[outer_test])[:, 1]
        for model_family in MODEL_FAMILIES:
            fixed = clone(_make_estimator(model_family))
            fixed.fit(matrix[train], labels[train])
            fixed_probability[model_family][outer_test] = fixed.predict_proba(
                matrix[outer_test]
            )[:, 1]

        fold_metrics = _probability_metrics(
            labels[test_eligible], nested_probability[test_eligible]
        )
        fold_reports.append(
            {
                "fold": fold_index,
                "train_case_count": len(train),
                "test_case_count": len(test_eligible),
                "train_seed_count": len(set(groups[train])),
                "test_seed_count": len(set(groups[test_eligible])),
                "selected_model_family": selected_model,
                "inner_model_comparison": inner_reports,
                "outer_metrics": fold_metrics,
            }
        )
        if (
            permutation_repeats > 0
            and len(test_eligible) >= 4
            and len(np.unique(labels[test_eligible])) == 2
        ):
            importance = permutation_importance(
                model,
                matrix[test_eligible],
                labels[test_eligible],
                scoring="roc_auc",
                n_repeats=permutation_repeats,
                random_state=RANDOM_STATE + fold_index,
            )
            for name, value in zip(feature_names, importance.importances_mean):
                importances[name].append(float(value))

    if not np.all(np.isfinite(nested_probability)):
        raise AssertionError("incomplete nested grouped predictions")
    fixed_reports = {}
    for model_family, probability in fixed_probability.items():
        if not np.all(np.isfinite(probability)):
            raise AssertionError("incomplete fixed-model grouped predictions")
        fixed_reports[model_family] = {
            "metrics": _probability_metrics(labels[eligible], probability[eligible]),
            "leave_one_seed_out_evaluation": _leave_one_seed_metrics(
                labels[eligible], probability[eligible], groups[eligible]
            ),
        }
    importance_rows = [
        {
            "feature": name,
            "held_out_permutation_auc_decrease_mean": float(np.mean(values)),
            "fold_count": len(values),
        }
        for name, values in importances.items()
        if values
    ]
    importance_rows.sort(
        key=lambda row: row["held_out_permutation_auc_decrease_mean"], reverse=True
    )
    return {
        "nested_probability_oof": nested_probability,
        "fixed_probability_oof": fixed_probability,
        "nested_metrics": _probability_metrics(
            labels[eligible], nested_probability[eligible]
        ),
        "nested_leave_one_seed_out_evaluation": _leave_one_seed_metrics(
            labels[eligible], nested_probability[eligible], groups[eligible]
        ),
        "folds": fold_reports,
        "fixed_model_comparison": fixed_reports,
        "held_out_permutation_importance": importance_rows,
    }


def _policy_metrics(
    population: ObservationPopulation,
    safety_probability: np.ndarray,
    benefit_probability: np.ndarray,
    *,
    evaluation_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    if evaluation_mask is None:
        evaluation_mask = np.ones(len(population.rows), dtype=bool)
    command = evaluation_mask & (safety_probability >= SAFETY_THRESHOLD) & (
        benefit_probability >= BENEFIT_THRESHOLD
    )
    count = int(command.sum())
    safe = int(population.safety[command].sum())
    operational_benefit = population.compact_dose & population.benefit
    available_benefit = operational_benefit[evaluation_mask]
    benefit = int(operational_benefit[command].sum())
    dose = int(population.compact_dose[command].sum())
    safe_and_dose = int(
        (population.safety[command] & population.compact_dose[command]).sum()
    )
    return {
        "thresholds": {
            "safety": SAFETY_THRESHOLD,
            "conditional_benefit": BENEFIT_THRESHOLD,
        },
        "command_case_count": count,
        "command_seed_count": len(set(population.groups[command])),
        "selected_safety_precision": safe / count if count else 0.0,
        "selected_safe_and_dose_precision": safe_and_dose / count if count else 0.0,
        "joint_benefit_precision": benefit / count if count else 0.0,
        "joint_benefit_recall": benefit / int(available_benefit.sum())
        if available_benefit.sum() else 0.0,
        "selected_dose_success_rate": dose / count if count else 0.0,
        "evaluation_case_count": int(evaluation_mask.sum()),
        "evaluation_seed_count": len(set(population.groups[evaluation_mask])),
        "feasible_always_wall_safety_rate": float(
            np.mean(population.safety[evaluation_mask])
        ),
        "feasible_always_wall_joint_benefit_rate": float(
            np.mean(operational_benefit[evaluation_mask])
        ),
        "command_case_ids": [
            str(row["case_id"]) for row, selected in zip(population.rows, command) if selected
        ],
    }


def _policy_variability(
    population: ObservationPopulation,
    safety_probability: np.ndarray,
    benefit_probability: np.ndarray,
    outer_folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    by_fold = []
    for fold_index, (_train, test) in enumerate(outer_folds):
        mask = np.zeros(len(population.rows), dtype=bool)
        mask[test] = True
        by_fold.append(
            {
                "fold": fold_index,
                **_policy_metrics(
                    population,
                    safety_probability,
                    benefit_probability,
                    evaluation_mask=mask,
                ),
            }
        )
    leave_seed_out = []
    for group in np.unique(population.groups):
        leave_seed_out.append(
            {
                "excluded_seed_group": str(group),
                **_policy_metrics(
                    population,
                    safety_probability,
                    benefit_probability,
                    evaluation_mask=population.groups != group,
                ),
            }
        )
    return {
        "outer_fold_policy_metrics": by_fold,
        "leave_one_seed_out_metric_sensitivity_without_refit": leave_seed_out,
    }


def family_model_screen(
    population: ObservationPopulation,
    families: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    outer_folds = development_outer_folds(population)
    targets = target_masks(population)
    reports = []
    prediction_payload: dict[str, Any] = {}
    for family_name, feature_names in families.items():
        matrix = matrix_for(population, feature_names)
        target_reports = {}
        target_predictions = {}
        for target_name, (eligible, labels) in targets.items():
            target_report = _target_oof(
                matrix,
                feature_names,
                labels,
                eligible,
                population.groups,
                outer_folds,
            )
            target_predictions[target_name] = target_report.pop("nested_probability_oof")
            target_report.pop("fixed_probability_oof")
            target_reports[target_name] = target_report
        policy = _policy_metrics(
            population,
            target_predictions["safety_admissible"],
            target_predictions["conditional_joint_benefit"],
        )
        reports.append(
            {
                "family": family_name,
                "feature_count": len(feature_names),
                "feature_names": list(feature_names),
                "targets": target_reports,
                "fixed_threshold_policy": policy,
                "fixed_threshold_policy_variability": _policy_variability(
                    population,
                    target_predictions["safety_admissible"],
                    target_predictions["conditional_joint_benefit"],
                    outer_folds,
                ),
            }
        )
        prediction_payload[family_name] = {
            name: values.tolist() for name, values in target_predictions.items()
        }
    return {
        "families": reports,
        "prediction_payload": prediction_payload,
        "outer_folds": [
            {
                "fold": index,
                "train_seed_groups": sorted(set(str(value) for value in population.groups[train])),
                "test_seed_groups": sorted(set(str(value) for value in population.groups[test])),
            }
            for index, (train, test) in enumerate(outer_folds)
        ],
    }


def nested_feature_model_selection(
    population: ObservationPopulation,
    families: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Select feature family and small model only inside each outer seed fold."""
    from sklearn.base import clone

    outer_folds = development_outer_folds(population)
    targets = target_masks(population)
    matrices = {
        family: matrix_for(population, names) for family, names in families.items()
    }
    target_reports = {}
    target_predictions = {}
    for target_name, (eligible, labels) in targets.items():
        probability = np.full(len(labels), np.nan, dtype=float)
        fold_reports = []
        for fold_index, (outer_train, outer_test) in enumerate(outer_folds):
            train = outer_train[eligible[outer_train]]
            candidates = []
            for family_order, (family_name, names) in enumerate(families.items()):
                _selected, model_reports = _select_model_inner(
                    matrices[family_name][train],
                    labels[train],
                    population.groups[train],
                    random_state=RANDOM_STATE + fold_index,
                )
                for model_report in model_reports:
                    candidates.append(
                        {
                            "family": family_name,
                            "family_order": family_order,
                            "feature_count": len(names),
                            **model_report,
                        }
                    )
            selected = max(
                candidates,
                key=lambda row: (
                    float(row["metrics"].get("roc_auc", -1.0)),
                    float(row["metrics"].get("average_precision", -1.0)),
                    -int(row["family_order"]),
                    -int(row["model_order"]),
                ),
            )
            model = clone(_make_estimator(str(selected["model_family"])))
            matrix = matrices[str(selected["family"])]
            model.fit(matrix[train], labels[train])
            probability[outer_test] = model.predict_proba(matrix[outer_test])[:, 1]
            test = outer_test[eligible[outer_test]]
            fold_reports.append(
                {
                    "fold": fold_index,
                    "selected_family": selected["family"],
                    "selected_model_family": selected["model_family"],
                    "inner_metrics": selected["metrics"],
                    "outer_metrics": _probability_metrics(
                        labels[test], probability[test]
                    ),
                }
            )
        target_predictions[target_name] = probability
        target_reports[target_name] = {
            "nested_metrics": _probability_metrics(
                labels[eligible], probability[eligible]
            ),
            "nested_leave_one_seed_out_evaluation": _leave_one_seed_metrics(
                labels[eligible], probability[eligible], population.groups[eligible]
            ),
            "selected_family_counts": dict(
                Counter(row["selected_family"] for row in fold_reports)
            ),
            "selected_model_counts": dict(
                Counter(row["selected_model_family"] for row in fold_reports)
            ),
            "folds": fold_reports,
        }
    policy = _policy_metrics(
        population,
        target_predictions["safety_admissible"],
        target_predictions["conditional_joint_benefit"],
    )
    return {
        "targets": target_reports,
        "fixed_threshold_policy": policy,
        "fixed_threshold_policy_variability": _policy_variability(
            population,
            target_predictions["safety_admissible"],
            target_predictions["conditional_joint_benefit"],
            outer_folds,
        ),
        "prediction_payload": {
            name: values.tolist() for name, values in target_predictions.items()
        },
    }


def leave_one_family_out_ablation(
    population: ObservationPopulation,
    families: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    combined = set(families["G_combined_physical"])
    outer_folds = development_outer_folds(population)
    targets = target_masks(population)
    report = []
    for omitted in (
        "A_static_traffic",
        "B_temporal_speed_gap_density",
        "C_lane_resolved_traffic",
        "D_wall_geometry_placement_authority",
        "E_predecision_dose_assembly_projection",
        "F_follower_compression_rear_safety",
    ):
        names = tuple(sorted(combined - set(families[omitted])))
        matrix = matrix_for(population, names)
        targets_report = {}
        for target_name, (eligible, labels) in targets.items():
            result = _target_oof(
                matrix,
                names,
                labels,
                eligible,
                population.groups,
                outer_folds,
                permutation_repeats=0,
            )
            targets_report[target_name] = result["nested_metrics"]
        report.append(
            {
                "omitted_family": omitted,
                "remaining_feature_count": len(names),
                "targets": targets_report,
            }
        )
    return {"leave_one_family_out": report}


def shuffled_label_controls(
    population: ObservationPopulation,
    feature_names: Sequence[str],
    *,
    repeats: int = 20,
) -> dict[str, Any]:
    from sklearn.base import clone

    matrix = matrix_for(population, feature_names)
    outer_folds = development_outer_folds(population)
    targets = target_masks(population)
    rng = np.random.default_rng(RANDOM_STATE + 99)
    report = {}
    for target_name, (eligible, labels) in targets.items():
        target_report = {}
        for model_family in MODEL_FAMILIES:
            null_auc = []
            for _ in range(repeats):
                shuffled = labels.copy()
                shuffled_values = shuffled[eligible].copy()
                rng.shuffle(shuffled_values)
                shuffled[eligible] = shuffled_values
                probability = np.full(len(labels), np.nan, dtype=float)
                valid = True
                for train_all, test_all in outer_folds:
                    train = train_all[eligible[train_all]]
                    if len(np.unique(shuffled[train])) < 2:
                        valid = False
                        break
                    model = clone(_make_estimator(model_family))
                    model.fit(matrix[train], shuffled[train])
                    probability[test_all] = model.predict_proba(matrix[test_all])[:, 1]
                if valid:
                    value = _safe_auc(
                        shuffled[eligible], probability[eligible], direction_invariant=False
                    )
                    if value is not None:
                        null_auc.append(value)
            target_report[model_family] = {
                "shuffle_repeats_requested": repeats,
                "valid_shuffle_repeats": len(null_auc),
                "null_auc_mean": float(np.mean(null_auc)) if null_auc else None,
                "null_auc_95": (
                    [float(np.quantile(null_auc, 0.025)), float(np.quantile(null_auc, 0.975))]
                    if null_auc else None
                ),
            }
        report[target_name] = target_report
    return report


def nearest_neighbor_observability(
    population: ObservationPopulation,
    feature_names: Sequence[str],
) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    matrix = matrix_for(population, feature_names)
    folds = development_outer_folds(population)
    targets = target_masks(population)
    report = {}
    for target_name, (eligible, labels) in targets.items():
        nearest_label = np.full(len(labels), np.nan)
        five_probability = np.full(len(labels), np.nan)
        nearest_distance = np.full(len(labels), np.nan)
        opposite_pairs = []
        for train_all, test_all in folds:
            train = train_all[eligible[train_all]]
            test = test_all[eligible[test_all]]
            scaler = StandardScaler().fit(matrix[train])
            train_matrix = scaler.transform(matrix[train])
            test_matrix = scaler.transform(matrix[test])
            for local_index, row_index in enumerate(test):
                distances = np.sqrt(np.mean((train_matrix - test_matrix[local_index]) ** 2, axis=1))
                order = np.argsort(distances)
                nearest = train[order[0]]
                nearest_label[row_index] = labels[nearest]
                nearest_distance[row_index] = distances[order[0]]
                k = order[: min(5, len(order))]
                five_probability[row_index] = float(np.mean(labels[train[k]]))
                opposite = [index for index in order if labels[train[index]] != labels[row_index]]
                if opposite:
                    other = train[opposite[0]]
                    deltas = np.abs(test_matrix[local_index] - train_matrix[opposite[0]])
                    top = np.argsort(deltas)[::-1][:8]
                    opposite_pairs.append(
                        {
                            "case_id": str(population.rows[row_index]["case_id"]),
                            "opposite_case_id": str(population.rows[other]["case_id"]),
                            "seed": int(population.rows[row_index]["seed"]),
                            "opposite_seed": int(population.rows[other]["seed"]),
                            "decision_step": int(population.rows[row_index]["decision_step"]),
                            "opposite_decision_step": int(population.rows[other]["decision_step"]),
                            "standardized_rms_distance": float(distances[opposite[0]]),
                            "label": int(labels[row_index]),
                            "opposite_label": int(labels[other]),
                            "largest_standardized_differences": [
                                {
                                    "feature": str(feature_names[index]),
                                    "absolute_standardized_difference": float(deltas[index]),
                                }
                                for index in top
                            ],
                        }
                    )
        indices = np.flatnonzero(eligible)
        auc = _safe_auc(
            labels[indices], five_probability[indices], direction_invariant=False
        )
        threshold = float(np.quantile(nearest_distance[indices], 0.25))
        close = indices[nearest_distance[indices] <= threshold]
        opposite_pairs.sort(key=lambda row: row["standardized_rms_distance"])
        report[target_name] = {
            "eligible_case_count": len(indices),
            "independent_seed_count": len(set(population.groups[indices])),
            "one_nearest_neighbor_accuracy": float(
                accuracy_score(labels[indices], nearest_label[indices])
            ),
            "five_neighbor_probability_auc": auc,
            "nearest_distance_first_quartile": threshold,
            "close_neighbor_opposite_label_fraction": float(
                np.mean(nearest_label[close] != labels[close])
            ) if len(close) else None,
            "closest_opposite_label_pairs": opposite_pairs[:20],
        }
    return report


def shadow_rollout_reconstructibility(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    row = next(
        value for value in rows
        if bool(value.get("predecision_supervisor", {}).get("admitted"))
    )
    timing = row.get("timing_scan") or {}
    feature_names = set((row.get("features") or {}).keys())
    raw_vehicle_state_saved = bool(
        row.get("predecision_vehicle_snapshot")
        or timing.get("vehicle_snapshot")
        or timing.get("per_vehicle_state")
    )
    candidate_observations = [
        {
            "name": "projected_rear_ttc_1_2_5_10s",
            "status": "not_recorded",
            "available_proxy": "preflight_min_rear_ttc_s_capped_at_t0",
        },
        {
            "name": "required_follower_deceleration_and_compression_rate",
            "status": "not_recorded",
            "available_proxy": "aggregate_preflight_min_acceleration_and_follower_gap",
        },
        {
            "name": "projected_minimum_front_and_rear_gaps",
            "status": "t0_only",
            "available_proxy": "preflight_min_front_gap_m_and_preflight_min_rear_gap_m",
        },
        {
            "name": "vehicles_expected_to_encounter_wall_and_per_lane_gap_ttc_quantiles",
            "status": "not_recorded",
            "available_proxy": "lane_counts_and_aggregate_minimum_ttc_only",
        },
        {
            "name": "release_safety_margin",
            "status": "partial_proxy",
            "available_proxy": "obs_v2_dose_release_boundary_margin_m_without_future_follower_state",
        },
        {
            "name": "predicted_assembly_and_speed_convergence_time",
            "status": "not_recorded",
            "available_proxy": "predicted_span_and_projected_duration_only",
        },
        {
            "name": "projected_member_span",
            "status": "recorded",
            "available_proxy": "preflight_predicted_max_span_m_and_terminal_span_m",
        },
        {
            "name": "lane_coverage_stability_and_replacement_cav_availability",
            "status": "partial_proxy",
            "available_proxy": "lagged_band_coverage_and_placement_counts_not_member_specific",
        },
        {
            "name": "command_saturation_and_realized_authority_projection",
            "status": "not_recorded",
            "available_proxy": "preflight_min_acceleration_mps2",
        },
        {
            "name": "upstream_arrival_minus_downstream_discharge",
            "status": "derivable_partial",
            "available_proxy": "obs_v2_dose_upstream_flow_60s_vph_minus_postmerge_discharge_rate_60s",
        },
        {
            "name": "cumulative_excess_demand_20_60s",
            "status": "not_exactly_recorded",
            "available_proxy": "ramp_arrival_and_storage_boundary_terms_missing",
        },
        {
            "name": "downstream_storage",
            "status": "recorded_proxy",
            "available_proxy": "postmerge_vehicle_count_density_and_occupancy_proxy",
        },
        {
            "name": "shockwave_direction_and_speed",
            "status": "direction_proxy_only",
            "available_proxy": "spatial_speed_gradient_and_temporal_speed_slopes",
        },
        {
            "name": "compression_persistence_autocorrelation",
            "status": "recorded_aggregate",
            "available_proxy": "premerge_speed_and_gap_autocorrelation",
        },
        {
            "name": "projected_wall_induced_inflow_reduction",
            "status": "not_recorded",
            "available_proxy": None,
        },
    ]
    return {
        "can_compute_exact_shadow_rollout_from_saved_rows": raw_vehicle_state_saved,
        "saved_row_has_raw_predecision_vehicle_state": raw_vehicle_state_saved,
        "saved_timing_scan_keys": sorted(timing),
        "saved_scalar_feature_count": sum(
            _number(value) is not None for value in (row.get("features") or {}).values()
        ),
        "candidate_observations": candidate_observations,
        "required_new_predecision_extractor": {
            "latest_timestamp": "declared_decision_before_real_branch_command",
            "state": [
                "per-vehicle road lane position speed acceleration length and type in the declared influence region",
                "leader and follower identity gap relative speed and IDM parameters",
                "frozen damping command and projected command authority for each candidate",
                "candidate wall members same-lane reserves and release-boundary distance",
                "upstream ramp merge and downstream detector counts over the frozen causal history",
            ],
            "prohibited_persisted_inputs": [
                "actual treatment dose",
                "actual completion or release reason",
                "actual treatment outcome",
                "oracle label",
                "counterfactual effect",
            ],
        },
        "deterministic_shadow_rollout_protocol": {
            "planning_horizon_s": 10.0,
            "report_times_s": [1.0, 2.0, 5.0, 10.0],
            "branch_source": "cloned_decision_state_with_frozen_routes_and_rng",
            "arms": [
                "frozen_damping_shadow",
                "identical_damping_plus_frozen_v5_4_wall_shadow",
            ],
            "required_outputs": [
                "projected_duration_fraction",
                "projected_assembly_success_and_time_s",
                "projected_speed_convergence_time_s",
                "projected_member_span_max_m",
                "projected_lane_coverage_min_fraction",
                "projected_min_front_and_rear_ttc_s_at_1_2_5_10s",
                "projected_min_front_and_rear_gap_m",
                "projected_follower_compression_rate_mps",
                "projected_required_follower_deceleration_mps2",
                "projected_command_saturation_fraction",
                "projected_replacement_available_by_lane",
                "projected_wall_induced_inflow_reduction_vph",
            ],
            "causal_guard": (
                "The shadow simulation finishes before the real branch action and may "
                "not read the realized experimental counterfactual."
            ),
        },
        "current_feature_presence_checks": {
            "predicted_span": "preflight_predicted_max_span_m" in feature_names,
            "postmerge_storage": "obs_v2_postmerge_occupancy_proxy" in feature_names,
            "gap_autocorrelation": "obs_v2_premerge_gap_lag1_autocorr_60s" in feature_names,
        },
    }


__all__ = [
    "ObservationPopulation",
    "audit_features",
    "current_model_error_analysis",
    "development_outer_folds",
    "family_model_screen",
    "feature_families",
    "leave_one_family_out_ablation",
    "matrix_for",
    "nearest_neighbor_observability",
    "nested_feature_model_selection",
    "population_integrity",
    "prepare_population",
    "shuffled_label_controls",
    "shadow_rollout_reconstructibility",
    "target_masks",
    "univariate_information",
]
