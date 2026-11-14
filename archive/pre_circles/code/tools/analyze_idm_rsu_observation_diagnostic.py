"""Matched development-only audit of padded versus live classic RSU fields.

The tool reuses frozen P30 outcomes. It never generates treatment branches and
never permits postdecision values in model inputs. Outer evaluation is grouped
leave-one-seed-out; classification thresholds are selected by an inner grouped
cross-validation using only the outer training fold.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np

from experiments.io import atomic_write_json
from tools.audit_idm_phase5_observation_sufficiency import auroc, evaluate


SCHEMA = "idm_rsu_observation_diagnostic/1"
PHYSICAL_OBSERVATION_DIM = 134
CLASSIC_RSU = slice(9, 17)
PADDED_CLASSIC_RSU = np.asarray(
    [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32
)
AUTHORITY_FEATURE_NAMES = (
    "runway_m",
    "predicted_dose_s",
    "terminal_rear_headroom_m",
    "target_speed_difference_mps",
    "preview_projection_margin_mps2",
)
PRIMARY_MODEL = "regularized_logistic_nested_threshold_v1"


def _finite(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_authority(candidate: dict, observation: np.ndarray) -> list[float]:
    previews = candidate.get("terminal_speed_previews") or {}
    p30 = previews.get("p30") or {}
    desired = _finite(p30.get("desired_acceleration_mps2"))
    projected = _finite(p30.get("projected_acceleration_mps2"), desired)
    reference = _finite(
        p30.get("reference_speed_mps"),
        _finite(candidate.get("preview_reference_speed_mps")),
    )
    return [
        _finite(candidate.get("runway_m")),
        _finite(candidate.get("predicted_dose_s")),
        _finite(candidate.get("prospective_rear_headroom_m"), 250.0),
        _finite(observation[1]) * 35.0 - reference,
        projected - desired,
    ]


def _classic_is_padded(observation: np.ndarray) -> bool:
    return bool(
        np.array_equal(
            observation[CLASSIC_RSU].astype(np.float32), PADDED_CLASSIC_RSU
        )
    )


def _nonclassic_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(
        np.array_equal(left[: CLASSIC_RSU.start], right[: CLASSIC_RSU.start])
        and np.array_equal(left[CLASSIC_RSU.stop :], right[CLASSIC_RSU.stop :])
    )


def load_matched_rows(p30_roots: list[Path], live_roots: list[Path]) -> list[dict]:
    if len(p30_roots) != len(live_roots):
        raise ValueError("P30 and live root lists must have equal length")
    rows: list[dict] = []
    for p30_root, live_root in zip(p30_roots, live_roots):
        p30_root = p30_root.resolve()
        live_root = live_root.resolve()
        summary = _load_json(p30_root / "summary.json")
        p30_manifest = _load_json(p30_root / "manifest.json")
        live_manifest = _load_json(live_root / "manifest.json")
        demand = int(summary["protocol"]["demand"])
        recurrence = int(summary["protocol"]["decision_index"])
        if int(live_manifest["demand"]) != demand:
            raise RuntimeError("live diagnostic demand differs from P30 source")
        if int(live_manifest["decision_index"]) != recurrence:
            raise RuntimeError("live diagnostic recurrence differs from P30 source")
        if live_manifest.get("decision_source_path") != str(p30_root):
            raise RuntimeError("live diagnostic does not point to the supplied P30 source")
        if live_manifest.get("decision_source_purpose") != "rsu_observation_replay":
            raise RuntimeError("live diagnostic is not an observation-only replay")
        if live_manifest.get("action_set") != p30_manifest.get("action_set"):
            raise RuntimeError("treatment action set changed in observation replay")

        live_by_seed = {
            int(path.stem.split("seed", 1)[1]): _load_json(path)
            for path in (live_root / "discoveries").glob(f"m{demand}.seed*.json")
        }
        for case in summary["cases"]:
            seed = int(case["seed"])
            source = _load_json(
                p30_root / "discoveries" / f"m{demand}.seed{seed}.json"
            )
            source_decision = source.get("decision")
            live = live_by_seed.get(seed)
            live_decision = None if live is None else live.get("decision")
            if source_decision is None:
                if live_decision is not None:
                    raise RuntimeError(f"live replay invented missing recurrence seed{seed}")
                continue
            if live_decision is None:
                raise RuntimeError(f"live replay lost source recurrence seed{seed}")
            source_candidate = source_decision["candidates"][0]
            live_candidate = live_decision["candidates"][0]
            source_id = str(source_candidate["vehicle_id"])
            live_id = str(live_candidate["vehicle_id"])
            source_obs = np.asarray(
                source_decision["candidate_observations"][source_id][
                    :PHYSICAL_OBSERVATION_DIM
                ],
                dtype=np.float32,
            )
            live_obs = np.asarray(
                live_decision["candidate_observations"][live_id][
                    :PHYSICAL_OBSERVATION_DIM
                ],
                dtype=np.float32,
            )
            if source_obs.size != PHYSICAL_OBSERVATION_DIM:
                raise RuntimeError("source observation dimension drifted")
            if live_obs.size != PHYSICAL_OBSERVATION_DIM:
                raise RuntimeError("live observation dimension drifted")
            matched = live_decision.get("matched_decision_source") or {}
            metadata = (live_decision.get("candidate_observation_metadata") or {}).get(
                live_id, {}
            )
            effect = next(
                row for row in case["effects"] if int(row["candidate_slot"]) == 1
            )
            identity = {
                "same_seed": int(live["seed"]) == seed,
                "same_recurrence": int(live_decision["decision_index"]) == recurrence,
                "same_decision_step": int(live_decision["decision_step"])
                == int(source_decision["decision_step"]),
                "same_candidate": source_id == live_id,
                "same_physical_state_fingerprint": bool(
                    matched.get("physical_state_identity")
                ),
                "same_treatment_fingerprint": bool(matched.get("treatment_identity")),
                "same_nonclassic_observation": _nonclassic_equal(source_obs, live_obs)
                and bool(matched.get("nonclassic_observation_identity")),
            }
            if not all(identity.values()):
                raise RuntimeError(
                    f"physical observation replay identity failed seed{seed} r{recurrence}: "
                    f"{identity}"
                )
            rows.append(
                {
                    "demand": demand,
                    "seed": seed,
                    "recurrence": recurrence,
                    "decision_step": int(source_decision["decision_step"]),
                    "vehicle_id": source_id,
                    "distance_to_merge_m": float(
                        source_candidate["distance_to_merge_m"]
                    ),
                    "runway_m": float(source_candidate["runway_m"]),
                    "source_classic_rsu_padded": _classic_is_padded(source_obs),
                    "live_classic_rsu_padded": _classic_is_padded(live_obs),
                    "outcome": str(effect["classification"]),
                    "qualified": bool(effect["constraints_pass"]),
                    "effect": float(effect["safety_composite_reference_units"]),
                    "padded_observation": source_obs.tolist(),
                    "live_observation": live_obs.tolist(),
                    "authority_features": _candidate_authority(
                        source_candidate, source_obs
                    ),
                    "authority_feature_names": list(AUTHORITY_FEATURE_NAMES),
                    "identity": identity,
                    "rsu_metadata": metadata,
                }
            )
    return sorted(rows, key=lambda row: (row["recurrence"], row["seed"]))


def _sample_weights(labels: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(labels), dtype=float)
    for seed in np.unique(seeds):
        selected = seeds == seed
        weights[selected] = 1.0 / max(int(selected.sum()), 1)
    for label in (0, 1):
        selected = labels == label
        total = float(weights[selected].sum())
        if total > 0.0:
            weights[selected] *= 0.5 / total
    return weights


def _standardize(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(train, axis=0)
    std = np.nanstd(train, axis=0)
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=1.0)
    std[std < 1e-6] = 1.0
    return (
        np.nan_to_num((train - mean) / std, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num((test - mean) / std, nan=0.0, posinf=0.0, neginf=0.0),
    )


def _regularized_logistic_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seeds_train: np.ndarray,
    x_test: np.ndarray,
) -> np.ndarray:
    if len(np.unique(y_train)) < 2:
        return np.full(len(x_test), float(np.mean(y_train)), dtype=float)
    train, test = _standardize(x_train, x_test)
    train = np.column_stack([train, np.ones(len(train))])
    test = np.column_stack([test, np.ones(len(test))])
    weights = _sample_weights(y_train, seeds_train)
    coefficients = np.zeros(train.shape[1], dtype=float)
    for _ in range(2500):
        logits = np.clip(train @ coefficients, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        gradient = train.T @ (weights * (probability - y_train))
        gradient[:-1] += 0.02 * coefficients[:-1]
        coefficients -= 0.05 * gradient
    return 1.0 / (1.0 + np.exp(-np.clip(test @ coefficients, -30.0, 30.0)))


def _boosted_stumps_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seeds_train: np.ndarray,
    x_test: np.ndarray,
) -> np.ndarray:
    if len(np.unique(y_train)) < 2:
        return np.full(len(x_test), float(np.mean(y_train)), dtype=float)
    train, test = _standardize(x_train, x_test)
    weights = _sample_weights(y_train, seeds_train)
    prior = float(np.clip(np.sum(weights * y_train) / np.sum(weights), 1e-4, 1 - 1e-4))
    train_logit = np.full(len(train), math.log(prior / (1.0 - prior)), dtype=float)
    test_logit = np.full(len(test), train_logit[0], dtype=float)
    learning_rate = 0.05
    for _ in range(60):
        probability = 1.0 / (1.0 + np.exp(-np.clip(train_logit, -30.0, 30.0)))
        gradient = weights * (y_train - probability)
        hessian = weights * probability * (1.0 - probability)
        best: tuple[float, int, float, float, float] | None = None
        for feature in range(train.shape[1]):
            unique = np.unique(train[:, feature])
            if unique.size < 2:
                continue
            thresholds = 0.5 * (unique[:-1] + unique[1:])
            for threshold in thresholds:
                left = train[:, feature] <= threshold
                right = ~left
                if not left.any() or not right.any():
                    continue
                g_left, h_left = gradient[left].sum(), hessian[left].sum()
                g_right, h_right = gradient[right].sum(), hessian[right].sum()
                value_left = float(g_left / (h_left + 1.0))
                value_right = float(g_right / (h_right + 1.0))
                gain = float(
                    g_left * g_left / (h_left + 1.0)
                    + g_right * g_right / (h_right + 1.0)
                )
                candidate = (gain, feature, float(threshold), value_left, value_right)
                if best is None or candidate > best:
                    best = candidate
        if best is None or best[0] <= 1e-12:
            break
        _, feature, threshold, value_left, value_right = best
        train_update = np.where(
            train[:, feature] <= threshold, value_left, value_right
        )
        test_update = np.where(
            test[:, feature] <= threshold, value_left, value_right
        )
        train_logit += learning_rate * np.clip(train_update, -2.0, 2.0)
        test_logit += learning_rate * np.clip(test_update, -2.0, 2.0)
    return 1.0 / (1.0 + np.exp(-np.clip(test_logit, -30.0, 30.0)))


MODELS: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = {
    PRIMARY_MODEL: _regularized_logistic_predict,
    "gradient_boosted_stumps_v1": _boosted_stumps_predict,
}


def _balanced_accuracy(labels: np.ndarray, active: np.ndarray) -> float:
    helpful = labels == 1
    harmful = labels == 0
    if not helpful.any() or not harmful.any():
        return float("nan")
    return 0.5 * (float(np.mean(active[helpful])) + float(np.mean(~active[harmful])))


def _select_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    candidates = sorted(
        set(np.linspace(0.05, 0.95, 19).tolist())
        | set(float(value) for value in scores)
    )
    best = None
    for threshold in candidates:
        active = scores >= threshold
        balanced = _balanced_accuracy(labels, active)
        harmful_recall = float(np.mean(~active[labels == 0]))
        helpful_recall = float(np.mean(active[labels == 1]))
        key = (balanced, harmful_recall, helpful_recall, threshold)
        if best is None or key > best[0]:
            best = (key, threshold)
    return float(best[1]) if best is not None else 0.5


def _inner_threshold(
    features: np.ndarray,
    labels: np.ndarray,
    seeds: np.ndarray,
    predictor: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
) -> float:
    scores = np.zeros(len(labels), dtype=float)
    for held_seed in sorted(set(seeds.tolist())):
        test = seeds == held_seed
        train = ~test
        scores[test] = predictor(
            features[train], labels[train], seeds[train], features[test]
        )
    return _select_threshold(scores, labels)


def _calibration(scores: np.ndarray, labels: np.ndarray) -> tuple[float, list[dict]]:
    bins = []
    weighted_error = 0.0
    for lower, upper in ((0.0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.000001)):
        selected = (scores >= lower) & (scores < upper)
        if not selected.any():
            continue
        mean_score = float(np.mean(scores[selected]))
        observed = float(np.mean(labels[selected]))
        fraction = float(np.mean(selected))
        weighted_error += fraction * abs(mean_score - observed)
        bins.append(
            {
                "lower": lower,
                "upper": min(upper, 1.0),
                "count": int(selected.sum()),
                "mean_predicted_helpful_probability": mean_score,
                "observed_helpful_fraction": observed,
            }
        )
    return weighted_error, bins


def _metrics(scores: np.ndarray, labels: np.ndarray, active: np.ndarray) -> dict:
    harmful = labels == 0
    helpful = labels == 1
    calibration_error, calibration_bins = _calibration(scores, labels)
    return {
        "auroc": auroc(labels, scores),
        "balanced_accuracy": _balanced_accuracy(labels, active),
        "brier_score": float(np.mean((scores - labels) ** 2)),
        "expected_calibration_error_3bin": calibration_error,
        "harmful_case_recall": float(np.mean(~active[harmful])),
        "helpful_case_recall": float(np.mean(active[helpful])),
        "activation_coverage": float(np.mean(active)),
        "harmful_activations": int(np.sum(active & harmful)),
        "helpful_activations": int(np.sum(active & helpful)),
        "calibration_bins": calibration_bins,
    }


def _bootstrap_intervals(
    scores: np.ndarray,
    labels: np.ndarray,
    active: np.ndarray,
    seeds: np.ndarray,
    replicates: int,
) -> dict:
    rng = np.random.default_rng(20260829)
    unique = np.unique(seeds)
    values: dict[str, list[float]] = {
        "auroc": [],
        "balanced_accuracy": [],
        "brier_score": [],
        "harmful_case_recall": [],
        "helpful_case_recall": [],
        "activation_coverage": [],
    }
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.where(seeds == seed)[0] for seed in sampled])
        if len(np.unique(labels[indices])) < 2:
            continue
        row = _metrics(scores[indices], labels[indices], active[indices])
        for key in values:
            values[key].append(float(row[key]))
    return {
        key: {
            "lower_95": float(np.percentile(items, 2.5)) if items else float("nan"),
            "upper_95": float(np.percentile(items, 97.5)) if items else float("nan"),
            "valid_replicates": len(items),
        }
        for key, items in values.items()
    }


def _evaluate_nested(
    features: np.ndarray,
    labels: np.ndarray,
    seeds: np.ndarray,
    recurrences: np.ndarray,
    model_name: str,
    bootstrap_replicates: int,
) -> dict:
    predictor = MODELS[model_name]
    scores = np.zeros(len(labels), dtype=float)
    active = np.zeros(len(labels), dtype=bool)
    folds = []
    for held_seed in sorted(set(seeds.tolist())):
        test = seeds == held_seed
        train = ~test
        threshold = _inner_threshold(
            features[train], labels[train], seeds[train], predictor
        )
        scores[test] = predictor(
            features[train], labels[train], seeds[train], features[test]
        )
        active[test] = scores[test] >= threshold
        folds.append(
            {
                "held_seed": int(held_seed),
                "test_cases": int(test.sum()),
                "inner_selected_threshold": threshold,
            }
        )
    result = _metrics(scores, labels, active)
    result["folds"] = folds
    result["scores"] = scores.tolist()
    result["executed_gate"] = active.astype(int).tolist()
    result["clustered_seed_bootstrap_95ci"] = _bootstrap_intervals(
        scores, labels, active, seeds, bootstrap_replicates
    )
    result["coverage_harm_curve"] = [
        {
            "threshold": threshold,
            **{
                key: value
                for key, value in _metrics(
                    scores, labels, scores >= threshold
                ).items()
                if key
                in {
                    "activation_coverage",
                    "harmful_activations",
                    "helpful_activations",
                    "harmful_case_recall",
                    "helpful_case_recall",
                }
            },
        }
        for threshold in np.linspace(0.1, 0.9, 9)
    ]
    result["leave_one_seed_out_metric_sensitivity"] = [
        {
            "removed_seed": int(seed),
            "auroc": auroc(labels[seeds != seed], scores[seeds != seed])
            if len(np.unique(labels[seeds != seed])) == 2
            else float("nan"),
        }
        for seed in sorted(set(seeds.tolist()))
    ]
    result["leave_one_recurrence_out_metric_sensitivity"] = [
        {
            "removed_recurrence": int(recurrence),
            "auroc": auroc(
                labels[recurrences != recurrence], scores[recurrences != recurrence]
            )
            if len(np.unique(labels[recurrences != recurrence])) == 2
            else float("nan"),
        }
        for recurrence in sorted(set(recurrences.tolist()))
    ]
    return result


def _paired_delta_interval(
    left_scores: np.ndarray,
    right_scores: np.ndarray,
    labels: np.ndarray,
    seeds: np.ndarray,
    replicates: int,
) -> dict:
    rng = np.random.default_rng(20260829)
    unique = np.unique(seeds)
    deltas = []
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.where(seeds == seed)[0] for seed in sampled])
        if len(np.unique(labels[indices])) < 2:
            continue
        deltas.append(
            auroc(labels[indices], right_scores[indices])
            - auroc(labels[indices], left_scores[indices])
        )
    return {
        "point_delta": auroc(labels, right_scores) - auroc(labels, left_scores),
        "lower_95": float(np.percentile(deltas, 2.5)) if deltas else float("nan"),
        "upper_95": float(np.percentile(deltas, 97.5)) if deltas else float("nan"),
        "valid_replicates": len(deltas),
    }


def _dominance_audit(
    padded_scores: np.ndarray,
    live_scores: np.ndarray,
    labels: np.ndarray,
    seeds: np.ndarray,
    recurrences: np.ndarray,
) -> dict:
    def removals(groups: np.ndarray, name: str) -> list[dict]:
        rows = []
        for value in sorted(set(groups.tolist())):
            selected = groups != value
            if len(np.unique(labels[selected])) < 2:
                delta = float("nan")
            else:
                delta = auroc(labels[selected], live_scores[selected]) - auroc(
                    labels[selected], padded_scores[selected]
                )
            rows.append({name: int(value), "live_minus_padded_auroc": delta})
        return rows

    seed_rows = removals(seeds, "removed_seed")
    recurrence_rows = removals(recurrences, "removed_recurrence")
    finite = [
        float(row["live_minus_padded_auroc"])
        for row in seed_rows + recurrence_rows
        if math.isfinite(float(row["live_minus_padded_auroc"]))
    ]
    return {
        "leave_one_seed_out": seed_rows,
        "leave_one_recurrence_out": recurrence_rows,
        "all_finite_removal_deltas_positive": bool(finite)
        and all(value > 0.0 for value in finite),
    }


def _baselines(rows: list[dict]) -> dict:
    labels = np.asarray(
        [1 if row["outcome"] == "helpful" else 0 for row in rows], dtype=int
    )
    always = np.ones(len(rows), dtype=bool)
    never = np.zeros(len(rows), dtype=bool)
    conservative = np.asarray(
        [row["authority_features"][1] >= 20.0 for row in rows], dtype=bool
    )

    def result(active: np.ndarray) -> dict:
        return {
            "activation_coverage": float(np.mean(active)),
            "harmful_activations": int(np.sum(active & (labels == 0))),
            "helpful_activations": int(np.sum(active & (labels == 1))),
            "harmful_case_recall": float(np.mean(~active[labels == 0])),
            "helpful_case_recall": float(np.mean(active[labels == 1])),
        }

    return {
        "always_activate": result(always),
        "never_activate": result(never),
        "conservative_predicted_dose_at_least_20s": result(conservative),
    }


def _classic_rsu_descriptive(rows: list[dict]) -> dict:
    names = (
        "pre_merge_count_density",
        "pre_merge_normalized_speed",
        "merge_count_density",
        "merge_normalized_speed",
        "ramp_count_density",
        "ramp_normalized_speed",
        "ramp_queue",
        "merge_outflow",
    )
    values = np.asarray(
        [row["live_observation"][CLASSIC_RSU] for row in rows], dtype=float
    )
    labels = np.asarray(
        [1 if row["outcome"] == "helpful" else 0 for row in rows], dtype=int
    )
    effects = np.asarray([row["effect"] for row in rows], dtype=float)
    result = {}
    for index, name in enumerate(names):
        feature = values[:, index]
        raw_auroc = auroc(labels, feature)
        correlation = (
            float(np.corrcoef(feature, effects)[0, 1])
            if float(np.std(feature)) > 0.0
            else float("nan")
        )
        result[name] = {
            "helpful_mean": float(np.mean(feature[labels == 1])),
            "harmful_mean": float(np.mean(feature[labels == 0])),
            "helpful_minus_harmful_mean": float(
                np.mean(feature[labels == 1]) - np.mean(feature[labels == 0])
            ),
            "univariate_auroc_as_recorded_direction": raw_auroc,
            "best_posthoc_direction_auroc_diagnostic_only": max(
                raw_auroc, auroc(labels, -feature)
            ),
            "pearson_correlation_with_paired_effect": correlation,
        }
    return result


def analyze(
    p30_roots: list[Path],
    live_roots: list[Path],
    bootstrap_replicates: int = 2000,
) -> dict:
    rows = load_matched_rows(p30_roots, live_roots)
    nonneutral = [
        row
        for row in rows
        if row["qualified"] and row["outcome"] in {"helpful", "harmful"}
    ]
    if not nonneutral:
        raise RuntimeError("no qualified nonneutral P30 cases")
    labels = np.asarray(
        [1 if row["outcome"] == "helpful" else 0 for row in nonneutral], dtype=int
    )
    seeds = np.asarray([row["seed"] for row in nonneutral], dtype=int)
    recurrences = np.asarray([row["recurrence"] for row in nonneutral], dtype=int)
    padded = np.asarray([row["padded_observation"] for row in nonneutral], dtype=float)
    live = np.asarray([row["live_observation"] for row in nonneutral], dtype=float)
    authority = np.asarray([row["authority_features"] for row in nonneutral], dtype=float)
    families = {
        "padded_current_observation_134": padded,
        "live_classic_rsu_observation_134": live,
        "live_classic_rsu_plus_authority_139": np.column_stack([live, authority]),
    }
    model_results = {
        model_name: {
            family_name: _evaluate_nested(
                values,
                labels,
                seeds,
                recurrences,
                model_name,
                bootstrap_replicates,
            )
            for family_name, values in families.items()
        }
        for model_name in MODELS
    }
    # Preserve the previous fixed-0.5 weighted-logistic evaluator as a direct
    # comparison; it is not used for the new primary pass/fail decision.
    existing_comparison = {
        name: evaluate(values, labels, seeds, np.full(len(labels), 2000), slice(None))
        for name, values in families.items()
    }
    primary = model_results[PRIMARY_MODEL]
    padded_primary = primary["padded_current_observation_134"]
    live_primary = primary["live_classic_rsu_observation_134"]
    delta_interval = _paired_delta_interval(
        np.asarray(padded_primary["scores"]),
        np.asarray(live_primary["scores"]),
        labels,
        seeds,
        bootstrap_replicates,
    )
    dominance = _dominance_audit(
        np.asarray(padded_primary["scores"]),
        np.asarray(live_primary["scores"]),
        labels,
        seeds,
        recurrences,
    )
    gates = {
        "all_matched_physical_treatment_identity": all(
            all(row["identity"].values()) for row in rows
        ),
        "all_29_outside_legacy_range_are_padded": (
            sum(row["distance_to_merge_m"] > 500.0 for row in rows) == 29
            and sum(row["source_classic_rsu_padded"] for row in rows) == 29
        ),
        "all_diagnostic_candidates_have_live_classic_rsu": all(
            not row["live_classic_rsu_padded"]
            and bool(row["rsu_metadata"].get("rsu_available"))
            for row in rows
        ),
        "live_rsu_auroc_improves_by_at_least_0p10": (
            live_primary["auroc"] - padded_primary["auroc"] >= 0.10
        ),
        "live_rsu_grouped_auroc_at_least_0p65": live_primary["auroc"] >= 0.65,
        "live_rsu_balanced_accuracy_at_least_0p60": (
            live_primary["balanced_accuracy"] >= 0.60
        ),
        "gate_reduces_harmful_activations": (
            live_primary["harmful_activations"]
            < int(np.sum(labels == 0))
        ),
        "gate_retains_nontrivial_treatment_coverage": (
            live_primary["activation_coverage"] >= 0.30
            and live_primary["helpful_case_recall"] >= 0.50
        ),
        "improvement_not_dominated_by_one_seed_or_recurrence": dominance[
            "all_finite_removal_deltas_positive"
        ],
    }
    cutoff_rows = [
        {
            key: row[key]
            for key in (
                "demand",
                "seed",
                "recurrence",
                "decision_step",
                "vehicle_id",
                "distance_to_merge_m",
                "runway_m",
                "source_classic_rsu_padded",
                "live_classic_rsu_padded",
                "outcome",
                "qualified",
                "identity",
                "rsu_metadata",
            )
        }
        for row in rows
    ]
    return {
        "schema": SCHEMA,
        "development_only": True,
        "inputs": {
            "p30": [str(path.resolve()) for path in p30_roots],
            "live_rsu": [str(path.resolve()) for path in live_roots],
        },
        "forbidden_inputs": [
            "future outcome as feature",
            "postdecision dose",
            "postdecision traffic state",
            "seed",
            "filename",
        ],
        "cutoff_audit": {
            "case_count": len(rows),
            "outside_500m_count": sum(
                row["distance_to_merge_m"] > 500.0 for row in rows
            ),
            "padded_classic_rsu_count": sum(
                row["source_classic_rsu_padded"] for row in rows
            ),
            "live_diagnostic_classic_rsu_count": sum(
                not row["live_classic_rsu_padded"] for row in rows
            ),
            "maximum_candidate_distance_to_merge_m": max(
                row["distance_to_merge_m"] for row in rows
            ),
            "by_recurrence": {
                str(recurrence): {
                    "cases": sum(row["recurrence"] == recurrence for row in rows),
                    "outside_500m": sum(
                        row["recurrence"] == recurrence
                        and row["distance_to_merge_m"] > 500.0
                        for row in rows
                    ),
                }
                for recurrence in sorted({row["recurrence"] for row in rows})
            },
            "qualified_outcomes": dict(
                Counter(row["outcome"] for row in rows if row["qualified"])
            ),
            "rows": cutoff_rows,
        },
        "model_contract": {
            "primary_model": PRIMARY_MODEL,
            "outer_split": "leave-one-seed-out",
            "threshold_selection": "inner leave-one-seed-out training folds only",
            "primary_label": "helpful=1 versus harmful=0 on qualified nonneutral P30 cases",
            "case_count": len(nonneutral),
            "helpful_count": int(np.sum(labels == 1)),
            "harmful_count": int(np.sum(labels == 0)),
            "authority_feature_names": list(AUTHORITY_FEATURE_NAMES),
            "confidence_interval": (
                "95% percentile bootstrap clustered by seed over fixed outer-fold predictions"
            ),
            "nontrivial_coverage_floor": 0.30,
            "helpful_recall_floor": 0.50,
        },
        "models": model_results,
        "classic_rsu_field_descriptive_diagnostic": _classic_rsu_descriptive(
            nonneutral
        ),
        "existing_observation_analyzer_comparison": existing_comparison,
        "baselines": _baselines(nonneutral),
        "primary_live_minus_padded_auroc": delta_interval,
        "primary_improvement_dominance": dominance,
        "gates": gates,
        "live_existing_rsu_improves_observability": all(gates.values()),
        "claim_boundary": (
            "Small development-only grouped diagnostic; passing would justify a new "
            "corridor-profile proposal, not predictor verification or MAPPO training."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p30", type=Path, nargs="+", required=True)
    parser.add_argument("--live", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_replicates < 200:
        raise ValueError("at least 200 grouped bootstrap replicates are required")
    payload = analyze(args.p30, args.live, args.bootstrap_replicates)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "summary.json", payload)
    print(
        json.dumps(
            {
                "cutoff_audit": payload["cutoff_audit"],
                "gates": payload["gates"],
                "live_existing_rsu_improves_observability": payload[
                    "live_existing_rsu_improves_observability"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
