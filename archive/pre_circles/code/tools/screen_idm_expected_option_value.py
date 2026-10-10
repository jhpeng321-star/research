"""Screen a conservative expected-value gate for the IDM P30 option.

This is a full-information contextual-bandit diagnostic: the paired replay
provides the reward of both abstention (zero) and activation for every row.
Model selection and the deployment threshold use grouped out-of-fold
predictions from the dataset's training seeds only.  The validation split is
opened exactly once after those choices are frozen.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from experiments.io import atomic_write_json
from idm.option_reward import IDM_BINARY_OPTION_REWARD_V1
from idm.paired_option_dataset import (
    load_dataset,
    option_context_from_candidate,
)


RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _context(record: Any) -> np.ndarray:
    discovery = _read(
        Path(record.source) / "discoveries" / f"m{record.demand}.seed{record.seed}.json"
    )["decision"]
    candidate = next(
        row for row in discovery["candidates"] if row["vehicle_id"] == record.vehicle_id
    )
    return np.asarray(option_context_from_candidate(candidate, record.penetration), dtype=float)


def feature_families(records: list[Any]) -> dict[str, np.ndarray]:
    physical = np.asarray([record.observation for record in records], dtype=float)
    context = np.asarray([_context(record) for record in records], dtype=float)
    controller = physical[:, 121:134]
    zone_controller = np.column_stack([physical[:, 9:21], controller])
    compact = np.column_stack([zone_controller, context])
    penetration = context[:, :1]
    # A small prespecified nonlinear family lets loading features have
    # different value at different penetrations without exposing demand.
    penetration_interactions = np.column_stack(
        [
            compact,
            penetration**2,
            penetration * zone_controller,
        ]
    )
    return {
        "option_context_13": context,
        "zone_controller_option_38": compact,
        "penetration_interactions_64": penetration_interactions,
        "physical_134": physical,
        "physical_plus_option_147": np.column_stack([physical, context]),
    }


def _cell_weights(records: list[Any]) -> np.ndarray:
    keys = [(round(record.penetration, 6), record.demand) for record in records]
    counts = Counter(keys)
    weights = np.asarray([1.0 / counts[key] for key in keys], dtype=float)
    return weights / float(np.mean(weights))


def fit_ridge(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray]:
    mean = np.average(x, axis=0, weights=weights)
    variance = np.average((x - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(np.maximum(variance, 1e-8))
    standardized = (x - mean) / scale
    design = np.column_stack([standardized, np.ones(len(x))])
    root_weights = np.sqrt(weights)[:, None]
    weighted_design = design * root_weights
    weighted_target = y * root_weights[:, 0]
    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_target,
    )
    return {"mean": mean, "scale": scale, "coefficients": coefficients}


def predict_ridge(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    standardized = (x - model["mean"]) / model["scale"]
    design = np.column_stack([standardized, np.ones(len(x))])
    return design @ model["coefficients"]


def _classification(record: Any) -> str:
    if record.constraints_pass and record.physical_utility > IDM_BINARY_OPTION_REWARD_V1.deadband:
        return "helpful"
    if (not record.constraints_pass) or (
        record.physical_utility < -IDM_BINARY_OPTION_REWARD_V1.deadband
    ):
        return "harmful"
    return "neutral"


def policy_metrics(records: list[Any], scores: np.ndarray, threshold: float) -> dict[str, Any]:
    active = scores >= float(threshold)
    rewards = np.asarray([record.activate_reward for record in records], dtype=float)
    physical = np.asarray([record.physical_utility for record in records], dtype=float)
    classes = np.asarray([_classification(record) for record in records])
    active_count = int(np.sum(active))
    denominator = max(active_count, 1)
    per_penetration = {}
    for penetration in sorted({record.penetration for record in records}):
        mask = np.asarray([record.penetration == penetration for record in records], dtype=bool)
        active_cell = active & mask
        cell_coverage = float(np.mean(active[mask]))
        per_penetration[f"{penetration:.2f}"] = {
            "records": int(np.sum(mask)),
            "activations": int(np.sum(active_cell)),
            "mean_policy_reward": (
                float(np.mean(rewards[active_cell])) * cell_coverage if np.any(active_cell) else 0.0
            ),
            "mean_policy_physical_effect": (
                float(np.mean(physical[active_cell])) * cell_coverage
                if np.any(active_cell)
                else 0.0
            ),
        }
    helpful = int(np.sum(active & (classes == "helpful")))
    harmful = int(np.sum(active & (classes == "harmful")))
    neutral = int(np.sum(active & (classes == "neutral")))
    return {
        "threshold": float(threshold),
        "records": len(records),
        "activations": active_count,
        "activation_coverage": active_count / len(records),
        "mean_policy_reward": float(np.mean(np.where(active, rewards, 0.0))),
        "mean_policy_physical_effect": float(np.mean(np.where(active, physical, 0.0))),
        "helpful_activations": helpful,
        "harmful_activations": harmful,
        "neutral_activations": neutral,
        "helpful_fraction_of_activations": helpful / denominator,
        "harmful_fraction_of_activations": harmful / denominator,
        "helpful_minus_harmful_fraction": (helpful - harmful) / denominator,
        "score_min": float(np.min(scores)),
        "score_mean": float(np.mean(scores)),
        "score_max": float(np.max(scores)),
        "per_penetration": per_penetration,
    }


def _threshold_candidates(scores: np.ndarray) -> list[float]:
    unique = np.unique(scores)
    boundaries = [float(unique[-1] + 1e-9), float(unique[0] - 1e-9)]
    boundaries.extend(float(value) for value in unique)
    return sorted(set(boundaries))


def choose_threshold(records: list[Any], scores: np.ndarray) -> tuple[float, dict]:
    candidates = [
        policy_metrics(records, scores, threshold) for threshold in _threshold_candidates(scores)
    ]
    admissible = [
        row
        for row in candidates
        if row["activation_coverage"] >= 0.20
        and row["harmful_fraction_of_activations"] <= 0.25
        and row["helpful_minus_harmful_fraction"] >= 0.15
    ]
    pool = admissible or candidates
    selected = max(
        pool,
        key=lambda row: (
            row["mean_policy_reward"],
            row["helpful_minus_harmful_fraction"],
            row["activation_coverage"],
        ),
    )
    selected = {**selected, "training_constraints_feasible": bool(admissible)}
    return float(selected["threshold"]), selected


def grouped_oof_predictions(records: list[Any], x: np.ndarray, alpha: float) -> np.ndarray:
    predictions = np.zeros(len(records), dtype=float)
    seeds = np.asarray([record.seed for record in records], dtype=int)
    target = np.asarray([record.activate_reward for record in records], dtype=float)
    for seed in sorted(set(seeds.tolist())):
        test = seeds == seed
        train = ~test
        model = fit_ridge(
            x[train],
            target[train],
            _cell_weights([record for index, record in enumerate(records) if train[index]]),
            alpha,
        )
        predictions[test] = predict_ridge(model, x[test])
    return predictions


def _always_baselines(records: list[Any]) -> dict[str, Any]:
    zeros = np.zeros(len(records), dtype=float)
    return {
        "always_abstain": policy_metrics(records, zeros, 1.0),
        "always_activate": policy_metrics(records, zeros, 0.0),
    }


def analyze(dataset: Path) -> dict[str, Any]:
    records, manifest = load_dataset(dataset, split=None)
    train_seeds = set(int(seed) for seed in manifest["train_seeds"])
    validation_seeds = set(int(seed) for seed in manifest["validation_seeds"])
    train_records = [record for record in records if record.seed in train_seeds]
    validation_records = [record for record in records if record.seed in validation_seeds]
    all_features = feature_families(records)
    train_indices = np.asarray([record.seed in train_seeds for record in records])
    validation_indices = np.asarray([record.seed in validation_seeds for record in records])
    target_train = np.asarray([record.activate_reward for record in train_records], dtype=float)
    results = {}
    for name, values in all_features.items():
        x_train = values[train_indices]
        x_validation = values[validation_indices]
        candidates = []
        for alpha in RIDGE_ALPHAS:
            oof = grouped_oof_predictions(train_records, x_train, alpha)
            threshold, metrics = choose_threshold(train_records, oof)
            candidates.append(
                {
                    "alpha": alpha,
                    "threshold": threshold,
                    "training_oof": metrics,
                    "oof_mse": float(np.mean((oof - target_train) ** 2)),
                }
            )
        selected = max(
            candidates,
            key=lambda row: (
                row["training_oof"]["training_constraints_feasible"],
                row["training_oof"]["mean_policy_reward"],
                -row["oof_mse"],
            ),
        )
        model = fit_ridge(
            x_train,
            target_train,
            _cell_weights(train_records),
            float(selected["alpha"]),
        )
        validation_scores = predict_ridge(model, x_validation)
        validation = policy_metrics(
            validation_records, validation_scores, float(selected["threshold"])
        )
        positive_penetrations = sum(
            row["mean_policy_reward"] > 0.0 for row in validation["per_penetration"].values()
        )
        gates = {
            "training_threshold_constraints_feasible": selected["training_oof"][
                "training_constraints_feasible"
            ],
            "validation_activation_coverage_at_least_20pct": validation["activation_coverage"]
            >= 0.20,
            "validation_mean_reward_over_0p02": validation["mean_policy_reward"] > 0.02,
            "validation_helpful_minus_harmful_at_least_15pp": validation[
                "helpful_minus_harmful_fraction"
            ]
            >= 0.15,
            "validation_harmful_fraction_at_most_25pct": validation[
                "harmful_fraction_of_activations"
            ]
            <= 0.25,
            "validation_positive_reward_in_four_penetrations": positive_penetrations >= 4,
        }
        results[name] = {
            "selected_alpha": selected["alpha"],
            "selected_threshold": selected["threshold"],
            "selection_candidates": candidates,
            "validation": validation,
            "validation_positive_penetration_count": positive_penetrations,
            "gates": gates,
            "passes": all(gates.values()),
        }
    return {
        "schema": "idm_expected_option_value_screen/1",
        "development_only": True,
        "dataset": str(dataset.resolve()),
        "dataset_sha256": manifest["dataset_sha256"],
        "train_seeds": sorted(train_seeds),
        "validation_seeds": sorted(validation_seeds),
        "forbidden_actor_inputs": [
            "configured demand",
            "seed",
            "future outcome",
            "realized dose",
            "reward",
        ],
        "model_selection": "seed-grouped training OOF only",
        "validation_opened_after_selection": True,
        "train_baselines": _always_baselines(train_records),
        "validation_baselines": _always_baselines(validation_records),
        "feature_families": results,
        "authorized_families": [name for name, row in results.items() if row["passes"]],
        "expected_value_actor_smoke_authorized": any(row["passes"] for row in results.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.dataset.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
