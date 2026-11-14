"""Limited-domain (5-10% penetration) fixed-P30 observation/value screen.

Causal, predecision, penetration-free features only; penetration is an
operating-domain admission rule applied outside the actor (frozen protocol
docs/IDM_FIXED_P30_LIMITED_DOMAIN_PROTOCOL.md).  The value screen reuses the
seed-grouped-ridge machinery of screen_idm_expected_option_value.py; the
classification screen is new.  Held development seeds 4129--4132 are opened
once, after model/threshold selection on 4121--4128.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.io import atomic_write_json
from idm.option_reward import IDM_BINARY_OPTION_REWARD_V1
from idm.paired_option_dataset import load_dataset
from tools.screen_idm_expected_option_value import (
    RIDGE_ALPHAS,
    _always_baselines,
    _cell_weights,
    _classification,
    _context,
    _threshold_candidates,
    choose_threshold,
    fit_ridge,
    grouped_oof_predictions,
    policy_metrics,
    predict_ridge,
)

PENETRATIONS = (0.05, 0.10)
DEMAND_GATE_MIN_CELLS = 2
GATE_MEAN_REWARD_MARGIN = 0.01
AUROC_TARGET = 0.70
BALACC_TARGET = 0.65


def feature_families(records: list[Any]) -> dict[str, np.ndarray]:
    """Causal penetration-free families; the domain rule keeps penetration
    outside the actor observation."""
    physical = np.asarray([record.observation for record in records], dtype=float)
    context = np.asarray(
        [option_context_without_penetration(record) for record in records], dtype=float
    )
    controller = physical[:, 121:134]
    zone_controller = np.column_stack([physical[:, 9:21], controller])
    return {
        "option_context_12": context,
        "zone_controller_option_37": np.column_stack([zone_controller, context]),
        "physical_134": physical,
        "physical_plus_option_146": np.column_stack([physical, context]),
    }


def option_context_without_penetration(record: Any) -> np.ndarray:
    return np.asarray(_context(record), dtype=float)[1:]


def _binary_labels(records: list[Any]):
    classes = np.asarray([_classification(record) for record in records])
    qualified = np.asarray(
        [record.constraints_pass and cls in ("helpful", "harmful") for record, cls in zip(records, classes)]
    )
    labels = np.asarray([1.0 if cls == "helpful" else 0.0 for cls in classes])
    return classes, labels, qualified


def classification_metrics(records: list[Any], scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    positives = float(np.sum(labels > 0))
    negatives = float(np.sum(labels <= 0))
    if positives == 0 or negatives == 0:
        return {
            "auroc": None,
            "average_precision": None,
            "balanced_accuracy": None,
            "positives": int(positives),
            "negatives": int(negatives),
            "sample_size": len(labels),
        }
    paired = 0.0
    positive_scores = scores[labels > 0]
    negative_scores = scores[labels <= 0]
    for pos in positive_scores:
        paired += float(np.sum(pos > negative_scores)) + 0.5 * float(
            np.sum(pos == negative_scores)
        )
    auroc = float(paired / (positives * negatives))
    order = np.argsort(-scores)
    order_pos = labels[order] > 0
    precisions = np.cumsum(order_pos) / np.arange(1, len(labels) + 1)
    average_precision = float(np.mean(precisions[order_pos]))
    thresholds = sorted(set(scores.tolist() + [float(np.mean(scores))]))
    best_bal = 0.0
    for value in thresholds:
        predicted = scores >= value
        tp = float(np.sum(predicted & (labels > 0)))
        fp = float(np.sum(predicted & (labels <= 0)))
        fn = float(np.sum(~predicted & (labels > 0)))
        tn = float(np.sum(~predicted & (labels <= 0)))
        tpr = tp / positives if positives else 0.0
        tnr = tn / negatives if negatives else 0.0
        best_bal = max(best_bal, 0.5 * (tpr + tnr))
    return {
        "auroc": auroc,
        "average_precision": average_precision,
        "balanced_accuracy": best_bal,
        "positives": int(positives),
        "negatives": int(negatives),
        "sample_size": len(labels),
    }


def fit_logistic(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    mean = np.mean(x, axis=0)
    scale = np.sqrt(np.maximum(np.var(x, axis=0), 1e-8))
    x_std = (x - mean) / scale
    design = np.column_stack([x_std, np.ones(len(x))])
    weights = np.zeros(design.shape[1])
    for _ in range(250):
        linear = design @ weights
        prob = 1.0 / (1.0 + np.exp(-np.clip(linear, -35.0, 35.0)))
        gradient = design.T @ (prob - y) + alpha * np.concatenate([weights[:-1], [0.0]])
        hessian = (design * (prob * (1.0 - prob))[:, None]).T @ design
        hessian += np.diag(np.concatenate([np.full(design.shape[1] - 1, alpha), [0.0]]))
        step = np.linalg.solve(hessian + np.eye(design.shape[1]) * 1e-6, gradient)
        weights = weights - step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return {"mean": mean, "scale": scale, "coefficients": weights}


def predict_logistic(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    design = np.column_stack([(x - model["mean"]) / model["scale"], np.ones(len(x))])
    return design @ model["coefficients"]


def screen_classification(
    train_seeds: set[int], validation_seeds: set[int], records: list[Any], values: dict[str, np.ndarray]
) -> dict[str, Any]:
    results = {}
    train_mask = np.asarray([record.seed in train_seeds for record in records], dtype=bool)
    val_mask = ~train_mask
    train_rows = [record for record, m in zip(records, train_mask) if m]
    val_rows = [record for record, m in zip(records, val_mask) if m]
    _, ytr, qtr = _binary_labels(train_rows)
    _, yval, qval = _binary_labels(val_rows)
    if not np.any(qtr) or not np.any(qval):
        return {
            "notes": "insufficient qualified helpful/harmful rows on one split",
            "train_qualified": int(np.sum(qtr)),
            "validation_qualified": int(np.sum(qval)),
        }
    for name, x in values.items():
        best = None
        for alpha in (0.01, 0.1, 1.0, 10.0):
            # alpha selection on seed-grouped training OOF only
            oof = np.zeros(len(train_rows), dtype=float)
            for seed in sorted(train_seeds):
                test = np.asarray([record.seed == seed for record in train_rows], dtype=bool)
                train = ~test
                if not np.any(test & qtr) or not np.any(train & qtr):
                    oof[test] = 0.0
                    continue
                model = fit_logistic(x[train_mask][train][qtr[train]], ytr[train][qtr[train]], alpha)
                oof[test] = predict_logistic(model, x[train_mask][test])
            qualified_train = [row for row, keep in zip(train_rows, qtr) if keep]
            tr_metrics = classification_metrics(qualified_train, oof[qtr], ytr[qtr])
            if best is None or (tr_metrics["auroc"] or -1.0) > (best["training_oof"]["auroc"] or -1.0):
                best = {"alpha": alpha, "training_oof": tr_metrics}
        model = fit_logistic(
            x[train_mask][qtr], ytr[qtr], float(best["alpha"])
        )
        held_scores = predict_logistic(model, x[val_mask][qval])
        qualified_val = [row for row, keep in zip(val_rows, qval) if keep]
        held = classification_metrics(qualified_val, held_scores, yval[qval])
        results[name] = {
            "selected_alpha": best["alpha"],
            "training_grouped_oof": best["training_oof"],
            "held": held,
            "targets": {"auroc": AUROC_TARGET, "balanced_accuracy": BALACC_TARGET},
            "passes": (held["auroc"] or 0.0) >= AUROC_TARGET and (held["balanced_accuracy"] or 0.0)
            >= BALACC_TARGET,
        }
    return results


def analyze(dataset: Path) -> dict[str, Any]:
    records, manifest = load_dataset(dataset, split=None)
    records = [record for record in records if float(record.penetration) in PENETRATIONS]
    train_seeds = set(int(seed) for seed in manifest["train_seeds"])
    validation_seeds = set(int(seed) for seed in manifest["validation_seeds"])
    train_records = [record for record in records if record.seed in train_seeds]
    validation_records = [record for record in records if record.seed in validation_seeds]
    values = feature_families(records)
    train_indices = np.asarray([record.seed in train_seeds for record in records])
    validation_indices = np.asarray([record.seed in validation_seeds for record in records])
    target_train = np.asarray([record.activate_reward for record in train_records], dtype=float)

    results = {}
    for name, x in values.items():
        x_train = x[train_indices]
        x_validation = x[validation_indices]
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
        model = fit_ridge(x_train, target_train, _cell_weights(train_records), float(selected["alpha"]))
        validation_scores = predict_ridge(model, x_validation)
        validation = policy_metrics(
            validation_records, validation_scores, float(selected["threshold"])
        )
        per_cell_positive = sum(
            row["mean_policy_reward"] > 0.0
            for row in _per_demand(validation_records, validation_scores, float(selected["threshold"])).values()
        )
        gates = {
            "validation_activation_coverage_at_least_20pct": validation["activation_coverage"] >= 0.20,
            "validation_mean_reward_positive": validation["mean_policy_reward"] > 0.0,
            "validation_helpful_more_frequent_than_harmful": validation["helpful_activations"]
            > validation["harmful_activations"],
            "validation_harmful_fraction_at_most_25pct": validation["harmful_fraction_of_activations"] <= 0.25,
            "validation_above_never_activate": validation["mean_policy_reward"] > 0.0,
            "validation_above_always_activate": validation["mean_policy_reward"]
            > _always_baselines(validation_records)["always_activate"]["mean_policy_reward"],
            "validation_above_domain_rule_by_margin": validation["mean_policy_reward"]
            - _always_baselines(validation_records)["always_activate"]["mean_policy_reward"]
            >= GATE_MEAN_REWARD_MARGIN,
            "validation_positive_value_in_more_than_one_demand_cell": per_cell_positive >= DEMAND_GATE_MIN_CELLS,
            "validation_no_service_or_safety_constraint_added": not any(
                not record.constraints_pass
                for record, score in zip(validation_records, validation_scores)
                if score >= float(selected["threshold"])
            ),
        }
        results[name] = {
            "selected_alpha": selected["alpha"],
            "selected_threshold": selected["threshold"],
            "selection_candidates": candidates,
            "validation": validation,
            "validation_per_demand": _per_demand(
                validation_records, validation_scores, float(selected["threshold"])
            ),
            "validation_positive_demand_cell_count": per_cell_positive,
            "gates": gates,
            "passes": all(gates.values()),
        }

    classification = screen_classification(
        train_seeds, validation_seeds, records, values
    )
    domain_record_counts = {
        "penetration_counts": {f"{p:.2f}": sum(1 for r in records if float(r.penetration) == p) for p in PENETRATIONS},
        "demand_counts": {str(d): sum(1 for r in records if int(r.demand) == d) for d in sorted({int(r.demand) for r in records})},
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "constraint_failed_records": sum(1 for r in records if not r.constraints_pass),
        "classification_counts": dict((c, sum(1 for r in records if _classification(r) == c)) for c in ("helpful", "harmful", "neutral")),
        "neutral_qualified_records": sum(
            1 for r in records if r.constraints_pass and _classification(r) == "neutral"
        ),
    }
    return {
        "schema": "idm_limited_domain_p30_screen/1",
        "protocol": "docs/IDM_FIXED_P30_LIMITED_DOMAIN_PROTOCOL.md",
        "development_only": True,
        "dataset": str(dataset.resolve()),
        "domain": {"penetrations": list(PENETRATIONS)},
        "train_seeds": sorted(train_seeds),
        "validation_seeds": sorted(validation_seeds),
        "domain_record_counts": domain_record_counts,
        "forbidden_actor_inputs": [
            "seed",
            "future observation",
            "realized dose",
            "treatment outcome",
            "reward",
            "release reason",
            "branch identity",
            "postdecision measurement",
            "penetration (admission rule outside the actor)",
        ],
        "model_selection": "seed-grouped training OOF only; penetration excluded from features",
        "validation_opened_after_selection": True,
        "train_baselines": _always_baselines(train_records),
        "validation_baselines": _always_baselines(validation_records),
        "classification_screen": classification,
        "value_screen": results,
        "authorized_value_families": [
            name for name, row in results.items() if row["passes"]
        ],
        "rl_authorized": any(row["passes"] for row in results.values()),
    }


def _per_demand(records: list[Any], scores: np.ndarray, threshold: float) -> dict[str, Any]:
    out = {}
    for demand in sorted({int(record.demand) for record in records}):
        mask = np.asarray([int(record.demand) == demand for record in records], dtype=bool)
        active = mask & (scores >= float(threshold))
        rewards = np.asarray([record.activate_reward for record in records], dtype=float)
        physical = np.asarray([record.physical_utility for record in records], dtype=float)
        coverage = float(np.mean(scores[mask] >= float(threshold)))
        out[str(demand)] = {
            "records": int(np.sum(mask)),
            "activations": int(np.sum(active)),
            "mean_policy_reward": (
                float(np.mean(rewards[active])) * coverage if np.any(active) else 0.0
            ),
            "mean_policy_physical_effect": (
                float(np.mean(physical[active])) * coverage if np.any(active) else 0.0
            ),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.dataset.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True)[:4000])


if __name__ == "__main__":
    main()
