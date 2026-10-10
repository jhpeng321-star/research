"""Small, dependency-light models for the counterfactual activation gate."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class LogisticModel:
    """Standardized L2-regularized binary logistic regression."""

    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    intercept: float
    coefficients: np.ndarray
    l2: float
    iterations: int

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=float)
        standardized = (matrix - self.mean) / self.scale
        logits = self.intercept + standardized @ self.coefficients
        return _sigmoid(logits)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "standardization_mean": self.mean.tolist(),
            "standardization_scale": self.scale.tolist(),
            "intercept": self.intercept,
            "standardized_coefficients": {
                name: float(value) for name, value in zip(self.feature_names, self.coefficients)
            },
            "l2": self.l2,
            "iterations": self.iterations,
        }


@dataclass(frozen=True)
class RidgeModel:
    """Standardized L2-regularized regression for one continuous target."""

    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    intercept: float
    coefficients: np.ndarray
    l2: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=float)
        standardized = (matrix - self.mean) / self.scale
        return self.intercept + standardized @ self.coefficients

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "standardization_mean": self.mean.tolist(),
            "standardization_scale": self.scale.tolist(),
            "intercept": self.intercept,
            "standardized_coefficients": {
                name: float(value) for name, value in zip(self.feature_names, self.coefficients)
            },
            "l2": self.l2,
        }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    negative_exp = np.exp(values[~positive])
    result[~positive] = negative_exp / (1.0 + negative_exp)
    return result


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    feature_names: Sequence[str],
    l2: float = 1.0,
    max_iterations: int = 100,
    tolerance: float = 1e-9,
) -> LogisticModel:
    """Fit logistic regression with Newton updates and no intercept penalty."""
    matrix = np.asarray(features, dtype=float)
    target = np.asarray(labels, dtype=float)
    if matrix.ndim != 2 or target.ndim != 1 or len(matrix) != len(target):
        raise ValueError("features must be 2D and labels must be a matching vector")
    if matrix.shape[1] != len(feature_names):
        raise ValueError("feature_names do not match the matrix width")
    if len(np.unique(target)) < 2:
        raise ValueError("logistic training requires both label classes")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features contain non-finite values")
    if l2 < 0.0:
        raise ValueError("l2 must be non-negative")

    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (matrix - mean) / scale
    design = np.column_stack((np.ones(len(matrix)), standardized))
    beta = np.zeros(design.shape[1], dtype=float)
    penalty = np.diag(np.r_[0.0, np.full(matrix.shape[1], float(l2))])

    iterations = 0
    for iterations in range(1, max_iterations + 1):
        probability = _sigmoid(design @ beta)
        gradient = design.T @ (probability - target) + penalty @ beta
        curvature = np.clip(probability * (1.0 - probability), 1e-8, None)
        hessian = design.T @ (design * curvature[:, None]) + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        beta -= step
        if float(np.linalg.norm(step, ord=np.inf)) <= tolerance:
            break

    return LogisticModel(
        feature_names=tuple(feature_names),
        mean=mean,
        scale=scale,
        intercept=float(beta[0]),
        coefficients=beta[1:].copy(),
        l2=float(l2),
        iterations=iterations,
    )


def fit_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    feature_names: Sequence[str],
    l2: float = 1.0,
) -> RidgeModel:
    """Fit one standardized ridge model without penalizing the intercept."""
    matrix = np.asarray(features, dtype=float)
    target = np.asarray(targets, dtype=float)
    if matrix.ndim != 2 or target.ndim != 1 or len(matrix) != len(target):
        raise ValueError("features must be 2D and targets must be a matching vector")
    if matrix.shape[1] != len(feature_names):
        raise ValueError("feature_names do not match the matrix width")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(target)):
        raise ValueError("features and targets must be finite")
    if l2 < 0.0:
        raise ValueError("l2 must be non-negative")

    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (matrix - mean) / scale
    design = np.column_stack((np.ones(len(matrix)), standardized))
    penalty = np.diag(np.r_[0.0, np.full(matrix.shape[1], float(l2))])
    normal = design.T @ design + penalty
    rhs = design.T @ target
    try:
        beta = np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(normal) @ rhs
    return RidgeModel(
        feature_names=tuple(feature_names),
        mean=mean,
        scale=scale,
        intercept=float(beta[0]),
        coefficients=beta[1:].copy(),
        l2=float(l2),
    )


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
    """Report scale-aware and sign-aware metrics for a continuous effect."""
    target = np.asarray(targets, dtype=float)
    prediction = np.asarray(predictions, dtype=float)
    if target.shape != prediction.shape or target.ndim != 1:
        raise ValueError("targets and predictions must be matching vectors")
    residual = prediction - target
    mse = float(np.mean(residual**2))
    baseline_mse = float(np.mean((target - target.mean()) ** 2))
    denominator = float(np.sum((target - target.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / denominator if denominator > 0.0 else 0.0
    target_std = float(target.std())
    prediction_std = float(prediction.std())
    correlation = (
        float(np.corrcoef(target, prediction)[0, 1])
        if target_std > 0.0 and prediction_std > 0.0
        else None
    )
    return {
        "mean_target": float(target.mean()),
        "mean_prediction": float(prediction.mean()),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": math.sqrt(mse),
        "mean_baseline_rmse": math.sqrt(baseline_mse),
        "r2": r2,
        "pearson_correlation": correlation,
        "sign_accuracy": float(np.mean(np.signbit(target) == np.signbit(prediction))),
    }


def choose_l2_logo_regression(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    *,
    feature_names: Sequence[str],
    candidates: Iterable[float],
) -> tuple[float, Dict[str, float]]:
    """Select ridge strength by grouped held-out root-mean-square error."""
    matrix = np.asarray(features, dtype=float)
    target = np.asarray(targets, dtype=float)
    group_values = np.asarray(groups)
    scores: Dict[str, float] = {}
    for candidate in candidates:
        predictions = np.zeros(len(target), dtype=float)
        for group in sorted(np.unique(group_values)):
            test = group_values == group
            train = ~test
            model = fit_ridge(
                matrix[train],
                target[train],
                feature_names=feature_names,
                l2=float(candidate),
            )
            predictions[test] = model.predict(matrix[test])
        scores[str(float(candidate))] = float(np.sqrt(np.mean((predictions - target) ** 2)))
    selected = min(
        (float(key) for key in scores),
        key=lambda value: (scores[str(value)], value),
    )
    return selected, scores


def nested_logo_regression_predictions(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    *,
    feature_names: Sequence[str],
    l2_candidates: Sequence[float] = (0.1, 1.0, 10.0, 100.0),
) -> Dict[str, Any]:
    """Return leakage-safe continuous predictions from nested grouped CV."""
    matrix = np.asarray(features, dtype=float)
    target = np.asarray(targets, dtype=float)
    group_values = np.asarray(groups)
    predictions = np.zeros(len(target), dtype=float)
    fold_details = []
    for group in sorted(np.unique(group_values)):
        test = group_values == group
        train = ~test
        selected_l2, inner_scores = choose_l2_logo_regression(
            matrix[train],
            target[train],
            group_values[train],
            feature_names=feature_names,
            candidates=l2_candidates,
        )
        model = fit_ridge(
            matrix[train],
            target[train],
            feature_names=feature_names,
            l2=selected_l2,
        )
        fold_predictions = model.predict(matrix[test])
        predictions[test] = fold_predictions
        fold_details.append(
            {
                "held_out_group": int(group),
                "train_case_count": int(train.sum()),
                "test_case_count": int(test.sum()),
                "selected_l2": selected_l2,
                "inner_rmse_by_l2": inner_scores,
                "test_predictions": fold_predictions.tolist(),
            }
        )
    return {"predictions": predictions, "folds": fold_details}


def log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    target = np.asarray(labels, dtype=float)
    probability = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0 - 1e-12)
    return float(
        -np.mean(target * np.log(probability) + (1.0 - target) * np.log(1.0 - probability))
    )


def choose_l2_logo(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    feature_names: Sequence[str],
    candidates: Iterable[float],
) -> tuple[float, Dict[str, float]]:
    """Select L2 by leave-one-group-out log loss."""
    matrix = np.asarray(features, dtype=float)
    target = np.asarray(labels, dtype=int)
    group_values = np.asarray(groups)
    scores: Dict[str, float] = {}
    for candidate in candidates:
        predictions = np.zeros(len(target), dtype=float)
        for group in sorted(np.unique(group_values)):
            test = group_values == group
            train = ~test
            model = fit_logistic(
                matrix[train],
                target[train],
                feature_names=feature_names,
                l2=float(candidate),
            )
            predictions[test] = model.predict_proba(matrix[test])
        scores[str(float(candidate))] = log_loss(target, predictions)
    selected = min((float(key) for key in scores), key=lambda value: (scores[str(value)], value))
    return selected, scores


def nested_logo_predictions(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    feature_names: Sequence[str],
    l2_candidates: Sequence[float] = (0.1, 1.0, 10.0),
) -> Dict[str, Any]:
    """Return leakage-safe probabilities from nested leave-one-group-out CV."""
    matrix = np.asarray(features, dtype=float)
    target = np.asarray(labels, dtype=int)
    group_values = np.asarray(groups)
    probabilities = np.zeros(len(target), dtype=float)
    fold_details = []
    for group in sorted(np.unique(group_values)):
        test = group_values == group
        train = ~test
        selected_l2, inner_scores = choose_l2_logo(
            matrix[train],
            target[train],
            group_values[train],
            feature_names=feature_names,
            candidates=l2_candidates,
        )
        model = fit_logistic(
            matrix[train],
            target[train],
            feature_names=feature_names,
            l2=selected_l2,
        )
        fold_probabilities = model.predict_proba(matrix[test])
        probabilities[test] = fold_probabilities
        fold_details.append(
            {
                "held_out_group": int(group),
                "train_case_count": int(train.sum()),
                "test_case_count": int(test.sum()),
                "train_positive_count": int(target[train].sum()),
                "test_positive_count": int(target[test].sum()),
                "selected_l2": selected_l2,
                "inner_log_loss_by_l2": inner_scores,
                "test_probabilities": fold_probabilities.tolist(),
            }
        )
    return {"probabilities": probabilities, "folds": fold_details}


def binary_metrics(
    labels: np.ndarray, probabilities: np.ndarray, *, threshold: float = 0.5
) -> Dict[str, Any]:
    target = np.asarray(labels, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    prediction = (probability >= threshold).astype(int)
    tp = int(np.sum((target == 1) & (prediction == 1)))
    tn = int(np.sum((target == 0) & (prediction == 0)))
    fp = int(np.sum((target == 0) & (prediction == 1)))
    fn = int(np.sum((target == 1) & (prediction == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denominator) if denominator else 0.0
    positives = probability[target == 1]
    negatives = probability[target == 0]
    comparisons = [
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    ]
    roc_auc = sum(comparisons) / len(comparisons) if comparisons else None
    return {
        "threshold": float(threshold),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": (tp + tn) / len(target),
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision_soft": precision,
        "recall_soft": recall,
        "specificity_off": specificity,
        "f1_soft": f1,
        "matthews_correlation": mcc,
        "roc_auc": roc_auc,
        "log_loss": log_loss(target, probability),
        "brier_score": float(np.mean((probability - target) ** 2)),
        "predicted_soft_count": int(prediction.sum()),
        "predicted_off_count": int(len(prediction) - prediction.sum()),
    }
