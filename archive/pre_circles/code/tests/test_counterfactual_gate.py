import numpy as np

from counterfactual.gate_model import (
    binary_metrics,
    fit_logistic,
    fit_ridge,
    nested_logo_predictions,
    nested_logo_regression_predictions,
    regression_metrics,
)


def test_logistic_model_learns_simple_separation():
    features = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
    labels = np.asarray([0, 0, 1, 1])
    model = fit_logistic(features, labels, feature_names=("signal",), l2=0.1)
    probabilities = model.predict_proba(features)

    assert probabilities[0] < probabilities[1] < 0.5
    assert probabilities[3] > probabilities[2] > 0.5


def test_nested_logo_predicts_every_case_once_without_group_leakage():
    features = np.asarray([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0], [-0.5], [0.5]])
    labels = np.asarray([0, 0, 0, 1, 1, 1, 0, 1])
    groups = np.asarray([1, 1, 2, 2, 3, 3, 4, 4])
    result = nested_logo_predictions(
        features,
        labels,
        groups,
        feature_names=("signal",),
        l2_candidates=(0.1, 1.0),
    )

    assert len(result["probabilities"]) == len(labels)
    assert np.all(np.isfinite(result["probabilities"]))
    assert {fold["held_out_group"] for fold in result["folds"]} == {1, 2, 3, 4}
    assert all(fold["test_case_count"] == 2 for fold in result["folds"])


def test_binary_metrics_report_unsafe_relevant_confusion_counts():
    labels = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.6, 0.7, 0.4])
    metrics = binary_metrics(labels, probabilities, threshold=0.5)

    assert metrics["true_positive"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["balanced_accuracy"] == 0.5


def test_ridge_model_learns_continuous_effect():
    features = np.asarray([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    targets = np.asarray([-4.0, -2.0, 0.0, 2.0, 4.0])
    model = fit_ridge(
        features,
        targets,
        feature_names=("signal",),
        l2=0.0,
    )

    assert np.allclose(model.predict(features), targets)


def test_nested_logo_regression_predicts_every_case_once():
    features = np.asarray([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0], [-0.5], [0.5]])
    targets = 2.0 * features[:, 0]
    groups = np.asarray([1, 1, 2, 2, 3, 3, 4, 4])
    result = nested_logo_regression_predictions(
        features,
        targets,
        groups,
        feature_names=("signal",),
        l2_candidates=(0.0, 1.0),
    )

    assert len(result["predictions"]) == len(targets)
    assert np.all(np.isfinite(result["predictions"]))
    assert {fold["held_out_group"] for fold in result["folds"]} == {1, 2, 3, 4}


def test_regression_metrics_are_exact_for_perfect_prediction():
    targets = np.asarray([-1.0, 0.0, 2.0])
    metrics = regression_metrics(targets, targets.copy())

    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["sign_accuracy"] == 1.0
