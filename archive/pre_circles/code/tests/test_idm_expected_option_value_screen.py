from dataclasses import dataclass

import numpy as np

from tools.screen_idm_expected_option_value import (
    choose_threshold,
    fit_ridge,
    policy_metrics,
    predict_ridge,
)


@dataclass
class Record:
    activate_reward: float
    physical_utility: float
    constraints_pass: bool
    penetration: float
    demand: int = 2000
    seed: int = 1


def test_weighted_ridge_learns_expected_value_order():
    x = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.asarray([-1.0, -0.5, 0.5, 1.0])
    model = fit_ridge(x, y, np.ones(4), alpha=0.01)
    prediction = predict_ridge(model, x)
    assert np.all(np.diff(prediction) > 0.0)


def test_threshold_requires_nontrivial_bounded_harm_activation():
    records = [
        Record(0.4, 0.4, True, 0.1),
        Record(0.3, 0.3, True, 0.1),
        Record(0.0, 0.0, True, 0.2),
        Record(-0.4, -0.4, True, 0.2),
        Record(-2.0, 0.2, False, 0.3),
    ]
    scores = np.asarray([0.8, 0.7, 0.2, -0.2, -0.5])
    threshold, selected = choose_threshold(records, scores)
    assert selected["training_constraints_feasible"]
    assert selected["activation_coverage"] >= 0.2
    assert selected["harmful_fraction_of_activations"] <= 0.25
    assert threshold > -0.2


def test_policy_reward_counts_abstention_as_zero():
    records = [
        Record(0.4, 0.4, True, 0.1),
        Record(-0.3, -0.3, True, 0.1),
    ]
    result = policy_metrics(records, np.asarray([0.5, -0.5]), 0.0)
    assert result["activations"] == 1
    assert result["mean_policy_reward"] == 0.2
