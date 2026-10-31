"""Meaningful categorical likelihood, causal evidence and mixture invariants."""
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest


@pytest.fixture(scope="module")
def fit():
    tools = os.environ.get("RBFIT_TOOLS", str(Path(__file__).resolve().parents[2]/"tools"/"rbfit"))
    sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location("rbfit_directional", Path(tools)/"lane_directional.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wrong_destination_is_not_a_match(fit):
    source = np.array([2, 2, 0, 1])
    actual = fit.decision_labels(source, np.array([0, 1, 1, 0]))
    predicted = fit.decision_labels(source, np.array([1, 0, 1, 0]))
    assert (actual != 0).all() and (predicted != 0).all()
    assert (actual == predicted).tolist() == [False, False, True, True]


def test_persistent_driver_mixture_and_no_information(fit):
    # Two profiles disagree, each explains a distinct driver's observable events.
    counts = np.zeros((3, 2, 3, 3, 3))
    counts[0, 0, 2, 1, 1] = counts[0, 1, 2, 2, 1] = 20
    counts[1, 0, 2, 1, 2] = counts[1, 1, 2, 2, 2] = 20
    # Both candidates are exactly equally informative for driver2.
    counts[2, :, 0, 0, 0] = 20
    weights = np.array([0.4, 0.6])
    emission = np.full((3,3,3), 0.05)
    for s in range(3):
        np.fill_diagonal(emission[s], 0.9)
    posterior, _ = fit.posterior_from_counts(counts, weights, emission)
    assert posterior[0,0] > 1-1e-12
    assert posterior[1,1] > 1-1e-12
    np.testing.assert_allclose(posterior[2], weights)
    no_exposure, _ = fit.posterior_from_counts(np.zeros_like(counts), weights, emission)
    np.testing.assert_allclose(no_exposure, np.tile(weights, (3,1)))


def test_directional_em_monotone_and_valid(fit):
    rng = np.random.default_rng(12)
    counts = np.zeros((20, 3, 3, 3, 3), dtype=int)
    for d in range(20):
        source = rng.integers(0,3,100)
        actual = rng.choice(3,100,p=[0.8,0.1,0.1])
        predictions = np.stack([actual, rng.integers(0,3,100), np.zeros(100,dtype=int)], axis=1)
        np.add.at(counts, (d, np.arange(3)[None,:], source[:,None], predictions, actual[:,None]), 1)
    weights, emission, posterior, trace, ll = fit.fit_directional(counts)
    assert np.min(np.diff(trace)) > -1e-8
    assert ll >= trace[-1]-1e-8
    np.testing.assert_allclose(emission.sum(-1), 1)
    np.testing.assert_allclose(posterior.sum(-1), 1)
    assert weights[0] > 0.99
    broken = counts.copy()
    broken[0,0,0,0,0] += 1
    with pytest.raises(ValueError, match="exposure"):
        fit.fit_directional(broken)


def test_shrinkage_zero_evidence_and_endpoints(fit):
    posterior = np.array([[1.0,0.0],[0.0,1.0],[0.1,0.9]])
    population = np.array([0.25,0.75])
    n = np.array([10,2,0])
    result = fit.shrink_weights(posterior, population, n, 5)
    np.testing.assert_allclose(result.sum(-1), 1)
    np.testing.assert_allclose(result[2], population)
    np.testing.assert_allclose(fit.shrink_weights(posterior,population,n,np.inf), np.tile(population,(3,1)))
    np.testing.assert_allclose(fit.shrink_weights(posterior,population,n,0)[:2], posterior[:2])


def test_count_record_target_and_candidate_axes(fit):
    record = {"vid": np.array([4,4,7]), "source": np.array([2,1,0]),
              "actual": np.array([1,0,2]), "prediction": np.array([[1,2],[0,2],[1,2]])}
    counts = fit.count_records(record, np.array([4,7]), 2, np.array([True,True,False]))
    assert counts[0,0,2,1,1] == counts[0,1,2,2,1] == 1
    assert counts[0,0,1,0,0] == counts[0,1,1,2,0] == 1
    assert not counts[1].any()
