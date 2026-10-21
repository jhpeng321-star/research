"""Synthetic probability and data-boundary contracts for the offline merge fitter.

No recorded episodes, fitted parameters, or evaluation artifacts are needed.  The
first-passage oracle is the logistic law of a single persistent threshold; the
soft-hazard oracle is the existing recorded-state sequence likelihood.
"""
import math
from pathlib import Path

import pytest
import torch

from tools import merge_fit_v2 as fit
from wms.train.losses.merge_core_seqtf import seq_tf_chunk


ROOT = Path(__file__).resolve().parents[1]
DTYPE = torch.float64
REFUSAL = (ValueError, PermissionError, SystemExit)


def _double(values):
    return torch.tensor(values, dtype=DTYPE)


def _assert_distribution(result, n_steps):
    assert isinstance(result, dict)
    for key in ("log_mass", "log_survival", "cdf"):
        value = result[key]
        assert isinstance(value, torch.Tensor), key
        assert value.shape == (n_steps,), key
        assert value.dtype == DTYPE, key
        assert not torch.isnan(value).any(), key
    mass = result["log_mass"].exp()
    survival = result["log_survival"].exp()
    cdf = result["cdf"]
    assert (mass >= 0).all()
    assert ((cdf >= 0) & (cdf <= 1)).all()
    assert (cdf[1:] >= cdf[:-1]).all()
    torch.testing.assert_close(mass.cumsum(0), cdf, atol=2e-13, rtol=2e-13)
    torch.testing.assert_close(survival + cdf, torch.ones_like(cdf),
                               atol=2e-13, rtol=2e-13)
    torch.testing.assert_close(mass.sum() + survival[-1], _double(1.0),
                               atol=2e-13, rtol=2e-13)


def test_first_passage_is_running_max_logistic_not_repeated_hazards():
    c = _double([-2.0, 0.5, -1.0, 0.5, 2.0, 1.0])
    result = fit.first_passage(c, _double(1.0))
    _assert_distribution(result, len(c))
    expected_cdf = _double([1 / (1 + math.exp(2)),
                            1 / (1 + math.exp(-0.5)),
                            1 / (1 + math.exp(-0.5)),
                            1 / (1 + math.exp(-0.5)),
                            1 / (1 + math.exp(-2)),
                            1 / (1 + math.exp(-2))])
    expected_mass = torch.diff(torch.cat((_double([0.0]), expected_cdf)))
    torch.testing.assert_close(result["cdf"], expected_cdf, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(result["log_mass"].exp(), expected_mass,
                               atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(result["log_survival"].exp(), 1 - expected_cdf,
                               atol=1e-14, rtol=1e-14)
    assert torch.isneginf(result["log_mass"][[2, 3, 5]]).all()


@pytest.mark.parametrize("last_observed", [0, 2, 4])
def test_first_passage_censoring_preserves_unspent_probability(last_observed):
    c = _double([-1.5, 0.0, -0.5, 1.0, 0.25, 3.0])
    full = fit.first_passage(c, _double(1.0))
    prefix = fit.first_passage(c[:last_observed + 1], _double(1.0))
    _assert_distribution(prefix, last_observed + 1)
    for key in ("log_mass", "log_survival", "cdf"):
        torch.testing.assert_close(prefix[key], full[key][:last_observed + 1])
    assert prefix["log_survival"][-1].exp() > 0


def test_first_passage_competing_exit_and_final_censor_conserve_mass():
    result = fit.first_passage(_double([-1.0, 0.2, -0.4, 1.2, 0.7]), _double(1.0))
    # An independent exit is checked after each merge opportunity.  Paths that
    # exited earlier contribute neither later merge mass nor final censor mass.
    exit_hazard = _double([0.1, 0.0, 0.3, 0.2, 0.4])
    exit_survival_before = torch.cat((_double([1.0]),
                                     (1 - exit_hazard).cumprod(0)[:-1]))
    merge_mass = result["log_mass"].exp() * exit_survival_before
    exit_mass = result["log_survival"].exp() * exit_survival_before * exit_hazard
    censored_mass = result["log_survival"][-1].exp() * (1 - exit_hazard).prod()
    assert merge_mass.sum() > 0 and exit_mass.sum() > 0 and censored_mass > 0
    torch.testing.assert_close(merge_mass.sum() + exit_mass.sum() + censored_mass,
                               _double(1.0), atol=2e-13, rtol=2e-13)


@pytest.mark.parametrize("positive", [False, True])
def test_first_passage_extreme_finite_records_keep_finite_log_probabilities(positive):
    c = _double([1000.0, 1001.0, 999.0] if positive else [-1001.0, -1000.0, -1002.0])
    result = fit.first_passage(c, _double(1.0))
    _assert_distribution(result, len(c))
    # Computing sigmoid(new) - sigmoid(old) loses this mass in either tail.
    expected_second = -1000.0 + math.log1p(-math.exp(-1))
    assert torch.isfinite(result["log_mass"][:2]).all()
    assert result["log_mass"][1].item() == pytest.approx(expected_second, abs=1e-10)
    assert torch.isneginf(result["log_mass"][2])
    if positive:
        torch.testing.assert_close(result["log_survival"], _double([-1000, -1001, -1001]))
    else:
        assert result["log_mass"][0].item() == pytest.approx(-1001.0, abs=1e-10)


def test_first_passage_negative_infinity_veto_does_not_erase_a_previous_record():
    result = fit.first_passage(_double([-math.inf, -0.5, -math.inf, 0.5]), _double(1.0))
    _assert_distribution(result, 4)
    assert torch.isneginf(result["log_mass"][[0, 2]]).all()
    assert result["cdf"][0].item() == 0
    assert result["log_survival"][0].item() == 0
    assert result["cdf"][2].item() == result["cdf"][1].item()
    assert result["log_survival"][2].item() == result["log_survival"][1].item()


def test_first_passage_all_veto_is_certain_censoring():
    result = fit.first_passage(torch.full((5,), -math.inf, dtype=DTYPE), _double(1.0))
    _assert_distribution(result, 5)
    assert torch.isneginf(result["log_mass"]).all()
    assert torch.equal(result["cdf"], _double([0] * 5))
    assert torch.equal(result["log_survival"], _double([0] * 5))


def test_first_passage_all_free_is_an_immediate_atom():
    result = fit.first_passage(torch.full((5,), math.inf, dtype=DTYPE), _double(1.0))
    _assert_distribution(result, 5)
    assert result["log_mass"][0].item() == 0
    assert torch.isneginf(result["log_mass"][1:]).all()
    assert torch.isneginf(result["log_survival"]).all()
    assert torch.equal(result["cdf"], _double([1] * 5))


def test_first_passage_free_atom_consumes_only_remaining_survival():
    result = fit.first_passage(_double([-1.0, math.inf, math.inf, -math.inf]), _double(1.0))
    _assert_distribution(result, 4)
    assert result["log_mass"][1].item() == pytest.approx(-math.log1p(math.exp(-1)))
    assert torch.isneginf(result["log_mass"][2:]).all()
    assert torch.isneginf(result["log_survival"][1:]).all()


@pytest.mark.parametrize("previous_row", [0, 1, 7, 103])
def test_aligned_row_is_the_next_observation_authorized_by_previous_row(previous_row):
    result = fit.aligned_row(previous_row)
    assert isinstance(result, int)
    assert result == previous_row + 1


def test_alignment_scores_the_gap_before_the_observed_lane_change():
    # The generous gap is on the authorizing row, not on the observed event row.
    gap_margin = _double([-3.0, 2.0, -8.0, -8.0])
    lane = [2, 2, 1, 1]
    observed_event = next(i for i in range(1, len(lane)) if lane[i - 1] == 2 and lane[i] == 1)
    authorizing_row = observed_event - 1
    assert fit.aligned_row(authorizing_row) == observed_event
    result = fit.first_passage(gap_margin, _double(1.0))
    assert result["log_mass"][authorizing_row].exp() > 0.8
    assert torch.isneginf(result["log_mass"][observed_event])


def _features():
    # A single car with changing gaps, closing speeds, acceleration, and each
    # kind of one-sided missing neighbour.  No scene builder is used as an oracle.
    rows = {
        "gap_l": [7.0, 12.0, 4.0, 16.0, 9.0],
        "gap_f": [13.0, 6.0, 11.0, 5.0, 17.0],
        "v": [7.0, 8.0, 9.0, 8.0, 7.5],
        "v_l": [6.0, 9.0, 7.0, 8.5, 8.0],
        "v_f": [9.0, 7.0, 10.0, 6.5, 8.5],
        "a": [0.2, -0.4, 0.1, 0.0, 0.3],
        "a_l": [-0.2, 0.1, 0.3, -0.1, 0.0],
        "a_f": [0.4, -0.2, 0.0, 0.1, -0.3],
    }
    features = {key: _double(value) for key, value in rows.items()}
    features["has_l"] = torch.tensor([True, False, True, True, True])
    features["has_f"] = torch.tensor([True, True, True, False, True])
    return features


def _params(sigma=0.6):
    return {key: _double(value) for key, value in {
        "s0": 2.1, "T": 0.85, "b": 2.3, "tau_a": 0.7,
        "temperature": 1.2, "sigma": sigma,
    }.items()}


def _handcrafted_scene(features, event, last_observed):
    n = len(features["v"])
    return {
        **{key: value.reshape(1, n, 1).clone() for key, value in features.items()},
        "k": 1, "n_steps": n, "dt_s": 0.5, "n_cars": 1.0,
        "feasible": torch.ones(1, n, 1, dtype=torch.bool),
        "supervised": torch.ones(1, n, 1, dtype=torch.bool),
        "identity": torch.zeros(1, 1, dtype=torch.long),
        "entry": torch.zeros(1, 1, dtype=torch.long),
        "run_start": torch.zeros(1, n, 1, dtype=torch.long),
        "rec_event": torch.tensor([[event]]),
        "last_obs": torch.tensor([[last_observed]]),
        "has_exposure": torch.ones(1, 1, dtype=torch.bool),
    }


@pytest.mark.parametrize("n_quad,sigma", [(1, 0.0), (8, 0.6), (8, 12.0)])
def test_soft_hazard_matches_seq_tf_chunk_for_every_event_and_censor(n_quad, sigma):
    features, params = _features(), _params(sigma)
    n = len(features["v"])
    result = fit.soft_hazard(features, params, n_quad=n_quad)
    _assert_distribution(result, n)
    for step in range(n):
        event_scene = _handcrafted_scene(features, event=step, last_observed=step - 1)
        event_nll, n_cars, _, metrics = seq_tf_chunk(
            event_scene, 0, n, **params, n_quad=n_quad, log_post=None)
        assert n_cars.item() == 1 and metrics["events"] == 1
        torch.testing.assert_close(result["log_mass"][step], -event_nll,
                                   atol=2e-11, rtol=2e-11)
        censor_scene = _handcrafted_scene(features, event=-1, last_observed=step)
        censor_nll, _, _, metrics = seq_tf_chunk(
            censor_scene, 0, n, **params, n_quad=n_quad, log_post=None)
        assert metrics["censored"] == 1
        torch.testing.assert_close(result["log_survival"][step], -censor_nll,
                                   atol=2e-11, rtol=2e-11)


def test_soft_hazard_default_quadrature_is_eight_nodes():
    features, params = _features(), _params()
    default = fit.soft_hazard(features, params)
    explicit = fit.soft_hazard(features, params, n_quad=8)
    for key in ("log_mass", "log_survival", "cdf"):
        torch.testing.assert_close(default[key], explicit[key], atol=0, rtol=0)


def _assert_bootstrap_result(result):
    assert isinstance(result, dict)
    assert {"estimate", "ci95", "status"} <= result.keys()
    assert isinstance(result["status"], str) and result["status"]
    assert math.isfinite(float(result["estimate"]))
    assert len(result["ci95"]) == 2
    low, high = map(float, result["ci95"])
    assert math.isfinite(low) and math.isfinite(high) and low <= high


def test_world_bootstrap_is_reproducible_for_paired_differences():
    values = [-3.0, -1.0, 2.0, 4.0, 6.0, 4.0]
    worlds = ["a", "a", "b", "b", "c", "c"]
    first = fit.world_bootstrap(values, worlds, n_boot=1000, seed=73)
    second = fit.world_bootstrap(values, worlds, n_boot=1000, seed=73)
    _assert_bootstrap_result(first)
    assert first == second
    assert first["estimate"] == pytest.approx(2.0)
    assert first["ci95"][0] < first["ci95"][1]
    # Constant shifts of supplied paired differences shift the whole interval.
    shifted = fit.world_bootstrap([x + 5 for x in values], worlds, n_boot=1000, seed=73)
    assert shifted["estimate"] == pytest.approx(first["estimate"] + 5)
    assert shifted["ci95"] == pytest.approx([x + 5 for x in first["ci95"]])


def test_world_bootstrap_resamples_whole_worlds_not_individual_rows():
    result = fit.world_bootstrap([-10.0, 10.0, -3.0, 3.0, -7.0, 7.0],
                                 ["a", "a", "b", "b", "c", "c"],
                                 n_boot=1000, seed=17)
    _assert_bootstrap_result(result)
    # Every world has zero mean, despite large within-world variation.
    assert result["estimate"] == pytest.approx(0.0)
    assert result["ci95"] == pytest.approx([0.0, 0.0])


def test_world_bootstrap_is_invariant_to_replication_within_one_world():
    base = fit.world_bootstrap([-4.0, 0.0, 2.0, 6.0, 8.0, 10.0],
                               ["a", "a", "b", "b", "c", "c"],
                               n_boot=1000, seed=29)
    replicated = fit.world_bootstrap([-4.0, 0.0] * 7 + [2.0, 6.0, 8.0, 10.0],
                                     ["a"] * 14 + ["b", "b", "c", "c"],
                                     n_boot=1000, seed=29)
    _assert_bootstrap_result(base)
    _assert_bootstrap_result(replicated)
    assert base["estimate"] == pytest.approx(11 / 3)
    assert replicated["estimate"] == pytest.approx(base["estimate"])
    assert replicated["ci95"] == pytest.approx(base["ci95"])


@pytest.mark.parametrize("token", ["sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1"])
def test_guard_refuses_forbidden_substrings_even_in_mixed_case_filenames(token):
    path = ROOT / "tests" / f"synthetic_{token.upper()}_result.json"
    with pytest.raises(REFUSAL):
        fit.guard(path)


@pytest.mark.parametrize("kind", ["parent", "prefix_sibling", "traversal"])
def test_guard_refuses_paths_outside_workspace_without_opening_them(kind):
    paths = {
        "parent": ROOT.parent / "synthetic_merge_fit.json",
        "prefix_sibling": ROOT.parent / (ROOT.name + "-other") / "synthetic.json",
        "traversal": ROOT / "tests" / ".." / ".." / "synthetic_merge_fit.json",
    }
    with pytest.raises(REFUSAL):
        fit.guard(paths[kind])


def test_guard_accepts_an_unrestricted_workspace_path():
    path = ROOT / "tests" / "synthetic_merge_fit_result.json"
    assert Path(fit.guard(path)).resolve() == path.resolve()


@pytest.mark.parametrize("seed", [964000, 964001, 964099, 964198, 964199])
def test_check_seed_refuses_evaluation_seed_range(seed):
    with pytest.raises(REFUSAL):
        fit.check_seed(seed)


@pytest.mark.parametrize("seed", [0, 73, 963999, 964200])
def test_check_seed_accepts_training_seeds_and_range_boundaries(seed):
    fit.check_seed(seed)
