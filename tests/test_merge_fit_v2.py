"""Synthetic probability and data-boundary contracts for the offline merge fitter.

No recorded episodes, fitted parameters, or evaluation artifacts are needed.  The
first-passage oracle is the logistic law of a single persistent threshold; the
soft-hazard oracle is the existing recorded-state sequence likelihood.
"""
import math
from pathlib import Path
import json
import tempfile
import numpy as np

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


def test_alignment_shift_scores_the_authorizing_output_row():
    # Cached SUMO authorization belongs to the first post-merge output row.
    # Recorded next-row geometry is diagnostic ONLY. Runtime scores the causal
    # proposal from its preceding input row, timestamped at this output row.
    gap_margin = _double([-3.0, -2.0, 2.0, -8.0])
    lane = [2, 2, 1, 1]
    observed_event = next(i for i in range(1, len(lane)) if lane[i - 1] == 2 and lane[i] == 1)
    previous_waiting_row = observed_event - 1
    authorizing_row = fit.aligned_row(previous_waiting_row)
    assert authorizing_row == observed_event
    result = fit.first_passage(gap_margin, _double(1.0))
    assert result["log_mass"][authorizing_row].exp() > 0.7
    assert result["log_mass"][authorizing_row] != result["log_mass"][previous_waiting_row]


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


@pytest.mark.parametrize("values", [[-2., -1., 0.], [1000., 1001., 1002.],
                                   [-1002., -1001., -1000.], [-math.inf, -1., math.inf]])
def test_first_passage_exact_record_gradients_are_finite(values):
    c = _double(values).requires_grad_()
    tau = _double(.8).requires_grad_()
    result = fit.first_passage(c, tau)
    supported = torch.isfinite(result["log_mass"])
    (-result["log_mass"][supported].sum()).backward()
    assert torch.isfinite(c.grad).all()
    assert torch.isfinite(tau.grad)
    assert tau.grad != 0


def _sequence(event=True, physical=True, unknown_age=False, world="a"):
    features = {k: v.tolist() for k, v in _features().items()}
    # Last step is a strict first-passage record, with both sides present.
    features["gap_l"][-1] = 100.
    features["gap_f"][-1] = 100.
    return {"world": world, "source": "synthetic", "episode": world+"__native", "vehicle": 7,
            "slot": 0, "entry_row": 2, "event_row": 7 if event else None,
            "legacy": features, "updated": features, "physical": [physical]*5,
            "physical_reasons": ["ok" if physical else "target_interval_intersection"]*5,
            "event": event, "terminal_kind": "event" if event else "horizon_censor",
            "unknown_age": unknown_age, "queue": "front", "demand": 1900,
            "dt": .5, "times": [.5, 1., 1.5, 2., 2.5],
            "observed_s": 2.5 if event else None, "horizon_s": 300.}


def _initial():
    return {k: float(v) for k, v in _params(.3).items()}


def test_physical_veto_reports_zero_event_without_exclusion_or_floor():
    seq = _sequence(physical=False)
    model = fit.Candidate("P1-P3", _initial())
    result = fit.evaluate(model, [seq], 8)
    assert result["nll"]["count"] == 1
    assert result["nll"]["mean"] == math.inf
    assert result["zero_mass_event_count"] == 1
    assert result["physical_support"]["share"] == 0
    assert len(result["physical_support"]["failures"]) == 1
    _, info = fit.optimize("P1-P3", _initial(), [seq], iters=4, lr=.03, shrinkage=0, n_quad=8)
    assert info["status"] == "exact_support_failure"
    assert info["updates"] == 0


@pytest.mark.parametrize("kind", ["M0-refit", "P1", "P1+P2", "P1-P3"])
def test_supported_fit_has_finite_gradients_and_updates(kind):
    _, info = fit.optimize(kind, _initial(), [_sequence(), _sequence(event=False)],
                           iters=4, lr=.03, shrinkage=0, n_quad=8)
    assert info["status"] == "fitted", info
    assert info["updates"] == 4
    assert all(t["finite_nonzero_gradient_count"] > 0 for t in info["trace"])


def test_unknown_age_is_retained_as_full_cohort_timing_bounds():
    result = fit.evaluate(fit.Candidate("M0", _initial()), [_sequence(unknown_age=True)], 8)
    assert result["timing"]["unknown_age_events"] == 1
    assert result["timing"]["signed_median_bounds_s"] == [-math.inf, math.inf]
    assert result["timing"]["absolute_median_bounds_s"] == [0., math.inf]


def test_unreached_median_is_a_bound_and_post_event_tails_are_not_continued():
    seq = _sequence()
    seq["updated"]["gap_l"] = [.0001]*5
    seq["updated"]["gap_f"] = [.0001]*5
    result = fit.evaluate(fit.Candidate("P1-P3", _initial()), [seq], 8)
    row = result["cars"][0]
    assert not row["median_reached"]
    assert row["error_bounds_s"] == [0., math.inf]
    assert row["tail_noncompletion"]["120"]["model_bounds"][0] == 0
    assert row["tail_noncompletion"]["120"]["model_bounds"][1] > .99
    assert result["population_cdf"]["quantiles"]["50"]["identifiable"] is False


def test_geometry_checks_all_intervals_with_variable_lengths_and_cav_neighbors():
    # Ego front 10, rear 6. Adjacent front 20/rear 10 merely touches.
    import numpy as np
    kin = np.array([[10, 0, 0, 1, 3, 0, 4], [20, 0, 1, 0, 4, 0, 10],
                    [0, 0, 1, 0, 2, 0, 8], [500, 0, 1, 0, 8, 0, 5]], float)
    features, ok, reason = fit.geometry(kin, np.ones(4, bool), 0)
    assert ok and reason == "ok"
    assert features["gap_l"] == 0
    assert features["gap_f"] == 6
    kin[1, 0] -= .01
    _, ok, reason = fit.geometry(kin, np.ones(4, bool), 0)
    assert not ok and reason == "target_interval_intersection"


def test_strict_json_preserves_infinite_losses_as_explicit_strings():
    import json
    import numpy as np
    result = fit.clean_json({"loss": math.inf, "bound": -math.inf, "flag": np.bool_(True)})
    assert json.loads(json.dumps(result, allow_nan=False)) == {
        "loss": "+Infinity", "bound": "-Infinity", "flag": True}


@pytest.fixture
def workspace_tmp():
    parent = ROOT / "tmp"
    parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="r9_fit_test_", dir=parent) as directory:
        yield Path(directory)


def _write_episode(root, number, *, reused=False):
    world = f"w963{number:03d}m1900p10t0"
    episode = world+"__native"
    path = root / "episodes" / episode
    path.mkdir(parents=True)
    kin = np.zeros((8, 3, 7), np.float32)
    kin[:, :, 6] = 5
    kin[:, :, 4] = 1
    kin[:, 0, 3] = 1
    kin[5:, 0, 3] = 0
    kin[5:, 0, 2] = 1
    kin[:, 1:, 2] = 1
    kin[:, 0, 0] = np.arange(8)*.5-.5
    kin[:, 1, 0] = 15+np.arange(8)*2
    kin[:, 2, 0] = -15-np.arange(8)*2
    # Future states deliberately incompatible with any causal proposal.
    kin[5:, 0, 0] = 60
    mask = np.ones((8, 3), np.uint8)
    vid = np.tile(np.arange(3), (8, 1)).astype(np.int32)
    if reused:
        vid[5:, 0] = 99  # Same-slot successor must NOT be a ramp merge.
    for key, array in {"veh_kin": kin, "veh_mask": mask, "vid": vid,
                       "is_cav": np.zeros_like(mask), "time_s": np.arange(8)*.5}.items():
        np.save(path/(key+".npy"), array, allow_pickle=False)
    meta = {"world": world, "episode": episode, "seed": 963000+number,
            "dir": "episodes/"+episode, "warmup_s": 0, "horizon_s": 3.5,
            "demand_vph": 1900, "arm": "native", "split_role": "train"}
    (path/"episode.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def test_episode_loader_alignment_is_causal_and_identity_bound(workspace_tmp):
    meta = _write_episode(workspace_tmp, 1)
    (workspace_tmp/"C4_EPISODES_MANIFEST.json").write_text(
        json.dumps({"split": "TRAIN", "episodes": [meta]}), encoding="utf-8")
    episodes, _ = fit.discover(["synthetic="+str(workspace_tmp)])
    snapshots = []
    def adapter(**kwargs):
        assert kwargs["kin"].shape == (3, 7)  # Not a time sequence.
        snapshots.append(kwargs["kin"].copy())
        return fit.causal_proposal(kwargs["kin"], kwargs["dt_s"])
    seqs, data = fit.build_dataset(episodes, adapter)
    assert len(seqs) == 1 and seqs[0]["event_row"] == 5
    assert seqs[0]["entry_row"] == 1
    assert seqs[0]["times"] == [.5, 1., 1.5, 2.]
    # Score input row 4 advanced exactly once, not the jumped recorded row 5.
    expected = fit.causal_proposal(snapshots[4], .5)
    features, _, _ = fit.geometry(expected, np.ones(3, bool), 0)
    assert seqs[0]["updated"]["gap_l"][-1] == features["gap_l"]
    assert seqs[0]["updated"]["gap_l"][-1] > 0
    assert data["episodes"][0]["events"] == 1


def test_same_slot_replacement_does_not_invent_event(workspace_tmp):
    meta = _write_episode(workspace_tmp, 1, reused=True)
    (workspace_tmp/"C4_EPISODES_MANIFEST.json").write_text(
        json.dumps({"split": "TRAIN", "episodes": [meta]}), encoding="utf-8")
    episodes, _ = fit.discover(["synthetic="+str(workspace_tmp)])
    seqs, _ = fit.build_dataset(episodes)
    assert len(seqs) == 1 and not seqs[0]["event"]
    assert seqs[0]["terminal_kind"] == "teleport_or_identity_exit"


def test_cli_nonempty_audit_grouped_fit_and_bootstrap(workspace_tmp):
    metas = [_write_episode(workspace_tmp, n) for n in range(10)]
    (workspace_tmp/"C4_EPISODES_MANIFEST.json").write_text(
        json.dumps({"split": "TRAIN", "episodes": metas}), encoding="utf-8")
    out, md = workspace_tmp/"fit.json", workspace_tmp/"fit.md"
    assert fit.main(["--source", "synthetic="+str(workspace_tmp), "--models", "M0,M0-refit,P1,P1+P2,P1-P3",
                     "--audit-frac", "0.2", "--inner-folds", "2", "--bootstrap", "40", "--seed", "73",
                     "--round8-params", "default", "--iters", "2", "--shrinkage", "0", "0.01",
                     "--out", str(out), "--md", str(md)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert len(report["split"]["audit_worlds"]) == 2
    assert len(report["split"]["fit_worlds"]) == 8
    assert set(report["split"]["audit_worlds"]).isdisjoint(report["split"]["fit_worlds"])
    for result in report["models"].values():
        audit = result["audit"]
        assert audit["nll"]["count"] == 2
        assert audit["nll_world_bootstrap"]["status"] == "ok"
        assert audit["physical_support"]["share"] == 1
        assert audit["zero_mass_event_count"] == 0
        assert set(audit["paired_vs"]) == {"M0", "M0-refit"}
        assert audit["noncompletion"]["all"]["tails"]["120"]["cars"] == 2
    assert report["models"]["P1-P3"]["fit"]["selection_status"] == "ok"
    assert "DIAGNOSTIC" in report["causal_proposal"]["name"]
    assert report["verdict"] == "CHANGES"
    assert md.is_file()


def test_round8_checkpoint_transform_and_learned_residual_refusal(workspace_tmp):
    from wms.model.merge_core import MergeCoreNet
    core = MergeCoreNet(input_dim=0)
    path = workspace_tmp/"round8.pt"
    state = {"merge_core."+k: v for k, v in core.state_dict().items()}
    torch.save({"model": state}, path)
    params, identity = fit.load_params(path)
    expected = {k: float(v.detach()) for k, v in core.params().items()}
    expected["sigma"] = float(core.alpha_scale().detach())
    assert params == expected
    assert len(identity["sha256"]) == 64
    state["merge_core.residual_head.2.bias"] = torch.ones(1)
    torch.save({"model": state}, path)
    with pytest.raises(ValueError, match="residual/location"):
        fit.load_params(path)


def test_untouched_m0_preserves_config_scalars_exactly():
    initial, _ = fit.load_params("default")
    model = fit.Candidate("M0", initial)
    assert {k: float(v) for k, v in model.params().items()} == initial
    assert not any(p.requires_grad for p in model.parameters())


def test_current_actions_use_only_issued_past_controls_and_reset_identity(workspace_tmp):
    times = np.arange(5)*.5
    vid = np.zeros((5, 1), np.int32)
    vid[1, 0] = 99
    mask = np.ones((5, 1), bool)
    arrays = {"control_time_s": np.array([0., 1.]), "controller": np.array([[3], [0]]),
              "issued_acceleration_mps2": np.array([[2.], [-3.]]),
              "issued_target_speed_mps": np.array([[10.], [20.]])}
    for key, value in arrays.items():
        np.save(workspace_tmp/(key+".npy"), value, allow_pickle=False)
    actions = fit.current_actions({"path": workspace_tmp}, times, vid, mask)
    assert actions["issued_acceleration_mps2"].flatten().tolist() == [2., 0., -3., -3.]
    assert actions["valid"].flatten().tolist() == [True, False, True, True]


def test_insufficient_inner_worlds_cannot_be_labeled_selected():
    from types import SimpleNamespace
    args = SimpleNamespace(inner_folds=4, shrinkage=[0], iters=1, lr=.03, n_quad=8)
    _, result = fit.select_fit("M0-refit", _initial(), [_sequence(world="a"), _sequence(world="b")], args)
    assert result["selection_status"] == "unresolved"
    assert result["inner"][0]["mean_validation_nll"] == math.inf


def test_p2_soft_hazard_uses_one_persistent_logistic_not_normal_or_tick_mixtures():
    initial = _initial()
    initial["sigma"] = .8
    model = fit.Candidate("P1+P2", initial)
    seq = _sequence()
    features = {k: torch.tensor(v, dtype=torch.bool if k.startswith("has_") else DTYPE)
                for k, v in seq["updated"].items()}
    p = model.params()
    x, w = np.polynomial.legendre.leggauss(8)
    u = (x+1)/2
    z = _double(np.log(u)-np.log1p(-u))
    alpha = torch.exp(p["tau"]*z).unsqueeze(1)
    df = p["s_f"]+p["T_f"]*features["v_f"]+torch.relu(features["v_f"]**2-features["v"]**2)/(2*p["b_f"])
    dl = p["s_l"]+p["T_l"]*features["v"]+torch.relu(features["v"]**2-features["v_l"]**2)/(2*p["b_l"])
    ml = torch.where(features["has_l"], features["gap_l"]-dl/alpha, math.inf)
    mf = torch.where(features["has_f"], features["gap_f"]-df/alpha, math.inf)
    hazards = torch.sigmoid(torch.minimum(ml, mf)/p["temperature"])
    weights = _double(w/2)
    persistent = (weights*((1-hazards[:, :-1]).prod(1))*hazards[:, -1]).sum()
    marginal = (weights[:, None]*hazards).sum(0)
    per_tick = (1-marginal[:-1]).prod()*marginal[-1]
    expected = -persistent.log()
    actual = model.nll(fit.pack([seq]))[0]
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert abs(float((actual+per_tick.log()).detach())) > 1e-4
