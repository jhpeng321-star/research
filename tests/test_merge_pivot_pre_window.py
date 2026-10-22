"""F2 recorded pre-window survival: causal replay, certification and exact scores."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wms.data.collate import collate
from wms.data.c4_episodes import C4WindowLoader
from wms.data.dataset import TrajectoryDataset, _PackWindowLoader
from wms.data.recorded_prefix import recorded_prefix
from wms.model import ModelSpec, WmSModel
from wms.model.merge_pivot import log_event_mass, log_survival
from wms.physics.config import PhysicsConfig
from wms.train.inputs import build_targets, prepare_inputs
from wms.train.losses.base import LossContext
from wms.train.losses.merge_pivot import (
    MergePivotConfig, MergePivotTerm, build_pre_window_history, build_recorded_geometry,
    merge_pivot_metrics, pre_window_maximum, score_chunk,
)
from wms.train.settings import TrainSettings


def _arrays():
    states = np.zeros((6, 3, 7), np.float32)
    states[..., 6] = 5.
    for i in range(6):
        states[i, :, 0] = [-6 + 4 * i, 35 + 6 * i, -30 + 3 * i]
        states[i, :, 4] = [8., 12., 6.]
        states[i, 0, 3 if i < 5 else 2] = 1.
        states[i, 1:, 2] = 1.
    return dict(states=states[3:], present=np.ones((3, 3), bool),
                history_kin=states[:4], history_mask=np.ones((4, 3), bool),
                history_target_mask=np.ones((3, 3), bool),
                history_is_cav=np.zeros((4, 3), bool))


def _fixture(arrays=None, **spec):
    torch.manual_seed(32)
    model = WmSModel(ModelSpec(rollout_mode="hybrid", lane_model="hazard_v5",
                               merge_decision="first_passage", **spec), physics=PhysicsConfig()).eval()
    dataset = TrajectoryDataset.from_arrays(_arrays() if arrays is None else arrays,
                                            merge_pivot_history=True)
    batch = collate([dataset[0]])
    settings = TrainSettings(rollout_mode="hybrid", lane_model="hazard_v5",
                             merge_decision="first_passage", **spec)
    inp = prepare_inputs(batch, model, settings, torch.device("cpu"), 2)
    targets = build_targets(inp, batch, settings, .5, torch.device("cpu"), physics=model.physics)
    targets.prepared["merge_pivot_source"] = model, inp
    return model, inp, targets


def _prepare(model, targets):
    term = MergePivotTerm(MergePivotConfig(tf_weight=1.))
    term.prepare(targets)
    prep = targets.prepared["merge_pivot"]
    # Known record-setting boundary sequence with both observed neighbour sides.
    # Its geometry remains detached, just like the frozen model replay it replaces.
    for side in ("f", "l"):
        prep["pre_window_history"][0]["geometry"][f"gap_{side}"][:, :, 0] = torch.tensor([10., 15., 20.])
        prep["geometry"][f"gap_{side}"][:, :, 0] = torch.tensor([25., 40.])
    return term, prep


def test_ragged_raw_prefix_is_opt_in_and_never_reconstructed_from_features():
    arrays = _arrays()
    off = TrajectoryDataset.from_arrays(arrays)[0]
    assert "merge_pivot_prefix" not in off
    on = TrajectoryDataset.from_arrays(arrays, merge_pivot_history=True)[0]
    other = deepcopy(on)
    other["merge_pivot_prefix"] = None
    batch = collate([on, other])
    assert len(batch["merge_pivot_prefix"]) == 2
    assert batch["merge_pivot_prefix"][1] is None
    no_ownership = dict(arrays)
    del no_ownership["history_is_cav"]
    assert TrajectoryDataset.from_arrays(no_ownership, merge_pivot_history=True)[0]["merge_pivot_prefix"] is None
    features_only = {key: val for key, val in arrays.items() if not key.startswith("history_")}
    features_only["hist"] = np.ones((3, 10, 6), np.float32)
    assert TrajectoryDataset.from_arrays(features_only, merge_pivot_history=True)[0]["merge_pivot_prefix"] is None


def test_observed_entry_recovers_waiting_identity_and_reports_auditable_counts():
    model, inp, targets = _fixture()
    term, prep = _prepare(model, targets)
    initial = pre_window_maximum(model, prep["pre_window_history"], inp.states[:, 0, :, 0])
    assert torch.isfinite(initial[0, 0]) and torch.isnan(initial[0, 1:]).all()
    c = model.merge_pivot(prep["geometry"])
    tau = model.merge_pivot.params()["tau"]
    history = prep["pre_window_history"][0]
    prefix_c = model.merge_pivot(history["geometry"])
    prior = torch.where(history["history_mask"], prefix_c, -torch.inf).amax(1)[:, 0]
    expected = (-log_event_mass(torch.maximum(prior, c[:, 0, 0]), c[:, 1, 0], tau)
                + log_survival(prior, tau)).sum()
    actual = term(LossContext(pred={}, current=inp.initial, targets=targets, c0=0, c1=2)).value
    assert torch.allclose(actual, expected, atol=1e-6, rtol=0)
    parameters = tuple(model.merge_pivot.parameters())
    actual_grad = torch.autograd.grad(actual, parameters)
    expected_grad = torch.autograd.grad(expected, parameters)
    for got, want in zip(actual_grad, expected_grad):
        assert torch.isfinite(got).all() and torch.allclose(got, want, atol=1e-6, rtol=0)
    metrics = merge_pivot_metrics(prep, .5)
    assert metrics["merge_pivot_unknown_age_before"] == 1
    assert metrics["merge_pivot_unknown_age_after"] == 0
    assert metrics["merge_pivot_pre_window_recovered_sequences"] == [[0, 0, 0]]
    assert metrics["merge_pivot_pre_window_replayed_intervals"] == 3
    assert metrics["merge_pivot_tf_unknown_age"] == 0
    assert metrics["merge_pivot_tf_positive_mass_events"] == 1
    assert metrics["merge_pivot_tf_scored_intervals"] == 2


def test_pre_window_boundary_graph_rebuilt_for_each_tbptt_backward():
    model, inp, targets = _fixture()
    term, _ = _prepare(model, targets)
    full = term(LossContext(pred={}, current=inp.initial, targets=targets, c0=0, c1=2)).value
    full.backward()
    expected = {key: val.grad.clone() for key, val in model.merge_pivot.named_parameters()}
    model.zero_grad(set_to_none=True)
    term, _ = _prepare(model, targets)
    total = 0.
    for c0, c1 in [(0, 1), (1, 2)]:
        value = term(LossContext(pred={}, current=inp.initial, targets=targets, c0=c0, c1=c1)).value
        total += float(value.detach())
        value.backward()
    assert total == pytest.approx(float(full.detach()), abs=1e-6)
    for key, val in model.merge_pivot.named_parameters():
        assert torch.isfinite(val.grad).all()
        assert torch.allclose(val.grad, expected[key], atol=1e-6, rtol=0), key
    assert expected["tau_raw"].abs() > 0
    assert all(p.grad is None for key, p in model.named_parameters() if not key.startswith("merge_pivot."))


@pytest.mark.parametrize("case", ["no_prefix", "starts_waiting", "identity_break", "missing_tick",
                                 "prior_event", "prior_exit", "unknown_commands", "bad_scene",
                                 "short_commands", "unobserved_direct"])
def test_incomplete_entry_or_terminated_history_retains_unknown_age(case):
    model, inp, targets = _fixture()
    prefix = inp.merge_pivot_prefix[0]
    if case == "no_prefix":
        inp.merge_pivot_prefix = None
    elif case == "starts_waiting":
        for key in ("states", "present", "is_cav"):
            prefix[key] = prefix[key][-2:]
        prefix["continue"] = prefix["continue"][-1:]
    elif case == "identity_break":
        prefix["continue"][-1, 0] = False
    elif case == "missing_tick":
        prefix["present"][1, 0] = False
        prefix["continue"][1, 0] = False
    elif case == "prior_event":
        prefix["states"][2, 0, 1:4] = [0, 1, 0]
    elif case == "prior_exit":
        prefix["states"][2, 0, 0] = 100.
    elif case == "unknown_commands":
        prefix["is_cav"][:, 1] = True
    elif case in ("short_commands", "unobserved_direct"):
        prefix["is_cav"][:, 1] = True
        count = 1 if case == "short_commands" else 2
        ids = np.zeros((count, 3), np.int64)
        ids[:, 1] = 3
        prefix["action_timeline"] = dict(
            controller=ids, acceleration_mps2=np.zeros((count, 3)),
            target_speed_mps=np.zeros((count, 3)), command_known=np.zeros((count, 3), bool))
    else:
        prefix["states"][2, 1, 1:4] = 0
    term = MergePivotTerm(MergePivotConfig(tf_weight=1.))
    term.prepare(targets)
    value = term(LossContext(pred={}, current=inp.initial, targets=targets, c0=0, c1=2)).value
    metrics = merge_pivot_metrics(targets.prepared["merge_pivot"], .5)
    assert value == 0
    assert metrics["merge_pivot_unknown_age_before"] == metrics["merge_pivot_unknown_age_after"] == 1
    assert metrics["merge_pivot_tf_unknown_age_events"] == 1
    assert metrics["merge_pivot_tf_scored_intervals"] == 0


def test_prefix_replay_uses_same_causal_motion_and_current_cav_commands():
    model, inp, _ = _fixture()
    prefix = inp.merge_pivot_prefix[0]
    prefix["is_cav"][:, 1] = True
    ids = np.zeros((2, 3), np.int64)
    ids[:, 1] = 3
    prefix["action_timeline"] = dict(controller=ids, acceleration_mps2=np.full((2, 3), -2.),
                                      target_speed_mps=np.full((2, 3), 2.), control_dt_s=1.)
    history = build_pre_window_history(model, inp, MergePivotConfig())[0]
    raw = SimpleNamespace(states=torch.as_tensor(prefix["states"])[None],
                          present=torch.as_tensor(prefix["present"])[None],
                          is_cav=torch.as_tensor(prefix["is_cav"])[None],
                          continuity=torch.as_tensor(prefix["continue"])[None],
                          hist=None, hist_mask=None, observed_hist=None, observed_hist_mask=None,
                          controller_id=None, action_timeline={key: (val if key == "control_dt_s" else
                              torch.as_tensor(val)[None]) for key, val in prefix["action_timeline"].items()})
    expected = build_recorded_geometry(model, raw, SimpleNamespace(n_steps=3))
    for key in expected:
        assert torch.equal(expected[key], history["geometry"][key]), key
        assert not history["geometry"][key].requires_grad
    # No t0 or later decision is replayed. Even t0 motion is not a prefix input.
    prefix["states"][-1, :, 4:6] += 500.
    inp.states[:, 0, :, 4:6] += 500.
    inp.states[:, 1:, :, 0] += 10000.
    inp.teacher = torch.full((1, 3, 3, 4), 99999.)
    changed = build_pre_window_history(model, inp, MergePivotConfig())[0]
    for key in expected:
        assert torch.equal(expected[key], changed["geometry"][key]), key


def test_earlier_slot_occupant_never_contributes_to_recovered_maximum():
    arrays = _arrays()
    # First occupant is already waiting. New occupant is observed outside the
    # zone before entering, with explicit reuse even though presence never drops.
    arrays["history_kin"][0, 0, 0] = 50.
    arrays["history_target_mask"][0, 0] = False
    model, inp, targets = _fixture(arrays)
    term, prep = _prepare(model, targets)
    history = prep["pre_window_history"][0]
    assert history["certified"][0, 0]
    assert not history["history_mask"][0, 0, 0]
    before = pre_window_maximum(model, [history], inp.states[:, 0, :, 0])
    for side in ("f", "l"):
        history["geometry"][f"gap_{side}"][:, 0, 0] = 1e12
    after = pre_window_maximum(model, [history], inp.states[:, 0, :, 0])
    assert torch.equal(before[:, :1], after[:, :1])


@pytest.mark.parametrize("prior", [-torch.inf, torch.inf])
def test_prior_limit_cases_are_exact_and_never_floored(prior):
    c = torch.tensor([[[.5]]])
    yes = torch.ones_like(c, dtype=torch.bool)
    nll, _, counts = score_chunk(c, torch.tensor(.7), yes, yes, known_entry=~yes,
                                 reset=~yes, initial_maximum=torch.tensor([[prior]]))
    if prior == -torch.inf:
        assert torch.allclose(nll, -log_event_mass(torch.tensor(-torch.inf), c.squeeze(), torch.tensor(.7)))
    else:
        assert torch.isposinf(nll) and counts["zero_survival"] == 1


def test_prefix_all_vetoes_certifies_entry_with_negative_infinity_not_unknown():
    model, inp, targets = _fixture()
    term, prep = _prepare(model, targets)
    history = prep["pre_window_history"][0]
    history["history_mask"].zero_()
    initial = pre_window_maximum(model, [history], inp.states[:, 0, :, 0])
    assert torch.isneginf(initial[0, 0])
    assert torch.isnan(initial[0, 1:]).all()
    assert torch.isfinite(term(LossContext(pred={}, current=inp.initial, targets=targets, c0=0, c1=2)).value)


def test_pack_raw_prefix_reads_only_same_run_past_and_no_truth_or_outcomes():
    arrays = _arrays()
    all_states = np.concatenate((np.zeros((2, 3, 7), np.float32), arrays["history_kin"],
                                 np.zeros((3, 3, 7), np.float32)))
    allowed = dict(veh_kin=all_states, veh_mask=np.ones((9, 3), bool),
                   target_mask=np.ones((9, 3), bool), veh_step=np.zeros((9, 3, 14)))
    reads = []

    class Bounded:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, index):
            rows = index[0] if isinstance(index, tuple) else index
            assert rows.start == 2
            assert rows.stop <= (5 if self.name == "target_mask" else 6)
            reads.append(self.name)
            return allowed[self.name][index]

    corpus = SimpleNamespace(run_bounds=lambda tick: (2, 9), array=lambda key: Bounded(key))
    prefix = recorded_prefix(corpus, 5)
    assert set(reads) == set(allowed)
    assert np.array_equal(prefix["states"], arrays["history_kin"])
    assert prefix["continue"].shape == (3, 3)


def test_prefix_direct_commands_do_not_read_the_t0_command():
    accesses = []

    class Commands:
        def __getitem__(self, index):
            accesses.append(index)
            assert index.start == 10 and index.stop == 13 and index.step == 2
            return np.ones((2, 3, 2), np.float32)

    loader = _PackWindowLoader(SimpleNamespace(array=lambda key: Commands(), run_end=lambda t: 50),
                               [], [], [], horizon_steps=2, start_shift_prob=0., timelines=None,
                               entries_v2=False, history_s=None, history_stride_s=1.)
    timeline = loader._fixed_timeline("direct", {"tick": 10}, 4, 3, decision_steps=3)
    assert len(accesses) == 1 and timeline["controller"].shape == (2, 3)


def test_history_is_ignored_outside_first_passage_mode():
    model, inp, _ = _fixture()
    batch = dict(states=inp.states, present=inp.present, merge_pivot_prefix=[{"invalid": True}])
    model.merge_decision = "secure_gap"
    settings = TrainSettings(rollout_mode="hybrid", lane_model="hazard_v5")
    assert prepare_inputs(batch, model, settings, torch.device("cpu"), 2).merge_pivot_prefix is None


def test_entry_first_observed_at_t0_has_exact_empty_survival_prior():
    arrays = _arrays()
    arrays["history_kin"] = arrays["history_kin"][-2:].copy()
    arrays["history_kin"][0, 0, 0] = -100.
    arrays["history_mask"] = arrays["history_mask"][-2:]
    arrays["history_is_cav"] = arrays["history_is_cav"][-2:]
    arrays["history_target_mask"] = arrays["history_target_mask"][-1:]
    model, inp, targets = _fixture(arrays)
    term = MergePivotTerm(MergePivotConfig(tf_weight=1.))
    term.prepare(targets)
    prep = targets.prepared["merge_pivot"]
    assert torch.isneginf(prep["pre_window_initial_maximum"][0, 0])
    assert prep["known"][0, 0, 0] and not prep["known_before"][0, 0, 0]
    assert prep["unknown_age_sequences"] == []


def test_pre_window_non_record_setting_event_is_infinite_and_enumerated():
    model, inp, targets = _fixture()
    term, prep = _prepare(model, targets)
    for side in ("f", "l"):
        prep["pre_window_history"][0]["geometry"][f"gap_{side}"][:, :, 0] = 1e6
    value = term(LossContext(pred={}, current=inp.initial, targets=targets, c0=0, c1=2)).value
    assert torch.isposinf(value)
    metrics = merge_pivot_metrics(prep, .5)
    assert metrics["merge_pivot_tf_zero_mass_events"] == 1
    assert metrics["merge_pivot_tf_unknown_age_events"] == 0


def test_onpolicy_score_uses_same_pre_window_survival_prior(monkeypatch):
    model, inp, targets = _fixture()
    term, prep = _prepare(model, targets)
    term.cfg = MergePivotConfig(weight=1.)
    initial = pre_window_maximum(model, prep["pre_window_history"], inp.states[:, 0, :, 0])
    c = torch.full((1, 2, 3), -torch.inf)
    c[0, :, 0] = torch.stack((initial[0, 0] + .2, initial[0, 0] + .5))
    pred = dict(merge_pivot_c=c, merge_pivot_eligible=torch.ones_like(c, dtype=torch.bool),
                **{"merge_pivot_" + key: val for key, val in model.merge_pivot.params().items()})
    monkeypatch.setattr(term, "_comparison", lambda ctx, prep: (
        torch.ones_like(c, dtype=torch.bool), torch.zeros_like(c, dtype=torch.bool)))
    value = term(LossContext(pred=pred, current=inp.initial, targets=targets, c0=0, c1=2)).value
    tau = model.merge_pivot.params()["tau"]
    expected = (-log_event_mass(c[:, 0, 0], c[:, 1, 0], tau)
                + log_survival(initial[:, 0], tau)).sum()
    assert torch.allclose(value, expected, atol=1e-6, rtol=0)
    assert merge_pivot_metrics(prep, .5)["merge_pivot_on_unknown_age"] == 0


def test_ragged_missing_rows_and_entrant_padding_never_invent_priors():
    model, inp, _ = _fixture()
    histories = build_pre_window_history(model, inp, MergePivotConfig())
    maximum = pre_window_maximum(model, histories + [None], torch.zeros(2, 5))
    assert torch.isfinite(maximum[0, 0])
    assert torch.isnan(maximum[0, 1:]).all() and torch.isnan(maximum[1]).all()


def test_c4_prefix_carries_historical_neighbours_and_pre_t0_commands_only(tmp_path):
    raw = _arrays()
    commands = np.arange(9, dtype=np.float32).reshape(3, 3)
    arrays = dict(veh_kin=raw["history_kin"], veh_mask=raw["history_mask"].copy(),
                  target_mask=raw["history_target_mask"], is_cav=raw["history_is_cav"],
                  time_s=np.arange(4) * .5, control_time_s=np.arange(3),
                  controller=np.zeros((3, 3), np.int64),
                  issued_acceleration_mps2=commands, issued_target_speed_mps=commands + 10)
    arrays["veh_mask"][-1, 2] = False  # departed neighbour is still in the historical scene
    ep = SimpleNamespace(array=lambda name: arrays[name], run_bounds=lambda tick: (0, 4))
    loader = C4WindowLoader(tmp_path, [], [], horizon_steps=2, history_s=None,
                            history_stride_s=1., entries_v2=False, merge_pivot_history=True)
    prefix = loader.recorded_prefix(ep, 2)
    assert prefix["present"][0, 2]
    assert np.array_equal(prefix["action_timeline"]["acceleration_mps2"], commands[:1])
    assert prefix["states"].shape[0] == 3
    arrays["is_cav"][:, 1] = True
    arrays["control_time_s"] = np.arange(3) + 1.
    assert loader.recorded_prefix(ep, 2) is None  # no command at history start


def test_absent_slot_packing_differences_cannot_erase_other_observed_histories():
    # An empty slot's lane one-hot can differ in packed windows versus native arrays.
    # Use the explicit raw-record API, since array histories check their t0 anchor.
    model, inp, _ = _fixture()
    prefix = inp.merge_pivot_prefix[0]
    prefix["present"][:, 2] = False
    prefix["states"][-1, 2, 1:4] = 0
    inp.present[:, 0, 2] = False
    histories = build_pre_window_history(model, inp, MergePivotConfig())
    assert histories[0] is not None and histories[0]["certified"][0, 0]


def test_c4_source_opt_in_reaches_real_window_loader_without_changing_default(monkeypatch, tmp_path):
    from wms.data import c4_episodes as c4

    raw = _arrays()
    states = np.concatenate((raw["history_kin"], raw["states"][1:]), 0)
    arrays = dict(veh_kin=states, veh_mask=np.ones((6, 3), bool),
                  target_mask=np.ones((6, 3), bool), vid=np.tile(np.arange(3), (6, 1)),
                  is_cav=np.zeros((6, 3), bool), time_s=np.arange(6) * .5,
                  control_time_s=np.arange(3), controller=np.zeros((3, 3), np.int64),
                  issued_acceleration_mps2=np.zeros((3, 3), np.float32),
                  issued_target_speed_mps=np.zeros((3, 3), np.float32))
    ep = SimpleNamespace(array=lambda name: arrays[name], run_bounds=lambda tick: (0, 6),
                         run_start=lambda tick: 0, has=lambda key: key in arrays)
    doc = dict(k_slots=3, n_episodes=1, episodes=[dict(
        world="w32m2700p20t1", arm="native", policy="native", episode="toy", dir="toy",
        warmup_s=1., split_role="train", controller="native", controllers_used=["native"])])
    monkeypatch.setattr(c4, "read_source_manifest", lambda root: doc)
    monkeypatch.setattr(c4, "_open_episode", lambda path: ep)
    monkeypatch.setattr(c4, "_sha256", lambda path: "synthetic")
    off = c4.build_c4_datasets(tmp_path, horizon_steps=2)["train"][0]
    on = c4.build_c4_datasets(tmp_path, horizon_steps=2, merge_pivot_history=True)["train"][0]
    assert "merge_pivot_prefix" not in off
    assert on["merge_pivot_prefix"]["states"].shape == (3, 3, 7)
    model, _, _ = _fixture()
    batch = collate([on])
    settings = TrainSettings(rollout_mode="hybrid", lane_model="hazard_v5",
                             merge_decision="first_passage")
    inp = prepare_inputs(batch, model, settings, torch.device("cpu"), 2)
    histories = build_pre_window_history(model, inp, MergePivotConfig())
    assert histories[0]["certified"][0, 0]


def test_long_history_prefix_reencoding_at_identity_birth_is_causal():
    # Supply the ordinary t0 encoder feature history required by this model;
    # the prefix replay must construct its OWN historical encoder inputs.
    arrays = _arrays()
    arrays["history_target_mask"][0, 0] = False
    torch.manual_seed(32)
    model = WmSModel(ModelSpec(rollout_mode="hybrid", lane_model="hazard_v5",
                               merge_decision="first_passage", history_s=2.,
                               history_stride_s=.5), physics=PhysicsConfig()).eval()
    record = TrajectoryDataset.from_arrays(arrays, history_s=2., history_stride_s=.5,
                                           merge_pivot_history=True)[0]
    batch = collate([record])
    settings = TrainSettings(rollout_mode="hybrid", lane_model="hazard_v5",
                             merge_decision="first_passage", history_s=2., history_stride_s=.5)
    inp = prepare_inputs(batch, model, settings, torch.device("cpu"), 2)
    before = build_pre_window_history(model, inp, MergePivotConfig())[0]
    inp.observed_hist.fill_(9999.)
    after = build_pre_window_history(model, inp, MergePivotConfig())[0]
    assert before["certified"][0, 0]
    for key in before["geometry"]:
        assert torch.equal(before["geometry"][key], after["geometry"][key]), key


def test_missing_raw_continuity_sidecar_retains_unknown_without_feature_fallback():
    arrays = _arrays()
    del arrays["history_target_mask"]
    record = TrajectoryDataset.from_arrays(arrays, merge_pivot_history=True)[0]
    assert record["merge_pivot_prefix"] is None


def test_prefix_rng_is_independent_of_future_window_length_and_preserves_caller_rng(monkeypatch):
    from wms.train.losses import merge_pivot as loss

    model, _, targets = _fixture()
    original = loss.build_recorded_geometry

    def stochastic_replay(model, inp, targets):
        torch.rand(targets.n_steps)  # stand-in for stochastic frozen forward calls
        geo = original(model, inp, targets)
        geo["gap_f"] = geo["gap_f"] + torch.rand_like(geo["gap_f"])
        return geo

    monkeypatch.setattr(loss, "build_recorded_geometry", stochastic_replay)
    term = MergePivotTerm(MergePivotConfig(tf_weight=1.))
    rng = torch.random.get_rng_state().clone()
    term.prepare(targets)
    before = targets.prepared["merge_pivot"]["pre_window_history"][0]["geometry"]["gap_f"]
    targets.n_steps = 1
    term.prepare(targets)
    after = targets.prepared["merge_pivot"]["pre_window_history"][0]["geometry"]["gap_f"]
    assert torch.equal(before, after)
    assert torch.equal(rng, torch.random.get_rng_state())
