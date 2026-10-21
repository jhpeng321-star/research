"""R9K CPU contract checks on synthetic recorded states, never evaluation worlds."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from wms.config.load import load_train_config
from wms.config.train import TrainConfig
from wms.model.leader_change import (LEADER_CHANGE_FEATURES, VALID_MASKED,
    LeaderChangeTracker, feature_matrix, leader_geometry, recorded_leader_change)
from wms.model.residual_net import HdvConfig
from wms.model.spec import ModelSpec
from wms.model.history import LONG_HISTORY_F
from wms.model.wms import WmSModel
from wms.train.inputs import build_targets, prepare_inputs, state_from_kin
from wms.train.losses.base import UnrollContext
from wms.train.losses.compose import build_loss
from wms.train.losses.cutin_reaction import (CutinReactionConfig, CutinReactionLoss,
    cutin_reaction_terms, insertion_events, insertion_plan, one_step_predicted_accel)
from wms.train.settings import TrainSettings
from wms.train.trainer import build_optimizer
from wms.train.unroll import UnrollRunner

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def sequence(two=False):
    states = torch.zeros(1, 9, 4, 7)
    states[..., 0] = torch.tensor([0., 100., 40., 20.])
    states[..., 4] = 10.
    states[..., 5] = -1.
    states[..., 6] = 5.
    lanes = torch.tensor([1, 1, 2, 2]).expand(1, 9, 4).clone()
    lanes[:, 2:, 2] = 1
    if two:
        lanes[:, 4:, 3] = 1
    states[..., 1:4] = F.one_hot(lanes, 3)
    states[:, 6:, 0, 5] = .5  # response recovery is observed, not a fixed window
    return {"states": states, "present": torch.ones(1, 9, 4, dtype=torch.bool),
            "is_cav": torch.zeros(1, 9, 4, dtype=torch.bool)}


def targets(batch):
    # Use the actual trainer's target construction, including its output-row mask.
    class Model:
        long_history = False
        rollout_mode = "residual"
        physics = None
    cfg = TrainSettings()
    inp = prepare_inputs(batch, Model(), cfg, torch.device("cpu"), 8)
    return build_targets(inp, batch, cfg, .5, torch.device("cpu"))


def context(model, t):
    return UnrollContext(model, {"veh_kin": t.states[:, 0],
                                "present": t.loss_valid[:, 0],
                                "is_cav": t.is_cav[:, 0]}, t.loss_valid[:, 0], t)


class AccelOutput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.accel = torch.nn.Parameter(torch.zeros(1, 8, 4))
        self.calls = []

    def forward(self, state, *, k_steps, **kwargs):
        assert state["veh_kin"].shape == (1, 4, 7)
        self.calls.append((k_steps, state["veh_kin"].detach().clone()))
        return {"accel_mps2": self.accel[:, k_steps]}


def test_no_events_and_weight_zero_never_calls_model():
    batch = sequence()
    batch["states"][..., 1:4] = torch.tensor([0., 1., 0.])
    t = targets(batch)
    assert insertion_events(t, CutinReactionConfig(weight=1)) == []
    model = AccelOutput()
    zero = CutinReactionLoss(CutinReactionConfig())(context(model, t))
    assert zero.value.item() == 0 and model.calls == []
    assert cutin_reaction_terms(CutinReactionConfig()) == []


def test_two_insertions_have_first_lane1_timestamps_and_data_windows():
    t = targets(sequence(two=True))
    events = insertion_events(t, CutinReactionConfig(weight=1))
    assert [(e["step"], e["follower"]) for e in events] == [(2, 0), (4, 0)]
    assert [(e["start"], e["end"]) for e in events] == [(0, 5), (2, 5)]
    assert not any(e["censored"] for e in events)
    plan = insertion_plan(t, CutinReactionConfig(weight=1))
    assert plan["event_weights"].sum().item() == pytest.approx(1.)
    assert plan["event_weights"][0, 2, 0] > plan["event_weights"][0, 0, 0]


def test_response_maximum_never_exceeds_six_seconds():
    batch = sequence()
    batch["states"][:, :, 0, 5] = -1
    t = targets(batch)
    event = insertion_events(t, CutinReactionConfig(weight=1, pre_s=0, max_window_s=1))[0]
    assert (event["start"], event["end"], event["censored"]) == (2, 3, True)
    with pytest.raises(ValueError):
        CutinReactionConfig(max_window_s=6.1)


@pytest.mark.parametrize("explicit", [False, True])
def test_slot_reuse_censors_window_and_cannot_create_insertion(explicit):
    batch = sequence()
    if not explicit:
        batch["present"][:, 4, 0] = False
        batch["present"][:, 1, 2] = False
    t = targets(batch)
    if explicit:
        reset = torch.zeros_like(t.loss_valid)
        reset[:, 4, 0] = True
        reset[:, 2, 2] = True
        t.prepared["cutin_reaction_inputs"] = {"reset": reset}
    assert insertion_events(t, CutinReactionConfig(weight=1)) == []
    # Restore inserter continuity, but never follow the replacement follower.
    if explicit:
        reset[:, 2, 2] = False
    else:
        t.loss_valid[:, 1, 2] = True
    event = insertion_events(t, CutinReactionConfig(weight=1))[0]
    assert event["end"] == 2


def test_nearest_cav_follower_is_not_replaced_by_distant_hdv():
    batch = sequence()
    batch["is_cav"][:, :, 0] = True
    assert insertion_events(targets(batch), CutinReactionConfig(weight=1)) == []


def test_event_balancing_controls_and_accel_output_gradients():
    t = targets(sequence(two=True))
    cfg = CutinReactionConfig(weight=3, control_weight=2)
    model = AccelOutput()
    loss = CutinReactionLoss(cfg)
    plan = insertion_plan(t, cfg)
    value = loss(context(model, t)).value
    errors = F.huber_loss(model.accel, t.states[:, 1:, :, 5], delta=.5, reduction="none")
    events = [errors[e["batch"], e["start"]:e["end"] + 1, e["follower"]].mean()
              for e in plan["events"]]
    expected = 3 * (torch.stack(events).mean() + 2 * errors[plan["control"]].mean())
    torch.testing.assert_close(value, expected)
    value.backward()
    assert model.accel.grad[0, 2, 0] > 0  # follower over-braking error reaches accel
    assert model.accel.grad[0, 2, 1] > 0  # ordinary braking control remains anchored
    assert model.accel.grad[0, 0, 3] == 0  # unrelated ramp row
    assert loss.last_stats["n_events"] == 2


def test_empty_enabled_batch_backward_and_repeated_calls_are_safe():
    batch = sequence()
    batch["present"].zero_()
    t = targets(batch)
    model = AccelOutput()
    loss = CutinReactionLoss(CutinReactionConfig(weight=1))
    loss(context(model, t)).value.backward()
    assert model.calls == []
    t = targets(sequence())
    for _ in range(2):
        loss(context(model, t)).value.backward()  # no cached freed autograd graph


def test_acceleration_labels_are_next_row_and_never_model_inputs():
    t = targets(sequence())
    t.states.requires_grad_()
    model = AccelOutput()
    CutinReactionLoss(CutinReactionConfig(weight=1))(context(model, t)).value.backward()
    assert t.states.grad is None
    for step, seen in model.calls:
        assert torch.equal(seen, t.states[:, step].detach())
    # Last output targets positive recovery, earlier outputs target braking.
    assert model.accel.grad[0, 4, 0] > 0 and model.accel.grad[0, 5, 0] < 0


def test_tracker_types_masks_and_flags_persist_across_chunks():
    batch = sequence()
    kin, present = batch["states"], batch["present"]
    # cut-in at 2, leader exits at 5, follower's own lane change at 7.
    kin[:, 5:, 2, 1:4] = torch.tensor([1., 0., 0.])
    kin[:, 7:, 0, 1:4] = torch.tensor([1., 0., 0.])
    whole = recorded_leader_change(kin, present)
    assert not whole["t_since_change_s_valid"][0, 0, 0]
    assert whole["changed_cut_in"][0, 2:5, 0].tolist() == [1., 1., 1.]
    assert whole["t_since_change_s"][0, 4, 0] == 1
    assert whole["changed_leader_exit"][0, 5, 0] == 1
    assert whole["changed_own_lane"][0, 7, 0] == 1
    tracker = LeaderChangeTracker(1, 4)
    first = recorded_leader_change(kin[:, :4], present[:, :4], tracker=tracker)
    tracker.detach_()
    restored = LeaderChangeTracker(1, 4)
    restored.load_state_dict(tracker.state_dict())
    last = recorded_leader_change(kin[:, 4:], present[:, 4:], tracker=restored)
    for name in whole:
        assert torch.equal(whole[name], torch.cat((first[name], last[name]), 1)), name
    matrix = feature_matrix({name: value[:, 0] for name, value in whole.items()})
    assert matrix.shape[-1] == len(LEADER_CHANGE_FEATURES) + len(VALID_MASKED)
    assert torch.isfinite(matrix).all()


def test_newly_observed_leader_is_not_a_cut_in_and_absence_clears_history():
    batch = sequence()
    batch["present"][:, 1, 2] = False
    batch["present"][:, 4, 0] = False
    out = recorded_leader_change(batch["states"], batch["present"])
    assert out["changed_cut_in"][0, 2, 0] == 0
    assert not out["t_since_change_s_valid"][0, 5, 0]


def test_explicit_identity_reset_of_follower_and_leader():
    batch = sequence()
    reset = torch.zeros_like(batch["present"])
    reset[:, 4, 0] = True
    out = recorded_leader_change(batch["states"], batch["present"], reset=reset)
    assert not out["t_since_change_s_valid"][0, 4, 0]
    reset.zero_()
    reset[:, 4, 2] = True
    out = recorded_leader_change(batch["states"], batch["present"], reset=reset)
    assert out["changed_cut_in"][0, 4, 0] == 0
    assert not out["prev_leader_gap_m_valid"][0, 4, 0]


def test_zero_gap_history_is_valid_and_cav_is_an_ordinary_leader():
    batch = sequence()
    batch["states"][..., 1, 0] = 5
    out = recorded_leader_change(batch["states"], batch["present"])
    # Leader slot 1 exits, revealing slot 2; previous zero bumper gap is real.
    batch["present"][:, 3:, 1] = False
    out = recorded_leader_change(batch["states"], batch["present"])
    assert out["prev_leader_gap_m_valid"][0, 3, 0]
    assert out["prev_leader_gap_m"][0, 3, 0] == 0


def test_geometry_uses_nearest_front_position_not_vehicle_length():
    s = torch.tensor([[0., 10., 20.]])
    geo = leader_geometry(s, torch.ones_like(s), s, torch.ones_like(s).bool(),
                          torch.tensor([[5., 1., 19.]]))
    assert geo["index"][0, 0] == 1


@pytest.mark.parametrize("shape", [(1, 0, 4, 7), (1, 2, 0, 7)])
def test_empty_recorded_tracker(shape):
    out = recorded_leader_change(torch.zeros(shape), torch.zeros(shape[:3], dtype=torch.bool))
    assert all(value.shape == shape[:3] for value in out.values())


def small_model(**kwargs):
    torch.manual_seed(31)
    return WmSModel(ModelSpec(history_hidden=16, k_hidden=8, **kwargs),
                    hdv=HdvConfig(d_model=16, n_heads=2, n_layers=1, ffn=32)).eval()


@pytest.mark.parametrize("slow", [False, True])
def test_real_model_reachability_optimizer_update_and_recorded_history(slow):
    model = small_model(temporal_residual_mode="slow" if slow else "direct")
    t = targets(sequence())
    term = CutinReactionLoss(CutinReactionConfig(weight=1))
    opt, _ = build_optimizer(model, lr=1e-3, weight_decay=0)
    value = term(context(model, t)).value
    value.backward()
    reached = {name for name, p in model.named_parameters()
               if p.grad is not None and p.grad.abs().sum() > 0}
    assert any(name.startswith("residual_net") for name in reached)
    assert any(name.startswith("temporal.") for name in reached)
    if slow:
        assert "temporal_tau_raw" in reached
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    before = {name: p.detach().clone() for name, p in model.named_parameters() if name in reached}
    opt.step()
    assert any(not torch.equal(before[name], p) for name, p in model.named_parameters() if name in before)


def test_temporal_replay_uses_only_causal_recorded_prefix():
    model = small_model()
    t = targets(sequence())
    first = one_step_predicted_accel(model, t)
    modified = targets(sequence())
    modified.states[:, 5:, :, 4:6] += 2
    second = one_step_predicted_accel(model, modified)
    assert torch.equal(first[:, :5], second[:, :5])


def test_round8_trainable_paths_and_frozen_parameters():
    settings = TrainSettings.from_config(load_train_config(ROOT / "configs/c4_wm_v10round8.toml"))
    torch.manual_seed(32)
    model = WmSModel(ModelSpec.from_mapping(
        load_train_config(ROOT / "configs/c4_wm_v10round8.toml").to_legacy_flat())).eval()
    optimizer, report = build_optimizer(model, lr=1e-4, weight_decay=0,
        freeze_prefixes=settings.freeze_prefixes, lr_scale=settings.lr_scale)
    assert report["unmatched_freeze_prefixes"] == []
    t = targets(sequence())
    history = t.states.new_zeros(1, 4, model.driver_history_len, LONG_HISTORY_F)
    history[..., 0] = 10 / 30
    history[..., 2:5] = t.states[:, 0, :, None, 1:4]
    history_mask = torch.zeros(history.shape[:-1], dtype=torch.bool)
    history_mask[..., -2:] = True  # causal observations available before window t0
    t.prepared["cutin_reaction_inputs"] = {"driver_history": history,
                                          "driver_history_mask": history_mask}
    loss = CutinReactionLoss(CutinReactionConfig(weight=1))(context(model, t)).value
    loss.backward()
    reached = {name for name, p in model.named_parameters()
               if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0}
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(name.startswith("residual_net.accel") for name in reached)
    assert any(name.startswith("temporal.") for name in reached)
    assert model.rollout_mode == "hybrid"
    assert any(name.startswith("driver_encoder.") for name in reached)
    frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    assert "merge_core.temperature_raw" in frozen
    optimizer.step()
    assert all(torch.equal(frozen[name], p) for name, p in model.named_parameters() if name in frozen)
    # Real initialized-model reachability evidence; no claim about a trained checkpoint.
    print("R9K round8 reached:", sorted({name.split('.')[0] for name in reached}))


def test_hybrid_replay_can_reach_inferred_idm_parameters():
    model = small_model(rollout_mode="hybrid")
    t = targets(sequence())
    CutinReactionLoss(CutinReactionConfig(weight=1))(context(model, t)).value.backward()
    assert any(name.startswith("driver_encoder.") and p.grad is not None
               and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for name, p in model.named_parameters())


def test_prepared_history_reaches_replay_and_slot_reset_clears_it():
    model = small_model()
    t = targets(sequence())
    history = torch.ones(1, 4, model.history_len, model.history_features)
    mask = torch.ones(history.shape[:-1], dtype=torch.bool)
    reset = torch.zeros_like(t.loss_valid)
    reset[:, 0, 0] = True
    reset[:, 4, 0] = True
    t.prepared["cutin_reaction_inputs"] = {"history": history, "history_mask": mask, "reset": reset}
    masks = []
    def observe(module, args, kwargs):
        masks.append(kwargs["history_mask"].clone())
    handle = model.register_forward_pre_hook(observe, with_kwargs=True)
    one_step_predicted_accel(model, t)
    handle.remove()
    assert not masks[0][0, 0].any()
    assert masks[0][0, 1].all()
    assert masks[4][0, 0].sum() == 1
    assert history.eq(1).all() and mask.all()  # never mutate runner inputs


def test_model_default_checkpoint_remains_strictly_loadable():
    old = small_model()
    state = old.state_dict()
    new = small_model()
    result = new.load_state_dict(state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert not any("leader_change" in name for name in state)


@pytest.mark.parametrize("slow", [False, True])
def test_replay_matches_rollout_with_every_recorded_state_revealed(slow):
    model = small_model(temporal_residual_mode="slow" if slow else "direct")
    t = targets(sequence())
    state = state_from_kin(t.states[:, 0], t.loss_valid[:, 0])
    state["is_cav"] = t.is_cav[:, 0]
    replay = one_step_predicted_accel(model, t)
    rollout = model.rollout(state, n_steps=t.n_steps,
        observe_states=t.states, observe_mask=t.loss_valid, is_cav_seq=t.is_cav)
    torch.testing.assert_close(replay, rollout["accel"], rtol=0, atol=1e-6)


def test_config_toml_settings_composer_and_default_flat_hash(tmp_path):
    path = tmp_path / "cutin.toml"
    path.write_text('[model]\nleader_change_features = true\n'
                    '[loss.cutin_reaction]\nweight = 0.7\ncontrol_weight = 2.0\n'
                    'pre_s = 0.5\nmax_window_s = 4.0\nhuber_delta_mps2 = 0.8\n', encoding="utf-8")
    cfg = load_train_config(path)
    settings = TrainSettings.from_config(cfg)
    assert settings.leader_change_features is True
    assert TrainConfig.from_legacy_flat(cfg.to_legacy_flat()).to_legacy_flat() == cfg.to_legacy_flat()
    composer = build_loss(settings)
    term = next(term for term in composer.unroll_terms if term.name == "cutin_reaction")
    assert term.config == CutinReactionConfig(weight=.7, control_weight=2., pre_s=.5,
                                              max_window_s=4., huber_delta_mps2=.8)
    assert "cutin_reaction" in cfg.required_log_terms()
    assert "cutin_reaction" not in build_loss(TrainSettings()).names
    old = load_train_config(ROOT / "configs/c4_wm_v10round7b.toml").to_legacy_flat()
    digest = hashlib.sha256(json.dumps(old, sort_keys=True, default=str).encode()).hexdigest()
    assert digest == "db0d9376e013e22febbdd44ebd75516fc9585384c0f48192d0897b9e300ef011"
    assert not any(key.startswith("cutin_reaction") or key == "leader_change_features" for key in old)


@pytest.mark.parametrize("field,value", [("weight", -1), ("weight", float("nan")),
    ("control_weight", 0), ("pre_s", 2), ("max_window_s", 7), ("huber_delta_mps2", 0)])
def test_config_validation_at_all_boundaries(field, value):
    with pytest.raises(ValueError):
        CutinReactionConfig(**{field: value})
    with pytest.raises(ValueError):
        TrainConfig.from_dict({"loss": {"cutin_reaction": {field: value}}})
    with pytest.raises(ValueError):
        TrainSettings(**{f"cutin_reaction_{field}": value})


def test_real_runner_default_off_identical_and_enabled_backward():
    model = small_model()
    cfg = TrainSettings(truncation_steps=2)
    batch = sequence()
    off = UnrollRunner(model, cfg, build_loss(cfg), projection=None, device=torch.device("cpu"))
    baseline = off.run(batch, unroll_steps=8, backward=False)
    explicit = replace(cfg, cutin_reaction_weight=0, leader_change_features=False)
    other = UnrollRunner(model, explicit, build_loss(explicit), projection=None,
                         device=torch.device("cpu")).run(batch, unroll_steps=8, backward=False)
    assert baseline == other
    on = replace(cfg, cutin_reaction_weight=1)
    model.zero_grad()
    runner = UnrollRunner(model, on, build_loss(on), projection=None, device=torch.device("cpu"))
    record = runner.run(batch, unroll_steps=8, backward=True)
    assert record["cutin_reaction"] > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
