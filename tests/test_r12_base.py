"""R12: base calibration switches (physical driver params from the learned centre, residual read-out re-init),
the replay visibility fix (filled presence), round-12 config pins and the documented rule file."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wms.cli.assemble import build_model, build_physics
from wms.config.load import load_raw, load_train_config
from wms.config.train import TrainConfig
from wms.model.checkpoint import save_checkpoint
from wms.model.driver_encoder import PHYSICAL_LOWER, PHYSICAL_UPPER
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.physics.config import PhysicsConfig
from wms.train.inputs import build_targets, prepare_inputs
from wms.train.losses.compose import build_loss
from wms.train.losses.cutin_reaction import _recorded, one_step_replay, replay_present
from wms.train.settings import TrainSettings
from wms.train.trainer import CheckpointInitError, WmSTrainer, build_optimizer

from test_r10b_courtesy import sequence, small_model

ROOT = Path(__file__).resolve().parents[1]
ROUND9 = ROOT / "configs/c4_wm_v10round9.toml"
ROUND11 = ROOT / "configs/c4_wm_v10round11.toml"
ROUND12 = ROOT / "configs/c4_wm_v10round12.toml"
HEAD = "follower_courtesy."


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def leaves(node, prefix=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaves(value, f"{prefix}{key}.")
    else:
        yield prefix[:-1], node


# ------------------------------------------------------------------ defaults and config pins
def test_new_switches_default_to_the_round11_behaviour_and_are_omitted_from_legacy_flat():
    cfg = TrainConfig.from_dict({})
    assert cfg.model.driver_param_init == "checkpoint" and cfg.model.residual_accel_reinit is False
    assert cfg.loss.recorded_replay_present == "loss_valid"
    flat = cfg.to_legacy_flat()
    assert not any(k in flat for k in ("driver_param_init", "residual_accel_reinit", "recorded_replay_present"))
    s = TrainSettings()
    assert (s.driver_param_init, s.residual_accel_reinit, s.recorded_replay_present) == (
        "checkpoint", False, "loss_valid")
    assert not any(k in load_train_config(ROUND11).to_legacy_flat()
                   for k in ("driver_param_init", "residual_accel_reinit", "recorded_replay_present"))


@pytest.mark.parametrize("bad", [{"model": {"driver_param_init": "centre"}},          # legacy mode
                                 {"model": {"driver_param_init": "mean"}},
                                 {"model": {"residual_accel_reinit": 1}},
                                 {"loss": {"recorded_replay_present": "all"}}])
def test_bad_switch_values_are_refused(bad):
    with pytest.raises(Exception):
        TrainConfig.from_dict(bad)


def test_round12_differs_from_round9_only_by_the_registered_changes():
    a, b = dict(leaves(load_raw(ROUND9))), dict(leaves(load_raw(ROUND12)))
    changed = {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
    allowed = {"name", "runtime.out_dir", "model.init_from", "model.driver_param_mode", "model.driver_param_init",
               "model.residual_accel_reinit", "model.follower_courtesy", "model.follower_courtesy_hidden",
               "model.follower_courtesy_embedding", "model.follower_courtesy_rollout_grad",
               "model.follower_courtesy_version", "model.follower_courtesy_init_zero",
               "loss.recorded_replay_present", "loss.cutin_reaction.weight", "loss.follower_courtesy.weight",
               "loss.follower_courtesy.huber_delta_mps2", "loss.follower_courtesy.contrast",
               "curriculum.stage_steps", "curriculum.max_hours", "optim.freeze_prefixes",
               "optim.lr_scale.merge_pivot", "optim.lr_scale.driver_encoder.head",
               "optim.lr_scale.driver_encoder.centre_logits", "optim.lr_scale.residual_net",
               "optim.lr_scale.temporal", "optim.lr_scale.follower_courtesy"}
    assert changed <= allowed, sorted(changed - allowed)
    assert b["model.init_from"].endswith("runs/c4_wm_v10_round9/wm_s_stage300.pt")
    # round-9 decisions inherited unchanged
    for key in ("model.merge_decision", "model.merge_passage_law", "model.merge_pivot_state_st",
                "model.checkpoint_forward", "model.merge_negotiation", "physics.merge_end_wait",
                "physics.ramp_lane_end_stop", "physics.ramp_risk_retain", "loss.merge_core.weight",
                "loss.merge_core.tf_weight", "loss.merge_core.yield_weight", "curriculum.unroll_s",
                "curriculum.truncation_steps_per_stage", "optim.lr_per_stage", "optim.batch_size_per_stage"):
        assert a[key] == b[key], key
    assert b["model.driver_param_mode"] == "physical" and b["model.driver_param_init"] == "centre"
    assert b["model.residual_accel_reinit"] is True and b["loss.recorded_replay_present"] == "filled"
    assert b["loss.cutin_reaction.weight"] == 0.0                      # fix B dropped
    assert b["model.follower_courtesy_version"] == 2 and b["model.follower_courtesy_init_zero"] is True
    assert b["model.follower_courtesy_rollout_grad"] is False and b["loss.follower_courtesy.contrast"] is True


def test_round12_effective_motion_objective_equals_round9():
    """The base retrain uses the round-8/9 motion objective unchanged (only the R12 keys may differ)."""
    f9, f12 = load_train_config(ROUND9).to_legacy_flat(), load_train_config(ROUND12).to_legacy_flat()
    r12_keys = {"name", "out_dir", "init_from", "driver_param_mode", "driver_param_init", "residual_accel_reinit",
                "recorded_replay_present", "follower_courtesy", "follower_courtesy_hidden",
                "follower_courtesy_embedding", "follower_courtesy_rollout_grad", "follower_courtesy_version",
                "follower_courtesy_init_zero", "follower_courtesy_weight", "follower_courtesy_huber_delta_mps2",
                "follower_courtesy_contrast", "stage_steps", "freeze_prefixes", "lr_scale"}
    diff = {k for k in set(f9) | set(f12) if f9.get(k, "<absent>") != f12.get(k, "<absent>")}
    assert diff <= r12_keys, sorted(diff - r12_keys)
    for key in ("position_weight", "speed_weight", "aggregate_weight", "agg_delay_weight", "agg_delay_paired_weight",
                "paired_diff_weight", "brake_persist_weight", "lane_event_weight", "lane_flow_weight",
                "residual_mag_weight", "driver_prior_weight", "driver_prior_target", "courtesy_weight", "grad_clip"):
        assert f12[key] == f9[key], key


def test_round12_trainable_set_loss_composition_and_frozen_merge_law():
    cfg = load_train_config(ROUND12)
    settings = TrainSettings.from_config(cfg)
    names = build_loss(settings).names
    assert "cutin_reaction" not in names and names[-1] == "follower_courtesy"
    assert settings.recorded_replay_present == "filled" and settings.driver_param_mode == "physical"
    model = build_model(cfg, build_physics(cfg))
    _, report = build_optimizer(model, lr=1e-4, weight_decay=0, freeze_prefixes=tuple(cfg.optim.freeze_prefixes),
                                lr_scale=dict(cfg.optim.lr_scale))
    assert report["unmatched_freeze_prefixes"] == [] and report["unmatched_lr_scale_prefixes"] == []
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    allowed = ("driver_encoder.head.", "driver_encoder.centre_logits", "residual_net.", "temporal.", HEAD)
    assert trainable and all(n.startswith(allowed) for n in trainable)
    assert not any(n.startswith(("residual_net.lane", "merge_pivot", "merge_core", "gate_", "lane_v5",
                                 "driver_encoder.history_proj", "driver_encoder.interaction_proj")) for n in trainable)
    assert "driver_encoder.centre_logits" in trainable and any(n.startswith("residual_net.accel") for n in trainable)
    assert any(n.startswith(HEAD) for n in trainable)


def test_round12_lr_rule_file_matches_the_config():
    rule = json.loads((ROOT / "docs/r12_lr_rule.json").read_text(encoding="utf-8"))
    cfg = load_train_config(ROUND12)
    assert rule["status"] == "MEASURED"
    assert cfg.optim.lr_scale["driver_encoder.head"] == pytest.approx(rule["chosen"]["head_scale"])
    assert cfg.optim.lr_scale["driver_encoder.centre_logits"] == pytest.approx(rule["chosen"]["centre_scale"])
    assert cfg.loss.follower_courtesy.weight == pytest.approx(rule["courtesy_weight"])
    assert cfg.optim.lr_scale["follower_courtesy"] == pytest.approx(rule["courtesy_lr_scale"])


def test_round12_gate_script_is_launchable():
    gate = (ROOT / "scripts/round12_gate.sh").read_bytes()
    assert b"\r" not in gate and b"configs/c4_wm_v10round12.toml" in gate
    assert b"runs/c4_wm_v10_round9/wm_s_stage300.pt" in gate and b"wms.train.round12_gate" in gate
    assert b"selection.val_max_windows=4" in gate and b"selection.val_max_windows=16" in gate
    assert b"R12_INIT" in gate and b"set -u -o pipefail -o noclobber" in gate


# ------------------------------------------------------------------ init switches
def _models(tmp_path, *, physical=True, centre=True, reinit=True):
    physics = PhysicsConfig()
    src_cfg = TrainConfig.from_dict({"model": {"rollout_mode": "hybrid"}})
    torch.manual_seed(0)
    src = WmSModel(ModelSpec.from_mapping(src_cfg.to_legacy_flat()), physics=physics)
    with torch.no_grad():                                        # a "trained" legacy head and live accel read-out
        src.driver_encoder.head[-1].weight.normal_(0, 0.5)
        src.driver_encoder.head[-1].bias.fill_(-3.0)
        src.residual_net.accel[-1].weight.normal_(0, 0.5)
        src.residual_net.accel[-1].bias.fill_(-2.0)
    init = tmp_path / "r9.pt"
    save_checkpoint(init, src, src_cfg)
    model_cfg = {"rollout_mode": "hybrid", "residual_accel_reinit": reinit}
    if physical:
        model_cfg.update(driver_param_mode="physical", driver_param_init="centre" if centre else "checkpoint")
    cfg = TrainConfig.from_dict({"model": model_cfg})
    torch.manual_seed(1)
    dst = WmSModel(ModelSpec.from_mapping(cfg.to_legacy_flat()), physics=physics)
    fake = SimpleNamespace(model=dst, settings=TrainSettings.from_config(cfg), out_dir=tmp_path / "out")
    return src, dst, fake, init


def test_centre_init_puts_every_observed_driver_exactly_on_the_learned_centre(tmp_path):
    src, dst, fake, init = _models(tmp_path)
    report = WmSTrainer._init_from_checkpoint(fake, str(init))
    assert report["driver_param_init"]["zeroed_rows"] == [0, 1, 2, 3, 4]
    assert set(report["new_in_model"]) == {"driver_encoder.centre_logits", "driver_encoder.physical_lower",
                                           "driver_encoder.physical_upper"}
    enc = dst.driver_encoder
    k, h = 6, 5
    out = enc(torch.randn(1, k, h, enc.history_features), torch.ones(1, k, h, dtype=torch.bool),
              torch.randn(1, k, 4), torch.rand(1, k, 9) + 1.0)
    lo, hi = enc.physical_lower, enc.physical_upper
    centre = lo + (hi - lo) * torch.sigmoid(enc.centre_logits)
    assert torch.allclose(out["values"][0, :, :5], centre.expand(k, 5), atol=1e-6)
    # lane-parameter means and every uncertainty row are the checkpoint's
    w_src, w_dst = src.driver_encoder.head[-1].weight, enc.head[-1].weight
    assert torch.equal(w_src[5:], w_dst[5:]) and torch.equal(src.driver_encoder.head[-1].bias[5:],
                                                            enc.head[-1].bias[5:])
    # physics-default centre (V6ENC): a, b, T, s0 of the default driver
    a, b, t, s0, _ = dst.physics.idm()
    assert torch.allclose(centre[1:], torch.tensor([t, s0, a, b]), atol=1e-5)
    assert json.loads((tmp_path / "out" / "INIT_FROM.json").read_text())["driver_param_init"]["mode"] == "centre"


def test_physical_values_reach_the_full_range_and_stay_inside_it(tmp_path):
    _, dst, fake, init = _models(tmp_path)
    WmSTrainer._init_from_checkpoint(fake, str(init))
    enc = dst.driver_encoder
    with torch.no_grad():
        enc.head[-1].weight[:5].normal_(0, 5.0)                 # extreme per-driver deviations
    out = enc(torch.randn(1, 64, 5, enc.history_features) * 3, torch.ones(1, 64, 5, dtype=torch.bool),
              torch.randn(1, 64, 4) * 3, torch.rand(1, 64, 9) + 1.0)["values"][0, :, :5]
    lo = torch.tensor(PHYSICAL_LOWER) * torch.tensor([dst.physics.speed_limit_mps, 1, 1, 1, 1])
    hi = torch.tensor(PHYSICAL_UPPER) * torch.tensor([dst.physics.speed_limit_mps, 1, 1, 1, 1])
    assert bool(((out >= lo) & (out <= hi)).all())
    assert float(out[:, 3].min()) < 1.0 and float(out[:, 4].min()) < 1.6   # below the legacy floors 1.78 / 2.87


def test_residual_reinit_zeroes_only_the_delta_row_with_a_live_gradient(tmp_path):
    src, dst, fake, init = _models(tmp_path)
    report = WmSTrainer._init_from_checkpoint(fake, str(init))
    last, ref = dst.residual_net.accel[-1], src.residual_net.accel[-1]
    assert report["residual_accel_reinit"]["zeroed_rows"] == [0] and report["residual_accel_reinit"]["abs_sum_before"] > 0
    assert float(last.weight[0].abs().sum() + last.bias[0].abs()) == 0.0
    assert torch.equal(last.weight[1], ref.weight[1]) and torch.equal(last.bias[1], ref.bias[1])
    x = torch.randn(3, last.in_features, requires_grad=False)
    y = torch.tanh(last(x)[:, 0]).sum()
    y.backward()
    assert float(last.weight.grad[0].abs().sum()) > 0             # tanh'(0) = 1: alive


def test_switches_off_load_the_checkpoint_unchanged(tmp_path):
    src, dst, fake, init = _models(tmp_path, physical=False, reinit=False)
    report = WmSTrainer._init_from_checkpoint(fake, str(init))
    assert "driver_param_init" not in report and "residual_accel_reinit" not in report
    for key, value in src.state_dict().items():
        assert torch.equal(dst.state_dict()[key], value), key


def test_init_switches_need_init_from_and_physical_mode(tmp_path):
    _, dst, fake, _ = _models(tmp_path)
    with pytest.raises(CheckpointInitError):
        WmSTrainer._init_from_checkpoint(fake, None)
    fake.settings = TrainSettings.from_config(TrainConfig.from_dict({"model": {"residual_accel_reinit": True}}))
    with pytest.raises(CheckpointInitError):
        WmSTrainer._init_from_checkpoint(fake, None)


# ------------------------------------------------------------------ replay visibility (fix D)
def junction_batch(n=9, window=None):
    """sequence() with the far lane-1 leader (slot 1) in a junction transit at t = 3, 4: present (filled lane) but
    not loss_valid.  ``window`` > n gives a window longer than the unroll."""
    rows = window or n
    batch = sequence(n=rows, cav=False)
    valid = batch["present"].clone()
    valid[:, 3:5, 1] = False
    batch["loss_valid"] = valid
    return batch


def built(batch, n, filled):
    class Model:
        long_history = False
        rollout_mode = "residual"
        physics = None
    cfg = TrainSettings()
    inp = prepare_inputs(batch, Model(), cfg, torch.device("cpu"), n)
    tg = build_targets(inp, batch, cfg, .5, torch.device("cpu"))
    tg.prepared["cutin_reaction_inputs"] = {"present": inp.present} if filled else {}
    return tg


class Recorder:
    """Stub model: records the replay state presence; its carried counter shows recurrent-state resets."""

    def __init__(self):
        self.present, self.counter = [], []
        self.checkpoint_forward = False

    def __call__(self, state, k_steps=0, **kwargs):
        self.present.append(state["present"].clone())
        count = state.get("temporal_residual_mps2", torch.zeros_like(state["veh_kin"][..., 0])) + 1
        self.counter.append(count.clone())
        return {"accel_mps2": torch.zeros_like(count), "temporal_residual_mps2": count}


@pytest.mark.parametrize("window", [None, 13])
def test_filled_replay_keeps_junction_vehicles_in_the_state_and_never_resets_them(window):
    n = 8
    batch = junction_batch(n + 1, window)
    off, on = built(batch, n, False), built(batch, n, True)
    assert torch.equal(replay_present(off), _recorded(off)[2])                  # default = loss_valid rows
    assert bool(replay_present(on)[0, 3, 1]) and not bool(_recorded(on)[2][0, 3, 1])
    assert replay_present(on).shape == (1, n + 1, 5)                           # cut to the unroll
    r_off, r_on = Recorder(), Recorder()
    one_step_replay(r_off, off)
    one_step_replay(r_on, on)
    assert not bool(r_off.present[3][0, 1]) and bool(r_on.present[3][0, 1])
    # loss_valid replay: the transit vehicle restarts its recurrent state after the transit; filled: continuous
    assert float(r_off.counter[5][0, 1]) == 1.0
    assert float(r_on.counter[5][0, 1]) == 6.0
    # supervision rows are unchanged by the fix
    assert torch.equal(_recorded(on)[5], _recorded(off)[5])


def test_filled_replay_is_causal_future_rows_do_not_change_step_t():
    torch.manual_seed(3)
    model = small_model(courtesy=True, randomize=True)
    n = 8
    batch = junction_batch(n + 1)
    acc, _ = one_step_replay(model, built(batch, n, True))
    changed = {k: v.clone() for k, v in batch.items()}
    changed["states"][:, 6:, :, 0] += 7.0
    changed["states"][:, 6:, :, 4] += 3.0
    acc2, _ = one_step_replay(model, built(changed, n, True))
    assert torch.equal(acc[:, :6], acc2[:, :6])


def test_runner_supplies_filled_presence_only_when_switched_on():
    import inspect

    from wms.train import unroll
    source = inspect.getsource(unroll.UnrollRunner.run)
    assert 'getattr(cfg, "recorded_replay_present", "loss_valid") == "filled"' in source
    assert 'targets.prepared["cutin_reaction_inputs"]["present"] = inp.present' in source


def test_courtesy_offset_is_exactly_zero_without_partner_under_filled_replay():
    torch.manual_seed(4)
    model = small_model(courtesy=True, randomize=True)
    n = 8
    _, out = one_step_replay(model, built(junction_batch(n + 1), n, True),
                             collect=("courtesy_accel_mps2", "courtesy_applies"))
    offset, applies = out["courtesy_accel_mps2"], out["courtesy_applies"].bool()
    assert bool((offset[~applies] == 0).all())


def test_trainer_logs_the_learned_centre_only_in_physical_mode():
    import inspect
    source = inspect.getsource(WmSTrainer.train_batch)
    assert 'record["driver_centre"]' in source and '"physical"' in source
