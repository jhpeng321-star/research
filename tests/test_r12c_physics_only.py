"""R12C: physics-only core calibration (``model.residual_accel_off`` + ``model.follower_courtesy_off``).

Synthetic CPU states only (no data, no SUMO).  Covers:
* default off: round-12 config / flat / spec / settings unchanged (keys omitted at default), bit-identical
  to 1878f842 with the switches off (forward + short-rollout equality is checked against a clean 1878f842
  worktree by the launcher; here: identical key sets and lane/temporal tensors with the flags on);
* on: the BRAKE_DIAG 'physics only (residual and courtesy off)' arm as a training switch -- the HDV
  acceleration equals the physics core exactly, ``raw_residual_mps2`` and ``courtesy_accel_mps2`` are
  exact zeros, the lane read-outs are untouched, the emissions matter for the flags-off model;
* gradients: the driver-encoder head still trains through the physics core; the temporal acceleration
  row is dead; the switch is part of the checkpoint (save -> strict reload keeps it);
* configs/c4_wm_v10round12c.toml = round 12 + exactly the R12C keys (emissions off, restricted trainable
  set, single 120 s / 30-step diagnostic schedule, val 4 windows, prior left to the gate's off override);
* the round-12C gate's trainable-tensor-list check on the optimizer groups line.
"""
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from wms.config.load import load_raw, load_train_config
from wms.model.checkpoint import load_checkpoint, save_checkpoint
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.physics.config import PhysicsConfig
from wms.train.round12c_gate import EXPECTED_TRAINABLE_TENSORS, optimizer_report_errors
from wms.train.settings import TrainSettings
from wms.train.trainer import build_optimizer

from test_r10b_courtesy import sequence, state_at_step
from test_r11_courtesy import leaves, model_v

ROOT = Path(__file__).resolve().parents[1]
ROUND12 = ROOT / "configs/c4_wm_v10round12.toml"
ROUND12C = ROOT / "configs/c4_wm_v10round12c.toml"
ALL_OFF = dict(temporal_accel_off=True, residual_accel_off=True, follower_courtesy_off=True)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def live_emissions(model):
    """Non-zero temporal / residual / courtesy read-outs so the emissions are not trivially zero."""
    torch.manual_seed(11)
    with torch.no_grad():
        for module in filter(None, (getattr(model, "temporal", None) and model.temporal[-1],
                                    getattr(model.residual_net, "accel", None) and model.residual_net.accel[-1])):
            module.weight.copy_(torch.randn_like(module.weight) * 0.5)
            module.bias.copy_(torch.randn_like(module.bias) * 0.5)
        if model.follower_courtesy is not None:
            for p in model.follower_courtesy.parameters():
                p.copy_(torch.randn_like(p) * 0.1)
    return model


def test_round12_defaults_keep_the_switches_off_and_omitted():
    cfg = load_train_config(ROUND12)
    assert cfg.model.temporal_accel_off is False
    assert cfg.model.residual_accel_off is False
    assert cfg.model.follower_courtesy_off is False
    flat = cfg.to_legacy_flat()
    for key in ("temporal_accel_off", "residual_accel_off", "follower_courtesy_off"):
        assert key not in flat
    spec = ModelSpec.from_mapping(flat)
    assert spec.temporal_accel_off is False and spec.residual_accel_off is False
    assert spec.follower_courtesy_off is False
    settings = TrainSettings.from_config(cfg)
    assert settings.residual_accel_off is False and settings.follower_courtesy_off is False
    model = WmSModel(ModelSpec())
    assert model.residual_accel_off is False and model.follower_courtesy_off is False


def test_switch_validation():
    with pytest.raises(ValueError):
        ModelSpec(residual_accel_off=1)
    with pytest.raises(ValueError):
        ModelSpec(follower_courtesy_off=1)
    with pytest.raises(ValueError):
        load_train_config(ROUND12, ["model.residual_accel_off=2"])
    with pytest.raises(ValueError):
        load_train_config(ROUND12, ["model.follower_courtesy_off=no"])


def test_physics_only_arm_zeroes_every_learned_emission_and_keeps_the_lane_path():
    base = live_emissions(model_v(randomize=True))
    on = live_emissions(model_v(randomize=True, **ALL_OFF))
    on.load_state_dict(base.state_dict())
    batch = sequence(cav=False)
    differs = False
    with torch.no_grad():
        for t in range(3):
            state = state_at_step(batch, t)
            a, c = on(state, k_steps=2), base(state, k_steps=2)
            assert a.keys() == c.keys()
            assert torch.equal(a["lane_logits"], c["lane_logits"])       # lane read-outs untouched
            assert bool((a["raw_residual_mps2"] == 0).all())             # residual + temporal emission = 0
            assert bool((a["courtesy_accel_mps2"] == 0).all())          # courtesy offset = 0
            hdv = a["hdv_mask"].bool()
            physics = a["physics_accel_mps2"]
            lo, hi = float(on.physics.accel_min_mps2), float(on.physics.accel_max_mps2)
            assert bool((physics[hdv] >= lo).all()) and bool((physics[hdv] <= hi).all())
            assert torch.equal(a["accel_mps2"][hdv], physics[hdv])       # HDV accel = physics core exactly
            differs |= not torch.equal(a["accel_mps2"][hdv], c["accel_mps2"][hdv])
    assert differs, "the emissions must matter when on (otherwise the test is vacuous)"


def test_flags_off_bit_identity_forward_keys_and_values():
    """With the switches off the model output must be the round-12 (1878f842) computation itself."""
    base = live_emissions(model_v(randomize=True))
    off = live_emissions(model_v(randomize=True, temporal_accel_off=False, residual_accel_off=False,
                                 follower_courtesy_off=False))
    off.load_state_dict(base.state_dict())
    batch = sequence(cav=False)
    with torch.no_grad():
        for t in range(3):
            state = state_at_step(batch, t)
            a, b = off(state, k_steps=2), base(state, k_steps=2)
            assert a.keys() == b.keys()
            for key in a:
                if torch.is_tensor(a[key]):
                    assert torch.equal(a[key], b[key]), key


def test_switch_on_driver_head_trains_through_physics_temporal_row_dead():
    from wms.train.losses.compose import build_loss
    from wms.train.unroll import UnrollRunner

    grads = {}
    for flag, kw in ((False, {}), (True, ALL_OFF)):
        model = live_emissions(model_v(randomize=True, driver_param_mode="physical",
                                       rollout_mode="driver_params", **kw)).train()
        cfg = TrainSettings(truncation_steps=8)
        UnrollRunner(model, cfg, build_loss(cfg), projection=None,
                     device=torch.device("cpu")).run(sequence(cav=False), unroll_steps=8, backward=True)
        grads[flag] = {n: (None if p.grad is None else p.grad.clone()) for n, p in model.named_parameters()}
    head = grads[True]["driver_encoder.head.0.weight"]
    assert head is not None and torch.isfinite(head).all() and float(head.abs().sum()) > 0
    temporal_w = grads[True]["temporal.2.weight"] if "temporal.2.weight" in grads[True] else None
    if temporal_w is not None and grads[False]["temporal.2.weight"] is not None:
        assert float(grads[False]["temporal.2.weight"][0].abs().sum()) > 0
        assert float(temporal_w[0].abs().sum()) == 0.0


def test_switches_are_part_of_the_checkpoint(tmp_path):
    torch.manual_seed(2)
    model = live_emissions(WmSModel(ModelSpec(follower_courtesy=True, follower_courtesy_version=2,
                                              **ALL_OFF)))
    (tmp_path / "physics.json").write_text(json.dumps(asdict(PhysicsConfig())), encoding="utf-8")
    flat = {k: v for k, v in asdict(model.spec).items() if v is not None}
    save_checkpoint(tmp_path / "wm_s.pt", model, flat)
    loaded = load_checkpoint(tmp_path / "wm_s.pt", strict=True)
    for flag in ("temporal_accel_off", "residual_accel_off", "follower_courtesy_off"):
        assert getattr(loaded.model.spec, flag) is True and getattr(loaded.model, flag) is True
    state = state_at_step(sequence(cav=False), 1)
    with torch.no_grad():
        assert torch.equal(loaded.model(state, k_steps=1)["accel_mps2"],
                           model.eval()(state, k_steps=1)["accel_mps2"])


def test_round12c_config_is_round12_plus_exactly_the_r12c_keys():
    a, b = dict(leaves(load_raw(ROUND12))), dict(leaves(load_raw(ROUND12C)))
    changed = {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
    expected = {"name", "runtime.out_dir", "model.temporal_accel_off", "model.residual_accel_off",
                "model.follower_courtesy_off", "curriculum.unroll_s", "curriculum.stage_steps",
                "curriculum.truncation_steps_per_stage", "curriculum.max_hours", "optim.lr_per_stage",
                "optim.batch_size_per_stage", "optim.freeze_prefixes",
                "optim.lr_scale.residual_net", "optim.lr_scale.temporal",
                "optim.lr_scale.follower_courtesy", "selection.val_max_windows"}
    assert changed == expected, sorted(changed ^ expected)
    assert a["model.init_from"] == b["model.init_from"]                     # init identical to round 12
    cfg = load_train_config(ROUND12C)
    assert cfg.model.temporal_accel_off and cfg.model.residual_accel_off and cfg.model.follower_courtesy_off
    assert cfg.model.driver_param_init == "centre" and cfg.model.residual_accel_reinit is True
    assert cfg.model.follower_courtesy_init_zero is True
    assert list(cfg.curriculum.unroll_s) == [120.0] and list(cfg.curriculum.stage_steps) == [30]
    assert list(cfg.curriculum.truncation_steps_per_stage) == [120]
    assert float(cfg.curriculum.max_hours) == 2.5
    assert int(cfg.selection.val_max_windows) == 4
    assert list(cfg.optim.batch_size_per_stage) == [4] and list(cfg.optim.lr_per_stage) == [5.0e-5]
    freeze = list(cfg.optim.freeze_prefixes)
    for prefix in ("residual_net", "temporal", "follower_courtesy", "merge_pivot"):
        assert prefix in freeze
    assert dict(cfg.optim.lr_scale) == {"driver_encoder.head": 10.0, "driver_encoder.centre_logits": 1000.0}
    settings = TrainSettings.from_config(cfg)
    assert settings.residual_accel_off and settings.follower_courtesy_off and settings.temporal_accel_off
    assert list(settings.unroll_curriculum_s) == [120.0]
    flat = cfg.to_legacy_flat()
    assert flat["residual_accel_off"] is True and flat["follower_courtesy_off"] is True
    spec = ModelSpec.from_mapping(flat)
    assert spec.residual_accel_off and spec.follower_courtesy_off and spec.temporal_accel_off


def test_round12c_trainable_set_is_exactly_the_driver_encoder_and_physics_centre():
    from wms.cli.assemble import build_physics

    cfg = load_train_config(ROUND12C)
    model = WmSModel(ModelSpec.from_mapping(cfg.to_legacy_flat()), physics=build_physics(cfg))
    optimizer, report = build_optimizer(model, lr=1.0e-4, weight_decay=0.0,
                                        freeze_prefixes=tuple(cfg.optim.freeze_prefixes),
                                        lr_scale=dict(cfg.optim.lr_scale))
    assert sorted(report["trainable_tensor_names"]) == sorted(EXPECTED_TRAINABLE_TENSORS)
    assert report["trainable_tensors"] == 5 and not report["unmatched_freeze_prefixes"]
    assert optimizer.param_groups[0]["params"]


def test_gate_optimizer_report_check(tmp_path):
    good = tmp_path / "good.log"
    report = {"trainable_tensor_names": list(EXPECTED_TRAINABLE_TENSORS),
              "freeze_prefixes": ["residual_net", "temporal", "follower_courtesy"], "groups": []}
    good.write_text("WM-S optimizer groups: " + json.dumps(report) + "\n", encoding="utf-8")
    metrics = {}
    assert optimizer_report_errors(good, report["freeze_prefixes"], metrics) == []
    assert metrics["trainable_tensor_names"] == sorted(EXPECTED_TRAINABLE_TENSORS)
    bad = tmp_path / "bad.log"
    bad.write_text("WM-S optimizer groups: " + json.dumps({**report, "trainable_tensor_names":
                                                          ["residual_net.accel.2.weight"]}) + "\n",
                   encoding="utf-8")
    assert optimizer_report_errors(bad, report["freeze_prefixes"], {})
    assert optimizer_report_errors(tmp_path / "none.log", (), {})
