"""R13F: temporal acceleration emission disabled (``model.temporal_accel_off``) + the round-13F config / gate.

Synthetic CPU states only (no data, no SUMO).  Covers:
* default off: round-12 config / flat / spec / settings unchanged (key omitted at default);
* on: forward outputs bit-identical to the BRAKE_DIAG ``diag_zero_temporal`` (zerotemp) override, the
  temporal acceleration contribution exactly 0, every temporal-path tensor without gradient through it;
* the switch is part of the checkpoint (save -> strict reload keeps it);
* configs/c4_wm_v10round13f.toml = round 12 + exactly the R13F/R13A keys, init identical to round 12, the
  absolute weight equal to the documented parity rule (docs/r13a_absolute_weight_rule.json);
* the round-13F verdict switch check.
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
from wms.train.losses.compose import build_loss
from wms.train.round13f_gate import check_r13f_checkpoints
from wms.train.settings import TrainSettings
from wms.train.unroll import UnrollRunner

from test_r10b_courtesy import sequence, state_at_step
from test_r11_courtesy import leaves, model_v

ROOT = Path(__file__).resolve().parents[1]
ROUND12 = ROOT / "configs/c4_wm_v10round12.toml"
ROUND13A = ROOT / "configs/c4_wm_v10round13a.toml"
ROUND13F = ROOT / "configs/c4_wm_v10round13f.toml"
RULE = ROOT / "docs/r13a_absolute_weight_rule.json"
TEMPORAL = ("history_proj.", "k_embedding.", "temporal.")


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def live_temporal(model):
    """A non-zero temporal read-out so the emission is not trivially zero."""
    torch.manual_seed(11)
    with torch.no_grad():
        model.temporal[-1].weight.copy_(torch.randn_like(model.temporal[-1].weight) * 0.5)
        model.temporal[-1].bias.copy_(torch.randn_like(model.temporal[-1].bias) * 0.5)
    return model


def test_round12_defaults_keep_the_switch_off_and_omitted():
    cfg = load_train_config(ROUND12)
    assert cfg.model.temporal_accel_off is False
    flat = cfg.to_legacy_flat()
    assert "temporal_accel_off" not in flat
    assert ModelSpec.from_mapping(flat).temporal_accel_off is False
    assert TrainSettings.from_config(cfg).temporal_accel_off is False
    assert WmSModel(ModelSpec()).temporal_accel_off is False


def test_switch_validation():
    with pytest.raises(ValueError):
        ModelSpec(temporal_accel_off=1)
    with pytest.raises(ValueError):
        load_train_config(ROUND12, ["model.temporal_accel_off=2"])


def test_switch_equals_the_zerotemp_override_bit_for_bit_and_differs_from_on():
    base = live_temporal(model_v(randomize=True))
    off = live_temporal(model_v(randomize=True, temporal_accel_off=True))
    off.load_state_dict(base.state_dict())
    diag = live_temporal(model_v(randomize=True))
    diag.load_state_dict(base.state_dict())
    diag.diag_zero_temporal = True
    batch = sequence(cav=False)
    differs = False
    with torch.no_grad():
        for t in range(3):
            state = state_at_step(batch, t)
            # k_steps = 1: the learned gate is exactly 0 at k = 0 (the residual would not act)
            a, b, c = off(state, k_steps=1), diag(state, k_steps=1), base(state, k_steps=1)
            assert a.keys() == b.keys() == c.keys()
            for key in a:
                if torch.is_tensor(a[key]):
                    assert torch.equal(a[key], b[key]), key
            differs |= not torch.equal(a["accel_mps2"], c["accel_mps2"])
    assert differs, "the temporal emission must matter when on (otherwise the test is vacuous)"


def test_switch_on_temporal_accel_row_gets_no_gradient_rest_trains():
    """The accel read-out row (temporal.2 row 0) is dead when off; the lane rows of the temporal MLP
    (used by this small model's lane path; unused by the hazard_v5 round-12 model) keep training."""
    grads = {}
    for flag in (False, True):
        model = live_temporal(model_v(randomize=True, temporal_accel_off=flag)).train()
        cfg = TrainSettings(truncation_steps=8)
        UnrollRunner(model, cfg, build_loss(cfg), projection=None,
                     device=torch.device("cpu")).run(sequence(cav=False), unroll_steps=8, backward=True)
        grads[flag] = {n: (None if p.grad is None else p.grad.clone()) for n, p in model.named_parameters()}
    on_w, on_b = grads[False]["temporal.2.weight"], grads[False]["temporal.2.bias"]
    assert float(on_w[0].abs().sum()) > 0 and float(on_b[0].abs()) > 0
    off_w, off_b = grads[True]["temporal.2.weight"], grads[True]["temporal.2.bias"]
    assert off_w is None or (float(off_w[0].abs().sum()) == 0.0 and float(off_b[0].abs()) == 0.0)
    rest = [n for n, g in grads[True].items() if not n.startswith(TEMPORAL) and g is not None
            and torch.isfinite(g).all() and float(g.abs().sum()) > 0]
    assert any(n.startswith("residual_net.") for n in rest)


def test_switch_is_part_of_the_checkpoint(tmp_path):
    torch.manual_seed(2)
    model = live_temporal(WmSModel(ModelSpec(temporal_accel_off=True)))
    (tmp_path / "physics.json").write_text(json.dumps(asdict(PhysicsConfig())), encoding="utf-8")
    flat = {k: v for k, v in asdict(model.spec).items() if v is not None}
    save_checkpoint(tmp_path / "wm_s.pt", model, flat)
    loaded = load_checkpoint(tmp_path / "wm_s.pt", strict=True)
    assert loaded.model.spec.temporal_accel_off is True and loaded.model.temporal_accel_off is True
    state = state_at_step(sequence(cav=False), 1)
    with torch.no_grad():
        assert torch.equal(loaded.model(state, k_steps=1)["accel_mps2"],
                           model.eval()(state, k_steps=1)["accel_mps2"])


def _rule_weight():
    rule = json.loads(RULE.read_text(encoding="utf-8"))
    assert rule["w_contrast"] == 9.067 and rule["n_windows"] == 207
    return float(rule["absolute_weight"])


def test_round13f_config_is_round12_plus_exactly_the_r13f_keys():
    a, b = dict(leaves(load_raw(ROUND12))), dict(leaves(load_raw(ROUND13F)))
    changed = {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
    expected = {"name", "runtime.out_dir", "model.temporal_accel_off", "model.follower_courtesy_state_grad",
                "model.follower_courtesy_intent", "loss.follower_courtesy.absolute_weight"}
    assert changed == expected, sorted(changed ^ expected)
    assert a["model.init_from"] == b["model.init_from"]                     # init identical to round 12
    cfg = load_train_config(ROUND13F)
    assert cfg.model.temporal_accel_off is True
    assert cfg.model.follower_courtesy_state_grad is True and cfg.model.follower_courtesy_intent == "pivot"
    assert cfg.loss.follower_courtesy.absolute_weight == _rule_weight() == 4.525
    assert cfg.model.driver_param_init == "centre" and cfg.model.residual_accel_reinit is True
    assert cfg.model.follower_courtesy_init_zero is True
    flat = cfg.to_legacy_flat()
    assert flat["temporal_accel_off"] is True
    settings = TrainSettings.from_config(cfg)
    assert settings.temporal_accel_off is True and settings.follower_courtesy_absolute_weight == 4.525
    spec = ModelSpec.from_mapping(flat)
    assert spec.temporal_accel_off and spec.follower_courtesy_state_grad and spec.follower_courtesy_intent == "pivot"


def test_round13a_config_uses_the_documented_absolute_weight():
    assert load_train_config(ROUND13A).loss.follower_courtesy.absolute_weight == _rule_weight()


def test_r13f_verdict_switch_check(tmp_path):
    (tmp_path / "physics.json").write_text(json.dumps(asdict(PhysicsConfig())), encoding="utf-8")

    def write(**spec_kw):
        torch.manual_seed(3)
        spec = ModelSpec(rollout_mode="driver_params", merge_decision="first_passage", merge_passage_law="hybrid",
                         lane_model="hazard_v5",
                         lane_decision="sample", follower_courtesy=True, follower_courtesy_version=2, **spec_kw)
        model = WmSModel(spec, physics=PhysicsConfig())
        flat = {k: v for k, v in asdict(spec).items() if v is not None}
        flat["follower_courtesy_absolute_weight"] = 4.525
        for name in ("wm_s_stage120.pt", "wm_s_stage300.pt", "wm_s.pt"):
            save_checkpoint(tmp_path / name, model, flat)

    write(temporal_accel_off=True, follower_courtesy_state_grad=True, follower_courtesy_intent="pivot")
    reports = check_r13f_checkpoints(tmp_path, 4.525)
    assert len(reports) == 3 and all(r["temporal_accel_off"] for r in reports)
    with pytest.raises(ValueError, match="absolute_weight"):
        check_r13f_checkpoints(tmp_path, 9.067)
    write(temporal_accel_off=False, follower_courtesy_state_grad=True, follower_courtesy_intent="pivot")
    with pytest.raises(ValueError, match="temporal_accel_off"):
        check_r13f_checkpoints(tmp_path, 4.525)
