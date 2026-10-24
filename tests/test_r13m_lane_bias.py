"""R13M: "keep the mean, remove the noise" -- per-lane constant temporal emission (``model.temporal_accel_lane_bias``).

Synthetic CPU states only.  Default off (no parameter, keys omitted); on: the pre-gate emission is exactly
b[step-start lane] (0 off road), with b = 0 the model equals ``temporal_accel_off`` bit for bit; only b (not the
history MLP's acceleration row) gets gradient; the init switch; checkpoint round trip; configs.
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
from wms.train.settings import TrainSettings
from wms.train.unroll import UnrollRunner

from test_r10b_courtesy import sequence, state_at_step
from test_r11_courtesy import leaves, model_v

ROOT = Path(__file__).resolve().parents[1]
ROUND12 = ROOT / "configs/c4_wm_v10round12.toml"
ROUND13F = ROOT / "configs/c4_wm_v10round13f.toml"
ROUND13M = ROOT / "configs/c4_wm_v10round13m.toml"
INIT_RULE = ROOT / "docs/r13m_lane_bias_init.json"


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def live(model):
    torch.manual_seed(11)
    with torch.no_grad():
        model.temporal[-1].weight.copy_(torch.randn_like(model.temporal[-1].weight) * 0.5)
        model.temporal[-1].bias.copy_(torch.randn_like(model.temporal[-1].bias) * 0.5)
    return model


def test_default_has_no_parameter_and_omits_keys():
    assert WmSModel(ModelSpec()).temporal_lane_bias is None
    assert "temporal_lane_bias" not in WmSModel(ModelSpec()).state_dict()
    flat = load_train_config(ROUND12).to_legacy_flat()
    assert not {"temporal_accel_lane_bias", "temporal_lane_bias_init"} & set(flat)
    with pytest.raises(ValueError):
        ModelSpec(temporal_accel_lane_bias=True, temporal_accel_off=True)
    with pytest.raises(ValueError):
        load_train_config(ROUND12, ["model.temporal_lane_bias_init=[1.0,0.0,0.0]"])   # needs the flag


def test_emission_is_the_lane_constant_and_zero_bias_equals_off():
    off = live(model_v(randomize=True, temporal_accel_off=True))
    lb = live(model_v(randomize=True, temporal_accel_lane_bias=True))
    lb.load_state_dict({**off.state_dict(), "temporal_lane_bias": torch.zeros(3)})
    batch = sequence(cav=False)
    with torch.no_grad():
        for t in range(3):
            state = state_at_step(batch, t)
            a, b = off(state, k_steps=1), lb(state, k_steps=1)
            for key in a:
                if torch.is_tensor(a[key]):
                    assert torch.equal(a[key], b[key]), key
        lb.temporal_lane_bias.copy_(torch.tensor([0.7, -0.3, 1.1]))
        lb.diag_emit_temporal = True
        state = state_at_step(batch, 1)
        out = lb(state, k_steps=1)
        kin = state["veh_kin"]
        on = kin[..., 1:4].sum(-1) > 0
        want = torch.where(on, lb.temporal_lane_bias[kin[..., 1:4].argmax(-1)], torch.zeros(()))
        assert torch.equal(out["temporal_accel_pregate_mps2"], want)
        assert not torch.equal(out["accel_mps2"], off(state, k_steps=1)["accel_mps2"])
    assert "temporal_accel_pregate_mps2" not in off(state_at_step(batch, 1), k_steps=1)


def test_only_the_bias_carries_the_temporal_acceleration_gradient():
    model = live(model_v(randomize=True, temporal_accel_lane_bias=True)).train()
    with torch.no_grad():
        model.temporal_lane_bias.copy_(torch.tensor([0.5, 0.5, 0.5]))
    cfg = TrainSettings(truncation_steps=8)
    UnrollRunner(model, cfg, build_loss(cfg), projection=None,
                 device=torch.device("cpu")).run(sequence(cav=False), unroll_steps=8, backward=True)
    assert model.temporal_lane_bias.grad is not None and float(model.temporal_lane_bias.grad.abs().sum()) > 0
    g = model.temporal[-1].weight.grad
    assert g is None or float(g[0].abs().sum()) == 0.0


def test_init_switch_sets_the_values():
    from wms.train.trainer import WmSTrainer

    class Stub:
        pass
    stub = Stub()
    stub.model = model_v(randomize=True, temporal_accel_lane_bias=True)
    stub.settings = TrainSettings(temporal_accel_lane_bias=True, temporal_lane_bias_init=(2.5, 0.25, -0.5))
    report = WmSTrainer._r12_reinit(stub)
    assert report["temporal_lane_bias_init"] == [2.5, 0.25, -0.5]
    assert torch.equal(stub.model.temporal_lane_bias.detach(), torch.tensor([2.5, 0.25, -0.5]))


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(2)
    model = WmSModel(ModelSpec(temporal_accel_lane_bias=True))
    with torch.no_grad():
        model.temporal_lane_bias.copy_(torch.tensor([1.0, 2.0, 3.0]))
    (tmp_path / "physics.json").write_text(json.dumps(asdict(PhysicsConfig())), encoding="utf-8")
    save_checkpoint(tmp_path / "wm_s.pt", model, {k: v for k, v in asdict(model.spec).items() if v is not None})
    loaded = load_checkpoint(tmp_path / "wm_s.pt", strict=True)
    assert loaded.model.temporal_accel_lane_bias is True
    assert torch.equal(loaded.model.temporal_lane_bias.detach(), torch.tensor([1.0, 2.0, 3.0]))


def test_round13m_config_is_round13f_with_the_mean_variant():
    a, b = dict(leaves(load_raw(ROUND13F))), dict(leaves(load_raw(ROUND13M)))
    changed = {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
    expected = {"name", "runtime.out_dir", "model.temporal_accel_off", "model.temporal_accel_lane_bias",
                "model.temporal_lane_bias_init"}
    assert changed == expected, sorted(changed ^ expected)
    assert a["model.init_from"] == b["model.init_from"] == dict(leaves(load_raw(ROUND12)))["model.init_from"]
    cfg = load_train_config(ROUND13M)
    rule = json.loads(INIT_RULE.read_text(encoding="utf-8"))
    assert cfg.model.temporal_accel_lane_bias is True and cfg.model.temporal_accel_off is False
    assert list(cfg.model.temporal_lane_bias_init) == rule["temporal_lane_bias_init"]
    settings = TrainSettings.from_config(cfg)
    assert tuple(settings.temporal_lane_bias_init) == tuple(rule["temporal_lane_bias_init"])
    assert ModelSpec.from_mapping(cfg.to_legacy_flat()).temporal_accel_lane_bias is True
