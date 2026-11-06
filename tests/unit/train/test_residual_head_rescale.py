"""R5MERGE: exact-identity-at-load residual-head output rescale."""
from __future__ import annotations

import pytest
import torch

from wms.config.legacy_flat import FLAT_TO_PATH, OMIT_WHEN_DEFAULT
from wms.config.train import ModelConfig, TrainConfig
from wms.model.residual_net import HdvConfig, HdvResidualNet
from wms.model.spec import ModelSpec
from wms.train.settings import TrainSettings

KIN = dict(d_model=16, n_heads=2, n_layers=1, ffn=32)


def _inputs(seed: int = 3):
    generator = torch.Generator().manual_seed(seed)
    kin = torch.randn(2, 5, 7, generator=generator)
    kin[..., 1:4] = 0.0
    kin[..., 2] = 1.0
    kin[..., 4] = kin[..., 4].abs()
    present = torch.ones(2, 5, dtype=torch.bool)
    return kin, present


def test_off_path_creates_no_parameter_and_keeps_the_head():
    net = HdvResidualNet(HdvConfig(**KIN))
    assert net.residual_head_log_scale is None
    assert "residual_head_log_scale" not in net.state_dict()


def test_rescale_is_exactly_identity_at_load():
    torch.manual_seed(11)
    base = HdvResidualNet(HdvConfig(**KIN)).eval()
    rescale = HdvResidualNet(HdvConfig(**KIN, residual_head_rescale=True)).eval()
    assert rescale.residual_head_log_scale is not None
    assert float(rescale.residual_head_log_scale.detach()) == 0.0
    result = rescale.load_state_dict(base.state_dict(), strict=False)
    assert result.missing_keys == ["residual_head_log_scale"]
    kin, present = _inputs()
    with torch.no_grad():
        expected = base(kin, present)
        actual = rescale(kin, present)
    for key in ("accel_delta_mps2", "accel_log_sigma", "lane_logit_delta"):
        assert torch.equal(actual[key], expected[key]), key


def test_rescale_parameter_receives_a_gradient():
    torch.manual_seed(5)
    net = HdvResidualNet(HdvConfig(**KIN, residual_head_rescale=True)).train()
    kin, present = _inputs()
    out = net(kin, present)
    out["accel_delta_mps2"].sum().backward()
    grad = net.residual_head_log_scale.grad
    assert grad is not None and torch.isfinite(grad)
    assert float(grad.abs()) > 0.0
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0.0
               for p in net.accel.parameters())


def test_spec_switch_builds_the_parameter_in_the_model():
    from wms.model.factory import build_model

    off = build_model(ModelSpec())
    on = build_model(ModelSpec(residual_head_rescale=True))
    assert off.residual_net.residual_head_log_scale is None
    assert on.residual_net.residual_head_log_scale is not None
    assert float(on.residual_net.residual_head_log_scale.detach()) == 0.0


def test_rescale_switch_is_wired_through_the_flat_config():
    flat = TrainConfig().to_legacy_flat()
    assert "residual_head_rescale" not in flat
    assert "residual_head_rescale" in OMIT_WHEN_DEFAULT
    assert FLAT_TO_PATH["residual_head_rescale"] == "model.residual_head_rescale"
    cfg = TrainConfig(model=ModelConfig(residual_head_rescale=True))
    flat_on = cfg.to_legacy_flat()
    assert flat_on["residual_head_rescale"] is True
    assert TrainConfig.from_legacy_flat(flat_on).model.residual_head_rescale is True
    spec = ModelSpec.from_mapping(flat_on)
    assert spec.residual_head_rescale is True
    settings = TrainSettings.from_config(cfg)
    assert settings.residual_head_rescale is True
    with pytest.raises(Exception):
        ModelSpec(residual_head_rescale="yes")
