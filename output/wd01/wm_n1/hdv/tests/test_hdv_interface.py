"""P2 interface: kinematics-only contract, shapes, ablation and zero-residual."""
from __future__ import annotations

import pytest
import torch

from hdv import KIN_F, HdvConfig, HdvResidualNet, input_schema
from synth_hdv import random_kin


def test_kin_fields_exclude_cav_and_actions():
    from hdv.config import KIN_FIELDS
    assert "is_cav" not in KIN_FIELDS
    assert not any(f.startswith("action_") for f in KIN_FIELDS)
    assert KIN_F == len(KIN_FIELDS)


def test_config_ablation_switch():
    assert HdvConfig().in_features == KIN_F
    assert HdvConfig(use_cav_flag=True).in_features == KIN_F + 1


def test_forward_shapes_and_finite():
    cfg = HdvConfig()
    net = HdvResidualNet(cfg)
    net.eval()
    kin, mask = random_kin(batch=3, slots=7)
    out = net(kin, mask)
    assert out["accel_delta_mps2"].shape == (3, 7)
    assert out["lane_logit_delta"].shape == (3, 7, 3)
    assert out["attn"].shape == (3, cfg.n_heads, 7, 7)
    assert torch.isfinite(out["accel_delta_mps2"]).all()
    assert torch.isfinite(out["lane_logit_delta"]).all()
    # range is bounded by the registered scale.
    assert out["accel_delta_mps2"].abs().max() <= cfg.accel_delta_scale_mps2


def test_forward_rejects_cav_column_in_headline():
    net = HdvResidualNet()
    with pytest.raises(ValueError):
        net.assert_no_cav_flag(torch.zeros(2, 4, KIN_F + 1))
    with pytest.raises(ValueError):
        net(torch.zeros(2, 4, KIN_F + 1), torch.ones(2, 4, dtype=torch.bool))


def test_zero_residual_switch_is_exact_noop():
    net = HdvResidualNet(HdvConfig(zero_residual=True))
    net.eval()
    kin, mask = random_kin(batch=2, slots=6)
    out = net(kin, mask)
    assert torch.count_nonzero(out["accel_delta_mps2"]) == 0
    assert torch.count_nonzero(out["lane_logit_delta"]) == 0
    assert torch.count_nonzero(out["accel_log_sigma"]) == 0


def test_input_schema_headline_has_no_forbidden_channel():
    s = input_schema()
    assert s["optional_inputs"] == ()
    assert s["allowed_inputs"] == ("kin", "kin_mask")
    for name in ("action", "action_code", "strategy", "strategy_name",
                 "is_cav", "cav_flag"):
        assert name not in s["allowed_inputs"]
        assert name not in s["optional_inputs"]
        assert name not in s["kin_fields"]
        assert name in s["forbidden_inputs"]
    assert s["use_cav_flag"] is False


def test_parameter_count_reported(capsys):
    full = HdvResidualNet(HdvConfig())
    flat = HdvResidualNet(HdvConfig(n_layers=1))
    n_full = int(sum(p.numel() for p in full.parameters()))
    n_flat = int(sum(p.numel() for p in flat.parameters()))
    with capsys.disabled():
        print("\n[hdv] parameter_count full=%d one-layer=%d" % (n_full, n_flat))
    assert 0 < n_flat < n_full
