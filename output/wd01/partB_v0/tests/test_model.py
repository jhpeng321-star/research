"""Model contract tests: shapes, parameter budget, conditioning, heads."""
from __future__ import annotations

import json

import pytest
import torch

from partB_v0.model import (POLICIES, V0ValueConfig, V0ValueNet,
                            count_parameters, policy_id, twohot_encode)


def _batch(b=3):
    cfg = V0ValueConfig()          # widths are config-driven (v0.4: 20/11)
    return dict(
        lane_bins=torch.randn(b, 3, 320, 3), lane_mask=torch.ones(b, 3, 320),
        lane_geom=torch.ones(b, 3, 320),
        cav_tokens=torch.randn(b, cfg.max_cavs, cfg.cav_raw),
        cav_mask=torch.ones(b, cfg.max_cavs),
        wall_tokens=torch.zeros(b, cfg.max_walls, cfg.wall_raw),
        wall_mask=torch.zeros(b, cfg.max_walls),
        demand=torch.randn(b, 21), context=torch.randn(b, 9),
        tau=torch.linspace(30.0, 570.0, b),
        policy=torch.arange(b) % 2)


def test_forward_shapes_and_default_heads():
    out = V0ValueNet()(_batch())
    assert set(out) == {"v_delay", "v_exp", "v_brake", "p_coll"}
    assert out["v_delay"].shape == (3, 255)
    assert out["p_coll"].shape == (3,)


def test_optional_heads_disabled_by_default_and_available():
    # P_closing_pair / future_min_ttc_q05 have NO V0 labels: off by default
    assert "p_closing_pair" not in V0ValueNet()(_batch())
    cfg = V0ValueConfig(enable_p_closing_pair=True,
                        enable_future_min_ttc_q05=True)
    out = V0ValueNet(cfg)(_batch())
    assert out["p_closing_pair"].shape == (3,)
    assert out["future_min_ttc_q05"].shape == (3,)


def test_parameter_count_in_target_band():
    n = count_parameters(V0ValueNet())
    print(f"params per member: {n} ({n / 1e6:.2f} M)")
    assert 1e6 <= n <= 3e6


def test_tau_conditioning_changes_value_head():
    net = V0ValueNet().eval()
    b = _batch()
    b["tau"] = torch.full((3,), 30.0)
    with torch.no_grad():
        a = net(b)["v_delay"]
        b["tau"] = torch.full((3,), 570.0)
        c = net(b)["v_delay"]
    assert (a - c).abs().max() > 1e-4


def test_policy_conditioning_changes_value_head():
    net = V0ValueNet().eval()
    b = _batch()
    b["policy"] = torch.zeros(3, dtype=torch.long)
    with torch.no_grad():
        a = net(b)["v_delay"]
        b["policy"] = torch.ones(3, dtype=torch.long)
        c = net(b)["v_delay"]
    assert (a - c).abs().max() > 1e-4


def test_pre_fix_flag_changes_value_head():
    net = V0ValueNet().eval()
    b = _batch()
    with torch.no_grad():
        b["context"] = b["context"].clone()
        b["context"][:, -1] = 0.0
        a = net(b)["v_delay"]
        b["context"][:, -1] = 1.0
        c = net(b)["v_delay"]
    assert (a - c).abs().max() > 1e-4


def test_policy_table_extensible():
    # v0.3.2 fix #3: the campaign '__native' arms are random per-CAV
    # switching schedules -- the canonical policy is switching_mix
    assert POLICIES == ("switching_mix", "damping_fleet")
    assert policy_id("switching_mix") == 0 \
        and policy_id("damping_fleet") == 1
    with pytest.raises(ValueError):
        policy_id("wall_member")
    # reserved slots: embedding has 4 rows, clamp keeps unknown ids safe
    net = V0ValueNet()
    assert net.policy_emb.num_embeddings == 4


def test_config_json_round_trip():
    cfg = V0ValueConfig(enable_p_closing_pair=True, d_model=64)
    cfg2 = V0ValueConfig.from_json(json.loads(json.dumps(cfg.to_json())))
    assert cfg == cfg2
    # unknown keys ignored (forward compatibility)
    assert V0ValueConfig.from_json({"d_model": 96, "future": "x"}) \
        .d_model == 96


def test_checkpoint_is_weights_only_loadable(tmp_path):
    net = V0ValueNet()
    p = tmp_path / "member_0.pt"
    torch.save(net.state_dict(), p)
    sd = torch.load(p, map_location="cpu", weights_only=True)
    net2 = V0ValueNet()
    net2.load_state_dict(sd)
    b = _batch(2)
    with torch.no_grad():
        assert torch.allclose(net(b)["v_delay"], net2(b)["v_delay"])


def test_gradient_flows_through_twohot_loss():
    net = V0ValueNet()
    out = net(_batch())
    target = twohot_encode(torch.tensor([100.0, 5000.0, 900.0]))
    loss = torch.nn.functional.cross_entropy(out["v_delay"], target)
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads) > 10
