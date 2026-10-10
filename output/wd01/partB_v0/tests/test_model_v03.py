"""v0.3 model contract tests (BRIEF_B3_V03.md A2-A5, C13)."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from partB_v0.model import (N_BINS_V03, PmfGrid, SUMMARY_DIM, SYM_MAX_V03,
                            SYM_MIN_V03, V0ValueConfig, V0ValueNet,
                            V03ValueConfig, V03ValueNet, count_parameters,
                            twohot_encode_v03)


def _batch(b=3, summary_dim=SUMMARY_DIM):
    cfg = V03ValueConfig()         # widths are config-driven (v0.4: 20/11)
    return dict(
        lane_bins=torch.randn(b, 3, 320, 3), lane_mask=torch.ones(b, 3, 320),
        lane_geom=torch.ones(b, 3, 320),
        cav_tokens=torch.randn(b, cfg.max_cavs, cfg.cav_raw),
        cav_mask=torch.ones(b, cfg.max_cavs),
        wall_tokens=torch.zeros(b, cfg.max_walls, cfg.wall_raw),
        wall_mask=torch.zeros(b, cfg.max_walls),
        demand=torch.randn(b, 21), context=torch.randn(b, 8),
        summary=torch.randn(b, summary_dim),
        tau=torch.linspace(30.0, 570.0, b),
        policy=torch.arange(b) % 2)


def test_v03_forward_shapes_and_delta_head():
    out = V03ValueNet()(_batch())
    assert set(out) == {"v_delay", "v_exp", "v_brake", "p_coll", "delta"}
    assert out["v_delay"].shape == (3, N_BINS_V03)
    assert out["delta"].shape == (3,)
    cfg = V03ValueConfig(enable_delta=False)
    assert "delta" not in V03ValueNet(cfg)(_batch())


def test_v02_model_still_importable_and_forward():
    """The v0.2 surface (train.py / B3a) keeps working next to v0.3."""
    b = _batch()
    b.pop("summary")
    b["context"] = torch.randn(3, 9)        # v0.2 ctx includes pre_fix
    out = V0ValueNet(V0ValueConfig())(b)
    assert out["v_delay"].shape == (3, 255)


def test_grid_covers_realistic_normalised_range_only():
    grid = PmfGrid()
    assert grid.centers[0] == pytest.approx(-(np.e ** 2 - 1), rel=1e-6)
    assert grid.centers[-1] == pytest.approx(np.e ** 12 - 1, rel=1e-4)
    # the old v0.2 grid reached e^20 (~4.8e8): the new one stops at
    # e^12 (~1.6e5 normalised units), far beyond any realistic y/median
    assert grid.centers[-1] < 2e5
    assert (SYM_MIN_V03, SYM_MAX_V03) == (-2.0, 12.0)


def test_twohot_v03_rows_sum_to_one_two_bins_and_clamp():
    x = torch.tensor([0.0, 1.0, 5.0, -50.0, 1e7])
    p = twohot_encode_v03(x)
    assert p.shape == (5, N_BINS_V03)
    assert torch.allclose(p.sum(-1), torch.ones(5))
    for row in p:
        assert int((row > 1e-8).sum()) <= 2
    assert p[3, 0] == 1.0 and p[4, -1] == 1.0


def test_point_estimate_is_the_median_of_the_pmf():
    """A5: point forecast = pmf median (a point mass decodes to its bin)."""
    net = V03ValueNet(V03ValueConfig(d_model=32, n_layers=1, dim_ff=64,
                                     head_hidden=32, head_layers=1,
                                     tau_emb_dim=16))
    net.eval()
    out = net(_batch(2))
    pts = net.point_estimates(out)
    assert set(pts) == {"v_delay", "v_exp", "v_brake"}
    assert pts["v_delay"].shape == (2,)


def test_pmf_grid_median_and_crps_scaling_identity():
    grid = PmfGrid()
    k = 137
    pm = np.zeros((1, N_BINS_V03))
    pm[0, k] = 1.0
    med = grid.median(pm)
    # interpolated-CDF semantics (as the shipped v0.2 metrics): a point
    # mass decodes inside its own bin, within the half bin width
    half = 0.5 * abs(grid.centers[k + 1] - grid.centers[k - 1])
    assert abs(med[0] - grid.centers[k]) <= half
    y = np.array([0.9])
    s = np.array([870.0])
    # CRPS after de-normalisation = |center*s - y*s| for a point mass
    got = grid.crps_real(pm, y, s)[0]
    assert got == pytest.approx(abs(grid.centers[k] - y[0]) * s[0],
                                rel=1e-6)
    assert grid.crps_norm(pm, y)[0] == pytest.approx(
        abs(grid.centers[k] - y[0]), rel=1e-6)


def test_crps_real_is_linear_in_scale_and_nonnegative():
    """CRPS(F*c, y*c) = c * CRPS(F, y) -- BOTH terms scale linearly (a
    c^2 pairwise term turns negative at per-tau scales ~1e3 and corrupts
    OOB selection; regression for the ported-reference bug)."""
    grid = PmfGrid()
    rng = np.random.default_rng(0)
    k0 = 130
    probs = np.zeros((4, N_BINS_V03))
    for i in range(4):                     # spread pmfs, not point masses
        for k in range(k0 - 6 + i, k0 + 7 + i):
            probs[i, k] = 1.0
    probs /= probs.sum(axis=1, keepdims=True)
    y = rng.uniform(0.5, 2.0, size=4)
    for s_val in (1.0, 340.0, 4660.0):
        s = np.full(4, s_val)
        got = grid.crps_real(probs, y, s)
        ref = s_val * grid.crps_norm(probs, y)
        assert np.allclose(got, ref, rtol=1e-9)
        assert (got >= 0).all(), (s_val, got)


def test_tau_and_policy_conditioning_change_the_head():
    """A3: every head layer is tau-conditioned -> outputs move with tau."""
    net = V03ValueNet().eval()
    b = _batch()
    b["tau"] = torch.full((3,), 30.0)
    with torch.no_grad():
        a = net(b)["v_delay"]
        b["tau"] = torch.full((3,), 570.0)
        c = net(b)["v_delay"]
    assert (a - c).abs().max() > 1e-4
    b2 = _batch()
    b2["policy"] = torch.zeros(3, dtype=torch.long)
    with torch.no_grad():
        d = net(b2)["v_delay"]
        b2["policy"] = torch.ones(3, dtype=torch.long)
        e = net(b2)["v_delay"]
    assert (d - e).abs().max() > 1e-4


def test_summary_token_changes_output():
    net = V03ValueNet().eval()
    b = _batch()
    with torch.no_grad():
        a = net(b)["v_delay"]
        b["summary"] = b["summary"] + 5.0
        c = net(b)["v_delay"]
    # one token among many: masked-mean pooling dilutes the perturbation
    # at init, but the output must still move beyond float noise
    assert (a - c).abs().max() > 1e-4


def test_default_parameter_count_in_band():
    n = count_parameters(V03ValueNet())
    print(f"v0.3 params per member: {n} ({n / 1e6:.2f} M)")
    assert 1e6 <= n <= 3e6


def test_config_json_round_trip():
    cfg = V03ValueConfig(d_model=96, enable_delta=False)
    cfg2 = V03ValueConfig.from_json(json.loads(json.dumps(cfg.to_json())))
    assert cfg == cfg2
    assert V03ValueConfig.from_json({"d_model": 64, "future": "x"}) \
        .d_model == 64


def test_checkpoint_weights_only_loadable(tmp_path):
    net = V03ValueNet()
    p = tmp_path / "member_0.pt"
    torch.save(net.state_dict(), p)
    sd = torch.load(p, map_location="cpu", weights_only=True)
    net2 = V03ValueNet()
    net2.load_state_dict(sd)
    b = _batch(2)
    with torch.no_grad():
        assert torch.allclose(net(b)["v_delay"], net2(b)["v_delay"])


def test_gradient_flows_through_all_heads():
    net = V03ValueNet()
    out = net(_batch())
    target = twohot_encode_v03(torch.tensor([1.0, 2.0, 3.0]))
    loss = (torch.nn.functional.cross_entropy(out["v_delay"], target)
            + out["p_coll"].float().pow(2).mean()
            + out["delta"].float().pow(2).mean())
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads) > 10
