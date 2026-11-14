"""Encoder/head invariants: masking, permutation, shapes, parameter count."""
from __future__ import annotations

import torch

from model import constants as C
from model.net import WmN1Config, WmN1Net, parameter_count
from synth_model import make_batch, small_cfg


def test_output_shapes():
    net = WmN1Net(small_cfg()).eval()
    b = make_batch(B=2, K=8)
    with torch.no_grad():
        out = net(**b)
    B, K = 2, 8
    assert out["slot_hidden"].shape == (B, K, 48)
    assert out["step"]["accel_mu"].shape == (B, K, C.N_MIX)
    assert out["step"]["lane_logits"].shape == (B, K, C.N_LANE_CLASSES)
    assert out["arrival"]["count_mu"].shape == (B, C.N_ENTRANCES)
    assert out["outcome_quantiles"].shape == (
        B, C.N_TAU, C.N_OUTCOME_TARGETS, C.N_QUANTILES)
    assert out["collision_logits"].shape == (B, C.N_TAU)


def test_default_member_parameter_count_in_target_band():
    n = parameter_count(WmN1Net(WmN1Config()))
    print(f"default member parameters = {n}")
    assert 2_000_000 <= n <= 5_000_000, n


def test_masking_invariance_to_padded_slots():
    net = WmN1Net(small_cfg()).eval()
    b = make_batch(B=2, K=8)
    with torch.no_grad():
        base = net(**b)
        # append 4 padded slots with random junk features, mask = 0
        pad = 4
        gen = torch.Generator().manual_seed(7)
        veh = torch.cat([b["veh_step"],
                         torch.randn(2, pad, C.VEH_F, generator=gen)], dim=1)
        mask = torch.cat([b["veh_mask"],
                          torch.zeros(2, pad, dtype=torch.bool)], dim=1)
        plan = torch.cat([b["plan"],
                          torch.randn(2, pad, C.N_SEGMENTS, C.PLAN_F,
                                      generator=gen)], dim=1)
        out = net(veh_step=veh, veh_mask=mask, plan=plan, wall=b["wall"],
                  wall_mask=b["wall_mask"], counter=b["counter"])
    for key in ("slot_hidden",):
        assert torch.allclose(base[key], out[key][:, :8], atol=1e-5)
    assert torch.allclose(base["global_token"], out["global_token"], atol=1e-5)
    assert torch.allclose(base["outcome_quantiles"],
                          out["outcome_quantiles"], atol=1e-5)


def test_permutation_equivariance_over_slots():
    net = WmN1Net(small_cfg()).eval()
    b = make_batch(B=1, K=6)
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    with torch.no_grad():
        base = net(**b)
        out = net(veh_step=b["veh_step"][:, perm],
                  veh_mask=b["veh_mask"][:, perm],
                  plan=b["plan"][:, perm], wall=b["wall"],
                  wall_mask=b["wall_mask"], counter=b["counter"])
    assert torch.allclose(base["slot_hidden"][:, perm],
                          out["slot_hidden"], atol=1e-5)
    assert torch.allclose(base["global_token"], out["global_token"], atol=1e-5)
    assert torch.allclose(base["outcome_quantiles"],
                          out["outcome_quantiles"], atol=1e-5)


def test_paired_contrast_is_quantile_difference():
    from model.net import OutcomeHead
    qa = torch.randn(3, C.N_TAU, C.N_OUTCOME_TARGETS, C.N_QUANTILES)
    qb = torch.randn(3, C.N_TAU, C.N_OUTCOME_TARGETS, C.N_QUANTILES)
    assert torch.allclose(OutcomeHead.paired_contrast(qa, qb), qa - qb)
