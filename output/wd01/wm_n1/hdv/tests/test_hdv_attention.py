"""P2 attention contract: slot masking, permutation equivariance, normality."""
from __future__ import annotations

import torch

from hdv import HdvConfig, HdvResidualNet
from synth_hdv import random_kin


def test_masking_isolates_absent_slots():
    net = HdvResidualNet(HdvConfig())
    net.eval()
    kin, mask = random_kin(batch=2, slots=6, seed=1)
    mask[:, 3] = False
    mask[:, 5] = False
    out0 = net(kin, mask)

    perturbed = kin.clone()
    perturbed[:, 3, :] += 1000.0
    perturbed[:, 5, :] -= 500.0
    out1 = net(perturbed, mask)

    present = mask
    assert torch.allclose(out0["accel_delta_mps2"][present],
                          out1["accel_delta_mps2"][present], atol=1e-6)
    assert torch.allclose(out0["lane_logit_delta"][present],
                          out1["lane_logit_delta"][present], atol=1e-6)
    # absent slots produce exactly zero correction.
    assert out0["accel_delta_mps2"][~mask].abs().max() == 0
    assert out0["lane_logit_delta"][~mask].abs().max() == 0


def test_slot_permutation_equivariance():
    net = HdvResidualNet(HdvConfig())
    net.eval()
    kin, mask = random_kin(batch=2, slots=6, seed=2)
    perm = torch.tensor([2, 0, 4, 1, 5, 3])
    out = net(kin, mask)
    out_perm = net(kin[:, perm], mask[:, perm])
    assert torch.allclose(out["accel_delta_mps2"][:, perm],
                          out_perm["accel_delta_mps2"], atol=1e-5)
    assert torch.allclose(out["lane_logit_delta"][:, perm],
                          out_perm["lane_logit_delta"], atol=1e-5)


def test_attention_rows_are_distributions_over_valid_neighbours():
    cfg = HdvConfig()
    net = HdvResidualNet(cfg)
    net.eval()
    kin, mask = random_kin(batch=2, slots=6, seed=3)
    mask[:, 4] = False
    out = net(kin, mask)
    attn = out["attn"]
    assert attn.shape == (2, cfg.n_heads, 6, 6)
    row_sums = attn.sum(dim=-1)                    # [B, H, K]
    for b in range(2):
        # present query slots: rows normalise to 1 (self is always allowed).
        assert torch.allclose(row_sums[b, :, mask[b]],
                              torch.ones_like(row_sums[b, :, mask[b]]),
                              atol=1e-5)
        # absent query slots: exactly zero.
        assert row_sums[b, :, ~mask[b]].abs().max() == 0


def test_attention_neighbourhood_is_radius_and_lane_limited():
    cfg = HdvConfig()
    net = HdvResidualNet(cfg)
    net.eval()
    kin, mask = random_kin(batch=1, slots=4, seed=4)
    # keep only slots 0 and 1, same lane, 200 m apart (> radius 150 m).
    mask = torch.tensor([[True, True, False, False]])
    kin[0, 0, 1:4] = torch.tensor([1.0, 0.0, 0.0])
    kin[0, 1, 1:4] = torch.tensor([1.0, 0.0, 0.0])
    kin[0, 0, 0] = 0.0
    kin[0, 1, 0] = 200.0
    out = net(kin, mask)
    attn = out["attn"]
    # slot 0 cannot attend to the far slot 1 (only its own row stays).
    assert attn[0, :, 0, 1].abs().max() == 0
    assert torch.allclose(attn[0, :, 0, 0], torch.ones(cfg.n_heads))
