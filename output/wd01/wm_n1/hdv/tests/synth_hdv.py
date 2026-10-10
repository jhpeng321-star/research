"""Deterministic synthetic kinematics fixtures for the P2 tests (no data)."""
from __future__ import annotations

import torch

from hdv import KIN_F


def random_kin(batch: int = 2, slots: int = 6, *, seed: int = 0):
    """``(kin [B,K,KIN_F], mask [B,K])`` with deterministic random values."""
    g = torch.Generator().manual_seed(int(seed))
    kin = torch.zeros(batch, slots, KIN_F)
    kin[..., 0] = torch.rand(batch, slots, generator=g) * 300.0   # s_down
    kin[..., 4] = torch.rand(batch, slots, generator=g) * 30.0    # v
    kin[..., 5] = (torch.rand(batch, slots, generator=g) - 0.5) * 4.0
    kin[..., 6] = 5.0
    lane = torch.randint(0, 3, (batch, slots), generator=g)
    for li in range(3):
        kin[..., 1 + li] = (lane == li).to(kin.dtype)
    mask = torch.ones(batch, slots, dtype=torch.bool)
    return kin, mask
