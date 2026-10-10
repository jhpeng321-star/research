"""Deterministic synthetic rollout state for the P4 sampler tests (no data)."""
from __future__ import annotations

import torch

from rollout2.state import WorldState


def make_world(batch: int = 2, slots: int = 6, *, seed: int = 0,
               free_slots: int = 2) -> WorldState:
    g = torch.Generator().manual_seed(int(seed))
    present = torch.ones(batch, slots, dtype=torch.bool)
    if free_slots:
        present[:, slots - free_slots:] = False
    s = torch.rand(batch, slots, generator=g) * 100.0
    v = torch.rand(batch, slots, generator=g) * 20.0
    a = torch.zeros(batch, slots)
    lane = torch.randint(0, 3, (batch, slots), generator=g)
    length = torch.full((batch, slots), 5.0)
    is_cav = torch.zeros(batch, slots, dtype=torch.bool)
    wall_group = torch.full((batch, slots), -1, dtype=torch.long)
    slot_id = torch.arange(batch * slots, dtype=torch.long).view(batch, slots)
    return WorldState(s_m=s, v_mps=v, a_mps2=a, lane=lane, length_m=length,
                      present=present, is_cav=is_cav, wall_group=wall_group,
                      slot_id=slot_id)
