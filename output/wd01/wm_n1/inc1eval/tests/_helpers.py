"""Synthetic-world helpers for the INC1 tests (zero SUMO, zero outcomes)."""
from __future__ import annotations

import torch

from rollout2.state import WorldState


def make_world(rows, *, slots: int = 8, dtype=torch.float32):
    """Build a single-world ``WorldState`` from ``(lane, s, v, is_cav)`` rows.

    Slot order is the row order; ``length_m`` is 5 m for every vehicle and
    ``slot_id`` is the slot index.
    """
    b_n, k_n = 1, int(slots)
    s = torch.zeros((b_n, k_n), dtype=dtype)
    v = torch.zeros((b_n, k_n), dtype=dtype)
    a = torch.zeros((b_n, k_n), dtype=dtype)
    lane = torch.full((b_n, k_n), -1, dtype=torch.int64)
    length = torch.full((b_n, k_n), 5.0, dtype=dtype)
    present = torch.zeros((b_n, k_n), dtype=torch.bool)
    is_cav = torch.zeros((b_n, k_n), dtype=torch.bool)
    for i, row in enumerate(rows):
        ln, ss, vv = row[0], row[1], row[2]
        cav = bool(row[3]) if len(row) > 3 else False
        s[0, i] = ss
        v[0, i] = vv
        lane[0, i] = ln
        present[0, i] = True
        is_cav[0, i] = cav
    return WorldState(
        s_m=s, v_mps=v, a_mps2=a, lane=lane, length_m=length, present=present,
        is_cav=is_cav, wall_group=torch.full((b_n, k_n), -1,
                                             dtype=torch.int64),
        slot_id=torch.arange(k_n).expand(b_n, k_n).contiguous())


def free_rng():
    import numpy as np

    return np.random.default_rng(0)
