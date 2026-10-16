"""RSU own-sensor field: per-cell aggregates over the RSU zones.

The same function runs on imagined WM-S states and on states converted from SUMO, so the
group signal sees identical inputs in training and deployment.  Only vehicles inside the
RSU zones contribute; CAV share uses ``is_cav`` (CAVs identify themselves to the RSU).
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor

from .contract import (F_RSU_CELL, RSU_CELL_M, RSU_MAINLINE_S_M, RSU_RAMP_S_M,
                       SPEED_LIMIT_MPS)

QUEUE_SPEED_MPS = 5.0


def _cells(lo: float, hi: float) -> list[tuple[float, float]]:
    n = int(math.ceil((hi - lo) / RSU_CELL_M - 1e-9))
    return [(lo + i * RSU_CELL_M, min(hi, lo + (i + 1) * RSU_CELL_M)) for i in range(n)]


#: (lane, s_lo, s_hi) per cell: lane 0, lane 1 over the mainline zone, then lane 2 (ramp).
CELLS: tuple[tuple[int, float, float], ...] = tuple(
    [(0, a, b) for a, b in _cells(*RSU_MAINLINE_S_M)]
    + [(1, a, b) for a, b in _cells(*RSU_MAINLINE_S_M)]
    + [(2, a, b) for a, b in _cells(*RSU_RAMP_S_M)])
N_RSU_CELLS = len(CELLS)


def cell_geometry(device=None, dtype=torch.float32) -> Tensor:
    """[N, 4] = (cell centre s / 1000, lane one-hot x3) for the encoder."""
    rows = []
    for lane, lo, hi in CELLS:
        onehot = [0.0, 0.0, 0.0]
        onehot[lane] = 1.0
        rows.append([0.5 * (lo + hi) / 1000.0] + onehot)
    return torch.tensor(rows, device=device, dtype=dtype)


def build_rsu_field(state: Mapping[str, Tensor]) -> Tensor:
    """RSU field [B, N_RSU_CELLS, F_RSU_CELL] (contract.RSU_FIELD_FEATURES order)."""
    s, v = state["s_m"], state["v_mps"]
    lane = state["lane"].long()
    present = state.get("present", torch.ones_like(s, dtype=torch.bool)).bool()
    is_cav = state.get("is_cav", torch.zeros_like(present)).bool()
    lane_t = torch.tensor([c[0] for c in CELLS], device=s.device)
    lo = torch.tensor([c[1] for c in CELLS], device=s.device, dtype=s.dtype)
    hi = torch.tensor([c[2] for c in CELLS], device=s.device, dtype=s.dtype)
    # member [B, N, K]
    member = (present[:, None, :] & (lane[:, None, :] == lane_t[None, :, None])
              & (s[:, None, :] >= lo[None, :, None]) & (s[:, None, :] < hi[None, :, None]))
    w = member.to(s.dtype)
    count = w.sum(-1)
    occupied = count > 0
    denom = count.clamp_min(1.0)
    mean_v = (w * v[:, None, :]).sum(-1) / denom
    var_v = (w * (v[:, None, :] - mean_v[..., None]) ** 2).sum(-1) / denom
    mean_v = torch.where(occupied, mean_v, torch.full_like(mean_v, SPEED_LIMIT_MPS))
    length_km = ((hi - lo) / 1000.0)[None, :]
    cav_share = (w * is_cav[:, None, :].to(s.dtype)).sum(-1) / denom
    queue = (occupied & (mean_v < QUEUE_SPEED_MPS)).to(s.dtype)
    field = torch.stack((count / length_km, mean_v, (var_v.clamp_min(0.0) + 1e-6).sqrt(), queue,
                         occupied.to(s.dtype), cav_share), dim=-1)
    assert field.shape[-1] == F_RSU_CELL
    return field


#: Network scaling of the field features (physical units are kept in the field itself).
FIELD_SCALE = (100.0, SPEED_LIMIT_MPS, 10.0, 1.0, 1.0, 1.0)
