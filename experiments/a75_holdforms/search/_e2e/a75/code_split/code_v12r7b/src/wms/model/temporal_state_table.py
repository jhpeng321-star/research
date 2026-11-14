"""R13S state-conditional mean of the temporal acceleration emission (S-FACTORIAL "T mean" arm, made trainable).

S-FACTORIAL (E:/agent-work/c4-factorial/FACTORIAL.md, scripts/factworker.py + fit_table.py) replaced the learned
temporal acceleration emission by its STATE-CONDITIONAL MEAN: a table over (lane, speed, bumper gap to the same-lane
leader, closing speed, position, steps since observation k), fitted on the as-is rollouts of the checkpoint, with a
backoff over five coarser levels (a cell is used when it holds >= 30 fitted rows).  That arm kept every delay ratio
and removed the hard brakes.  This module reproduces the arm's features, bins, levels and backoff exactly; the
model holds the five mean tables as PARAMETERS (initialised from the fitted table, trainable) and the counts as
buffers, so the emission is a deterministic function of the current state and k only.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from torch import Tensor

#: S-FACTORIAL bin edges (factworker.py), bucketize right=False == numpy searchsorted side="left".
V_B = (1.0, 2.0, 4.0, 7.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0)          # 11 bins
GAP_B = (2.0, 5.0, 10.0, 20.0, 40.0, 80.0)                              # 7 bins + no leader = 8
CL_B = (-2.0, -0.5, 0.5, 2.0, 5.0)                                      # 6 bins
S_B = (-400.0, -150.0, 0.0, 75.44, 200.0)                               # 6 bins
K_B = (10.0, 40.0, 100.0)                                               # 4 bins
SHAPE = (3, 11, 8, 6, 6, 4)
#: backoff levels (indices into lane, v, gap, closing, s, k), finest first
LEVELS = ((0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5), (0, 1, 4, 5), (0, 1, 5), (0, 5))
MIN_ROWS = 30
NO_LEADER_GAP = 1.0e3


def level_shape(level: int) -> tuple[int, ...]:
    return tuple(SHAPE[i] for i in LEVELS[level])


@torch.no_grad()
def state_features(kin: Tensor, present: Tensor, k: Tensor) -> tuple[Tensor, ...]:
    """(lane, v, gap, closing, s, k) ``[B, K]`` exactly as S-FACTORIAL ``state_feats`` (no leader: gap 1e3, closing 0)."""
    s = kin[..., 0]
    onehot = kin[..., 1:4]
    lane = torch.where(onehot.sum(-1) > 0, onehot.argmax(-1), torch.full_like(s, -1, dtype=torch.long))
    v = kin[..., 4]
    length = kin[..., 6]
    present = present.bool()
    same = (lane[..., :, None] == lane[..., None, :]) & present[..., None, :] & present[..., :, None]
    ahead = (s[..., None, :] > s[..., :, None]) & same
    gapm = s[..., None, :] - s[..., :, None] - length[..., None, :]
    gapm = torch.where(ahead, gapm, torch.full_like(gapm, 1e4))
    gap, j = gapm.min(-1)
    v_leader = torch.gather(v, -1, j)
    has = gap < NO_LEADER_GAP
    gap = torch.where(has, gap, torch.full_like(gap, NO_LEADER_GAP))
    closing = torch.where(has, v - v_leader, torch.zeros_like(v))
    kk = k.expand_as(s) if k.ndim == 0 else k
    return lane, v, gap, closing, s, kk.to(s.dtype)


@torch.no_grad()
def bin_index(lane: Tensor, v: Tensor, gap: Tensor, closing: Tensor, s: Tensor, k: Tensor) -> tuple[Tensor, ...]:
    def edges(values):
        return torch.tensor(values, dtype=v.dtype, device=v.device)
    lane, v, gap, closing, s, k = (t.contiguous() for t in (lane, v, gap, closing, s, k))
    vb = torch.bucketize(v, edges(V_B))
    gb = torch.where(gap >= 999, torch.full_like(vb, 7), torch.bucketize(gap, edges(GAP_B)))
    cb = torch.bucketize(closing, edges(CL_B))
    sb = torch.bucketize(s, edges(S_B))
    kb = torch.bucketize(k, edges(K_B))
    return lane.clamp(0, 2), vb, gb, cb, sb, kb


def lookup(means: list[Tensor], counts: list[Tensor], bins: tuple[Tensor, ...]) -> Tensor:
    """Backoff lookup: the finest level whose cell holds >= MIN_ROWS rows, the coarsest level otherwise."""
    out = None
    for level in reversed(range(len(LEVELS))):
        idx = tuple(bins[i] for i in LEVELS[level])
        value = means[level][idx]
        if out is None:
            out = value
        else:
            out = torch.where(counts[level][idx] >= MIN_ROWS, value, out)
    return out


def fit(x: np.ndarray, y: np.ndarray) -> dict:
    """Per-level cell means and counts of ``y`` (S-FACTORIAL ``fit_table.fit``); ``x`` columns lane, v, gap, closing, s, k."""
    lane, v, gap, clos, s, k = (x[:, i] for i in range(6))
    b = (np.clip(lane, 0, 2).astype(int), np.searchsorted(V_B, v, side="left"),
         np.where(gap >= 999, 7, np.searchsorted(GAP_B, gap, side="left")),
         np.searchsorted(CL_B, clos, side="left"), np.searchsorted(S_B, s, side="left"),
         np.searchsorted(K_B, k, side="left"))
    out = {}
    for level, keep in enumerate(LEVELS):
        shape = level_shape(level)
        flat = np.ravel_multi_index(tuple(b[i] for i in keep), shape)
        n = np.bincount(flat, minlength=int(np.prod(shape)))
        total = np.bincount(flat, y, minlength=int(np.prod(shape)))
        out[f"mean{level}"] = (total / np.maximum(n, 1)).reshape(shape).astype(np.float32)
        out[f"n{level}"] = n.reshape(shape).astype(np.int64)
    return out


def check_table(table: Mapping[str, np.ndarray]) -> None:
    for level in range(len(LEVELS)):
        for key in (f"mean{level}", f"n{level}"):
            if key not in table or tuple(np.shape(table[key])) != level_shape(level):
                raise ValueError(f"temporal state table: {key} missing or shape != {level_shape(level)}")
        if not np.isfinite(np.asarray(table[f"mean{level}"])).all():
            raise ValueError(f"temporal state table: mean{level} not finite")


__all__ = ["CL_B", "GAP_B", "K_B", "LEVELS", "MIN_ROWS", "SHAPE", "S_B", "V_B", "bin_index", "check_table",
           "fit", "level_shape", "lookup", "state_features"]
