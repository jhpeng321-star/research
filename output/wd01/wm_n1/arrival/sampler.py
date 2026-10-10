"""P4 arrival sampling + insertion into the rollout state.

Insertions mirror ``model.rollout._insert_arrivals`` (up to
``max_insert_per_entrance`` per entrance, behind the current lane occupancy)
but act on the P3 :class:`rollout2.state.WorldState`.  Sampling is deterministic
under a caller-supplied numpy generator.

Entrance lanes (frozen): entrance 0 (mainline) inserts on lane 1
(``main_right``); entrance 1 (ramp) inserts on lane 2 (``accel``).  Inserted
vehicles are generic HDVs (``is_cav=False``, no wall membership).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from model.constants import NOMINAL_LENGTH_M


def _as_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class ArrivalSampler:
    """Negative-binomial count + normal insertion-speed sampler."""

    def __init__(self, cfg=None) -> None:
        from .head import ArrivalConfig

        self.cfg = cfg or ArrivalConfig()

    # ---------------------------------------------------------------- counts
    def sample_counts(self, head_out: dict, *, rng) -> torch.Tensor:
        """-> ``int64 [B, 2]`` clipped counts (<= max_insert_per_entrance)."""
        if rng is None:
            raise ValueError("ArrivalSampler.sample_counts needs an rng")
        mu = _as_numpy(head_out["count_mu"]).astype(np.float64)
        log_alpha = _as_numpy(head_out["count_log_alpha"]).astype(np.float64)
        r = np.log1p(np.exp(log_alpha)) + 1e-4            # softplus
        p = np.clip(r / (r + np.abs(mu) + 1e-6), 1e-6, 1.0 - 1e-6)
        counts = rng.negative_binomial(np.maximum(r, 1e-3), p,
                                       size=mu.shape)
        counts = np.clip(counts, 0, int(self.cfg.max_insert_per_entrance))
        device = getattr(head_out.get("count_mu"), "device", None)
        return torch.as_tensor(counts.astype(np.int64), device=device)

    # ---------------------------------------------------------------- speeds
    def sample_speeds(self, head_out: dict, *, rng) -> torch.Tensor:
        """-> ``float [B, 2]`` non-negative insertion speeds."""
        if rng is None:
            raise ValueError("ArrivalSampler.sample_speeds needs an rng")
        mu = _as_numpy(head_out["speed_mu"]).astype(np.float64)
        log_sigma = _as_numpy(head_out["speed_log_sigma"]).astype(np.float64)
        sigma = np.exp(log_sigma)
        speeds = mu + sigma * rng.standard_normal(size=mu.shape)
        device = getattr(head_out.get("speed_mu"), "device", None)
        return torch.as_tensor(np.maximum(speeds, 0.0).astype(np.float32),
                               device=device)

    # ---------------------------------------------------------------- insert
    def insert(self, world, counts, speeds) -> Tuple[object, torch.Tensor]:
        """Insert arrivals into empty slots; returns ``(world, inserted)``.

        ``world`` is a :class:`rollout2.state.WorldState`.  ``counts`` is
        ``[B, 2]`` int, ``speeds`` is ``[B, 2]`` m/s.  Returns a NEW world
        (input tensors are not mutated) and ``int64 [B, 2]`` inserted counts.
        """
        cnt = _as_numpy(counts).astype(np.int64)
        spd = _as_numpy(speeds).astype(np.float64)
        n_ent = int(self.cfg.n_entrances)
        if cnt.ndim != 2 or cnt.shape[1] != n_ent:
            raise ValueError("counts must be [B, %d]" % n_ent)
        if spd.shape != cnt.shape:
            raise ValueError("speeds must match counts shape")

        b = int(world.present.shape[0])
        s = world.s_m.clone()
        v = world.v_mps.clone()
        a = world.a_mps2.clone()
        lane = world.lane.clone()
        length = world.length_m.clone()
        present = world.present.clone()
        is_cav = world.is_cav.clone()
        wall_group = world.wall_group.clone()
        slot_id = world.slot_id.clone()

        nxt = int(slot_id.max().item()) + 1 if slot_id.numel() else 0
        inserted = torch.zeros(b, n_ent, dtype=torch.int64,
                               device=present.device)
        cap = int(self.cfg.max_insert_per_entrance)

        for bi in range(b):
            for e in range(n_ent):
                entry_lane = 1 if e == 0 else 2
                n = int(min(max(int(cnt[bi, e]), 0), cap))
                for _ in range(n):
                    free = (~present[bi]).nonzero(as_tuple=False)
                    if free.numel() == 0:
                        break
                    slot = int(free[0, 0])
                    occ_lane = present[bi] & (lane[bi] == entry_lane)
                    if bool(occ_lane.any()):
                        pos = float(s[bi][occ_lane].min()) - NOMINAL_LENGTH_M
                    else:
                        pos = -NOMINAL_LENGTH_M
                    speed = float(max(spd[bi, e], 0.0))
                    s[bi, slot] = pos
                    v[bi, slot] = speed
                    a[bi, slot] = 0.0
                    lane[bi, slot] = entry_lane
                    length[bi, slot] = NOMINAL_LENGTH_M
                    present[bi, slot] = True
                    is_cav[bi, slot] = False
                    wall_group[bi, slot] = -1
                    slot_id[bi, slot] = nxt
                    nxt += 1
                    inserted[bi, e] += 1

        new_world = type(world)(
            s_m=s, v_mps=v, a_mps2=a, lane=lane, length_m=length,
            present=present, is_cav=is_cav, wall_group=wall_group,
            slot_id=slot_id)
        return new_world, inserted
