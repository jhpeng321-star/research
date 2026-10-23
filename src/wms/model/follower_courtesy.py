"""R10B learned pre-merge follower courtesy (default OFF).

S-YIELD (R9_YIELD.md) found that in SUMO only the *target follower* of a waiting ramp car (the
nearest lane-1 vehicle behind or level with it) is held back before the merge, by an amount that
depends on the situation (bumper gap, the ramp car's distance to the lane end, its wait, its speed)
and not measurably on the driver.  This module adds ONE learned acceleration offset to such a
follower.  Nothing here encodes *when* to yield:

* the pair is defined by geometry only (no distance window): every ramp-lane car's target follower is
  the nearest lane-1 car with ``s_follower <= s_ramp``; a follower that is the target of several ramp
  cars is paired with the nearest of them.  A lane-1 car that is nobody's target follower has no
  partner and its offset is EXACTLY zero (the head is never evaluated into its acceleration);
* the inputs are continuous, smoothly normalised (``asinh``) current-state quantities; there is no
  threshold, clamp, bin or indicator of "waiting" / "near the end";
* the output is an unbounded linear read-out of a small MLP whose last layer is zero-initialised, so a
  freshly enabled head is an exact no-op and the network learns where the effect is ~0;
* no driver type, simulator cooperative state or future quantity is an input.  The per-driver
  embedding is an optional ABLATION input (``follower_courtesy_embedding``), off by default because the
  data show no driver dependence.

The inputs are computed from the state at the step START (recorded state in the teacher-forced replay,
rolled-out state at runtime) with the same function, so training and runtime see identical features.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

#: Per-follower pair inputs (all zero for a slot without a ramp partner).
COURTESY_PAIR_FIELDS = (
    "gap_bumper",       # asinh((s_ramp - len_ramp - s_follower) / 5 m); negative = overlap
    "dend",             # asinh((lane end - s_ramp) / 20 m)
    "v_follower",       # follower speed / 10 m/s
    "v_ramp",           # ramp car speed / 10 m/s
    "dv",               # (v_ramp - v_follower) / 5 m/s
    "wait",             # 1 - exp(-t_wait / 30 s): ramp car's carried merge wait (or time in its lane)
    "intent",           # merge head's detached P(merge) of the ramp car (0 when unavailable)
    "has_leader",       # follower has a lane-1 leader
    "gap_leader",       # asinh(follower bumper gap to its lane-1 leader / 10 m) (0 without leader)
    "closing_leader",   # (v_follower - v_leader) / 5 m/s (0 without leader)
)
COURTESY_PAIR_F = len(COURTESY_PAIR_FIELDS)


def _lane(kin: Tensor) -> Tensor:
    onehot = kin[..., 1:4]
    lane = onehot.argmax(-1)
    return torch.where(onehot.sum(-1) > 0, lane, torch.full_like(lane, -1))


def follower_courtesy_pairs(kin: Tensor, present: Tensor, physics, *,
                            merge_wait_s: Optional[Tensor] = None,
                            lane_age: Optional[Tensor] = None,
                            merge_prob: Optional[Tensor] = None) -> dict[str, Tensor]:
    """Pair every lane-1 follower with its ramp partner; return features ``[B, K, F]``.

    Returns ``features`` (zeros where there is no partner), ``has_partner [B, K]`` and
    ``partner [B, K]`` (slot index of the ramp car; meaningless where ``has_partner`` is false).
    Only current-state quantities enter (``merge_wait_s`` is the carried waiting time, a function
    of the past; ``merge_prob`` is detached).
    """
    kin = kin.float()
    present = present.bool()
    s, v, length = kin[..., 0], kin[..., 4], kin[..., 6]
    lane = _lane(kin)
    b, k = s.shape
    ramp = present & (lane == int(getattr(physics, "merge_lane", 2)))
    main = present & (lane == int(getattr(physics, "merge_target_lane", 1)))
    inf = torch.full((b, k, k), float("inf"), dtype=s.dtype, device=s.device)
    d = s.unsqueeze(-1) - s.unsqueeze(-2)                      # [B, j, i] = s_j - s_i
    # Target follower of each ramp car j: nearest lane-1 car i with s_i <= s_j.
    behind = ramp.unsqueeze(-1) & main.unsqueeze(-2) & (d >= 0.0)
    dmin, f_of_j = torch.where(behind, d, inf).min(-1)          # [B, j]
    has_f = torch.isfinite(dmin)
    slots = torch.arange(k, device=s.device)
    owner = has_f.unsqueeze(-1) & (f_of_j.unsqueeze(-1) == slots)   # [B, j, i]
    dpart, partner = torch.where(owner, d, inf).min(-2)          # nearest owning ramp car per i
    has_partner = torch.isfinite(dpart) & main
    partner = torch.where(has_partner, partner, torch.zeros_like(partner))
    # Follower's own lane-1 leader (strictly ahead in its lane).
    ahead = main.unsqueeze(-2) & main.unsqueeze(-1) & (d.transpose(-1, -2) > 0.0)  # [B, i, l]
    lead_key = torch.where(ahead, d.transpose(-1, -2), inf)
    dlead, leader = lead_key.min(-1)
    has_leader = torch.isfinite(dlead) & main
    leader = torch.where(has_leader, leader, torch.zeros_like(leader))

    def at(x: Tensor, index: Tensor) -> Tensor:
        return x.gather(-1, index)

    zero = torch.zeros_like(s)
    s_r, v_r, len_r = at(s, partner), at(v, partner), at(length, partner)
    gap_b = s_r - len_r - s
    end = float(getattr(physics, "merge_zone_s1_m", 75.44))
    if merge_wait_s is not None:
        wait_all = torch.as_tensor(merge_wait_s, dtype=s.dtype, device=s.device)
    elif lane_age is not None:
        wait_all = (torch.as_tensor(lane_age, dtype=s.dtype, device=s.device)
                    * float(getattr(physics, "dt_s", 0.5)))
    else:
        wait_all = zero
    intent_all = zero if merge_prob is None else torch.as_tensor(
        merge_prob, dtype=s.dtype, device=s.device).detach()
    gap_l = at(s, leader) - at(length, leader) - s
    lead_on = has_leader.to(s.dtype)
    feats = torch.stack((
        torch.asinh(gap_b / 5.0),
        torch.asinh((end - s_r) / 20.0),
        v / 10.0,
        v_r / 10.0,
        (v_r - v) / 5.0,
        # Smooth saturating map: the lane-age fallback uses a large sentinel for "in this lane for
        # at least the observed history", which this maps to ~1 instead of an outlier.
        -torch.expm1(-at(wait_all, partner).clamp_min(0.0) / 30.0),
        at(intent_all, partner),
        lead_on,
        torch.where(has_leader, torch.asinh(gap_l / 10.0), zero),
        torch.where(has_leader, (v - at(v, leader)) / 5.0, zero),
    ), -1)
    feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    feats = torch.where(has_partner.unsqueeze(-1), feats, torch.zeros_like(feats))
    return {"features": feats.detach(), "has_partner": has_partner, "partner": partner}


class FollowerCourtesyHead(nn.Module):
    """Pair MLP -> one acceleration offset (m/s^2) per follower; zero last layer (exact no-op)."""

    def __init__(self, hidden: int = 32, embed_dim: int = 0) -> None:
        super().__init__()
        if int(hidden) < 1 or int(embed_dim) < 0:
            raise ValueError("follower courtesy needs hidden >= 1 and embed_dim >= 0")
        self.n_in = COURTESY_PAIR_F + int(embed_dim)
        self.embed_dim = int(embed_dim)
        self.net = nn.Sequential(
            nn.Linear(self.n_in, int(hidden)), nn.GELU(),
            nn.Linear(int(hidden), int(hidden)), nn.GELU(),
            nn.Linear(int(hidden), 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.shape[-1] != self.n_in:
            raise ValueError(f"follower courtesy expects {self.n_in} inputs")
        return self.net(inputs)[..., 0]


__all__ = ["COURTESY_PAIR_F", "COURTESY_PAIR_FIELDS", "FollowerCourtesyHead",
           "follower_courtesy_pairs"]
