"""Learned merge negotiation between a ramp car and its lane-1 follower.

A small MLP/attention block over the (ramp car, lane-1 follower) pair.  It returns

* ``yield_accel`` -- a non-positive acceleration correction for the follower (the
  learned courtesy / yield response),
* ``ramp_accel_delta`` -- a small signed longitudinal residual for the ramp car,
  conditioned on the target-gap geometry (R5YIELD),
* ``merge_logit_delta`` -- an additive correction on the ramp car's merge logit
  (gap acceptance conditioned on the follower's response), and
* ``required_brake`` / ``conflict_logit`` -- the interaction / safety head's predicted
  post-insertion braking requirement and conflict probability (R5YIELD).

The last layer of every output head is zero-initialised, so enabling the block is an
exact no-op until trained.  Inputs are the lane-1 gap and relative speed of the nearest
follower and leader, the next-behind (second nearest) follower gap, the ramp car's
distance to the acceleration-lane end, waiting time, the follower's current and EMA
acceleration (deceleration history), and the merge head's current intent (probability,
logit and detached logit trend).  The block is stateless per step; the rollout carries
the follower-acceleration EMA and the previous intent logit as plain state entries.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

#: Per-pair features (physical units before the network's own normalisation).
PAIR_FEATURE_NAMES = (
    "ramp", "dend", "in_zone", "ramp_v", "ramp_wait", "ramp_acc",
    "follower", "gap_f", "rel_speed_f", "follower_acc", "follower_decel",
    "follower_acc_ema", "leader", "gap_l", "rel_speed_l", "intent",
)
N_PAIR_FEATURES = len(PAIR_FEATURE_NAMES)

#: R5YIELD extension: gap geometry + detached merge-intent score / trend.
EXT_FEATURE_NAMES = (
    "intent_logit", "intent_trend", "next_behind", "next_gap_f", "rel_speed_next",
)
N_EXT_FEATURES = N_PAIR_FEATURES + len(EXT_FEATURE_NAMES)


def _lane(kin: Tensor) -> Tensor:
    onehot = kin[..., 1:4]
    lane = onehot.argmax(-1)
    return torch.where(onehot.sum(-1) > 0, lane, torch.full_like(lane, -1))


def pair_features(kin: Tensor, present: Tensor, physics,
                  *, merge_prob: Optional[Tensor] = None,
                  merge_logit: Optional[Tensor] = None,
                  prev_merge_logit: Optional[Tensor] = None,
                  wait_s: Optional[Tensor] = None,
                  accel_ema: Optional[Tensor] = None,
                  follower_max_gap_m: float = 150.0,
                  leader_max_gap_m: float = 150.0) -> dict[str, Tensor]:
    """Features of every ramp slot's nearest lane-1 follower / leader.

    Returns ``features [B, K, N_PAIR_FEATURES]`` (the frozen V10NEGOT inputs) and
    ``ext_features [B, K, N_EXT_FEATURES]`` (the R5YIELD extension), both on the ramp
    slots (zeros elsewhere), plus ``follower_index [B, K]``, ``valid_pair [B, K]``
    (ramp car with a lane-1 follower) and ``ramp [B, K]``.  No future information: all
    quantities are the current state.  ``merge_logit`` / ``prev_merge_logit`` are the
    detached merge-head score and its previous-step value (the trend is their
    difference).
    """
    present = present.bool()
    kin = kin.float()
    s, v, a, length = kin[..., 0], kin[..., 4], kin[..., 5], kin[..., 6]
    lane = _lane(kin)
    b, k = s.shape
    ramp = present & (lane == int(getattr(physics, "merge_lane", 2)))
    target_lane = int(getattr(physics, "merge_target_lane", 1))
    target = present & (lane == target_lane)
    ds = s.unsqueeze(-1) - s.unsqueeze(-2)                  # [B, i(ramp), j(target)]
    inf = torch.full_like(ds, float("inf"))
    # Nearest follower behind: 0 <= s_i - s_j <= follower_max_gap_m.
    behind = target.unsqueeze(-2) & (ds >= 0.0) & (ds <= float(follower_max_gap_m))
    f_key = torch.where(behind, ds, inf)
    gap_f, f_idx = f_key.min(-1)
    has_f = torch.isfinite(gap_f)
    # Next-behind follower (second smallest gap): blank the nearest target column.
    safe_f = f_idx.clamp(0, max(k - 1, 0))
    behind2 = behind.scatter(-1, safe_f.unsqueeze(-1), False)
    f2_key = torch.where(behind2, ds, inf)
    gap_f2, f2_idx = f2_key.min(-1)
    has_f2 = torch.isfinite(gap_f2)
    safe_f2 = f2_idx.clamp(0, max(k - 1, 0))
    # Nearest leader ahead: 0 <= s_j - s_i <= leader_max_gap_m.
    ahead = target.unsqueeze(-2) & (ds <= 0.0) & (-ds <= float(leader_max_gap_m))
    l_key = torch.where(ahead, -ds, inf)
    gap_l, l_idx = l_key.min(-1)
    has_l = torch.isfinite(gap_l)
    safe_l = l_idx.clamp(0, max(k - 1, 0))
    v_f = v.gather(-1, safe_f)
    v_f2 = v.gather(-1, safe_f2)
    a_f = a.gather(-1, safe_f)
    len_f = length.gather(-1, safe_f)
    len_f2 = length.gather(-1, safe_f2)
    v_l = v.gather(-1, safe_l)
    len_l = length.gather(-1, safe_l)
    gap_f = torch.where(has_f, (gap_f - len_f).clamp_min(0.0), torch.zeros_like(gap_f))
    gap_f2 = torch.where(has_f2, (gap_f2 - len_f2).clamp_min(0.0), torch.zeros_like(gap_f2))
    gap_l = torch.where(has_l, (gap_l - len_l).clamp_min(0.0), torch.zeros_like(gap_l))
    zero = torch.zeros_like(s)
    ema = zero if accel_ema is None else torch.as_tensor(
        accel_ema, device=s.device, dtype=s.dtype).clamp(-9.0, 3.0)
    if wait_s is None:
        wait = zero
    else:
        wait = torch.as_tensor(wait_s, device=s.device, dtype=s.dtype).clamp_min(0.0)
    intent = zero if merge_prob is None else torch.as_tensor(
        merge_prob, device=s.device, dtype=s.dtype).clamp(0.0, 1.0).detach()
    logit = zero if merge_logit is None else torch.as_tensor(
        merge_logit, device=s.device, dtype=s.dtype).detach()
    prev_logit = zero if prev_merge_logit is None else torch.as_tensor(
        prev_merge_logit, device=s.device, dtype=s.dtype).detach()
    end = float(getattr(physics, "merge_zone_s1_m", 75.44))
    dend = (end - s).clamp(-100.0, 500.0)
    in_zone = (s >= float(getattr(physics, "merge_zone_s0_m", 0.0))) & (s <= end)
    feats = torch.stack((
        ramp.to(s.dtype),
        dend / 50.0,
        (ramp & in_zone).to(s.dtype),
        (v / 20.0).clamp(0.0, 2.0),
        (wait / 20.0).clamp(0.0, 3.0),
        (a / 5.0).clamp(-2.0, 1.0),
        has_f.to(s.dtype),
        (gap_f / 50.0).clamp(0.0, 3.0),
        ((v - v_f) / 10.0).clamp(-3.0, 3.0),
        (a_f / 5.0).clamp(-2.0, 1.0),
        ((-a_f).clamp_min(0.0) / 5.0).clamp(0.0, 2.0),
        (ema / 5.0).clamp(-2.0, 1.0),
        has_l.to(s.dtype),
        (gap_l / 50.0).clamp(0.0, 3.0),
        ((v - v_l) / 10.0).clamp(-3.0, 3.0),
        intent,
    ), -1)
    extras = torch.stack((
        (logit / 10.0).clamp(-3.0, 3.0),
        ((logit - prev_logit) / 5.0).clamp(-3.0, 3.0),
        has_f2.to(s.dtype),
        (gap_f2 / 50.0).clamp(0.0, 3.0),
        ((v - v_f2) / 10.0).clamp(-3.0, 3.0),
    ), -1)
    feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    extras = torch.nan_to_num(extras, nan=0.0, posinf=0.0, neginf=0.0)
    keep = (ramp & present).unsqueeze(-1)
    feats = torch.where(keep, feats, torch.zeros_like(feats))
    extras = torch.where(keep, extras, torch.zeros_like(extras))
    return {"features": feats, "ext_features": torch.cat((feats, extras), -1),
            "follower_index": safe_f, "next_follower_index": safe_f2,
            "valid_pair": ramp & has_f, "ramp": ramp}


class MergeNegotiationNet(nn.Module):
    """Zero-initialised pair MLP: follower yield, ramp adaptation, safety and merge delta."""

    def __init__(self, hidden: int = 64, yield_scale_mps2: float = 1.5,
                 merge_scale: float = 1.0, ramp_scale_mps2: float = 0.5,
                 ext_hidden: int = 32, ramp_zone_m: float = 60.0) -> None:
        super().__init__()
        if hidden < 1 or ext_hidden < 1:
            raise ValueError("merge negotiation hidden widths must be >= 1")
        if yield_scale_mps2 <= 0.0 or merge_scale <= 0.0 or ramp_scale_mps2 <= 0.0:
            raise ValueError("merge negotiation output scales must be positive")
        if ramp_zone_m <= 0.0:
            raise ValueError("merge negotiation ramp zone must be positive")
        self.hidden = int(hidden)
        self.ext_hidden = int(ext_hidden)
        self.yield_scale_mps2 = float(yield_scale_mps2)
        self.merge_scale = float(merge_scale)
        self.ramp_scale_mps2 = float(ramp_scale_mps2)
        self.ramp_zone_m = float(ramp_zone_m)
        # Frozen V10NEGOT head (same key names / shapes as the old checkpoints).
        self.net = nn.Sequential(
            nn.Linear(N_PAIR_FEATURES, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        # R5YIELD heads: separate modules so old checkpoints keep loading with strict=False.
        self.yield_head = self._head(N_EXT_FEATURES, 1, ext_hidden)
        self.ramp_head = self._head(N_EXT_FEATURES, 1, ext_hidden)
        self.safety_head = self._head(N_EXT_FEATURES, 2, ext_hidden)
        # The merge decision sees the safety head's detached outputs as extra features.
        self.merge_safety_head = self._head(N_EXT_FEATURES + 2, 1, ext_hidden)
        # Attention over the two output channels is deliberately absent: the block is a
        # per-pair MLP whose outputs are scattered to the follower / ramp slots.
        self.attn = None

    @staticmethod
    def _head(n_in: int, n_out: int, hidden: int) -> nn.Sequential:
        head = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Linear(hidden, n_out))
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
        return head

    def forward(self, features: Tensor, *, follower_index: Tensor,
                valid_pair: Tensor, ramp: Tensor,
                ext_features: Optional[Tensor] = None) -> dict[str, Tensor]:
        """Extended outputs on ramp slots (all action outputs are zero at initialisation)."""
        if features.shape[-1] != N_PAIR_FEATURES:
            raise ValueError(f"merge negotiation needs {N_PAIR_FEATURES} pair features")
        out = self.net(features)
        if ext_features is None:
            ext = features.new_zeros((*features.shape[:-1], N_EXT_FEATURES))
        else:
            if ext_features.shape[-1] != N_EXT_FEATURES:
                raise ValueError(
                    f"merge negotiation needs {N_EXT_FEATURES} extended features")
            ext = ext_features
        base_yield = torch.where(ramp, out[..., 0], torch.zeros_like(out[..., 0]))
        yield_logit = base_yield + torch.where(
            ramp, self.yield_head(ext)[..., 0], torch.zeros_like(out[..., 0]))
        base_delta = torch.where(ramp, self.merge_scale * torch.tanh(out[..., 1]),
                                 torch.zeros_like(out[..., 1]))
        # The R5YIELD heads are only trained in the lane-end approach zone.
        near = ramp & (features[..., 1] * 50.0 < self.ramp_zone_m)
        # tanh(logit / 2) = 2 sigmoid(logit) - 1: zero at the zero-initialised logit.
        yield_accel = (-self.yield_scale_mps2 * torch.tanh(yield_logit / 2.0)
                       * valid_pair.to(yield_logit.dtype))
        ramp_delta = (self.ramp_scale_mps2 * torch.tanh(self.ramp_head(ext)[..., 0])
                      * near.to(yield_logit.dtype))
        safety = self.safety_head(ext)
        brake_raw = torch.where(near, safety[..., 0], torch.zeros_like(safety[..., 0]))
        required_brake = ((nn.functional.softplus(brake_raw) - math.log(2.0))
                          .clamp_min(0.0) * near.to(yield_logit.dtype))
        conflict_logit = torch.where(near, safety[..., 1],
                                     torch.zeros_like(safety[..., 1]))
        conflict_prob = torch.sigmoid(conflict_logit)
        merge_in = torch.cat((ext, conflict_prob.detach().unsqueeze(-1),
                              required_brake.detach().unsqueeze(-1)), -1)
        safety_delta = (self.merge_scale * torch.tanh(self.merge_safety_head(merge_in)[..., 0])
                        * near.to(yield_logit.dtype))
        return {"yield_logit": yield_logit, "yield_accel": yield_accel,
                "ramp_accel_delta": ramp_delta,
                "merge_logit_delta": base_delta + safety_delta,
                "required_brake": required_brake, "conflict_logit": conflict_logit,
                "conflict_prob": conflict_prob,
                "follower_index": follower_index, "valid_pair": valid_pair,
                "ramp": ramp}


def scatter_follower_accel(yield_accel: Tensor, follower_index: Tensor,
                           valid_pair: Tensor, k: int) -> Tensor:
    """Sum the per-ramp yield accelerations onto their lane-1 follower slots ``[B, K]``."""
    out = torch.zeros((*yield_accel.shape[:1], k), dtype=yield_accel.dtype,
                      device=yield_accel.device)
    value = torch.where(valid_pair, yield_accel, torch.zeros_like(yield_accel))
    return out.scatter_add(-1, follower_index.clamp(0, max(k - 1, 0)), value)


__all__ = ["N_PAIR_FEATURES", "N_EXT_FEATURES", "PAIR_FEATURE_NAMES", "EXT_FEATURE_NAMES",
           "MergeNegotiationNet", "pair_features", "scatter_follower_accel"]
