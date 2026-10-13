"""WMLANE A: dedicated per-HDV lane-change hazard model.

``model.lane_model = "hazard"`` replaces the gated residual lane-logit delta by an
event model: for every learned (HDV) slot an MLP maps interpretable features of the
two adjacent target lanes (front/rear gaps and speeds, the speed advantage over the
current-lane leader), the ego state (speed, desired-speed deficit, lane index, time
since the last change, distance to the merge end) and the car's own learned driver
embedding to two logit deltas: *left* (lane - 1) and *right* (lane + 1).

The output stays in the absolute-lane format of the rest of WM-S (``[B, K, 3]``
logits over lanes 0/1/2, the current lane at the physics 'stay' logit 0), so the
rollout decision (argmax), the lane loss, the flip-rate guard and every diagnostic
work unchanged.  The physics feasibility mask is HARD: a target the physics masked
(``<= MASKED_LOGIT``: unsafe gap, ramp entry, non-adjacent, forced-merge-only rule)
keeps its masked logit, and inside the last ``MANDATORY_MERGE_WINDOW_M`` before the
acceleration-lane end the physics forced-merge logit is kept as is (the learned
hazard can delay a ramp merge, never strand a car at the lane end).

The final layer is zero-initialised, so a fresh hazard model reproduces the physics
prior exactly (under ``learned_discretionary``: no discretionary change at all).
"""
from __future__ import annotations

from typing import Mapping, Optional

import torch
from torch import Tensor, nn

from wms.physics.config import PhysicsConfig
from wms.physics.mobil import MASKED_LOGIT
from wms.physics.neighbours import lane_neighbour, lane_of, lead_neighbour

from .lane_head import MANDATORY_MERGE_WINDOW_M

#: Per-target-lane features: valid, lead gap, lead dv, follow gap, follow dv,
#: speed advantage, physics prior logit, physics feasible.
_PER_TARGET_F = 8
#: Ego features: v, a, desired-speed deficit, v/v0, current lead gap/dv, lane one-hot (3),
#: lane age, never-changed flag, merge-end distance, in-merge-zone flag, position.
_EGO_F = 14
#: Width of :func:`hazard_features`.
HAZARD_F = 2 * _PER_TARGET_F + _EGO_F
#: Lane-change directions handled by the hazard: (left, right) lane offsets.
DIRECTIONS = (-1, 1)
_AGE_SENTINEL = 999.0


def _target_features(kin: Tensor, present: Tensor, lane_base: Tensor, target: Tensor,
                     physics: PhysicsConfig, v0: Tensor, current_lead_v: Tensor
                     ) -> tuple[Tensor, Tensor]:
    """Features ``[B, K, 8]`` of one adjacent target lane and its physics prior logit."""
    v = kin[..., 4]
    valid = (target >= 0) & (target <= 2)
    target_c = target.clamp(0, 2)
    lead_gap, lead_v = lane_neighbour(kin, present, target_c, physics, ahead=True)
    foll_gap, foll_v = lane_neighbour(kin, present, target_c, physics, ahead=False)
    free = float(physics.free_road_gap_m) - 1.0e-3
    # Without a leader the car could reach its desired speed in that lane.
    lead_speed = torch.where(lead_gap >= free, v0, lead_v)
    advantage = (torch.minimum(lead_speed, v0) - torch.minimum(current_lead_v, v0)) / 10.0
    prior = lane_base.gather(-1, target_c.unsqueeze(-1)).squeeze(-1)
    prior = torch.where(valid, prior, torch.full_like(prior, MASKED_LOGIT))
    feasible = prior > MASKED_LOGIT
    vf = valid.to(kin.dtype)
    feats = torch.stack((vf, vf * lead_gap.clamp(0, 250) / 100, vf * (lead_v - v) / 30,
                         vf * foll_gap.clamp(0, 250) / 100, vf * (foll_v - v) / 30,
                         vf * advantage, prior.clamp(MASKED_LOGIT, -MASKED_LOGIT) / 8.0,
                         feasible.to(kin.dtype)), -1)
    return feats, prior


def hazard_features(kin: Tensor, present: Tensor, lane_base: Tensor, physics: PhysicsConfig,
                    driver_params: Optional[Mapping[str, Tensor]],
                    lane_age: Optional[Tensor]) -> tuple[Tensor, Tensor]:
    """Hazard features ``[B, K, HAZARD_F]`` and physics prior logits ``[B, K, 2]`` (left, right)."""
    s, v, a = kin[..., 0], kin[..., 4], kin[..., 5]
    lane = lane_of(kin)
    v0 = (torch.full_like(v, float(physics.v0_mps)) if driver_params is None
          else driver_params["v0"].to(v.dtype)).clamp_min(1.0)
    lead_gap, lead_v = lead_neighbour(kin, present, physics)
    free = float(physics.free_road_gap_m) - 1.0e-3
    current_lead_v = torch.where(lead_gap >= free, v0, lead_v)
    chunks, priors = [], []
    for offset in DIRECTIONS:
        feats, prior = _target_features(kin, present, lane_base, lane + offset, physics, v0,
                                        current_lead_v)
        chunks.append(feats)
        priors.append(prior)
    age = torch.full_like(s, _AGE_SENTINEL) if lane_age is None else lane_age.to(s.dtype)
    in_zone = ((lane == physics.merge_lane) & (s >= physics.merge_zone_s0_m)
               & (s <= physics.merge_zone_s1_m))
    ego = torch.stack((v / 30, a / 3, (v0 - v) / 10, (v / v0).clamp(0, 2),
                       lead_gap.clamp(0, 250) / 100, (lead_v - v) / 30,
                       kin[..., 1], kin[..., 2], kin[..., 3],
                       (age * physics.dt_s).clamp(0, 60) / 10,
                       (age >= _AGE_SENTINEL).to(s.dtype),
                       (physics.merge_zone_s1_m - s).clamp(-250, 250) / 100,
                       in_zone.to(s.dtype), (s / 1000.0).clamp(-1, 5)), -1)
    features = torch.cat((*chunks, ego), -1)
    return torch.where(present.unsqueeze(-1), features, torch.zeros_like(features)), \
        torch.stack(priors, -1)


class HazardLaneModel(nn.Module):
    """MLP over :func:`hazard_features` + driver embedding -> bounded (left, right) deltas."""

    def __init__(self, embed_dim: int, hidden: int = 64, cap: float = 10.0) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.cap = float(cap)
        self.net = nn.Sequential(nn.Linear(HAZARD_F + self.embed_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, len(DIRECTIONS)))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: Tensor, embedding: Optional[Tensor]) -> Tensor:
        """Logit deltas ``[B, K, 2]`` for (left, right), soft-capped at ``cap``."""
        if features.shape[-1] != HAZARD_F:
            raise ValueError(f"hazard features must end in {HAZARD_F}")
        if embedding is None:
            embedding = features.new_zeros(*features.shape[:-1], self.embed_dim)
        if embedding.shape[:-1] != features.shape[:-1] or embedding.shape[-1] != self.embed_dim:
            raise ValueError("driver embedding does not match the hazard features")
        raw = self.net(torch.cat((features, embedding.to(features.dtype)), -1))
        return self.cap * torch.tanh(raw / self.cap)


def hazard_lane_logits(module: HazardLaneModel, kin: Tensor, present: Tensor, learned: Tensor,
                       lane_base: Tensor, physics: PhysicsConfig,
                       driver_params: Optional[Mapping[str, Tensor]],
                       driver_embedding: Optional[Tensor],
                       lane_age: Optional[Tensor]) -> tuple[Tensor, Tensor]:
    """Absolute-lane logits ``[B, K, 3]`` and the per-direction deltas ``[B, K, 2]``.

    Only ``learned`` slots receive a delta; masked (infeasible) targets and the
    mandatory end-of-lane merge keep their physics logit.
    """
    features, priors = hazard_features(kin, present, lane_base, physics, driver_params,
                                       lane_age)
    delta = module(features, driver_embedding)
    s = kin[..., 0]
    lane = lane_of(kin)
    mandatory = ((lane == physics.merge_lane)
                 & (s >= physics.merge_zone_s1_m - MANDATORY_MERGE_WINDOW_M)
                 & (s <= physics.merge_zone_s1_m))
    logits = lane_base
    for j, offset in enumerate(DIRECTIONS):
        target = lane + offset
        valid = (target >= 0) & (target <= 2)
        target_c = target.clamp(0, 2).unsqueeze(-1)
        prior = priors[..., j]
        forced = mandatory & (target == physics.merge_target_lane)
        apply = valid & learned & (prior > MASKED_LOGIT) & ~forced
        current = logits.gather(-1, target_c).squeeze(-1)
        value = torch.where(apply, prior + delta[..., j], current)
        logits = logits.scatter(-1, target_c, value.unsqueeze(-1))
    return logits, delta


def relative_logits(lane_logits: Tensor, lane: Tensor) -> Tensor:
    """Absolute ``[..., 3]`` lane logits -> relative (stay, left, right) logits.

    A non-existent direction (left of lane 0, right of lane 2) gets ``MASKED_LOGIT``.
    """
    lane = lane.long().clamp(0, 2)
    stay = lane_logits.gather(-1, lane.unsqueeze(-1)).squeeze(-1)
    out = [stay]
    for offset in DIRECTIONS:
        target = lane + offset
        valid = (target >= 0) & (target <= 2)
        value = lane_logits.gather(-1, target.clamp(0, 2).unsqueeze(-1)).squeeze(-1)
        out.append(torch.where(valid, value, torch.full_like(value, MASKED_LOGIT)))
    return torch.stack(out, -1)
