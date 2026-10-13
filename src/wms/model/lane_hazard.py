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

LANEFIX switches (2026-10-14, LANEFIX_NOTES.md; every default reproduces the code above):

* ``features="v2"`` appends ranking features (:func:`hazard_features_v2_extra`): per lane
  (current and both targets) the mean speed of the cars within 100 m ahead relative to the
  driver's desired speed v0, the leader's acceleration (speed trend) and the density within
  100 / 200 m, plus the time the car has been blocked behind a slower leader (a recurrent
  count carried by the rollout, see :func:`blocked_indicator`).  All from the current
  (rollout or recorded) state and the car's own past; no future information.
* ``merge_zone="full"``: the hazard owns the ramp merge 2 -> 1 from the start of the
  acceleration lane (constant base logit :data:`MERGE_HAZARD_PRIOR_LOGIT` + learned delta
  instead of the physics urgency ramp); the physics merge is forced (logit >=
  :data:`FORCED_MERGE_LOGIT`) only in the last :data:`MERGE_FULL_FORCE_WINDOW_M`.
* ``offsets`` (overtake, return, merge): constant per-category logit offsets (calibration).
* ``detach_delta``: the learned delta enters the returned logits detached (forward value
  unchanged) - ``loss.lane_hazard_decouple``: the rollout lane CE / flip-rate guard then
  never reach the hazard parameters; the teacher-forced event loss stays its only objective.
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
#: LANEFIX v2 extras per lane: lookahead speed deficit vs v0, leader acceleration,
#: density within 100 m and within 200 m.
_LANE_STATS_F = 4
#: LANEFIX v2 extras: both target lanes + current lane stats + blocked time + blocked flag.
HAZARD_V2_EXTRA_F = 3 * _LANE_STATS_F + 2
#: Width of the v2 features (v1 + extras).
HAZARD_F_V2 = HAZARD_F + HAZARD_V2_EXTRA_F
HAZARD_FEATURE_SETS = ("v1", "v2")
#: Lookahead (m) of the lane mean-speed feature; density ranges (m); jam spacing (m/veh)
#: normalising the densities to ~1 at jam density.
LOOKAHEAD_SPEED_M = 100.0
DENSITY_RANGES_M = (100.0, 200.0)
JAM_SPACING_M = 7.5
#: A car is 'blocked' while its same-lane leader is within BLOCKED_GAP_M and slower than
#: the car's desired speed v0 minus BLOCKED_SPEED_MARGIN_MPS.
BLOCKED_GAP_M = 50.0
BLOCKED_SPEED_MARGIN_MPS = 2.0
#: The blocked-time feature saturates at 60 s.
BLOCKED_CAP_S = 60.0
#: Merge-zone ownership (LANEFIX): legacy (physics urgency ramp + 25 m mandatory window) or
#: the hazard owns ramp merges from the start of the acceleration lane.
MERGE_HAZARD_ZONES = ("forced_late", "full")
#: ``merge_zone='full'``: base logit of the ramp merge (vs stay 0) before the learned delta,
#: ~ the TRAIN per-step merge rate of lane-2 rows in the zone (181 / 10994 = 1.6 %, V3s60
#: teacher-forced diagnostic on TRAIN windows) so a zero hazard merges at the base rate.
MERGE_HAZARD_PRIOR_LOGIT = -4.0
#: ``merge_zone='full'``: forced-merge window before the acceleration-lane end and the
#: minimum logit of a feasible merge there (p(stay) ~ 3e-4 per step).
MERGE_FULL_FORCE_WINDOW_M = 5.0
FORCED_MERGE_LOGIT = 8.0
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


def _driver_v0(v: Tensor, physics: PhysicsConfig,
               driver_params: Optional[Mapping[str, Tensor]]) -> Tensor:
    return (torch.full_like(v, float(physics.v0_mps)) if driver_params is None
            else driver_params["v0"].to(v.dtype)).clamp_min(1.0)


def blocked_indicator(kin: Tensor, present: Tensor, physics: PhysicsConfig,
                      driver_params: Optional[Mapping[str, Tensor]]) -> Tensor:
    """``[B, K]`` bool: the car's same-lane leader (a vehicle, not the lane end) is within
    :data:`BLOCKED_GAP_M` and slower than the car's v0 minus :data:`BLOCKED_SPEED_MARGIN_MPS`."""
    with torch.no_grad():
        v0 = _driver_v0(kin[..., 4], physics, driver_params)
        gap, lead_v = lane_neighbour(kin, present, lane_of(kin), physics, ahead=True)
        has = gap < float(physics.free_road_gap_m) - 1.0e-3
        return (present.bool() & has & (gap <= BLOCKED_GAP_M)
                & (lead_v < v0 - BLOCKED_SPEED_MARGIN_MPS))


def update_blocked(previous: Tensor, indicator: Tensor) -> Tensor:
    """Consecutive blocked steps including the current one (0 when not blocked now)."""
    return torch.where(indicator, previous.detach() + 1.0, torch.zeros_like(previous))


def initial_lane_blocked(present: Tensor, dtype: torch.dtype,
                         driver_history: Optional[Tensor],
                         driver_history_mask: Optional[Tensor],
                         v0: Tensor, stride_steps: int) -> Tensor:
    """Blocked steps before t0 ``[B, K]`` from the observed pre-t0 long driver history.

    Uses the long-history channels (v/30, gap/100, leader dv/30, leader-seen flag) of every
    sample strictly before t0 (the last sample is t0 itself, whose state the rollout's first
    step scores): the trailing run of blocked samples times ``stride_steps``.  Without a
    long history (or with fewer than two samples) every car starts at 0.
    """
    zeros = torch.zeros(present.shape, dtype=dtype, device=present.device)
    if driver_history is None or driver_history_mask is None \
            or driver_history.shape[-1] < 9 or driver_history.shape[-2] <= 1:
        return zeros
    with torch.no_grad():
        hist = driver_history[..., :-1, :].to(dtype)
        seen = driver_history_mask[..., :-1].bool()
        v = hist[..., 0] * 30.0
        gap = hist[..., 5] * 100.0
        lead_v = v + hist[..., 7] * 30.0
        has = hist[..., 8] > 0.5
        blocked = (seen & has & (gap <= BLOCKED_GAP_M)
                   & (lead_v < v0.detach().to(dtype).unsqueeze(-1) - BLOCKED_SPEED_MARGIN_MPS))
        run = torch.cumprod(blocked.flip(-1).to(dtype), -1).sum(-1)
        return torch.where(present.bool(), run * float(stride_steps), zeros)


def _ahead_context(kin: Tensor, present: Tensor) -> dict:
    """Pairwise tensors shared by the per-lane :func:`lane_ahead_stats` calls."""
    s = kin[..., 0]
    ds = s.unsqueeze(-2) - s.unsqueeze(-1)          # [B,i,j] = s_j - s_i
    return {"ds": ds, "ahead": present.unsqueeze(-2) & (ds > 0),
            "lane": lane_of(kin).unsqueeze(-2), "within": [ds <= r for r in DENSITY_RANGES_M],
            "look": ds <= LOOKAHEAD_SPEED_M}


def lane_ahead_stats(kin: Tensor, present: Tensor, query_lane: Tensor, v0: Tensor,
                     context: Optional[dict] = None) -> Tensor:
    """Per-lane ranking features ``[B, K, 4]`` of the traffic ahead in ``query_lane``.

    (mean speed of the cars within :data:`LOOKAHEAD_SPEED_M` ahead, capped at v0, minus v0)
    / 10 (0 on a free lane), the nearest leader's acceleration / 3 (speed trend; 0 without
    one) and the number of cars within each :data:`DENSITY_RANGES_M` ahead times
    :data:`JAM_SPACING_M` / range (~1 at jam density).  Current state only.
    """
    ctx = context if context is not None else _ahead_context(kin, present)
    v, a = kin[..., 4], kin[..., 5]
    ds = ctx["ds"]
    ahead = ctx["ahead"] & (ctx["lane"] == query_lane.unsqueeze(-1))
    nearest, index = torch.where(ahead, ds, torch.full_like(ds, float("inf"))).min(-1)
    lead_a = torch.where(torch.isfinite(nearest), torch.gather(a, -1, index),
                         torch.zeros_like(a))
    window = ahead & ctx["look"]
    count = window.sum(-1)
    speed_sum = torch.matmul(window.to(v.dtype), v.unsqueeze(-1)).squeeze(-1)
    mean_v = torch.where(count > 0, speed_sum / count.clamp_min(1).to(v.dtype), v0)
    out = [(torch.minimum(mean_v, v0) - v0) / 10.0, lead_a / 3.0]
    for rng, within in zip(DENSITY_RANGES_M, ctx["within"]):
        n = (ahead & within).sum(-1).to(v.dtype)
        out.append(n * JAM_SPACING_M / rng)
    return torch.stack(out, -1)


def hazard_features_v2_extra(kin: Tensor, present: Tensor, physics: PhysicsConfig,
                             driver_params: Optional[Mapping[str, Tensor]],
                             lane_blocked: Optional[Tensor]) -> Tensor:
    """LANEFIX v2 extras ``[B, K, HAZARD_V2_EXTRA_F]``: left/right target-lane and current-lane
    :func:`lane_ahead_stats` and the blocked time (s, capped, / 10) with its 'blocked now' flag.

    ``lane_blocked`` is the consecutive blocked-step count INCLUDING the current state
    (:func:`update_blocked`); ``None`` means 0.
    """
    v = kin[..., 4]
    v0 = _driver_v0(v, physics, driver_params)
    lane = lane_of(kin)
    context = _ahead_context(kin, present)
    chunks = []
    for offset in DIRECTIONS:
        target = lane + offset
        valid = ((target >= 0) & (target <= 2)).to(kin.dtype).unsqueeze(-1)
        chunks.append(valid * lane_ahead_stats(kin, present, target.clamp(0, 2), v0, context))
    chunks.append(lane_ahead_stats(kin, present, lane, v0, context))
    blocked = torch.zeros_like(v) if lane_blocked is None else lane_blocked.to(v.dtype)
    blocked_s = (blocked * physics.dt_s).clamp(0.0, BLOCKED_CAP_S) / 10.0
    chunks.append(torch.stack((blocked_s, (blocked > 0).to(v.dtype)), -1))
    return torch.cat(chunks, -1)


def hazard_feasible(lane_base: Tensor, kin: Tensor) -> Tensor:
    """``[B, K, 3]`` bool: lanes the physics mask allows (the current lane always)."""
    current = torch.nn.functional.one_hot(lane_of(kin), 3).bool()
    return (lane_base > MASKED_LOGIT) | current


def sample_lane_choice(logits: Tensor, feasible: Tensor, sample: Tensor,
                       uniform: Tensor) -> Tensor:
    """Lane choice ``[B, K]``: inverse-CDF draw from the masked categorical on ``sample``
    slots, argmax elsewhere.

    The categorical is ``softmax(logits)`` over the feasible lanes (infeasible = -inf), lanes
    ordered 0, 1, 2; ``uniform [B, K]`` in [0, 1) is the draw (one per slot, so rows sharing
    draws - common random numbers - choose monotonically coupled lanes).  A draw that lands
    on an infeasible lane through float rounding falls back to the argmax.
    """
    argmax = logits.argmax(-1)
    masked = torch.where(feasible, logits, torch.full_like(logits, float("-inf")))
    probs = torch.softmax(masked.detach().float(), -1)
    cdf = probs.cumsum(-1)
    u = uniform.to(cdf.dtype).unsqueeze(-1)
    drawn = (u >= cdf[..., :2]).sum(-1)
    ok = feasible.gather(-1, drawn.unsqueeze(-1)).squeeze(-1)
    drawn = torch.where(ok, drawn, argmax)
    return torch.where(sample.bool(), drawn, argmax)


class HazardLaneModel(nn.Module):
    """MLP over :func:`hazard_features` + driver embedding -> bounded (left, right) deltas."""

    def __init__(self, embed_dim: int, hidden: int = 64, cap: float = 10.0,
                 n_features: int = HAZARD_F) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.cap = float(cap)
        self.n_features = int(n_features)
        self.net = nn.Sequential(nn.Linear(self.n_features + self.embed_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, len(DIRECTIONS)))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: Tensor, embedding: Optional[Tensor]) -> Tensor:
        """Logit deltas ``[B, K, 2]`` for (left, right), soft-capped at ``cap``."""
        if features.shape[-1] != self.n_features:
            raise ValueError(f"hazard features must end in {self.n_features}")
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
                       lane_age: Optional[Tensor], *, features: str = "v1",
                       lane_blocked: Optional[Tensor] = None,
                       merge_zone: str = "forced_late",
                       offsets: Optional[tuple[float, float, float]] = None,
                       detach_delta: bool = False) -> tuple[Tensor, Tensor]:
    """Absolute-lane logits ``[B, K, 3]`` and the per-direction deltas ``[B, K, 2]``.

    Only ``learned`` slots receive a delta; masked (infeasible) targets and the
    mandatory end-of-lane merge keep their physics logit.  The keyword switches are the
    LANEFIX options of the module docstring (defaults: the WMLANE A behaviour).
    """
    feats, priors = hazard_features(kin, present, lane_base, physics, driver_params,
                                    lane_age)
    if features == "v2":
        extra = hazard_features_v2_extra(kin, present, physics, driver_params, lane_blocked)
        feats = torch.cat((feats, torch.where(present.unsqueeze(-1), extra,
                                              torch.zeros_like(extra))), -1)
    delta = module(feats, driver_embedding)
    applied_delta = delta.detach() if detach_delta else delta
    full = merge_zone == "full"
    s = kin[..., 0]
    lane = lane_of(kin)
    window = MERGE_FULL_FORCE_WINDOW_M if full else MANDATORY_MERGE_WINDOW_M
    mandatory = ((lane == physics.merge_lane)
                 & (s >= physics.merge_zone_s1_m - window)
                 & (s <= physics.merge_zone_s1_m))
    use_offsets = offsets is not None and any(float(x) != 0.0 for x in offsets)
    logits = lane_base
    for j, offset in enumerate(DIRECTIONS):
        target = lane + offset
        valid = (target >= 0) & (target <= 2)
        target_c = target.clamp(0, 2).unsqueeze(-1)
        prior = priors[..., j]
        is_merge = (lane == physics.merge_lane) & (target == physics.merge_target_lane)
        forced = mandatory & is_merge
        feasible = valid & learned & (prior > MASKED_LOGIT)
        apply = feasible & ~forced
        current = logits.gather(-1, target_c).squeeze(-1)
        base = prior
        if full:
            # The hazard owns the ramp merge from the start of the acceleration lane.
            base = torch.where(is_merge, torch.full_like(prior, MERGE_HAZARD_PRIOR_LOGIT), prior)
        value = base + applied_delta[..., j]
        if use_offsets:
            ovt, ret, mrg = (float(x) for x in offsets)
            value = value + torch.where(
                (lane == 1) & (target == 0), torch.full_like(value, ovt),
                torch.where((lane == 0) & (target == 1), torch.full_like(value, ret),
                            torch.where(is_merge, torch.full_like(value, mrg),
                                        torch.zeros_like(value))))
        value = torch.where(apply, value, current)
        if full:
            # Still forced at the end: a feasible merge in the last window is taken.
            value = torch.where(feasible & forced, current.clamp_min(FORCED_MERGE_LOGIT), value)
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
