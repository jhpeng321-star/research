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
    "intent",           # ramp car's merge intent: lane-v5 merge head's P(merge) (round 10b-12 default),
                        # R13A ``follower_courtesy_intent='pivot'`` the pivot clearance CDF at t, 'none' 0
    "has_leader",       # follower has a lane-1 leader
    "gap_leader",       # asinh(follower bumper gap to its lane-1 leader / 10 m) (0 without leader)
    "closing_leader",   # (v_follower - v_leader) / 5 m/s (0 without leader)
)
COURTESY_PAIR_F = len(COURTESY_PAIR_FIELDS)
#: R11 (``follower_courtesy_version = 2``) inputs APPENDED to the round-10b ones (so a round-10b head
#: initialises a version-2 head exactly by zero-expanding its first layer):
COURTESY_V2_FIELDS = (
    "a_ramp",           # ramp car's acceleration in its state at t (kin[..., 5]: the acceleration that
                        # produced its current state; recorded backward difference / applied rollout accel) / 3
    "a_ramp_prev",      # the same one step earlier (short motion-history slot t-1; 0 when unobserved) / 3
    "zone_wait",        # asinh(t_zone / 10 s): time since the ramp car entered the merge zone (carried,
                        # causal; ``courtesy_zone_s``) - replaces the dead lane-age wait of round 10b
)
COURTESY_V2_F = COURTESY_PAIR_F + len(COURTESY_V2_FIELDS)
#: Carried state key of the R11 merge-zone time (s, including the current state).
ZONE_STATE_KEY = "courtesy_zone_s"


def n_pair_inputs(version: int) -> int:
    """Number of pair inputs of a courtesy head of ``version`` (1 = round 10b, 2 = round 11)."""
    if int(version) not in (1, 2):
        raise ValueError("follower_courtesy_version must be 1 or 2")
    return COURTESY_PAIR_F if int(version) == 1 else COURTESY_V2_F


def _lane(kin: Tensor) -> Tensor:
    onehot = kin[..., 1:4]
    lane = onehot.argmax(-1)
    return torch.where(onehot.sum(-1) > 0, lane, torch.full_like(lane, -1))


def in_merge_zone(kin: Tensor, present: Tensor, physics) -> Tensor:
    """Ramp-lane cars inside the geometric merge zone ``[merge_zone_s0_m, merge_zone_s1_m]`` (the
    acceleration lane, where a merge is physically possible; road geometry, not a behaviour window)."""
    s = kin[..., 0]
    return (present.bool() & (_lane(kin) == int(getattr(physics, "merge_lane", 2)))
            & (s >= float(getattr(physics, "merge_zone_s0_m", 0.0)))
            & (s <= float(getattr(physics, "merge_zone_s1_m", 75.44))))


def update_zone_time(prev: Optional[Tensor], kin: Tensor, present: Tensor, physics) -> Tensor:
    """R11 merge-zone time ``[B, K]`` (s) INCLUDING the state ``kin``: ``+dt`` while the car is in
    the merge zone, 0 outside it / absent (a discrete causal counter; no autograd graph)."""
    with torch.no_grad():
        kin = kin.detach()
        base = (torch.zeros(kin.shape[:-1], dtype=torch.float32, device=kin.device)
                if prev is None else prev.detach().float())
        inside = in_merge_zone(kin, present, physics)
        return torch.where(inside, base + float(getattr(physics, "dt_s", 0.5)),
                           torch.zeros_like(base))


def initial_zone_time(present: Tensor, kin: Tensor, physics, driver_history: Optional[Tensor],
                      driver_history_mask: Optional[Tensor], stride_steps: int) -> Tensor:
    """Merge-zone time at a rollout / replay start, INCLUDING the t0 state.

    Pre-t0 part from the observed long-history samples strictly before t0 (channels v/30, lane
    one-hot 2:5, displacement/250 relative to t0 at index 6 - the conventions of
    :func:`wms.model.merge_wait.initial_merge_wait`): ``+stride * dt`` per sample in the zone, reset
    outside it / unseen.  Without a long history every car starts at 0.  Then the t0 state is applied.
    """
    zone = torch.zeros(present.shape, dtype=torch.float32, device=present.device)
    if driver_history is not None and driver_history_mask is not None \
            and driver_history.shape[-1] >= 9 and driver_history.shape[-2] > 1:
        period = float(getattr(physics, "dt_s", 0.5)) * float(stride_steps)
        s0 = float(getattr(physics, "merge_zone_s0_m", 0.0))
        s1 = float(getattr(physics, "merge_zone_s1_m", 75.44))
        with torch.no_grad():
            hist = driver_history[..., :-1, :].float()
            seen_h = driver_history_mask[..., :-1].bool()
            s_t0 = kin[..., 0].detach().float()
            for h in range(hist.shape[-2]):
                lane = hist[..., h, 2:5].argmax(-1)
                s = s_t0 + hist[..., h, 6] * 250.0
                inside = (seen_h[..., h] & (lane == int(getattr(physics, "merge_lane", 2)))
                          & (s >= s0) & (s <= s1))
                zone = torch.where(inside, zone + period, torch.zeros_like(zone))
    return update_zone_time(zone, kin, present, physics)


def follower_courtesy_pairs(kin: Tensor, present: Tensor, physics, *,
                            merge_wait_s: Optional[Tensor] = None,
                            lane_age: Optional[Tensor] = None,
                            merge_prob: Optional[Tensor] = None,
                            version: int = 1,
                            zone_time_s: Optional[Tensor] = None,
                            history: Optional[Tensor] = None,
                            history_mask: Optional[Tensor] = None,
                            state_grad: bool = False) -> dict[str, Tensor]:
    """Pair every lane-1 follower with its ramp partner; return features ``[B, K, F]``.

    Returns ``features`` (zeros where there is no partner), ``has_partner [B, K]`` and
    ``partner [B, K]`` (slot index of the ramp car; meaningless where ``has_partner`` is false).
    Only current-state quantities enter (``merge_wait_s`` is the carried waiting time, a function
    of the past; ``merge_prob`` is detached).

    ``version=2`` (R11) appends the ramp car's acceleration at t and t-1 (``history`` slot -2 of the
    short motion history, channel 1 = accel/3; zero where unobserved) and its merge-zone time
    ``zone_time_s`` (carried; missing -> the current state only, :func:`update_zone_time`).

    ``state_grad`` (R13A, default False = round 12: every feature detached) keeps the continuous
    derivative of the gathered pair features with respect to the state ``kin`` (and a supplied
    differentiable ``merge_prob``).  The discrete pairing (partner / leader indices, presence and
    lane masks) stays an explicit, non-differentiable selection; the carried counters (wait, zone
    time) and the short-history slot t-1 stay detached.
    """
    n_pair_inputs(version)
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
    if merge_prob is None:
        intent_all = zero
    else:
        intent_all = torch.as_tensor(merge_prob, dtype=s.dtype, device=s.device)
        intent_all = intent_all if state_grad else intent_all.detach()
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
    if int(version) >= 2:
        acc = kin[..., 5]
        if history is not None and history_mask is not None and history.shape[-2] >= 2:
            prev_all = torch.where(history_mask[..., -2].bool(),
                                   history[..., -2, 1].detach().to(s.dtype) * 3.0, zero)
        else:
            prev_all = zero
        zone_all = (update_zone_time(None, kin, present, physics) if zone_time_s is None
                    else torch.as_tensor(zone_time_s, dtype=s.dtype, device=s.device))
        feats = torch.cat((feats, torch.stack((
            at(acc, partner) / 3.0,
            at(prev_all, partner) / 3.0,
            torch.asinh(at(zone_all, partner).clamp_min(0.0) / 10.0),
        ), -1)), -1)
    feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    feats = torch.where(has_partner.unsqueeze(-1), feats, torch.zeros_like(feats))
    return {"features": feats if state_grad else feats.detach(), "has_partner": has_partner,
            "partner": partner}


#: R13A courtesy ``intent`` input sources (``model.follower_courtesy_intent``).
INTENT_SOURCES = ("lane_v5", "pivot", "none")


def pivot_clearance_intent(pivot_net: nn.Module, kin: Tensor, present: Tensor, physics,
                           is_cav: Optional[Tensor] = None) -> Tensor:
    """R13A causal courtesy intent ``[B, K]`` from the first-passage pivot, at the step-START state.

    ``sigmoid(c_t / tau)`` with ``c_t`` the pivot's own boundary (min log clearance ratio to its
    learned two-sided gap requirement) evaluated on the CURRENT state ``kin`` (before this step's
    longitudinal update, so there is no acceleration/intent algebraic loop) and ``tau`` its
    persistent-latent scale: the fraction of the pivot's population latent ``z ~ Logistic(0, 1)``
    for which ``c_t + tau z >= 0``.  Deterministic and latent-free (the drawn latent and the hybrid
    per-decision noise never enter pre-merge motion).  Exact limits: ineligible / vetoed (``c = -inf``)
    -> 0, unconstrained (``c = +inf``) -> 1.  HDV ramp cars use the pivot's decision geometry (CAVs
    excluded from interval reservation, as in :meth:`WmSModel.pivot_decision_geometry`); a CAV ramp
    car is scored on its own neighbours with the same physical veto (overlap / invalid scene), since
    its lane decision is commanded, not drawn.  The pivot's parameters enter DETACHED (the courtesy
    losses never train the pivot); the state derivative is kept.
    """
    from torch.func import functional_call

    from . import merge_pivot as pivot

    present = present.bool()
    cav = torch.zeros_like(present) if is_cav is None else is_cav.bool()
    geo = pivot.geometry(kin, present, physics, decision_mask=~cav)
    own_physical = geo["ramp"] & geo["scene_valid"] & ~geo["overlap"]
    geo = {**geo, "eligible": torch.where(cav, own_physical, geo["eligible"])}
    params = {name: value.detach() for name, value in pivot_net.named_parameters()}
    c = functional_call(pivot_net, params, (geo,))
    tau = torch.nn.functional.softplus(params["tau_raw"]).to(c.dtype)
    finite = torch.isfinite(c)
    safe = torch.where(finite, c, torch.zeros_like(c))
    limit = (c == torch.inf).to(c.dtype)
    return torch.where(finite, torch.sigmoid(safe / tau), limit)


class FollowerCourtesyHead(nn.Module):
    """Pair MLP -> one acceleration offset (m/s^2) per follower; zero last layer (exact no-op)."""

    def __init__(self, hidden: int = 32, embed_dim: int = 0, version: int = 1) -> None:
        super().__init__()
        if int(hidden) < 1 or int(embed_dim) < 0:
            raise ValueError("follower courtesy needs hidden >= 1 and embed_dim >= 0")
        self.version = int(version)
        self.n_pair = n_pair_inputs(self.version)
        self.n_in = self.n_pair + int(embed_dim)
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


def expand_head_inputs(source: dict, own: dict, prefix: str = "follower_courtesy.") -> dict:
    """Zero-expand a narrower stored first layer (round-10b head, 10 pair inputs) to a version-2
    head (13 pair inputs; the new inputs are APPENDED, so the expanded head computes exactly the
    stored function).  Only the plain pair-input heads (no embedding ablation) are expanded."""
    key = prefix + "net.0.weight"
    stored, want = source.get(key), own.get(key)
    if stored is None or want is None or tuple(stored.shape) == tuple(want.shape):
        return source
    if (stored.ndim != 2 or want.ndim != 2 or stored.shape[0] != want.shape[0]
            or stored.shape[1] != COURTESY_PAIR_F or want.shape[1] != COURTESY_V2_F):
        return source
    out = dict(source)
    grown = torch.zeros(tuple(want.shape), dtype=stored.dtype, device=stored.device)
    grown[:, :COURTESY_PAIR_F] = stored
    out[key] = grown
    return out


__all__ = ["COURTESY_PAIR_F", "COURTESY_PAIR_FIELDS", "COURTESY_V2_F", "COURTESY_V2_FIELDS",
           "FollowerCourtesyHead", "INTENT_SOURCES", "ZONE_STATE_KEY", "expand_head_inputs",
           "follower_courtesy_pairs", "in_merge_zone", "initial_zone_time", "n_pair_inputs",
           "pivot_clearance_intent", "update_zone_time"]
