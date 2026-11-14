"""Differentiable insertion-blocking surrogate for the imagined entry queue (SRCGRAD_NOTES.md).

The imagined entry queue (``TorchEntryProcessV2`` backlog) is a discrete count: the reward's
``source_wait = dt * backlog`` therefore carries no gradient to the actor, which then learns that
keeping vehicles OUT of the network (slow CAVs near an entrance block insertion) is free.  This
module gives that term a gradient while keeping its VALUE exact (straight-through):

    entry_queue = backlog + q_sur - q_sur.detach()

    q_sur[b]   = sum_e D[b, e] * p_block[b, e]                       (vehicles)
    D[b, e]    = pending[b, e] (before this step's insertion) + rate[b, e] * dt
    p_block    = p_lane(0) * p_lane(1)   mainline (an arrival is blocked only if BOTH lanes fail)
                 p_lane(2)               ramp
    p_lane(l)  = 1 - sigmoid((g - req) / tau) * Phi((g - gap_mean) / sqrt(gap_std^2 + tau^2))
    g          = soft-min over present vehicles on lane l of (s - length - entry_s[l])
                 (softmax(-gap / tau_min) weights; also give the soft rear speed v_lead)
    req        = min_gap + headway * v_new + relu(v_new - v_lead)^2 / (2 * decel)    (EntryConfig)
    v_new      = relu(speed_mean[e] * mean speed of the n_nearby rearmost vehicles on l)

i.e. the soft version of ``TorchEntryProcessV2._try_lane``: the SUMO-like safe-gap test
``safe_to_insert`` AND the sampled per-arrival required gap ``u_gap ~ N(gap_mean, gap_std)`` (whose
exact acceptance probability is the normal CDF; ``tau`` widens it so the gradient stays smooth).
``d source_wait / d p_block[e] = dt * D[e]``: losing one service opportunity at an entrance
delays every vehicle that is waiting there (the backlog plus the expected arrival of this step)
by one step -- the first-order queueing cost of a lost insertion slot.  The expected arrival
``rate * dt`` replaces the realised 0/1 arrival indicator (its conditional expectation; the
blocking probability does not depend on whether an arrival happens to be due), which keeps the
gradient dense.  An empty lane gives ``g = +large`` and ``p_lane = 0`` (always free).

ENTRYFIX ``insert_rule = "sumo_like"`` (POLICYV5; ``cfg.insert_rule``, "v2" above unchanged): SUMO
departSpeed="max" insertion lowers the entrant's speed to the safe insertion speed and refuses ONLY
when that speed is below ``sumo_min_speed_mps`` (0: when the bumper gap is inside the vType minGap);
there is no speed ratio and no sampled required gap.  The surrogate is the soft version of
``TorchEntryProcessV2._try_lane_sumo`` per vehicle class ``vt`` (weights = the entrance's observed
class mix, as the process draws ``vt``):

    p_lane(l, vt) = 1 - sigmoid((g - length[vt] - eps - req[vt]) / tau)
    req[vt]       = minGap[vt] + relu(tau_s[vt] v_min + (v_min^2 - v_lead^2) / (2 decel[vt]))
    p_block[e]    = sum_vt share[e, vt] * prod_{l in e} p_lane(l, vt)

(``eps`` = SUMO's departPos="base" offset; ``v_min = 0`` gives ``req = minGap``.)  Mainline arrivals
try both lanes ("any" lane mode, largest gap first), so they are refused only when both lanes are.

V6ENTRY ``insert_rule = "sumo_like_v2"``: the soft test moves to the END of the insertion step
(``bumper' = bumper + relu(v_lead) dt``, the rearmost vehicle's within-step movement) and drops the
leader's speed credit entirely (stop-speed departure), mirroring the hard rule's v2 branch.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

#: Soft-min temperature over candidate rear vehicles (metres).
SOFTMIN_TAU_M = 2.0
#: Temperature of the safe-gap sigmoid / widening of the sampled-gap CDF (metres).
BLOCK_TAU_M = 1.0
_EMPTY_GAP_M = 1.0e4
#: Entrance -> candidate lanes, as in ``TorchEntryProcessV2._insert_entrance``.
ENTRANCE_LANES = ((0, 1), (2,))


def history_tensors(process, rows: int, device, dtype) -> dict[str, Tensor]:
    """Per-row ``rate`` / ``speed_mean`` / ``gap_mean`` / ``gap_std`` ``[B, 2]`` of an entry process
    (+ ``vtype_share`` ``[B, 2, N_VTYPES]``, read only by the ``sumo_like`` surrogate)."""
    hists = [process._hist(b) for b in range(rows)]
    out = {}
    for name in ("rate", "speed_mean", "gap_mean", "gap_std", "vtype_share"):
        out[name] = torch.as_tensor(np.stack([np.asarray(getattr(h, name), np.float64)
                                              for h in hists]), dtype=dtype, device=device)
    return out


def lane_block_probability(state, present: Tensor, lane_index: int, entry_s: float,
                           speed_ratio: Tensor, gap_mean: Tensor, gap_std: Tensor, cfg) -> Tensor:
    """Soft probability ``[B]`` that an arrival trying lane ``lane_index`` is refused."""
    s, v = state["s_m"], state["v_mps"]
    length = state.get("length_m")
    if length is None:
        length = torch.full_like(s, 5.0)
    in_lane = present & (state["lane"] == lane_index)
    any_vehicle = in_lane.any(-1)
    gap = torch.where(in_lane, s - length - entry_s, torch.full_like(s, _EMPTY_GAP_M))
    weight = torch.softmax(-gap / SOFTMIN_TAU_M, dim=-1)
    weight = torch.where(in_lane, weight, torch.zeros_like(weight))
    weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-12)
    soft_gap = torch.where(any_vehicle, (weight * gap).sum(-1), torch.full_like(s[:, 0], _EMPTY_GAP_M))
    v_lead = (weight * torch.where(in_lane, v, torch.zeros_like(v))).sum(-1)
    # Local speed = mean speed of the n_nearby rearmost vehicles (hard selection, as the process).
    n_near = max(1, min(int(cfg.n_nearby), s.shape[-1]))
    order = torch.topk(torch.where(in_lane, s, torch.full_like(s, math.inf)), n_near, dim=-1,
                       largest=False).indices
    near_mask = in_lane.gather(-1, order).to(v.dtype)
    local = (v.gather(-1, order) * near_mask).sum(-1) / near_mask.sum(-1).clamp_min(1.0)
    local = torch.where(any_vehicle, local, torch.full_like(local, float(cfg.default_speed_mps)))
    v_new = torch.relu(speed_ratio * local)
    required = (cfg.min_gap_m + cfg.headway_s * v_new
                + torch.relu(v_new - v_lead).square() / (2.0 * cfg.decel_mps2))
    p_safe = torch.sigmoid((soft_gap - required) / BLOCK_TAU_M)
    spread = torch.sqrt(gap_std.clamp_min(0.0).square() + BLOCK_TAU_M ** 2)
    p_gap = 0.5 * (1.0 + torch.erf((soft_gap - gap_mean) / (spread * math.sqrt(2.0))))
    blocked = 1.0 - p_safe * p_gap
    return torch.where(any_vehicle, blocked, torch.zeros_like(blocked))


def sumo_lane_block_probability(state, present: Tensor, lane_index: int, entry_s: float,
                                vtype: int, cfg, dt_s: float = 0.5) -> Tensor:
    """``sumo_like`` rule: soft probability ``[B]`` that a class-``vtype`` arrival is refused on
    ``lane_index`` (bumper gap behind the lane's rearmost vehicle inside the SUMO requirement).

    ``sumo_like_v2`` (V6ENTRY): the soft test of the end-of-step gap ``bumper + relu(v_lead)
    dt_s`` against the textbook vType minGap alone (the entrant departs at the stop speed; the
    leader gets no speed credit), mirroring ``TorchEntryProcessV2._try_lane_sumo``'s v2 branch."""
    from wms.model.entries import (SUMO_POSITION_EPS_M, SUMO_V2_DECEL_MPS2, SUMO_V2_MIN_GAP_M,
                                   SUMO_V2_TAU_S, VTYPE_DECEL_MPS2, VTYPE_LENGTH_M,
                                   VTYPE_MIN_GAP_M, VTYPE_TAU_S)

    rule_v2 = str(getattr(cfg, "insert_rule", "v2")) == "sumo_like_v2"

    s, v = state["s_m"], state["v_mps"]
    length = state.get("length_m")
    if length is None:
        length = torch.full_like(s, 5.0)
    in_lane = present & (state["lane"] == lane_index)
    any_vehicle = in_lane.any(-1)
    gap = torch.where(in_lane, s - length - entry_s, torch.full_like(s, _EMPTY_GAP_M))
    weight = torch.softmax(-gap / SOFTMIN_TAU_M, dim=-1)
    weight = torch.where(in_lane, weight, torch.zeros_like(weight))
    weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-12)
    soft_gap = torch.where(any_vehicle, (weight * gap).sum(-1), torch.full_like(s[:, 0], _EMPTY_GAP_M))
    v_lead = (weight * torch.where(in_lane, v, torch.zeros_like(v))).sum(-1)
    bumper = soft_gap - float(VTYPE_LENGTH_M[vtype]) - SUMO_POSITION_EPS_M
    v_min = max(0.0, float(getattr(cfg, "sumo_min_speed_mps", 0.0)))
    if rule_v2:
        bumper = bumper + torch.relu(v_lead) * float(dt_s)
        v_lead = torch.zeros_like(v_lead)
    tau_vt = float(SUMO_V2_TAU_S[vtype] if rule_v2 else VTYPE_TAU_S[vtype])
    decel = max(float(SUMO_V2_DECEL_MPS2[vtype] if rule_v2 else VTYPE_DECEL_MPS2[vtype]), 1.0e-6)
    min_gap_vt = float(SUMO_V2_MIN_GAP_M[vtype] if rule_v2 else VTYPE_MIN_GAP_M[vtype])
    required = min_gap_vt + torch.relu(
        tau_vt * v_min + (v_min ** 2 - v_lead.square()) / (2.0 * decel))
    blocked = 1.0 - torch.sigmoid((bumper - required) / BLOCK_TAU_M)
    return torch.where(any_vehicle, blocked, torch.zeros_like(blocked))


def _sumo_entrance_block_probability(state, present: Tensor, hist: dict[str, Tensor],
                                     entry_s, cfg, dt_s: float = 0.5) -> Tensor:
    from wms.model.entries import N_VTYPES

    if getattr(cfg, "sumo_lane_mode", "any") != "any":
        raise ValueError("the sumo_like queue surrogate models sumo_lane_mode='any' only")
    share = hist.get("vtype_share")
    rows = state["s_m"].shape[0]
    if share is None:
        share = torch.zeros(rows, len(ENTRANCE_LANES), N_VTYPES, dtype=state["s_m"].dtype,
                            device=state["s_m"].device)
    share = share.to(dtype=state["s_m"].dtype)
    total = share.sum(-1, keepdim=True)
    # As TorchEntryProcessV2: class drawn from the normalised share; no observed class -> class 0.
    fallback = torch.zeros_like(share)
    fallback[..., 0] = 1.0
    share = torch.where(total > 0, share / total.clamp_min(1e-12), fallback)
    out = []
    for e, lanes in enumerate(ENTRANCE_LANES):
        p_e = None
        for vt in range(share.shape[-1]):
            p = None
            for ln in lanes:
                lane_p = sumo_lane_block_probability(state, present, ln, float(entry_s[ln]),
                                                     vt, cfg, dt_s)
                p = lane_p if p is None else p * lane_p
            term = share[:, e, vt] * p
            p_e = term if p_e is None else p_e + term
        out.append(p_e)
    return torch.stack(out, -1)


def entrance_block_probability(state, present: Tensor, hist: dict[str, Tensor], entry_s, cfg,
                               dt_s: float = 0.5) -> Tensor:
    """``p_block`` ``[B, 2]`` (mainline, ramp) from the pre-insertion state."""
    if getattr(cfg, "insert_rule", "v2") in ("sumo_like", "sumo_like_v2"):
        return _sumo_entrance_block_probability(state, present, hist, entry_s, cfg, dt_s)
    out = []
    for e, lanes in enumerate(ENTRANCE_LANES):
        p = None
        for ln in lanes:
            lane_p = lane_block_probability(state, present, ln, float(entry_s[ln]),
                                            hist["speed_mean"][:, e], hist["gap_mean"][:, e],
                                            hist["gap_std"][:, e], cfg)
            p = lane_p if p is None else p * lane_p
        out.append(p)
    return torch.stack(out, -1)


def queue_surrogate(state, present: Tensor, pending_before: Tensor, hist: dict[str, Tensor],
                    entry_s, cfg, dt: float) -> tuple[Tensor, Tensor]:
    """``(q_sur [B], p_block [B, 2])``; ``q_sur = sum_e (pending_e + rate_e * dt) * p_block_e``."""
    p_block = entrance_block_probability(state, present, hist, entry_s, cfg, dt)
    demand = pending_before.to(p_block.dtype) + hist["rate"] * dt
    return (demand * p_block).sum(-1), p_block


def straight_through(exact: Tensor, surrogate: Tensor) -> Tensor:
    """Value ``exact``, gradient of ``surrogate``."""
    return exact + (surrogate - surrogate.detach())
