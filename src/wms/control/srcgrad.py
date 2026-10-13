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
    """Per-row ``rate`` / ``speed_mean`` / ``gap_mean`` / ``gap_std`` ``[B, 2]`` of an entry process."""
    hists = [process._hist(b) for b in range(rows)]
    out = {}
    for name in ("rate", "speed_mean", "gap_mean", "gap_std"):
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


def entrance_block_probability(state, present: Tensor, hist: dict[str, Tensor], entry_s, cfg) -> Tensor:
    """``p_block`` ``[B, 2]`` (mainline, ramp) from the pre-insertion state."""
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
    p_block = entrance_block_probability(state, present, hist, entry_s, cfg)
    demand = pending_before.to(p_block.dtype) + hist["rate"] * dt
    return (demand * p_block).sum(-1), p_block


def straight_through(exact: Tensor, surrogate: Tensor) -> Tensor:
    """Value ``exact``, gradient of ``surrogate``."""
    return exact + (surrogate - surrogate.detach())
