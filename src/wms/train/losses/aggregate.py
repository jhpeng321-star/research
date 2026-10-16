"""Aggregate-consistency loss (C4, 2026-10-13): match the RSU cell fields and reward components.

The C4 policy and its reward read AGGREGATE traffic -- the RSU own-sensor cell field
(``wms.control.rsu_field``: density per 50 m cell per lane, mean speed, queue flag) and
the team-reward components (``wms.control.reward``: on-road delay, throughput past the
merge end) -- while the per-car Huber/CE terms only score individual HDVs.  This module
compares differentiable versions of those aggregates between the rollout and the recorded
window:

* **field** (``agg_field``): per RSU cell, soft cell membership ``sigmoid((s - lo)/tau) -
  sigmoid((s - hi)/tau)`` times the (hard) lane indicator; density = soft count / cell
  length, mean speed = membership-weighted mean with a ``SPEED_LIMIT`` pseudo-count (the
  empty-cell convention of ``build_rsu_field``), queue = soft (slow AND occupied).  Fields
  are averaged over ``bin_s`` bins, normalised by per-feature scales and compared per cell.
* **reward** (``agg_reward``): per bin, on-road delay ``dt * relu(1 - v / SPEED_LIMIT)``
  (whole road and RSU zone; error per recorded vehicle-second) and throughput = soft
  crossings of ``MERGE_S1_M`` by same-vehicle transitions (error in vehicles / scale).
* **flow** (``agg_flow``, optional): soft crossings of every RSU cell boundary (all lanes
  of the cross-section) per bin -- by vehicle conservation, matching every boundary flux
  matches the inflow - outflow balance of each cell.

Recorded side: with the entry process active and the windows carrying the recorded
on-road presence (``road_present``/``road_continue``/``road_valid``, emitted by the
entries-v2 loaders), the recorded aggregate is the FULL road, including vehicles that
arrive after t0, and the predicted aggregate includes the entrants.  Otherwise both sides
use the t0 cohort only (recorded alive-since-t0 presence; padding entrant slots excluded).
Steps whose recorded future is unavailable (window padding) are masked out.

Normalisation: each component is a weighted mean over every (row, bin) of the whole
unroll (``prepare`` computes the unroll-wide weight), so the chunk contributions add up to
one horizon-invariant mean, as for the per-car terms.  Flag-gated: with
``loss.aggregate.weight = 0`` the composer never builds these terms (bit-identical).

The RSU geometry and reward constants are restated here because the train layer may not
import ``wms.control`` (import graph); ``tests/unit/train/test_wmloss_aggregate.py``
asserts they equal ``wms.control.contract`` / ``rsu_field.CELLS``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import LossContext, LossTerm, TermValue, UnrollTargets

# ---- geometry / reward constants (== wms.control.contract; asserted by the unit tests) ----
SPEED_LIMIT_MPS = 31.29
MERGE_S1_M = 75.44
RSU_MAINLINE_S_M = (-800.0, 225.44)
RSU_RAMP_S_M = (-378.66, 75.44)
RSU_CELL_M = 50.0
QUEUE_SPEED_MPS = 5.0
CROSSING_WIDTH_M = 1.0        # reward._CROSSING_WIDTH_M


def _edges(lo: float, hi: float) -> list[float]:
    n = int(math.ceil((hi - lo) / RSU_CELL_M - 1e-9))
    return [lo + i * RSU_CELL_M for i in range(n)] + [hi]


MAINLINE_EDGES = tuple(_edges(*RSU_MAINLINE_S_M))
RAMP_EDGES = tuple(_edges(*RSU_RAMP_S_M))
#: (lane, lo, hi) per cell, in ``wms.control.rsu_field.CELLS`` order.
CELLS: tuple[tuple[int, float, float], ...] = tuple(
    [(0, a, b) for a, b in zip(MAINLINE_EDGES[:-1], MAINLINE_EDGES[1:])]
    + [(1, a, b) for a, b in zip(MAINLINE_EDGES[:-1], MAINLINE_EDGES[1:])]
    + [(2, a, b) for a, b in zip(RAMP_EDGES[:-1], RAMP_EDGES[1:])])
#: Cross-section boundaries of the flow term (union of both grids, all lanes).
FLOW_EDGES = tuple(sorted(set(MAINLINE_EDGES) | set(RAMP_EDGES)))
FIELD_NAMES = ("density", "mean_speed", "queue")
TERM_NAMES = ("agg_field", "agg_reward", "agg_flow")


@dataclass(frozen=True)
class AggregateLossConfig:
    """Weights, bins, horizon ramp, smoothing temperatures and normalising scales."""

    dt_s: float = 0.5
    weight: float = 1.0
    field_weight: float = 1.0
    reward_weight: float = 1.0
    flow_weight: float = 0.0
    bin_s: float = 5.0
    every_steps: int = 1
    horizon_start_s: float = 0.0
    horizon_full_s: float = 0.0
    cell_temp_m: float = 5.0
    density_scale_vpkm: float = 50.0
    speed_scale_mps: float = 10.0
    throughput_scale_veh: float = 2.0
    flow_scale_veh: float = 2.0
    speed_prior_count: float = 0.25
    queue_temp_mps: float = 1.0
    occupied_temp: float = 0.15
    huber_delta: float = 1.0

    def __post_init__(self) -> None:
        if self.dt_s <= 0.0 or self.bin_s <= 0.0 or self.every_steps < 1:
            raise ValueError("aggregate: dt_s/bin_s must be positive and every_steps >= 1")
        if min(self.weight, self.field_weight, self.reward_weight, self.flow_weight) < 0.0:
            raise ValueError("aggregate weights must be non-negative")
        if self.horizon_start_s < 0.0 or self.horizon_full_s < 0.0:
            raise ValueError("aggregate horizons must be non-negative")
        for value in (self.cell_temp_m, self.density_scale_vpkm, self.speed_scale_mps,
                      self.throughput_scale_veh, self.flow_scale_veh, self.queue_temp_mps,
                      self.occupied_temp, self.huber_delta, self.speed_prior_count):
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError("aggregate temperatures/scales must be finite and positive")

    @property
    def bin_steps(self) -> int:
        return max(1, int(round(self.bin_s / self.dt_s)))

    def ramp(self, target_index: Tensor) -> Tensor:
        """Horizon weight of the state ``target_index`` (time ``index * dt``) in ``[0, 1]``."""
        t = target_index.to(torch.float32) * float(self.dt_s)
        start, full = float(self.horizon_start_s), float(self.horizon_full_s)
        if full <= start:
            return (t >= start).to(torch.float32)
        return ((t - start) / (full - start)).clamp(0.0, 1.0)


# ------------------------------------------------------------------ differentiable fields
def soft_cell_fields(s: Tensor, v: Tensor, lane: Tensor, present: Tensor, *,
                     cfg: AggregateLossConfig) -> dict[str, Tensor]:
    """Soft RSU cell fields and per-step reward aggregates of ``[B, T, K]`` states.

    Returns ``density``/``mean_speed``/``queue`` ``[B, T, N_cells]`` and ``delay_road``,
    ``count_road``, ``delay_zone``, ``count_zone`` ``[B, T]``.
    """
    dtype = s.dtype
    mask = present.to(dtype)
    tau = float(cfg.cell_temp_m)
    main = torch.sigmoid((s.unsqueeze(-1) - s.new_tensor(MAINLINE_EDGES)) / tau)  # [B,T,K,E]
    ramp = torch.sigmoid((s.unsqueeze(-1) - s.new_tensor(RAMP_EDGES)) / tau)
    d_main = main[..., :-1] - main[..., 1:]
    d_ramp = ramp[..., :-1] - ramp[..., 1:]
    lanes = [mask * (lane == i).to(dtype) for i in range(3)]
    counts, vsums = [], []
    for i, member in ((0, d_main), (1, d_main), (2, d_ramp)):
        counts.append(torch.einsum("btk,btkc->btc", lanes[i], member))
        vsums.append(torch.einsum("btk,btkc->btc", lanes[i] * v, member))
    count = torch.cat(counts, -1)
    vsum = torch.cat(vsums, -1)
    length_km = s.new_tensor([(hi - lo) / 1000.0 for _, lo, hi in CELLS])
    prior = float(cfg.speed_prior_count)
    mean_v = (vsum + prior * SPEED_LIMIT_MPS) / (count + prior)
    queue = (torch.sigmoid((QUEUE_SPEED_MPS - mean_v) / float(cfg.queue_temp_mps))
             * torch.sigmoid((count - 0.5) / float(cfg.occupied_temp)))
    zone = ((lanes[0] + lanes[1]) * (main[..., 0] - main[..., -1])
            + lanes[2] * (ramp[..., 0] - ramp[..., -1]))
    delay = float(cfg.dt_s) * F.relu(1.0 - v / SPEED_LIMIT_MPS)
    return {"density": count / length_km, "mean_speed": mean_v, "queue": queue,
            "delay_road": (mask * delay).sum(-1), "count_road": mask.sum(-1),
            "delay_zone": (zone * delay).sum(-1), "count_zone": zone.sum(-1)}


def soft_crossings(s0: Tensor, s1: Tensor, cont: Tensor, edges, temp_m: float) -> Tensor:
    """Soft same-vehicle forward crossings of each boundary, ``[B, T, len(edges)]``."""
    e = s0.new_tensor(list(edges))
    up = (torch.sigmoid((s1.unsqueeze(-1) - e) / temp_m)
          - torch.sigmoid((s0.unsqueeze(-1) - e) / temp_m))
    return (F.relu(up) * cont.to(s0.dtype).unsqueeze(-1)).sum(-2)


def _huber(x: Tensor, delta: float) -> Tensor:
    return F.huber_loss(x, torch.zeros_like(x), delta=float(delta), reduction="none")


def _bin_matrix(target_index: Tensor, bin_steps: int, dtype) -> Tensor:
    """One-hot ``[T, J]`` membership of each chunk step in its absolute-index bin."""
    bins = torch.div(target_index - 1, int(bin_steps), rounding_mode="floor")
    unique, inverse = torch.unique(bins, return_inverse=True)
    return F.one_hot(inverse, num_classes=int(unique.numel())).to(dtype)


# ------------------------------------------------------------------ recorded side
@dataclass
class RoadTargets:
    """Recorded on-road presence used by the aggregate loss (attached by ``build_targets``)."""

    present: Tensor     # [B, T+1, Kr] recorded on-road presence (full road or t0 cohort)
    cont: Tensor        # [B, T, Kr] same vehicle in the slot from t to t+1
    valid: Tensor       # [B, T+1] recorded tick available (not window padding)
    k: int              # recorded slot count (predicted slots beyond it are excluded if not full)
    full: bool          # True: full road incl. arrivals (predicted entrants are counted)
    lane: Optional[Tensor] = None   # [B, T+1, Kr] recorded lane (-1 = junction), else from states


def road_targets(batch, states: Tensor, present: Tensor, n_k: int, *, entries_on: bool,
                 device) -> RoadTargets:
    """Full-road recorded presence when entries run and the batch carries it, else the cohort.

    The recorded lane comes from the raw teacher (``cav_teacher[..., 1]``, -1 on a junction
    internal lane, which ``states`` clips to lane 0) when the batch has it.
    """
    lane = None
    teacher = batch.get("cav_teacher") if hasattr(batch, "get") else None
    if teacher is not None and teacher.shape[:3] == present[..., :n_k].shape:
        lane = teacher[..., 1].to(device).round().long()
    if entries_on and "road_present" in batch and "road_valid" in batch:
        rp = batch["road_present"].to(device).bool()
        cont = batch["road_continue"].to(device).bool()
        valid = batch["road_valid"].to(device).bool()
        cont = cont & rp[:, :-1] & rp[:, 1:]
        return RoadTargets(rp, cont, valid, int(rp.shape[-1]), True, lane)
    cohort = present[..., :n_k].bool()
    cont = cohort[:, :-1] & cohort[:, 1:]
    valid = cohort.any(-1)
    valid[:, 0] = True
    return RoadTargets(cohort, cont, valid, int(n_k), False, lane)


def _recorded_kin(states: Tensor, road: RoadTargets, lo: int, hi: int):
    """Recorded ``(s, lane, v)`` of state indices ``[lo, hi)`` for the first ``road.k`` slots."""
    rec = states[:, lo:hi, :road.k]
    if road.lane is not None:
        lane = road.lane[:, lo:hi]
    else:
        onehot = rec[..., 1:4]
        lane = torch.where(onehot.sum(-1) > 0, onehot.argmax(-1),
                           torch.full_like(onehot[..., 0], -1, dtype=torch.long))
    return rec[..., 0], lane, rec[..., 4]


# ------------------------------------------------------------------ the loss
class AggregateLoss:
    """Shared per-chunk computation of the three aggregate components (cached on ``ctx``)."""

    def __init__(self, cfg: AggregateLossConfig) -> None:
        self.cfg = cfg

    def step_masks(self, targets: UnrollTargets, lo: int, hi: int):
        """Per transition ``i in [lo, hi)`` (target state ``i+1``): valid and weights."""
        road: RoadTargets = targets.road
        idx = torch.arange(lo + 1, hi + 1, device=targets.states.device)
        valid = road.valid[:, lo + 1:hi + 1].to(targets.states.dtype)
        ramp = self.cfg.ramp(idx).to(valid.dtype)
        every = ((idx % int(self.cfg.every_steps)) == 0).to(valid.dtype)
        return idx, valid, ramp, every

    def prepare(self, targets: UnrollTargets) -> None:
        if targets.road is None:
            raise ValueError("aggregate loss requires UnrollTargets.road")
        _, valid, ramp, every = self.step_masks(targets, 0, targets.n_steps)
        targets.prepared["aggregate"] = {
            "w_field": float((valid * ramp * every).sum()),
            "w_reward": float((valid * ramp).sum())}

    def components(self, ctx: LossContext) -> dict[str, Tensor]:
        cached = ctx.cache.get("aggregate")
        if cached is not None:
            return cached
        cfg, t = self.cfg, ctx.targets
        road: RoadTargets = t.road
        totals = t.prepared["aggregate"]
        c0, c1 = ctx.c0, ctx.c1
        idx, valid, ramp, every = self.step_masks(t, c0, c1)
        k = None if road.full else road.k
        pred = ctx.pred

        def cut(x):
            return x if k is None else x[..., :k]

        ps, pv, pl, pm = (cut(pred["positions"]), cut(pred["speeds"]), cut(pred["lanes"]),
                          cut(pred["masks"]).bool())
        prev_s = torch.cat((cut(ctx.current["s_m"]).unsqueeze(1), ps[:, :-1]), 1)
        prev_m = torch.cat((cut(ctx.current["present"]).bool().unsqueeze(1), pm[:, :-1]), 1)
        pf = soft_cell_fields(ps, pv, pl, pm, cfg=cfg)
        with torch.no_grad():
            rs_all, rl_all, rv_all = _recorded_kin(t.states, road, c0, c1 + 1)
            rm = road.present[:, c0 + 1:c1 + 1]
            rf = soft_cell_fields(rs_all[:, 1:], rv_all[:, 1:], rl_all[:, 1:], rm, cfg=cfg)
            rcont = road.cont[:, c0:c1]
        bins = _bin_matrix(idx, cfg.bin_steps, ps.dtype)                      # [T, J]
        zero = ps.sum() * 0.0 + 0.0          # graph-connected +0.0 (never -0.0 in logs)
        out = {name: zero for name in TERM_NAMES}
        # ---- fields: bin means over valid (subsampled) steps, per-cell normalised error
        if cfg.field_weight > 0.0 and totals["w_field"] > 0.0:
            m = valid * every                                                  # [B, T]
            w_bin = torch.einsum("bt,tj->bj", m * ramp, bins)
            n_bin = torch.einsum("bt,tj->bj", m, bins).clamp_min(1e-6)
            scales = (cfg.density_scale_vpkm, cfg.speed_scale_mps, 1.0)
            err = 0.0
            for name, scale in zip(FIELD_NAMES, scales):
                fp = torch.einsum("btc,bt,tj->bjc", pf[name], m, bins) / n_bin.unsqueeze(-1)
                fr = torch.einsum("btc,bt,tj->bjc", rf[name], m, bins) / n_bin.unsqueeze(-1)
                err = err + _huber((fp - fr) / float(scale), cfg.huber_delta).mean(-1)
            err = err / len(FIELD_NAMES)
            out["agg_field"] = (cfg.weight * cfg.field_weight
                                * (w_bin * err).sum() / totals["w_field"])
        if totals["w_reward"] > 0.0 and (cfg.reward_weight > 0.0 or cfg.flow_weight > 0.0):
            w_bin = torch.einsum("bt,tj->bj", valid * ramp, bins)

            def bin_sum(x):  # [B, T(, E)] -> [B, J(, E)] over valid steps
                if x.ndim == 2:
                    return torch.einsum("bt,bt,tj->bj", x, valid, bins)
                return torch.einsum("bte,bt,tj->bje", x, valid, bins)

            pcont = prev_m & pm
            if cfg.reward_weight > 0.0:
                errs = []
                for zone in ("road", "zone"):
                    denom = bin_sum(rf[f"count_{zone}"].clamp_min(1.0)) * float(cfg.dt_s)
                    errs.append((bin_sum(pf[f"delay_{zone}"]) - bin_sum(rf[f"delay_{zone}"]))
                                / denom.clamp_min(1e-6))
                xp = soft_crossings(prev_s, ps, pcont, (MERGE_S1_M,), CROSSING_WIDTH_M)[..., 0]
                with torch.no_grad():
                    xr = soft_crossings(rs_all[:, :-1], rs_all[:, 1:], rcont, (MERGE_S1_M,),
                                        CROSSING_WIDTH_M)[..., 0]
                errs.append((bin_sum(xp) - bin_sum(xr)) / float(cfg.throughput_scale_veh))
                err = torch.stack([_huber(e, cfg.huber_delta) for e in errs], 0).mean(0)
                out["agg_reward"] = (cfg.weight * cfg.reward_weight
                                     * (w_bin * err).sum() / totals["w_reward"])
            if cfg.flow_weight > 0.0:
                fxp = soft_crossings(prev_s, ps, pcont, FLOW_EDGES, float(cfg.cell_temp_m))
                with torch.no_grad():
                    fxr = soft_crossings(rs_all[:, :-1], rs_all[:, 1:], rcont, FLOW_EDGES,
                                         float(cfg.cell_temp_m))
                err = _huber((bin_sum(fxp) - bin_sum(fxr)) / float(cfg.flow_scale_veh),
                             cfg.huber_delta).mean(-1)
                out["agg_flow"] = (cfg.weight * cfg.flow_weight
                                   * (w_bin * err).sum() / totals["w_reward"])
        ctx.cache["aggregate"] = out
        return out


class AggregateTerm(LossTerm):
    """One component (``agg_field`` / ``agg_reward`` / ``agg_flow``) of :class:`AggregateLoss`."""

    def __init__(self, name: str, shared: AggregateLoss, *, owner: bool) -> None:
        if name not in TERM_NAMES:
            raise ValueError(f"unknown aggregate term {name!r}")
        self.name = name
        self.shared = shared
        self.owner = owner          # the first enabled term prepares the unroll totals

    def prepare(self, targets: UnrollTargets) -> None:
        if self.owner:
            self.shared.prepare(targets)

    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        return TermValue(self.shared.components(ctx)[self.name])


def aggregate_terms(cfg: AggregateLossConfig) -> list[AggregateTerm]:
    """Enabled aggregate chunk terms (empty when the total weight is zero)."""
    if cfg.weight <= 0.0:
        return []
    shared = AggregateLoss(cfg)
    names = [name for name, w in zip(TERM_NAMES, (cfg.field_weight, cfg.reward_weight,
                                                  cfg.flow_weight)) if w > 0.0]
    return [AggregateTerm(name, shared, owner=(i == 0)) for i, name in enumerate(names)]
