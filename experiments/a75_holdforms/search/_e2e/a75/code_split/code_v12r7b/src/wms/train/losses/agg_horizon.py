"""A5 long-horizon aggregate delay loss (2026-10-19; default OFF).

Motivation (``onroad_diag/ONROAD_DIAG.md``): the WM's native on-road delay is 1.00x SUMO at
120 s but 1.36x at 300 s (yield off); 91 % of the excess accrues in 120-300 s, and the policy
deviation penalty is WM +195 vs SUMO +106 veh-s.  The per-car Huber/CE terms score 120 s
positions; the long-run *aggregate* the C4 policy actually optimises -- team delay, queue and
throughput, in the regions where the excess lives -- is not directly supervised beyond the
short horizon.

This module matches the WM rollout to the recorded SUMO over the *long* window
(``start_s`` .. ``horizon_s``, defaults 120-300 s), splitting the on-road team delay by region
(upstream / merge zone / ramp / downstream), plus queue length (all regions, so the ramp queue
is a component) and throughput (soft crossings of fixed sections, including the ramp merge
count).  It reuses the C4 geometry and the soft-field helpers of
:mod:`wms.train.losses.aggregate` and the paired-branch structure of
:mod:`wms.train.losses.paired_diff`:

* **agg_delay_region**: per ``bin_s`` bin and region ``r``, the recorded-vehicle-second
  normalised delay error ``(sum_t pred_delay_r - sum_t rec_delay_r) / (sum_t rec_count_r *
  dt)`` under a unit-delta Huber, averaged over the four regions.  ``delay = dt * relu(1 -
  v / SPEED_LIMIT)`` per present on-road vehicle; region membership is a soft function of ``s``
  (merge-zone membership = 1 - upstream - downstream) times the hard lane indicator (ramp =
  lane ``merge_lane``).
* **agg_queue**: per bin and region, ``(pred_queue_r - rec_queue_r) / queue_scale`` (Huber),
  averaged over regions; ``queue = sum sigmoid((queue_speed - v) / temp)`` per present on-road
  vehicle.  Region 2 is the ramp queue count.
* **agg_throughput**: per bin, the soft same-vehicle forward crossings of the fixed sections
  ``sections_m`` (all lanes) and of the ramp end ``merge_s1_m`` by lane-``merge_lane`` vehicles
  (the merges), each ``/ throughput_scale`` (Huber), averaged.
* **agg_pair** (only with ``paired_weight > 0`` and a paired batch): the paired DIFFERENCE of
  every aggregate above, ``(A_arm - A_ref)_pred - (A_arm - A_ref)_rec``, exactly as
  :class:`~wms.train.losses.paired_diff.PairedDiffLoss` does for the reward aggregates.  With
  an exactly predicted reference arm the term reduces to the arm's own aggregate error.

Simulating the whole recorded window under ``no_grad`` (``_recorded_kin``) keeps the graph to
the rollout; the recorded lane is the raw teacher lane where available.  As for the existing
aggregate loss, every component is a weighted mean over the whole unroll (``prepare`` sums the
unroll-wide weight), so truncated-BPTT chunks add up to one horizon-invariant mean provided the
bins align with the chunk boundaries.  Flag-gated: with ``weight = 0`` and ``paired_weight = 0``
the composer never builds these terms (bit-identical).

The RSU/reward constants are imported from :mod:`wms.train.losses.aggregate` (which asserts them
against ``wms.control.contract``), never re-derived here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .aggregate import (MERGE_S1_M, SPEED_LIMIT_MPS, RoadTargets, _bin_matrix, _huber,
                        _recorded_kin, road_targets, soft_crossings)
from .base import LossContext, LossTerm, TermValue, UnrollTargets

#: Region order of every per-region aggregate (ramp = ``merge_lane``).
REGION_NAMES = ("upstream", "merge", "ramp", "downstream")
TERM_NAMES = ("agg_delay_region", "agg_queue", "agg_throughput", "agg_pair")
PAIR_TERM = "agg_pair"
#: Default fixed cross-sections (s_down frame; s = 0 = merge start, 75.44 = ramp end).
DEFAULT_SECTIONS_M = (-400.0, 0.0, MERGE_S1_M, 400.0)


@dataclass(frozen=True)
class AggHorizonConfig:
    """Weights, the long window, region geometry and the normalising scales."""

    dt_s: float = 0.5
    weight: float = 1.0
    paired_weight: float = 0.0
    start_s: float = 120.0
    horizon_s: float = 300.0
    fade_s: float = 0.0
    bin_s: float = 5.0
    merge_s0_m: float = 0.0
    merge_s1_m: float = MERGE_S1_M
    merge_lane: int = 2
    road_end_m: float = 553.33
    sections_m: Sequence[float] = DEFAULT_SECTIONS_M
    queue_speed_mps: float = 5.0
    region_temp_m: float = 5.0
    queue_temp_mps: float = 1.0
    delay_scale_veh_s: float = 1.0
    queue_scale_veh: float = 1.0
    throughput_scale_veh: float = 1.0
    huber_delta: float = 1.0

    def __post_init__(self) -> None:
        if self.dt_s <= 0.0 or self.bin_s <= 0.0:
            raise ValueError("agg_delay dt_s/bin_s must be positive")
        if min(self.weight, self.paired_weight, self.start_s, self.horizon_s,
               self.fade_s) < 0.0:
            raise ValueError("agg_delay weights/window must be non-negative")
        for name in ("merge_s0_m", "merge_s1_m", "road_end_m", "queue_speed_mps",
                     "region_temp_m", "queue_temp_mps", "delay_scale_veh_s",
                     "queue_scale_veh", "throughput_scale_veh", "huber_delta"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"agg_delay {name} must be finite")
        for name in ("region_temp_m", "queue_temp_mps", "delay_scale_veh_s",
                     "queue_scale_veh", "throughput_scale_veh", "huber_delta"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"agg_delay {name} must be positive")
        if self.merge_s1_m <= self.merge_s0_m:
            raise ValueError("agg_delay merge_s1_m must exceed merge_s0_m")
        if int(self.merge_lane) < 1:
            raise ValueError("agg_delay merge_lane must be >= 1")
        sections = [float(x) for x in self.sections_m]
        if not sections or self.road_end_m <= 0.0:
            raise ValueError("agg_delay sections/road_end_m must be non-empty/positive")

    @property
    def enabled(self) -> bool:
        return self.weight > 0.0 or self.paired_weight > 0.0

    @property
    def bin_steps(self) -> int:
        return max(1, int(round(self.bin_s / self.dt_s)))

    def window_weight(self, target_index: Tensor) -> Tensor:
        """Long-window weight of state ``target_index`` (time ``index * dt``) in ``[0, 1]``.

        ``1`` for ``start_s <= t <= horizon_s`` (``horizon_s <= 0``: no upper end), zero
        outside; a positive ``fade_s`` linearly ramps in at ``start_s`` and out at
        ``horizon_s``.
        """
        t = target_index.to(torch.float32) * float(self.dt_s)
        lo, hi, fade = float(self.start_s), float(self.horizon_s), float(self.fade_s)
        if hi <= lo:
            inside = (t >= lo).to(torch.float32)
            return inside
        if fade > 0.0:
            up = ((t - lo) / fade).clamp(0.0, 1.0)
            down = ((hi - t) / fade).clamp(0.0, 1.0)
            return up * down
        return ((t >= lo) & (t <= hi)).to(torch.float32)


def region_delay_queue(s: Tensor, v: Tensor, lane: Tensor, present: Tensor, *,
                       cfg: AggHorizonConfig) -> dict[str, Tensor]:
    """Per-region on-road delay / queue / count of ``[B, T, K]`` kinematic fields.

    ``delay`` ``[B,T,4]`` in veh-s per step, ``queue`` ``[B,T,4]`` soft vehicle counts,
    ``count`` ``[B,T,4]`` soft present vehicle counts (recorded-side denominator).
    """
    dtype = s.dtype
    tau = float(cfg.region_temp_m)
    mask = present.to(dtype)
    on = torch.sigmoid((float(cfg.road_end_m) - s) / tau)
    main = mask * (lane < int(cfg.merge_lane)).to(dtype) * on
    ramp = mask * (lane == int(cfg.merge_lane)).to(dtype) * on
    up = torch.sigmoid((float(cfg.merge_s0_m) - s) / tau)
    down = torch.sigmoid((s - float(cfg.merge_s1_m)) / tau)
    merge = (1.0 - up - down).clamp_min(0.0)
    member = torch.stack((main * up, main * merge, ramp, main * down), dim=-1)   # [B,T,K,4]
    delay = float(cfg.dt_s) * F.relu(1.0 - v / SPEED_LIMIT_MPS)
    slow = torch.sigmoid((float(cfg.queue_speed_mps) - v) / float(cfg.queue_temp_mps))
    return {"delay": torch.einsum("btk,btkr->btr", delay, member),
            "queue": torch.einsum("btk,btkr->btr", slow, member),
            "count": member.sum(dim=-2)}


def crossings(prev_s: Tensor, s: Tensor, prev_lane: Tensor, lane: Tensor, cont: Tensor,
              present: Tensor, *, cfg: AggHorizonConfig) -> dict[str, Tensor]:
    """Soft section crossings and ramp->mainline merge counts of the chunk.

    ``sections`` ``[B,T]`` (every fixed boundary), ``merge`` ``[B,T]`` (lane-``merge_lane``
    vehicles entering the merge zone from the ramp; differentiable w.r.t. the rollout ``s``).
    """
    dtype = s.dtype
    tau = float(cfg.region_temp_m)
    prev_present = cont.to(dtype)
    sections = soft_crossings(prev_s, s, prev_present, tuple(float(x) for x in cfg.sections_m),
                              tau).sum(-1)
    in_zone = (torch.sigmoid((s - float(cfg.merge_s0_m)) / tau)
               * torch.sigmoid((float(cfg.merge_s1_m) - s) / tau))
    transformed = ((prev_lane == int(cfg.merge_lane))
                   & (lane <= int(cfg.merge_lane) - 1)).to(dtype)
    merge = (transformed * in_zone * present.to(dtype)).sum(-1)
    return {"sections": sections, "merge": merge}


def _binsum(x: Tensor, valid: Tensor, bins: Tensor) -> Tensor:
    """Sum ``x`` over valid steps into global bins: ``[B,T]`` or ``[B,T,R]`` -> ``[B,J(,R)]``."""
    if x.ndim == 2:
        return torch.einsum("bt,bt,tj->bj", x, valid, bins)
    return torch.einsum("btr,bt,tj->bjr", x, valid, bins)


class AggHorizonLoss:
    """Shared per-chunk computation of the long-horizon aggregate terms (cached on ``ctx``)."""

    def __init__(self, cfg: AggHorizonConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------ masks / prep
    def step_masks(self, targets: UnrollTargets, lo: int, hi: int):
        road: RoadTargets = targets.road
        device = targets.states.device
        idx = torch.arange(lo + 1, hi + 1, device=device)
        valid = road.valid[:, lo + 1:hi + 1].to(targets.states.dtype)
        window = self.cfg.window_weight(idx).to(valid.dtype).unsqueeze(0)
        return idx, valid, window

    @staticmethod
    def _pair_valid(targets: UnrollTargets, lo: int, hi: int) -> Tensor:
        pair = targets.pair
        valid = pair.road.valid[:, lo + 1:hi + 1]
        return valid & valid[pair.ref] & pair.arm.view(-1, 1)

    def prepare(self, targets: UnrollTargets) -> None:
        if targets.road is None:
            raise ValueError("agg_delay loss requires UnrollTargets.road")
        _, valid, window = self.step_masks(targets, 0, targets.n_steps)
        targets.prepared["agg_horizon"] = {"w": float((valid * window).sum())}
        pair = getattr(targets, "pair", None)
        if pair is not None:
            pvalid = self._pair_valid(targets, 0, targets.n_steps).to(valid.dtype)
            targets.prepared["agg_horizon_pair"] = {
                "w": float((pvalid * window).sum()), "n_arms": int(pair.arm.sum())}
        else:
            targets.prepared["agg_horizon_pair"] = {"w": 0.0, "n_arms": 0}

    # ------------------------------------------------------------------ aggregate views
    def _predicted(self, ctx: LossContext, k: Optional[int]):
        pred = ctx.pred
        c0, c1 = ctx.c0, ctx.c1

        def cut(x):
            return x if k is None else x[..., :k]

        s, v, lane = cut(pred["positions"]), cut(pred["speeds"]), cut(pred["lanes"]).long()
        present = cut(pred["masks"]).bool()
        prev_lane = ctx.current.get("lane")
        if torch.is_tensor(prev_lane):
            prev_lane = cut(prev_lane).long()
            prev_lane = torch.cat((prev_lane.unsqueeze(1), lane[:, :-1]), dim=1)
        else:                                   # adapter without a lane state: no ramp transition
            prev_lane = lane
        prev_present = cut(ctx.current["present"]).bool()
        prev_s = torch.cat((cut(ctx.current["s_m"]).unsqueeze(1), s[:, :-1]), dim=1)
        prev_p = torch.cat((prev_present.unsqueeze(1), present[:, :-1]), dim=1)
        fields = region_delay_queue(s, v, lane, present, cfg=self.cfg)
        cross = crossings(prev_s, s, prev_lane, lane, prev_p & present, present, cfg=self.cfg)
        return {**fields, **cross}

    def _recorded(self, targets: UnrollTargets, road: RoadTargets, c0: int, c1: int):
        rs_all, rl_all, rv_all = _recorded_kin(targets.states, road, c0, c1 + 1)
        rm = road.present[:, c0 + 1:c1 + 1]
        fields = region_delay_queue(rs_all[:, 1:], rv_all[:, 1:], rl_all[:, 1:], rm, cfg=self.cfg)
        cont = road.cont[:, c0:c1]
        cross = crossings(rs_all[:, :-1], rs_all[:, 1:], rl_all[:, :-1], rl_all[:, 1:], cont,
                          rm, cfg=self.cfg)
        return {**fields, **cross}

    # ------------------------------------------------------------------ components
    def components(self, ctx: LossContext) -> dict[str, Tensor]:
        cached = ctx.cache.get("agg_horizon")
        if cached is not None:
            return cached
        cfg, t = self.cfg, ctx.targets
        road: RoadTargets = t.road
        totals = t.prepared["agg_horizon"]
        c0, c1 = ctx.c0, ctx.c1
        idx, valid, window = self.step_masks(t, c0, c1)
        k = None if road.full else road.k
        pred = self._predicted(ctx, k)
        with torch.no_grad():
            rec = self._recorded(t, road, c0, c1)
        bins = _bin_matrix(idx, cfg.bin_steps, valid.dtype)                         # [T, J]
        zero = ctx.pred["positions"].sum() * 0.0 + 0.0
        out = {name: zero for name in TERM_NAMES}
        if totals["w"] <= 0.0 or cfg.weight <= 0.0:
            out.update(self._pair_out(ctx, zero))
            ctx.cache["agg_horizon"] = out
            return out
        # ---- per-region delay and queue (recorded-normalised), throughput crossings
        w_bin = torch.einsum("bt,tj->bj", valid * window, bins)                  # [B, J]
        delay_err = []
        for r in range(len(REGION_NAMES)):
            denom = _binsum(rec["count"][..., r].clamp_min(1.0), valid, bins) \
                * float(cfg.dt_s)
            delay_err.append((_binsum(pred["delay"][..., r], valid, bins)
                              - _binsum(rec["delay"][..., r], valid, bins))
                             / denom.clamp_min(1e-6))
        err_delay = torch.stack([_huber(e, cfg.huber_delta) for e in delay_err], 0).mean(0)
        out["agg_delay_region"] = (cfg.weight * cfg.delay_scale_veh_s
                                   * (w_bin * err_delay).sum() / totals["w"])
        queue_err = [(_binsum(pred["queue"][..., r], valid, bins)
                      - _binsum(rec["queue"][..., r], valid, bins)) / float(cfg.queue_scale_veh)
                     for r in range(len(REGION_NAMES))]
        err_queue = torch.stack([_huber(e, cfg.huber_delta) for e in queue_err], 0).mean(0)
        out["agg_queue"] = (cfg.weight * (w_bin * err_queue).sum() / totals["w"])
        thr = [_binsum(pred[key], valid, bins) - _binsum(rec[key], valid, bins)
               for key in ("sections", "merge")]
        err_thr = torch.stack([_huber(e / float(cfg.throughput_scale_veh), cfg.huber_delta)
                               for e in thr], 0).mean(0)
        out["agg_throughput"] = (cfg.weight * (w_bin * err_thr).sum() / totals["w"])
        out.update(self._pair_out(ctx, zero))
        ctx.cache["agg_horizon"] = out
        return out

    def _pair_out(self, ctx: LossContext, zero: Tensor) -> dict[str, Tensor]:
        """The ``agg_pair`` component (zero unless a paired batch and ``paired_weight > 0``)."""
        cfg, t = self.cfg, ctx.targets
        pair = getattr(t, "pair", None)
        prep = t.prepared.get("agg_horizon_pair")
        if (cfg.paired_weight <= 0.0 or pair is None or prep is None
                or prep["w"] <= 0.0 or prep["n_arms"] == 0):
            return {PAIR_TERM: zero}
        c0, c1 = ctx.c0, ctx.c1
        idx, _, window = self.step_masks(t, c0, c1)
        valid = self._pair_valid(t, c0, c1).to(zero.dtype)
        road = pair.road
        pred = self._predicted(ctx, road.k)
        with torch.no_grad():
            rec = self._recorded(t, road, c0, c1)
        bins = _bin_matrix(idx, cfg.bin_steps, zero.dtype)

        def bin_sum(x: Tensor) -> Tensor:        # per arm row, over that arm's valid steps
            return _binsum(x, valid, bins)

        def diff(x: Tensor) -> Tensor:
            return bin_sum(x) - bin_sum(x[pair.ref])

        derrs = []
        for r in range(len(REGION_NAMES)):
            denom = bin_sum(rec["count"][..., r].clamp_min(1.0)) * float(cfg.dt_s)
            derrs.append(_huber((diff(pred["delay"][..., r]) - diff(rec["delay"][..., r]))
                                / denom.clamp_min(1e-6), cfg.huber_delta))
        qerrs = [_huber((diff(pred["queue"][..., r]) - diff(rec["queue"][..., r]))
                        / float(cfg.queue_scale_veh), cfg.huber_delta)
                 for r in range(len(REGION_NAMES))]
        terr = [_huber((diff(pred[key]) - diff(rec[key])) / float(cfg.throughput_scale_veh),
                       cfg.huber_delta) for key in ("sections", "merge")]
        err_all = (torch.stack(derrs, 0).mean(0) + torch.stack(qerrs, 0).mean(0)
                   + torch.stack(terr, 0).mean(0)) / 3.0
        w_bin = torch.einsum("bt,tj->bj", valid, bins)
        return {PAIR_TERM: cfg.paired_weight * (w_bin * err_all).sum() / prep["w"]}


class AggHorizonTerm(LossTerm):
    """One component of :class:`AggHorizonLoss` (the first enabled term prepares the unroll)."""

    def __init__(self, name: str, shared: AggHorizonLoss, *, owner: bool) -> None:
        if name not in TERM_NAMES:
            raise ValueError(f"unknown agg_delay term {name!r}")
        self.name = name
        self.shared = shared
        self.owner = owner

    def prepare(self, targets: UnrollTargets) -> None:
        if self.owner:
            self.shared.prepare(targets)

    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        return TermValue(self.shared.components(ctx)[self.name])


def agg_horizon_terms(cfg: AggHorizonConfig) -> list[AggHorizonTerm]:
    """Enabled long-horizon aggregate chunk terms (empty when both weights are zero)."""
    if not cfg.enabled:
        return []
    names = list(TERM_NAMES[:3]) if cfg.weight > 0.0 else []
    if cfg.paired_weight > 0.0:
        names.append(PAIR_TERM)
    shared = AggHorizonLoss(cfg)
    return [AggHorizonTerm(name, shared, owner=(i == 0)) for i, name in enumerate(names)]


__all__ = ["AggHorizonConfig", "AggHorizonLoss", "AggHorizonTerm", "agg_horizon_terms",
           "region_delay_queue", "crossings", "TERM_NAMES", "REGION_NAMES", "PAIR_TERM"]
