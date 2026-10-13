"""PAIRLOSS paired-difference loss (2026-10-14): learn CAV-action EFFECTS from paired branches.

A paired batch (:class:`wms.data.c4_pairs.PairedBatchDataset`) holds pair groups of GENRAND
branch windows: every arm of a group starts from the same SUMO state at t0 (identical
pre-history, HDV roster and slots) and differs only in the CAV actions after t0, which the
model sees only through the recorded CAV teacher (realized CAV motion).  The per-car terms
already fit each arm; this loss fits the DIFFERENCE an action change makes, which is what the
C4 policy optimises.  For every non-reference arm ``a`` of a group with reference ``r`` (the
all-native arm, else the first arm):

* ``pair_pos`` / ``pair_speed`` - per t0 HDV and step, where the vehicle is supervised in BOTH
  arms (``hdv_mask``): ``huber((p_a - p_r) - (x_a - x_r)) / delta`` (prediction ``p``, record
  ``x``; Huber deltas = the per-car position / speed deltas), step-weighted like the per-car
  terms and normalised over the whole unroll;
* ``pair_agg`` - per ``bin_s`` bin, the cohort aggregates the C4 reward reads (on-road delay of
  the whole road and of the RSU zone, soft throughput past the merge end), computed with the
  aggregate loss's soft fields (:func:`~wms.train.losses.aggregate.soft_cell_fields`,
  :func:`~wms.train.losses.aggregate.soft_crossings`): ``huber((A_a - A_r)_pred - (A_a -
  A_r)_rec)`` with the aggregate reward normalisation of arm ``a`` (delay per recorded
  vehicle-second, throughput / scale).  When the reference prediction equals its record this is
  exactly the ``agg_reward`` error of arm ``a``.  The t0 cohort is used (not the full road):
  entrants are drawn per row by the entry process, so their noise would not cancel in the
  difference.

Batches without pair groups contribute a graph-connected zero (keys stay in the log).  With
``loss.paired_diff.weight = 0`` the composer never builds these terms (bit-identical).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .aggregate import (CROSSING_WIDTH_M, MERGE_S1_M, AggregateLossConfig, RoadTargets,
                        _bin_matrix, _huber, _recorded_kin, road_targets, soft_cell_fields,
                        soft_crossings)
from .base import LossContext, LossTerm, TermValue, UnrollTargets

TERM_NAMES = ("pair_pos", "pair_speed", "pair_agg")


@dataclass(frozen=True)
class PairedDiffConfig:
    """Weights and normalisers of the paired-difference terms."""

    dt_s: float = 0.5
    weight: float = 1.0
    position_weight: float = 1.0
    speed_weight: float = 1.0
    aggregate_weight: float = 1.0
    position_delta_m: float = 5.0
    speed_delta_mps: float = 1.0
    bin_s: float = 5.0
    cell_temp_m: float = 5.0

    def __post_init__(self) -> None:
        if min(self.weight, self.position_weight, self.speed_weight,
               self.aggregate_weight) < 0.0:
            raise ValueError("paired_diff weights must be non-negative")
        if self.position_delta_m <= 0.0 or self.speed_delta_mps <= 0.0 or self.dt_s <= 0.0:
            raise ValueError("paired_diff deltas / dt must be positive")

    def aggregate(self) -> AggregateLossConfig:
        """Aggregate-loss config whose soft fields / scales the ``pair_agg`` term reuses."""
        return AggregateLossConfig(dt_s=self.dt_s, bin_s=self.bin_s, cell_temp_m=self.cell_temp_m)


@dataclass
class PairTargets:
    """Pair structure of one paired batch (attached by ``build_targets``)."""

    ref: Tensor          # [B] long: batch row of the row's group reference (itself for refs)
    arm: Tensor          # [B] bool: a non-reference arm of a pair group
    road: RoadTargets    # recorded t0-cohort presence (aggregate conventions, cohort mode)


def pair_targets(batch: Mapping[str, Any], states: Tensor, present: Tensor, n_k: int, *,
                 device) -> Optional[PairTargets]:
    """Pair targets of a paired batch (``batch["pair_ref"]``), else ``None``."""
    if "pair_ref" not in batch:
        return None
    ref = batch["pair_ref"].to(device).long()
    rows = torch.arange(ref.shape[0], device=device)
    if bool((ref < 0).any()) or bool((ref >= ref.shape[0]).any()):
        raise ValueError("pair_ref out of range")
    road = road_targets(batch, states, present, int(n_k), entries_on=False, device=device)
    return PairTargets(ref=ref, arm=ref != rows, road=road)


class PairedDiffLoss:
    """Shared per-chunk computation of the three paired terms (cached on ``ctx``)."""

    def __init__(self, cfg: PairedDiffConfig) -> None:
        self.cfg = cfg
        self.agg = cfg.aggregate()

    @staticmethod
    def _pair_mask(targets: UnrollTargets, pair: PairTargets) -> Tensor:
        m = targets.hdv_mask
        return m & m[pair.ref] & pair.arm.view(-1, 1, 1)

    @staticmethod
    def _valid_pair(targets: UnrollTargets, pair: PairTargets, lo: int, hi: int) -> Tensor:
        valid = pair.road.valid[:, lo + 1:hi + 1]
        return valid & valid[pair.ref] & pair.arm.view(-1, 1)

    def prepare(self, targets: UnrollTargets) -> None:
        pair: Optional[PairTargets] = getattr(targets, "pair", None)
        if pair is None:
            targets.prepared["paired_diff"] = None
            return
        m = self._pair_mask(targets, pair).to(targets.weights.dtype)
        valid = self._valid_pair(targets, pair, 0, targets.n_steps)
        targets.prepared["paired_diff"] = {
            "mask": m, "w_car": float((targets.weights * m).sum()),
            "w_agg": float(valid.sum()), "n_arms": int(pair.arm.sum())}

    def components(self, ctx: LossContext) -> dict[str, Tensor]:
        cached = ctx.cache.get("paired_diff")
        if cached is not None:
            return cached
        cfg, t = self.cfg, ctx.targets
        pred = ctx.pred
        zero = pred["positions"].sum() * 0.0 + 0.0
        out = {name: zero for name in TERM_NAMES}
        prep = t.prepared.get("paired_diff")
        if prep is None or prep["n_arms"] == 0:
            ctx.cache["paired_diff"] = out
            return out
        pair: PairTargets = t.pair
        ref = pair.ref
        c0, c1 = ctx.c0, ctx.c1
        # ---- per-vehicle differences (t0 HDVs supervised in both arms)
        m = prep["mask"][:, c0:c1]
        w = t.weights[:, c0:c1]
        for name, key, rec, delta, weight in (
                ("pair_pos", "positions", ctx.rec_pos, cfg.position_delta_m, cfg.position_weight),
                ("pair_speed", "speeds", ctx.rec_spd, cfg.speed_delta_mps, cfg.speed_weight)):
            if weight <= 0.0 or prep["w_car"] <= 0.0:
                continue
            p = pred[key]
            dp = p - p[ref]
            dr = (rec - rec[ref]).to(p.dtype)
            err = F.huber_loss(dp, dr, delta=float(delta), reduction="none") / float(delta)
            out[name] = cfg.weight * weight * (w * m * err).sum() / prep["w_car"]
        # ---- cohort aggregate differences per bin (aggregate-loss soft fields)
        if cfg.aggregate_weight > 0.0 and prep["w_agg"] > 0.0:
            out["pair_agg"] = cfg.weight * cfg.aggregate_weight * self._aggregate(ctx, prep)
        ctx.cache["paired_diff"] = out
        return out

    def _aggregate(self, ctx: LossContext, prep: dict) -> Tensor:
        agg, t = self.agg, ctx.targets
        pair: PairTargets = t.pair
        road = pair.road
        ref, c0, c1 = pair.ref, ctx.c0, ctx.c1
        k = road.k
        pred = ctx.pred
        ps, pv, pl = (pred["positions"][..., :k], pred["speeds"][..., :k],
                      pred["lanes"][..., :k])
        pm = pred["masks"][..., :k].bool()
        prev_s = torch.cat((ctx.current["s_m"][..., :k].unsqueeze(1), ps[:, :-1]), 1)
        prev_m = torch.cat((ctx.current["present"][..., :k].bool().unsqueeze(1), pm[:, :-1]), 1)
        pf = soft_cell_fields(ps, pv, pl, pm, cfg=agg)
        xp = soft_crossings(prev_s, ps, prev_m & pm, (MERGE_S1_M,), CROSSING_WIDTH_M)[..., 0]
        with torch.no_grad():
            rs_all, rl_all, rv_all = _recorded_kin(t.states, road, c0, c1 + 1)
            rm = road.present[:, c0 + 1:c1 + 1]
            rf = soft_cell_fields(rs_all[:, 1:], rv_all[:, 1:], rl_all[:, 1:], rm, cfg=agg)
            xr = soft_crossings(rs_all[:, :-1], rs_all[:, 1:], road.cont[:, c0:c1],
                                (MERGE_S1_M,), CROSSING_WIDTH_M)[..., 0]
        idx = torch.arange(c0 + 1, c1 + 1, device=ps.device)
        bins = _bin_matrix(idx, agg.bin_steps, ps.dtype)                          # [T, J]
        valid = self._valid_pair(t, pair, c0, c1).to(ps.dtype)                   # [B, T]

        def bin_sum(x: Tensor) -> Tensor:   # rows' own steps where BOTH arms are valid
            return torch.einsum("bt,bt,tj->bj", x, valid, bins)

        def diff(x: Tensor) -> Tensor:      # arm minus its reference, over the arm's steps
            return bin_sum(x) - bin_sum(x[ref])

        errs = []
        for zone in ("road", "zone"):
            denom = bin_sum(rf[f"count_{zone}"].clamp_min(1.0)) * float(agg.dt_s)
            errs.append((diff(pf[f"delay_{zone}"]) - diff(rf[f"delay_{zone}"]))
                        / denom.clamp_min(1e-6))
        errs.append((diff(xp) - diff(xr)) / float(agg.throughput_scale_veh))
        err = torch.stack([_huber(e, agg.huber_delta) for e in errs], 0).mean(0)
        w_bin = torch.einsum("bt,tj->bj", valid, bins)
        return (w_bin * err).sum() / prep["w_agg"]


class PairedDiffTerm(LossTerm):
    """One component (``pair_pos`` / ``pair_speed`` / ``pair_agg``) of :class:`PairedDiffLoss`."""

    def __init__(self, name: str, shared: PairedDiffLoss, *, owner: bool) -> None:
        if name not in TERM_NAMES:
            raise ValueError(f"unknown paired_diff term {name!r}")
        self.name = name
        self.shared = shared
        self.owner = owner

    def prepare(self, targets: UnrollTargets) -> None:
        if self.owner:
            self.shared.prepare(targets)

    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        return TermValue(self.shared.components(ctx)[self.name])


def paired_diff_terms(cfg: PairedDiffConfig) -> list[PairedDiffTerm]:
    """Enabled paired-difference chunk terms (empty when the total weight is zero)."""
    if cfg.weight <= 0.0:
        return []
    shared = PairedDiffLoss(cfg)
    names = [name for name, w in zip(TERM_NAMES, (cfg.position_weight, cfg.speed_weight,
                                                  cfg.aggregate_weight)) if w > 0.0]
    return [PairedDiffTerm(name, shared, owner=(i == 0)) for i, name in enumerate(names)]
