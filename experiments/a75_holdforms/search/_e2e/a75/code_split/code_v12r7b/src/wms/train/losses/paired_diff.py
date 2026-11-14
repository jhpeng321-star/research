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
* Opt-in ``pair_return`` / ``pair_rank`` fit the full-horizon t0-HDV negative-delay
  return difference, using a common t0 HDV count times commonly observed duration as
  the scale. Return regression is Huber; ranking uses a veh-s margin and ignores
  recorded effects at or below that margin. See :class:`PairedDiffLoss` for replay.

Batches without pair groups contribute a graph-connected zero (keys stay in the log).  With
``loss.paired_diff.weight = 0`` the composer never builds these terms (bit-identical).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .aggregate import (CROSSING_WIDTH_M, MERGE_S1_M, SPEED_LIMIT_MPS, AggregateLossConfig, RoadTargets,
                        _bin_matrix, _huber, _recorded_kin, road_targets, soft_cell_fields,
                        soft_crossings)
from .base import LossContext, LossTerm, TermValue, UnrollTargets

# Public legacy contract: callers/tests use this tuple for the default paired terms.
TERM_NAMES = ("pair_pos", "pair_speed", "pair_agg")
RETURN_TERM_NAMES = ("pair_return", "pair_rank")


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
    return_weight: float = 0.0
    rank_weight: float = 0.0
    rank_margin_veh_s: float = 5.0

    def __post_init__(self) -> None:
        if min(self.weight, self.position_weight, self.speed_weight,
               self.aggregate_weight) < 0.0:
            raise ValueError("paired_diff weights must be non-negative")
        if any(not math.isfinite(w) or w < 0.0 for w in (self.return_weight, self.rank_weight)):
            raise ValueError("paired_diff return/rank weights must be finite and non-negative")
        if not math.isfinite(self.rank_margin_veh_s) or self.rank_margin_veh_s < 0.0:
            raise ValueError("paired_diff rank margin must be finite and non-negative")
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
    observed_valid: Optional[Tensor] = None  # explicit observed ticks, including empty-road ticks


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
    observed = batch.get("road_valid")
    return PairTargets(ref=ref, arm=ref != rows, road=road,
                       observed_valid=None if observed is None else observed.to(device).bool())


class PairedDiffLoss:
    """Shared paired terms, including optional whole-horizon negative-delay objectives.

    Runner replay API (all differences are arm minus reference, shape ``[B]``, veh-s):

    * After ``prepare``, sum ``predicted_return_difference(ctx)`` over a no-grad probe's
      chunks. This helper caches each chunk at ``ctx.cache['paired_return_difference']``;
      it does not accumulate implicitly, so repeated term evaluation cannot double count.
    * Set ``targets.prepared['paired_return_reference'] = probe_sum.detach()`` before
      gradient replay. Every chunk uses that WHOLE-horizon prediction with only its local
      gradient attached. Replay must reproduce the probe trajectory / randomness.
    * ``return_losses(targets, probe_sum)`` gives weighted whole-horizon losses for probe
      or evaluation logging. ``recorded_return_difference(targets, c0, c1)`` exposes the
      matching recorded chunk sum (omitting bounds gives the full horizon).

    Without a reference, a full single chunk evaluates directly. No-grad partial chunks
    expose their differences but return zero return/rank terms; the runner must sum the
    differences and call ``return_losses``. Gradient-enabled partial chunks require replay.
    TermValue logs the replay objective only at c0 == 0; every chunk retains gradients.
    """

    def __init__(self, cfg: PairedDiffConfig) -> None:
        self.cfg = cfg
        self.agg = cfg.aggregate()

    @property
    def return_enabled(self) -> bool:
        return self.cfg.weight > 0.0 and (self.cfg.return_weight > 0.0 or self.cfg.rank_weight > 0.0)

    @property
    def return_names(self) -> tuple[str, ...]:
        return tuple(name for name, weight in zip(
            RETURN_TERM_NAMES, (self.cfg.return_weight, self.cfg.rank_weight)) if weight > 0.0)

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
            if self.return_enabled:
                targets.prepared["paired_return"] = None
            return
        m = self._pair_mask(targets, pair).to(targets.weights.dtype)
        valid = self._valid_pair(targets, pair, 0, targets.n_steps)
        targets.prepared["paired_diff"] = {
            "mask": m, "w_car": float((targets.weights * m).sum()),
            "w_agg": float(valid.sum()), "n_arms": int(pair.arm.sum())}
        if self.return_enabled:
            self._prepare_return(targets)

    @torch.no_grad()
    def _prepare_return(self, targets: UnrollTargets) -> None:
        pair: PairTargets = targets.pair
        k = pair.road.k
        hdv0 = pair.road.present[:, 0, :k] & ~targets.is_cav[:, 0, :k].bool()
        # The paired roster is identical at t0. Intersection also makes the scale symmetric
        # for imperfect inputs; neither arm's later occupancy enters the denominator.
        cohort = hdv0 & hdv0[pair.ref]
        observed = pair.road.valid if pair.observed_valid is None else pair.observed_valid
        observed = observed[:, 1:targets.n_steps + 1]
        common = observed & observed[pair.ref]
        count = cohort.sum(-1)
        horizon_s = common.sum(-1).to(targets.states.dtype) * float(self.cfg.dt_s)
        scale = count.to(horizon_s.dtype) * horizon_s
        active = pair.arm & (scale > 0.0)
        prep = {"cohort": cohort, "valid": common, "scale": scale.clamp_min(1e-6),
                "active": active, "hdv_count": count, "horizon_s": horizon_s}
        targets.prepared["paired_return"] = prep
        recorded = self.recorded_return_difference(targets)
        prep["recorded_difference"] = recorded
        prep["rank_active"] = active & (recorded.abs() > self.cfg.rank_margin_veh_s)

    def _return_step_difference(self, targets: UnrollTargets, speeds: Tensor, present: Tensor,
                                c0: int, c1: int) -> Tensor:
        pair: PairTargets = targets.pair
        prep = targets.prepared["paired_return"]
        # Use the same t0 HDVs for both arms of EACH comparison, even when a reference is
        # shared by several arms. Entrant slots and CAVs never contribute.
        cohort = prep["cohort"].unsqueeze(1)
        delay = float(self.cfg.dt_s) * F.relu(1.0 - speeds / SPEED_LIMIT_MPS)
        own = (delay * (present.bool() & cohort).to(delay.dtype)).sum(-1)
        ref = (delay[pair.ref] * (present[pair.ref].bool() & cohort).to(delay.dtype)).sum(-1)
        valid = prep["valid"][:, c0:c1] & pair.arm.unsqueeze(-1)
        return (ref - own) * valid.to(delay.dtype)  # negative delay: higher is better

    @torch.no_grad()
    def recorded_return_difference(self, targets: UnrollTargets, c0: int = 0,
                                   c1: Optional[int] = None) -> Tensor:
        """Recorded cumulative return difference in veh-s; bounds use chunk indices."""
        c1 = targets.n_steps if c1 is None else c1
        if targets.prepared.get("paired_return") is None:
            return targets.states.new_zeros(targets.states.shape[0])
        road = targets.pair.road
        speed = targets.states[:, c0 + 1:c1 + 1, :road.k, 4]
        present = road.present[:, c0 + 1:c1 + 1]
        return self._return_step_difference(targets, speed, present, c0, c1).sum(-1)

    def predicted_return_difference(self, ctx: LossContext) -> Tensor:
        """Differentiable chunk return difference ``[B]`` in veh-s, cached for the runner."""
        if "paired_return_difference" not in ctx.cache:
            t = ctx.targets
            speed = ctx.pred["speeds"]
            if t.prepared.get("paired_return") is None:
                difference = speed.sum(dim=(1, 2)) * 0.0
            else:
                k = t.pair.road.k
                difference = self._return_step_difference(
                    t, speed[..., :k], ctx.pred["masks"][..., :k], ctx.c0, ctx.c1).sum(-1)
            ctx.cache["paired_return_difference"] = difference
        return ctx.cache["paired_return_difference"]

    def return_losses(self, targets: UnrollTargets, difference: Tensor) -> dict[str, Tensor]:
        """Weighted WHOLE-horizon losses from raw predicted differences in veh-s.

        Return regression is unit-delta Huber of the common-scale-normalised error.
        Ranking is ``relu(margin - sign(recorded) * predicted) / scale`` only for
        ``abs(recorded) > margin``. Both reductions average over observed nonempty pairs;
        ties / sub-margin pairs contribute exactly zero to ranking.
        """
        zero = difference.sum() * 0.0
        out = {name: zero for name in self.return_names}
        prep = targets.prepared.get("paired_return")
        if prep is None:
            return out
        if difference.shape != prep["scale"].shape:
            raise ValueError("paired return difference must have shape [B] in veh-s")
        scale = prep["scale"].to(difference)
        recorded = prep["recorded_difference"].to(difference)
        active = prep["active"]
        denom = active.sum().clamp_min(1).to(difference.dtype)
        if self.cfg.return_weight > 0.0:
            error = _huber((difference - recorded) / scale, 1.0)
            out["pair_return"] = (self.cfg.weight * self.cfg.return_weight
                                  * torch.where(active, error, 0.0).sum() / denom)
        if self.cfg.rank_weight > 0.0:
            hinge = F.relu(self.cfg.rank_margin_veh_s - recorded.sign() * difference) / scale
            out["pair_rank"] = (self.cfg.weight * self.cfg.rank_weight
                                * torch.where(prep["rank_active"], hinge, 0.0).sum() / denom)
        return out

    @torch.no_grad()
    def recorded_effect_bins(self, targets: UnrollTargets) -> dict[str, Tensor]:
        """Full-horizon logging metadata, independent of TBPTT chunk boundaries.

        ``effect_bin`` is -1 / 0 / +1 for negative / within-margin / positive recorded
        effects. Filter by ``active`` to exclude references and unobserved/empty pairs.
        """
        prep = targets.prepared.get("paired_return")
        if prep is None:
            return {}
        recorded = prep["recorded_difference"]
        effect_bin = torch.where(prep["rank_active"], recorded.sign(), 0.0).long()
        return {"difference_veh_s": recorded, "effect_bin": effect_bin,
                "active": prep["active"], "rank_active": prep["rank_active"],
                "scale_veh_s": prep["scale"], "hdv_count": prep["hdv_count"],
                "horizon_s": prep["horizon_s"]}

    def _return_components(self, ctx: LossContext) -> dict[str, Tensor]:
        chunk = self.predicted_return_difference(ctx)
        reference = ctx.targets.prepared.get("paired_return_reference")
        if reference is not None:
            if reference.shape != chunk.shape or reference.requires_grad:
                raise ValueError("paired_return_reference must be a detached whole-horizon [B] tensor")
            whole = reference.to(chunk) + (chunk - chunk.detach())
        elif ctx.c0 == 0 and ctx.c1 == ctx.targets.n_steps:
            whole = chunk
        elif not torch.is_grad_enabled():
            # A probe cannot evaluate a nonlinear horizon objective before the final chunk.
            return {name: chunk.sum() * 0.0 for name in self.return_names}
        else:
            raise ValueError("paired return/rank TBPTT requires paired_return_reference from a "
                             "no-grad whole-horizon probe; sum predicted_return_difference(ctx)")
        return self.return_losses(ctx.targets, whole)

    def components(self, ctx: LossContext) -> dict[str, Tensor]:
        cached = ctx.cache.get("paired_diff")
        if cached is not None:
            return cached
        cfg, t = self.cfg, ctx.targets
        pred = ctx.pred
        zero = pred["positions"].sum() * 0.0 + 0.0
        out = {name: zero for name in TERM_NAMES}
        if self.return_enabled:
            out.update({name: zero for name in self.return_names})
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
        if self.return_enabled:
            out.update(self._return_components(ctx))
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
    """One legacy or opt-in whole-horizon component of :class:`PairedDiffLoss`."""

    def __init__(self, name: str, shared: PairedDiffLoss, *, owner: bool) -> None:
        if name not in TERM_NAMES + RETURN_TERM_NAMES:
            raise ValueError(f"unknown paired_diff term {name!r}")
        self.name = name
        self.shared = shared
        self.owner = owner

    def prepare(self, targets: UnrollTargets) -> None:
        if self.owner:
            self.shared.prepare(targets)

    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        value = self.shared.components(ctx)[self.name]
        if self.name in RETURN_TERM_NAMES:
            return TermValue(value, log=value.detach() if ctx.c0 == 0 else value.detach() * 0.0)
        return TermValue(value)


def paired_diff_terms(cfg: PairedDiffConfig) -> list[PairedDiffTerm]:
    """Enabled paired-difference chunk terms (empty when the total weight is zero)."""
    if cfg.weight <= 0.0:
        return []
    shared = PairedDiffLoss(cfg)
    names = [name for name, w in zip(TERM_NAMES, (cfg.position_weight, cfg.speed_weight,
                                                   cfg.aggregate_weight)) if w > 0.0]
    names.extend(shared.return_names)
    return [PairedDiffTerm(name, shared, owner=(i == 0)) for i, name in enumerate(names)]
