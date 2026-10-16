"""Merge-zone calibration loss (v7ft, owner-approved 2026-10-17).

Diagnosis (docs/C4_DIAGNOSIS_20261016.md, runs/active/C4-DIAG-20261016/README.md): rolled
120 s from recorded states with CAVs teacher-forced, the v5m WM is too congested at the
merge -- HDV speed in the merge zone (s in [0, 75.44) m) is 1-3.7 m/s below SUMO and the
HDV discharge past the merge end (s = 75.44 m) is 5-8 % lower.  This module adds two
opt-in terms that score exactly those two quantities on the training rollout:

* **speed** (``merge_speed``): per cross-section cell (all lanes; default edges
  ``[-100, 0, 75.44, 300]`` m) the HDV mean speed of the rollout vs the record.  Soft
  cell membership ``sigmoid((s - lo)/tau) - sigmoid((s - hi)/tau)``; sums are pooled over
  ``bin_s`` bins of valid steps; the predicted mean is shrunk toward the (detached)
  recorded mean with ``speed_prior_count`` vehicle-steps, so an empty predicted cell is
  not an error and identical trajectories give exactly zero.  Error =
  Huber((pred - rec) / speed_scale_mps), weighted by the recorded soft vehicle-steps of
  the (row, bin, cell) and normalised by the unroll-wide recorded vehicle-steps
  (``prepare``), so chunk contributions add up to one horizon-invariant mean.
* **discharge** (``merge_discharge``): soft same-vehicle forward crossings of
  ``line_m`` by HDVs (``sigmoid`` width ``cross_temp_m``), accumulated from t0: the
  cumulative difference ``E_t = sum_{u<=t} (crossings_pred - crossings_rec)`` per row,
  scored as the mean over valid steps of Huber(E_t / discharge_scale_veh).  Across
  truncated-BPTT chunks the previous chunks' difference is carried as a detached
  constant (keyed by the chunk start, so a recomputed chunk never double-counts).

HDVs only: CAVs are teacher-forced (the recorded type ``is_cav`` per step; predicted
entrants take the entry process's type).  The recorded side is the same population as the
aggregate loss (``losses.aggregate.road_targets``): the full road incl. arrivals when the
entry process runs and the windows carry it, else the t0 cohort.  Window padding
(recorded tick unavailable) is masked out of both terms.

Validation metrics (no gradient; ``merge_metrics``): ``merge_speed_err_mps`` (recorded
vehicle-step weighted mean |pred - rec| over the bins and cells), ``merge_speed_bias_mps``
(signed) and per cell ``merge_speed_bias_mps_c{i}``; ``merge_discharge_err_frac`` =
(sum pred - sum rec) / sum rec of the HDV crossings over the unroll (signed; negative = the
WM discharges fewer) and ``merge_discharge_abs_err_frac`` = sum |pred - rec| / sum rec.

Flag-gated: with both weights 0 the composer never builds these terms (bit-identical).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .aggregate import _bin_matrix, _huber, soft_crossings
from .base import LossContext, LossTerm, TermValue, UnrollTargets

TERM_NAMES = ("merge_speed", "merge_discharge")
METRIC_NAMES = ("merge_speed_err_mps", "merge_speed_bias_mps", "merge_discharge_err_frac",
                "merge_discharge_abs_err_frac")


@dataclass(frozen=True)
class MergeLossConfig:
    """Weights, cell edges, merge line, smoothing temperatures and normalising scales."""

    dt_s: float = 0.5
    speed_weight: float = 0.0
    discharge_weight: float = 0.0
    cell_edges_m: tuple[float, ...] = (-100.0, 0.0, 75.44, 300.0)
    line_m: float = 75.44
    bin_s: float = 5.0
    cell_temp_m: float = 2.0
    cross_temp_m: float = 5.0
    speed_scale_mps: float = 1.0
    discharge_scale_veh: float = 2.0
    speed_prior_count: float = 1.0
    huber_delta: float = 1.0

    def __post_init__(self) -> None:
        edges = tuple(float(x) for x in self.cell_edges_m)
        if len(edges) < 2 or any(not math.isfinite(x) for x in edges) or any(
                b <= a for a, b in zip(edges[:-1], edges[1:])):
            raise ValueError("merge loss cell_edges_m must be >= 2 finite increasing values")
        if min(self.speed_weight, self.discharge_weight) < 0.0 or not all(
                math.isfinite(x) for x in (self.speed_weight, self.discharge_weight)):
            raise ValueError("merge loss weights must be finite and non-negative")
        if not math.isfinite(self.line_m):
            raise ValueError("merge loss line_m must be finite")
        for value in (self.dt_s, self.bin_s, self.cell_temp_m, self.cross_temp_m,
                      self.speed_scale_mps, self.discharge_scale_veh, self.speed_prior_count,
                      self.huber_delta):
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError("merge loss temperatures/scales must be finite and positive")

    @property
    def enabled(self) -> bool:
        return self.speed_weight > 0.0 or self.discharge_weight > 0.0

    @property
    def bin_steps(self) -> int:
        return max(1, int(round(self.bin_s / self.dt_s)))


def cell_sums(s: Tensor, v: Tensor, hdv: Tensor, edges, temp_m: float) -> tuple[Tensor, Tensor]:
    """Soft HDV count and speed sum per cross-section cell, ``[B, T, C]`` each."""
    mask = hdv.to(s.dtype)
    sig = torch.sigmoid((s.unsqueeze(-1) - s.new_tensor(list(edges))) / float(temp_m))
    member = (sig[..., :-1] - sig[..., 1:]) * mask.unsqueeze(-1)            # [B,T,K,C]
    return member.sum(-2), (member * v.unsqueeze(-1)).sum(-2)


def _recorded_is_cav(targets: UnrollTargets, k: int) -> Tensor:
    cav = targets.is_cav.bool()
    if cav.shape[-1] >= k:
        return cav[..., :k]
    pad = cav.new_zeros((*cav.shape[:-1], k - cav.shape[-1]))
    return torch.cat((cav, pad), -1)


def predicted_hdv_type(ctx: LossContext, k: int) -> Tensor:
    """Per-step predicted CAV flag ``[B, T, k]``: recorded type; entrants: entry-process type."""
    t = ctx.targets
    cav = _recorded_is_cav(t, k)[:, ctx.c0 + 1:ctx.c1 + 1]
    entrant = ctx.pred.get("entrant_seq")
    state = ctx.pred.get("state")
    if (torch.is_tensor(entrant) and isinstance(state, dict)
            and torch.is_tensor(state.get("is_cav"))):
        ent = entrant.bool()[..., :k]
        end_cav = state["is_cav"].bool()[..., :k].unsqueeze(1)
        if ent.shape == cav.shape:
            cav = torch.where(ent, end_cav.expand_as(cav), cav)
    return cav


class MergeLoss:
    """Shared per-chunk computation of the two merge terms (cached on ``ctx``)."""

    def __init__(self, cfg: MergeLossConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------ recorded side (once)
    def prepare(self, targets: UnrollTargets) -> None:
        road = targets.road
        if road is None:
            raise ValueError("merge loss requires UnrollTargets.road")
        cfg, n = self.cfg, targets.n_steps
        with torch.no_grad():
            rec = targets.states[:, :n + 1, :road.k]
            rs, rv = rec[..., 0], rec[..., 4]
            hdv = road.present[:, :n + 1].bool() & ~_recorded_is_cav(targets, road.k)[:, :n + 1]
            valid = road.valid[:, 1:n + 1].to(rs.dtype)                     # [B, n]
            count, _ = cell_sums(rs[:, 1:], rv[:, 1:], hdv[:, 1:], cfg.cell_edges_m,
                                 cfg.cell_temp_m)
            cont = road.cont[:, :n].bool() & hdv[:, :-1] & hdv[:, 1:]
        zeros = rs.new_zeros(rs.shape[0])
        # The per-chunk recorded sums are recomputed in ``components`` with exactly the
        # predicted side's operations, so identical trajectories give an exact zero.
        targets.prepared["merge"] = {
            "rec_s": rs, "rec_v": rv, "rec_hdv": hdv, "rec_cont": cont, "valid": valid,
            "w_speed": float((count * valid.unsqueeze(-1)).sum()),
            "w_discharge": float(valid.sum()),
            "carry": {0: zeros}, "pred_total": {0: zeros}, "rec_total": {0: zeros},
            "speed_stats": {}, "n_cells": len(cfg.cell_edges_m) - 1}

    # ------------------------------------------------------------ per chunk
    def components(self, ctx: LossContext) -> dict[str, Tensor]:
        cached = ctx.cache.get("merge")
        if cached is not None:
            return cached
        cfg, t = self.cfg, ctx.targets
        road = t.road
        prep = t.prepared["merge"]
        c0, c1 = ctx.c0, ctx.c1
        k = None if road.full else road.k

        def cut(x):
            return x if k is None else x[..., :k]

        ps, pv, pm = cut(ctx.pred["positions"]), cut(ctx.pred["speeds"]), cut(ctx.pred["masks"])
        kk = ps.shape[-1]
        hdv = pm.bool() & ~predicted_hdv_type(ctx, kk)
        prev_s = torch.cat((cut(ctx.current["s_m"]).unsqueeze(1), ps[:, :-1]), 1)
        prev_hdv = torch.cat((cut(ctx.current["present"]).bool().unsqueeze(1), hdv[:, :-1]), 1)
        vmask = prep["valid"][:, c0:c1]                                       # [B, T]
        zero = ps.sum() * 0.0 + 0.0          # graph-connected +0.0 (never -0.0 in logs)
        out = {name: zero for name in TERM_NAMES}

        # ---- speed: bin-pooled cell means, recorded-vehicle-step weighted Huber error
        idx = torch.arange(c0 + 1, c1 + 1, device=ps.device)
        bins = _bin_matrix(idx, cfg.bin_steps, ps.dtype)                     # [T, J]
        pcount, pvsum = cell_sums(ps, pv, hdv, cfg.cell_edges_m, cfg.cell_temp_m)
        np_ = torch.einsum("btc,bt,tj->bjc", pcount, vmask, bins)
        p_ = torch.einsum("btc,bt,tj->bjc", pvsum, vmask, bins)
        rs, rv = prep["rec_s"], prep["rec_v"]
        rcount, rvsum = cell_sums(rs[:, c0 + 1:c1 + 1], rv[:, c0 + 1:c1 + 1],
                                  prep["rec_hdv"][:, c0 + 1:c1 + 1], cfg.cell_edges_m,
                                  cfg.cell_temp_m)
        nr = torch.einsum("btc,bt,tj->bjc", rcount, vmask, bins)
        r_ = torch.einsum("btc,bt,tj->bjc", rvsum, vmask, bins)
        # pred mean shrunk toward the recorded mean with ``prior`` vehicle-steps:
        # (p + prior r/nr)/(np + prior) - r/nr == (p nr - r np) / (nr (np + prior))
        prior = float(cfg.speed_prior_count)
        err = (p_ * nr - r_ * np_) / (nr.clamp_min(1e-6) * (np_ + prior))
        if cfg.speed_weight > 0.0 and prep["w_speed"] > 0.0:
            out["merge_speed"] = (cfg.speed_weight
                                  * (nr * _huber(err / float(cfg.speed_scale_mps),
                                                 cfg.huber_delta)).sum() / prep["w_speed"])
        with torch.no_grad():
            prep["speed_stats"][c0] = (
                (nr * err.abs()).sum((0, 1)).detach(), (nr * err).sum((0, 1)).detach(),
                nr.sum((0, 1)).detach())

        # ---- discharge: cumulative soft-crossing difference since t0 (carried across chunks)
        cont = prev_hdv & hdv
        xp = soft_crossings(prev_s, ps, cont, (float(cfg.line_m),),
                            float(cfg.cross_temp_m))[..., 0] * vmask         # [B, T]
        xr = soft_crossings(rs[:, c0:c1], rs[:, c0 + 1:c1 + 1], prep["rec_cont"][:, c0:c1],
                            (float(cfg.line_m),), float(cfg.cross_temp_m))[..., 0] * vmask
        carry = prep["carry"].get(c0, xr.new_zeros(xr.shape[0]))
        cum = carry.unsqueeze(1) + torch.cumsum(xp - xr, dim=1)
        if cfg.discharge_weight > 0.0 and prep["w_discharge"] > 0.0:
            out["merge_discharge"] = (cfg.discharge_weight
                                      * (vmask * _huber(cum / float(cfg.discharge_scale_veh),
                                                        cfg.huber_delta)).sum()
                                      / prep["w_discharge"])
        with torch.no_grad():
            prep["carry"][c1] = cum[:, -1].detach()
            prep["pred_total"][c1] = (prep["pred_total"].get(c0, carry * 0.0)
                                      + xp.sum(1).detach())
            prep["rec_total"][c1] = prep["rec_total"].get(c0, carry * 0.0) + xr.sum(1)
        ctx.cache["merge"] = out
        return out


def merge_metrics(prep: Optional[dict], n_steps: int) -> dict[str, float]:
    """Whole-unroll validation metrics from the per-chunk statistics (empty if absent)."""
    if not prep or not prep.get("speed_stats"):
        return {}
    stats = list(prep["speed_stats"].values())
    abs_c = sum(s[0] for s in stats)
    sgn_c = sum(s[1] for s in stats)
    w_c = sum(s[2] for s in stats)
    w = float(w_c.sum())
    out = {"merge_speed_err_mps": float(abs_c.sum()) / w if w > 0 else 0.0,
           "merge_speed_bias_mps": float(sgn_c.sum()) / w if w > 0 else 0.0}
    for i in range(int(prep["n_cells"])):
        wi = float(w_c[i])
        out[f"merge_speed_bias_mps_c{i}"] = float(sgn_c[i]) / wi if wi > 0 else 0.0
    last = max(prep["pred_total"])
    pred, rec = prep["pred_total"][last], prep["rec_total"][last]
    rec_sum = float(rec.sum())
    out["merge_discharge_err_frac"] = (float(pred.sum()) - rec_sum) / rec_sum if rec_sum > 0 else 0.0
    out["merge_discharge_abs_err_frac"] = (float((pred - rec).abs().sum()) / rec_sum
                                           if rec_sum > 0 else 0.0)
    out["merge_discharge_pred_veh"] = float(pred.sum())
    out["merge_discharge_rec_veh"] = rec_sum
    return out


class MergeTerm(LossTerm):
    """One component (``merge_speed`` / ``merge_discharge``) of :class:`MergeLoss`."""

    def __init__(self, name: str, shared: MergeLoss, *, owner: bool) -> None:
        if name not in TERM_NAMES:
            raise ValueError(f"unknown merge term {name!r}")
        self.name = name
        self.shared = shared
        self.owner = owner          # the first enabled term prepares the unroll totals

    def prepare(self, targets: UnrollTargets) -> None:
        if self.owner:
            self.shared.prepare(targets)

    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        return TermValue(self.shared.components(ctx)[self.name])


def merge_terms(cfg: MergeLossConfig) -> list[MergeTerm]:
    """Enabled merge chunk terms (empty when both weights are zero)."""
    if not cfg.enabled:
        return []
    shared = MergeLoss(cfg)
    names = [name for name, w in zip(TERM_NAMES, (cfg.speed_weight, cfg.discharge_weight))
             if w > 0.0]
    return [MergeTerm(name, shared, owner=(i == 0)) for i, name in enumerate(names)]


__all__ = ["MergeLossConfig", "MergeLoss", "MergeTerm", "merge_terms", "merge_metrics",
           "cell_sums", "TERM_NAMES", "METRIC_NAMES"]
