"""R5THR integrated, censored first-passage merge loss (``loss.merge_first_passage``).

Scores the SAME recorded ramp car's merge interval on the absolute clock, on the free-running
rollout states of the unroll, with the per-driver threshold ``u`` integrated out:

* ``z_t`` = the lane-v5 merge-head logit on the model's own rolled-out state (with gradient);
* ``M_t`` = running maximum of ``z`` over ENVELOPE-FEASIBLE in-zone steps only (an infeasible
  step never enters, so it can never create event mass);
* ``F`` = unit-logistic CDF at the learned location ``loc``; the event mass is
  ``F(M_m) - F(M_{m-1})``, right censoring is ``1 - F(M_c)``, an interval ``(l, r]`` is
  ``F(M_r) - F(M_l)``.  Log-CDF differences are evaluated in logit space (stable).
* Chunks telescope: a censored chunk contributes ``log S(M_{c0-1}) - log S(M_L)`` and an event
  chunk contributes ``log(mass) - log S(M_{c0-1})`` (conditional on survival to the chunk
  start), so truncated-BPTT chunks add up to the whole-unroll likelihood.  Delayed entry is
  conditioned on survival before the car's first at-risk step (``S(-inf) = 1``).
* ``rank_weight``: warm-start structured ranking ``softplus(M_{m-1} - z_m)`` on events (a later
  event below an earlier feasible maximum has exactly zero likelihood mass).
* A recorded event below an earlier feasible maximum is counted as ``unsupported_events`` and
  floored at :data:`wms.model.merge_threshold.LOG_TINY` instead of being hidden.

On-policy safety (advice section 2): the recorded time is valid supervision only while the
interaction stays comparable.  A DETACHED compatibility mask predeclared by the config
(``pos_tol_m``, ``speed_tol_mps``, ``gap_tol_m``, ``gap_speed_tol_mps``) compares ego
position/speed and the target-lane leader/follower configuration; the car is censored at the
last compatible step.  The recorded merge is never forced into a different gap.  A model merge
before the recorded one is not benign censoring: it is censored with the still-valid recorded
survival and counted in ``early_merges``.

Diagnostics: ``merge_fp_nll``, ``merge_fp_events``, ``merge_fp_censored_admin``,
``merge_fp_censored_divergence`` (+ by wait-time bin), ``merge_fp_unsupported_events``,
``merge_fp_early_merges``, ``merge_fp_location_reg``.

Flag-gated: ``loss.merge_first_passage.weight = 0`` and ``metrics = false`` never build the
term.  When on it replaces ``loss.merge_timing`` / ``loss.merge_survival`` (composer).
"""
from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from wms.model.merge_threshold import (log_event_mass, log_survival,
                                       target_gap_configuration)

from .base import LossContext, LossTerm, TermValue, UnrollTargets

TERM_NAME = "merge_first_passage"
#: Divergence-censoring wait-time bins (seconds) for the diagnostic counters.
WAIT_BINS_S = ((0.0, 10.0), (10.0, 30.0), (30.0, 60.0), (60.0, float("inf")))


@dataclass(frozen=True)
class MergeFirstPassageConfig:
    """Weights, geometry and the predeclared compatibility tolerances."""

    weight: float = 0.0
    rank_weight: float = 0.0
    metrics: bool = False
    pos_tol_m: float = 25.0
    speed_tol_mps: float = 5.0
    gap_tol_m: float = 20.0
    gap_speed_tol_mps: float = 5.0
    location_reg_weight: float = 1.0e-3
    merge_lane: int = 2
    target_lane: int = 1
    zone_s0_m: float = 0.0
    zone_s1_m: float = 75.44

    def __post_init__(self) -> None:
        for name in ("weight", "rank_weight", "location_reg_weight"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"merge_first_passage {name} must be finite and >= 0")
        for name in ("pos_tol_m", "speed_tol_mps", "gap_tol_m", "gap_speed_tol_mps"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"merge_first_passage {name} must be finite and positive")

    @property
    def enabled(self) -> bool:
        return self.weight > 0.0 or self.rank_weight > 0.0 or bool(self.metrics)


def _first_index(mask: Tensor, dim: int = 1) -> Tensor:
    """Index of the first True along ``dim`` (-1 where none)."""
    n = mask.shape[dim]
    idx = torch.arange(n, device=mask.device).view(
        *([1] * dim), n, *([1] * (mask.ndim - dim - 1)))
    big = torch.where(mask, idx.expand_as(mask), torch.full_like(mask, n, dtype=torch.long))
    first = big.min(dim).values
    return torch.where(first >= n, torch.full_like(first, -1), first)


def _last_index(mask: Tensor, dim: int = 1) -> Tensor:
    """Index of the last True along ``dim`` (-1 where none)."""
    n = mask.shape[dim]
    idx = torch.arange(n, device=mask.device).view(
        *([1] * dim), n, *([1] * (mask.ndim - dim - 1)))
    small = torch.where(mask, idx.expand_as(mask), torch.full_like(mask, -1, dtype=torch.long))
    return small.max(dim).values


def _kin(s: Tensor, lane: Tensor, v: Tensor, a: Tensor, length: Tensor) -> Tensor:
    """``[B, T, K, 7]`` kinematics from per-slot fields."""
    onehot = F.one_hot(lane.clamp(0, 2), 3).to(s.dtype)
    return torch.cat((s.unsqueeze(-1), onehot, v.unsqueeze(-1), a.unsqueeze(-1),
                      length.unsqueeze(-1)), -1)


def _gap_compatible(cfg: MergeFirstPassageConfig, pred_kin: Tensor, rec_kin: Tensor,
                    pred_present: Tensor, rec_present: Tensor) -> Tensor:
    """Detached target-gap compatibility of the model vs the record on the same slot."""
    shim = types.SimpleNamespace(merge_lane=int(cfg.merge_lane),
                                 merge_target_lane=int(cfg.target_lane))
    with torch.no_grad():
        p = target_gap_configuration(pred_kin, pred_present, shim)
        r = target_gap_configuration(rec_kin, rec_present, shim)
        same = (p["has_f"] == r["has_f"]) & (p["has_l"] == r["has_l"])
        gap_ok = (p["gap_f"] - r["gap_f"]).abs() <= float(cfg.gap_tol_m)
        gap_ok = gap_ok & ((p["gap_l"] - r["gap_l"]).abs() <= float(cfg.gap_tol_m))
        rel_ok = (p["rel_f"] - r["rel_f"]).abs() <= float(cfg.gap_speed_tol_mps)
        rel_ok = rel_ok & ((p["rel_l"] - r["rel_l"]).abs() <= float(cfg.gap_speed_tol_mps))
        ok = same & gap_ok & rel_ok
        # Only ramp slots are constrained; absent/other slots are neutral.
        return torch.where(p["ramp"], ok, torch.ones_like(ok))


def _wait_bin(wait_s: Tensor) -> Tensor:
    """Index of the wait-time bin (``WAIT_BINS_S``) of every censored car."""
    out = torch.zeros_like(wait_s, dtype=torch.long)
    for i, (lo, _hi) in enumerate(WAIT_BINS_S):
        out = torch.where(wait_s >= float(lo), torch.full_like(out, i), out)
    return out


class MergeFirstPassageTerm(LossTerm):
    """One chunk of the integrated censored first-passage NLL."""

    name = TERM_NAME

    def __init__(self, cfg: MergeFirstPassageConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------ recorded side (once)
    def prepare(self, targets: UnrollTargets) -> None:
        cfg, n = self.cfg, targets.n_steps
        with torch.no_grad():
            rec = targets.states[:, :n + 1].float()
            k = rec.shape[2]
            onehot = rec[..., 1:4]
            lane = onehot.argmax(-1)
            lane = torch.where(onehot.sum(-1) > 0, lane, torch.full_like(lane, -1))
            cav = targets.is_cav[:, :n + 1, :k].bool()
            valid = targets.loss_valid[:, :n + 1, :k].bool() & ~cav
            ramp = valid[:, :n] & (lane[:, :n] == int(cfg.merge_lane))
            event = ramp & valid[:, 1:n + 1] & (lane[:, 1:n + 1] == int(cfg.target_lane))
            rec_event = _first_index(event)
            entry = _first_index(ramp)
            last_risk = _last_index(ramp)
            ramp_any = ramp.any(1)
            targets.prepared["merge_first_passage"] = {
                "rec_s": rec[:, :n, :k, 0], "rec_v": rec[:, :n, :k, 4],
                "rec_kin": rec[:, :n], "rec_present": valid[:, :n],
                "ramp": ramp, "event": event, "rec_event": rec_event, "entry": entry,
                "last_risk": last_risk, "ramp_any": ramp_any,
                "n_cars": float(ramp_any.sum()), "k": k, "n_steps": int(n),
                "dt_s": float(targets.dt_s),
                "m_carry": {0: torch.full(rec_event.shape, float("-inf"),
                                          dtype=rec.dtype, device=rec_event.device)},
                "pred_merge": {0: torch.full_like(rec_event, -1)},
                "stats": {},
            }

    # ------------------------------------------------------------ per chunk
    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        cfg, t = self.cfg, ctx.targets
        prep = t.prepared["merge_first_passage"]
        c0, c1, k = ctx.c0, ctx.c1, prep["k"]
        T = c1 - c0
        pred = ctx.pred
        head = pred.get("lane_v5_head_logit")
        if T <= 0 or not torch.is_tensor(head):
            return None

        def cut(x: Tensor) -> Tensor:
            return x[..., :k] if x.shape[-1] > k else x

        lanes = cut(pred["lanes"]).long()
        masks = cut(pred["masks"]).bool()
        pos = cut(pred["positions"])
        spd = cut(pred["speeds"])
        acc = cut(pred["accel"])
        lane_dec = torch.cat((cut(ctx.current["lane"]).long().unsqueeze(1), lanes[:, :-1]), 1)
        pres_dec = torch.cat((cut(ctx.current["present"]).bool().unsqueeze(1), masks[:, :-1]), 1)
        s_dec = torch.cat((cut(ctx.current["s_m"]).unsqueeze(1), pos[:, :-1]), 1)
        v_dec = torch.cat((cut(ctx.current["v_mps"]).unsqueeze(1), spd[:, :-1]), 1)
        a_dec = torch.cat((cut(ctx.current["a_mps2"]).unsqueeze(1), acc[:, :-1]), 1)
        length = prep["rec_kin"][:, c0:c1, :, 6]
        z = cut(head)[..., :k]
        acts = cut(pred["lane_v5_acts"]).bool() if torch.is_tensor(
            pred.get("lane_v5_acts")) else torch.ones_like(pres_dec)
        loc = pred.get("merge_threshold_location")
        loc = torch.zeros_like(z) if not torch.is_tensor(loc) else cut(loc)[..., :k]

        # The recorded car's own slots only: no CAVs, no post-t0 entrants (no record).
        rec_cav = t.is_cav[:, c0:c1, :k].bool()
        entrant = pred.get("entrant_seq")
        ent = (cut(entrant)[:, :, :k].bool() if torch.is_tensor(entrant)
               and cut(entrant).shape[:2] == pres_dec.shape else torch.zeros_like(pres_dec))
        own = ~rec_cav & ~ent
        ramp_dec = pres_dec & (lane_dec == int(cfg.merge_lane)) & own
        in_zone = ramp_dec & (s_dec >= float(cfg.zone_s0_m)) & (s_dec <= float(cfg.zone_s1_m))
        feasible = in_zone & acts & own

        # ---- running maximum, reset on a new car in the slot; gradient through the max
        m_prev = prep["m_carry"].get(c0)
        if m_prev is None:
            m_prev = torch.full((z.shape[0], k), float("-inf"), dtype=z.dtype, device=z.device)
        prev_pres = torch.cat((cut(ctx.current["present"]).bool().unsqueeze(1),
                               pres_dec[:, :-1]), 1)
        new_car = pres_dec & ~prev_pres
        # A slot reused at the chunk boundary is a new car: do not inherit its maximum.
        carried_present = ctx.current.get("merge_fp_present")
        if torch.is_tensor(carried_present):
            new0 = pres_dec[:, 0] & ~carried_present.bool()
            m_prev = torch.where(new0, torch.full_like(m_prev, float("-inf")), m_prev)
        m_list: list[Tensor] = []
        m = m_prev
        for i in range(T):
            m = torch.where(new_car[:, i] | ~pres_dec[:, i],
                            torch.full_like(m, float("-inf")), m)
            m = torch.where(feasible[:, i], torch.maximum(m, z[:, i]), m)
            m_list.append(m)
        m_seq = torch.stack(m_list, 1)                                  # [B, T, K]
        m_before = torch.stack([m_prev] + m_list[:-1], 1)
        prep["m_carry"][c1] = m_seq[:, -1].detach()

        # ---- detached compatibility mask (same car, absolute clock)
        pred_kin = _kin(s_dec, lane_dec, v_dec, a_dec, length)
        pos_ok = (s_dec - prep["rec_s"][:, c0:c1]).abs() <= float(cfg.pos_tol_m)
        spd_ok = (v_dec - prep["rec_v"][:, c0:c1]).abs() <= float(cfg.speed_tol_mps)
        gap_ok = _gap_compatible(cfg, pred_kin, prep["rec_kin"][:, c0:c1],
                                 pres_dec, prep["rec_present"][:, c0:c1])
        compat = (pos_ok & spd_ok & gap_ok) | ~ramp_dec
        alive = torch.cumprod(compat.float(), 1) > 0.5
        idx = torch.arange(c0, c1, device=z.device).view(1, T, 1)
        last_ok = torch.where(ramp_dec & alive, idx.expand_as(ramp_dec),
                              torch.full_like(idx, -1)).max(1).values

        # ---- model's own first merge (global, carried across chunks; reset on slot reuse)
        departed = ramp_dec & masks & (lanes != int(cfg.merge_lane)) & ~new_car
        before_merge = prep["pred_merge"].get(c0,
                                              torch.full_like(departed[:, 0], -1))
        pred_merge = before_merge
        for i in range(T):
            pred_merge = torch.where(new_car[:, i],
                                     torch.full_like(pred_merge, -1), pred_merge)
            pred_merge = torch.where((pred_merge < 0) & departed[:, i],
                                     torch.full_like(pred_merge, c0 + i), pred_merge)
        prep["pred_merge"][c1] = pred_merge

        rec_event = prep["rec_event"]
        entry = prep["entry"]
        early = (pred_merge >= 0) & ((rec_event < 0) | (pred_merge < rec_event))
        in_chunk = (rec_event >= c0) & (rec_event < c1)
        if bool(in_chunk.any()):
            local_e = (rec_event - c0).clamp(0, T - 1)
            at_event_step = F.one_hot(local_e, T).bool().permute(0, 2, 1)
        else:
            at_event_step = torch.zeros_like(ramp_dec)
        take_event = (in_chunk.unsqueeze(1) & at_event_step & pres_dec & masks
                      & ~early.unsqueeze(1) & (last_ok >= rec_event).unsqueeze(1))
        event_any = take_event.any(1)

        def pick(x: Tensor, mask: Tensor) -> Tensor:
            # ``0 * -inf`` would be NaN when the running maximum is still empty.
            return torch.where(mask, x, torch.zeros_like(x)).sum(1)

        m_e = pick(m_seq, take_event)
        m_prev_e = pick(m_before, take_event)
        loc_e = pick(loc, take_event)
        z_e = pick(z, take_event)
        mass = torch.sigmoid(m_e - loc_e) - torch.sigmoid(m_prev_e - loc_e)
        supported = event_any & (mass > 1.0e-12)
        log_mass = log_event_mass(m_prev_e, m_e, loc_e)
        # Survival before the car's delayed entry (chunk start, or its first at-risk step).
        den_idx = (entry.clamp_min(c0) - c0).clamp(0, T - 1)
        m_den = torch.zeros_like(m_prev)
        for i in range(T):
            m_den = torch.where(den_idx == i, m_before[:, i], m_den)
        log_denom = log_survival(m_den, loc[:, 0])

        # Censor contribution at the last compatible step (compatibility, early merge,
        # window end).  A car already merged in the record before this chunk is skipped.
        last = torch.full_like(rec_event, c1 - 1)
        last = torch.where((last_ok >= 0) & (last_ok < last), last_ok, last)
        last = torch.where(early & (pred_merge >= 0) & (pred_merge < last), pred_merge, last)
        has_censor = (~event_any) & (last >= entry) & (last >= c0) & (
            (rec_event < 0) | (rec_event >= c1) | early | (last_ok < rec_event))
        last_local = (last - c0).clamp(0, T - 1)
        sel_c = (F.one_hot(last_local, T).bool().permute(0, 2, 1)
                 & has_censor.unsqueeze(1))
        m_c = pick(m_seq, sel_c)
        loc_c = pick(loc, sel_c)
        logp_c = log_survival(m_c, loc_c) - log_denom
        logp_e = log_mass - log_denom

        nll = torch.where(event_any, -logp_e, torch.zeros_like(logp_e))
        nll = torch.where(has_censor, -logp_c, nll)
        zero = z.sum() * 0.0 + 0.0
        n_cars = max(float(prep["n_cars"]), 1.0)

        # Warm-start ranking: a recorded event below an earlier feasible maximum has no mass.
        rank = F.softplus(m_prev_e.detach() - z_e)
        rank_sum = (rank * event_any.to(rank.dtype)).sum()
        loc_reg = (loc * ramp_dec).pow(2).sum() / max(float(ramp_dec.sum()), 1.0)

        value = None
        if cfg.weight > 0.0 or cfg.rank_weight > 0.0 or cfg.location_reg_weight > 0.0:
            value = (float(cfg.weight) * nll.sum() / n_cars
                     + float(cfg.rank_weight) * rank_sum / n_cars
                     + float(cfg.location_reg_weight) * loc_reg + zero)

        # ---- diagnostics (no gradient)
        with torch.no_grad():
            wait = ((last - entry).clamp_min(0).to(z.dtype) * float(prep["dt_s"]))
            wb = _wait_bin(wait)
            stats = prep["stats"].setdefault(c0, {
                "nll": 0.0, "events": 0.0, "unsupported": 0.0, "early": 0.0,
                "censored_admin": 0.0, "censored_divergence": 0.0, "location_reg": 0.0,
                "wait_bins": [0.0] * len(WAIT_BINS_S)})
            stats["nll"] += float(nll.detach().sum())
            stats["events"] += float(event_any.sum())
            stats["unsupported"] += float((event_any & ~supported).sum())
            stats["early"] += float((early & (before_merge < 0) & (pred_merge >= 0)).sum())
            divergent = has_censor & ((last_ok < c1 - 1) | early)
            stats["censored_divergence"] += float(divergent.sum())
            stats["censored_admin"] += float(
                (has_censor & ~divergent & (rec_event < 0) & (c1 == prep["n_steps"])).sum())
            stats["location_reg"] += float(loc_reg.detach())
            bins = stats["wait_bins"]
            for i in range(len(WAIT_BINS_S)):
                bins[i] += float((divergent & (wb == i)).sum())
        return None if value is None else TermValue(value)


def merge_first_passage_metrics(prep: Optional[dict]) -> dict:
    """Whole-unroll diagnostics from the per-chunk statistics (empty if absent)."""
    if not prep or not prep.get("stats"):
        return {}
    stats = list(prep["stats"].values())

    def total(key: str) -> float:
        return sum(float(s.get(key, 0.0)) for s in stats)

    n_cars = max(float(prep["n_cars"]), 1.0)
    bins = [sum(float(s["wait_bins"][i]) for s in stats)
            for i in range(len(WAIT_BINS_S))]
    out = {
        "merge_fp_nll": total("nll") / n_cars,
        "merge_fp_events": total("events"),
        "merge_fp_censored_admin": total("censored_admin"),
        "merge_fp_censored_divergence": total("censored_divergence"),
        "merge_fp_unsupported_events": total("unsupported"),
        "merge_fp_early_merges": total("early"),
        "merge_fp_location_reg": total("location_reg"),
        "merge_fp_n_cars": n_cars,
    }
    for i, (lo, hi) in enumerate(WAIT_BINS_S):
        key = f"merge_fp_censored_divergence_wait_{int(lo)}_{'inf' if math.isinf(hi) else int(hi)}"
        out[key] = bins[i]
    return out


def merge_first_passage_terms(cfg: MergeFirstPassageConfig) -> list[MergeFirstPassageTerm]:
    """The first-passage chunk term (empty when weight, rank and metrics are all off)."""
    return [MergeFirstPassageTerm(cfg)] if cfg.enabled else []


__all__ = ["MergeFirstPassageConfig", "MergeFirstPassageTerm", "merge_first_passage_terms",
           "merge_first_passage_metrics", "TERM_NAME", "WAIT_BINS_S"]
