"""Frailty, censored discrete-time survival merge loss (``loss.merge_frailty``).

Scores the SAME recorded ramp car's merge interval on the absolute clock, on the
free-running rollout states of the unroll, with the per-driver latent ``w`` integrated out
by Gauss-Hermite quadrature:

* ``z_t`` = the lane-v5 merge-head logit on the model's own rolled-out state (with gradient),
  plus the negotiation delta when that block is active;
* per-step hazard ``h_t(w) = sigmoid(z_t + w)`` on ENVELOPE-FEASIBLE in-zone steps only
  (infeasible steps give no merge opportunity, factor 1);
* ``w = loc + sigma * eps``, ``eps ~ Normal(0, 1)``; ``loc`` / ``sigma`` are the model's
  learned frailty location / positive scale;
* event likelihood ``prod_{t < T} (1 - h_t) * h_T`` and right-censoring
  ``prod_{t <= c} (1 - h_t)``, conditioned by construction on delayed entry (steps before
  the car's first at-risk step never contribute);
* ``log L = log sum_q w_q [ product ... ]`` with Gaussian quadrature nodes/weights.

Unlike the first-passage threshold, ``h_T > 0`` for every finite score, so **every recorded
merge has positive likelihood** even when it happens below an earlier feasible maximum.  The
familiar ``prod(1-h)`` is evaluated in log space (``logsigmoid``), so no zero-mass event is
hidden behind an epsilon; a genuinely non-finite car mass is counted in
``merge_frailty_zero_likelihood_events``.

On-policy safety reuses the R5THR machinery: a DETACHED compatibility mask (ego
position/speed and the target-lane leader/follower configuration, predeclared tolerances,
entrant mask fix) censors at the last comparable step, and a model merge before the recorded
one is censored with the still-valid recorded survival and counted in ``early_merges``.
Chunks telescope multiplicatively in log space, so truncated-BPTT chunks add up (each chunk
integrates the latent with its own prior; the rollout carries the sampled latent).

Flag-gated: ``loss.merge_frailty.weight = 0`` and ``metrics = false`` never build the term.
When on it REPLACES ``loss.merge_first_passage`` / ``merge_timing`` / ``merge_survival`` for
the merge head (composer).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from wms.model.merge_frailty import gauss_hermite

from .base import LossContext, LossTerm, TermValue, UnrollTargets
from .merge_first_passage import (WAIT_BINS_S, _first_index, _gap_compatible, _kin,
                                  _last_index, _wait_bin)

TERM_NAME = "merge_frailty"
#: Floor used only to classify a numerically dead likelihood for the diagnostic counter.
LOG_LIKELIHOOD_FLOOR = math.log(1.0e-30)


@dataclass(frozen=True)
class MergeFrailtyConfig:
    """Weights, quadrature order, geometry and predeclared compatibility tolerances."""

    weight: float = 0.0
    n_quad: int = 8
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
        for name in ("weight", "location_reg_weight"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"merge_frailty {name} must be finite and >= 0")
        if not (1 <= int(self.n_quad) <= 64):
            raise ValueError("merge_frailty n_quad must be in [1, 64]")
        for name in ("pos_tol_m", "speed_tol_mps", "gap_tol_m", "gap_speed_tol_mps"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"merge_frailty {name} must be finite and positive")

    @property
    def enabled(self) -> bool:
        return self.weight > 0.0 or bool(self.metrics)


class MergeFrailtyTerm(LossTerm):
    """One chunk of the frailty censored discrete-time survival NLL."""

    name = TERM_NAME

    def __init__(self, cfg: MergeFrailtyConfig) -> None:
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
            targets.prepared["merge_frailty"] = {
                "rec_s": rec[:, :n, :k, 0], "rec_v": rec[:, :n, :k, 4],
                "rec_kin": rec[:, :n], "rec_present": valid[:, :n],
                "ramp": ramp, "event": event, "rec_event": rec_event, "entry": entry,
                "last_risk": last_risk, "ramp_any": ramp_any,
                "n_cars": float(ramp_any.sum()), "k": k, "n_steps": int(n),
                "dt_s": float(targets.dt_s),
                "pred_merge": {0: torch.full_like(rec_event, -1)},
                "diverged": torch.zeros_like(rec_event, dtype=torch.bool),
                "stats": {},
            }

    # ------------------------------------------------------------ per chunk
    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        cfg, t = self.cfg, ctx.targets
        prep = t.prepared["merge_frailty"]
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
        delta = pred.get("merge_negotiation_merge_delta")
        if torch.is_tensor(delta):
            z = z + cut(delta)[..., :k].to(z.dtype)
        acts = cut(pred["lane_v5_acts"]).bool() if torch.is_tensor(
            pred.get("lane_v5_acts")) else torch.ones_like(pres_dec)
        loc = pred.get("merge_frailty_location")
        loc = torch.zeros_like(z) if not torch.is_tensor(loc) else cut(loc)[..., :k].to(z.dtype)
        sigma = pred.get("merge_frailty_sigma")
        if torch.is_tensor(sigma):
            sigma = torch.as_tensor(sigma, dtype=z.dtype, device=z.device)
            sigma = sigma.expand_as(z)
        else:
            sigma = torch.ones_like(z)

        rec_cav = t.is_cav[:, c0:c1, :k].bool()
        entrant = pred.get("entrant_seq")
        ent_seq = cut(entrant) if torch.is_tensor(entrant) else None
        ent = (ent_seq[:, :, :k].bool() if ent_seq is not None
               and ent_seq.shape == pres_dec.shape else torch.zeros_like(pres_dec))
        own = ~rec_cav & ~ent
        ramp_dec = pres_dec & (lane_dec == int(cfg.merge_lane)) & own
        in_zone = ramp_dec & (s_dec >= float(cfg.zone_s0_m)) & (s_dec <= float(cfg.zone_s1_m))
        feasible = in_zone & acts & own

        step = torch.arange(c0, c1, device=z.device).view(1, T, 1)
        entry = prep["entry"]
        at_risk = feasible & (step >= entry.clamp_min(c0).unsqueeze(1))

        # ---- detached compatibility mask (same car, absolute clock)
        pred_kin = _kin(s_dec, lane_dec, v_dec, a_dec, length)
        pos_ok = (s_dec - prep["rec_s"][:, c0:c1]).abs() <= float(cfg.pos_tol_m)
        spd_ok = (v_dec - prep["rec_v"][:, c0:c1]).abs() <= float(cfg.speed_tol_mps)
        gap_ok = _gap_compatible(cfg, pred_kin, prep["rec_kin"][:, c0:c1],
                                 pres_dec & own, prep["rec_present"][:, c0:c1])
        compat = (pos_ok & spd_ok & gap_ok) | ~ramp_dec
        alive = torch.cumprod(compat.float(), 1) > 0.5
        last_ok = torch.where(ramp_dec & alive, step.expand_as(ramp_dec),
                              torch.full_like(step.expand_as(ramp_dec), -1)).max(1).values

        # ---- model's own first merge (global, carried across chunks; reset on slot reuse)
        prev_pres = torch.cat((cut(ctx.current["present"]).bool().unsqueeze(1),
                               pres_dec[:, :-1]), 1)
        new_car = pres_dec & ~prev_pres
        departed = ramp_dec & masks & (lanes != int(cfg.merge_lane)) & ~new_car
        before_merge = prep["pred_merge"].get(c0, torch.full_like(departed[:, 0], -1))
        pred_merge = before_merge
        for i in range(T):
            pred_merge = torch.where(new_car[:, i],
                                     torch.full_like(pred_merge, -1), pred_merge)
            pred_merge = torch.where((pred_merge < 0) & departed[:, i],
                                     torch.full_like(pred_merge, c0 + i), pred_merge)
        prep["pred_merge"][c1] = pred_merge

        rec_event = prep["rec_event"]
        early = (pred_merge >= 0) & ((rec_event < 0) | (pred_merge < rec_event))
        in_chunk = (rec_event >= c0) & (rec_event < c1)
        at_event_step = (F.one_hot((rec_event - c0).clamp(0, T - 1), T).bool()
                         .permute(0, 2, 1)) if bool(in_chunk.any()) else torch.zeros_like(
            ramp_dec)
        take_event = (in_chunk.unsqueeze(1) & at_event_step & pres_dec & masks
                      & ~early.unsqueeze(1) & (last_ok >= rec_event).unsqueeze(1))
        event_any = take_event.any(1)                                    # [B, K]

        last = torch.where(last_ok >= 0, last_ok, torch.full_like(last_ok, -1))
        last = torch.where(early & (pred_merge >= 0) & ((last < 0) | (pred_merge < last)),
                           pred_merge, last)
        censorable = (~event_any) & prep["ramp_any"] & (entry < c1) & (
            (rec_event < 0) | (rec_event >= c1) | early | (last_ok < rec_event))
        has_censor = censorable & (last >= entry) & (last >= c0)

        # ---- Gauss-Hermite integration of the persistent latent w = loc + sigma * eps
        nodes, weights = gauss_hermite(cfg.n_quad, dtype=z.dtype, device=z.device)
        logits = (z.unsqueeze(1) + loc.unsqueeze(1)
                  + sigma.unsqueeze(1) * nodes.view(1, -1, 1, 1))      # [B, Q, T, K]
        log_h = F.logsigmoid(logits)
        log_1mh = F.logsigmoid(-logits)

        # Event: survival over all at-risk steps except the event step, then h at the event.
        sel = take_event.unsqueeze(1)                                    # [B, 1, T, K]
        surv_mask = at_risk.unsqueeze(1) & ~sel
        log_surv = (log_1mh * surv_mask).sum(2)                           # [B, Q, K]
        log_h_e = (log_h * sel).sum(2)                                    # [B, Q, K]
        event_q = log_surv + log_h_e

        # Censoring: survival through the last comparable step (inclusive).
        censor_mask = at_risk & (step <= (last - c0).clamp(-1, T - 1).unsqueeze(1))
        censor_q = (log_1mh * censor_mask.unsqueeze(1)).sum(2)            # [B, Q, K]

        used = event_any | has_censor
        loglik_q = torch.where(event_any.unsqueeze(1), event_q, censor_q)
        logw = torch.log(weights.clamp_min(torch.finfo(weights.dtype).tiny))
        mix = torch.logsumexp(loglik_q + logw.view(1, -1, 1), dim=1)     # [B, K]
        mix = torch.nan_to_num(mix, nan=LOG_LIKELIHOOD_FLOOR,
                               neginf=LOG_LIKELIHOOD_FLOOR)

        nll = -(mix * used.to(mix.dtype)).sum()
        zero_likelihood = event_any & (mix <= LOG_LIKELIHOOD_FLOOR)
        zero = z.sum() * 0.0 + 0.0
        n_cars = max(float(prep["n_cars"]), 1.0)
        loc_reg = (loc * ramp_dec).pow(2).sum() / max(float(ramp_dec.sum()), 1.0)

        value = None
        if cfg.weight > 0.0:
            value = (float(cfg.weight) * nll / n_cars
                     + float(cfg.location_reg_weight) * loc_reg + zero)

        # ---- diagnostics (no gradient)
        with torch.no_grad():
            stats = prep["stats"].setdefault(c0, {
                "nll": 0.0, "events": 0.0, "early": 0.0, "zero": 0.0,
                "censored_admin": 0.0, "censored_divergence": 0.0, "location_reg": 0.0,
                "sigma": 0.0, "location": 0.0, "n_active": 0.0,
                "wait_bins": [0.0] * len(WAIT_BINS_S)})
            stats["nll"] += float(-mix.detach()[used].sum())
            stats["events"] += float(event_any.sum())
            stats["zero"] += float(zero_likelihood.sum())
            stats["early"] += float((early & (before_merge < 0) & (pred_merge >= 0)).sum())
            last_possible = torch.minimum(torch.full_like(last_ok, c1 - 1),
                                          prep["last_risk"])
            reason = censorable & ((last_ok < last_possible) | early)
            divergent = reason & ~prep["diverged"]
            prep["diverged"] = prep["diverged"] | reason
            stats["censored_divergence"] += float(divergent.sum())
            stats["censored_admin"] += float(
                (has_censor & ~divergent & (rec_event < 0) & (c1 == prep["n_steps"])).sum())
            stats["location_reg"] += float(loc_reg.detach())
            active = ramp_dec[:, 0] & own[:, 0]
            stats["n_active"] += float(active.sum())
            stats["location"] += float(loc[:, 0].detach()[active].sum())
            stats["sigma"] += float(sigma[:, 0].detach()[active].sum())
            wait = ((last - entry).clamp_min(0).to(z.dtype) * float(prep["dt_s"]))
            wb = _wait_bin(wait)
            bins = stats["wait_bins"]
            for i in range(len(WAIT_BINS_S)):
                bins[i] += float((divergent & (wb == i)).sum())
        return None if value is None else TermValue(value)


def merge_frailty_metrics(prep: Optional[dict]) -> dict:
    """Whole-unroll diagnostics from the per-chunk statistics (empty if absent)."""
    if not prep or not prep.get("stats"):
        return {}
    stats = list(prep["stats"].values())

    def total(key: str) -> float:
        return sum(float(s.get(key, 0.0)) for s in stats)

    n_cars = max(float(prep["n_cars"]), 1.0)
    n_active = max(total("n_active"), 1.0)
    bins = [sum(float(s["wait_bins"][i]) for s in stats)
            for i in range(len(WAIT_BINS_S))]
    out = {
        "merge_frailty_nll": total("nll") / n_cars,
        "merge_frailty_events": total("events"),
        "merge_frailty_censored_admin": total("censored_admin"),
        "merge_frailty_censored_divergence": total("censored_divergence"),
        "merge_frailty_early_merges": total("early"),
        "merge_frailty_zero_likelihood_events": total("zero"),
        "merge_frailty_location_reg": total("location_reg"),
        "merge_frailty_sigma": total("sigma") / n_active,
        "merge_frailty_location_mean": total("location") / n_active,
        "merge_frailty_n_cars": n_cars,
    }
    for i, (lo, hi) in enumerate(WAIT_BINS_S):
        key = (f"merge_frailty_censored_divergence_wait_{int(lo)}_"
               f"{'inf' if math.isinf(hi) else int(hi)}")
        out[key] = bins[i]
    return out


def merge_frailty_terms(cfg: MergeFrailtyConfig) -> list[MergeFrailtyTerm]:
    """The frailty chunk term (empty when weight and metrics are both off)."""
    return [MergeFrailtyTerm(cfg)] if cfg.enabled else []


__all__ = ["MergeFrailtyConfig", "MergeFrailtyTerm", "merge_frailty_terms",
           "merge_frailty_metrics", "TERM_NAME"]
