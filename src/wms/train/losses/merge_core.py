"""SECUREGAP merge-core losses (``loss.merge_core``).

Two likelihoods and one regression, all flag-gated (every weight 0 = never built):

``weight``
    Censored discrete-time survival NLL of the SAME recorded ramp car's merge on the
    free-running rollout states, with the persistent per-driver assertiveness ``alpha =
    exp(loc + sigma * eps)`` integrated out by Gauss-Hermite quadrature.  The hazard is the
    core's smooth ``sigmoid(margin / temperature)`` on envelope-feasible in-zone steps, so
    every recorded merge has positive likelihood.  The R5FR compatibility censoring is reused
    (same car / absolute clock; censor at the last comparable state and count early merges).

``tf_weight``
    Teacher-forced acceptance likelihood of the core on the RECORDED states: the same
    two-sided secure-gap margin (population alpha integrated by quadrature) is evaluated on
    the recorded kinematics and fit to the recorded next-step merge event.  This directly
    supervises ``s0``, ``T``, ``b`` and ``tau_a``.

``yield_weight``
    Huber follower-yield regression on the recorded pre-merge follower acceleration; it reuses
    the R5YIELD ``YieldHuberTerm`` with ``merge_core_yield_accel`` as the prediction.

The term is created only when ``model.merge_decision='secure_gap'``; a default hazard model
never builds it and is bit-identical.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from wms.model.merge_core import core_features, secure_gap_sides
from wms.model.merge_frailty import gauss_hermite

from .base import LossContext, LossTerm, TermValue, UnrollTargets
from .merge_first_passage import (WAIT_BINS_S, _first_index, _gap_compatible, _kin,
                                  _last_index, _wait_bin)

TERM_NAME = "merge_core"
LOG_LIKELIHOOD_FLOOR = math.log(1.0e-30)


@dataclass(frozen=True)
class MergeCoreConfig:
    """Weights, quadrature order, geometry and predeclared compatibility tolerances."""

    weight: float = 0.0
    tf_weight: float = 0.0
    yield_weight: float = 0.0
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
        for name in ("weight", "tf_weight", "yield_weight", "location_reg_weight"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"merge_core {name} must be finite and >= 0")
        if not (1 <= int(self.n_quad) <= 64):
            raise ValueError("merge_core n_quad must be in [1, 64]")
        for name in ("pos_tol_m", "speed_tol_mps", "gap_tol_m", "gap_speed_tol_mps"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"merge_core {name} must be finite and positive")

    @property
    def enabled(self) -> bool:
        return (self.weight > 0.0 or self.tf_weight > 0.0 or self.yield_weight > 0.0
                or bool(self.metrics))


def _geometry(cfg: MergeCoreConfig) -> SimpleNamespace:
    return SimpleNamespace(merge_lane=int(cfg.merge_lane),
                           merge_target_lane=int(cfg.target_lane))


def _core_margin(alpha: Tensor, gap_l: Tensor, gap_f: Tensor, v: Tensor, v_l: Tensor,
                 v_f: Tensor, a: Tensor, a_l: Tensor, a_f: Tensor, has_l: Tensor,
                 has_f: Tensor, s0: Tensor, T: Tensor, b: Tensor, tau_a: Tensor,
                 residual: Optional[Tensor], require_both: Optional[Tensor] = None) -> Tensor:
    ml, mf = secure_gap_sides(gap_l, gap_f, v, v_l, v_f, a, a_l, a_f, has_l, has_f,
                              s0, T, b, tau_a, alpha, require_both)
    margin = torch.minimum(ml, mf)
    return margin if residual is None else margin + residual


class MergeCoreTerm(LossTerm):
    """Survival + teacher-forced + optional diagnostics of the secure-gap core."""

    name = TERM_NAME

    def __init__(self, cfg: MergeCoreConfig) -> None:
        self.cfg = cfg
        self.geometry = _geometry(cfg)

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
            targets.prepared["merge_core"] = {
                "rec_s": rec[:, :n, :k, 0], "rec_v": rec[:, :n, :k, 4],
                "rec_kin": rec[:, :n], "rec_present": valid[:, :n],
                "rec_lane": lane[:, :n], "rec_next_lane": lane[:, 1:n + 1],
                "rec_next_present": valid[:, 1:n + 1],
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
        prep = t.prepared["merge_core"]
        c0, c1, k = ctx.c0, ctx.c1, prep["k"]
        T = c1 - c0
        pred = ctx.pred
        if T <= 0:
            return None
        keys = ("merge_core_gap_l", "merge_core_gap_f", "merge_core_alpha_log_location",
                "merge_core_alpha_scale", "merge_core_s0")
        if not all(torch.is_tensor(pred.get(key)) for key in keys):
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

        gap_l = cut(pred["merge_core_gap_l"]).float()
        gap_f = cut(pred["merge_core_gap_f"]).float()
        v_l = cut(pred["merge_core_v_l"]).float()
        v_f = cut(pred["merge_core_v_f"]).float()
        a_l = cut(pred["merge_core_a_l"]).float()
        a_f = cut(pred["merge_core_a_f"]).float()
        has_l = cut(pred["merge_core_has_l"]).bool()
        has_f = cut(pred["merge_core_has_f"]).bool()
        loc = cut(pred["merge_core_alpha_log_location"]).float()
        sigma = cut(pred["merge_core_alpha_scale"]).float()
        residual = pred.get("merge_core_residual")
        residual = None if not torch.is_tensor(residual) else cut(residual).float()
        require = pred.get("merge_core_require_both")
        require = None if not torch.is_tensor(require) else cut(require).bool()
        s0 = pred["merge_core_s0"].float().mean()
        Tp = pred["merge_core_T"].float().mean()
        b = pred["merge_core_b"].float().mean()
        tau = pred["merge_core_tau_a"].float().mean()
        temperature = pred["merge_core_temperature"].float().mean().clamp_min(1.0e-3)

        rec_cav = t.is_cav[:, c0:c1, :k].bool()
        entrant = pred.get("entrant_seq")
        ent_seq = cut(entrant) if torch.is_tensor(entrant) else None
        ent = (ent_seq[:, :, :k].bool() if ent_seq is not None
               and ent_seq.shape == pres_dec.shape else torch.zeros_like(pres_dec))
        own = ~rec_cav & ~ent
        ramp_dec = pres_dec & (lane_dec == int(cfg.merge_lane)) & own
        in_zone = ramp_dec & (s_dec >= float(cfg.zone_s0_m)) & (s_dec <= float(cfg.zone_s1_m))
        acts = cut(pred["lane_v5_acts"]).bool() if torch.is_tensor(
            pred.get("lane_v5_acts")) else torch.ones_like(pres_dec)
        feasible = in_zone & acts & own

        step = torch.arange(c0, c1, device=loc.device).view(1, T, 1)
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

        # ---- Gauss-Hermite integration of the persistent assertiveness alpha
        nodes, weights = gauss_hermite(cfg.n_quad, dtype=loc.dtype, device=loc.device)
        alpha_q = torch.exp((loc.unsqueeze(1) + sigma.unsqueeze(1) * nodes.view(1, -1, 1, 1))
                            .clamp(-8.0, 8.0))                       # [B, Q, T, K]
        base = (gap_l.unsqueeze(1), gap_f.unsqueeze(1), v_dec.unsqueeze(1),
                v_l.unsqueeze(1), v_f.unsqueeze(1), a_dec.unsqueeze(1),
                a_l.unsqueeze(1), a_f.unsqueeze(1), has_l.unsqueeze(1),
                has_f.unsqueeze(1))
        margin_q = _core_margin(alpha_q, *base, s0, Tp, b, tau,
                                None if residual is None else residual.unsqueeze(1),
                                None if require is None else require.unsqueeze(1))
        logits = margin_q / temperature
        log_h = F.logsigmoid(logits)
        log_1mh = F.logsigmoid(-logits)

        sel = take_event.unsqueeze(1)
        surv_mask = at_risk.unsqueeze(1) & ~sel
        log_surv = (log_1mh * surv_mask).sum(2)
        log_h_e = (log_h * sel).sum(2)
        event_q = log_surv + log_h_e
        censor_mask = at_risk & (step <= (last - c0).clamp(-1, T - 1).unsqueeze(1))
        censor_q = (log_1mh * censor_mask.unsqueeze(1)).sum(2)

        used = event_any | has_censor
        loglik_q = torch.where(event_any.unsqueeze(1), event_q, censor_q)
        logw = torch.log(weights.clamp_min(torch.finfo(weights.dtype).tiny))
        mix = torch.logsumexp(loglik_q + logw.view(1, -1, 1), dim=1)
        mix = torch.nan_to_num(mix, nan=LOG_LIKELIHOOD_FLOOR, neginf=LOG_LIKELIHOOD_FLOOR)
        nll = -(mix * used.to(mix.dtype)).sum()
        zero_likelihood = event_any & (mix <= LOG_LIKELIHOOD_FLOOR)
        n_cars = max(float(prep["n_cars"]), 1.0)
        loc_reg = (loc * ramp_dec).pow(2).sum() / max(float(ramp_dec.sum()), 1.0)

        # ---- teacher-forced acceptance likelihood on the recorded states
        tf_nll, tf_count, tf_metrics = self._teacher_forced(prep, c0, c1, s0, Tp, b, tau,
                                                            temperature, sigma)

        value = mix.new_zeros(())
        if cfg.weight > 0.0:
            value = value + float(cfg.weight) * nll / n_cars
        if cfg.tf_weight > 0.0 and tf_count > 0.0:
            value = value + float(cfg.tf_weight) * tf_nll / tf_count
        value = value + float(cfg.location_reg_weight) * loc_reg

        # ---- diagnostics (no gradient)
        with torch.no_grad():
            stats = prep["stats"].setdefault(c0, {
                "nll": 0.0, "events": 0.0, "early": 0.0, "zero": 0.0,
                "censored_admin": 0.0, "censored_divergence": 0.0, "location_reg": 0.0,
                "sigma": 0.0, "location": 0.0, "n_active": 0.0,
                "tf_nll": 0.0, "tf_rows": 0.0, "tf_events": 0.0,
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
            stats["tf_nll"] += tf_metrics.get("nll", 0.0)
            stats["tf_rows"] += tf_metrics.get("rows", 0.0)
            stats["tf_events"] += tf_metrics.get("events", 0.0)
            wait = ((last - entry).clamp_min(0).to(logits.dtype) * float(prep["dt_s"]))
            wb = _wait_bin(wait)
            bins = stats["wait_bins"]
            for i in range(len(WAIT_BINS_S)):
                bins[i] += float((divergent & (wb == i)).sum())
        return TermValue(value, log=value.detach())

    def _teacher_forced(self, prep: dict, c0: int, c1: int, s0: Tensor, Tp: Tensor,
                        b: Tensor, tau: Tensor, temperature: Tensor,
                        sigma: Tensor) -> tuple[Tensor, float, dict]:
        cfg = self.cfg
        rec_kin = prep["rec_kin"][:, c0:c1]
        present = prep["rec_present"][:, c0:c1]
        bsz, n, k = rec_kin.shape[0], rec_kin.shape[1], rec_kin.shape[2]
        if bsz * n == 0:
            zero = s0.new_zeros(())
            return zero, 0.0, {}
        flat = rec_kin.reshape(bsz * n, k, 7)
        flat_present = present.reshape(bsz * n, k)
        feat = core_features(flat, flat_present, self.geometry)
        gap_l = feat["gap_l"].view(bsz, n, k)
        gap_f = feat["gap_f"].view(bsz, n, k)
        v = flat[..., 4].view(bsz, n, k)
        v_l = feat["v_l"].view(bsz, n, k)
        v_f = feat["v_f"].view(bsz, n, k)
        a = flat[..., 5].view(bsz, n, k)
        a_l = feat["a_l"].view(bsz, n, k)
        a_f = feat["a_f"].view(bsz, n, k)
        has_l = feat["has_l"].view(bsz, n, k)
        has_f = feat["has_f"].view(bsz, n, k)
        ramp = (present & (prep["rec_lane"][:, c0:c1] == int(cfg.merge_lane))
                & (flat[..., 0].view(bsz, n, k) >= float(cfg.zone_s0_m))
                & (flat[..., 0].view(bsz, n, k) <= float(cfg.zone_s1_m)))
        valid = ramp & (has_l | has_f)
        label = (valid & prep["rec_next_present"][:, c0:c1]
                 & (prep["rec_next_lane"][:, c0:c1] == int(cfg.target_lane))).float()
        if float(valid.sum()) <= 0.0:
            zero = s0.new_zeros(())
            return zero, 0.0, {"nll": 0.0, "rows": 0.0, "events": 0.0}
        nodes, weights = gauss_hermite(cfg.n_quad, dtype=s0.dtype, device=s0.device)
        alpha_q = torch.exp((sigma.mean(1).unsqueeze(1).unsqueeze(1)
                             * nodes.view(1, -1, 1, 1)).clamp(-8.0, 8.0))
        margin_q = _core_margin(alpha_q, gap_l.unsqueeze(1), gap_f.unsqueeze(1),
                                v.unsqueeze(1), v_l.unsqueeze(1), v_f.unsqueeze(1),
                                a.unsqueeze(1), a_l.unsqueeze(1), a_f.unsqueeze(1),
                                has_l.unsqueeze(1), has_f.unsqueeze(1), s0, Tp, b, tau,
                                None, None)
        logits = margin_q / temperature
        y = label.unsqueeze(1)
        ll = y * F.logsigmoid(logits) + (1.0 - y) * F.logsigmoid(-logits)
        logw = torch.log(weights.clamp_min(torch.finfo(weights.dtype).tiny))
        row_ll = torch.logsumexp(ll + logw.view(1, -1, 1, 1), dim=1)
        nll = -(row_ll * valid).sum()
        return nll, float(valid.sum()), {"nll": float(nll.detach()),
                                         "rows": float(valid.sum()),
                                         "events": float(label.sum())}


def merge_core_metrics(prep: Optional[dict]) -> dict:
    """Whole-unroll diagnostics from the per-chunk statistics (empty if absent)."""
    if not prep or not prep.get("stats"):
        return {}
    stats = list(prep["stats"].values())

    def total(key: str) -> float:
        return sum(float(s.get(key, 0.0)) for s in stats)

    n_cars = max(float(prep["n_cars"]), 1.0)
    n_active = max(total("n_active"), 1.0)
    tf_rows = max(total("tf_rows"), 1.0)
    out = {
        "merge_core_nll": total("nll") / n_cars,
        "merge_core_events": total("events"),
        "merge_core_censored_admin": total("censored_admin"),
        "merge_core_censored_divergence": total("censored_divergence"),
        "merge_core_early_merges": total("early"),
        "merge_core_zero_likelihood_events": total("zero"),
        "merge_core_location_reg": total("location_reg"),
        "merge_core_alpha_loc_mean": total("location") / n_active,
        "merge_core_alpha_sigma_mean": total("sigma") / n_active,
        "merge_core_tf_nll": total("tf_nll") / tf_rows,
        "merge_core_tf_rows": total("tf_rows"),
        "merge_core_tf_events": total("tf_events"),
        "merge_core_n_cars": n_cars,
    }
    for i, (lo, hi) in enumerate(WAIT_BINS_S):
        key = (f"merge_core_censored_divergence_wait_{int(lo)}_"
               f"{'inf' if math.isinf(hi) else int(hi)}")
        out[key] = sum(float(s["wait_bins"][i]) for s in stats)
    return out


def merge_core_terms(cfg: MergeCoreConfig) -> list[MergeCoreTerm]:
    """The secure-gap chunk term (empty when every weight and metrics are off)."""
    return [MergeCoreTerm(cfg)] if cfg.enabled else []


__all__ = ["MergeCoreConfig", "MergeCoreTerm", "merge_core_terms", "merge_core_metrics",
           "TERM_NAME"]
