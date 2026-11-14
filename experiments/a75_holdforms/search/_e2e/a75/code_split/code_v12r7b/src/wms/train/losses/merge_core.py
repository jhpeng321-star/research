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

from wms.model.merge_core import core_features, load_init, secure_gap_sides
from wms.model.merge_frailty import gauss_hermite

from .base import LossContext, LossTerm, TermValue, UnrollTargets
from .merge_first_passage import (WAIT_BINS_S, _first_index, _gap_compatible, _kin,
                                  _last_index, _wait_bin)

TERM_NAME = "merge_core"
LOG_LIKELIHOOD_FLOOR = math.log(1.0e-30)

#: Version-2 per-car terminal categories (exactly one per recorded ramp identity).
TERMINAL_EVENT = 0
TERMINAL_ADMIN = 1
TERMINAL_SUPPORT_POS = 2
TERMINAL_SUPPORT_SPEED = 3
TERMINAL_SUPPORT_GAP = 4
TERMINAL_SUPPORT_TOPOLOGY = 5
TERMINAL_SUPPORT_IDENTITY = 6
TERMINAL_EARLY = 7
TERMINAL_ZERO_EXPOSURE = 8
TERMINAL_NAMES = {
    TERMINAL_EVENT: "event", TERMINAL_ADMIN: "censored_admin",
    TERMINAL_SUPPORT_POS: "support_failure_pos",
    TERMINAL_SUPPORT_SPEED: "support_failure_speed",
    TERMINAL_SUPPORT_GAP: "support_failure_gap",
    TERMINAL_SUPPORT_TOPOLOGY: "support_failure_topology",
    TERMINAL_SUPPORT_IDENTITY: "support_failure_identity",
    TERMINAL_EARLY: "early_nonevent", TERMINAL_ZERO_EXPOSURE: "zero_exposure",
}


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
    # Round 8 (version 2): fixed normalisers, sigma regulariser and the repaired on-policy
    # supported-prefix survival.  Every new field defaults to version-1 behaviour.
    version: int = 1
    z_tf: float = 1.0
    z_on: float = 1.0
    sigma_reg_weight: float = 0.0
    sigma_fit: float = 0.0

    def __post_init__(self) -> None:
        for name in ("weight", "tf_weight", "yield_weight", "location_reg_weight",
                     "sigma_reg_weight"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"merge_core {name} must be finite and >= 0")
        if int(self.version) not in (1, 2):
            raise ValueError("merge_core version must be 1 or 2")
        for name in ("z_tf", "z_on"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"merge_core {name} must be finite and > 0")
        if not (math.isfinite(float(self.sigma_fit)) and float(self.sigma_fit) >= 0.0):
            raise ValueError("merge_core sigma_fit must be finite and >= 0")
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
            if int(cfg.version) == 2:
                self._prepare_v2(targets, targets.prepared["merge_core"])

    def _prepare_v2(self, targets: UnrollTargets, prep: dict) -> None:
        """Round-8 persistent support / posterior / birth state (version 2 only)."""
        shape = prep["rec_event"].shape
        device = prep["rec_event"].device
        prep["v2"] = {
            "support": torch.ones(shape, dtype=torch.bool, device=device),
            "started": torch.zeros(shape, dtype=torch.bool, device=device),
            "terminal": torch.zeros(shape, dtype=torch.bool, device=device),
            "birth_loc": torch.zeros(shape, dtype=torch.float32, device=device),
            "birth_sigma": torch.zeros(shape, dtype=torch.float32, device=device),
            "birth_step": torch.full(shape, -1, dtype=torch.long, device=device),
            "category": torch.full(shape, -1, dtype=torch.long, device=device),
            "reason": torch.zeros(shape, dtype=torch.long, device=device),
            "scored": torch.zeros(shape, dtype=torch.long, device=device),
            "opportunity": torch.zeros(shape, dtype=torch.long, device=device),
            "log_post": None,
            "tf_log_post": None,
            "scene": None,
            "n_eligible": max(float(prep["n_cars"]), 1.0),
        }
        if float(self.cfg.tf_weight) > 0.0:
            try:
                from . import merge_core_seqtf as seqtf
            except ImportError as exc:  # agent B's branch is not merged yet
                raise RuntimeError(
                    "merge_core version 2 with tf_weight > 0 needs "
                    "wms.train.losses.merge_core_seqtf (agent B); use tf_weight=0 until "
                    "that module is on the branch") from exc
            prep["v2"]["scene"] = seqtf.build_recorded_scene(
                targets, merge_lane=int(self.cfg.merge_lane),
                zone_s0_m=float(self.cfg.zone_s0_m),
                zone_s1_m=float(self.cfg.zone_s1_m), k=int(prep["k"]))

    # ------------------------------------------------------------ per chunk
    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        cfg, t = self.cfg, ctx.targets
        if int(cfg.version) == 2:
            return self._call_v2(ctx)
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

    def _call_v2(self, ctx: LossContext) -> Optional[TermValue]:
        """Round-8 repaired on-policy supported-prefix survival (version 2)."""
        cfg, t = self.cfg, ctx.targets
        prep = t.prepared["merge_core"]
        v2 = prep.get("v2")
        if v2 is None:  # prepare() was not called with version 2
            return None
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

        def chunk_view(x: Tensor) -> Tensor:
            """Chunk-sized view: real rollouts already are; tests may pass the whole unroll."""
            x = cut(x)
            if x.ndim >= 2 and x.shape[1] > T:
                x = x[:, c0:c1]
            return x

        lanes = chunk_view(pred["lanes"]).long()
        masks = chunk_view(pred["masks"]).bool()
        pos = chunk_view(pred["positions"])
        spd = chunk_view(pred["speeds"])
        acc = chunk_view(pred["accel"])
        lane_dec = torch.cat((cut(ctx.current["lane"]).long().unsqueeze(1), lanes[:, :-1]), 1)
        pres_dec = torch.cat((cut(ctx.current["present"]).bool().unsqueeze(1), masks[:, :-1]), 1)
        s_dec = torch.cat((cut(ctx.current["s_m"]).unsqueeze(1), pos[:, :-1]), 1)
        v_dec = torch.cat((cut(ctx.current["v_mps"]).unsqueeze(1), spd[:, :-1]), 1)
        a_dec = torch.cat((cut(ctx.current["a_mps2"]).unsqueeze(1), acc[:, :-1]), 1)
        length = prep["rec_kin"][:, c0:c1, :, 6]

        gap_l = chunk_view(pred["merge_core_gap_l"]).float()
        gap_f = chunk_view(pred["merge_core_gap_f"]).float()
        v_l = chunk_view(pred["merge_core_v_l"]).float()
        v_f = chunk_view(pred["merge_core_v_f"]).float()
        a_l = chunk_view(pred["merge_core_a_l"]).float()
        a_f = chunk_view(pred["merge_core_a_f"]).float()
        has_l = chunk_view(pred["merge_core_has_l"]).bool()
        has_f = chunk_view(pred["merge_core_has_f"]).bool()
        loc_step = chunk_view(pred["merge_core_alpha_log_location"]).float()
        sigma_step = chunk_view(pred["merge_core_alpha_scale"]).float()
        residual = pred.get("merge_core_residual")
        residual = None if not torch.is_tensor(residual) else chunk_view(residual).float()
        require = pred.get("merge_core_require_both")
        require = None if not torch.is_tensor(require) else chunk_view(require).bool()

        def param(key: str) -> Tensor:
            value = pred[key].float()
            if value.ndim == 1 and value.shape[0] > T:
                value = value[c0:c1]
            return value.mean()

        s0 = param("merge_core_s0")
        Tp = param("merge_core_T")
        b = param("merge_core_b")
        tau = param("merge_core_tau_a")
        temperature = param("merge_core_temperature").clamp_min(1.0e-3)

        rec_cav = t.is_cav[:, c0:c1, :k].bool()
        entrant = pred.get("entrant_seq")
        ent_seq = chunk_view(entrant) if torch.is_tensor(entrant) else None
        ent = (ent_seq[:, :, :k].bool() if ent_seq is not None
               and ent_seq.shape == pres_dec.shape else torch.zeros_like(pres_dec))
        own = ~rec_cav & ~ent
        ramp_dec = pres_dec & (lane_dec == int(cfg.merge_lane)) & own
        in_zone = ramp_dec & (s_dec >= float(cfg.zone_s0_m)) & (s_dec <= float(cfg.zone_s1_m))
        acts = chunk_view(pred["lane_v5_acts"]).bool() if torch.is_tensor(
            pred.get("lane_v5_acts")) else torch.ones_like(pres_dec)
        feasible = in_zone & acts & own

        rec_event = prep["rec_event"]
        last_risk = prep["last_risk"]
        rec_ramp = prep["ramp"][:, c0:c1]
        rec_present = prep["rec_present"][:, c0:c1]

        # ---- detached compatibility mask (same car, absolute clock); support persists
        pred_kin = _kin(s_dec, lane_dec, v_dec, a_dec, length)
        pos_ok = (s_dec - prep["rec_s"][:, c0:c1]).abs() <= float(cfg.pos_tol_m)
        spd_ok = (v_dec - prep["rec_v"][:, c0:c1]).abs() <= float(cfg.speed_tol_mps)
        gap_ok = _gap_compatible(cfg, pred_kin, prep["rec_kin"][:, c0:c1],
                                 pres_dec & own, rec_present)
        topo_ok = (own == rec_present) | ~ramp_dec
        compat = pos_ok & spd_ok & gap_ok & topo_ok
        pos_bad = ramp_dec & ~pos_ok
        spd_bad = ramp_dec & ~spd_ok
        gap_bad = ramp_dec & ~gap_ok

        prev_pres = torch.cat((cut(ctx.current["present"]).bool().unsqueeze(1),
                               pres_dec[:, :-1]), 1)
        new_car = pres_dec & ~prev_pres

        # persistent per-identity state (reset only on a genuinely new identity)
        support = v2["support"].clone()
        started = v2["started"].clone()
        terminal = v2["terminal"].clone()
        category = v2["category"].clone()
        reason = v2["reason"].clone()
        scored = v2["scored"].clone()
        opportunity = v2["opportunity"].clone()
        birth_loc = v2["birth_loc"].to(loc_step.dtype).clone()
        birth_sigma = v2["birth_sigma"].to(sigma_step.dtype).clone()
        birth_step = v2["birth_step"].clone()
        log_post = v2["log_post"]
        scored_chunk = 0
        opportunity_chunk = 0

        nodes, weights = gauss_hermite(cfg.n_quad, dtype=loc_step.dtype,
                                       device=loc_step.device)
        logw = torch.log(weights.clamp_min(torch.finfo(weights.dtype).tiny))
        logw = logw.view(1, -1, 1)
        on_nll = loc_step.new_zeros(())
        counts: dict[int, int] = {code: 0 for code in TERMINAL_NAMES}
        unsupported = 0
        born_any = torch.zeros_like(started)
        observed_intervals = 0.0
        supported_intervals = 0.0

        def register(mask: Tensor, code: int) -> None:
            nonlocal category
            counts[code] += int(mask.sum())
            category = torch.where(mask, torch.full_like(category, code), category)

        for i in range(T):
            new = new_car[:, i]
            if bool(new.any()):
                lost = new & started & ~terminal
                register(lost & (opportunity > 0), TERMINAL_SUPPORT_IDENTITY)
                support = torch.where(new, torch.ones_like(support), support)
                started = torch.where(new, torch.zeros_like(started), started)
                terminal = torch.where(new, torch.zeros_like(terminal), terminal)
                category = torch.where(new, torch.full_like(category, -1), category)
                reason = torch.where(new, torch.zeros_like(reason), reason)
                scored = torch.where(new, torch.zeros_like(scored), scored)
                opportunity = torch.where(new, torch.zeros_like(opportunity), opportunity)
                birth_loc = torch.where(new, torch.zeros_like(birth_loc), birth_loc)
                birth_sigma = torch.where(new, torch.zeros_like(birth_sigma), birth_sigma)
                birth_step = torch.where(new, torch.full_like(birth_step, -1), birth_step)
                if log_post is not None:
                    log_post = torch.where(new.unsqueeze(1), logw.expand_as(log_post),
                                           log_post)

            ramp_t = ramp_dec[:, i]
            first = ramp_t & ~started
            if bool(first.any()):
                birth_loc = torch.where(first, loc_step[:, i], birth_loc)
                birth_sigma = torch.where(first, sigma_step[:, i], birth_sigma)
                birth_step = torch.where(first, torch.full_like(birth_step, c0 + i),
                                         birth_step)
                born_any = born_any | first
            started = started | first
            alive = ~terminal

            opp_t = ramp_t & feasible[:, i] & alive & started
            opportunity = opportunity + opp_t.long()
            opportunity_chunk += int(opp_t.sum())
            observed_intervals += float((rec_ramp[:, i] & alive & started).sum())
            support = torch.where(ramp_t, support & compat[:, i], support)
            supported_intervals += float((ramp_t & support & alive & started).sum())

            fail = ramp_t & pres_dec[:, i] & ~compat[:, i] & alive & started
            if bool(fail.any()):
                code = torch.where(
                    pos_bad[:, i], torch.full_like(reason, TERMINAL_SUPPORT_POS),
                    torch.where(spd_bad[:, i], torch.full_like(reason, TERMINAL_SUPPORT_SPEED),
                                torch.where(gap_bad[:, i],
                                            torch.full_like(reason, TERMINAL_SUPPORT_GAP),
                                            torch.full_like(reason, TERMINAL_SUPPORT_TOPOLOGY))))
                reason = torch.where(fail, code, reason)
                for c in (TERMINAL_SUPPORT_POS, TERMINAL_SUPPORT_SPEED, TERMINAL_SUPPORT_GAP,
                          TERMINAL_SUPPORT_TOPOLOGY):
                    register(fail & (code == c) & (opportunity > 0), c)
                register(fail & (opportunity == 0), TERMINAL_ZERO_EXPOSURE)
                terminal = terminal | fail
                support = torch.where(fail, torch.zeros_like(support), support)
                alive = ~terminal

            missing = rec_ramp[:, i] & ~pres_dec[:, i] & alive & started
            if bool(missing.any()):
                register(missing & (opportunity > 0), TERMINAL_SUPPORT_IDENTITY)
                register(missing & (opportunity == 0), TERMINAL_ZERO_EXPOSURE)
                terminal = terminal | missing
                alive = ~terminal

            post = lanes[:, i]
            off = ramp_t & (post != int(cfg.merge_lane)) & (post != int(cfg.target_lane)) \
                & alive & started
            if bool(off.any()):
                register(off & (opportunity > 0), TERMINAL_SUPPORT_TOPOLOGY)
                register(off & (opportunity == 0), TERMINAL_ZERO_EXPOSURE)
                terminal = terminal | off
                alive = ~terminal

            at_risk = ramp_t & support & feasible[:, i] & alive & started
            ev = torch.zeros_like(ramp_t)
            is_e = (rec_event == (c0 + i)) & (rec_event >= 0) & alive & started
            if bool(is_e.any()):
                took = is_e & at_risk
                unsupported_mask = is_e & ~at_risk
                ev = took
                if bool(took.any()):
                    register(took, TERMINAL_EVENT)
                if bool(unsupported_mask.any()):
                    register(unsupported_mask & (opportunity > 0), TERMINAL_SUPPORT_GAP)
                    register(unsupported_mask & (opportunity == 0), TERMINAL_ZERO_EXPOSURE)
                    unsupported += int(unsupported_mask.sum())
                terminal = terminal | is_e
                alive = ~terminal

            is_censor = ((rec_event < 0) & (c0 + i == last_risk) & alive & started)
            surv = torch.zeros_like(ramp_t)
            if bool(is_censor.any()):
                censor_risk = is_censor & at_risk
                surv = censor_risk
                register(is_censor & (opportunity > 0), TERMINAL_ADMIN)
                register(is_censor & (opportunity == 0), TERMINAL_ZERO_EXPOSURE)
                terminal = terminal | is_censor
                alive = ~terminal

            early = (ramp_t & (post == int(cfg.target_lane)) & alive & started
                     & ((rec_event < 0) | (c0 + i < rec_event)))
            if bool(early.any()):
                early_risk = early & at_risk
                surv = surv | early_risk
                register(early & (opportunity > 0), TERMINAL_EARLY)
                register(early & (opportunity == 0), TERMINAL_ZERO_EXPOSURE)
                terminal = terminal | early
                alive = ~terminal

            has_event = rec_event >= 0
            before_event = ~has_event | (c0 + i < rec_event)
            normal = (at_risk & ~ev & ~is_censor & ~early & (c0 + i <= last_risk)
                      & before_event & ~terminal)
            surv = surv | normal
            used = ev | surv
            scored = scored + used.long()
            scored_chunk += int(used.sum())

            if bool(used.any()):
                alpha_q = torch.exp((birth_loc.unsqueeze(1)
                                     + birth_sigma.unsqueeze(1) * nodes.view(1, -1, 1)
                                     ).clamp(-8.0, 8.0))              # [B, Q, K]
                margin_q = _core_margin(
                    alpha_q, gap_l[:, i].unsqueeze(1), gap_f[:, i].unsqueeze(1),
                    v_dec[:, i].unsqueeze(1), v_l[:, i].unsqueeze(1),
                    v_f[:, i].unsqueeze(1), a_dec[:, i].unsqueeze(1),
                    a_l[:, i].unsqueeze(1), a_f[:, i].unsqueeze(1),
                    has_l[:, i].unsqueeze(1), has_f[:, i].unsqueeze(1), s0, Tp, b, tau,
                    None if residual is None else residual[:, i].unsqueeze(1),
                    None if require is None else require[:, i].unsqueeze(1))
                logits = margin_q / temperature
                log_h = F.logsigmoid(logits)
                log_1mh = F.logsigmoid(-logits)
                zero = logits.new_zeros(())
                g = (torch.where(surv.unsqueeze(1), log_1mh, zero)
                     + torch.where(ev.unsqueeze(1), log_h, zero))
                if log_post is None:
                    log_post = logw.expand(g.shape[0], -1, g.shape[-1])
                A = log_post + g
                d = torch.logsumexp(A, dim=1)
                on_nll = on_nll + torch.where(used, -d, d.new_zeros(())).sum()
                log_post = torch.where(used.unsqueeze(1), A - d.unsqueeze(1), log_post)

        # finalise identities whose only terminal is the end of the unroll
        if c1 == prep["n_steps"]:
            remaining = prep["ramp_any"] & ~terminal
            if bool(remaining.any()):
                register(remaining & (opportunity > 0), TERMINAL_ADMIN)
                register(remaining & (opportunity == 0), TERMINAL_ZERO_EXPOSURE)
                terminal = terminal | remaining
            uncounted = prep["ramp_any"] & ~terminal
            if bool(uncounted.any()):
                register(uncounted, TERMINAL_ZERO_EXPOSURE)
                terminal = terminal | uncounted

        v2["support"] = support
        v2["started"] = started
        v2["terminal"] = terminal
        v2["category"] = category
        v2["reason"] = reason
        v2["scored"] = scored
        v2["opportunity"] = opportunity
        v2["birth_loc"] = birth_loc.detach()
        v2["birth_sigma"] = birth_sigma.detach()
        v2["birth_step"] = birth_step
        v2["log_post"] = None if log_post is None else log_post.detach()

        # ---- regularisers (mean over births in this chunk, i.e. once per identity)
        born = born_any
        if bool(born.any()):
            loc_reg = birth_loc[born].pow(2).mean()
            sigma_fit = float(cfg.sigma_fit)
            if sigma_fit <= 0.0:
                sigma_fit = float(load_init()["alpha_log_sigma"])
            sigma_reg = (torch.log(birth_sigma[born].clamp_min(1.0e-8) / sigma_fit)
                         .pow(2).mean())
        else:
            loc_reg = loc_step.new_zeros(())
            sigma_reg = sigma_step.new_zeros(())

        n_eligible = max(float(prep.get("n_cars", 0.0)), v2.get("n_eligible", 1.0), 1.0)
        value = on_nll.new_zeros(())
        if float(cfg.weight) > 0.0:
            value = value + float(cfg.weight) * on_nll / n_eligible / float(cfg.z_on)
        tf_metrics: dict = {}
        tf_nll = None
        tf_cars = None
        if float(cfg.tf_weight) > 0.0:
            scene = v2.get("scene")
            if scene is None:
                raise RuntimeError(
                    "merge_core version 2 with tf_weight > 0 needs the recorded seq-TF scene")
            from . import merge_core_seqtf as seqtf
            tf_nll, tf_cars, tf_post, tf_metrics = seqtf.seq_tf_chunk(
                scene, c0, c1, s0=s0, T=Tp, b=b, tau_a=tau, temperature=temperature,
                sigma=birth_sigma, n_quad=int(cfg.n_quad), log_post=v2.get("tf_log_post"))
            v2["tf_log_post"] = tf_post
            value = value + (float(cfg.tf_weight) * tf_nll / n_eligible
                             / float(cfg.z_tf))
        value = value + float(cfg.location_reg_weight) * loc_reg
        value = value + float(cfg.sigma_reg_weight) * sigma_reg

        # ---- diagnostics (no gradient)
        with torch.no_grad():
            stats = prep["stats"].setdefault(c0, {
                "nll": 0.0, "events": 0.0, "early": 0.0, "zero": 0.0,
                "censored_admin": 0.0, "censored_divergence": 0.0, "location_reg": 0.0,
                "sigma": 0.0, "location": 0.0, "n_active": 0.0,
                "tf_nll": 0.0, "tf_rows": 0.0, "tf_events": 0.0, "tf_cars": 0.0,
                "wait_bins": [0.0] * len(WAIT_BINS_S)})
            for key in ("on_nll", "observed_intervals", "supported_intervals",
                        "scored_intervals", "opportunity_intervals", "unsupported_events",
                        "sigma_reg", "tf_cars"):
                stats.setdefault(key, 0.0)
            for code in TERMINAL_NAMES:
                stats.setdefault(f"terminal_{TERMINAL_NAMES[code]}", 0.0)
            stats["nll"] += float(on_nll.detach())
            stats["on_nll"] += float(on_nll.detach())
            stats["events"] += float(counts[TERMINAL_EVENT])
            stats["early"] += float(counts[TERMINAL_EARLY])
            stats["censored_admin"] += float(counts[TERMINAL_ADMIN])
            stats["censored_divergence"] += float(
                counts[TERMINAL_SUPPORT_POS] + counts[TERMINAL_SUPPORT_SPEED]
                + counts[TERMINAL_SUPPORT_GAP] + counts[TERMINAL_SUPPORT_TOPOLOGY]
                + counts[TERMINAL_SUPPORT_IDENTITY])
            stats["zero"] += float(counts[TERMINAL_ZERO_EXPOSURE])
            stats["unsupported_events"] += float(unsupported)
            stats["location_reg"] += float(loc_reg.detach())
            stats["sigma_reg"] += float(sigma_reg.detach())
            stats["scored_intervals"] += float(scored_chunk)
            stats["opportunity_intervals"] += float(opportunity_chunk)
            stats["observed_intervals"] += observed_intervals
            stats["supported_intervals"] += supported_intervals
            for code, name in TERMINAL_NAMES.items():
                stats[f"terminal_{name}"] += float(counts[code])
            if tf_nll is not None:
                stats["tf_nll"] += float(tf_metrics.get("nll", float(tf_nll.detach())))
                stats["tf_rows"] += float(tf_metrics.get("scored_rows",
                                                         tf_metrics.get("rows", 0.0)))
                stats["tf_events"] += float(tf_metrics.get("events", 0.0))
                stats["tf_cars"] += float(tf_cars.detach() if torch.is_tensor(tf_cars)
                                          else tf_cars)
            active = ramp_dec[:, 0] & own[:, 0]
            stats["n_active"] += float(active.sum())
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
    dt = float(prep.get("dt_s", 0.0))
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
            "merge_core_tf_nll_per_car": total("tf_nll") / max(total("tf_cars"), 1.0),
            "merge_core_tf_rows": total("tf_rows"),
            "merge_core_tf_events": total("tf_events"),
            "merge_core_n_cars": n_cars,
        }
    if prep.get("v2") is not None:
        out.update({
            "merge_core_version": 2,
            "merge_core_on_nll": total("on_nll") / n_cars,
            "merge_core_sigma_reg": total("sigma_reg"),
            "merge_core_unsupported_events": total("unsupported_events"),
            "merge_core_observed_intervals": total("observed_intervals"),
            "merge_core_supported_intervals": total("supported_intervals"),
            "merge_core_scored_intervals": total("scored_intervals"),
            "merge_core_opportunity_intervals": total("opportunity_intervals"),
            "merge_core_observed_seconds": total("observed_intervals") * dt,
            "merge_core_scored_seconds": total("scored_intervals") * dt,
            "merge_core_zero_exposure_rate": total("terminal_zero_exposure") / n_cars,
        })
        terminal_total = 0.0
        for name in TERMINAL_NAMES.values():
            value = total(f"terminal_{name}")
            out[f"merge_core_terminal_{name}"] = value
            terminal_total += value
        out["merge_core_terminal_total"] = terminal_total
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
