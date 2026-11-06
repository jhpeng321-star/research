"""Per-vehicle merge-timing (discrete-time survival) loss (V8WM, owner-approved 2026-10-17).

Diagnosis (runs/active/C4-V7FT-20261017/wm_diag/WM_DIAGNOSIS_v7.md): both v5m and v7 leave
66-86 of 736 ramp vehicles in lane 2 for the whole 120 s (SUMO: 13) while the others leave too
early (v7 -1.5 s); the cell-averaged v7 merge loss could not see this (not identity-matched).
This term scores, per recorded ramp vehicle, WHEN it leaves the ramp lane, under the model's
own merge hazard evaluated on its own free-running rollout state.

**Hazard.**  For a slot in the ramp lane (``merge_lane``, 2) at decision step ``t`` (state
``t`` -> ``t + 1``) of the rollout, ``h_t = sigmoid(head_t - stay_t)``: ``head_t`` is the
row's own transition logit of the hazard_v5 lane network (the merge head) on the model's
rolled-out state, returned WITH gradient by the model (``emit_merge_hazard``; recomputed
outside the ``lane_hazard_decouple`` detach), ``stay_t`` the (detached) logit of the current
lane.  Steps where the physics forbids the merge (mask), the forced last-5 m window and CAV /
entrant slots are not at risk: ``h_t`` is 0 there (or 1 in the forced window), with no
learnable parameter, and they are left out of the likelihood.  V8WM2
``physics.forced_window_mode='hazard'``: the window's feasible steps are at risk too (the
model's ``lane_v5_acts`` include them; ``h_t`` uses the floored head logit).  Models without the hazard_v5
network use ``softmax(lane_logits)`` of the rollout (target lane 1; masked = logit <=
``MASKED_LOGIT``).  The lane features are computed without gradient (``lane_v5.step_features``),
so the gradient reaches the merge head only (not the motion state).

**Event time.**  Recorded side (``targets.states`` lanes, ``targets.loss_valid`` HDV steps):
a recorded HDV-step ``t`` is at risk when the vehicle is valid at ``t`` and ``t + 1`` and in
the ramp lane at ``t``; the event is the first transition out of it (``t*``).  A vehicle that
stays (or whose record ends) is right-censored.  Per at-risk step where the MODEL's vehicle is
also in the ramp lane, present, not an entrant and its hazard is learnable:

    NLL = - [ sum_{t < t*} log(1 - h_t) + log h_{t*} ]       (censored: survival part only)

The event term is dropped (counted in ``merge_timing_event_masked``) when the model state
forbids the merge at ``t*`` (its hazard is 0 by physics) and when the model's vehicle already
left the lane (its early departure was already penalised by the survival terms).

**Normalisation.**  Sum over (row, slot, step) divided by the number of recorded ramp
vehicles with at least one at-risk step in the unroll (computed once in ``prepare``), times
``weight``; truncated-BPTT chunk contributions add up to the whole-unroll per-vehicle mean.
Window padding is never at risk (``loss_valid``).

**Metrics** (no gradient; ``merge_timing_metrics``), over the t0 ramp roster (recorded HDVs in
the ramp lane at or before the lane end at t0): stuck fraction = never leaving lane 2 within the
unroll, recorded vs predicted (only vehicles whose record covers the unroll or merges),
signed / absolute / median merge-time error (s, vehicles that merge in both), ramp-HDV |ds|
over all supervised steps and at the last step, and the pooled sums/counts the validation uses.

Flag-gated: ``loss.merge_timing.weight = 0`` and ``metrics = false`` never build the term.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from wms.physics.mobil import MASKED_LOGIT

from .base import LossContext, LossTerm, TermValue, UnrollTargets

TERM_NAME = "merge_timing"
#: Forced-merge window before the acceleration-lane end (``wms.model.lane_v5.FORCE_WINDOW_M``).
FORCE_WINDOW_M = 5.0
#: Pooled validation keys (sums / counts; ``pooled_metrics`` turns them into ratios).
SUM_KEYS = ("ramp_abs_ds_sum_m", "ramp_ds_count", "ramp_final_abs_ds_sum_m", "ramp_final_count",
            "merge_timing_stuck_rec_n", "merge_timing_stuck_pred_n", "merge_timing_stuck_den",
            "merge_timing_abs_err_sum_s", "merge_timing_err_sum_s", "merge_timing_n_err")
ERRORS_KEY = "merge_timing_errors_s"


@dataclass(frozen=True)
class MergeTimingConfig:
    """Weight (0 = metrics only), lane geometry and step."""

    weight: float = 0.0
    metrics: bool = False
    merge_lane: int = 2
    target_lane: int = 1
    zone_end_m: float = 75.44
    force_window_m: float = FORCE_WINDOW_M
    #: V8WM2: the forced window keeps the learned hazard (its steps are at risk).
    forced_window_hazard: bool = False

    def __post_init__(self) -> None:
        if not (math.isfinite(float(self.weight)) and self.weight >= 0.0):
            raise ValueError("merge_timing weight must be finite and non-negative")

    @property
    def enabled(self) -> bool:
        return self.weight > 0.0 or bool(self.metrics)


def _first_index(mask: Tensor, dim: int = 1) -> Tensor:
    """Index of the first True along ``dim`` (-1 where none)."""
    n = mask.shape[dim]
    idx = torch.arange(n, device=mask.device).view(*([1] * dim), n, *([1] * (mask.ndim - dim - 1)))
    big = torch.where(mask, idx.expand_as(mask), torch.full_like(mask, n, dtype=torch.long))
    first = big.min(dim).values
    return torch.where(first >= n, torch.full_like(first, -1), first)


class MergeTimingTerm(LossTerm):
    """The survival NLL of the recorded lane-2 departure times (one chunk term)."""

    name = TERM_NAME

    def __init__(self, cfg: MergeTimingConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------ recorded side (once)
    def prepare(self, targets: UnrollTargets) -> None:
        cfg, n = self.cfg, targets.n_steps
        with torch.no_grad():
            rec = targets.states[:, :n + 1]
            k = rec.shape[2]
            lanes = rec[..., 1:4].argmax(-1)
            cav = targets.is_cav[:, :n + 1, :k].bool()
            if cav.shape[-1] < k:
                cav = torch.cat((cav, cav.new_zeros((*cav.shape[:-1], k - cav.shape[-1]))), -1)
            valid = targets.loss_valid[:, :n + 1, :k].bool() & ~cav
            lane2 = lanes == cfg.merge_lane
            risk = valid[:, :n] & valid[:, 1:] & lane2[:, :n]                    # [B, n, K]
            event = risk & ~lane2[:, 1:]
            survive = risk & lane2[:, 1:]
            vehicles = risk.any(1)
            rec_merge = _first_index(event)                                      # [B, K]
            ramp0 = valid[:, 0] & lane2[:, 0] & (rec[:, 0, :, 0] <= cfg.zone_end_m)
            complete = (rec_merge >= 0) | valid[:, n]
        targets.prepared["merge_timing"] = {
            "event": event, "survive": survive, "n_vehicles": float(vehicles.sum()),
            "rec_merge": rec_merge, "ramp0": ramp0, "complete": complete, "k": k,
            "dt_s": float(targets.dt_s), "n_steps": int(n),
            "pred_merge": {0: torch.full_like(rec_merge, -1)}, "stats": {}}

    # ------------------------------------------------------------ per chunk
    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        cfg, t = self.cfg, ctx.targets
        prep = t.prepared["merge_timing"]
        c0, c1, k = ctx.c0, ctx.c1, prep["k"]
        pred = ctx.pred

        def cut(x):
            return x[..., :k] if x.shape[-1] > k else x

        lanes, masks, pos = cut(pred["lanes"]).long(), cut(pred["masks"]).bool(), cut(pred["positions"])
        lane_dec = torch.cat((cut(ctx.current["lane"]).long().unsqueeze(1), lanes[:, :-1]), 1)
        pres_dec = torch.cat((cut(ctx.current["present"]).bool().unsqueeze(1), masks[:, :-1]), 1)
        s_dec = torch.cat((cut(ctx.current["s_m"]).unsqueeze(1), pos[:, :-1]), 1)
        in_lane = pres_dec & (lane_dec == cfg.merge_lane)
        entrant = pred.get("entrant_seq")
        if torch.is_tensor(entrant) and cut(entrant).shape == in_lane.shape:
            in_lane = in_lane & ~cut(entrant).bool()          # predicted newcomers: no record
        logits = pred["lane_logits"][:, :, :k]

        # ---- hazard on the model's own state: log h and log(1 - h), learnable rows
        head = pred.get("lane_v5_head_logit")
        neg_delta = pred.get("merge_negotiation_merge_delta")
        if torch.is_tensor(head):
            acts = cut(pred["lane_v5_acts"]).bool()
            stay = logits.gather(-1, lane_dec.clamp(0, 2).unsqueeze(-1)).squeeze(-1).detach()
            x = cut(head).to(stay.dtype)
            if torch.is_tensor(neg_delta):
                # V10NEGOT: the survival likelihood trains the negotiated merge logit.
                x = x + cut(neg_delta).to(x.dtype)
            x = x - stay
            log_h, log_1mh = F.logsigmoid(x), F.logsigmoid(-x)
        else:
            lsm = torch.log_softmax(logits, -1)
            log_h = lsm[..., cfg.target_lane]
            others = lsm.clone()
            others[..., cfg.target_lane] = float("-inf")
            log_1mh = torch.logsumexp(others, -1)
            acts = logits[..., cfg.target_lane] > MASKED_LOGIT + 1e-6
            if not cfg.forced_window_hazard:
                forced = (s_dec >= cfg.zone_end_m - cfg.force_window_m) & (s_dec <= cfg.zone_end_m)
                acts = acts & ~forced
        at_risk = in_lane & acts
        surv = prep["survive"][:, c0:c1] & at_risk
        evt = prep["event"][:, c0:c1] & at_risk
        zero = log_h.sum() * 0.0 + 0.0
        nll = -(torch.where(surv, log_1mh, torch.zeros_like(log_1mh)).sum()
                + torch.where(evt, log_h, torch.zeros_like(log_h)).sum())
        value = None
        if cfg.weight > 0.0:
            value = (cfg.weight * nll / prep["n_vehicles"] if prep["n_vehicles"] > 0.0
                     else zero)

        # ---- statistics (no gradient)
        with torch.no_grad():
            rec_event = prep["event"][:, c0:c1]
            ramp0 = prep["ramp0"]
            moved = in_lane & masks & (lanes != cfg.merge_lane)
            first = _first_index(moved)
            before = prep["pred_merge"].get(c0, torch.full_like(prep["rec_merge"], -1))
            prep["pred_merge"][c1] = torch.where((before < 0) & (first >= 0), first + c0, before)
            hdv = t.hdv_mask[:, c0:c1, :k].bool() & masks & ramp0.unsqueeze(1)
            ds = (pos - t.states[:, c0 + 1:c1 + 1, :k, 0]).abs()
            stats = {
                "risk_steps": float((surv | evt).sum()), "events_used": float(evt.sum()),
                "events_masked": float((rec_event & in_lane & ~acts).sum()),
                "events_left": float((rec_event & ~in_lane).sum()),
                "nll": float(nll.detach()),
                "ramp_abs_ds_sum_m": float(torch.where(hdv, ds, torch.zeros_like(ds)).sum()),
                "ramp_ds_count": float(hdv.sum())}
            if c1 == prep["n_steps"]:
                last = hdv[:, -1]
                stats["ramp_final_abs_ds_sum_m"] = float(
                    torch.where(last, ds[:, -1], torch.zeros_like(ds[:, -1])).sum())
                stats["ramp_final_count"] = float(last.sum())
            prep["stats"][c0] = stats
        return None if value is None else TermValue(value)


def merge_timing_metrics(prep: Optional[dict]) -> dict:
    """Whole-unroll metrics (record keys) from the per-chunk statistics (empty if absent)."""
    if not prep or not prep.get("stats"):
        return {}
    stats = list(prep["stats"].values())

    def total(key):
        return sum(s.get(key, 0.0) for s in stats)

    dt = prep["dt_s"]
    pred_merge = prep["pred_merge"][max(prep["pred_merge"])]
    rec_merge, ramp0, complete = prep["rec_merge"], prep["ramp0"], prep["complete"]
    den_mask = ramp0 & complete
    both = ramp0 & (rec_merge >= 0) & (pred_merge >= 0)
    errors = ((pred_merge - rec_merge)[both].double() * dt).tolist()
    out = {
        "merge_timing_vehicles": prep["n_vehicles"],
        "merge_timing_ramp_t0": float(ramp0.sum()),
        "merge_timing_risk_steps": total("risk_steps"),
        "merge_timing_events_used": total("events_used"),
        "merge_timing_event_masked": total("events_masked"),
        "merge_timing_event_left": total("events_left"),
        "merge_timing_nll_per_vehicle": (total("nll") / prep["n_vehicles"]
                                         if prep["n_vehicles"] > 0 else 0.0),
        "merge_timing_stuck_den": float(den_mask.sum()),
        "merge_timing_stuck_rec_n": float((den_mask & (rec_merge < 0)).sum()),
        "merge_timing_stuck_pred_n": float((den_mask & (pred_merge < 0)).sum()),
        "merge_timing_n_err": float(len(errors)),
        "merge_timing_err_sum_s": float(sum(errors)),
        "merge_timing_abs_err_sum_s": float(sum(abs(e) for e in errors)),
        "ramp_abs_ds_sum_m": total("ramp_abs_ds_sum_m"),
        "ramp_ds_count": total("ramp_ds_count"),
        "ramp_final_abs_ds_sum_m": total("ramp_final_abs_ds_sum_m"),
        "ramp_final_count": total("ramp_final_count"),
        ERRORS_KEY: errors,
    }
    out.update(_ratios(out, errors))
    return out


def _ratios(sums: dict, errors: list) -> dict:
    """Ratios / medians from pooled sums (one batch, or a whole validation pass)."""
    def div(a, b):
        return float(sums[a]) / float(sums[b]) if float(sums[b]) > 0 else 0.0

    med = float(torch.tensor(errors, dtype=torch.float64).median()) if errors else 0.0
    return {"merge_timing_stuck_rec_frac": div("merge_timing_stuck_rec_n", "merge_timing_stuck_den"),
            "merge_timing_stuck_pred_frac": div("merge_timing_stuck_pred_n",
                                                "merge_timing_stuck_den"),
            "merge_timing_mean_err_s": div("merge_timing_err_sum_s", "merge_timing_n_err"),
            "merge_timing_mean_abs_err_s": div("merge_timing_abs_err_sum_s", "merge_timing_n_err"),
            "merge_timing_median_err_s": med,
            "ramp_abs_ds_m": div("ramp_abs_ds_sum_m", "ramp_ds_count"),
            "ramp_final_abs_ds_m": div("ramp_final_abs_ds_sum_m", "ramp_final_count")}


def pooled_metrics(mean: dict, errors: list, prefix: str = "mean_") -> dict:
    """Validation: ratios of the batch-pooled sums (``mean`` holds batch means of the sums,
    so mean/mean = total/total) and the median over all vehicles' merge-time errors."""
    sums = {key: mean.get(prefix + key, 0.0) for key in SUM_KEYS}
    return {prefix + key: value for key, value in _ratios(sums, errors).items()}


def merge_timing_terms(cfg: MergeTimingConfig) -> list[MergeTimingTerm]:
    """The merge-timing chunk term (empty when weight 0 and metrics off)."""
    return [MergeTimingTerm(cfg)] if cfg.enabled else []


__all__ = ["MergeTimingConfig", "MergeTimingTerm", "merge_timing_terms", "merge_timing_metrics",
           "pooled_metrics", "TERM_NAME", "SUM_KEYS", "ERRORS_KEY"]
