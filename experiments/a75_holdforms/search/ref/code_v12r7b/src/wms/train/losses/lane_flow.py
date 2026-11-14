"""On-policy mainline lane-flow loss (V8LANE, 2026-10-17; default OFF).

Diagnosis (runs/active/C4-V7FT-20261017/lane_diag/LANE_DIAG.md): the overtake (lane 1 -> 0) and
return (0 -> 1) heads of the hazard_v5 lane network are calibrated on SUMO states (teacher-forced
BCE) but over-fire 1.2-1.6x on the WM's own rollout states, and a sampled rollout overshoots the
lane-0 share (0.44-0.51 vs SUMO 0.35).  The teacher-forced BCE never sees the rollout states.

This term calibrates the two mainline heads ON-POLICY, at the population level (the identity /
timing of a single change is not predictable from the state, so per-vehicle matching is not
the target):

* **Expected counts.**  For every rollout step, slot in lane 1 (overtake) / lane 0 (return),
  present, HDV, not an entrant, whose own transition the physics allows (``lane_v5_acts``):
  ``h = sigmoid(head - stay)`` (``head`` = the model's own-head logit WITH gradient,
  ``emit_merge_hazard``; ``stay`` the detached current-lane logit).  ``E_h`` = sum of ``h`` over
  the chunk and the batch (the expected number of changes if the rollout samples).
* **Recorded counts.**  ``C_h`` = recorded overtakes / returns in the same chunk (HDV,
  ``loss_valid`` at both ends; window padding is never counted).
* **Occupancy feedback** (``balance`` >= 0, 0 = off).  ``x`` = (number of supervised HDVs the
  model has in lane 0) - (number the record has in lane 0), over HDVs on the mainline in both at
  the chunk start (detached).  The return target becomes ``C_ret + balance * max(x, 0)`` and the
  overtake target ``C_ovt + balance * max(-x, 0)``: a rollout that already holds too many cars in
  lane 0 is asked for more returns (the keep-right pressure the head lacks off-policy).
* **Loss** = ``weight * sum_h (E_h - T_h log E_h) / n_windows`` (Poisson NLL of the pooled count;
  minimised at ``E_h = T_h``; constant terms dropped, so the value can be negative).

The gradient reaches the overtake / return heads only (features are computed without gradient;
the merge head is never in these rows).  Flag-gated: ``weight = 0`` never builds the term.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from .base import LossContext, LossTerm, TermValue, UnrollTargets

TERM_NAME = "lane_flow"
#: Mainline heads: (name, source lane, target lane).
HEADS = (("overtake", 1, 0), ("return", 0, 1))
_EPS = 1.0e-3


@dataclass(frozen=True)
class LaneFlowConfig:
    weight: float = 0.0
    balance: float = 0.0

    def __post_init__(self) -> None:
        for name in ("weight", "balance"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"lane_flow {name} must be finite and non-negative")

    @property
    def enabled(self) -> bool:
        return self.weight > 0.0


class LaneFlowTerm(LossTerm):
    """Pooled on-policy Poisson count matching of the overtake / return hazards."""

    name = TERM_NAME

    def __init__(self, cfg: LaneFlowConfig) -> None:
        self.cfg = cfg

    def prepare(self, targets: UnrollTargets) -> None:
        n = targets.n_steps
        with torch.no_grad():
            rec = targets.states[:, :n + 1]
            k = rec.shape[2]
            lanes = rec[..., 1:4].argmax(-1)
            cav = targets.is_cav[:, :n + 1, :k].bool()
            if cav.shape[-1] < k:
                cav = torch.cat((cav, cav.new_zeros((*cav.shape[:-1], k - cav.shape[-1]))), -1)
            valid = targets.loss_valid[:, :n + 1, :k].bool() & ~cav
            both = valid[:, :n] & valid[:, 1:]
            events = {name: both & (lanes[:, :n] == src) & (lanes[:, 1:] == dst)
                      for name, src, dst in HEADS}
        targets.prepared[TERM_NAME] = {"events": events, "lanes": lanes, "valid": valid, "k": k,
                                       "n_windows": float(rec.shape[0]), "stats": {}}

    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        cfg, t = self.cfg, ctx.targets
        prep = t.prepared[TERM_NAME]
        c0, c1, k = ctx.c0, ctx.c1, prep["k"]
        pred = ctx.pred
        head = pred.get("lane_v5_head_logit")
        if not torch.is_tensor(head):
            raise ValueError("lane_flow needs model.lane_model='hazard_v5' (lane_v5_head_logit)")

        def cut(x):
            return x[..., :k] if x.shape[-1] > k else x

        lanes, masks = cut(pred["lanes"]).long(), cut(pred["masks"]).bool()
        lane_dec = torch.cat((cut(ctx.current["lane"]).long().unsqueeze(1), lanes[:, :-1]), 1)
        pres_dec = torch.cat((cut(ctx.current["present"]).bool().unsqueeze(1), masks[:, :-1]), 1)
        rows = pres_dec & cut(pred["lane_v5_acts"]).bool()
        rows = rows & ~cut(t.is_cav[:, c0:c1]).bool()
        entrant = pred.get("entrant_seq")
        if torch.is_tensor(entrant) and cut(entrant).shape == rows.shape:
            rows = rows & ~cut(entrant).bool()
        logits = pred["lane_logits"][:, :, :k]
        stay = logits.gather(-1, lane_dec.clamp(0, 2).unsqueeze(-1)).squeeze(-1).detach()
        h = torch.sigmoid(cut(head).to(stay.dtype) - stay)
        excess = 0.0
        if cfg.balance > 0.0:
            with torch.no_grad():
                rec_lane = prep["lanes"][:, c0]
                sup = (prep["valid"][:, c0] & cut(ctx.current["present"]).bool()
                       & (rec_lane <= 1) & (cut(ctx.current["lane"]).long() <= 1))
                excess = float((sup & (cut(ctx.current["lane"]).long() == 0)).sum()
                               - (sup & (rec_lane == 0)).sum())
        total = h.sum() * 0.0
        stats = {}
        for name, src, _ in HEADS:
            sel = rows & (lane_dec == src)
            expected = torch.where(sel, h, torch.zeros_like(h)).sum()
            recorded = float(prep["events"][name][:, c0:c1].sum())
            push = max(excess, 0.0) if name == "return" else max(-excess, 0.0)
            target = recorded + cfg.balance * push
            total = total + expected - target * torch.log(expected + _EPS)
            stats[f"{name}_expected"] = float(expected.detach())
            stats[f"{name}_recorded"] = recorded
        stats["lane0_excess"] = float(excess)
        prep["stats"][c0] = stats
        value = cfg.weight * total / max(prep["n_windows"], 1.0)
        return TermValue(value)


def lane_flow_metrics(prep: Optional[dict]) -> dict:
    """Whole-unroll sums (expected vs recorded per head, mean lane-0 excess) of the chunks."""
    if not prep or not prep.get("stats"):
        return {}
    stats = list(prep["stats"].values())
    out = {f"lane_flow_{key}": sum(s.get(key, 0.0) for s in stats)
           for key in ("overtake_expected", "overtake_recorded", "return_expected",
                       "return_recorded")}
    out["lane_flow_lane0_excess_mean"] = sum(s["lane0_excess"] for s in stats) / len(stats)
    return out


def lane_flow_terms(cfg: LaneFlowConfig) -> list[LaneFlowTerm]:
    return [LaneFlowTerm(cfg)] if cfg.enabled else []


__all__ = ["LaneFlowConfig", "LaneFlowTerm", "lane_flow_terms", "lane_flow_metrics", "TERM_NAME"]
