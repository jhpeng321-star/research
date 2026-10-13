"""WMLANE A: teacher-forced lane-change EVENT loss for the hazard lane model.

Once per unroll, the hazard model is evaluated on the RECORDED state of every
HDV-step ``t`` (0.5 s resolution) and its stay/left/right decision is scored against
the recorded transition ``t -> t+1`` with a focal, category-balanced cross entropy:

* categories: ``stay``, ``overtake`` (lane 1 -> 0), ``return`` (0 -> 1), ``merge``
  (ramp 2 -> 1) and ``other`` (any remaining adjacent change);
* every event category present in the batch gets the same total weight, together
  equal to the total 'stay' weight (``w_c = n_stay / (C * n_c)``), each per-step
  event weight clipped to ``[1, max_weight]``;
* focal factor ``(1 - p_y)^gamma`` (``gamma = 0`` is plain weighted CE);
* steps whose recorded action the physics feasibility mask forbids (masked logit)
  are not learnable and are dropped (counted in the stats).

Because the recorded state is used (no rollout drift), the loss measures decision
TIMING; the rollout lane loss and the flip-rate guard keep acting on the rollout.
The driver inputs come from ``targets.prepared['lane_event']`` (filled by the unroll
runner from the same t0 encoding the rollout uses).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from wms.model.lane_hazard import relative_logits
from wms.physics.mobil import MASKED_LOGIT

from .base import TermValue, UnrollContext, UnrollTerm

#: Event categories (index = category id).
CATEGORIES = ("stay", "overtake", "return", "merge", "other")


def lane_event_targets(lanes: Tensor, valid: Tensor, n_steps: int
                       ) -> tuple[Tensor, Tensor, Tensor]:
    """Relative action, category and validity ``[B, n, K]`` of the recorded transitions.

    ``lanes``/``valid`` are ``[B, T+1, K]``; a transition is valid when both ends are
    valid (on-road, supervised) and it moves at most one lane.
    """
    prev, nxt = lanes[:, :n_steps].long(), lanes[:, 1:n_steps + 1].long()
    mask = valid[:, :n_steps].bool() & valid[:, 1:n_steps + 1].bool() & ((nxt - prev).abs() <= 1)
    action = torch.where(nxt == prev, 0, torch.where(nxt < prev, 1, 2))
    category = torch.full_like(prev, 4)
    category = torch.where(nxt == prev, 0, category)
    category = torch.where((prev == 1) & (nxt == 0), 1, category)
    category = torch.where((prev == 0) & (nxt == 1), 2, category)
    category = torch.where((prev == 2) & (nxt == 1), 3, category)
    return action, category, mask


def category_weights(category: Tensor, max_weight: float) -> Tensor:
    """Per-category weights ``[5]`` (stay = 1) balancing the present event categories."""
    counts = torch.bincount(category.long(), minlength=len(CATEGORIES)).to(torch.float32)
    weights = torch.ones(len(CATEGORIES), dtype=torch.float32, device=category.device)
    present = counts[1:] > 0
    n_cats = int(present.sum())
    if n_cats:
        n_stay = counts[0].clamp_min(1.0)
        balanced = n_stay / (n_cats * counts[1:].clamp_min(1.0))
        weights[1:] = torch.where(present, balanced.clamp(1.0, float(max_weight)),
                                  torch.ones_like(balanced))
    return weights


def lane_event_loss(rel_logits: Tensor, action: Tensor, category: Tensor, *, gamma: float,
                    max_weight: float) -> tuple[Tensor, dict[str, Any]]:
    """Focal category-balanced CE over valid rows ``[N, 3]``; returns ``(loss, stats)``."""
    chosen = rel_logits.gather(-1, action.long().unsqueeze(-1)).squeeze(-1)
    feasible = chosen > MASKED_LOGIT + 1.0e-6
    stats = {"rows": int(action.numel()), "infeasible_events": int((~feasible).sum())}
    rel_logits, action, category = rel_logits[feasible], action[feasible], category[feasible]
    if action.numel() == 0:
        return rel_logits.sum() * 0.0, stats
    weights = category_weights(category, max_weight).to(rel_logits.dtype)
    w = weights[category.long()]
    logp = F.log_softmax(rel_logits, -1).gather(-1, action.long().unsqueeze(-1)).squeeze(-1)
    focal = (1.0 - logp.exp()).clamp_min(0.0).pow(float(gamma)) if gamma else 1.0
    loss = (w * focal * -logp).sum() / w.sum().clamp_min(1.0e-8)
    counts = torch.bincount(category.long(), minlength=len(CATEGORIES))
    stats.update({name: int(c) for name, c in zip(CATEGORIES, counts.tolist())})
    return loss, stats


def recorded_lane_age(lanes: Tensor, valid: Tensor, age0: Tensor, n_steps: int) -> Tensor:
    """Steps since the last recorded lane change ``[B, n, K]`` (``age0`` at t0)."""
    ages = [age0]
    age = age0
    for t in range(1, n_steps):
        changed = (lanes[:, t] != lanes[:, t - 1]) & valid[:, t].bool() & valid[:, t - 1].bool()
        age = torch.where(changed, torch.zeros_like(age), age + 1)
        ages.append(age)
    return torch.stack(ages, 1)


def recorded_event_logits(model: Any, states: Tensor, present: Tensor, is_cav: Tensor,
                          driver_params: Optional[Mapping[str, Tensor]],
                          driver_embedding: Optional[Tensor], lane_age: Tensor) -> Tensor:
    """Hazard lane logits ``[B, n, K, 3]`` on recorded states ``[B, n, K, 7]`` (one step each)."""
    b, n, k = states.shape[:3]
    flat = lambda x: x.reshape(b * n, *x.shape[2:])  # noqa: E731
    params = None
    if driver_params is not None:
        params = {name: value.unsqueeze(1).expand(b, n, k).reshape(b * n, k)
                  for name, value in driver_params.items()}
    embedding = None
    if driver_embedding is not None:
        embedding = driver_embedding.unsqueeze(1).expand(b, n, k, -1).reshape(b * n, k, -1)
    state = {"veh_kin": flat(states).float(), "present": flat(present).bool(),
             "is_cav": flat(is_cav).bool()}
    logits = model.lane_event_logits(state, driver_params=params, driver_embedding=embedding,
                                     lane_age=flat(lane_age))
    return logits.reshape(b, n, k, 3)


class LaneEventLoss(UnrollTerm):
    """Once-per-unroll teacher-forced event loss (``loss.lane_event.weight``)."""

    name = "lane_event"
    differentiable = True

    def __init__(self, weight: float, *, gamma: float = 2.0, max_weight: float = 20.0) -> None:
        self.weight = float(weight)
        self.gamma = float(gamma)
        self.max_weight = float(max_weight)
        self.last_stats: dict[str, Any] = {}

    def __call__(self, ctx: UnrollContext) -> Optional[TermValue]:
        prep = ctx.targets.prepared.get("lane_event")
        if prep is None or getattr(ctx.model, "lane_hazard", None) is None:
            return None
        t = ctx.targets
        n = t.n_steps
        states = t.states[:, :n + 1]
        present = t.loss_valid[:, :n + 1].bool()
        is_cav = t.is_cav[:, :n + 1].bool()
        lanes = states[..., 1:4].argmax(-1)
        valid = present & ~is_cav
        action, category, mask = lane_event_targets(lanes, valid, n)
        age = recorded_lane_age(lanes, valid, prep["lane_age0"].to(states.dtype), n)
        logits = recorded_event_logits(ctx.model, states[:, :n], present[:, :n], is_cav[:, :n],
                                       prep.get("driver_params"), prep.get("driver_embedding"),
                                       age)
        rel = relative_logits(logits, lanes[:, :n])
        comp, stats = lane_event_loss(rel[mask], action[mask], category[mask],
                                      gamma=self.gamma, max_weight=self.max_weight)
        self.last_stats = stats
        return TermValue(self.weight * comp, log=comp)
