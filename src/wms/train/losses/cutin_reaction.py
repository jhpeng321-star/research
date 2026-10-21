"""R9K: teacher-forced, event-weighted cut-in reaction loss (``loss.cutin_reaction``).

Evidence (runs/active/C4-V7FT-20261017/discharge/FOLLOWER_YIELD.md): the WM lane-1
follower over-brakes at ramp insertions - min accel ~-2.4 vs SUMO ~-0.9 m/s^2 at
matched gap/closing, with the excess arriving AFTER the insertion and SUMO
followers accelerating again afterwards.  Insertions are rare in the ordinary
position/speed Huber terms, so the follower response around them is barely
supervised.  This term re-weights exactly those steps, TEACHER-FORCED on the
recorded states (no rollout drift): for every recorded ramp->lane-1 insertion
(the C11 / ``scripts/c4/wm_discharge_check.py`` definition), the nearest lane-1
HDV follower behind the insertion point is located and the model's ONE-STEP
predicted acceleration on the recorded state of each window step is scored
against the recorded acceleration with Huber.

* **event term** (``weight``): per-event mean Huber over the event window, then
  the mean over events - every insertion contributes equally, however rare;
* **control term** (``control_weight``): the same one-step Huber on lane-1 HDV
  rows OUTSIDE every insertion window, so ordinary braking is re-anchored and the
  event weighting cannot be "solved" by cancelling braking globally (astra S2:
  non-insertion controls).

Window (data-derived, documented maximum): the follower is supervised from
``pre_s`` before the insertion (default 1 s, the C11 follower-metric definition)
until its recorded acceleration first returns to >= 0 (the recorded response
period - the follower accelerates again, per FOLLOWER_YIELD), censored at the
documented cap ``max_window_s`` (default 6 s = the C11 post-insertion window).
The cap is a metric-definition constant, not a behavioural threshold.

Both weights default to 0: the term is never built and existing configs are
bit-identical.  One shared one-step model call serves both parts; the value is
``weight * event + control_weight * control`` (single backward through one
graph).  Only the predicted acceleration carries gradient; every mask, window
and target is detached recorded data.  The one-step call ``model(state, k_steps=t)``
uses the unroll's absolute step index, so the gate value equals the one the
rollout would apply at that step of the same unroll.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import TermValue, UnrollContext, UnrollTerm, UnrollTargets

TERM_NAME = "cutin_reaction"


@dataclass(frozen=True)
class CutinReactionConfig:
    """Weights and window definition of the cut-in reaction terms (all default OFF)."""

    weight: float = 0.0
    control_weight: float = 0.0
    pre_s: float = 1.0
    max_window_s: float = 6.0
    huber_delta_mps2: float = 0.5
    merge_lane: int = 2
    target_lane: int = 1

    def __post_init__(self) -> None:
        for name in ("weight", "control_weight", "pre_s", "max_window_s",
                     "huber_delta_mps2"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"cutin_reaction {name} must be finite")
        if float(self.weight) < 0.0 or float(self.control_weight) < 0.0:
            raise ValueError("cutin_reaction weights must be >= 0 (0 = off)")
        if not 0.0 <= float(self.pre_s) <= 5.0:
            raise ValueError("cutin_reaction.pre_s must be in [0, 5] seconds")
        if not 1.0 <= float(self.max_window_s) <= 20.0:
            raise ValueError("cutin_reaction.max_window_s must be in [1, 20] seconds")
        if float(self.huber_delta_mps2) <= 0.0:
            raise ValueError("cutin_reaction.huber_delta_mps2 must be positive")
        if int(self.merge_lane) == int(self.target_lane) \
                or int(self.merge_lane) < 0 or int(self.target_lane) < 0:
            raise ValueError("cutin_reaction lanes must be distinct and non-negative")

    @property
    def enabled(self) -> bool:
        return float(self.weight) > 0.0 or float(self.control_weight) > 0.0

    def pre_steps(self, dt_s: float) -> int:
        return max(0, int(round(float(self.pre_s) / float(dt_s))))

    def post_max_steps(self, dt_s: float) -> int:
        return max(1, int(round(float(self.max_window_s) / float(dt_s))))

    @property
    def log_terms(self) -> tuple[str, ...]:
        return (TERM_NAME,) if self.enabled else ()


# ------------------------------------------------------------------ event scan
def _recorded_lanes(targets: UnrollTargets) -> tuple[Tensor, Tensor]:
    """Recorded lane index and lane validity ``[B, T+1, K]``."""
    onehot = targets.states[..., 1:4]
    lane = onehot.argmax(-1)
    return lane, onehot.sum(-1) > 0


@torch.no_grad()
def insertion_events(targets: UnrollTargets, config: CutinReactionConfig,
                     ) -> list[dict[str, Any]]:
    """Recorded ramp->lane-1 insertions with their nearest lane-1 HDV follower.

    Mirrors ``scripts/c4/wm_discharge_check.py``: an insertion at step ``w`` is a
    loss-valid slot on the merge lane at ``w`` and on the target lane at ``w+1``
    (CAV inserters included - the event is recorded geometry; only the follower
    must be a supervised HDV).  The follower is the nearest lane-1 vehicle BEHIND
    the inserter's position at the insertion observation ``w+1`` (largest ``s``
    below it).  Each event carries the data-derived window: steps
    ``[w - pre, w + post]`` with ``post`` = the follower's recorded response
    period (first step whose recorded accel is >= 0), censored at the cap.
    """
    n = targets.n_steps
    lane, lane_valid = _recorded_lanes(targets)
    valid = targets.loss_valid[:, : n + 1].bool()
    is_cav = targets.is_cav[:, : n + 1].bool()
    states = targets.states
    pre = config.pre_steps(float(targets.dt_s))
    post_max = config.post_max_steps(float(targets.dt_s))
    events: list[dict[str, Any]] = []
    if n <= 0:
        return events
    insert = (valid[:, :n] & valid[:, 1:n + 1]
              & (lane[:, :n] == int(config.merge_lane))
              & (lane[:, 1:n + 1] == int(config.target_lane)))
    for b in range(states.shape[0]):
        for j in range(states.shape[2]):
            steps = torch.nonzero(insert[b, :, j], as_tuple=False).flatten()
            for w_t in steps.tolist():
                w = int(w_t)
                s_ins = float(states[b, w + 1, j, 0])
                behind = (valid[b, w + 1] & (lane[b, w + 1] == int(config.target_lane))
                          & (states[b, w + 1, :, 0] < s_ins))
                behind[j] = False
                cand = torch.nonzero(behind, as_tuple=False).flatten().tolist()
                if not cand:
                    continue
                follower = max(cand, key=lambda k: float(states[b, w + 1, k, 0]))
                if bool(is_cav[b, w + 1, follower]):
                    # The supervised follower must be an HDV slot throughout the
                    # window; a CAV follower is teacher-forced (no gradient).
                    if all(bool(is_cav[b, u + 1, follower])
                           for u in range(max(0, w - pre), min(n, w + post_max + 1))):
                        continue
                # Response period: first post-insertion step whose recorded accel
                # is >= 0 (recovery), censored at the documented cap.
                post = post_max
                for d in range(1, post_max + 1):
                    u = w + d
                    if u >= n:
                        post = min(d, post_max)
                        break
                    if float(states[b, u + 1, follower, 5]) >= 0.0:
                        post = d
                        break
                start = max(0, w - pre)
                end = min(n - 1, w + post)
                events.append({
                    "batch": b, "inserter": j, "follower": follower,
                    "step": w, "start": start, "end": end,
                    "gap_m": s_ins - float(states[b, w + 1, follower, 0])
                    - float(states[b, w + 1, follower, 6]),
                    "censored": post == post_max,
                })
    return events


@torch.no_grad()
def insertion_plan(targets: UnrollTargets, config: CutinReactionConfig) -> dict:
    """Detached event/follower/control row masks of the whole unroll (cached)."""
    key = (TERM_NAME, float(config.weight), float(config.control_weight),
           float(config.pre_s), float(config.max_window_s), int(config.merge_lane),
           int(config.target_lane))
    cached = targets.prepared.get(TERM_NAME)
    if cached is not None and cached.get("key") == key:
        return cached
    n = targets.n_steps
    b_sz, k = targets.states.shape[0], targets.states.shape[2]
    lane, _ = _recorded_lanes(targets)
    hdv = targets.hdv_mask.bool()                      # [B, n, K] supervised rows
    window = torch.zeros(b_sz, n, k, dtype=torch.bool, device=targets.states.device)
    events = insertion_events(targets, config) if config.enabled else []
    for event in events:
        b, f = int(event["batch"]), int(event["follower"])
        window[b, event["start"]:event["end"] + 1, f] = True
    supervised = window & hdv
    control = (hdv
               & (lane[:, 1:n + 1] == int(config.target_lane))
               & ~supervised
               & ~window)
    plan = {
        "key": key, "events": events, "window": window, "supervised": supervised,
        "control": control, "n_events": len(events),
        "n_event_rows": int(supervised.sum()), "n_control_rows": int(control.sum()),
        "dt_s": float(targets.dt_s),
    }
    targets.prepared[TERM_NAME] = plan
    return plan


# ------------------------------------------------------- teacher-forced call
def _driver_inputs(model: Any, ctx: UnrollContext):
    """Driver parameters / embedding of the t0 encoding for the one-step call."""
    needs = (getattr(model, "rollout_mode", "residual") != "residual"
             or bool(getattr(model, "long_history", False))
             or bool(getattr(model, "_wmlane_embedding", False)))
    if not needs:
        return None, None
    prep = ctx.targets.prepared.get("lane_event")
    if isinstance(prep, dict) and prep.get("driver_params") is not None:
        return prep["driver_params"], prep.get("driver_embedding")
    _, embedding, params = model.encode_drivers(ctx.initial)
    return params, embedding


def _recorded_lane_age(lanes: Tensor, present: Tensor, n: int,
                       max_age: float = 999.0) -> Tensor:
    """Steps since the last recorded lane change ``[B, n, K]`` (causal)."""
    ages = []
    age = torch.zeros(lanes.shape[0], lanes.shape[2], dtype=lanes.dtype,
                      device=lanes.device)
    for t in range(n):
        if t:
            changed = (present[:, t] & present[:, t - 1]
                       & (lanes[:, t] != lanes[:, t - 1]))
            age = torch.where(changed, torch.zeros_like(age),
                              torch.where(present[:, t], age + 1.0,
                                          torch.zeros_like(age)))
        ages.append(age.clone())
    return torch.stack(ages, 1).clamp(max=max_age)


def one_step_predicted_accel(model: Any, targets: UnrollTargets, steps: Tensor,
                             *, driver_params=None, driver_embedding=None,
                             lane_age: Optional[Tensor] = None) -> Tensor:
    """Model one-step accelerations on the recorded states of ``steps`` ``[B, |S|, K]``.

    The state of every selected step is flattened into one batch; ``k_steps`` is
    the absolute step index, so the learned gate equals the rollout's gate at
    that step of the same unroll.  With gradient enabled the output carries
    gradient to every acceleration parameter path (IDM driver parameters,
    residual attention head, temporal path, negotiation / secure-gap yield heads).
    """
    n = targets.n_steps
    states = targets.states
    step_list = [int(t) for t in steps.tolist()]
    b_sz, k = states.shape[0], states.shape[2]

    def flat(x: Tensor, t: int) -> Tensor:
        return x[:, t].reshape(b_sz, k)

    veh = torch.stack([states[:, t] for t in step_list], dim=1).reshape(
        b_sz * len(step_list), k, 7).float()
    present = torch.stack([targets.loss_valid[:, t] for t in step_list],
                          dim=1).reshape(b_sz * len(step_list), k).bool()
    is_cav = torch.stack([targets.is_cav[:, t] for t in step_list], dim=1).reshape(
        b_sz * len(step_list), k).bool()
    state = {"veh_kin": veh, "present": present, "is_cav": is_cav}

    kwargs: dict[str, Any] = {}
    if driver_params is not None:
        params = {}
        for name, value in driver_params.items():
            value = torch.as_tensor(value).float()
            params[name] = value.unsqueeze(1).expand(
                b_sz, len(step_list), *value.shape[1:]).reshape(
                b_sz * len(step_list), *value.shape[1:])
        kwargs["driver_params"] = params
    if driver_embedding is not None:
        emb = torch.as_tensor(driver_embedding).float()
        kwargs["driver_embedding"] = emb.unsqueeze(1).expand(
            b_sz, len(step_list), *emb.shape[1:]).reshape(
            b_sz * len(step_list), *emb.shape[1:])
    if lane_age is not None:
        kwargs["lane_age"] = torch.stack([lane_age[:, t] for t in step_list],
                                         dim=1).reshape(-1)
    k_steps = torch.as_tensor(step_list, device=states.device,
                              dtype=torch.long).view(1, -1, 1).expand(
        b_sz, len(step_list), k).reshape(-1)
    out = model(state, k_steps=k_steps, **kwargs)
    accel = out["accel_mps2"]
    return accel.reshape(b_sz, len(step_list), k)


class CutinReactionLoss(UnrollTerm):
    """Once-per-unroll teacher-forced cut-in reaction term (``loss.cutin_reaction``).

    ``value = weight * event + control_weight * control`` with

    * ``event`` = mean over insertion events of the per-event mean Huber between
      the one-step predicted and recorded follower acceleration (every event
      equally weighted, however rare in the batch);
    * ``control`` = mean Huber of the same one-step prediction on lane-1 HDV
      rows outside every insertion window (ordinary braking anchor).
    """

    name = TERM_NAME
    differentiable = True

    def __init__(self, config: CutinReactionConfig) -> None:
        self.config = config
        self.last_stats: dict[str, Any] = {}

    def _predicted(self, ctx: UnrollContext) -> tuple[Tensor, Tensor]:
        """Teacher-forced predictions/targets ``[B, n, K]`` (cached per unroll)."""
        plan = insertion_plan(ctx.targets, self.config)
        cache = plan.setdefault("_tf", {})
        if cache.get("accel") is not None:
            return cache["accel"], cache["target"]
        targets = ctx.targets
        n = targets.n_steps
        model = ctx.model
        needed = plan["supervised"] | plan["control"]
        steps = torch.nonzero(needed.any((0, 2)), as_tuple=False).flatten()
        device = targets.states.device
        accel = torch.zeros(targets.states.shape[0], n, targets.states.shape[2],
                            device=device)
        if steps.numel():
            params, embedding = _driver_inputs(model, ctx)
            lane, _ = _recorded_lanes(targets)
            age = _recorded_lane_age(lane.float(), targets.loss_valid[:, :n].bool(), n)
            sel = one_step_predicted_accel(
                model, targets, steps, driver_params=params,
                driver_embedding=embedding, lane_age=age)
            accel[:, steps.tolist()] = sel
        target = targets.states[:, 1:n + 1, :, 5].detach().float()
        cache["accel"], cache["target"] = accel, target
        return accel, target

    def __call__(self, ctx: UnrollContext) -> Optional[TermValue]:
        if not self.config.enabled:
            return None
        plan = insertion_plan(ctx.targets, self.config)
        if plan["n_events"] == 0 and plan["n_control_rows"] == 0:
            zero = torch.zeros((), device=ctx.targets.states.device)
            self.last_stats = {"events": 0, "event_rows": 0, "control_rows": 0}
            return TermValue(zero, log=zero)
        accel, target = self._predicted(ctx)
        huber = F.smooth_l1_loss(accel, target,
                                 beta=float(self.config.huber_delta_mps2),
                                 reduction="none")

        event_value = None
        if plan["n_events"]:
            per_event = []
            for event in plan["events"]:
                b, f = int(event["batch"]), int(event["follower"])
                rows = plan["supervised"][b, event["start"]:event["end"] + 1, f]
                if bool(rows.any()):
                    per_event.append(huber[b, event["start"]:event["end"] + 1, f]
                                     [rows].mean())
            if per_event:
                event_value = torch.stack(per_event).mean()
        control_value = None
        if plan["n_control_rows"] and float(self.config.control_weight) > 0.0:
            rows = plan["control"]
            control_value = (huber * rows).sum() / rows.sum().clamp_min(1.0)

        parts = []
        if event_value is not None and float(self.config.weight) > 0.0:
            parts.append(float(self.config.weight) * event_value)
        if control_value is not None:
            parts.append(float(self.config.control_weight) * control_value)
        if not parts:
            zero = torch.zeros((), device=huber.device)
            self.last_stats = {"events": plan["n_events"],
                               "event_rows": plan["n_event_rows"],
                               "control_rows": plan["n_control_rows"]}
            return TermValue(zero, log=zero)
        value = parts[0] if len(parts) == 1 else parts[0] + parts[1]
        log = (event_value if event_value is not None and float(self.config.weight) > 0.0
               else (control_value if control_value is not None else value.detach()))
        self.last_stats = {
            "events": plan["n_events"], "event_rows": plan["n_event_rows"],
            "control_rows": int(plan["control"].sum()),
            "censored_windows": sum(1 for e in plan["events"] if e["censored"]),
        }
        return TermValue(value, log=log)


def cutin_reaction_terms(config: CutinReactionConfig) -> list[UnrollTerm]:
    """The cut-in reaction unroll term (empty when both weights are 0)."""
    return [CutinReactionLoss(config)] if config.enabled else []


__all__ = ["TERM_NAME", "CutinReactionConfig", "CutinReactionLoss",
           "insertion_events", "insertion_plan", "one_step_predicted_accel",
           "cutin_reaction_terms"]
