"""R9K recorded-state acceleration supervision, default off.

An insertion is observed at the FIRST lane-1 row e after a lane-2 row.
Prediction t uses recorded state t and scores recorded acceleration t+1 (SUMO
rows are post-step). Each event has unit mass, shared equally over its valid
follower rows. Overlapping events retain separate unit masses. Controls have a
separate unit mass over non-insertion lane-1 HDV transitions, including braking.

The response ends at the first nonnegative recorded acceleration AFTER e,
including that recovery transition, or at max_window_s (default/documented
maximum 6 s). pre_s is optional pre-event context, at most 1 s. These are
supervision bounds, never runtime braking rules. No recorded response label
is supplied to the model. There is no fixed minimum response duration.

UnrollTargets has stable slots, not vehicle IDs. Absence breaks identity;
datasets with immediate reuse must supply prepared['cutin_reaction_inputs']
['reset'] [B,T+1,K], true on the first row of a new occupant. That same optional
mapping accepts history/history_mask and driver_history/driver_history_mask
from the runner. Without them history starts explicitly unknown at window t0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from wms.model.history import (LONG_HISTORY_F, history_step_features,
                               initial_history, push_history)
from wms.model.leader_change import LeaderChangeTracker, feature_matrix
from wms.physics.mobil import update_merge_wait
from wms.physics.neighbours import lead_neighbour

from .base import TermValue, UnrollContext, UnrollTerm, UnrollTargets

TERM_NAME = "cutin_reaction"


@dataclass(frozen=True)
class CutinReactionConfig:
    weight: float = 0.0
    # Relative control weight; overall weight=0 disables BOTH terms.
    control_weight: float = 1.0
    pre_s: float = 1.0
    max_window_s: float = 6.0
    huber_delta_mps2: float = 0.5
    merge_lane: int = 2
    target_lane: int = 1

    def __post_init__(self) -> None:
        for name in ("weight", "control_weight", "pre_s", "max_window_s",
                     "huber_delta_mps2"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"cutin_reaction.{name} must be finite")
        if self.weight < 0 or self.control_weight <= 0:
            raise ValueError("cutin_reaction needs weight >= 0 and control_weight > 0")
        if not 0 <= self.pre_s <= 1 or not 0 < self.max_window_s <= 6:
            raise ValueError("cutin_reaction needs pre_s in [0,1], max_window_s in (0,6]")
        if self.huber_delta_mps2 <= 0:
            raise ValueError("cutin_reaction.huber_delta_mps2 must be positive")
        if self.merge_lane == self.target_lane or min(self.merge_lane, self.target_lane) < 0:
            raise ValueError("cutin_reaction lanes must be distinct and nonnegative")

    @property
    def enabled(self) -> bool:
        return self.weight > 0


def _recorded(targets: UnrollTargets):
    kin = targets.states[:, :targets.n_steps + 1].detach()
    onehot = kin[..., 1:4]
    lane = onehot.argmax(-1)
    valid = targets.loss_valid[:, :targets.n_steps + 1].bool() & (onehot.sum(-1) > 0)
    prep = targets.prepared.get("cutin_reaction_inputs", {})
    reset = prep.get("reset", torch.zeros_like(valid)).detach().bool()
    if reset.shape != valid.shape:
        raise ValueError("cutin_reaction reset must be [B,n_steps+1,K]")
    continuous = valid[:, :-1] & valid[:, 1:] & ~reset[:, 1:]
    hdv = (targets.hdv_mask.bool() & continuous
           & ~targets.is_cav[:, :-1].bool() & ~targets.is_cav[:, 1:].bool())
    return kin, lane, valid, reset, continuous, hdv


@torch.no_grad()
def insertion_events(targets: UnrollTargets, config: CutinReactionConfig) -> list[dict[str, Any]]:
    """Event rows and inclusive prediction-index windows, detached from autograd."""
    dt = float(targets.dt_s)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("cutin_reaction dt_s must be finite and positive")
    n = targets.n_steps
    kin, lane, valid, _, continuous, hdv = _recorded(targets)
    insert = continuous & (lane[:, :-1] == config.merge_lane) & (lane[:, 1:] == config.target_lane)
    events = []
    pre = int(math.floor(config.pre_s / dt + 1e-8))
    post = int(math.floor(config.max_window_s / dt + 1e-8))
    for b, w, j in torch.nonzero(insert, as_tuple=False).tolist():
        e = w + 1
        behind = (valid[b, e] & (lane[b, e] == config.target_lane)
                  & (kin[b, e, :, 0] < kin[b, e, j, 0]))
        candidates = torch.nonzero(behind, as_tuple=False).flatten()
        if not candidates.numel():
            continue
        f = int(candidates[kin[b, e, candidates, 0].argmax()])
        # Never skip a nearer CAV to supervise a more distant HDV.
        if bool(targets.is_cav[b, e, f]):
            continue
        start = e
        while start > max(0, e - pre):
            t = start - 1
            if not bool(hdv[b, t, f]) or int(lane[b, t, f]) != config.target_lane:
                break
            start = t
        end, recovered = e - 1, False
        # a[t+1] belongs to the transition from recorded t to recorded t+1.
        for t in range(e, min(n, e + post)):
            if not bool(hdv[b, t, f]) or int(lane[b, t, f]) != config.target_lane \
                    or int(lane[b, t + 1, f]) != config.target_lane:
                break
            end = t
            if float(kin[b, t + 1, f, 5]) >= 0:
                recovered = True
                break
        if end >= start:
            events.append({"batch": b, "inserter": j, "follower": f,
                           "step": e, "start": start, "end": end,
                           "censored": not recovered})
    return events


@torch.no_grad()
def insertion_plan(targets: UnrollTargets, config: CutinReactionConfig) -> dict[str, Any]:
    """Per-event unit-mass weights and non-insertion controls (no graph cache)."""
    _, lane, _, _, _, hdv = _recorded(targets)
    window = torch.zeros_like(hdv)
    mass = torch.zeros_like(hdv, dtype=torch.float32)
    events = insertion_events(targets, config)
    for event in events:
        b, f, lo, hi = (event[x] for x in ("batch", "follower", "start", "end"))
        rows = hdv[b, lo:hi + 1, f]
        window[b, lo:hi + 1, f] = True
        mass[b, lo:hi + 1, f] += rows.float() / rows.sum().clamp_min(1)
    if events:
        mass /= len(events)
    control = hdv & (lane[:, :-1] == config.target_lane) \
        & (lane[:, 1:] == config.target_lane) & ~window
    return {"events": events, "event_weights": mass, "window": window,
            "supervised": (mass > 0), "control": control,
            "n_events": len(events), "n_event_rows": int((mass > 0).sum()),
            "n_control_rows": int(control.sum())}


def one_step_predicted_accel(model: Any, targets: UnrollTargets,
                             grad_steps: Tensor | None = None) -> Tensor:
    """Causal recorded-state replay [B,n,K] (see :func:`one_step_replay`)."""
    return one_step_replay(model, targets, grad_steps)[0]


def one_step_replay(model: Any, targets: UnrollTargets, grad_steps: Tensor | None = None,
                    collect: tuple[str, ...] = ()) -> tuple[Tensor, dict[str, Tensor]]:
    """Causal recorded-state replay [B,n,K], including temporal and slow history.

    Returns ``(accel, collected)``: ``collected[key]`` stacks the DETACHED per-step model output
    ``key`` to ``[B, n, ...]`` for every key in ``collect`` the model returns (R10B: the courtesy
    head inputs share this replay; collecting never changes the computed values).

    ``grad_steps`` (optional bool [n]) limits autograd to the steps that carry a supervised
    row: the other steps still advance the causal recurrent state but run under ``no_grad``
    (their carried state is a constant for later steps, i.e. truncated backpropagation).
    ``None`` keeps every step differentiable.  With ``model.checkpoint_forward`` each
    differentiable step is activation-checkpointed exactly like ``WmSModel.step``.

    Kinematics and history features are recorded/detached. Only learned recurrent
    state (e.g. slow residual) and the model outputs carry gradients. No rollout
    positions, future accelerations, or event/window labels enter predictions.
    A fresh graph is built on every call, independent of previous runner backward.
    """
    kin, lane, valid, reset, _, _ = _recorded(targets)
    prep = targets.prepared.get("cutin_reaction_inputs", {})
    b, _, k = kin.shape[:3]
    if targets.n_steps == 0:
        return kin.new_empty(b, 0, k), {}
    history = prep.get("history")
    mask = prep.get("history_mask")
    if history is not None:
        history, mask = history.detach(), mask.detach().bool()
    driver_history = prep.get("driver_history")
    driver_mask = prep.get("driver_history_mask")
    if driver_history is not None:
        driver_history, driver_mask = driver_history.detach(), driver_mask.detach().bool()
    elif bool(getattr(model, "long_history", False)):
        # Explicitly masked unknown prefix, not fabricated observed history.
        driver_history = kin.new_zeros(b, k, model.driver_history_len, LONG_HISTORY_F)
        driver_mask = torch.zeros(driver_history.shape[:-1], dtype=torch.bool, device=kin.device)
    tracker = LeaderChangeTracker(b, k, dt=targets.dt_s, device=kin.device)
    carry: dict[str, Tensor] = {}
    recurrent: dict[str, Tensor] = {}
    age = prep.get("lane_age0")
    if age is None and hasattr(model, "_initial_lane_age"):
        age = model._initial_lane_age({}, valid[:, 0], kin[:, 0],
                                      driver_history, driver_mask)
    age = kin.new_full((b, k), 999.) if age is None else age.detach()
    params = embedding = None
    outputs = []
    collected: dict[str, list[Tensor]] = {key: [] for key in collect}
    for t in range(targets.n_steps):
        present = valid[:, t]
        new = reset[:, t] | (present & ~valid[:, t - 1] if t else torch.zeros_like(present))
        if t:
            age = torch.where(lane[:, t] != lane[:, t - 1], 0., age + 1)
            age = torch.where(new, 999., age)
        tracker.reset_slots(new)
        features = tracker.step(kin[:, t, :, 0], lane[:, t], kin[:, t, :, 4],
                                present, kin[:, t, :, 6])
        for tensors in (carry, recurrent):
            for key, value in tensors.items():
                expanded = new.reshape(b, k, *([1] * (value.ndim - 2)))
                tensors[key] = torch.where(expanded, torch.zeros_like(value), value)
        kwargs: dict[str, Any] = {"lane_age": age, **recurrent}
        if driver_history is not None:
            kwargs.update(driver_history=driver_history,
                          driver_history_mask=driver_mask & ~new[..., None])
        if hasattr(model, "history_len"):
            gap, _ = lead_neighbour(kin[:, t], present, model.physics)
            feat = history_step_features(kin[:, t], gap)
            if history is None:
                history, mask = initial_history(feat, present, model.history_len)
            else:
                history = torch.where(new[..., None, None], 0., history)
                mask = mask & ~new[..., None]
                if t:
                    history, mask = push_history(history, mask, feat, present)
            kwargs.update(history=history, history_mask=mask)
        state = {"veh_kin": kin[:, t].float(), "present": present,
                 "is_cav": targets.is_cav[:, t].bool(), **carry}
        if bool(getattr(model, "_tracks_merge_wait", False)):
            if t == 0:
                state["merge_wait_s"] = model.initial_merge_wait(
                    present, kin[:, t], driver_history, driver_mask)
            else:
                state["merge_wait_s"] = update_merge_wait(
                    carry.get("merge_wait_s"), kin[:, t], present, model.physics)
        if bool(getattr(model, "leader_change_features", False)):
            state["leader_change_features"] = feature_matrix(features)
        if hasattr(model, "encode_drivers") and (t == 0 or bool(new.any())):
            enc_kwargs = dict(history=history, history_mask=mask)
            if driver_history is not None:
                enc_kwargs.update(driver_history=driver_history,
                                  driver_history_mask=driver_mask & ~new[..., None])
            _, fresh_embedding, fresh_params = model.encode_drivers(state, **enc_kwargs)
            if t == 0:
                params, embedding = fresh_params, fresh_embedding
            elif fresh_params is not None and params is not None:
                params = {name: torch.where(new, fresh_params[name], value)
                          for name, value in params.items()}
            if t and embedding is not None and fresh_embedding is not None:
                embedding = torch.where(new[..., None], fresh_embedding, embedding)
        if params is not None:
            kwargs["driver_params"] = params
        if embedding is not None:
            kwargs["driver_embedding"] = embedding
        grad_here = grad_steps is None or bool(grad_steps[t])
        if not grad_here:
            with torch.no_grad():
                out = model(state, k_steps=t, **kwargs)
        elif bool(getattr(model, "checkpoint_forward", False)) and torch.is_grad_enabled():
            out = checkpoint(partial(model.forward, k_steps=t, **kwargs), state,
                             use_reentrant=False, preserve_rng_state=model.training)
        else:
            out = model(state, k_steps=t, **kwargs)
        outputs.append(out["accel_mps2"])
        for key, values in collected.items():
            if key in out:
                values.append(out[key].detach())
        carry = {key: out[key] for key in ("temporal_residual_mps2", "temporal_residual_present",
                                          "merge_neg_accel_ema", "merge_neg_intent_logit")
                 if key in out}
        if "merge_wait_s" in state:
            carry["merge_wait_s"] = state["merge_wait_s"]
        recurrent = {key: out[key] for key in ("lane_v5_acc", "lane_blocked") if key in out}
    stacked = {key: torch.stack(values, 1) for key, values in collected.items()
               if len(values) == targets.n_steps}
    return torch.stack(outputs, 1), stacked


class CutinReactionLoss(UnrollTerm):
    name = TERM_NAME
    differentiable = True

    def __init__(self, config: CutinReactionConfig, share_keys: tuple[str, ...] = (),
                 share_to: str | None = None) -> None:
        self.config = config
        self.last_stats: dict[str, Any] = {}
        # R10B: when another recorded-state term needs detached outputs of the SAME replay
        # (the courtesy loss), collect them here and leave them in targets.prepared[share_to].
        self.share_keys = tuple(share_keys)
        self.share_to = share_to

    def __call__(self, ctx: UnrollContext) -> TermValue:
        # A connected zero is safe for the runner's unconditional unroll backward,
        # including enabled batches without a single supervised row.
        zero = torch.zeros((), device=ctx.targets.states.device, requires_grad=torch.is_grad_enabled())
        if not self.config.enabled:
            return TermValue(zero, valid_count=0)
        if not torch.is_grad_enabled():
            # Validation / forward-only probes: a recorded-state replay there would only double
            # the cost and put a training-only term into validation losses.
            self.last_stats = {}
            return TermValue(zero, valid_count=0)
        plan = insertion_plan(ctx.targets, self.config)
        self.last_stats = {key: plan[key] for key in ("n_events", "n_event_rows", "n_control_rows")}
        self.last_stats["censored_windows"] = sum(e["censored"] for e in plan["events"])
        if not plan["n_event_rows"] and not plan["n_control_rows"]:
            return TermValue(zero, valid_count=0)
        needed = plan["supervised"] | plan["control"]
        accel, shared = one_step_replay(ctx.model, ctx.targets,
                                        grad_steps=needed.any(-1).any(0), collect=self.share_keys)
        if self.share_to is not None and shared:
            ctx.targets.prepared[self.share_to] = shared
        target = ctx.targets.states[:, 1:ctx.targets.n_steps + 1, :, 5].detach()
        # Invalid/padding rows must not poison a reduction via NaN * 0.
        error = F.huber_loss(accel[needed], target[needed],
                             delta=self.config.huber_delta_mps2, reduction="none")
        event = (error * plan["event_weights"][needed]).sum()
        control = (error * plan["control"][needed]).sum() / max(plan["n_control_rows"], 1)
        value = event + self.config.control_weight * control
        self.last_stats.update(event_loss=float(event.detach()), control_loss=float(control.detach()))
        return TermValue(self.config.weight * value, log=value,
                         valid_count=plan["n_event_rows"] + plan["n_control_rows"])


def cutin_reaction_terms(config: CutinReactionConfig) -> list[UnrollTerm]:
    return [CutinReactionLoss(config)] if config.enabled else []
