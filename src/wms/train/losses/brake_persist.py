"""BRAKE_DIAG persistence terms: multi-step speed change and onset quantiles.

BRAKE_DIAG (2026-10-20) found that WM HDV braking is a one-interval pulse: mean braking
duration 2.2 s vs SUMO 6.5 s, 69 % fresh onsets vs 9 %, and the leader's acceleration is
back to +0.11 m/s^2 one second later vs -1.44.  This module adds the learned fix
recommended in section 5 of the diagnosis, behind default-off keys:

``brake_persist``
    For every recorded HDV braking onset (recorded acceleration below
    ``-onset_decel_mps2`` after at least one second without braking), penalise the
    error of the accumulated speed change (the integral of deceleration) over the
    next ``window_s`` seconds between the WM rollout and the recording, per vehicle.

``brake_persist_quantile``
    Pinball losses at tau = 0.1/0.5/0.9 on per-onset peak deceleration and braking
    duration inside the same window.  Both predicted statistics are differentiable
    proxies: peak deceleration is a soft-min of the predicted accelerations and the
    duration is a sigmoid-counted number of braking steps.

``event_balance`` (optional)
    Inverse-frequency weights over (gap x closing speed x leader decel) cells computed
    from the recorded onsets of the whole unroll (BRAKE_DIAG section 5.1), so the
    over-produced stopped/opening close-gap seconds are down-weighted.  The value is
    the inverse-frequency exponent (0 = off, 1 = full balancing).

Every mask and target is computed from recorded data (detached); only the predicted
speeds carry gradient.  Onsets whose full window does not fit in the current BPTT
chunk are skipped on that chunk (the term is evaluated per chunk and no prediction
graph crosses a chunk boundary).  ``window_s`` is restricted to 3-5 s.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import LossContext, LossTerm, TermValue, UnrollTargets

#: Pinball quantiles of the peak-deceleration / braking-duration distributions.
QUANTILES = (0.1, 0.5, 0.9)
#: Cell layout of the event-balance weights: 4 gap x 4 closing x 4 leader-decel bins.
_GAP_BINS = 4
_CLOSING_BINS = 4
_DECEL_BINS = 4
_N_CELLS = _GAP_BINS * _CLOSING_BINS * _DECEL_BINS
#: No same-lane leader within ``max_gap_m`` (or no leader at all).
_NO_LEADER_BIN = 3


@dataclass(frozen=True)
class BrakePersistConfig:
    """Hyper-parameters of the persistence terms (all default OFF)."""

    weight: float = 0.0
    quantile_weight: float = 0.0
    event_balance: float = 0.0
    window_s: float = 4.0
    onset_decel_mps2: float = 1.0
    max_gap_m: float = 40.0
    speed_huber_delta_mps: float = 0.5
    softmin_temp_mps2: float = 0.25
    duration_temp_mps2: float = 0.25

    def __post_init__(self) -> None:
        if not all(math.isfinite(float(x)) for x in (
                self.weight, self.quantile_weight, self.event_balance, self.window_s,
                self.onset_decel_mps2, self.max_gap_m, self.speed_huber_delta_mps,
                self.softmin_temp_mps2, self.duration_temp_mps2)):
            raise ValueError("brake_persist hyper-parameters must be finite")
        if float(self.weight) < 0.0 or float(self.quantile_weight) < 0.0:
            raise ValueError("brake_persist weights must be >= 0 (0 = off)")
        if not 0.0 <= float(self.event_balance) <= 4.0:
            raise ValueError("brake_persist.event_balance must be in [0, 4] (0 = off)")
        if not 3.0 <= float(self.window_s) <= 5.0:
            raise ValueError("brake_persist.window_s must be in [3, 5] seconds")
        if float(self.onset_decel_mps2) <= 0.0 or float(self.max_gap_m) <= 0.0:
            raise ValueError("onset_decel_mps2 and max_gap_m must be positive")
        if float(self.speed_huber_delta_mps) <= 0.0 or float(self.softmin_temp_mps2) <= 0.0 \
                or float(self.duration_temp_mps2) <= 0.0:
            raise ValueError("brake_persist Huber/temperature scales must be positive")

    @property
    def enabled(self) -> bool:
        return float(self.weight) > 0.0 or float(self.quantile_weight) > 0.0

    def window_steps_for(self, dt_s: float) -> int:
        return max(1, int(round(float(self.window_s) / float(dt_s))))

    @property
    def log_terms(self) -> tuple[str, ...]:
        return tuple(name for name, on in (("brake_persist", self.weight > 0.0),
                                           ("brake_persist_quantile",
                                            self.quantile_weight > 0.0)) if on)


def _lead(x: Tensor, steps: int) -> Tensor:
    """``out[:, r] = x[:, r + steps]`` (future sample), zero-padded at the end."""
    steps = int(steps)
    if steps <= 0:
        return x
    out = torch.zeros_like(x)
    if steps < x.shape[1]:
        out[:, :x.shape[1] - steps] = x[:, steps:]
    return out


def _lag(x: Tensor, steps: int) -> Tensor:
    """``out[:, r] = x[:, r - steps]`` (past sample), zero-padded at the start."""
    steps = int(steps)
    if steps <= 0:
        return x
    out = torch.zeros_like(x)
    if steps < x.shape[1]:
        out[:, steps:] = x[:, :x.shape[1] - steps]
    return out


def _window_stack(x: Tensor, steps: int) -> Tensor:
    """``out[:, r, ..., j] = x[:, r + j, ...]`` (zero-padded) for a window of ``steps``."""
    return torch.stack([_lead(x, j) for j in range(int(steps))], dim=-1)


@torch.no_grad()
def onset_mask(targets: UnrollTargets, *, onset_decel_mps2: float = 1.0) -> Tensor:
    """``[B, n_steps, K]`` recorded HDV braking onsets after >= 1 s without braking."""
    n = targets.n_steps
    states = targets.states
    dt_s = float(targets.dt_s)
    a_rec = states[:, 1:n + 1, :, 5]
    valid = (targets.hdv_mask & ~targets.is_cav[:, 1:n + 1]
             & targets.loss_valid[:, 1:n + 1])
    braking = valid & (a_rec < -float(onset_decel_mps2))
    clear_steps = max(1, int(math.ceil(1.0 / dt_s - 1e-9)))
    clear = torch.ones_like(braking)
    recent_valid = torch.ones_like(braking)
    for lag in range(1, clear_steps + 1):
        clear &= ~_lag(braking, lag)
        recent_valid &= _lag(valid, lag)
    enough = torch.arange(n, device=a_rec.device) >= clear_steps
    return braking & clear & recent_valid & enough.view(1, n, 1)


@torch.no_grad()
def _cell_ids(targets: UnrollTargets, onset: Tensor, *, max_gap_m: float) -> Tensor:
    """``[B, n_steps, K]`` recorded (gap x closing x leader-decel) cell id, -1 off-onset."""
    b_n, n, k_n = onset.shape
    states = targets.states
    cells = torch.full(onset.shape, -1, dtype=torch.long, device=onset.device)
    for t in range(n):
        if not bool(onset[:, t].any()):
            continue
        s_t = states[:, t, :, 0]
        lane = states[:, t, :, 1:4].argmax(dim=-1)
        on_road = states[:, t, :, 1:4].sum(dim=-1) > 0
        valid_t = targets.loss_valid[:, t] & on_road
        gap = s_t.unsqueeze(-2) - s_t.unsqueeze(-1)          # [B, follower, leader]
        same_lane = lane.unsqueeze(-2) == lane.unsqueeze(-1)
        ahead = (valid_t.unsqueeze(-2) & valid_t.unsqueeze(-1) & same_lane
                 & (gap > 0.0) & (gap <= float(max_gap_m)))
        dist = torch.where(ahead, gap, torch.full_like(gap, float("inf")))
        nearest, index = dist.min(dim=-1)
        has_leader = torch.isfinite(nearest)
        v = states[:, t, :, 4]
        a = states[:, t, :, 5]
        v_leader = torch.gather(v, -1, index)
        a_leader = torch.gather(a, -1, index)
        closing = v - v_leader
        gap_bin = torch.where(nearest <= 10.0, 0,
                              torch.where(nearest <= 20.0, 1,
                                          torch.where(has_leader, 2, _NO_LEADER_BIN)))
        closing_bin = torch.where(closing < 0.0, 0,
                                  torch.where(closing < 3.0, 1,
                                              torch.where(closing < 6.0, 2, 3)))
        decel_bin = torch.where(~has_leader, _NO_LEADER_BIN,
                                torch.where(a_leader >= -1.0, 0,
                                            torch.where(a_leader >= -2.5, 1, 2)))
        cell = (gap_bin * _CLOSING_BINS + closing_bin) * _DECEL_BINS + decel_bin
        cells[:, t] = torch.where(onset[:, t], cell, torch.full_like(cell, -1))
    return cells


@torch.no_grad()
def _balanced_weights(onset: Tensor, cells: Tensor, strength: float) -> Tensor:
    """Inverse-frequency weights over recorded onset cells (mean 1 over events)."""
    weights = torch.ones(onset.shape, dtype=torch.float32, device=onset.device)
    if float(strength) <= 0.0:
        return weights
    active = cells[onset]
    if active.numel() == 0:
        return weights
    counts = torch.bincount(active, minlength=_N_CELLS).to(torch.float32)
    inv = 1.0 / counts.clamp_min(1.0)
    mean_inv = inv[active].mean().clamp_min(1e-12)
    balanced = (inv[active] / mean_inv).pow(float(strength))
    weights[onset] = balanced.to(weights.dtype)
    return weights


@torch.no_grad()
def prepare_brake_persist(config: BrakePersistConfig, targets: UnrollTargets) -> dict:
    """Compute (once per unroll) the recorded onsets, cell weights and window length."""
    key = (float(config.window_s), float(config.onset_decel_mps2),
           float(config.max_gap_m), float(config.event_balance))
    cached = targets.prepared.get("brake_persist")
    if cached is not None and cached.get("key") == key:
        return cached
    n = targets.n_steps
    dt_s = float(targets.dt_s)
    window = config.window_steps_for(dt_s)
    onset = onset_mask(targets, onset_decel_mps2=float(config.onset_decel_mps2))
    if window <= n:
        fits = torch.arange(n, device=onset.device) + window <= n
        onset = onset & fits.view(1, n, 1)
    else:
        onset = torch.zeros_like(onset)
    cells = _cell_ids(targets, onset, max_gap_m=float(config.max_gap_m))
    weights = _balanced_weights(onset, cells, float(config.event_balance))
    result = {"key": key, "onset": onset, "cells": cells, "weights": weights,
              "window_steps": int(window), "dt_s": dt_s, "count": int(onset.sum()),
              "n_cells": int(torch.unique(cells[onset]).numel()) if bool(onset.any()) else 0}
    targets.prepared["brake_persist"] = result
    return result


def _chunk_events(ctx: LossContext, plan: dict) -> tuple[Tensor, Tensor]:
    """Onset mask and event weights of this chunk, restricted to windows fully inside it."""
    c0, c1 = int(ctx.c0), int(ctx.c1)
    window = int(plan["window_steps"])
    length = c1 - c0
    onset = plan["onset"][:, c0:c1]
    weights = plan["weights"][:, c0:c1].to(ctx.pred["speeds"].dtype)
    if length < window:
        zero = torch.zeros_like(onset)
        return zero, weights * zero.to(weights.dtype)
    fits = torch.arange(length, device=onset.device) + window <= length
    mask = onset & fits.view(1, length, 1)
    return mask, weights


def _speed_sequence(ctx: LossContext) -> Tensor:
    """Predicted speed at times ``c0 .. c1`` (``[B, length + 1, K]``)."""
    return torch.cat((ctx.current["v_mps"].unsqueeze(1), ctx.pred["speeds"]), dim=1)


def _event_weighted(loss: Tensor, mask: Tensor, weights: Tensor) -> Tensor:
    event_w = weights * mask.to(weights.dtype)
    denom = event_w.sum().clamp_min(1e-6)
    return (loss * event_w).sum() / denom


def _pinball(pred: Tensor, target: Tensor, tau: float) -> Tensor:
    difference = target.detach() - pred
    return torch.maximum(float(tau) * difference, (float(tau) - 1.0) * difference)


class _BrakePersistTerm(LossTerm):
    """Shared prepare and chunk-event helpers."""

    def __init__(self, config: BrakePersistConfig) -> None:
        self.config = config

    def prepare(self, targets: UnrollTargets) -> None:
        prepare_brake_persist(self.config, targets)

    def _plan(self, ctx: LossContext) -> dict:
        return ctx.targets.prepared["brake_persist"]


class BrakePersistLoss(_BrakePersistTerm):
    """Huber error of the accumulated speed change over the onset window."""

    name = "brake_persist"

    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        plan = self._plan(ctx)
        mask, weights = _chunk_events(ctx, plan)
        window = int(plan["window_steps"])
        c0, c1 = int(ctx.c0), int(ctx.c1)
        predicted = _speed_sequence(ctx)
        recorded = ctx.targets.states[:, c0:c1 + 1, :, 4]
        delta_pred = _lead(predicted, window)[:, :c1 - c0] - predicted[:, :c1 - c0]
        delta_rec = _lead(recorded, window)[:, :c1 - c0] - recorded[:, :c1 - c0]
        huber = F.smooth_l1_loss(delta_pred, delta_rec.detach(),
                                 beta=float(self.config.speed_huber_delta_mps),
                                 reduction="none")
        value = _event_weighted(huber, mask, weights)
        return TermValue(float(self.config.weight) * value,
                         log=value if float(self.config.weight) > 0.0 else None)


class BrakePersistQuantileLoss(_BrakePersistTerm):
    """Pinball losses on per-onset peak deceleration and braking duration."""

    name = "brake_persist_quantile"

    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        plan = self._plan(ctx)
        mask, weights = _chunk_events(ctx, plan)
        window = int(plan["window_steps"])
        dt_s = float(plan["dt_s"])
        c0, c1 = int(ctx.c0), int(ctx.c1)
        predicted = _speed_sequence(ctx)
        a_pred = (predicted[:, 1:] - predicted[:, :-1]) / dt_s
        a_rec = ctx.targets.states[:, c0 + 1:c1 + 1, :, 5]
        pred_window = _window_stack(a_pred, window)
        rec_window = _window_stack(a_rec, window)
        peak_pred = -(-float(self.config.softmin_temp_mps2) * torch.logsumexp(
            -pred_window / float(self.config.softmin_temp_mps2), dim=-1))
        peak_rec = (-rec_window.min(dim=-1).values).clamp_min(0.0)
        a_threshold = -float(self.config.onset_decel_mps2)
        duration_pred = dt_s * torch.sigmoid(
            (a_threshold - pred_window) / float(self.config.duration_temp_mps2)).sum(dim=-1)
        duration_rec = dt_s * (rec_window < a_threshold).sum(dim=-1).to(a_pred.dtype)
        loss = torch.zeros_like(peak_pred)
        for tau in QUANTILES:
            loss = loss + _pinball(peak_pred, peak_rec, tau)
            loss = loss + _pinball(duration_pred, duration_rec, tau)
        value = _event_weighted(loss, mask, weights)
        return TermValue(float(self.config.quantile_weight) * value,
                         log=value if float(self.config.quantile_weight) > 0.0 else None)


def brake_persist_terms(config: BrakePersistConfig) -> list[LossTerm]:
    """The enabled persistence terms, in a fixed order (empty when both weights are 0)."""
    terms: list[LossTerm] = []
    if float(config.weight) > 0.0:
        terms.append(BrakePersistLoss(config))
    if float(config.quantile_weight) > 0.0:
        terms.append(BrakePersistQuantileLoss(config))
    return terms


__all__ = ["BrakePersistConfig", "BrakePersistLoss", "BrakePersistQuantileLoss",
           "brake_persist_terms", "onset_mask", "prepare_brake_persist", "QUANTILES"]
