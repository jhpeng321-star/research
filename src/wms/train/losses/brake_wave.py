"""WAVE (2026-10-20): onset-conditioned braking-wave terms.

The round-2 C6/C2 diagnosis (`WAVE.md`) found the mechanism behind the missing braking
wave: at ``a < -1.5 m/s^2`` onsets the IDM core is almost neutral (mean -0.14 m/s^2) and
the learned residual supplies the whole onset, but as a large ONE-STEP pulse (gate 0.35,
pre-gate residual ~ -7.5 m/s^2, 87 % of onsets last one 0.5 s step, next-step residual
+0.45).  On the steps where IDM does brake, the residual is positive on 78 % of them and
41 % of those brakes are cancelled below the onset threshold.  Learned driver ``b`` is not
the problem (p50 3.0-4.0 vs SUMO 4.2) and the evaluation rollout is deterministic.

These terms are the cheap learned fix, all default OFF (weight 0 = never built, so an
existing checkpoint is bit-identical):

``brake_wave`` (``weight``)
    Event-level speed-drop loss.  For every recorded HDV braking onset (fresh
    ``a < -onset_decel_mps2``) compare the minimum speed drop over the next ``window_s``
    seconds (the C6 definition) with the recorded drop; events are weighted by their
    recorded drop (mean weight 1), so deep SUMO-like brakes dominate.

``brake_wave_residual`` (``residual_weight``)
    One-sided penalty on a POSITIVE learned residual (``raw_residual_mps2``) on recorded
    braking steps: it stops the residual from cancelling the IDM brake it was trained to
    correct.

``brake_wave_spurious`` (``spurious_weight``)
    On recorded NON-braking steps (``a >= -spurious_decel_mps2``) penalise a predicted
    deceleration below ``-spurious_decel_mps2``: it removes the one-step pulses that
    over-produce C6 onsets (91.8 vs 4.7 onsets per 1000 HDV-steps).

Every mask/target is detached recorded data; only the predicted speeds carry gradient.
No parameters are added, and all weights default to 0, so loading an old checkpoint is
an exact no-op.  ``window_s`` is restricted to 3-5 s like the other persistence term.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import LossContext, LossTerm, TermValue, UnrollTargets
from .brake_persist import _chunk_events, _lead, onset_mask


@dataclass(frozen=True)
class BrakeWaveConfig:
    """Hyper-parameters of the braking-wave terms (all default OFF)."""

    weight: float = 0.0
    residual_weight: float = 0.0
    spurious_weight: float = 0.0
    window_s: float = 4.0
    onset_decel_mps2: float = 1.0
    spurious_decel_mps2: float = 1.5
    speed_huber_delta_mps: float = 0.5
    residual_huber_delta_mps2: float = 0.5

    def __post_init__(self) -> None:
        if not all(math.isfinite(float(x)) for x in (
                self.weight, self.residual_weight, self.spurious_weight, self.window_s,
                self.onset_decel_mps2, self.spurious_decel_mps2,
                self.speed_huber_delta_mps, self.residual_huber_delta_mps2)):
            raise ValueError("brake_wave hyper-parameters must be finite")
        if (float(self.weight) < 0.0 or float(self.residual_weight) < 0.0
                or float(self.spurious_weight) < 0.0):
            raise ValueError("brake_wave weights must be >= 0 (0 = off)")
        if not 3.0 <= float(self.window_s) <= 5.0:
            raise ValueError("brake_wave.window_s must be in [3, 5] seconds")
        if float(self.onset_decel_mps2) <= 0.0 or float(self.spurious_decel_mps2) <= 0.0:
            raise ValueError("brake_wave decel thresholds must be positive")
        if float(self.speed_huber_delta_mps) <= 0.0 \
                or float(self.residual_huber_delta_mps2) <= 0.0:
            raise ValueError("brake_wave Huber scales must be positive")

    def window_steps_for(self, dt_s: float) -> int:
        return max(1, int(round(float(self.window_s) / float(dt_s))))

    @property
    def enabled(self) -> bool:
        return (float(self.weight) > 0.0 or float(self.residual_weight) > 0.0
                or float(self.spurious_weight) > 0.0)

    @property
    def log_terms(self) -> tuple[str, ...]:
        return tuple(name for name, on in (("brake_wave", self.weight > 0.0),
                                           ("brake_wave_residual", self.residual_weight > 0.0),
                                           ("brake_wave_spurious",
                                            self.spurious_weight > 0.0)) if on)


def _plan(config: BrakeWaveConfig, targets: UnrollTargets) -> dict:
    """Recorded onsets, per-event drop weights and the window length (cached)."""
    key = ("brake_wave", float(config.window_s), float(config.onset_decel_mps2))
    cached = targets.prepared.get("brake_wave")
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
    v_rec = targets.states[:, :, :, 4]
    drop = torch.zeros_like(onset, dtype=v_rec.dtype)
    if bool(onset.any()):
        # Minimum speed drop over the window, exactly the C6 wave amplitude definition.
        values = torch.stack([_lead(v_rec, j) for j in range(window + 1)], dim=-1)
        with torch.no_grad():
            drop = (v_rec[:, :n] - values[:, :n].min(dim=-1).values).clamp_min(0.0)
    active = drop[onset]
    weights = torch.ones_like(drop)
    if active.numel() > 0:
        mean = active.mean().clamp_min(1e-6)
        balanced = (active / mean).clamp(0.0, 8.0)
        weights[onset] = balanced.to(weights.dtype)
    result = {"key": key, "onset": onset, "drop_target": drop, "weights": weights,
              "window_steps": int(window), "dt_s": dt_s, "count": int(onset.sum())}
    targets.prepared["brake_wave"] = result
    return result


def _pred_speed_sequence(ctx: LossContext) -> Tensor:
    """Predicted speeds at the chunk start and after every step ``[B, len+1, K]``."""
    return torch.cat((ctx.current["v_mps"].unsqueeze(1), ctx.pred["speeds"]), dim=1)


def _pred_accel(ctx: LossContext) -> Tensor:
    """Predicted acceleration of every chunk step ``[B, len, K]``."""
    sequence = _pred_speed_sequence(ctx)
    return (sequence[:, 1:] - sequence[:, :-1]) / float(ctx.targets.dt_s)


def _recorded_accel(ctx: LossContext) -> Tensor:
    return ctx.targets.states[:, ctx.c0 + 1:ctx.c1 + 1, :, 5]


def _valid(ctx: LossContext) -> Tensor:
    return (ctx.targets.hdv_mask[:, ctx.c0:ctx.c1]
            & ~ctx.targets.is_cav[:, ctx.c0 + 1:ctx.c1 + 1]
            & ctx.targets.loss_valid[:, ctx.c0 + 1:ctx.c1 + 1])


class BrakeWaveDropLoss(LossTerm):
    """C6-style minimum speed drop over a recorded onset window."""

    name = "brake_wave"

    def __init__(self, config: BrakeWaveConfig) -> None:
        self.config = config

    def prepare(self, targets: UnrollTargets) -> None:
        _plan(self.config, targets)

    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        if float(self.config.weight) <= 0.0:
            return None
        plan = _plan(self.config, ctx.targets)
        mask, weights = _chunk_events(ctx, plan)
        window = int(plan["window_steps"])
        sequence = _pred_speed_sequence(ctx)
        length = int(ctx.c1) - int(ctx.c0)
        if length < window:
            return TermValue(torch.zeros((), device=sequence.device, dtype=sequence.dtype),
                             log=torch.zeros((), device=sequence.device, dtype=sequence.dtype))
        start = sequence[:, :length]
        values = torch.stack([_lead(sequence, j)[:, :length] for j in range(window + 1)],
                             dim=-1)
        pred_drop = (start - values.min(dim=-1).values).clamp_min(0.0)
        rec_sequence = ctx.targets.states[:, ctx.c0:ctx.c1 + 1, :, 4]
        rec_start = rec_sequence[:, :length]
        rec_values = torch.stack([_lead(rec_sequence, j)[:, :length]
                                  for j in range(window + 1)], dim=-1)
        rec_drop = (rec_start - rec_values.min(dim=-1).values).clamp_min(0.0).detach()
        loss = F.smooth_l1_loss(pred_drop, rec_drop,
                                beta=float(self.config.speed_huber_delta_mps),
                                reduction="none")
        event_w = weights * mask.to(weights.dtype)
        denom = event_w.sum().clamp_min(1e-6)
        value = (loss * event_w).sum() / denom
        return TermValue(float(self.config.weight) * value,
                         log=value if float(self.config.weight) > 0.0 else None)


class BrakeWaveResidualLoss(LossTerm):
    """One-sided penalty on a positive learned residual during recorded braking."""

    name = "brake_wave_residual"

    def __init__(self, config: BrakeWaveConfig) -> None:
        self.config = config

    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        if float(self.config.residual_weight) <= 0.0:
            return None
        residual = ctx.raw_residual
        if residual is None:
            return None
        a_rec = _recorded_accel(ctx)
        mask = _valid(ctx) & (a_rec < -float(self.config.onset_decel_mps2))
        positive = residual.clamp_min(0.0)
        loss = F.smooth_l1_loss(positive, torch.zeros_like(positive),
                                beta=float(self.config.residual_huber_delta_mps2),
                                reduction="none")
        denom = mask.to(loss.dtype).sum().clamp_min(1.0)
        value = (loss * mask.to(loss.dtype)).sum() / denom
        return TermValue(float(self.config.residual_weight) * value, log=value)


class BrakeWaveSpuriousLoss(LossTerm):
    """Penalise one-step WM decel pulses on recorded non-braking steps."""

    name = "brake_wave_spurious"

    def __init__(self, config: BrakeWaveConfig) -> None:
        self.config = config

    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        if float(self.config.spurious_weight) <= 0.0:
            return None
        a_pred = _pred_accel(ctx)
        a_rec = _recorded_accel(ctx)
        mask = _valid(ctx) & (a_rec >= -float(self.config.spurious_decel_mps2))
        excess = (-a_pred - float(self.config.spurious_decel_mps2)).clamp_min(0.0)
        loss = F.smooth_l1_loss(excess, torch.zeros_like(excess),
                                beta=float(self.config.residual_huber_delta_mps2),
                                reduction="none")
        denom = mask.to(loss.dtype).sum().clamp_min(1.0)
        value = (loss * mask.to(loss.dtype)).sum() / denom
        return TermValue(float(self.config.spurious_weight) * value, log=value)


def brake_wave_terms(config: BrakeWaveConfig) -> list[LossTerm]:
    """The enabled wave terms, in a fixed order (empty when all weights are 0)."""
    terms: list[LossTerm] = []
    if float(config.weight) > 0.0:
        terms.append(BrakeWaveDropLoss(config))
    if float(config.residual_weight) > 0.0:
        terms.append(BrakeWaveResidualLoss(config))
    if float(config.spurious_weight) > 0.0:
        terms.append(BrakeWaveSpuriousLoss(config))
    return terms


def prepare_brake_wave(config: BrakeWaveConfig, targets: UnrollTargets) -> dict:
    """Public prepare hook (also used by the tests)."""
    return _plan(config, targets)


__all__ = ["BrakeWaveConfig", "BrakeWaveDropLoss", "BrakeWaveResidualLoss",
           "BrakeWaveSpuriousLoss", "brake_wave_terms", "prepare_brake_wave"]
