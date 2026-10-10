"""Braking-response fix for WM-S (loss terms + asymmetric residual cap).

Offline diagnosis DIAG120 found that long-horizon blow-ups are *missed jams* and
systematic *under-braking*: predicted speed is too high upstream and at the
merge, the error starts at one vehicle and propagates **backward** car to car
with a 1.0-1.5 s follower lag.  The supplied diagnosis identifies rare braking
events as under-weighted by a symmetric mean Huber loss, especially on
exactly the steps that carry the jam.

Everything here is flag-gated through :class:`BrakefixConfig`; with the default
config :attr:`BrakefixConfig.any_on` is ``False``, the trainer never calls into
this module and every loss/log value is byte-identical to the pre-fix trainer.

Terms (each its own flag; all are differentiable only through the *predicted*
quantity - every mask/multiplier is computed from detached values):

``asym_speed`` (``--asym-speed-loss ALPHA``)
    Elements where the prediction is too fast (``v_pred > v_rec`` for the speed
    loss, ``s_pred > s_rec`` for the position loss) get total weight ``ALPHA``;
    too-slow elements keep weight 1.  Logged term = ``(ALPHA - 1)`` x the
    too-fast Huber mass, so ``base + asym_speed`` is the asymmetric loss.

``decel_weight`` (``--decel-weight W --decel-threshold A``)
    Every HDV step whose *recorded* acceleration is below ``-A`` m/s^2 gets
    total weight ``W`` on the speed+position loss.  Logged term = ``(W - 1)`` x
    the (asym-weighted) Huber mass on those steps.

``propagation`` (``--propagation-loss W --prop-window S``)
    For every HDV follower whose recorded same-lane nearest leader (ahead,
    optionally limited by ``prop_max_gap_m``) brakes harder than ``-A`` at transition ``t``,
    the follower's predicted speed *and* acceleration are matched to the
    recording over the next ``S`` seconds after each braking step (transitions
    ``t+1 .. t+S/dt``; the
    measured follower lag is 1.0-1.5 s = 2-3 steps).  This directly supervises
    the backward wave.  Fully vectorised (pairwise leader search per time
    chunk, ``cumsum`` window), no Python loop over vehicles and no host sync.

Recorded acceleration: ``states[..., 5]`` at state ``t+1`` is assumed to label
the ``t -> t+1`` transition. Real TRAIN channel/finite-difference alignment
must be checked by the coordinator (see ``_out/BRAKEFIX_READY.md``).

``bounded_value_asym`` implements ``--residual-cap-neg-mps2``: a tanh cap on the
learned residual that may be larger on the braking (negative) side than on the
accelerating side.  ``None`` on a side means "no cap on that side".
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

#: Pairwise leader-search tensor budget (elements) per time chunk.
_PAIRWISE_BUDGET = 4_000_000


@dataclass(frozen=True)
class BrakefixConfig:
    """Hyper-parameters of the braking-response terms (all default OFF)."""

    asym_alpha: float = 1.0          # >1 enables the asymmetric speed/position loss
    decel_weight: float = 1.0        # >1 enables the braking-step weight multiplier
    decel_threshold_mps2: float = 3.0  # A: "braking" = recorded accel < -A (A > 0)
    prop_weight: float = 0.0         # >0 enables the backward-propagation loss
    prop_window_s: float = 3.0       # S: follower supervision window after onset
    prop_max_gap_m: Optional[float] = None  # unlimited recorded nearest-leader gap by default
    prop_accel_huber_mps2: float = 1.0
    dt_s: float = 0.5

    def __post_init__(self) -> None:
        if not all(math.isfinite(float(x)) for x in vars(self).values() if x is not None):
            raise ValueError("braking hyper-parameters must be finite")
        if float(self.asym_alpha) < 1.0:
            raise ValueError("asym_alpha must be >= 1 (1 = off)")
        if float(self.decel_weight) < 1.0:
            raise ValueError("decel_weight must be >= 1 (1 = off)")
        if float(self.decel_threshold_mps2) <= 0.0:
            raise ValueError("decel_threshold_mps2 must be > 0 (a deceleration magnitude)")
        if float(self.prop_weight) < 0.0:
            raise ValueError("prop_weight must be >= 0 (0 = off)")
        if float(self.prop_window_s) <= 0.0 or float(self.dt_s) <= 0.0:
            raise ValueError("prop_window_s and dt_s must be positive")
        if (self.prop_max_gap_m is not None and float(self.prop_max_gap_m) <= 0.0) \
                or float(self.prop_accel_huber_mps2) <= 0.0:
            raise ValueError("prop_max_gap_m and prop_accel_huber_mps2 must be positive")

    @property
    def asym_on(self) -> bool:
        return float(self.asym_alpha) > 1.0

    @property
    def decel_on(self) -> bool:
        return float(self.decel_weight) > 1.0

    @property
    def prop_on(self) -> bool:
        return float(self.prop_weight) > 0.0

    @property
    def any_on(self) -> bool:
        return self.asym_on or self.decel_on or self.prop_on

    @property
    def window_steps(self) -> int:
        return max(1, int(round(float(self.prop_window_s) / float(self.dt_s))))

    @property
    def log_terms(self) -> tuple[str, ...]:
        """Loss-term names this config adds to the train log / loss share."""
        out = []
        if self.asym_on:
            out.append("asym_speed")
        if self.decel_on:
            out.append("decel_weight")
        if self.prop_on:
            out.append("propagation")
        return tuple(out)


# --------------------------------------------------------------------------- caps
def bounded_value_asym(x: Tensor, cap_pos: Optional[float],
                       cap_neg: Optional[float]) -> Tensor:
    """tanh cap with separate positive/negative limits.

    ``cap_pos * tanh(x / cap_pos)`` for ``x >= 0`` and ``cap_neg * tanh(x /
    cap_neg)`` for ``x < 0``; ``None`` leaves that side uncapped (identity).  The
    two branches have unit slope at 0, so the function is C^1 and gradients are
    finite everywhere.  ``(None, None)`` is the identity.
    """
    for cap in (cap_pos, cap_neg):
        if cap is not None and (not math.isfinite(float(cap)) or float(cap) <= 0.0):
            raise ValueError("caps must be finite and > 0")
    pos = x if cap_pos is None else float(cap_pos) * torch.tanh(x / float(cap_pos))
    neg = x if cap_neg is None else float(cap_neg) * torch.tanh(x / float(cap_neg))
    return torch.where(x < 0, neg, pos)


def required_log_terms(log_path: Path, config_path: Optional[Path] = None) -> tuple[str, ...]:
    """Enabled losses from run metadata, independent of observed log keys.

    The on-path config is written before any updates, so interrupted runs also
    retain their required-term contract. Historical logs without metadata retain
    the old gate requirements. An explicit path can be a config or TRAIN receipt.
    """
    path = config_path
    if path is None:
        for name in ("BRAKEFIX_CONFIG.json", "TRAIN_RECEIPT.json"):
            candidate = Path(log_path).with_name(name)
            if candidate.is_file():
                path = candidate
                break
    if path is None:
        return ()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = data.get("config", data)
    return BrakefixConfig(asym_alpha=cfg.get("asym_speed_alpha", 1.0),
                          decel_weight=cfg.get("decel_weight", 1.0),
                          prop_weight=cfg.get("propagation_weight", 0.0)).log_terms


# ----------------------------------------------------------------- weight helpers
def asym_multiplier(pred: Tensor, target: Tensor, alpha: float) -> Tensor:
    """``alpha`` where ``pred > target`` (too fast / too far ahead) else ``1``."""
    return 1.0 + (float(alpha) - 1.0) * (pred.detach() > target).to(pred.dtype)


def braking_step_mask(a_rec: Tensor, valid: Tensor, threshold: float) -> Tensor:
    """Valid HDV steps whose recorded acceleration is below ``-threshold``."""
    return valid.bool() & (a_rec < -float(threshold))


def decel_multiplier(a_rec: Tensor, valid: Tensor, weight: float,
                     threshold: float) -> Tensor:
    """``weight`` on braking steps, ``1`` elsewhere (float, detached)."""
    mask = braking_step_mask(a_rec, valid, threshold)
    return 1.0 + (float(weight) - 1.0) * mask.to(a_rec.dtype)


# ------------------------------------------------------------------ propagation
@torch.no_grad()
def leader_brake_events(s: Tensor, lane: Tensor, valid: Tensor, length: Tensor,
                        a_next: Tensor, *, threshold: float, max_gap_m: Optional[float] = None,
                        budget: int = _PAIRWISE_BUDGET) -> Tensor:
    """Per follower slot and transition: does its recorded leader brake hard?

    ``s, lane, valid, length`` are ``[B, T+1, K]`` recorded state sequences
    (``valid`` = slot present and on the modelled road); ``a_next[b, t, k]`` is
    the recorded acceleration of the ``t -> t+1`` transition (``[B, T, K]``).
    Returns ``[B, T, K]`` bool: at state ``t`` the slot's nearest same-lane
    vehicle ahead (optionally bumper gap ``<= max_gap_m``) is still present at ``t + 1`` and
    its recorded acceleration is ``< -threshold``.  Pairwise search is done per
    time chunk (bounded memory); there is no loop over vehicles and no host sync.
    """
    b_n, t1, k_n = s.shape
    t_n = t1 - 1
    chunk = max(1, int(budget // max(1, b_n * k_n * k_n)))
    out = []
    for t0 in range(0, t_n, chunk):
        t1_ = min(t_n, t0 + chunk)
        st, ln, vd, le = (x[:, t0:t1_] for x in (s, lane, valid, length))
        ds = st.unsqueeze(-2) - st.unsqueeze(-1)            # [B,c,i,j] = s_j - s_i
        same = ln.unsqueeze(-2) == ln.unsqueeze(-1)
        cand = same & vd.unsqueeze(-2) & vd.unsqueeze(-1) & (ds > 0)
        dist = torch.where(cand, ds, torch.full_like(ds, float("inf")))
        dmin, jdx = dist.min(dim=-1)                          # nearest leader j
        has = torch.isfinite(dmin)
        gap = dmin - torch.gather(le, -1, jdx)
        a_lead = torch.gather(a_next[:, t0:t1_], -1, jdx)
        still = torch.gather(valid[:, t0 + 1:t1_ + 1], -1, jdx)
        close = torch.ones_like(has) if max_gap_m is None else gap <= float(max_gap_m)
        out.append(has & still & close & (a_lead < -float(threshold)))
    if not out:
        return torch.zeros(b_n, 0, k_n, dtype=torch.bool, device=s.device)
    return torch.cat(out, dim=1)


@torch.no_grad()
def propagation_window(events: Tensor, window_steps: int) -> Tensor:
    """Transitions within ``window_steps`` AFTER an onset (``t+1 .. t+L``).

    ``events[b, t, k]`` marks an onset at transition ``t``; the returned
    ``[B, T, K]`` bool is true at transition ``u`` iff an onset occurred in
    ``[u - L, u - 1]``.  ``cumsum`` formulation, vectorised over everything.
    """
    b_n, t_n, k_n = events.shape
    csum = torch.cat((torch.zeros(b_n, 1, k_n, dtype=torch.int32, device=events.device),
                      events.to(torch.int32).cumsum(dim=1)), dim=1)   # csum[u]=sum ev[<u]
    u = torch.arange(t_n, device=events.device)
    lo = (u - int(window_steps)).clamp_min(0)
    return (csum[:, u] - csum[:, lo]) > 0


def propagation_terms(pred_speed: Tensor, pred_accel: Tensor, rec_speed: Tensor,
                      rec_accel: Tensor, mask: Tensor, weights: Tensor,
                      denom: Tensor, *, speed_delta: float,
                      accel_delta: float) -> Tensor:
    """Masked, step-weighted Huber of follower speed and acceleration.

    Same normalisation convention as the base per-car terms (global valid-count
    ``denom`` and decaying step ``weights``) so the term is a pure re-weighting
    of follower steps in the propagation window.  Returns a 0-d tensor (no sync).
    """
    spd = F.huber_loss(pred_speed, rec_speed, delta=float(speed_delta), reduction="none")
    acc = F.huber_loss(pred_accel, rec_accel, delta=float(accel_delta), reduction="none")
    m = mask.to(weights.dtype)
    return ((spd + acc) * weights * m).sum() / denom


def realized_acceleration(speeds: Tensor, initial_speed: Tensor, dt_s: float) -> Tensor:
    """Acceleration of actual predicted speed transitions, including speed floor,
    projection and any configured state noise. Raw model acceleration stays intact.
    This uses the chunk's predicted initial speed, never a teacher-forced HDV state.
    """
    previous = torch.cat((initial_speed.unsqueeze(1), speeds[:, :-1]), dim=1)
    return (speeds - previous) / float(dt_s)


# --------------------------------------------------------------------- the terms
def braking_terms(cfg: BrakefixConfig, *, pos: Tensor, spd: Tensor,
                  pred_pos: Tensor, rec_pos: Tensor, pred_spd: Tensor,
                  rec_spd: Tensor, a_rec: Tensor, hdv_mask: Tensor,
                  weights: Tensor, denom: Tensor, position_weight: float,
                  speed_weight: float) -> dict[str, Tensor]:
    """Additive asym/decel loss terms for one chunk (0-d tensors, no host sync).

    ``pos``/``spd`` are the unreduced base Huber losses.  The returned terms
    satisfy ``base + sum(terms) == sum(w * m_asym * m_decel * base) / denom``.
    """
    m = hdv_mask.to(weights.dtype)
    base_p = float(position_weight) * pos * weights * m
    base_s = float(speed_weight) * spd * weights * m
    if cfg.asym_on:
        m_ap = asym_multiplier(pred_pos, rec_pos, cfg.asym_alpha)
        m_as = asym_multiplier(pred_spd, rec_spd, cfg.asym_alpha)
    else:
        m_ap = m_as = None
    out: dict[str, Tensor] = {}
    if cfg.asym_on:
        out["asym_speed"] = (((m_ap - 1.0) * base_p).sum()
                             + ((m_as - 1.0) * base_s).sum()) / denom
    if cfg.decel_on:
        m_d = decel_multiplier(a_rec, hdv_mask, cfg.decel_weight,
                               cfg.decel_threshold_mps2) - 1.0
        wp = base_p if m_ap is None else m_ap * base_p
        ws = base_s if m_as is None else m_as * base_s
        out["decel_weight"] = ((m_d * wp).sum() + (m_d * ws).sum()) / denom
    return out


# ----------------------------------------------------------------- audit helpers
def min_accel_reach(*, rollout_mode: str, decel_i: float, cap_pos: Optional[float],
                    cap_neg: Optional[float], accel_min: float = -9.0) -> float:
    """Most negative acceleration the model can emit (m/s^2, <= 0).

    Mirrors ``WmSModel.forward``: ``accel = clamp(physics + correction,
    accel_min, accel_max)`` where, in ``driver_params``/``hybrid`` mode, the
    physics term is first clamped to ``>= -decel_i``; ``driver_params`` has no
    residual; the correction is ``bounded_value_asym(gate * delta, cap_pos,
    cap_neg)`` with ``gate*delta`` unbounded when the side is uncapped.
    """
    if rollout_mode == "residual":
        phys = accel_min
    else:
        phys = max(-float(decel_i), accel_min)
    if rollout_mode == "driver_params":
        return phys
    neg_cap = cap_neg if cap_neg is not None else cap_pos
    if neg_cap is None:
        return accel_min
    return max(accel_min, phys - float(neg_cap))
