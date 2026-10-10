"""Optional paired-effect head: climatology-anchored dV quantiles (Idea A).

Owner-approved 2026-10-04 (WM-N1 Idea A, CAL-tune follow-up).  This is a
DEFAULT-OFF component.  It predicts, per registered tau, a distribution over
the scalar paired effect ``dV = delay(treatment) - delay(native)`` on the
scorer's 19 quantile levels
(:data:`model.constants.QUANTILE_LEVELS`, the CAL-tune scorer grid).  The
prediction is

    dV_quantiles = TRAIN_climatology[t tau, level] + gate * residual(token)

where ``TRAIN_climatology`` is the per-tau empirical quantiles of the paired
dV built from TRAIN labels only (the ``wm_n1_train_climatology/1`` JSON the
scorer also consumes), and ``gate`` is a scalar shrinkage parameter
initialised to **exactly zero**: an untrained head returns the climatology
EXACTLY.  ``residual`` is a small MLP over the state global token.

Training (in :mod:`train.idea_a`) uses the pinball loss on outcome-supported
sibling contrast pairs only, through the existing contrast-pair machinery.

No label, arm name or strategy one-hot is an input: the head reads the same
state global token the correction head reads.

Plan-aware mode (A9, default OFF).  With ``plan_aware=True`` the residual MLP
takes ``concat(state token, plan vector)`` where the plan vector is the
TREATMENT arm's CAV-motion summary -- the SAME 16-D ``summary_raw`` the
correction head consumes (the 10 s / 20-step CAV-motion summary, not an arm
name).  The plan embedding is a small new linear layer ``plan_proj``
(``PLAN_RAW_F -> plan_dim``, default ``plan_dim = d_model``) owned by this head;
it is deliberately self-contained so all plan-aware parameters live under the
``paired.`` checkpoint prefix.  The gate is still zero-initialised, so an
untrained plan-aware head returns the climatology EXACTLY for every plan.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import constants as C

#: Climatology document schema (matches ``evalh.scorer.baselines``).
CLIMATOLOGY_SCHEMA = "wm_n1_train_climatology/1"
#: Registered contrast whose per-tau dV climatology anchors the head.
DEFAULT_CONTRAST = "pi-native"
#: Floor (seconds) of the per-tau loss scale s_tau = q90 - q10 (A8 scale fix).
LOSS_SCALE_FLOOR_S = 1.0
#: Raw CAV-motion summary width the plan embedding consumes.  Mirrors
#: :data:`corr.summary.SUMMARY_RAW_F` (asserted equal in the paired-effect
#: tests; kept local to avoid a model -> corr import cycle).
PLAN_RAW_F = 16
#: A11 FULL-HORIZON plan input (default OFF).  ``--paired-effect-plan-horizon-s
#: H`` replaces the 10 s / 20-step ``summary_raw`` with a fixed-size summary of
#: the TREATMENT arm's planned CAV motion over ``[t0, t0 + H]``, binned in
#: :data:`LONG_PLAN_BIN_S`-second bins.  Each bin contributes
#: :data:`LONG_PLAN_FEATS_PER_BIN` features (documented in
#: :func:`long_plan_summary`); the raw width is
#: :func:`long_plan_raw_f` ``= 5 * ceil(H / 10)`` (never 16, so the head's
#: ``plan_raw_f`` selects long-vs-short unambiguously).  Direct-speed plans run
#: ~120 s (H=120 -> 12 bins -> 60 features).
LONG_PLAN_BIN_S = 10.0
LONG_PLAN_FEATS_PER_BIN = 5
#: Long-plan per-bin feature order (documented, unit-frozen by tests).
LONG_PLAN_FIELDS = ("mean_speed_mps", "min_speed_mps", "mean_accel_mps2",
                    "frac_decelerating", "mean_cav_count")


def long_plan_n_bins(horizon_s) -> int:
    """Fixed bin count ``ceil(H / 10 s)`` (>= 1) for the long plan summary."""
    h = float(horizon_s)
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError(f"plan horizon must be finite and > 0; got {horizon_s!r}")
    return max(1, int(math.ceil(h / LONG_PLAN_BIN_S - 1e-9)))


def long_plan_raw_f(horizon_s) -> int:
    """Long-plan raw width ``LONG_PLAN_FEATS_PER_BIN * long_plan_n_bins(H)``."""
    return LONG_PLAN_FEATS_PER_BIN * long_plan_n_bins(horizon_s)


def long_plan_summary(speeds, accel, cav_mask, *, horizon_s, dt: float = 0.5
                      ) -> torch.Tensor:
    """Long-horizon treatment-arm CAV-motion summary ``[B, W]``.

    A11: the same controller *planned* CAV channels the 16-D ``summary_raw``
    uses (``cav_plan_ch`` fields 0/1, de-scaled to SI m/s and m/s^2), read over
    ``[t0, t0 + H]`` and binned into ``horizon_s / 10 s`` fixed bins.  Inputs
    are ``[T, B, K]``: ``speeds`` (m/s), ``accel`` (m/s^2) and a binary
    ``cav_mask`` (1 = the CAV slot is alive at that step).  Steps beyond
    ``n_bins * round(10 s / dt)`` are ignored; a short window simply leaves the
    trailing bins at zero.  Per bin (all statistics over ACTIVE CAV-slot
    observations in that bin; empty bins stay zero):

    ``0`` mean speed, ``1`` min speed, ``2`` mean acceleration,
    ``3`` fraction with ``accel < 0`` (strictly decelerating),
    ``4`` mean number of active CAVs per tick in the bin.

    Pure, deterministic, no Python RNG; the same tensors always yield the same
    vector.  Training (``train.idea_a``) and any inference caller must build
    the vector through this function so both paths agree exactly.
    """
    sp = torch.as_tensor(speeds, dtype=torch.float32)
    ac = torch.as_tensor(accel, dtype=torch.float32)
    m = torch.as_tensor(cav_mask, dtype=torch.float32)
    if sp.dim() != 3 or ac.shape != sp.shape or m.shape != sp.shape:
        raise ValueError(
            "speeds/accel/cav_mask must all be [T,B,K] and equal; got "
            f"{tuple(sp.shape)}, {tuple(ac.shape)}, {tuple(m.shape)}")
    t_steps, batch, _k = sp.shape
    n_bins = long_plan_n_bins(horizon_s)
    bin_steps = max(1, int(round(LONG_PLAN_BIN_S / float(dt))))
    out = sp.new_zeros(batch, n_bins * LONG_PLAN_FEATS_PER_BIN)
    for b in range(n_bins):
        lo, hi = b * bin_steps, min(t_steps, (b + 1) * bin_steps)
        if hi <= lo:
            continue
        sv, av, mv = sp[lo:hi], ac[lo:hi], m[lo:hi]
        active = mv > 0
        den = active.sum(dim=(0, 2)).clamp(min=1.0)          # [B]
        mean_s = (sv * mv).sum(dim=(0, 2)) / den
        mean_a = (av * mv).sum(dim=(0, 2)) / den
        dec = (((av < 0) & active).sum(dim=(0, 2)).to(sp.dtype)) / den
        big = torch.finfo(sp.dtype).max
        min_s = torch.where(active, sv, torch.full_like(sv, big)).amin(
            dim=(0, 2))
        has = active.any(dim=0).any(dim=-1)                  # [B]
        min_s = torch.where(has, min_s, torch.zeros_like(min_s))
        count = active.sum(dim=(0, 2)) / float(hi - lo)      # mean CAV/tick
        col = b * LONG_PLAN_FEATS_PER_BIN
        out[:, col + 0] = mean_s
        out[:, col + 1] = min_s
        out[:, col + 2] = mean_a
        out[:, col + 3] = dec
        out[:, col + 4] = count
    return out


def sha256_file(path) -> str:
    """SHA-256 of a file's bytes (the climatology pin recorded in receipts)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_climatology(path, *, contrast: str = DEFAULT_CONTRAST,
                     taus=C.TAUS, levels=C.QUANTILE_LEVELS,
                     expected_sha256: str | None = None):
    """Load a TRAIN climatology document into a ``[N_TAU, NQ]`` anchor.

    Returns ``(anchor, sha256, contrast)``.  Fails closed on a bad schema, a
    missing ``(contrast, tau)`` cell, a wrong quantile width, a non-finite
    quantile or (when supplied) a SHA-256 mismatch.  ``TRAIN_CLIMATOLOGY.json``
    is TRAIN-derived; no CAL/HOLDOUT byte is read here.
    """
    p = Path(path)
    raw = p.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None:
        pin = str(expected_sha256).strip().lower()
        if sha != pin:
            raise ValueError(
                f"climatology {p} sha256 {sha} != pinned {pin}; refusing")
    doc = json.loads(raw.decode("utf-8"))
    if doc.get("schema") != CLIMATOLOGY_SCHEMA:
        raise ValueError(
            f"climatology schema {doc.get('schema')!r} != "
            f"{CLIMATOLOGY_SCHEMA!r}")
    lv = np.asarray(levels, dtype=float)
    nq = int(lv.size)
    cells = (doc.get("cells") or {}).get(str(contrast))
    if not isinstance(cells, dict):
        raise ValueError(
            f"climatology has no contrast cell {contrast!r}; available "
            f"{sorted((doc.get('cells') or {}).keys())}")
    anchor = np.zeros((len(taus), nq), dtype=np.float64)
    for i, tau in enumerate(taus):
        cell = cells.get(f"{float(tau):g}") or cells.get(str(tau))
        if cell is None:
            raise ValueError(
                f"climatology cell {contrast!r}@{tau:g}s is missing")
        q = np.asarray(cell.get("quantiles", []), dtype=np.float64)
        if q.shape != (nq,):
            raise ValueError(
                f"climatology cell {contrast!r}@{tau:g}s has {q.shape} "
                f"quantiles, expected ({nq},)")
        if not np.isfinite(q).all():
            raise ValueError(
                f"climatology cell {contrast!r}@{tau:g}s is non-finite")
        anchor[i] = np.sort(q)
    return anchor, sha, str(contrast)


def order_statistics(anchor) -> torch.Tensor:
    """Coerce an anchor to the non-decreasing ``[N_TAU, NQ]`` form."""
    a = torch.as_tensor(np.asarray(anchor, dtype=np.float64),
                        dtype=torch.float32)
    return torch.sort(a, dim=-1).values


class PairedEffectHead(nn.Module):
    """Climatology-anchored paired-dV quantile head with a zero-init gate."""

    def __init__(self, d_model: int, *, n_tau: int = C.N_TAU,
                 n_quantiles: int = C.N_QUANTILES,
                 hidden: int | None = None, plan_aware: bool = False,
                 plan_dim: int | None = None,
                 plan_raw_dim: int = PLAN_RAW_F) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_tau = int(n_tau)
        self.n_quantiles = int(n_quantiles)
        #: A9 plan-aware mode: residual trunk reads concat(token, plan_vector).
        self.plan_aware = bool(plan_aware)
        self.plan_raw_f = int(plan_raw_dim)
        self.plan_dim = int(plan_dim) if self.plan_aware and plan_dim \
            else (self.d_model if self.plan_aware else 0)
        h = int(hidden or self.d_model)
        trunk_in = self.d_model + (self.plan_dim if self.plan_aware else 0)
        self.trunk = nn.Sequential(nn.Linear(trunk_in, h), nn.GELU())
        self.residual = nn.Linear(h, self.n_tau * self.n_quantiles)
        #: Treatment-arm CAV-motion summary (``summary_raw``) embedding.  Only
        #: present in plan-aware mode; a small linear, no arm/strategy input.
        self.plan_proj = (nn.Linear(self.plan_raw_f, self.plan_dim)
                          if self.plan_aware else None)
        #: Shrinkage gate.  ``zeros`` => the untrained head is EXACTLY the
        #: climatology anchor; training learns how far to move from it.
        self.gate = nn.Parameter(torch.zeros(1))
        self.register_buffer(
            "climatology", torch.zeros(self.n_tau, self.n_quantiles),
            persistent=True)
        self.register_buffer(
            "levels", torch.tensor(C.QUANTILE_LEVELS, dtype=torch.float32),
            persistent=False)

    # ------------------------------------------------------------------ anchor
    @torch.no_grad()
    def set_climatology(self, anchor) -> None:
        """Install the ``[N_TAU, NQ]`` (or flat) climatology anchor."""
        a = order_statistics(anchor)
        if a.shape != self.climatology.shape:
            a = a.reshape(self.climatology.shape)
        self.climatology.copy_(a.to(self.climatology.dtype))

    def climatology_quantiles(self) -> torch.Tensor:
        return self.climatology

    def loss_scale(self, *, floor_s: float = LOSS_SCALE_FLOOR_S,
                   levels=C.QUANTILE_LEVELS) -> torch.Tensor:
        """Per-tau TRAIN scale ``s_tau`` = q90 - q10 of the climatology anchor.

        Used only to normalise the pinball loss (predictions stay in seconds),
        so the paired-effect term is O(1) like the other loss terms instead of
        hundreds of seconds.  Floored at ``floor_s``.
        """
        lv = [round(float(x), 4) for x in levels]
        lo, hi = lv.index(0.1), lv.index(0.9)
        s = self.climatology[:, hi] - self.climatology[:, lo]
        return torch.clamp(s, min=float(floor_s))

    # ------------------------------------------------------------------ forward
    def plan_vector(self, summary_raw: torch.Tensor) -> torch.Tensor | None:
        """Embed a treatment-arm ``summary_raw`` ``[..., PLAN_RAW_F]``.

        Returns ``None`` for a non-plan-aware head (the plan is not an input).
        Casts the summary to the ``plan_proj`` parameter dtype/device so a
        float64 loaded checkpoint consumes float32 summaries safely.
        """
        if not self.plan_aware:
            return None
        w = self.plan_proj.weight
        s = torch.as_tensor(summary_raw).to(dtype=w.dtype, device=w.device)
        return self.plan_proj(s)

    def forward(self, token: torch.Tensor,
                plan: torch.Tensor | None = None) -> torch.Tensor:
        """State token ``[B, d]`` -> paired-dV quantiles ``[B, N_TAU, NQ]``.

        A plan-aware head also consumes a treatment-arm CAV-motion summary
        ``plan`` ``[B, PLAN_RAW_F]``: the residual trunk reads
        ``concat(token, plan_proj(plan))``.  With the gate still at its ``0.0``
        initial value the output equals :attr:`climatology` broadcast over the
        batch, up to floating-point exactness (``0 * finite == 0``), so an
        untrained head returns the climatology exactly for every plan.
        """
        if self.plan_aware:
            if plan is None:
                raise ValueError(
                    "plan-aware PairedEffectHead requires a plan vector "
                    "(the treatment-arm summary_raw); got None")
            w = self.plan_proj.weight
            token = torch.as_tensor(token).to(dtype=w.dtype, device=w.device)
            x = torch.cat([token, self.plan_vector(plan)], dim=-1)
        else:
            x = token
        h = self.trunk(x)
        r = self.residual(h).view(
            *token.shape[:-1], self.n_tau, self.n_quantiles)
        return self.climatology + self.gate * r

    def dv_quantiles(self, token: torch.Tensor, plan: torch.Tensor | None = None,
                     *, monotone: bool = True) -> torch.Tensor:
        q = self.forward(token, plan=plan)
        if monotone:
            q = torch.sort(q, dim=-1).values
        return q


def paired_effect_pinball(dv_pred: torch.Tensor, dy: torch.Tensor, *,
                          mask: torch.Tensor | None = None,
                          levels=C.QUANTILE_LEVELS,
                          weight: float = 1.0,
                          scale: torch.Tensor | None = None) -> torch.Tensor:
    """Pinball loss of predicted dV quantiles against the realised dV.

    ``dv_pred`` ``[..., NQ]``; ``dy`` broadcasts to ``[..., 1]``; ``mask``
    ``[...]`` marks the outcome-supported cells.  Only masked cells contribute
    (an empty mask scores exactly ``0.0``); the mask is never faked in.
    ``scale`` (optional ``[N_TAU]``, matching the axis before NQ) divides the
    error per tau: a scale-free pinball (loss / s_tau).
    """
    lv = torch.as_tensor(levels, dtype=dv_pred.dtype, device=dv_pred.device)
    e = dy.unsqueeze(-1) - dv_pred
    if scale is not None:
        e = e / scale.to(e.dtype).unsqueeze(-1)
    loss = torch.maximum(lv * e, (lv - 1.0) * e)
    if mask is None:
        return float(weight) * loss.mean()
    m = mask.to(loss.dtype).unsqueeze(-1)
    denom = (m.sum() * int(dv_pred.shape[-1])).clamp(min=1.0)
    return float(weight) * (loss * m).sum() / denom


def paired_effect_rank_loss(score: torch.Tensor, delta: torch.Tensor,
                            valid: torch.Tensor, *,
                            scale: torch.Tensor | None = None,
                            weight: float = 1.0) -> torch.Tensor:
    """A11 within-decision pairwise ranking loss (finite, O(1)).

    ``score`` is the head's mean-dV prediction DIFFERENCE ``pred_i - pred_j``
    (``[..., N_TAU]``) for a pair of treatment arms of the SAME decision; the
    realised delay contrast difference is ``delta = dY_i - dY_j`` in seconds
    (``[..., N_TAU]``, lower delay = better).  The logistic penalty

        softplus(-sign(delta) * score / s_tau)

    is small exactly when the predicted order matches the realised order
    (``score < 0`` when ``delta < 0``, i.e. arm ``i`` predicted below ``j``)
    and grows when it does not.  ``valid`` marks the cells that may contribute;
    ties (``delta == 0``) are ignored, and an all-invalid/all-tie input scores
    exactly ``0.0`` (never a fake value).  ``scale`` (``[N_TAU]``) is the
    per-tau ``s_tau`` from :meth:`PairedEffectHead.loss_scale`; dividing by it
    keeps the term O(1) across the 30 s..570 s taus.
    """
    sc = torch.as_tensor(score, dtype=torch.float32)
    dl = torch.as_tensor(delta, dtype=sc.dtype)
    va = torch.as_tensor(valid, dtype=torch.bool)
    if sc.shape != dl.shape or sc.shape != va.shape:
        raise ValueError(
            f"score/delta/valid must share a shape; got {tuple(sc.shape)}, "
            f"{tuple(dl.shape)}, {tuple(va.shape)}")
    if scale is not None:
        sc = sc / scale.to(sc.dtype)
    e = -torch.sign(dl) * sc
    loss = torch.nn.functional.softplus(e)
    m = va & (dl != 0)
    denom = m.sum().to(loss.dtype).clamp(min=1.0)
    return float(weight) * (loss * m).sum() / denom


def untrained_head_returns_climatology(head: PairedEffectHead,
                                       anchor, token: torch.Tensor,
                                       *, plan: torch.Tensor | None = None,
                                       atol: float = 0.0) -> bool:
    """Test helper: assert the zero-gate head reproduces the anchor exactly."""
    head.eval()
    with torch.no_grad():
        q = head(token, plan=plan)
    ref = order_statistics(anchor).to(q.dtype)
    return bool(torch.allclose(q, ref.expand_as(q), atol=atol, rtol=0.0))


__all__ = [
    "CLIMATOLOGY_SCHEMA", "DEFAULT_CONTRAST", "PLAN_RAW_F", "LONG_PLAN_BIN_S",
    "LONG_PLAN_FEATS_PER_BIN", "LONG_PLAN_FIELDS", "sha256_file",
    "load_climatology", "order_statistics", "PairedEffectHead",
    "paired_effect_pinball", "paired_effect_rank_loss", "long_plan_n_bins",
    "long_plan_raw_f", "long_plan_summary",
    "untrained_head_returns_climatology",
]
