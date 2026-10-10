"""Read-only evaluation surfaces for the optional paired-effect head.

Exposes the climatology-anchored paired-effect head's per-tau ``dV`` quantiles
(WM-N1 Idea A, owner-approved 2026-10-04) so the CAL-tune scorer can consume a
``paired_head`` dV source directly, alongside the historical per-arm marginal
sources (``comonotone`` / ``independent``).  Defaults unchanged: a bundle with
no paired-effect head raises, and nothing else in ``wm_api`` changes.

The head reads the SAME A6 state global token the correction head reads
(:func:`corr.evaluation.state_token`); no label, arm name or strategy one-hot
enters.
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "paired_effect_quantiles_from_token", "paired_effect_quantiles",
    "paired_effect_quantiles_for_plan", "has_paired_effect_head",
]


def has_paired_effect_head(bundle) -> bool:
    """True when the bundle carries a trained paired-effect head module."""
    return isinstance(getattr(bundle, "paired_effect", None), torch.nn.Module)


def _head_of(bundle):
    head = getattr(bundle, "paired_effect", None)
    if not isinstance(head, torch.nn.Module):
        raise ValueError(
            "the bundle carries no paired-effect head; enable "
            "--paired-effect-head / attach model.paired_effect.PairedEffectHead")
    return head


@torch.no_grad()
def paired_effect_quantiles_from_token(bundle, token, *, monotone: bool = True
                                       ) -> np.ndarray:
    """Head dV quantiles for a state token ``[B, d]``.

    Returns a float64 ``[B, N_TAU, N_QUANTILES]`` array (quantiles non-decreasing
    when ``monotone``).  An untrained head (gate at zero) returns the TRAIN
    climatology anchor exactly.
    """
    head = _head_of(bundle)
    t = token if isinstance(token, torch.Tensor) else torch.as_tensor(token)
    q = head.dv_quantiles(t, monotone=monotone)
    return q.detach().cpu().to(torch.float64).numpy()


@torch.no_grad()
def paired_effect_quantiles(bundle, *, t0_kin, t0_mask, t0_counter,
                            t0_reference=None, d_model: int | None = None,
                            monotone: bool = True) -> np.ndarray:
    """Build the A6 state token from the t0 state, then read the head.

    Mirrors :func:`wm_api.engine.outcome_head`'s token construction so the
    paired-effect head sees exactly the training-time state input.  Returns the
    float64 ``[B, N_TAU, N_QUANTILES]`` dV-quantile array.
    """
    from corr.evaluation import state_token

    head = _head_of(bundle)
    d = int(d_model if d_model is not None else getattr(head, "d_model", 0))
    if d <= 0:
        raise ValueError("the paired-effect head exposes no d_model")
    kin = torch.as_tensor(np.asarray(t0_kin))
    batch = int(kin.shape[0])
    if t0_reference is None:
        t0_reference = torch.zeros(batch, d, dtype=kin.dtype)
    token = state_token(bundle, t0_reference, kin=t0_kin, mask=t0_mask,
                        counter=t0_counter, d_model=d)
    return paired_effect_quantiles_from_token(bundle, token,
                                              monotone=monotone)


def _as_batch1(value, *, ndim: int) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(value))
    while t.dim() < int(ndim):
        t = t.unsqueeze(0)
    return t


@torch.no_grad()
def paired_effect_quantiles_for_plan(bundle, *, t0_kin, t0_mask, t0_counter,
                                     summary_raw, plan_long=None,
                                     monotone: bool = True) -> np.ndarray:
    """Plan-aware head dV quantiles for ONE t0 state + one plan.

    FIXED INTERFACE (a parallel scorer depends on it).  Builds the A6 state
    token with the bundle's state encoder exactly as training's ``global_token``
    (``state_enc(kin, mask, counter)``), embeds the treatment arm's plan with
    the head's plan projection exactly as training does, and returns the float64
    ``[N_TAU, N_QUANTILES]`` paired-dV quantiles for that single
    ``(state, plan)`` pair.  Requires a plan-aware paired-effect head
    (``PairedEffectHead(plan_aware=True)``).

    A11 full-horizon plan (default OFF): the head's ``plan_raw_f`` selects the
    plan.  A head with the historical width consumes ``summary_raw`` (the 10 s
    / 20-step CAV-motion summary); a head built with
    ``--paired-effect-plan-horizon-s H`` (``plan_raw_f = 5 * ceil(H/10)``)
    requires the optional ``plan_long`` ``[long_plan_raw_f(H)]`` vector, built
    by the scorer with :func:`model.paired_effect.long_plan_summary` (the SAME
    function training uses).  Supplying ``plan_long`` for a short head is
    refused; omitting it for a long head is refused.

    ``t0_kin`` ``[K,7]`` / ``[1,K,7]``, ``t0_mask`` ``[K]`` / ``[1,K]``,
    ``t0_counter`` ``[2,30]`` / ``[1,2,30]``, ``summary_raw``
    ``[PLAN_RAW_F]`` / ``[1,PLAN_RAW_F]``, ``plan_long`` ``[W]`` / ``[1,W]``.
    """
    from corr.evaluation import state_token
    from model.paired_effect import PLAN_RAW_F

    head = _head_of(bundle)
    if not getattr(head, "plan_aware", False):
        raise ValueError(
            "paired_effect_quantiles_for_plan requires a plan-aware "
            "paired-effect head (enable --paired-effect-plan-aware)")
    d = int(getattr(head, "d_model", 0))
    if d <= 0:
        raise ValueError("the paired-effect head exposes no d_model")
    kin = _as_batch1(t0_kin, ndim=3)
    mask = _as_batch1(t0_mask, ndim=2)
    counter = _as_batch1(t0_counter, ndim=3)
    if int(kin.shape[0]) != 1:
        raise ValueError(
            "paired_effect_quantiles_for_plan scores exactly one t0 state; "
            f"got batch {int(kin.shape[0])}")
    expected = int(getattr(head, "plan_raw_f", 0))
    if plan_long is not None:
        if expected == int(PLAN_RAW_F):
            raise ValueError(
                "this paired-effect head uses the historical 10 s "
                "summary_raw plan; plan_long is only valid for a head built "
                "with --paired-effect-plan-horizon-s")
        plan = _as_batch1(plan_long, ndim=2)
    elif expected == int(PLAN_RAW_F):
        plan = _as_batch1(summary_raw, ndim=2)
    else:
        raise ValueError(
            f"this paired-effect head consumes a {expected}-D full-horizon "
            "plan; pass plan_long (built with model.paired_effect."
            "long_plan_summary)")
    if int(plan.shape[0]) != 1:
        raise ValueError(
            "paired_effect_quantiles_for_plan scores exactly one plan; "
            f"got batch {int(plan.shape[0])}")
    if int(plan.shape[-1]) != expected:
        raise ValueError(
            f"plan must have {expected} features; got "
            f"{int(plan.shape[-1])}")
    ref_param = next((p for p in head.parameters() if p.is_floating_point()),
                     None)
    ref = torch.zeros(1, d,
                      dtype=(ref_param.dtype if ref_param is not None
                             else kin.dtype),
                      device=(ref_param.device if ref_param is not None
                              else kin.device))
    token = state_token(bundle, ref, kin=kin, mask=mask, counter=counter,
                        d_model=d)
    q = head.dv_quantiles(token, plan=plan, monotone=monotone)
    return q.detach().cpu().to(torch.float64).numpy()[0]
