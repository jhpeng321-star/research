"""P5 CAV-motion OOD "is this motion familiar?" score primitives.

Calibration (thresholds, CAL-only conformal quantiles) is owned by P8
(``evalh/idea_a.py``); this module provides the strategy-independent score
primitives computed from the P5 CAV-motion summary only.  Neither primitive
accepts a strategy name, action or CAV flag.

Two complementary scores (both larger = less familiar):

* :func:`motion_density_fit` / :func:`motion_density_score` -- a diagonal
  Gaussian fitted on familiar (TRAIN) summaries; the score is the mean squared
  standardized distance (Mahalanobis^2 / F).
* :func:`ensemble_disagreement` -- the spread of member summaries for the SAME
  motion.
"""
from __future__ import annotations

from typing import Sequence, Union

import torch
from torch import Tensor


def _as_2d(summaries) -> Tensor:
    s = torch.as_tensor(summaries, dtype=torch.float32)
    if s.dim() != 2:
        raise ValueError(f"summaries must be [B,F]; got {tuple(s.shape)}")
    return s


def motion_density_fit(summaries, *,
                       min_summary_std: float = 1e-6) -> dict:
    """Fit a diagonal-Gaussian familiarity model on ``[N,F]`` summaries."""
    s = _as_2d(summaries)
    mean = s.mean(dim=0)
    std = s.std(dim=0, unbiased=False).clamp(min=float(min_summary_std))
    return {"mean": mean, "std": std, "dim": int(s.shape[1]),
            "min_summary_std": float(min_summary_std)}


def motion_density_score(summaries, model: dict) -> Tensor:
    """``[B,F]`` + fitted model -> ``[B]``; larger = less familiar."""
    s = _as_2d(summaries)
    mean = torch.as_tensor(model["mean"], dtype=s.dtype, device=s.device)
    std = torch.as_tensor(model["std"], dtype=s.dtype, device=s.device)
    z = (s - mean) / std.clamp(min=float(model.get("min_summary_std", 1e-6)))
    return (z * z).mean(dim=-1)


def ensemble_disagreement(
        summaries: Union[Tensor, Sequence]) -> Tensor:
    """``[M,B,F]`` member summaries -> ``[B]`` spread; larger = less familiar."""
    if isinstance(summaries, (list, tuple)):
        s = torch.stack([torch.as_tensor(m, dtype=torch.float32)
                         for m in summaries], dim=0)
    else:
        s = torch.as_tensor(summaries, dtype=torch.float32)
    if s.dim() != 3:
        raise ValueError(
            f"summaries must be [M,B,F] or a sequence of [B,F]; got "
            f"{tuple(s.shape)}")
    return s.std(dim=0, unbiased=False).mean(dim=-1)
