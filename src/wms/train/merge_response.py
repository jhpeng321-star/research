"""WMFIX objective: full-fleet intervention contrasts, no t0-HDV proxy."""
from __future__ import annotations

import torch


def paired_objective(predicted, recorded, references, scales, *, anchor_columns=None):
    """Each column is a measured component; returns normalized arm-minus-ref MSE.

    Source costs may be exact nondifferentiable tensors; never invent their gradient.
    Absolute anchors prevent cancellation while paired terms are the primary signal.
    """
    if predicted.shape != recorded.shape or predicted.ndim != 2:
        raise ValueError("expected identical [rows, components] tensors")
    if scales.shape != predicted.shape[1:] or not bool((scales > 0).all()):
        raise ValueError("positive TRAIN component scales required")
    references = torch.as_tensor(references, dtype=torch.long, device=predicted.device)
    if references.shape != predicted.shape[:1] or bool((references < 0).any()) \
            or bool((references >= predicted.shape[0]).any()):
        raise ValueError("invalid pair reference map")
    rows = torch.arange(len(references), device=predicted.device)
    active = references != rows
    if not bool(active.any()):
        raise ValueError("no intervention contrasts")
    if not bool((references[references] == references).all()):
        raise ValueError("references must be native self-reference rows")
    error = ((predicted - predicted[references]) -
             (recorded - recorded[references])) / scales
    # Give each paired world equal mass even with differing numbers of arms.
    world = torch.stack([error[active & (references == ref)].square().mean()
                         for ref in references[active].unique()]).mean()
    absolute_error = (predicted - recorded) / scales
    if anchor_columns is not None:
        if not anchor_columns:
            raise ValueError("absolute anchor columns must be explicit and nonempty")
        absolute_error = absolute_error[:, list(anchor_columns)]
    anchor = absolute_error.square().mean()
    return world + anchor


def smooth_crossings(positions, present, newborn, width, boundary=75.44, lanes=None):
    """Differentiable identity-continuous crossing surrogate for fixed slot lifetimes.

    Actual verdict uses the independent identity-bracketed labels.crossings counter.
    New births/reused slots are excluded; disappearance cannot create a crossing.
    """
    if width <= 0:
        raise ValueError("crossing width must be measured on TRAIN and positive")
    alive = present[:, 1:] & present[:, :-1] & ~newborn[:, 1:]
    if lanes is not None:
        alive = alive & (lanes[:, 1:] >= 0) & (lanes[:, 1:] < 2)
    clean = torch.where(present, positions, torch.zeros_like(positions))
    cdf = torch.sigmoid((clean - boundary) / width)
    delta = (cdf[:, 1:] - cdf[:, :-1]).clamp_min(0)
    return torch.where(alive, delta, torch.zeros_like(delta)).sum((-1, -2))
