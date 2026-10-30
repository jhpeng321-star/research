"""WMFIX objective: full-fleet intervention contrasts, no t0-HDV proxy.

Gradient terms are the DIFFERENTIABLE paired columns only -- upstream road,
the differentiable crossing surrogate and total road (WMFIX_REVIEW items 7/8).
The paired J column contains the nondifferentiable source waiting and the
source column is exact-but-detached: both are computed and reported, never
differentiated, so road contrasts cannot be rewarded for cancelling source
error.  Absolute anchors are normalized by their own TRAIN absolute-value RMS
(``anchor_scales``, prepare.py), not by the paired-contrast RMS, so the paired
response term sets the loss scale; the measured step0 shares are part of every
receipt (``objective_terms``).
"""
from __future__ import annotations

import torch

#: columns of the predicted/recorded component stack (data.COMPONENTS order)
J, UPSTREAM, SOURCE, DISCHARGE, ROAD = range(5)
#: differentiable columns whose paired contrasts carry gradients
GRADIENT_COLUMNS = (UPSTREAM, DISCHARGE, ROAD)
#: exact columns reported without gradients (J = road + source)
REPORT_COLUMNS = (J, SOURCE)
#: anchor columns (levels, differentiable), scaled by TRAIN absolute RMS
ANCHOR_COLUMNS = (DISCHARGE, ROAD)


def _paired_error(predicted, recorded, references):
    return (predicted - predicted[references]) - (recorded - recorded[references])


def paired_objective(predicted, recorded, references, scales, *, anchor_scales=None,
                     gradient_columns=GRADIENT_COLUMNS, anchor_columns=ANCHOR_COLUMNS):
    """Loss = normalized paired MSE over ``gradient_columns`` + rescaled anchors."""
    terms = _terms(predicted, recorded, references, scales, anchor_scales,
                   gradient_columns, anchor_columns)
    return terms["paired_gradient"] + terms["anchor"]


def objective_terms(predicted, recorded, references, scales, *, anchor_scales=None,
                    gradient_columns=GRADIENT_COLUMNS, anchor_columns=ANCHOR_COLUMNS,
                    report_columns=REPORT_COLUMNS):
    """Detached per-term loss values, shares and reported paired MAEs."""
    terms = _terms(predicted, recorded, references, scales, anchor_scales,
                   gradient_columns, anchor_columns)
    out = {key: float(value.detach()) for key, value in terms.items()
           if torch.is_tensor(value)}
    paired_error = _paired_error(predicted, recorded, references)
    references = torch.as_tensor(references, dtype=torch.long, device=predicted.device)
    active = references != torch.arange(len(references), device=predicted.device)
    for column in report_columns:
        error = paired_error[:, column][active] / scales[column]
        out[f"paired_mae_col{column}"] = float(error.detach().abs().mean())
    total = out["paired_gradient"] + out["anchor"]
    out["paired_share"] = out["paired_gradient"] / total if total > 0 else None
    out["anchor_share"] = out["anchor"] / total if total > 0 else None
    return out


def _terms(predicted, recorded, references, scales, anchor_scales,
           gradient_columns, anchor_columns):
    if predicted.shape != recorded.shape or predicted.ndim != 2:
        raise ValueError("expected identical [rows, components] tensors")
    scales = torch.as_tensor(scales, dtype=predicted.dtype, device=predicted.device)
    if scales.shape != predicted.shape[1:] or not bool((scales > 0).all()):
        raise ValueError("positive TRAIN component scales required")
    if not gradient_columns:
        raise ValueError("gradient columns must be nonempty")
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
    error = _paired_error(predicted, recorded, references) / scales
    # Give each paired world equal mass even with differing numbers of arms.
    world = torch.stack([error[active & (references == ref)][:, list(gradient_columns)]
                         .square().mean() for ref in references[active].unique()]).mean()
    if anchor_columns:
        if anchor_scales is None:
            raise ValueError("anchors require TRAIN absolute-value scales")
        anchor_scales = torch.as_tensor(anchor_scales, dtype=predicted.dtype,
                                        device=predicted.device)
        if anchor_scales.shape != predicted.shape[1:] or not bool((anchor_scales > 0).all()):
            raise ValueError("positive TRAIN anchor scales required")
        absolute_error = (predicted - recorded) / anchor_scales
        anchor = absolute_error[:, list(anchor_columns)].square().mean()
    else:
        anchor = torch.zeros((), device=predicted.device)
    return dict(paired_gradient=world, anchor=anchor)


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
