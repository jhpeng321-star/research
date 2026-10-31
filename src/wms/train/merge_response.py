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

from wms.model.entries import (ENTRY_S_M, NOMINAL_LENGTH_M, SUMO_POSITION_EPS_M)

#: columns of the predicted/recorded component stack (data.COMPONENTS order)
J, UPSTREAM, SOURCE, DISCHARGE, ROAD = range(5)
#: differentiable columns whose paired contrasts carry gradients
GRADIENT_COLUMNS = (UPSTREAM, DISCHARGE, ROAD)
#: exact columns reported without gradients (J = road + source)
REPORT_COLUMNS = (J, SOURCE)
#: anchor columns (levels, differentiable), scaled by TRAIN absolute RMS
ANCHOR_COLUMNS = (DISCHARGE, ROAD)
#: S-WMFIX-C fallback 1 (default OFF): TRAINING-ONLY gradient columns of the
#: smoothed-event objective.  The exact integer source ledger is replaced, in
#: the differentiable stack only, by the smoothed admission surrogate
#: (:func:`smooth_admission_source`), so the paired source contrast carries a
#: gradient through the hard insertion boundary; the crossing column keeps the
#: existing identity-bracketed :func:`smooth_crossings` surrogate.  Inference,
#: verdicts and every default path keep the exact registered columns above.
SMOOTH_GRADIENT_COLUMNS = (UPSTREAM, DISCHARGE, ROAD, SOURCE)
#: Default numerical smoothing width (m) of the admission gate surrogate.  This
#: is NOT a behavioural threshold: it only sets the gradient relaxation scale
#: of the sigmoid that replaces the hard insertion test inside TRAINING
#: objectives (the order of the inferred min-gap scale it smooths).
ADMISSION_WIDTH_M = 1.0


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


def entrance_gap_series(pre_pos, pre_speeds, pre_present, pre_lanes, pre_length,
                        entry_s_m=ENTRY_S_M, dt_s=0.5,
                        length_new_m=NOMINAL_LENGTH_M,
                        position_eps_m=SUMO_POSITION_EPS_M):
    """[B, T, 2] differentiable end-of-step entrance bumper gaps (mainline best, ramp).

    TRAINING-ONLY surrogate input of S-WMFIX-C fallback 1.  ``pre_*`` are the
    pre-insertion rollout series indexed ``t`` = the state the hard admission at
    step ``t`` actually tested (t0 snapshot, then each step's post-move state);
    ``T = pre_pos.shape[1]`` steps are evaluated.  The geometry mirrors
    ``LearnedAdmission.place`` exactly: bumper gap from the entrant's front
    (lane start + nominal length + eps) to the rearmost vehicle's rear bumper,
    credited with the leader's next-step movement ``max(0, v_lead) * dt``.
    Membership (presence/lane) uses the exact hard masks; only the gap itself
    stays differentiable through the recorded positions/speeds, exactly like
    the crossing surrogate.  Mainline takes the larger of lanes 0/1 (the
    incumbent "any" lane order); an empty lane is an infinite gap.
    """
    lanes = ((0, 1), (2,))
    out = []
    for group in lanes:
        per_lane = []
        for lane in group:
            member = pre_present.bool() & (pre_lanes.long() == int(lane))
            big = torch.full_like(pre_pos, float("inf"))
            s_min, index = torch.where(member, pre_pos, big).min(-1)
            rear = s_min - pre_length.gather(-1, index.unsqueeze(-1)).squeeze(-1)
            v_lead = pre_speeds.gather(-1, index.unsqueeze(-1)).squeeze(-1).clamp_min(0)
            front = float(entry_s_m[lane]) + float(length_new_m) + float(position_eps_m)
            per_lane.append(rear - front + v_lead * float(dt_s))
        best = per_lane[0] if len(per_lane) == 1 else torch.maximum(per_lane[0], per_lane[1])
        out.append(best)
    return torch.stack(out, -1)


def smooth_admission_source(queue0, due, gap_end, min_gap, width=ADMISSION_WIDTH_M,
                            dt_s=0.5):
    """TRAINING-ONLY differentiable surrogate of the exact due-inserted ledger.

    Mean-field relaxation of the hard admission event chain (WMFIX_FEAS_BLOCKED
    option (c)): the per-step Heaviside insertion test on the end-of-step
    bumper gap becomes a sigmoid gate ``p_t = sigmoid((gap_end_t - min_gap_t) /
    width)`` and each entrance queue drains through it,
    ``Q_{t+1} = (Q_t + due_t) * (1 - p_t)``, seeded with the recorded Q0.
    ``gap_end`` (from :func:`entrance_gap_series`) is the only differentiable
    input; demand, Q0 and the hard-path tested ``min_gap`` thresholds are
    detached constants, so the gradient path is exactly the smoothed insertion
    boundary of the events the exact ledger counted.  Returns
    ``(source [R], queue [R, T, 2])`` with the source integral in the scorer's
    0.5 s left-boundary rectangle convention.  The exact integer
    ``SourceLedger`` remains the only reported/verdict value; enabling this
    surrogate is a training-side flag, never set at inference.
    """
    if float(width) <= 0.0:
        raise ValueError("admission smoothing width must be positive")
    if queue0.shape != due.shape[:1] + (due.shape[2],) or due.shape[1] < 1:
        raise ValueError("queue0 [R,2] must match due [R,T,2]")
    if gap_end.shape != due.shape or min_gap.shape != due.shape:
        raise ValueError("gap_end/min_gap [R,T,2] must match due")
    gate = torch.sigmoid((gap_end - min_gap) / float(width))
    queue = []
    q = queue0.to(due.dtype)
    for t in range(due.shape[1]):
        queue.append(q)
        q = (q + due[:, t]) * (1.0 - gate[:, t])
    series = torch.stack(queue, 1)
    return float(dt_s) * series.sum((1, 2)), series
