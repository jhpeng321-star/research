"""WM-N1 losses (design section 3).

* step NLL: Gaussian-mixture acceleration + categorical lane change;
* arrival loss: negative-binomial count NLL + insertion-speed NLL;
* outcome pinball / CRPS over the 19 quantiles;
* collision BCE;
* paired CONTRAST loss on sibling arms (same world + t0), the pinball loss of
  the comonotone quantile difference ``V(P_a) - V(P_b)`` against the realised
  ``dY``, with the design weight ``lambda_c = 1`` (WM_N1_DESIGN.md section 3,
  line 115).

Sign convention: for ``dY = y_a - y_b`` and ``dq = q_a - q_b`` the pinball
loss is ``max(tau*(dY-dq), (tau-1)*(dY-dq))``, so a prediction whose sign
agrees with ``dY`` scores lower (used by the sign test).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from . import constants as C

LOG2PI = math.log(2.0 * math.pi)


def _masked_mean(x, mask):
    m = mask.to(x.dtype)
    return (x * m).sum() / m.sum().clamp(min=1.0)


# --------------------------------------------------------------- step NLL ---
def gmm_accel_nll(target, mu, log_sigma, mix_logits, mask):
    """NLL of ``target`` under a per-slot Gaussian mixture.

    Shapes: ``target`` [B,K]; ``mu``/``log_sigma``/``mix_logits`` [B,K,M].
    """
    var = torch.exp(2.0 * log_sigma) + 1e-6
    log_comp = (-0.5 * ((target.unsqueeze(-1) - mu) ** 2) / var
                - log_sigma - 0.5 * LOG2PI)
    log_pi = F.log_softmax(mix_logits, dim=-1)
    log_prob = torch.logsumexp(log_pi + log_comp, dim=-1)
    return _masked_mean(-log_prob, mask)


def lane_change_nll(lane_target, lane_logits, mask):
    """Cross-entropy over slots where a valid next-lane class exists."""
    valid = mask & (lane_target >= 0) & (lane_target < C.N_LANE_CLASSES)
    if valid.sum() == 0:
        return lane_logits.sum() * 0.0
    return F.cross_entropy(lane_logits[valid],
                           lane_target[valid].long())


# --------------------------------------------------------------- arrivals ---
def neg_binom_nll(count, mu, log_alpha):
    """Negative-binomial NLL with mean ``mu`` and gamma dispersion."""
    r = F.softplus(log_alpha) + 1e-4
    p = (r / (r + mu.abs() + 1e-6)).clamp(1e-6, 1.0 - 1e-6)
    log_pmf = (torch.lgamma(count + r) - torch.lgamma(r)
               - torch.lgamma(count + 1.0)
               + r * torch.log(p) + count * torch.log1p(-p))
    return (-log_pmf).mean()


def insertion_speed_nll(target_speed, mu, log_sigma, mask):
    var = torch.exp(2.0 * log_sigma) + 1e-6
    log_prob = (-0.5 * ((target_speed - mu) ** 2) / var - log_sigma
                - 0.5 * LOG2PI)
    return _masked_mean(-log_prob, mask)


def arrival_loss(arrivals, predic, speed_target=None, speed_mask=None):
    """Negative-binomial count loss (+ optional insertion-speed NLL)."""
    count = arrivals.float()
    loss = neg_binom_nll(count.unsqueeze(0) if count.dim() == 2 else count,
                         predic["count_mu"], predic["count_log_alpha"])
    if speed_target is not None:
        loss = loss + insertion_speed_nll(
            speed_target, predic["speed_mu"], predic["speed_log_sigma"],
            speed_mask)
    return loss


# --------------------------------------------------------------- outcomes ---
def pinball(pred_q, target, levels):
    """Pinball loss; ``pred_q`` [..., NQ], ``target`` broadcasts to [..., 1]."""
    e = target.unsqueeze(-1) - pred_q
    lv = levels.to(pred_q.dtype).view(*([1] * (pred_q.dim() - 1)), -1)
    return torch.maximum(lv * e, (lv - 1.0) * e)


def outcome_pinball(quantiles, targets, mask=None):
    """Mean pinball over tau/target/quantile; shapes [B,TAU,TGT,NQ]/[B,TAU,TGT]."""
    lv = torch.tensor(C.QUANTILE_LEVELS, dtype=quantiles.dtype,
                      device=quantiles.device)
    loss = pinball(quantiles, targets, lv)
    if mask is None:
        return loss.mean()
    m = mask.to(loss.dtype).unsqueeze(-1).unsqueeze(-1)
    return (loss * m).sum() / (m.sum() * C.N_QUANTILES).clamp(min=1.0)


def outcome_crps(quantiles, targets, mask=None):
    """CRPS = 2 * mean pinball (per-target average over the 19 levels)."""
    return 2.0 * outcome_pinball(quantiles, targets, mask)


def collision_bce(logits, target):
    """BCE over collision logits; ``logits`` [B,TAU], ``target`` [B,TAU]."""
    return F.binary_cross_entropy_with_logits(logits, target.float())


# --------------------------------------------------------------- contrast ---
def contrast_loss(q_a, q_b, y_a, y_b, mask=None, weight=C.CONTRAST_WEIGHT):
    """Paired CONTRAST loss on sibling arms (design section 3, line 115).

    ``q_a``/``q_b`` [..., NQ] are the marginal outcome quantiles of the two
    arms; ``y_a``/``y_b`` the realised outcomes.  Comonotone pairing uses the
    same quantile level, so the predicted contrast is ``dq = q_a - q_b`` and
    the target is ``dy = y_a - y_b``::

        loss = pinball(q_a - q_b, y_a - y_b) .

    With ``lambda_c = 1``.  Identical arm predictions give ``dq = 0``; when
    the two arms also realised the same outcome (``dy = 0``) the loss is 0.
    """
    lv = torch.tensor(C.QUANTILE_LEVELS, dtype=q_a.dtype, device=q_a.device)
    dq = q_a - q_b
    dy = y_a - y_b
    loss = pinball(dq, dy, lv)
    if mask is None:
        return weight * loss.mean()
    m = mask.to(loss.dtype).unsqueeze(-1).unsqueeze(-1)
    return weight * (loss * m).sum() / (m.sum() * C.N_QUANTILES).clamp(min=1.0)


# ------------------------------------------------------------- aggregator ---
DEFAULT_WEIGHTS = dict(step=1.0, lane=1.0, arrival=1.0, outcome=1.0,
                       collision=1.0, contrast=C.CONTRAST_WEIGHT)


def total_loss(pred, targets, weights=None):
    """Aggregate the training losses.

    ``pred`` comes from :meth:`WmN1Net.forward`; ``targets`` is a dict with
    ``accel`` [B,K], ``target_mask`` [B,K], ``lane`` [B,K] int, ``arrivals``
    [B,E], and outcome keys.  The contrast term is only added when sibling
    predictions/targets are supplied.
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    step = pred["step"]
    loss = w["step"] * gmm_accel_nll(
        targets["accel"], step["accel_mu"], step["accel_log_sigma"],
        step["accel_mix_logits"], targets["target_mask"])
    loss = loss + w["lane"] * lane_change_nll(
        targets["lane"], step["lane_logits"], targets["target_mask"])
    loss = loss + w["arrival"] * arrival_loss(targets["arrivals"],
                                              pred["arrival"])
    if "outcome_quantiles" in targets:
        loss = loss + w["outcome"] * outcome_pinball(
            pred["outcome_quantiles"], targets["outcome_quantiles"],
            targets.get("outcome_mask"))
    if "collision" in targets:
        loss = loss + w["collision"] * collision_bce(
            pred["collision_logits"], targets["collision"])
    if "contrast" in targets:
        c = targets["contrast"]
        loss = loss + contrast_loss(pred["outcome_quantiles"][c["a"]],
                                    pred["outcome_quantiles"][c["b"]],
                                    targets["outcome_quantiles"][c["a"]],
                                    targets["outcome_quantiles"][c["b"]],
                                    mask=c.get("mask"),
                                    weight=w["contrast"])
    return loss
