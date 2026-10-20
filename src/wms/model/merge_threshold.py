"""Leaned gap-acceptance threshold for the ramp merge (``model.merge_decision='threshold'``).

Replaces the per-step Bernoulli merge draw (lane-v5 merge head, sampled every ``dt_s``) with
one persistent per-driver threshold ``u`` drawn at the car's first ramp appearance and a
first-passage decision on the merge head's logit ``z_t``:

* ``u = loc + eps``, ``eps ~ Logistic(0, 1)`` (unit residual, centred location);
* ``loc`` is a small learned function of the car's own driver encoding (zero-initialised
  location, bounded, L2-regularised by the loss);
* ``M_t = max_{s <= t, feasible_s} z_s`` over ENVELOPE-FEASIBLE in-zone steps only (infeasible
  steps never enter the maximum);
* the car merges at the first feasible step with ``z_t > u``.

The threshold / running maximum / previous presence are plain rollout-state entries
(``merge_threshold_u``, ``merge_fp_max``, ``merge_fp_present``) carried across truncated-BPTT
chunks and slot moves.  A slot that was absent and is present again is a new car: its
threshold and maximum are reset.  The state entries are detached; the deployment decision is
a sampled threshold, while :mod:`wms.train.losses.merge_first_passage` integrates ``u`` out
for training.

Everything here is OFF unless ``ModelSpec.merge_decision == "threshold"``: the default
``"hazard"`` path never creates the module and is bit-identical.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

#: Threshold location bound (keeps the learned location on the score's support).
LOCATION_BOUND = 8.0
#: Numerically stable lower bound of a first-passage probability mass.
LOG_TINY = math.log(1.0e-12)


def _logistic_sample(shape, dtype, device, uniform: Optional[Tensor] = None) -> Tensor:
    """Unit-logistic residual sample ``log(r / (1 - r))`` (finite, clamped)."""
    if uniform is None:
        uniform = torch.rand(shape, dtype=dtype, device=device)
    r = uniform.to(dtype).clamp(1.0e-6, 1.0 - 1.0e-6)
    return torch.log(r) - torch.log1p(-r)


def logistic_location(loc: Tensor, uniform: Optional[Tensor] = None) -> Tensor:
    """One-unit-logistic threshold ``u = loc + Logistic(0, 1)`` per slot."""
    return loc + _logistic_sample(loc.shape, loc.dtype, loc.device, uniform)


def log_survival(m: Tensor, loc: Tensor) -> Tensor:
    """``log P(T > t) = log(1 - F(M_t))`` with ``F`` the unit-logistic CDF at ``loc``."""
    return torch.nn.functional.logsigmoid(loc - m)


def log_event_mass(m_prev: Tensor, m_cur: Tensor, loc: Tensor) -> Tensor:
    """Stable ``log(F(M_cur) - F(M_prev))`` (``-inf`` previous maximum = ``F(M_cur)``).

    ``M_cur >= M_prev``; an event below an earlier feasible maximum has zero mass and is
    returned as :data:`LOG_TINY` (finite, no gradient), so an unsupported event is diagnosed
    rather than hidden behind an epsilon.  The log-CDF difference is evaluated in logit space:
    with ``a = M_prev - loc``, ``b = M_cur - loc``,

        log(F(b) - F(a)) = a + log(expm1(b - a)) - softplus(a) - softplus(b).
    """
    finite = torch.isfinite(m_prev)
    a = (m_prev - loc)
    b = (m_cur - loc)
    # The sentinel branch is value-selected below; clamping only keeps expm1 finite.
    a_safe = torch.where(finite, a, torch.full_like(a, -60.0)).clamp(-60.0, 40.0)
    b_safe = b.clamp(-60.0, 40.0)
    gap = (b_safe - a_safe).clamp_min(1.0e-12)
    low = a_safe + torch.log(torch.expm1(gap)) \
        - torch.nn.functional.softplus(a_safe) \
        - torch.nn.functional.softplus(b_safe)
    out = torch.where(finite, low, torch.nn.functional.logsigmoid(b_safe))
    out = torch.nan_to_num(out, nan=LOG_TINY, posinf=0.0, neginf=LOG_TINY)
    return out.clamp(LOG_TINY, 0.0)


def first_passage_probs(m_seq: Tensor, loc: Tensor) -> tuple[Tensor, Tensor]:
    """Per-step event masses and the final survival of a non-decreasing max sequence.

    ``m_seq`` is ``[..., T]``; returns ``p [..., T]`` with ``p_t = F(M_t) - F(M_{t-1})`` and
    ``survival = 1 - F(M_T)``.  ``sum(p) + survival == 1`` exactly (in exact arithmetic).
    """
    prev = torch.cat((torch.full_like(m_seq[..., :1], float("-inf")), m_seq[..., :-1]), -1)
    p = torch.sigmoid(m_seq - loc) - torch.sigmoid(prev - loc)
    return p, 1.0 - torch.sigmoid(m_seq[..., -1] - loc)


class MergeThresholdNet(nn.Module):
    """Zero-initialised, bounded threshold-location function of the driver encoding.

    ``input_dim == 0`` (no structured driver encoding) gives one learned scalar location;
    otherwise a two-layer MLP maps the per-car driver embedding to a location in
    ``(-LOCATION_BOUND, LOCATION_BOUND)``.  Zero-initialised output weights make the location
    exactly 0 at creation, so enabling the module does not move the merge decision until it
    is trained.
    """

    def __init__(self, input_dim: int = 0, hidden: int = 32,
                 location_bound: float = LOCATION_BOUND) -> None:
        super().__init__()
        if input_dim < 0:
            raise ValueError("merge threshold input_dim must be >= 0")
        if hidden < 1:
            raise ValueError("merge threshold hidden width must be >= 1")
        if not (math.isfinite(location_bound) and location_bound > 0.0):
            raise ValueError("merge threshold location_bound must be finite and positive")
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.location_bound = float(location_bound)
        self.location_bias = nn.Parameter(torch.zeros(()))
        if self.input_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden), nn.GELU(),
                nn.Linear(self.hidden, 1))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        else:
            self.net = None

    def forward(self, features: Optional[Tensor], shape: tuple[int, ...]) -> Tensor:
        """Location ``[B, K]`` (bounded, centred) for ``shape`` slots."""
        bias = self.location_bias.expand(shape)
        if self.net is None or features is None:
            return self.location_bound * torch.tanh(bias)
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"merge threshold needs {self.input_dim} driver features")
        raw = self.net(features.to(self.location_bias.dtype))[..., 0]
        return self.location_bound * torch.tanh(raw + self.location_bias)


def update_threshold_state(z: Tensor, ramp: Tensor, feasible: Tensor, present: Tensor,
                           location: Tensor, *, u: Optional[Tensor] = None,
                           has_u: Optional[Tensor] = None,
                           m: Optional[Tensor] = None,
                           prev_present: Optional[Tensor] = None,
                           uniform: Optional[Tensor] = None) -> dict[str, Tensor]:
    """One step of the persistent threshold / running-maximum decision.

    ``ramp``: present in the ramp lane; ``feasible``: envelope-feasible AND in the merge
    zone (infeasible / out-of-zone steps never enter the maximum); ``z``: merge-head logit
    (detached here; the likelihood re-evaluates it with gradient).  Returns ``cross`` (merge
    now) and the carried ``u`` / ``has_u`` / ``m`` / ``prev_present``.
    """
    if u is None:
        u = torch.zeros_like(z)
    if has_u is None:
        has_u = torch.zeros_like(present, dtype=torch.bool)
    if m is None:
        m = torch.full_like(z, float("-inf"))
    if prev_present is None:
        prev_present = torch.zeros_like(present, dtype=torch.bool)
    z = z.detach()
    present = present.bool()
    new_car = present & ~prev_present.bool()
    keep = present & ~new_car & has_u.bool()
    u = torch.where(keep, u, torch.zeros_like(u))
    has = keep
    need = present & ramp.bool() & ~has
    if bool(need.any()):
        u = torch.where(need, logistic_location(location.detach(), uniform), u)
        has = has | need
    m = torch.where(new_car | ~present, torch.full_like(m, float("-inf")), m)
    m = torch.where(feasible.bool(), torch.maximum(m, z), m)
    cross = feasible.bool() & has & (z > u)
    return {"u": u, "has_u": has, "m": m, "prev_present": present, "cross": cross}


def target_gap_configuration(kin: Tensor, present: Tensor, physics,
                             max_gap_m: float = 150.0) -> dict[str, Tensor]:
    """Nearest lane-1 follower / leader gap and relative speed of every ramp slot.

    Returns ``gap_f``, ``gap_l`` (bumper-to-bumper, 0 when absent), ``rel_f``/``rel_l``
    (ego minus other speed, 0 when absent) and the ``has_f``/``has_l`` flags.  Used by the
    on-policy first-passage loss as the detached target-gap compatibility check.
    """
    present = present.bool()
    kin = kin.float()
    s, v, length = kin[..., 0], kin[..., 4], kin[..., 6]
    onehot = kin[..., 1:4]
    lane = onehot.argmax(-1)
    lane = torch.where(onehot.sum(-1) > 0, lane, torch.full_like(lane, -1))
    ramp_lane = int(getattr(physics, "merge_lane", 2))
    target_lane = int(getattr(physics, "merge_target_lane", 1))
    ramp = present & (lane == ramp_lane)
    target = present & (lane == target_lane)
    ds = s.unsqueeze(-1) - s.unsqueeze(-2)
    inf = torch.full_like(ds, float("inf"))
    k = s.shape[-1]
    behind = target.unsqueeze(-2) & (ds >= 0.0) & (ds <= float(max_gap_m))
    gap_f, f_idx = torch.where(behind, ds, inf).min(-1)
    has_f = torch.isfinite(gap_f)
    ahead = target.unsqueeze(-2) & (ds <= 0.0) & (-ds <= float(max_gap_m))
    gap_l, l_idx = torch.where(ahead, -ds, inf).min(-1)
    has_l = torch.isfinite(gap_l)
    safe_f = f_idx.clamp(0, max(k - 1, 0))
    safe_l = l_idx.clamp(0, max(k - 1, 0))
    v_f = v.gather(-1, safe_f)
    v_l = v.gather(-1, safe_l)
    gap_f = torch.where(has_f, (gap_f - length.gather(-1, safe_f)).clamp_min(0.0),
                        torch.zeros_like(gap_f))
    gap_l = torch.where(has_l, (gap_l - length.gather(-1, safe_l)).clamp_min(0.0),
                        torch.zeros_like(gap_l))
    rel_f = torch.where(has_f, v - v_f, torch.zeros_like(v))
    rel_l = torch.where(has_l, v - v_l, torch.zeros_like(v))
    keep = ramp
    return {"gap_f": torch.where(keep, gap_f, torch.zeros_like(gap_f)),
            "gap_l": torch.where(keep, gap_l, torch.zeros_like(gap_l)),
            "rel_f": torch.where(keep, rel_f, torch.zeros_like(rel_f)),
            "rel_l": torch.where(keep, rel_l, torch.zeros_like(rel_l)),
            "has_f": ramp & has_f, "has_l": ramp & has_l, "ramp": ramp}


__all__ = ["LOCATION_BOUND", "LOG_TINY", "MergeThresholdNet", "logistic_location",
           "log_survival", "log_event_mass", "first_passage_probs", "update_threshold_state",
           "target_gap_configuration"]
