"""Causal two-sided, persistent-logistic first passage (round-9 P1--P3).

All slot tensors have shape ``[..., K]``; kinematics have shape ``[..., K, 7]``
with columns ``s, lane0, lane1, lane2, v, a, length``.  Positions are FRONT
positions: a vehicle occupies ``[s - length, s]``.  This module consumes only
the current scene/action, never a recorded next state or simulator LC flags.

``geometry()['ramp']`` is membership in the inclusive ramp *risk zone*, even
when overlap/invalid observations veto that tick.  Feed it to ``update_state``
to make zone/lane exits absorbing.  Carry is explicit (including the latent,
running maximum and lifecycle); callers may detach it at TBPTT boundaries.
There are no behavioral clips, clearance padding or likelihood floors here.
"""
from __future__ import annotations

import math
from typing import Mapping, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .merge_core import load_init


class FirstPassageNet(nn.Module):
    """Seven positive softplus parameters, with no residual or driver inputs.

    ``init`` optionally overrides ``merge_core.load_init()`` keys (``s0_m``,
    ``T_s``, ``b_mps2``, ``alpha_log_sigma``), or any of the seven new names.
    The default logistic scale variance-matches the old Gaussian log scale.
    """

    def __init__(self, init: Optional[Mapping[str, float]] = None) -> None:
        super().__init__()
        old = {**load_init(), **({} if init is None else init)}
        values = {
            "s_f": old["s0_m"], "T_f": old["T_s"], "b_f": old["b_mps2"],
            "s_l": old["s0_m"], "T_l": old["T_s"], "b_l": old["b_mps2"],
            "tau": old["alpha_log_sigma"] * math.sqrt(3.0) / math.pi,
        }
        for name, default in values.items():
            value = float(old.get(name, default))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must have a finite positive initialization")
            raw = value + math.log(-math.expm1(-value))
            self.register_parameter(f"{name}_raw", nn.Parameter(torch.tensor(raw)))

    def params(self) -> dict[str, Tensor]:
        """Effective ``s_f,T_f,b_f,s_l,T_l,b_l,tau`` (plain softplus)."""
        return {name: F.softplus(getattr(self, f"{name}_raw"))
                for name in ("s_f", "T_f", "b_f", "s_l", "T_l", "b_l", "tau")}

    def boundary(self, geometry: Mapping[str, Tensor]) -> Tensor:
        """Exact min log clearance ratio; absent=+inf, veto/zero gap=-inf.

        Mask operands BEFORE arithmetic: an absent/invalid observation must
        neither invent event support nor poison gradients in other slots.
        """
        p = self.params()
        v = geometry["v"]
        eligible = geometry["eligible"].bool() & torch.isfinite(v)
        sides = []
        for side in ("f", "l"):
            has = geometry[f"has_{side}"].bool()
            gap, other = geometry[f"gap_{side}"], geometry[f"v_{side}"]
            valid = (eligible & has & torch.isfinite(gap) & (gap > 0.0)
                     & torch.isfinite(other))
            ego = torch.where(valid, v, torch.zeros_like(v))
            neighbor = torch.where(valid, other, torch.zeros_like(other))
            speed, slower = (neighbor, ego) if side == "f" else (ego, neighbor)
            req = (p[f"s_{side}"] + p[f"T_{side}"] * speed
                   + F.relu(speed.square() - slower.square()) / (2.0 * p[f"b_{side}"]))
            valid = valid & torch.isfinite(req) & (req > 0.0)
            safe_gap = torch.where(valid, gap, torch.ones_like(gap))
            safe_req = torch.where(valid, req, torch.ones_like(req))
            margin = safe_gap.log() - safe_req.log()
            constrained = torch.where(valid, margin, torch.full_like(v, -torch.inf))
            sides.append(torch.where(has, constrained, torch.full_like(v, torch.inf)))
        c = torch.minimum(*sides)
        return torch.where(eligible, c, torch.full_like(c, -torch.inf))

    def forward(self, geometry: Mapping[str, Tensor]) -> Tensor:
        return self.boundary(geometry)


def geometry(kin: Tensor, present: Tensor, physics,
             decision_mask: Optional[Tensor] = None,
             fixed_target: Optional[Tensor] = None) -> dict[str, Tensor]:
    """Full-scene neighbors, exact overlap veto and latent-independent reservations.

    Touching bumpers are physically eligible but constrain the learned boundary
    to -inf.  A missing/nonfinite PRESENT observation invalidates the scene;
    absent slots may contain arbitrary padding.  No distance cutoff is used.
    ``ramp`` includes both endpoints of the configured merge zone, independent
    of physical vetoes, for lifecycle tracking.
    ``decision_mask`` excludes nonlearned candidates from opportunity allocation;
    ``fixed_target`` marks immutable target occupancy from current actions only.
    """
    present = present.bool()
    observed = (torch.isfinite(kin).all(-1) & (kin[..., 1:4].sum(-1) > 0.0)
                & (kin[..., 6] > 0.0))
    scene_valid = (~present | observed).all(-1, keepdim=True)
    safe = torch.where((present & observed).unsqueeze(-1), kin, torch.zeros_like(kin))
    s, v, length = safe[..., 0], safe[..., 4], safe[..., 6]
    # Risk membership uses route/position observations, independently of a
    # missing speed/acceleration.  A temporary observation veto is not an exit.
    route = kin[..., 1:4]
    route_valid = torch.isfinite(route).all(-1) & (route.sum(-1) > 0.0)
    lane = torch.nan_to_num(route).argmax(-1)
    ramp_lane = int(getattr(physics, "merge_lane", 2))
    target_lane = int(getattr(physics, "merge_target_lane", 1))
    start = float(getattr(physics, "merge_zone_s0_m", 0.0))
    end = float(physics.merge_zone_s1_m)
    ramp = (present & route_valid & (lane == ramp_lane) & torch.isfinite(kin[..., 0])
            & (kin[..., 0] >= start) & (kin[..., 0] <= end))
    target = present & observed & (lane == target_lane)
    ds = s.unsqueeze(-1) - s.unsqueeze(-2)  # ego front - other front
    k = s.shape[-1]
    not_self = ~torch.eye(k, dtype=torch.bool, device=kin.device)
    candidates = target.unsqueeze(-2) & not_self
    rear = s - length
    intersects = (torch.maximum(rear.unsqueeze(-1), rear.unsqueeze(-2))
                  < torch.minimum(s.unsqueeze(-1), s.unsqueeze(-2)))
    overlap = (candidates & intersects).any(-1)
    physical = ramp & scene_valid & ~overlap
    if decision_mask is not None:
        physical = physical & decision_mask.bool()
    # Current-action immutable CAV insertions are a physical veto, but never
    # change PRE-change target-neighbor features.
    incoming_overlap = torch.zeros_like(overlap)
    incoming = torch.zeros_like(intersects)
    if fixed_target is not None:
        incoming = intersects & fixed_target.bool().unsqueeze(-2) & not_self
        incoming_overlap = incoming.any(-1)
        physical = physical & ~incoming_overlap
    # Allocate simultaneous opportunities BEFORE willingness/latent draws. Exact
    # interval reservations are deterministic and independent of z, preserving a
    # scalar exact first-passage law conditional on the supplied scene. Arbitration
    # depending on competing z AFTER crossing would require a joint likelihood.
    reserved = torch.zeros_like(physical)
    # Only potential entrants can reserve an interval. Preserve slot priority
    # exactly while avoiding K tiny kernels for mostly-mainline recorded scenes.
    slots = torch.nonzero(physical.reshape(-1, k).any(0)).flatten().tolist() if k else []
    for i in slots:
        blocked = (intersects[..., i, :] & reserved).any(-1)
        reserved[..., i] = physical[..., i] & ~blocked
    blocker = (candidates & intersects)
    blocker_index = (blocker.long().argmax(-1) if k else
                     torch.zeros_like(s, dtype=torch.long))
    fixed_index = incoming.long().argmax(-1) if k else torch.zeros_like(blocker_index)
    out = {"v": v, "s": s, "length": length, "lane": lane, "pre_target": target,
           "scene_valid": scene_valid.expand_as(present),
           "overlap_index": torch.where(overlap, blocker_index, -torch.ones_like(blocker_index)),
           "fixed_overlap_index": torch.where(incoming_overlap, fixed_index,
                                                -torch.ones_like(fixed_index)),
           "fixed_overlap": incoming_overlap,
           "ramp": ramp, "overlap": overlap,
           "physical_eligible": physical, "reservation_veto": physical & ~reserved,
           "eligible": reserved}
    for side, mask, distance in (
            ("f", candidates & (ds >= 0.0), ds),
            ("l", candidates & (ds <= 0.0), -ds)):
        if k == 0:
            out.update({f"gap_{side}": s.clone(), f"v_{side}": v.clone(),
                        f"has_{side}": present.clone(),
                        f"index_{side}": torch.zeros_like(s, dtype=torch.long)})
            continue
        dist, index = torch.where(mask, distance, torch.full_like(ds, torch.inf)).min(-1)
        has = torch.isfinite(dist)
        # Leader gap subtracts LEADER length; follower gap subtracts EGO length.
        subtract = length if side == "f" else length.gather(-1, index)
        gap = torch.where(has, dist, torch.zeros_like(dist)) - subtract
        out[f"gap_{side}"] = torch.where(has, gap, torch.zeros_like(gap))
        out[f"v_{side}"] = torch.where(has, v.gather(-1, index), torch.zeros_like(v))
        out[f"has_{side}"] = has
        out[f"index_{side}"] = torch.where(has, index, -torch.ones_like(index))
    return out


def _scale(tau, reference: Tensor) -> Tensor:
    return torch.as_tensor(tau, dtype=reference.dtype, device=reference.device)


class _Ratio(torch.autograd.Function):
    """Ordinary division with a zero-safe adjoint at saturated infinite logits.

    PyTorch's quotient backward can form ``0 * inf`` even when the derivative
    through a saturated sigmoid is exactly zero.  Mask that ZERO adjoint before
    multiplying, and divide twice instead of squaring a tiny denominator.  The
    forward and all nonzero derivatives are the ordinary quotient; no logit or
    gradient is clipped.  Unrepresentably large true derivatives may still
    overflow, as with any finite-precision representation.
    """

    @staticmethod
    def forward(ctx, numerator: Tensor, denominator: Tensor) -> Tensor:
        ctx.save_for_backward(numerator, denominator)
        return numerator / denominator

    @staticmethod
    def backward(ctx, grad: Tensor):
        numerator, denominator = ctx.saved_tensors
        safe = torch.where(grad != 0.0, numerator, torch.zeros_like(numerator))
        dn = grad / denominator
        dd = -((grad * safe) / denominator) / denominator
        return dn.sum_to_size(numerator.shape), dd.sum_to_size(denominator.shape)


def _ratio(numerator: Tensor, denominator: Tensor) -> Tensor:
    return _Ratio.apply(numerator, denominator)


def log_survival(C: Tensor, tau: Tensor) -> Tensor:
    """``log(1 - sigmoid(C/tau))``, exact at +/-inf (tau > 0).

    Infinite endpoints are constants with zero derivative, rather than inputs
    to a division whose backward would produce ``0 * inf`` for tau.
    """
    finite = torch.isfinite(C)
    safe = torch.where(finite, C, torch.zeros_like(C))
    result = F.logsigmoid(_ratio(-safe, _scale(tau, C)))
    limit = torch.where(C == -torch.inf, torch.zeros_like(C),
                        torch.full_like(C, -torch.inf))
    return torch.where(finite, result, limit)


def log_event_mass(previous: Tensor, current: Tensor, tau: Tensor) -> Tensor:
    """Exact log CDF increment, zero/-negative increments -> -inf, no floor.

    For finite increasing a,b use
    ``logsigmoid(b/tau) + logsigmoid(-a/tau) + log(1-exp(-(b-a)/tau))``.
    This avoids subtracting saturated CDFs or exponentiating a large gap.
    Invalid branches receive safe operands before every nonlinear operation.
    """
    previous, current = torch.broadcast_tensors(previous, current)
    scale = _scale(tau, current)
    pair = torch.isfinite(previous) & torch.isfinite(current) & (current > previous)
    a = torch.where(pair, previous, torch.zeros_like(previous))
    b = torch.where(pair, current, torch.ones_like(current))
    delta = b - a
    gap = _ratio(delta, scale)
    # A positive CDF increment can have an underflowed scaled gap.  Evaluate
    # its log in the original units, not log(0).  Below machine epsilon the
    # omitted higher terms in log((1-exp(-x))/x) round away; -x/2 retains its
    # first-order correction.  This is numerical evaluation, not a mass floor.
    small = gap < torch.finfo(gap.dtype).eps
    regular_gap = torch.where(small, torch.ones_like(gap), gap)
    correction = torch.where(small, delta.log() - scale.log() - 0.5 * gap,
                             torch.log(-torch.expm1(-regular_gap)))
    mass = (F.logsigmoid(_ratio(b, scale)) + F.logsigmoid(_ratio(-a, scale))
            + correction)
    result = torch.where(pair, mass, torch.full_like(mass, -torch.inf))
    from_prior = (previous == -torch.inf) & torch.isfinite(current)
    c = torch.where(from_prior, current, torch.zeros_like(current))
    result = torch.where(from_prior, F.logsigmoid(_ratio(c, scale)), result)
    to_certain = torch.isfinite(previous) & (current == torch.inf)
    p = torch.where(to_certain, previous, torch.zeros_like(previous))
    result = torch.where(to_certain, log_survival(p, scale), result)
    return torch.where((previous == -torch.inf) & (current == torch.inf),
                       torch.zeros_like(result), result)


def advance_longitudinal(kin: Tensor, accel: Tensor, dt: float,
                         stop_time_integration: bool = False) -> Tensor:
    """Return updated kin, matching WmSModel.step's single longitudinal update.

    Lane/length are copied; s,v,a are updated.  No exits, lane-end projection,
    lane changes or future observations are applied by this primitive.
    """
    s, v = kin[..., 0], kin[..., 4]
    next_v = (v + accel * dt).clamp_min(0.0)
    next_s = s + 0.5 * (v + next_v) * dt
    if stop_time_integration:
        decel = torch.where(accel < 0.0, -accel, torch.ones_like(accel))
        stopping_distance = v.square() / (2.0 * decel)
        next_s = torch.where(v + accel * dt < 0.0, s + stopping_distance, next_s)
    out = kin.clone()
    out[..., 0], out[..., 4], out[..., 5] = next_s, next_v, accel
    return out


def update_state(c: Tensor, tau: Tensor, ramp: Tensor, present: Tensor, *,
                 carry: Optional[Mapping[str, Tensor]] = None,
                 identity: Optional[Tensor] = None,
                 uniform: Optional[Tensor] = None) -> dict[str, Tensor]:
    """Advance persistent latent, exact maximum and absorbing risk lifecycle.

    ``ramp`` is risk membership, NOT instantaneous physical eligibility (the
    latter is encoded by c=-inf).  Leaving risk after starting is terminal.
    ``cross`` proposes a move: a conflict-canceled proposal remains at risk
    with the same latent and may retry.  A committed move leaves risk on the
    next call.  Presence gaps / changed identities reset the whole slot;
    neighbor switches do not.  ``identity`` is lifecycle metadata only.

    One Logistic(0,1) draw is needed on first risk entry.  Supplied uniforms
    must be strictly in (0,1); generated uniforms redraw exact RNG endpoints,
    never clip the latent tails.  No RNG is consumed if no slot needs a draw.
    The returned carry retains its autograd history; crossing is hard/bool.
    """
    carry = {} if carry is None else carry
    present, ramp = present.bool(), ramp.bool()
    old_present = carry.get("present", torch.zeros_like(present))
    keep = present & old_present.bool()
    old_identity = carry.get("identity")
    if identity is None:
        identity = old_identity if old_identity is not None else torch.zeros_like(present, dtype=torch.long)
    elif old_identity is not None:
        keep = keep & (identity == old_identity)
    started = keep & carry.get("started", torch.zeros_like(present)).bool()
    terminal = keep & carry.get("terminal", torch.zeros_like(present)).bool()
    terminal = terminal | (started & ~ramp)
    z = torch.where(keep, carry.get("z", torch.zeros_like(c)), torch.zeros_like(c))
    maximum = torch.where(keep, carry.get("maximum", torch.full_like(c, -torch.inf)),
                          torch.full_like(c, -torch.inf))
    need = present & ramp & ~started & ~terminal
    if bool(need.any()):
        n = int(need.sum().item())
        if uniform is None:
            u = torch.rand(n, dtype=c.dtype, device=c.device)
            endpoints = (u == 0.0) | (u == 1.0)
            while bool(endpoints.any()):
                u[endpoints] = torch.rand(int(endpoints.sum().item()), dtype=c.dtype, device=c.device)
                endpoints = (u == 0.0) | (u == 1.0)
        else:
            u = torch.broadcast_to(uniform.to(device=c.device, dtype=c.dtype), c.shape)[need]
            if not bool((torch.isfinite(u) & (u > 0.0) & (u < 1.0)).all()):
                raise ValueError("needed logistic uniforms must be strictly between 0 and 1")
        z = z.masked_scatter(need, u.log() - torch.log1p(-u))
    active = present & ramp & ~terminal
    started = started | need
    maximum = torch.where(active, torch.maximum(maximum, c), maximum)
    # Explicit limits: even an extreme latent cannot overcome a physical veto.
    finite = torch.isfinite(c)
    safe_c = torch.where(finite, c, torch.zeros_like(c))
    crossing = (safe_c + _scale(tau, c) * z >= 0.0) & finite
    cross = active & (crossing | (c == torch.inf))
    return {"z": z, "maximum": maximum, "present": present, "identity": identity.clone(),
            "terminal": terminal, "cross": cross, "started": started}


def hard_st(c: Tensor, tau: Tensor, z: Tensor, width: float | Tensor,
            eligible: Tensor) -> Tensor:
    """Hard 0/1 forward, sigmoid straight-through backward (NOT a likelihood).

    Width is solely a numerical backward estimator parameter.  Infinite c
    has an exact constant forward and zero backward; invalid/ineligible slots
    have zero derivative.  ``tau`` is population dispersion, not this width.
    """
    width = _scale(width, c)
    if not bool((torch.isfinite(width) & (width > 0.0)).all()):
        raise ValueError("straight-through width must be finite and positive")
    active = eligible.bool() & torch.isfinite(c) & torch.isfinite(z)
    safe_c = torch.where(active, c, torch.zeros_like(c))
    safe_z = torch.where(active, z, torch.zeros_like(z))
    margin = safe_c + _scale(tau, c) * safe_z
    soft = torch.where(active, torch.sigmoid(_ratio(margin, width)), torch.zeros_like(margin))
    hard = (eligible.bool() & ((active & (margin >= 0.0)) | (c == torch.inf))).to(c.dtype)
    return hard + (soft - soft.detach())


def resolve_conflicts(kin: Tensor, present: Tensor, current_lane: Tensor,
                      candidate_lane: Tensor, learned_mask: Tensor) -> Tensor:
    """Cancel only proposed learned/HDV changes with exact final intersections.

    Fixed final occupancy wins over any proposed HDV change.  Otherwise the
    lower slot index wins a simultaneous conflict.  Recheck after cancellation:
    restoring an origin lane can block another mover (including lane swaps).
    Pre-existing conflicts among fixed vehicles cannot be repaired here.
    No padding is added; bumper touching is allowed.  Returns final lanes.
    """
    result = candidate_lane.clone()
    k = present.shape[-1]
    if k == 0:
        return result
    present = present.bool()
    changeable = present & learned_mask.bool() & (candidate_lane != current_lane)
    s, rear = kin[..., 0], kin[..., 0] - kin[..., 6]
    intersects = (torch.maximum(rear.unsqueeze(-1), rear.unsqueeze(-2))
                  < torch.minimum(s.unsqueeze(-1), s.unsqueeze(-2)))
    pair = (intersects & present.unsqueeze(-1) & present.unsqueeze(-2)
            & ~torch.eye(k, dtype=torch.bool, device=kin.device))
    slots = torch.arange(k, device=kin.device)
    for _ in range(k):
        moving = changeable & (result != current_lane)
        conflict = pair & (result.unsqueeze(-1) == result.unsqueeze(-2))
        fixed_bad = moving & (conflict & ~moving.unsqueeze(-2)).any(-1)
        # Resolve fixed blockers first.  Otherwise cancel the first losing
        # pair in lexicographic (winner,loser) order, then recompute occupancy.
        simultaneous = (conflict & moving.unsqueeze(-1) & moving.unsqueeze(-2)
                        & (slots.unsqueeze(-1) < slots.unsqueeze(-2)))
        flat = simultaneous.reshape(*simultaneous.shape[:-2], k * k)
        has_pair = flat.any(-1)
        loser = flat.to(torch.long).argmax(-1) % k
        priority_bad = (slots == loser.unsqueeze(-1)) & has_pair.unsqueeze(-1)
        cancel = torch.where(fixed_bad.any(-1, keepdim=True), fixed_bad, priority_bad)
        if not bool(cancel.any()):
            break
        result = torch.where(cancel, current_lane, result)
    return result


__all__ = ["FirstPassageNet", "geometry", "log_survival", "log_event_mass",
           "advance_longitudinal", "update_state", "hard_st", "resolve_conflicts"]
