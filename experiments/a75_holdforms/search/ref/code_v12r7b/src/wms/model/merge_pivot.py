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
from functools import lru_cache
from typing import Mapping, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..physics.secure_gap import secure_gap_need, secure_gap_ok
from .merge_core import load_init


class FirstPassageNet(nn.Module):
    """Seven strict / eight hybrid softplus scalars; optional rollout driver veto.

    ``init`` optionally overrides ``merge_core.load_init()`` keys (``s0_m``,
    ``T_s``, ``b_mps2``, ``alpha_log_sigma``), or any effective parameter name.
    The default logistic scale variance-matches the old Gaussian log scale.
    """

    def __init__(self, init: Optional[Mapping[str, float]] = None, *,
                 hybrid: bool = False, w_init: float = 1.0) -> None:
        super().__init__()
        old = {**load_init(), **({} if init is None else init)}
        values = {
            "s_f": old["s0_m"], "T_f": old["T_s"], "b_f": old["b_mps2"],
            "s_l": old["s0_m"], "T_l": old["T_s"], "b_l": old["b_mps2"],
            "tau": old["alpha_log_sigma"] * math.sqrt(3.0) / math.pi,
        }
        self.hybrid = hybrid
        if hybrid:
            values["w"] = w_init
        for name, default in values.items():
            value = float(old.get(name, default))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must have a finite positive initialization")
            raw = value + math.log(-math.expm1(-value))
            self.register_parameter(f"{name}_raw", nn.Parameter(torch.tensor(raw)))

    def params(self) -> dict[str, Tensor]:
        """Effective gap parameters, tau, and (hybrid only) w; plain softplus."""
        names = ("s_f", "T_f", "b_f", "s_l", "T_l", "b_l", "tau")
        return {name: F.softplus(getattr(self, f"{name}_raw"))
                for name in (*names, "w") if hasattr(self, f"{name}_raw")}

    def boundary(self, geometry: Mapping[str, Tensor],
                 driver_params: Optional[Mapping[str, Tensor]] = None) -> Tensor:
        """Exact min log clearance ratio; absent=+inf, veto/zero gap=-inf.

        Mask operands BEFORE arithmetic: an absent/invalid observation must
        neither invent event support nor poison gradients in other slots.
        Supplying per-slot driver settings opts into the R13E secure-gap veto.
        Recorded replay omits them, preserving its original likelihood/support.
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
        c = torch.where(eligible, c, torch.full_like(c, -torch.inf))
        if driver_params is not None:
            secure = self.secure_gap_margin(geometry, driver_params) >= 0.0
            c = torch.where(secure, c, torch.full_like(c, -torch.inf))
        return c

    @staticmethod
    def secure_gap_margin(geometry: Mapping[str, Tensor],
                          driver_params: Mapping[str, Tensor]) -> Tensor:
        """Min log gap/required-gap; its SIGN is exactly GAPACC ``gs`` acceptance.

        Required bumper gap (wms.physics.secure_gap, GAPACC install_gate(use_T=True)):
        T_back*v_back + relu(v_back²/(2*b_back) - v_front²/(2*b_front)), with the actual
        learned settings gathered by geometry's neighbour indices (never slot order).
        NO standstill/s0 term (R13E review B3). A side passes iff its requirement is
        satisfied (gap >= need, direct comparison, no log rounding). Invalid settings or
        states fail closed (GAPACC would accept a non-finite need; not reproduced).
        The finite log ratio only feeds the labeled straight-through surrogate.
        No anticipation, relaxation, distance/speed cutoffs or overlap changes.
        """
        v = geometry["v"]
        settings = {name: driver_params[name].to(v) for name in ("time_headway", "decel")}
        if any(value.shape != v.shape for value in settings.values()):
            raise ValueError("secure-gap driver settings must have the geometry's per-slot shape")
        if v.shape[-1] == 0:
            return torch.full_like(v, torch.inf)
        eligible = geometry["eligible"].bool()
        tiny = torch.finfo(v.dtype).tiny
        sides = []
        for side in ("f", "l"):
            has = geometry[f"has_{side}"].bool()
            index = geometry[f"index_{side}"]
            gap, other = geometry[f"gap_{side}"], geometry[f"v_{side}"]
            neighbor = {name: value.gather(-1, index.clamp(0, v.shape[-1] - 1))
                        for name, value in settings.items()}
            back, front = (neighbor, settings) if side == "f" else (settings, neighbor)
            v_back, v_front = (other, v) if side == "f" else (v, other)
            t_back, b_back, b_front = back["time_headway"], back["decel"], front["decel"]
            # Mask operands BEFORE arithmetic (finite adjoints). Invalid settings or
            # states fail CLOSED (veto); on valid inputs the arithmetic and the direct
            # gap >= need comparison are GAPACC's, bit for bit.
            valid = (has & (index >= 0) & (index < v.shape[-1]) & ~torch.isnan(gap)
                     & torch.isfinite(v_back) & (v_back >= 0.0)
                     & torch.isfinite(v_front) & (v_front >= 0.0)
                     & torch.isfinite(t_back) & (t_back >= 0.0)
                     & torch.isfinite(b_back) & (b_back > 0.0)
                     & torch.isfinite(b_front) & (b_front > 0.0))
            v_back, v_front, t_back = (torch.where(valid, x, torch.zeros_like(x))
                                       for x in (v_back, v_front, t_back))
            b_back, b_front = (torch.where(valid, x, torch.ones_like(x)) for x in (b_back, b_front))
            need = secure_gap_need(v_back, v_front, t_back, b_back, b_front)
            ok = valid & secure_gap_ok(gap, need)
            finite = (valid & torch.isfinite(gap) & (gap > 0.0)
                      & torch.isfinite(need) & (need > 0.0))
            safe_gap = torch.where(finite, gap, torch.ones_like(gap))
            safe_need = torch.where(finite, need, torch.ones_like(need))
            ratio = safe_gap.log() - safe_need.log()
            # Force the surrogate's sign to the direct decision (1-ULP log rounding).
            ratio = torch.where(ok, ratio.clamp_min(0.0), ratio.clamp_max(-tiny))
            infinite = torch.where(ok, torch.full_like(v, torch.inf), torch.full_like(v, -torch.inf))
            margin = torch.where(finite, ratio, infinite)
            sides.append(torch.where(has, margin, torch.full_like(v, torch.inf)))
        margin = torch.minimum(*sides)
        return torch.where(eligible, margin, torch.full_like(margin, -torch.inf))

    def forward(self, geometry: Mapping[str, Tensor],
                driver_params: Optional[Mapping[str, Tensor]] = None) -> Tensor:
        return self.boundary(geometry, driver_params)


def geometry(kin: Tensor, present: Tensor, physics,
             decision_mask: Optional[Tensor] = None,
             fixed_target: Optional[Tensor] = None,
             route_shift: Optional[Tensor] = None) -> dict[str, Tensor]:
    """Full-scene neighbors, exact overlap veto and latent-independent reservations.

    Touching bumpers are physically eligible but constrain the learned boundary
    to -inf.  A missing/nonfinite PRESENT observation invalidates the scene;
    absent slots may contain arbitrary padding.  No distance cutoff is used.
    ``ramp`` includes both endpoints of the configured merge zone, independent
    of physical vetoes, for lifecycle tracking.
    ``decision_mask`` excludes nonlearned candidates from opportunity allocation;
    ``fixed_target`` marks immutable target occupancy from current actions only.
    ``route_shift`` (``wms.data.route_frame``) converts the collapsed pack-frame ``s`` of
    each vehicle to the lossless route coordinate ``x = s + shift`` in which bodies
    ``[x - length, x]``, gaps and overlaps are evaluated; it is the PRE-state shift
    carried along the single longitudinal update.  ``None`` keeps the pack line
    (WM rollout states, which are a junction-free line).  Returned ``s`` is always the
    pack value; ``route_s`` is the coordinate that the overlap veto used.
    """
    present = present.bool()
    observed = (torch.isfinite(kin).all(-1) & (kin[..., 1:4].sum(-1) > 0.0)
                & (kin[..., 6] > 0.0))
    scene_valid = (~present | observed).all(-1, keepdim=True)
    safe = torch.where((present & observed).unsqueeze(-1), kin, torch.zeros_like(kin))
    s, v, length = safe[..., 0], safe[..., 4], safe[..., 6]
    pack_s = s
    if route_shift is not None:
        shift = torch.where(present & observed & torch.isfinite(route_shift),
                            route_shift, torch.zeros_like(route_shift))
        s = s + shift.to(s.dtype)
    # Risk membership uses route/position observations, independently of a
    # missing speed/acceleration.  A temporary observation veto is not an exit.
    route = kin[..., 1:4]
    route_valid = torch.isfinite(route).all(-1) & (route.sum(-1) > 0.0)
    lane = torch.nan_to_num(route).argmax(-1)
    ramp_lane = int(getattr(physics, "merge_lane", 2))
    target_lane = int(getattr(physics, "merge_target_lane", 1))
    start = float(getattr(physics, "merge_zone_s0_m", 0.0))
    end = float(physics.merge_zone_s1_m)
    # Zone membership is a merge-edge coordinate: the route coordinate, so that a
    # vehicle inside a junction (stale pack ``s``) neither enters nor leaves the zone.
    zone_x = kin[..., 0] if route_shift is None else kin[..., 0] + torch.nan_to_num(route_shift)
    ramp = (present & route_valid & (lane == ramp_lane) & torch.isfinite(kin[..., 0])
            & (zone_x >= start) & (zone_x <= end))
    # Same lane and end test WITHOUT the start bound: the car is still on the ramp lane,
    # on or before the lane end.  Only used to keep an already started risk chain alive
    # (``ramp_risk_retain``); it never makes a car eligible.
    ramp_open = (present & route_valid & (lane == ramp_lane) & torch.isfinite(kin[..., 0])
                 & (zone_x <= end))
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
    out = {"v": v, "s": pack_s, "route_s": s, "length": length, "lane": lane, "pre_target": target,
           "scene_valid": scene_valid.expand_as(present),
           "overlap_index": torch.where(overlap, blocker_index, -torch.ones_like(blocker_index)),
           "fixed_overlap_index": torch.where(incoming_overlap, fixed_index,
                                                -torch.ones_like(fixed_index)),
           "fixed_overlap": incoming_overlap,
           "ramp": ramp, "ramp_open": ramp_open, "overlap": overlap,
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


def conditional_log_hazards(c: Tensor, tau: Tensor, w: Tensor, z: Tensor,
                            eligible: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
    """Hybrid conditional law, with exact vetoes/atoms and no support floor.

    Work in logs: even when exp(logh) underflows, its finite log probability
    remains scoreable. Infinite boundaries are constants, never divided.
    """
    finite = torch.isfinite(c)
    allowed = torch.ones_like(c, dtype=torch.bool) if eligible is None else eligible.bool()
    active = allowed & finite
    safe = torch.where(active, c, torch.zeros_like(c))
    safe_z = torch.where(active, z, torch.zeros_like(z))
    x = _ratio(safe + _scale(tau, c) * safe_z, _scale(w, c))
    certain = allowed & (c == torch.inf)
    logh = torch.where(active, -F.softplus(-x),
                       torch.where(certain, torch.zeros_like(x), torch.full_like(x, -torch.inf)))
    logwait = torch.where(active, -F.softplus(x),
                          torch.where(certain, torch.full_like(x, -torch.inf), torch.zeros_like(x)))
    return logh, logwait


def _logsumexp_supported(x: Tensor, dim: int) -> Tensor:
    """logsumexp with a zero adjoint for a genuinely empty support."""
    supported = torch.isfinite(x).any(dim, keepdim=True)
    safe = torch.where(supported, x, torch.zeros_like(x))
    result = torch.logsumexp(safe, dim)
    return torch.where(supported.squeeze(dim), result, torch.full_like(result, -torch.inf))


@lru_cache(maxsize=16)
def _legendre(order: int):
    import numpy as np

    if isinstance(order, bool) or int(order) != order or order < 2:
        raise ValueError("quadrature order must be an integer >= 2")
    x, weights = np.polynomial.legendre.leggauss(order)
    return (x + 1.0) / 2.0, weights / 2.0


def hybrid_log_masses(c: Tensor, tau: Tensor, w: Tensor, order: int = 32,
                      node_budget: Optional[int] = None) -> tuple[Tensor, Tensor]:
    """Integrate persistent z ONCE over a fixed supplied path (time LAST).

    Positive composite Gauss--Legendre weights integrate the logistic prior in
    probability coordinates. Split at each finite crossing and at +/-1,+/-8
    conditional-noise widths; these are numerical integration knots, not a
    truncated prior or decision threshold. Both unbounded tails are retained.
    Log-CDF interpolation keeps far-tail intervals representable. Refinement of
    ``order`` must be checked on the fitted paths, NLL AND parameter gradients.
    Memory is bounded by ``node_budget`` (elements of one [rows, live steps, nodes] block) via exact step/
    interval compaction, node grouping and checkpointing; time is still ~quadratic in live path length.
    """
    if c.shape[-1] == 0:
        return c.clone(), c.clone()
    # Accumulate likelihoods in float64, including for float32 WM parameters.
    c = c.to(torch.float64)
    atom = (c == torch.inf).long().cumsum(-1)
    after_atom = atom - (c == torch.inf).long() > 0
    c = torch.where(after_atom, torch.full_like(c, -torch.inf), c)
    tau, w = _scale(tau, c), _scale(w, c)
    if not bool((torch.isfinite(tau) & (tau > 0)).all()
                & (torch.isfinite(w) & (w > 0)).all()):
        raise ValueError("hybrid tau and w must be finite and positive")
    if c.ndim > 1:
        flat = c.reshape(-1, c.shape[-1])
        used = (flat != -torch.inf).any(-1)
        if not bool(used.all()):
            # Padding/never-eligible cars contribute exact zero survival loss,
            # and do not need a quadrature grid or an autograd graph per tick.
            le = torch.full_like(flat, -torch.inf)
            ls = torch.zeros_like(flat)
            if bool(used.any()):
                e, s = hybrid_log_masses(flat[used], tau, w, order, node_budget)
                le[used], ls[used] = e, s
            return le.reshape_as(c), ls.reshape_as(c)
    if c.ndim > 1 and not any(x.requires_grad for x in (c, tau, w)):
        flat = c.reshape(-1, c.shape[-1])
        unique, inverse = torch.unique(flat, dim=0, return_inverse=True)
        if unique.shape[0] < flat.shape[0]:
            le, ls = hybrid_log_masses(unique, tau, w, order, node_budget)
            return le[inverse].reshape_as(c), ls[inverse].reshape_as(c)
    return _hybrid_scan(c, tau, w, order, _HYBRID_NODE_BUDGET if node_budget is None else int(node_budget))


# Elements of one [rows, steps, nodes] block kept alive by autograd (about 12 float64 tensors of that
# size are saved per block, i.e. ~100 bytes per element). Larger blocks only trade memory for speed.
_HYBRID_NODE_BUDGET = 15_000_000


def _hybrid_block(cc: Tensor, tau: Tensor, w: Tensor, a: Tensor, b: Tensor, lw: Tensor, order: int
                  ) -> tuple[Tensor, Tensor]:
    """Event/survival log masses of a compacted path restricted to ONE group of integration intervals.

    ``cc`` is [R, A] (A steps, -inf padded), ``a``/``b``/``lw`` are [R, m] interval ends and log prior masses
    (``lw == -inf`` padding). Returns the [R, A] log-sum-exp over this group's Gauss nodes; the caller combines
    groups with a second log-sum-exp, which is the identical integral.
    """
    one = cc.new_tensor(1.)
    nodes_np, weights_np = _legendre(order)
    u = torch.as_tensor(nodes_np, device=cc.device, dtype=cc.dtype)
    weights = torch.as_tensor(weights_np, device=cc.device, dtype=cc.dtype)
    # sigmoid(knot) is never materialized, so z~1000 remains a finite node.
    la, lb = log_survival(-a, one), log_survival(-b, one)
    sa, sb = log_survival(a, one), log_survival(b, one)
    logu = torch.logaddexp(la.unsqueeze(-1) + torch.log1p(-u), lb.unsqueeze(-1) + u.log())
    logv = torch.logaddexp(sa.unsqueeze(-1) + torch.log1p(-u), sb.unsqueeze(-1) + u.log())
    z = (logu - logv).flatten(-2).unsqueeze(1)                    # [R, 1, nodes]
    lwn = (lw.unsqueeze(-1) + weights.log()).flatten(-2).unsqueeze(1)
    lh, ls = conditional_log_hazards(cc.unsqueeze(-1), tau, w, z)  # [R, A, nodes]
    running = ls.cumsum(1)
    previous = torch.cat((torch.zeros_like(running[:, :1]), running[:, :-1]), 1)
    return (_logsumexp_supported(lwn + previous + lh, -1),
            _logsumexp_supported(lwn + running, -1))


def _hybrid_scan(c: Tensor, tau: Tensor, w: Tensor, order: int, budget: int) -> tuple[Tensor, Tensor]:
    """Exact memory-bounded evaluation of the hybrid law (same integral and grid as the O(T^2) original).

    * Steps where a row is -inf have zero hazard and leave the running product unchanged, so each row is
      compacted to its own non-vetoed steps and the results are scattered back (events -inf, survival carried
      forward, 0 before the first live step).
    * Integration intervals of zero prior mass (coincident knots) contribute exactly nothing and are dropped
      per row; the knot set itself (including the 0 knot placed for vetoed steps) is unchanged.
    * Persistent-z nodes are independent, so they are processed in groups sized by ``budget`` and joined with a
      log-sum-exp; groups are gradient-checkpointed so only one [rows, steps, group-nodes] graph is alive.
    """
    shape = c.shape
    flat = c.reshape(-1, shape[-1])
    rows, steps = flat.shape
    live = flat != -torch.inf
    count = live.sum(-1)
    width = int(count.max())
    if width == 0:
        return torch.full_like(c, -torch.inf), torch.zeros_like(c)
    # Live steps first, in time order (stable); the tail is -inf padding.
    position = torch.sort((~live).to(torch.int8), dim=-1, stable=True).indices[:, :width]
    keep = torch.arange(width, device=c.device) < count.unsqueeze(-1)
    cc = torch.where(keep, flat.gather(1, position), torch.full_like(flat[:, :width], -torch.inf))
    # Adaptive mesh placement is numerical, not a model transformation. Hold it fixed in the adjoint and
    # integrate the derivative of the defining law (avoids spurious moving-boundary gradients).
    finite = torch.isfinite(flat)
    safe = torch.where(finite, flat, torch.zeros_like(flat))
    offsets = flat.new_tensor([-8., -1., 0., 1., 8.])
    knots = ((-safe.unsqueeze(-1) + w * offsets) / tau).detach()
    knots = torch.where(finite.unsqueeze(-1), knots, torch.zeros_like(knots))
    knots = knots.flatten(-2).sort(-1).values
    ends = torch.full_like(knots[..., :1], torch.inf)
    knots = torch.cat((-ends, knots, ends), -1)
    a, b = knots[..., :-1], knots[..., 1:]
    lw = log_event_mass(a, b, flat.new_tensor(1.))
    valid = lw != -torch.inf
    nvalid = valid.sum(-1)
    mmax = max(int(nvalid.max()), 1)
    pick = torch.sort((~valid).to(torch.int8), dim=-1, stable=True).indices[:, :mmax]
    ok = torch.arange(mmax, device=c.device) < nvalid.unsqueeze(-1)
    a = torch.where(ok, a.gather(1, pick), torch.zeros_like(lw[:, :mmax]))
    b = torch.where(ok, b.gather(1, pick), torch.ones_like(lw[:, :mmax]))
    lw = torch.where(ok, lw.gather(1, pick), torch.full_like(lw[:, :mmax], -torch.inf))
    group = max(1, min(mmax, budget // max(1, rows * width * order)))
    grad = torch.is_grad_enabled() and any(x.requires_grad for x in (cc, tau, w))
    events, survival = [], []
    for lo in range(0, mmax, group):
        args = (cc, tau, w, a[:, lo:lo + group], b[:, lo:lo + group], lw[:, lo:lo + group], order)
        if grad:
            from torch.utils.checkpoint import checkpoint
            e, s = checkpoint(_hybrid_block, *args, use_reentrant=False)
        else:
            e, s = _hybrid_block(*args)
        events.append(e)
        survival.append(s)
    if len(events) > 1:
        e, s = (_logsumexp_supported(torch.stack(x), 0) for x in (events, survival))
    else:
        e, s = events[0], survival[0]
    # Back to the full time axis.
    prior_live = live.long().cumsum(-1) - live.long()
    zero = torch.zeros_like(e[:, :1])
    ev = torch.full_like(flat, -torch.inf).scatter(1, position, e)
    ev = torch.where((prior_live == 0) & (flat == torch.inf), torch.zeros_like(ev), ev)
    slot = live.long().cumsum(-1) - 1
    sv = torch.where(slot >= 0, s.gather(1, slot.clamp_min(0)), zero.expand(rows, steps))
    return ev.reshape(shape), sv.reshape(shape)


def _keyed_logistic(seed: Tensor, identity: Tensor, step: Tensor, dtype) -> Tensor:
    """Counter-based 64-bit mixing; RNG key/counter are explicit rollout state.

    Integer overflow is intentional modulo-2**64 arithmetic. Logical shifts
    are expressed using masks because torch int64 right shifts are arithmetic.
    Midpoint 53-bit uniforms have no endpoints and need no clipping/redraw.
    """
    x = seed ^ (identity.long() * 6364136223846793005) ^ (step * 1442695040888963407)
    x = (x ^ ((x >> 30) & ((1 << 34) - 1))) * -4658895280553007687
    x = (x ^ ((x >> 27) & ((1 << 37) - 1))) * -7723592293110705685
    x = x ^ ((x >> 31) & ((1 << 33) - 1))
    # 52 bits ensure midpoint addition is exact in float64 at both endpoints.
    u = ((x & ((1 << 52) - 1)).double() + 0.5) / float(1 << 52)
    return (u.log() - torch.log1p(-u)).to(dtype)


def hybrid_quadrature_report(c: Tensor, tau: Tensor, w: Tensor, *,
                              event_index: Optional[int] = None, order: int = 32,
                              atol: float = 1.e-6) -> dict:
    """Refinement certificate for a fixed path/fit, including true-law adjoints.

    Compare order N,2N,4N; require BOTH successive refinements to meet the
    registered 1e-6 absolute parity tolerance on masses, summed sequence NLL,
    and c/tau/w gradients. No scale is changed to make a check pass. A false
    result calls for more quadrature/work on that fit, not a likelihood floor.
    ``None`` scores administrative censoring at the supplied path endpoint.
    """
    rows = []
    with torch.enable_grad():
        for n in (order, 2 * order, 4 * order):
            boundary = c.detach().double().requires_grad_(True)
            scale = _scale(tau, boundary).detach().requires_grad_(True)
            width = _scale(w, boundary).detach().requires_grad_(True)
            le, ls = hybrid_log_masses(boundary, scale, width, n)
            score = -(ls[..., -1] if event_index is None else le[..., event_index]).sum()
            if not bool(torch.isfinite(score)):
                return {"converged": False, "supported": False, "orders": [order, 2 * order, 4 * order]}
            # A deterministic sequence can be independent of every parameter.
            score = score + 0. * (boundary[torch.isfinite(boundary)].sum() + scale + width)
            gradients = torch.autograd.grad(score, (boundary, scale, width))
            rows.append((torch.cat((le.exp().flatten(), ls.exp().flatten())).detach(),
                         score.detach(), torch.cat([g.flatten() for g in gradients]).detach()))
    errors = [tuple(float((a - b).abs().max()) for a, b in zip(left, right))
              for left, right in zip(rows[:-1], rows[1:])]
    return {"converged": all(math.isfinite(e) and e <= atol for row in errors for e in row),
            "supported": True, "orders": [order, 2 * order, 4 * order], "atol": atol,
            "mass_errors": [row[0] for row in errors], "nll_errors": [row[1] for row in errors],
            "gradient_errors": [row[2] for row in errors]}


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


def ramp_lane_end_stop(kin: Tensor, next_s: Tensor, next_v: Tensor, present: Tensor, physics,
                       route_shift: Optional[Tensor] = None
                       ) -> tuple[Tensor, Tensor, Tensor]:
    """Real ramp-lane end (``physics.merge_zone_s1_m``): SUMO never moves a car past it.

    ``kin`` is the PRE-step snapshot (its one-hot lane is the lane the car drives on).
    A present ramp-lane car whose longitudinal update would put its front beyond the lane
    end (route coordinate ``x = s + shift``) is held at the end with speed 0.  No tolerance:
    the end is the network geometry already in ``physics`` (cast to the tensor dtype only).
    Returns ``(s, v, stopped)``; the caller decides which cars keep the stop (cars that
    change lanes in the same step do not).  Only used when
    ``physics.ramp_lane_end_stop`` is true.
    """
    route = torch.nan_to_num(kin[..., 1:4])
    on_ramp = present.bool() & (route.argmax(-1) == int(physics.merge_lane)) & (route.sum(-1) > 0.0)
    end = torch.as_tensor(float(physics.merge_zone_s1_m), dtype=next_s.dtype, device=next_s.device)
    shift = (torch.zeros_like(next_s) if route_shift is None else
             torch.nan_to_num(route_shift).to(next_s.dtype))
    stopped = on_ramp & torch.isfinite(next_s) & (next_s + shift > end)
    return (torch.where(stopped, end - shift, next_s),
            torch.where(stopped, torch.zeros_like(next_v), next_v), stopped)


def close_risk(risk: Tensor, ramp_open: Tensor, reset: Optional[Tensor] = None) -> Tensor:
    """Keep a started risk chain alive while the car is still on the ramp lane before its end.

    ``risk`` / ``ramp_open`` / ``reset`` are ``[B, T, K]`` booleans (``reset`` = a fresh
    identity starts at that tick).  A chain that has started at some tick stays at risk on later
    ticks where ``ramp_open`` holds even if ``risk`` is momentarily false.  The WM-predicted
    next position of a car crawling across the zone start (route x = ``merge_zone_s0_m``, the
    end of the 2.93 m junction) jitters around 0 while its true route x is monotone; such a tick
    is not a physical zone exit, so it must not be an absorbing risk exit.  Real exits (lane change,
    passing the lane end, absence, a new identity) still end the chain.  No constants.
    """
    out = risk.clone()
    active = risk[:, 0]
    for t in range(1, risk.shape[1]):
        keep = ramp_open[:, t] & active
        if reset is not None:
            keep = keep & ~reset[:, t]
        active = risk[:, t] | keep
        out[:, t] = active
    return out


def update_state(c: Tensor, tau: Tensor, ramp: Tensor, present: Tensor, *,
                 carry: Optional[Mapping[str, Tensor]] = None,
                 identity: Optional[Tensor] = None, retain: Optional[Tensor] = None,
                  uniform: Optional[Tensor] = None, w: Optional[Tensor] = None,
                  epsilon_uniform: Optional[Tensor] = None,
                  quadrature_order: int = 32,
                  diagnostics: bool = True) -> dict[str, Tensor]:
    """Advance persistent latent, exact maximum and absorbing risk lifecycle.

    S-SLOWROLL / S-SPEED2: ``diagnostics=False`` skips the hybrid quadrature that only
    produces the ``logS`` / ``event_mass`` / ``cdf`` diagnostics (returned as NaN).  It
    re-integrates each at-risk slot's whole boundary history every step (O(A^2) per step,
    O(A^3) per waiting car) and never feeds ``cross``, ``z``, ``epsilon`` or any RNG.

    ``ramp`` is risk membership, NOT instantaneous physical eligibility (the
    latter is encoded by c=-inf).  Leaving risk after starting is terminal.
    ``cross`` proposes a move: a conflict-canceled proposal remains at risk
    with the same latent and may retry.  A committed move leaves risk on the
    next call.  Presence gaps / changed identities reset the whole slot;
    neighbor switches do not.  ``identity`` is lifecycle metadata only.

    One Logistic(0,1) draw is needed on first risk entry. Supplied uniforms
    must be strictly in (0,1); generated uniforms redraw exact RNG endpoints,
    never clip the latent tails.  No RNG is consumed if no slot needs a draw.
    The returned carry retains its autograd history; crossing is hard/bool.
    With ``w`` supplied the hybrid law adds keyed per-decision epsilon. Stream
    keys/generations/counters and boundary history are explicit carry tensors;
    new births after resumption use saved keys, not the global RNG. Diagnostic
    quadrature includes the same persistent population prior. Native rollout
    detaches those diagnostics and uses the separately labeled ST estimator.
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
    if retain is not None:
        # ``close_risk`` for the online path: a started, non-terminal chain stays at risk
        # while the car is still on the ramp lane before its end (``geo['ramp_open']``).
        ramp = ramp | (retain.bool() & started & ~terminal)
    terminal = terminal | (started & ~ramp)
    z = torch.where(keep, carry.get("z", torch.zeros_like(c)), torch.zeros_like(c))
    maximum = torch.where(keep, carry.get("maximum", torch.full_like(c, -torch.inf)),
                           torch.full_like(c, -torch.inf))
    need = present & ramp & ~started & ~terminal
    if w is not None:
        # Allocate the saved per-slot stream keys once, even for slots whose
        # first identity arrives later. New births after checkpoint resumption
        # therefore never depend on the process-global RNG. Generation is a
        # stream namespace for presence reuse when explicit IDs are unavailable.
        seed = carry.get("rng_seed")
        if seed is None:
            seed = torch.randint(0, 1 << 52, c.shape, device=c.device, dtype=torch.long)
        generation = carry.get("rng_generation", torch.zeros_like(identity)) + need.long()
        stream_identity = identity.long() ^ (generation * 2862933555777941757)
        if uniform is None:
            draw = _keyed_logistic(seed, stream_identity, torch.zeros_like(identity), c.dtype)
            z = torch.where(need, draw, z)
    if bool(need.any()):
        n = int(need.sum().item())
        if uniform is None and w is None:
            u = torch.rand(n, dtype=c.dtype, device=c.device)
            endpoints = (u == 0.0) | (u == 1.0)
            while bool(endpoints.any()):
                u[endpoints] = torch.rand(int(endpoints.sum().item()), dtype=c.dtype, device=c.device)
                endpoints = (u == 0.0) | (u == 1.0)
        elif uniform is not None:
            u = torch.broadcast_to(uniform.to(device=c.device, dtype=c.dtype), c.shape)[need]
            if not bool((torch.isfinite(u) & (u > 0.0) & (u < 1.0)).all()):
                raise ValueError("needed logistic uniforms must be strictly between 0 and 1")
        if uniform is not None or w is None:
            z = z.masked_scatter(need, u.log() - torch.log1p(-u))
    active = present & ramp & ~terminal
    started = started | need
    maximum = torch.where(active, torch.maximum(maximum, c), maximum)
    # Explicit limits: even an extreme latent cannot overcome a physical veto.
    finite = torch.isfinite(c)
    safe_c = torch.where(finite, c, torch.zeros_like(c))
    crossing = (safe_c + _scale(tau, c) * z >= 0.0) & finite
    result = {"z": z, "maximum": maximum, "present": present, "identity": identity.clone(),
              "terminal": terminal, "cross": active & (crossing | (c == torch.inf)), "started": started}
    if w is None:
        return result
    counter = torch.where(keep, carry.get("decision_step", torch.zeros_like(identity)),
                          torch.zeros_like(identity))
    counter = counter + active.long()
    epsilon = _keyed_logistic(seed, stream_identity, counter, c.dtype)
    if epsilon_uniform is not None:
        u = torch.broadcast_to(epsilon_uniform.to(c), c.shape)
        if not bool((~active | (torch.isfinite(u) & (u > 0) & (u < 1))).all()):
            raise ValueError("epsilon uniforms must be strictly between 0 and 1")
        safe_u = torch.where(active, u, torch.full_like(u, .5))
        epsilon = safe_u.log() - torch.log1p(-safe_u)
    epsilon = torch.where(active, epsilon, torch.zeros_like(epsilon))
    margin = safe_c + _scale(tau, c) * z + _scale(w, c) * epsilon
    result["cross"] = active & ((finite & (margin >= 0)) | (c == torch.inf))
    history = carry.get("history", c.new_empty((*c.shape, 0)))
    history = torch.where(keep.unsqueeze(-1), history, torch.full_like(history, -torch.inf))
    history = torch.cat((history, torch.where(active, c, torch.full_like(c, -torch.inf)).unsqueeze(-1)), -1)
    if not diagnostics:
        nan = torch.full_like(c, torch.nan, dtype=torch.float64)
        result.update(epsilon=epsilon, rng_seed=seed, rng_generation=generation,
                      decision_step=counter, history=history,
                      logS=nan, event_mass=nan.clone(), cdf=nan.clone())
        return result
    logS = carry.get("logS", torch.zeros_like(c, dtype=torch.float64))
    event_mass = torch.zeros_like(c, dtype=torch.float64)
    if bool(active.any()):
        le, ls = hybrid_log_masses(history[active], tau, w, quadrature_order)
        logS = logS.masked_scatter(active, ls[..., -1])
        event_mass = event_mass.masked_scatter(active, le[..., -1].exp())
    result.update(epsilon=epsilon, rng_seed=seed, rng_generation=generation,
                  decision_step=counter, history=history,
                  logS=logS, event_mass=event_mass, cdf=-torch.expm1(logS))
    return result


def hard_st(c: Tensor, tau: Tensor, z: Tensor, width: float | Tensor,
              eligible: Tensor, *, w: Optional[Tensor] = None,
              epsilon: Optional[Tensor] = None,
              secure_gap_margin: Optional[Tensor] = None) -> Tensor:
    """Hard 0/1 forward, sigmoid straight-through backward (NOT a likelihood).

    Width is solely a numerical backward estimator parameter.  Infinite c
    has an exact constant forward and zero backward; invalid/ineligible slots
    have zero derivative.  ``tau`` is population dispersion, not this width.
    With R13E, c is the UNGATED willingness boundary: the hard secure-gap veto
    is ANDed in forward. Its labeled surrogate multiplies the willingness sigmoid
    by sigmoid(secure_gap_margin/tau), reusing the existing learned positive
    dimensionless scale (no new checkpoint tensors). This is NOT a likelihood or
    the derivative of the discontinuous hard veto. Physical vetoes still have
    exactly zero gradient, while finite rejected gaps retain driver/state credit.
    """
    width = _scale(width, c)
    if not bool((torch.isfinite(width) & (width > 0.0)).all()):
        raise ValueError("straight-through width must be finite and positive")
    active = eligible.bool() & torch.isfinite(c) & torch.isfinite(z)
    safe_c = torch.where(active, c, torch.zeros_like(c))
    safe_z = torch.where(active, z, torch.zeros_like(z))
    margin = safe_c + _scale(tau, c) * safe_z
    if w is not None:
        margin = margin + _scale(w, c) * torch.where(active, epsilon, torch.zeros_like(c))
    soft = torch.where(active, torch.sigmoid(_ratio(margin, width)), torch.zeros_like(margin))
    hard = (eligible.bool() & ((active & (margin >= 0.0)) | (c == torch.inf))).to(c.dtype)
    if secure_gap_margin is not None:
        finite_gap = eligible.bool() & torch.isfinite(secure_gap_margin)
        safe_gap = torch.where(finite_gap, secure_gap_margin, torch.zeros_like(c))
        gate = torch.where(finite_gap, torch.sigmoid(_ratio(safe_gap, _scale(tau, c))),
                           (eligible.bool() & (secure_gap_margin == torch.inf)).to(c.dtype))
        soft = soft * gate
        hard = hard * (secure_gap_margin >= 0.0).to(c.dtype)
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
           "conditional_log_hazards", "hybrid_log_masses", "hybrid_quadrature_report", "advance_longitudinal", "ramp_lane_end_stop", "close_risk",
           "update_state", "hard_st", "resolve_conflicts"]
