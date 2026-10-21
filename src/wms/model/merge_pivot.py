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

from .merge_core import load_init


class FirstPassageNet(nn.Module):
    """Seven strict / eight hybrid softplus scalars, no residual or driver inputs.

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
    if fixed_target is not None:
        incoming_overlap = (intersects & fixed_target.bool().unsqueeze(-2)
                            & not_self).any(-1)
        physical = physical & ~incoming_overlap
    # Allocate simultaneous opportunities BEFORE willingness/latent draws. Exact
    # interval reservations are deterministic and independent of z, preserving a
    # scalar exact first-passage law conditional on the supplied scene. Arbitration
    # depending on competing z AFTER crossing would require a joint likelihood.
    reserved = torch.zeros_like(physical)
    for i in range(k):
        blocked = (intersects[..., i, :] & reserved).any(-1)
        reserved[..., i] = physical[..., i] & ~blocked
    out = {"v": v, "ramp": ramp, "overlap": overlap,
           "physical_eligible": physical, "reservation_veto": physical & ~reserved,
           "eligible": reserved}
    for side, mask, distance in (
            ("f", candidates & (ds >= 0.0), ds),
            ("l", candidates & (ds <= 0.0), -ds)):
        if k == 0:
            out.update({f"gap_{side}": s.clone(), f"v_{side}": v.clone(),
                        f"has_{side}": present.clone()})
            continue
        dist, index = torch.where(mask, distance, torch.full_like(ds, torch.inf)).min(-1)
        has = torch.isfinite(dist)
        # Leader gap subtracts LEADER length; follower gap subtracts EGO length.
        subtract = length if side == "f" else length.gather(-1, index)
        gap = torch.where(has, dist, torch.zeros_like(dist)) - subtract
        out[f"gap_{side}"] = torch.where(has, gap, torch.zeros_like(gap))
        out[f"v_{side}"] = torch.where(has, v.gather(-1, index), torch.zeros_like(v))
        out[f"has_{side}"] = has
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


def hybrid_log_masses(c: Tensor, tau: Tensor, w: Tensor, order: int = 32
                      ) -> tuple[Tensor, Tensor]:
    """Integrate persistent z ONCE over a fixed supplied path (time LAST).

    Positive composite Gauss--Legendre weights integrate the logistic prior in
    probability coordinates. Split at each finite crossing and at +/-1,+/-8
    conditional-noise widths; these are numerical integration knots, not a
    truncated prior or decision threshold. Both unbounded tails are retained.
    Log-CDF interpolation keeps far-tail intervals representable. Refinement of
    ``order`` must be checked on the fitted paths, NLL AND parameter gradients.
    Cost is quadratic in path length; callers should batch only scored cars.
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
                e, s = hybrid_log_masses(flat[used], tau, w, order)
                le[used], ls[used] = e, s
            return le.reshape_as(c), ls.reshape_as(c)
    if c.ndim > 1 and not any(x.requires_grad for x in (c, tau, w)):
        flat = c.reshape(-1, c.shape[-1])
        unique, inverse = torch.unique(flat, dim=0, return_inverse=True)
        if unique.shape[0] < flat.shape[0]:
            le, ls = hybrid_log_masses(unique, tau, w, order)
            return le[inverse].reshape_as(c), ls[inverse].reshape_as(c)
    finite = torch.isfinite(c)
    safe = torch.where(finite, c, torch.zeros_like(c))
    offsets = c.new_tensor([-8., -1., 0., 1., 8.])
    # Adaptive mesh placement is numerical, not a model transformation. Hold it
    # fixed in the adjoint and integrate the derivative of the defining law.
    # This also avoids spurious moving-boundary gradients at coincident knots.
    knots = ((-safe.unsqueeze(-1) + w * offsets) / tau).detach()
    knots = torch.where(finite.unsqueeze(-1), knots, torch.zeros_like(knots))
    knots = knots.flatten(-2).sort(-1).values
    # Coincident numerical knots have zero prior mass; removing shared duplicate
    # columns preserves the integral and avoids grids dominated by veto padding.
    knots = torch.unique_consecutive(knots, dim=-1)
    ends = torch.full_like(knots[..., :1], torch.inf)
    knots = torch.cat((-ends, knots, ends), -1)
    a, b = knots[..., :-1], knots[..., 1:]
    logweights = log_event_mass(a, b, c.new_tensor(1.))
    nodes_np, weights_np = _legendre(order)
    u = torch.as_tensor(nodes_np, device=c.device, dtype=c.dtype)
    weights = torch.as_tensor(weights_np, device=c.device, dtype=c.dtype)
    # sigmoid(knot) is never materialized, so z~1000 remains a finite node.
    la, lb = log_survival(-a, c.new_tensor(1.)), log_survival(-b, c.new_tensor(1.))
    sa, sb = log_survival(a, c.new_tensor(1.)), log_survival(b, c.new_tensor(1.))
    logu = torch.logaddexp(la.unsqueeze(-1) + torch.log1p(-u), lb.unsqueeze(-1) + u.log())
    logv = torch.logaddexp(sa.unsqueeze(-1) + torch.log1p(-u), sb.unsqueeze(-1) + u.log())
    z = (logu - logv).flatten(-2)
    lw = (logweights.unsqueeze(-1) + weights.log()).flatten(-2)
    # Do not create [cars,time,knots,order] intermediates for long paths.
    running = torch.zeros_like(z)
    events, survival = [], []
    for step in range(c.shape[-1]):
        lh, ls = conditional_log_hazards(c[..., step, None], tau, w, z)
        le = _logsumexp_supported(lw + running + lh, -1)
        no_previous = (c[..., :step] == -torch.inf).all(-1)
        le = torch.where(no_previous & (c[..., step] == torch.inf), torch.zeros_like(le), le)
        events.append(le)
        running = running + ls
        surv = _logsumexp_supported(lw + running, -1)
        surv = torch.where((c[..., :step + 1] == -torch.inf).all(-1), torch.zeros_like(surv), surv)
        survival.append(surv)
    return torch.stack(events, -1), torch.stack(survival, -1)


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


def update_state(c: Tensor, tau: Tensor, ramp: Tensor, present: Tensor, *,
                 carry: Optional[Mapping[str, Tensor]] = None,
                 identity: Optional[Tensor] = None,
                  uniform: Optional[Tensor] = None, w: Optional[Tensor] = None,
                  epsilon_uniform: Optional[Tensor] = None,
                  quadrature_order: int = 32) -> dict[str, Tensor]:
    """Advance persistent latent, exact maximum and absorbing risk lifecycle.

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
             epsilon: Optional[Tensor] = None) -> Tensor:
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
    if w is not None:
        margin = margin + _scale(w, c) * torch.where(active, epsilon, torch.zeros_like(c))
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
           "conditional_log_hazards", "hybrid_log_masses", "hybrid_quadrature_report", "advance_longitudinal",
           "update_state", "hard_st", "resolve_conflicts"]
