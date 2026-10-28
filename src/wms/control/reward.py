"""Differentiable per-physics-step team reward and detached episode logging.

Slots must retain vehicle identity through a transition, including the final
position/speed of a vehicle that exits.

Entry queue (SRCWAIT_NOTES.md, 2026-10-13): vehicles that have been requested (arrived at
an entrance) but are not yet inserted are invisible to the slot state.  Callers may pass
their count per batch row, ``entry_queue`` [B] (argument, or ``state["entry_queue"]``):
the number of vehicles waiting to enter DURING the transition ``state -> next_state``
(may be a time average, i.e. fractional).  Each one costs ``dt`` vehicle-seconds of
``source_wait``, weighted exactly like on-road delay, counts in the per-vehicle
normalisation and enters the fairness CVaR as a vehicle with delay ``dt``.  Absent ->
the original (on-road only) reward, bit for bit.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from wms.core.constants import DEFAULT_LENGTH_M

from .contract import MERGE_S1_M, PHYSICS_DT_S, SPEED_LIMIT_MPS, RewardWeights

_KEYS = ("reward", "cost_ttc", "delay", "throughput", "smooth", "hard_brake",
         "fairness_delay", "source_wait")
_CVAR_ALPHA = 0.9
_CVAR_TEMPERATURE_S = 0.01
_CROSSING_WIDTH_M = 1.0
_TTC_TEMPERATURE_S = 0.05
_CLOSING_EPS_MPS = 0.01
_FREE_SPEED_FLOOR_MPS = 10.0      # V8POL: time-gap speed floor of the free-road test


def _cvar_bisection_eager(values: Tensor, present: Tensor, weight: Tensor,
                          count: Tensor) -> Tensor:
    """Detached 32-step bisection for the CVaR threshold eta (pure, RNG-free, no sync)."""
    tau = _CVAR_TEMPERATURE_S
    tail = 1.0 - _CVAR_ALPHA
    any_present = present.any(-1)
    lo = torch.where(present, values, torch.full_like(values, float("inf"))).amin(-1)
    hi = torch.where(present, values, torch.full_like(values, -float("inf"))).amax(-1)
    lo = torch.where(any_present, lo, torch.zeros_like(lo)) - 10.0 * tau
    hi = torch.where(any_present, hi, torch.zeros_like(hi)) + 10.0 * tau
    for _ in range(32):
        eta = 0.5 * (lo + hi)
        mass = torch.where(present, weight * torch.sigmoid((values - eta[:, None]) / tau),
                           torch.zeros_like(values)).sum(-1) / count
        lo = torch.where(mass > tail, eta, lo)
        hi = torch.where(mass > tail, hi, eta)
    return 0.5 * (lo + hi)


#: S-POLSPEED ``WMS_SPEED_GRAPH_CVAR``: one captured CUDA graph per input geometry.
_CVAR_GRAPHS: dict = {}


def _cvar_bisection(values: Tensor, present: Tensor, weight: Tensor, count: Tensor) -> Tensor:
    """Eager bisection, or (opt-in, CUDA only) a replay of the SAME kernels on the SAME shapes.

    The ~360 tiny launches of the bisection dominate the paired (row-local, [1,K]) reward.
    A CUDA graph replays exactly the captured kernels with identical launch geometry and
    inputs copied into static buffers, so eta is bit-identical; the bisection is detached,
    deterministic and draws no random numbers. Python dispatch modes (the LONGMEM random
    tape) are bypassed only for these non-random ops.
    """
    from wms.core import speed_flags
    if (not values.is_cuda or not speed_flags.enabled("graph_cvar")
            or torch.cuda.is_current_stream_capturing() or torch.is_grad_enabled()):
        return _cvar_bisection_eager(values, present, weight, count)
    from torch.utils._python_dispatch import _disable_current_modes
    inputs = (values, present, weight, count)
    key = tuple((tuple(t.shape), t.dtype, tuple(t.stride())) for t in inputs) + (values.device,)
    with _disable_current_modes():
        entry = _CVAR_GRAPHS.get(key)
        if entry is None:
            static = tuple(t.clone() for t in inputs)
            current = torch.cuda.current_stream(values.device)
            side = torch.cuda.Stream(values.device)
            side.wait_stream(current)
            with torch.cuda.stream(side):
                _cvar_bisection_eager(*static)          # warm-up outside capture
            current.wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                out = _cvar_bisection_eager(*static)
            entry = _CVAR_GRAPHS[key] = (graph, static, out)
        graph, static, out = entry
        for dst, src in zip(static, inputs):
            dst.copy_(src)
        graph.replay()
        return out.clone()


def _soft_cvar(delay: Tensor, present: Tensor, weight: Tensor | None = None) -> Tensor:
    """Bias-corrected, softplus Rockafellar--Uryasev upper-tail CVaR.

    Minimise eta + mean(tau*softplus((delay-eta)/tau))/(1-alpha).
    The sigmoid tail mass determines the optimum eta. Bisection is detached:
    the envelope theorem gives the derivative of the minimised objective
    directly through delay, without differentiating the numerical optimiser.
    Subtract its constant equal-delay entropy bias, so equal delays return
    exactly that delay (up to floating-point error), not a smoothing penalty.
    ``weight`` [B,K] (optional, >= 0) counts column k as ``weight`` identical vehicles
    (exactly equivalent to replicating it); default = 1 per present slot.
    """
    if delay.shape[-1] == 0:
        return delay.sum(-1)
    if weight is None:
        weight = present.to(delay.dtype)
    else:
        weight = torch.where(present, weight.to(delay.dtype), torch.zeros_like(delay))
        present = present & (weight > 0)
    count = weight.sum(-1).clamp_min(1).to(delay.dtype)
    tau = _CVAR_TEMPERATURE_S
    tail = 1.0 - _CVAR_ALPHA
    with torch.no_grad():
        eta = _cvar_bisection(delay.detach(), present, weight, count)
    excess = tau * F.softplus((delay - eta[:, None]) / tau)
    excess = torch.where(present, weight * excess, torch.zeros_like(excess)).sum(-1) / count
    entropy = -_CVAR_ALPHA * math.log(_CVAR_ALPHA) - tail * math.log(tail)
    cvar = eta + excess / tail - tau * entropy / tail
    return torch.where(present.any(-1), cvar.clamp_min(0.0), torch.zeros_like(cvar))


TTC_PAIRS = ("all", "cav", "cav_follower")


def _ttc_cost(state: Mapping[str, Tensor], limit_s: float, pairs: str = "all") -> Tensor:
    """Soft relu(limit - TTC), summed over directed closing same-lane pairs.

    A softplus smooths the positive part in seconds. A small denominator floor
    avoids division by zero; non-closing pairs contribute exactly zero.
    Pair membership/order is discrete; positions/speeds within it have gradients.
    ``pairs="cav"`` (V7POL) keeps only pairs whose follower or leader is a CAV
    (``state["is_cav"]``); human-human pairs then cost nothing.
    ``pairs="cav_follower"`` (V8POL) keeps only pairs whose FOLLOWER (the closing vehicle) is a CAV:
    a human closing on a slow CAV leader costs nothing (the CAV cannot fix that by braking).
    Index convention: entry [b, i, j] has ``ds = s_j - s_i >= 0`` (j at or ahead of i) and
    ``closing = relu(v_i - v_j)``, so i (dim -2) is the follower and j (dim -1) the leader.
    """
    if pairs not in TTC_PAIRS:
        raise ValueError(f"ttc_pairs must be one of {TTC_PAIRS}")
    present = state["present"].bool()
    s = torch.where(present, state["s_m"], torch.zeros_like(state["s_m"]))
    v = torch.where(present, state["v_mps"], torch.zeros_like(state["v_mps"]))
    lane = state["lane"]
    length = state.get("length_m", torch.full_like(s, DEFAULT_LENGTH_M))
    length = torch.where(present, length, torch.zeros_like(length))
    ds = s[:, None, :] - s[:, :, None]
    pair = (present[:, :, None] & present[:, None, :]
             & (lane[:, :, None] == lane[:, None, :])
             & (lane[:, :, None] >= 0) & (ds >= 0.0)
             & ~torch.eye(s.shape[-1], dtype=torch.bool, device=s.device)[None])
    if pairs in ("cav", "cav_follower"):
        if "is_cav" not in state:
            raise ValueError(f"ttc_pairs={pairs!r} needs state['is_cav']")
        cav = state["is_cav"].bool() & present
        if pairs == "cav":
            pair = pair & (cav[:, :, None] | cav[:, None, :])  # follower i or leader j is a CAV
        else:
            pair = pair & cav[:, :, None]                      # follower i is a CAV
    gap = ds - length[:, None, :]
    closing = F.relu(v[:, :, None] - v[:, None, :])
    ttc = gap.clamp_min(0.0) / closing.clamp_min(_CLOSING_EPS_MPS)
    exposure = _TTC_TEMPERATURE_S * F.softplus((limit_s - ttc) / _TTC_TEMPERATURE_S)
    return torch.where(pair & (closing > 0), exposure, torch.zeros_like(exposure)).sum((-2, -1))


def _free_speed_shortfall(state: Mapping[str, Tensor], next_state: Mapping[str, Tensor],
                          time_gap_s: float, range_m: float) -> Tensor:
    """V8POL free-road speed shortfall [B]: ``dt * sum_free relu(1 - v/SPEED_LIMIT) / n_cav``.

    A REWARD INCENTIVE, not a controller: it only adds a (differentiable) penalty for a CAV
    driving below the road speed limit SPEED_LIMIT_MPS (the same reference as the ``delay`` term,
    same ``dt * relu(1 - v/limit)`` form) while its own road ahead is free; it never sets or
    bounds an action.  CAV i (``state["is_cav"]``, present in ``state``) is "free" when, in
    ``next_state``, it has no same-lane leader within ``range_m`` metres (bumper gap) OR its time
    gap ``gap / max(v_i, _FREE_SPEED_FLOOR_MPS)`` exceeds ``time_gap_s``.  The speed floor stops a
    near-stopped CAV in a queue from counting as free.  The free mask is detached (discrete); the
    gradient flows only through the free CAVs' next speeds -- HDVs get no term and no gradient.
    Normalised by the number of present CAVs (0 without CAVs).
    """
    if "is_cav" not in state:
        raise ValueError("free_speed needs state['is_cav']")
    present = state["present"].bool()
    cav = state["is_cav"].bool() & present
    zero = torch.zeros_like(next_state["v_mps"])
    v = torch.where(present, next_state["v_mps"], zero)
    with torch.no_grad():
        s = torch.where(present, next_state["s_m"], zero).detach()
        lane = next_state["lane"]
        length = next_state.get("length_m", torch.full_like(s, DEFAULT_LENGTH_M))
        ds = s[:, None, :] - s[:, :, None]               # [b, i, j] = s_j - s_i (j ahead of i)
        ahead = (present[:, :, None] & present[:, None, :]
                 & (lane[:, :, None] == lane[:, None, :]) & (ds >= 0.0)
                 & ~torch.eye(s.shape[-1], dtype=torch.bool, device=s.device)[None])
        gap = torch.where(ahead, ds - length.to(s.dtype)[:, None, :],
                          torch.full_like(ds, float("inf"))).amin(-1)      # [b, i] leader gap
        headway = gap / v.detach().clamp_min(_FREE_SPEED_FLOOR_MPS)
        free = cav & (lane >= 0) & ((gap > range_m) | (headway > time_gap_s))
    shortfall = torch.where(free, PHYSICS_DT_S * F.relu(1.0 - v / SPEED_LIMIT_MPS), zero)
    return shortfall.sum(-1) / cav.sum(-1).clamp_min(1).to(shortfall.dtype)


def team_reward(state: Mapping[str, Tensor], next_state: Mapping[str, Tensor],
                accel: Tensor, weights: RewardWeights,
                entry_queue: Tensor | None = None) -> dict[str, Tensor]:
    """Return the seven contract metrics plus ``source_wait``, each a tensor of shape [B].

    The current state's present cohort is charged this step, including exits;
    delay uses *next* speeds: dt*relu(1-v_next/SPEED_LIMIT). Components are sums
    over that cohort, except fairness_delay (soft 90% CVaR in vehicle-seconds).
    Throughput is the positive increment in sigmoid((s-MERGE_S1)/1 metre).
    Smooth/hard_brake are sum(a**2) and sum(relu(-a-3)**2), for HDVs and CAVs.
    The additive weighted terms are divided by max(present_count, 1); the fairness
    penalty (already a per-vehicle tail delay) is subtracted undivided. TTC is a separate, nonnegative constraint,
    evaluated on next-state occupants, never folded into the reward.

    ``entry_queue`` [B] (or ``state["entry_queue"]``; module docstring): vehicles waiting to
    enter during this transition.  ``source_wait = dt * entry_queue`` is charged with
    ``weights.delay``, the normalisation count becomes ``present + entry_queue`` and the
    CVaR includes the queued vehicles at delay ``dt`` each.  ``source_wait`` is zero without it.
    A queue that requires grad (straight-through surrogate, SRCGRAD_NOTES.md) passes its gradient
    to ``source_wait`` only; values are unchanged.

    V7POL (defaults off): ``weights.merge_discharge`` adds ``merge_discharge * throughput`` to the
    reward without the count normalisation (``throughput`` already IS the soft crossing count of
    the merge end MERGE_S1_M); ``weights.ttc_pairs = "cav"`` restricts the TTC cost to pairs with a
    CAV follower or leader.

    V8POL (defaults off): ``weights.ttc_pairs = "cav_follower"`` restricts the TTC cost to pairs whose
    follower (closing vehicle) is a CAV; ``weights.free_speed`` subtracts ``free_speed *
    free_speed_shortfall`` (``_free_speed_shortfall``: mean over CAVs of the free-road speed
    shortfall, a reward incentive, not a controller; no count normalisation) and then also returns
    the ``free_speed_shortfall`` component.
    """
    if weights.ttc_limit_s <= 0.0:
        raise ValueError("ttc_limit_s must be positive")
    present = state["present"].bool()
    if accel.ndim != 2 or accel.shape != present.shape:
        raise ValueError("accel and present must have matching [B, K] shapes")
    zero = torch.zeros_like(next_state["v_mps"])
    v = torch.where(present, next_state["v_mps"], zero)
    delay_by_vehicle = torch.where(
        present, PHYSICS_DT_S * F.relu(1.0 - v / SPEED_LIMIT_MPS), zero)
    delay = delay_by_vehicle.sum(-1)
    s = torch.where(present, state["s_m"], zero)
    next_s = torch.where(present, next_state["s_m"], zero)
    crossing = (torch.sigmoid((next_s - MERGE_S1_M) / _CROSSING_WIDTH_M)
                - torch.sigmoid((s - MERGE_S1_M) / _CROSSING_WIDTH_M))
    throughput = torch.where(present, F.relu(crossing), zero).sum(-1)
    a = torch.where(present, accel, torch.zeros_like(accel))
    smooth = a.square().sum(-1)
    hard_brake = F.relu(-a - 3.0).square().sum(-1)
    if entry_queue is None:
        entry_queue = state.get("entry_queue")
    if entry_queue is None:
        source_wait = torch.zeros_like(delay)
        fairness = _soft_cvar(delay_by_vehicle, present)
        count = present.sum(-1).clamp_min(1).to(delay.dtype)
    else:
        raw = torch.as_tensor(entry_queue, dtype=delay.dtype, device=delay.device)
        raw = raw.reshape(-1).expand(delay.shape[0])
        queue = raw.detach().clamp_min(0.0)
        source_wait = PHYSICS_DT_S * queue
        if raw.requires_grad:
            # Straight-through queue (SRCGRAD_NOTES.md): exact value, surrogate gradient, used in
            # source_wait only; the normalisation count and the CVaR weight stay detached (a
            # blocked vehicle moves between road and queue, it does not change their number).
            source_wait = source_wait + PHYSICS_DT_S * (raw - raw.detach())
        # Queued vehicles = one extra CVaR column of weight `queue` at delay dt (max possible).
        fairness = _soft_cvar(
            torch.cat((delay_by_vehicle, torch.full_like(delay[:, None], PHYSICS_DT_S)), -1),
            torch.cat((present, (queue > 0)[:, None]), -1),
            torch.cat((present.to(delay.dtype), queue[:, None]), -1))
        count = (present.sum(-1).to(delay.dtype) + queue).clamp_min(1)
    # Additive terms are per-vehicle means; fairness is already a per-vehicle tail delay
    # (CVaR of vehicle delays), so it is NOT divided by the count again - otherwise it would
    # be ~count-times weaker than the mean delay (integration review 2026-10-12).
    reward = ((-weights.delay * (delay + source_wait) + weights.throughput * throughput
               - weights.smooth * smooth - weights.hard_brake * hard_brake) / count
              - weights.fairness * fairness)
    discharge_weight = float(getattr(weights, "merge_discharge", 0.0))
    if discharge_weight != 0.0:
        # V7POL: un-normalised merge-end discharge (vehicles crossing MERGE_S1_M this step).
        reward = reward + discharge_weight * throughput
    free_weight = float(getattr(weights, "free_speed", 0.0))
    extra = {}
    if free_weight != 0.0:
        # V8POL: free-road CAV speed incentive (reward term only, never an action rule).
        shortfall = _free_speed_shortfall(state, next_state,
                                          float(getattr(weights, "free_speed_time_gap_s", 3.0)),
                                          float(getattr(weights, "free_speed_range_m", 100.0)))
        reward = reward - free_weight * shortfall
        extra["free_speed_shortfall"] = shortfall
    pairs = getattr(weights, "ttc_pairs", "all")
    return {"reward": reward, "cost_ttc": _ttc_cost(next_state, weights.ttc_limit_s, pairs),
            "delay": delay, "throughput": throughput, "smooth": smooth,
            "hard_brake": hard_brake, "fairness_delay": fairness, "source_wait": source_wait,
            **extra}


def episode_summary(steps: Sequence[Mapping[str, Tensor]]) -> dict[str, float]:
    """Detached Python scalars for logs, averaging parallel episodes in a batch.

    Additive metrics are summed over physics steps; fairness_delay is the mean
    of the per-step tail delays, *not* CVaR of lifetime vehicle delay. The latter
    requires vehicle IDs and accumulated trajectories outside this interface.
    Empty input returns zeros. ``duration_s`` excludes any warmup.
    """
    summary = {key: 0.0 for key in _KEYS}
    if steps:
        for key in _KEYS:
            if not all(key in step for step in steps):
                continue
            values = torch.stack([step[key].detach() for step in steps])
            reduced = values.mean() if key == "fairness_delay" else values.sum(0).mean()
            summary[key] = float(reduced.cpu())
    summary["n_steps"] = float(len(steps))
    summary["duration_s"] = len(steps) * PHYSICS_DT_S
    return summary
