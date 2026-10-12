"""Differentiable per-physics-step team reward and detached episode logging.

Slots must retain vehicle identity through a transition, including the final
position/speed of a vehicle that exits. Source queues contribute only if they
are represented as present, stationary slots; unrepresented waiting is unknown.
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
         "fairness_delay")
_CVAR_ALPHA = 0.9
_CVAR_TEMPERATURE_S = 0.01
_CROSSING_WIDTH_M = 1.0
_TTC_TEMPERATURE_S = 0.05
_CLOSING_EPS_MPS = 0.01


def _soft_cvar(delay: Tensor, present: Tensor) -> Tensor:
    """Bias-corrected, softplus Rockafellar--Uryasev upper-tail CVaR.

    Minimise eta + mean(tau*softplus((delay-eta)/tau))/(1-alpha).
    The sigmoid tail mass determines the optimum eta. Bisection is detached:
    the envelope theorem gives the derivative of the minimised objective
    directly through delay, without differentiating the numerical optimiser.
    Subtract its constant equal-delay entropy bias, so equal delays return
    exactly that delay (up to floating-point error), not a smoothing penalty.
    """
    if delay.shape[-1] == 0:
        return delay.sum(-1)
    count = present.sum(-1).clamp_min(1).to(delay.dtype)
    tau = _CVAR_TEMPERATURE_S
    tail = 1.0 - _CVAR_ALPHA
    with torch.no_grad():
        values = delay.detach()
        any_present = present.any(-1)
        lo = torch.where(present, values, torch.full_like(values, float("inf"))).amin(-1)
        hi = torch.where(present, values, torch.full_like(values, -float("inf"))).amax(-1)
        lo = torch.where(any_present, lo, torch.zeros_like(lo)) - 10.0 * tau
        hi = torch.where(any_present, hi, torch.zeros_like(hi)) + 10.0 * tau
        for _ in range(32):
            eta = 0.5 * (lo + hi)
            mass = torch.where(present, torch.sigmoid((values - eta[:, None]) / tau),
                               torch.zeros_like(values)).sum(-1) / count
            lo = torch.where(mass > tail, eta, lo)
            hi = torch.where(mass > tail, hi, eta)
        eta = 0.5 * (lo + hi)
    excess = tau * F.softplus((delay - eta[:, None]) / tau)
    excess = torch.where(present, excess, torch.zeros_like(excess)).sum(-1) / count
    entropy = -_CVAR_ALPHA * math.log(_CVAR_ALPHA) - tail * math.log(tail)
    cvar = eta + excess / tail - tau * entropy / tail
    return torch.where(present.any(-1), cvar.clamp_min(0.0), torch.zeros_like(cvar))


def _ttc_cost(state: Mapping[str, Tensor], limit_s: float) -> Tensor:
    """Soft relu(limit - TTC), summed over directed closing same-lane pairs.

    A softplus smooths the positive part in seconds. A small denominator floor
    avoids division by zero; non-closing pairs contribute exactly zero.
    Pair membership/order is discrete; positions/speeds within it have gradients.
    """
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
    gap = ds - length[:, None, :]
    closing = F.relu(v[:, :, None] - v[:, None, :])
    ttc = gap.clamp_min(0.0) / closing.clamp_min(_CLOSING_EPS_MPS)
    exposure = _TTC_TEMPERATURE_S * F.softplus((limit_s - ttc) / _TTC_TEMPERATURE_S)
    return torch.where(pair & (closing > 0), exposure, torch.zeros_like(exposure)).sum((-2, -1))


def team_reward(state: Mapping[str, Tensor], next_state: Mapping[str, Tensor],
                accel: Tensor, weights: RewardWeights) -> dict[str, Tensor]:
    """Return exactly the seven contract metrics, each a tensor of shape [B].

    The current state's present cohort is charged this step, including exits;
    delay uses *next* speeds: dt*relu(1-v_next/SPEED_LIMIT). Components are sums
    over that cohort, except fairness_delay (soft 90% CVaR in vehicle-seconds).
    Throughput is the positive increment in sigmoid((s-MERGE_S1)/1 metre).
    Smooth/hard_brake are sum(a**2) and sum(relu(-a-3)**2), for HDVs and CAVs.
    The complete weighted objective, including the scalar fairness penalty, is
    divided by max(present_count, 1). TTC is a separate, nonnegative constraint,
    evaluated on next-state occupants, never folded into the reward.
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
    fairness = _soft_cvar(delay_by_vehicle, present)
    count = present.sum(-1).clamp_min(1).to(delay.dtype)
    reward = (-weights.delay * delay + weights.throughput * throughput
              - weights.smooth * smooth - weights.hard_brake * hard_brake
              - weights.fairness * fairness) / count
    return {"reward": reward, "cost_ttc": _ttc_cost(next_state, weights.ttc_limit_s),
            "delay": delay, "throughput": throughput, "smooth": smooth,
            "hard_brake": hard_brake, "fairness_delay": fairness}


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
            values = torch.stack([step[key].detach() for step in steps])
            reduced = values.mean() if key == "fairness_delay" else values.sum(0).mean()
            summary[key] = float(reduced.cpu())
    summary["n_steps"] = float(len(steps))
    summary["duration_s"] = len(steps) * PHYSICS_DT_S
    return summary
