"""Fixed-clock team reward/credit primitives for the new damping controller.

These do not change the rejected old per-vehicle reward or certify a trainer.
One row is one global simulation interval, including intervals after CAV exit.
"""

from dataclasses import asdict, dataclass
from math import isfinite

import numpy as np


@dataclass(frozen=True)
class TeamCostScales:
    braking_vehicle_seconds: float
    acceleration_energy: float
    requested_population: int
    service_weight: float = 0.1
    hard_event_penalty: float = 2.0

    def __post_init__(self) -> None:
        if not all(isfinite(float(v)) for v in asdict(self).values()):
            raise ValueError("nonfinite reward reference")
        if (
            min(self.braking_vehicle_seconds, self.acceleration_energy, self.requested_population)
            <= 0
        ):
            raise ValueError("positive frozen reference scales are required")
        if self.requested_population != int(self.requested_population):
            raise ValueError("requested population must be an integer")
        if self.service_weight < 0 or self.hard_event_penalty <= 0:
            raise ValueError("invalid penalty")


@dataclass(frozen=True)
class TeamIntervalCost:
    braking_vehicle_seconds: float
    acceleration_energy: float
    requested_not_arrived_vehicle_seconds: float
    added_collision_events: int = 0
    added_teleport_events: int = 0

    def __post_init__(self) -> None:
        if not all(isfinite(float(v)) and v >= 0 for v in asdict(self).values()):
            raise ValueError("interval costs must be finite and nonnegative")
        for value in (self.added_collision_events, self.added_teleport_events):
            if value != int(value):
                raise ValueError("event counts must be integers")


def team_reward(cost: TeamIntervalCost, scales: TeamCostScales) -> dict[str, float]:
    """Return one auditable global reward, independent of live actor count.

    `requested_not_arrived` includes uninserted source queues. Reference scales
    must come from declared development baselines, not the active population.
    `added_*_events` means newly occurring in this interval, not cumulative
    events or a presumed paired-treatment difference. Safety/service remain
    explicit episode acceptance constraints, regardless of scalar reward.
    """
    parts = {
        "braking": -0.5 * cost.braking_vehicle_seconds / scales.braking_vehicle_seconds,
        "energy": -0.5 * cost.acceleration_energy / scales.acceleration_energy,
        "service": -scales.service_weight
        * cost.requested_not_arrived_vehicle_seconds
        / scales.requested_population,
        "hard_events": -scales.hard_event_penalty
        * (cost.added_collision_events + cost.added_teleport_events),
    }
    return {**parts, "reward": sum(parts.values())}


def team_advantages(
    rewards,
    values,
    episode_ends,
    *,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute global-time GAE before gathering any per-CAV actor samples.

    `values` has T+1 entries, including a bootstrap for an unfinished chunk.
    `episode_ends` marks only true team-objective ends, never vehicle exits.
    Defaults preserve the finite-horizon undiscounted objective.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    ends = np.asarray(episode_ends)
    if rewards.ndim != 1 or values.shape != (len(rewards) + 1,) or ends.shape != rewards.shape:
        raise ValueError("expected rewards/ends[T] and values[T+1]")
    if not np.isfinite(rewards).all() or not np.isfinite(values).all():
        raise ValueError("nonfinite trajectory")
    if not np.isin(ends, [False, True]).all():
        raise ValueError("episode_ends must be boolean")
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("invalid discount or GAE lambda")
    advantages = np.zeros_like(rewards)
    carry = 0.0
    for t in reversed(range(len(rewards))):
        continuation = 1.0 - float(ends[t])
        delta = rewards[t] + gamma * values[t + 1] * continuation - values[t]
        carry = delta + gamma * gae_lambda * continuation * carry
        advantages[t] = carry
    return advantages, advantages + values[:-1]


def gather_actor_advantages(advantages, active_mask, *, normalize: bool = True) -> np.ndarray:
    """Normalize once over active GLOBAL TIMES, then mask/pad actor slots.

    Adding empty slots or more actors to a time step cannot alter the scalar
    team advantage. This helper does not replace a time-major rollout buffer.
    """
    advantages = np.asarray(advantages, dtype=np.float64)
    mask = np.asarray(active_mask)
    if mask.ndim != 2 or advantages.shape != (mask.shape[0],):
        raise ValueError("expected advantages[T] and active_mask[T,N]")
    if not np.isfinite(advantages).all() or not np.isin(mask, [False, True]).all():
        raise ValueError("invalid advantages or actor mask")
    mask = mask.astype(bool)
    active_times = mask.any(axis=1)
    scaled = advantages.copy()
    if normalize and active_times.any():
        mean, std = scaled[active_times].mean(), scaled[active_times].std()
        # Constant delayed harm must not turn into all-zero learning credit.
        # Production normalization belongs across the collected team batch,
        # not separately inside every CAV trajectory or one option sample.
        if std > 1e-8:
            scaled = (scaled - mean) / std
    return np.where(mask, scaled[:, None], 0.0)
