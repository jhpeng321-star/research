"""Pure contracts for the sequential timing-headroom pilot (Path B).

Common-horizon computation, rate normalization (frozen equations of
docs/IDM_SEQUENTIAL_TIMING_HEADROOM_PROTOCOL.md), action masks and the
retrospective timing oracle.  No TraCI calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

DEADBAND = 0.02
MINIMUM_DOSE_STEPS = 16
ORACLE_GAIN_GATE = 0.02
MEANINGFUL_SEED_MARGIN = 0.02
TIMING_ACTIONS = ("abstain", "activate_now", "defer_r2", "defer_r3")


@dataclass(frozen=True)
class TimingState:
    """One analyzable seed/cell state with its frozen schedule and baselines."""

    state_id: str
    seed: int
    penetration: float
    demand: int
    t1: int
    t2: int | None
    t3: int | None
    schedule_hash: str
    observations: dict[str, dict[str, Any]]  # per landmark: candidate id + hash
    schedule_identity: bool

    @property
    def horizon_steps(self) -> int:
        return common_horizon(self.t1, self.t2, self.t3)

    @property
    def masks(self) -> dict[str, bool]:
        return timing_masks(self.t1, self.t2, self.t3, self.observations)


def common_horizon(t1: int, t2: int | None, t3: int | None) -> int:
    """Frozen common terminal horizon H (controller steps), one per state."""
    if t3 is not None:
        return int(t3) + 60
    if t2 is not None:
        return int(t2) + 60
    return int(t1) + 60


def common_state_horizon(t1: int, t2: int | None, t3: int | None) -> int:
    """L = H - t1: the length of the identical [t1, H] measurement window."""
    return common_horizon(t1, t2, t3) - int(t1)


def timing_masks(t1: int, t2: int | None, t3: int | None, observations: Mapping[str, Any]) -> dict[str, bool]:
    """Frozen action feasibility from the native-IDM schedule."""
    def feasible(step: int | None, key: str) -> bool:
        if step is None:
            return False
        record = (observations or {}).get(key)
        if record is None or not bool(record.get("admissible", False)):
            return False
        return True

    return {
        "abstain": True,
        "activate_now": feasible(t1, "t1"),
        "defer_r2": feasible(t2, "t2"),
        "defer_r3": feasible(t3, "t3"),
    }


def rate_utility(
    *, severe_delta: float, energy_delta: float, horizon_steps: int, severe_ref: float, energy_ref: float
) -> float:
    """Frozen rate utility: paired reduction normalized by L and frozen refs."""
    if int(horizon_steps) <= 0 or severe_ref <= 0.0 or energy_ref <= 0.0:
        raise ValueError("rate utility requires a positive horizon and references")
    u_braking = severe_delta / (horizon_steps * severe_ref)
    u_energy = energy_delta / (horizon_steps * energy_ref)
    return float(0.5 * u_braking + 0.5 * u_energy)


def classify_timing(utility: float, constraints: Mapping[str, bool]) -> str:
    """Frozen classification using the fixed 0.02 deadband (never raised)."""
    if all(constraints.values()) and utility > DEADBAND:
        return "helpful"
    if (not all(constraints.values())) or utility < -DEADBAND:
        return "harmful"
    return "neutral"


def reward_timing(utility: float, constraints: Mapping[str, bool]) -> float:
    """Frozen paired option reward for the timing pilot (rate form)."""
    if not all(constraints.values()):
        return -2.0
    return float(__import__("math").tanh(utility))


def oracle_select(rows: list[Any], constrained: bool = True, dose_limited: bool = True) -> list[str]:
    """Constrained retrospective oracle over the four timing actions."""
    selected = []
    for row in rows:
        best, best_score = "abstain", 0.0
        for action in ("activate_now", "defer_r2", "defer_r3"):
            outcome = getattr(row, f"outcome_{action}", None)
            if outcome is None or not bool(getattr(outcome, "feasible", False)):
                continue
            if constrained and not bool(getattr(outcome, "constraints_pass", False)):
                continue
            if dose_limited and float(getattr(outcome, "dose_steps", 0)) < MINIMUM_DOSE_STEPS:
                continue
            score = float(getattr(outcome, "reward", 0.0))
            if score > best_score:
                best, best_score = action, score
        selected.append(best)
    return selected


def timing_gates(
    *,
    oracle_mean: float,
    fixed_best_mean: float,
    per_seed_gain: Mapping[int, float],
    options_used: Mapping[str, int],
    outcomes: Mapping[str, int],
    constraint_failures: int,
    loo_gains: Mapping[str, float] | None = None,
    cohort_diversity: Mapping[str, int] | None = None,
    replay_noise_ok: bool = True,
) -> dict[str, bool]:
    gain = float(oracle_mean - fixed_best_mean)
    meaningful = [seed for seed, delta in per_seed_gain.items() if delta >= MEANINGFUL_SEED_MARGIN]
    cohort = cohort_diversity or outcomes
    return {
        "improves_best_fixed_timing_rule_by_0p02": gain >= ORACLE_GAIN_GATE,
        "meaningful_gain_in_at_least_three_seeds": len(meaningful) >= 3,
        "more_than_one_activation_timing_selected": len(options_used) > 1,
        "no_single_seed_dominating": (
            loo_gains is None
            or all(float(value) >= 0.0 for value in loo_gains.values())
        ),
        "adequate_outcome_diversity": (
            int(cohort.get("helpful", 0)) >= 3
            and int(cohort.get("harmful", 0)) >= 3
            and int(outcomes.get("helpful", 0)) >= 3
        ),
        "hard_safety_service_compliance": constraint_failures == 0,
        "effects_exceed_replay_noise": bool(replay_noise_ok),
    }
