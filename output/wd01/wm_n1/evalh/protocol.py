"""Abstract predictor protocol and evaluation row types.

The WM-N1 model package (e1) is built in parallel with this harness, so the
harness never imports a concrete model.  It codes against the ``Predictor``
protocol below: any object exposing ``predict_quantiles`` and
``sample_rollouts`` is a valid model.  The trivial
:class:`evalh.climatology.ClimatologyPredictor` satisfies it for tests.

Row semantics follow the frozen label contract
(``features/FORMAT.md`` section 7): one outcome row per
``(world, t0, decision_t, tau, policy)`` with a realised delay ``y_delay_s``
and the safety counters.  Contrasts pair two policies on the SAME
``(world, t0, decision_t, tau)`` window.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

# The 19 quantile levels of the design's outcome heads (WM_N1_DESIGN.md
# section 2.4, line 85); shared by every predictor and by the comonotone
# pairing of the contrast metric.
QUANTILE_LEVELS = np.round(np.arange(1, 20) / 20.0, 2)   # 0.05 .. 0.95
N_QUANTILES = int(QUANTILE_LEVELS.size)

# Gating contrasts and taus, verbatim from REGISTRATION_WM_N1.md:82-84.
GATING_CONTRASTS = ("damping-native", "pi-native", "wall-native")
GATING_TAUS = (60.0, 120.0, 300.0)

# Short-horizon report points (REGISTRATION_WM_N1.md:87; design:195).
HORIZONS_S = (10.0, 30.0, 60.0, 120.0)


@dataclass
class OutcomeRow:
    """One realised (policy, tau) outcome on one decision window."""

    world: str
    t0_s: float
    decision_t_s: float
    tau_s: float
    policy: str
    y_delay_s: float
    collision: float = 0.0
    veh_steps: float = 0.0
    ttc_lt3_s: float = 0.0
    hard_brake_s: float = 0.0
    n_vehicles: float = 0.0
    n_arrivals: float = 0.0
    intervention: float = 0.0
    features: dict = field(default_factory=dict)

    @property
    def window(self) -> str:
        return f"{self.world}@{self.t0_s:g}+{self.decision_t_s:g}"


@dataclass
class RolloutRow:
    """One decision window with the TRUE short-horizon rollout.

    ``true_pos`` and ``true_speed`` are ``[H, V]`` over ``horizons_s`` and the
    vehicles present; ``true_dist`` / ``true_speed_scale`` are the per-window
    denominators the registration's "% of the true distance and speed" refers
    to (design:195-197).
    """

    world: str
    t0_s: float
    policy: str
    true_pos: np.ndarray            # [H, V]
    true_speed: np.ndarray          # [H, V]
    horizons_s: tuple = HORIZONS_S
    true_dist: float = 0.0
    true_speed_scale: float = 0.0
    veh_steps: float = 0.0
    collision: float = 0.0
    ttc_lt3_s: float = 0.0
    hard_brake_s: float = 0.0
    features: dict = field(default_factory=dict)


@dataclass
class RolloutBatch:
    """A predictor's sampled rollouts for a sequence of windows."""

    horizons_s: tuple
    pos: np.ndarray                 # [N, H, V] predicted downstream position
    speed: np.ndarray               # [N, H, V] predicted speed
    mask: Optional[np.ndarray] = None       # [N, H, V] 1 = valid vehicle
    collision: Optional[np.ndarray] = None  # [N] predicted collision count
    ttc_lt3_s: Optional[np.ndarray] = None  # [N] predicted TTC<3 exposure
    hard_brake_s: Optional[np.ndarray] = None  # [N] predicted hard-brake s
    intervention: Optional[np.ndarray] = None  # [N] validity projections


@runtime_checkable
class Predictor(Protocol):
    """The abstract model interface the harness evaluates.

    ``predict_quantiles`` is the outcome head (design section 2.4): it returns
    ``[len(rows), Q]`` quantiles of the delay ``V`` for each row's own
    ``(tau_s, policy)``.  ``sample_rollouts`` is the stochastic step decoder
    (design section 2.2), returning short-horizon trajectories for the
    short-horizon metrics.
    """

    name: str
    quantile_levels: np.ndarray

    def predict_quantiles(self, rows: Sequence[OutcomeRow]) -> np.ndarray:
        ...

    def sample_rollouts(self, rows: Sequence[RolloutRow],
                        horizons_s: tuple = HORIZONS_S) -> RolloutBatch:
        ...


class PredictorBase:
    """Convenience base: default quantile levels and helper validation."""

    name = "predictor"
    quantile_levels = QUANTILE_LEVELS

    def predict_quantiles(self, rows: Sequence[OutcomeRow]) -> np.ndarray:
        raise NotImplementedError

    def sample_rollouts(self, rows: Sequence[RolloutRow],
                        horizons_s: tuple = HORIZONS_S) -> RolloutBatch:
        raise NotImplementedError

    def _validate(self, q: np.ndarray, n_rows: int) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if q.shape != (n_rows, self.quantile_levels.size):
            raise ValueError(
                f"{self.name}: predict_quantiles returned {q.shape}, expected "
                f"({n_rows}, {self.quantile_levels.size})")
        return q
