"""Shared state interface for the scripted R5/R6 multi-CAV generators.

Addendum 2 (A2.1) of ``REGISTRATION_WM_N1_INC1_ADDENDUM_2.md``: R5 and R6 are
pure, deterministic, closed-loop scripted response generators used BOTH by the
SUMO arm (through ``CommandOwnership``, longitudinal commands only) and by the
WM rollout as the in-the-loop CAV mover.  This module defines the ONE shared
per-vehicle state record and the obstacle record both call sites render into.

Numpy-only package: no traci, no torch.  Nothing here reads an outcome, a
held-out split, or a live simulator.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VehState:
    """One vehicle's state at one control tick (the shared interface).

    ``s_down`` is the mainline downstream coordinate (merge start
    ``s_down = 0``); the distance to the obstacle is
    ``d = obstacle.s_down - s_down`` (A2.1).

    ``native_accel`` is the vehicle's native IDM acceleration at THIS tick
    (the released/native car-following acceleration the current call site
    computes).  ``None`` means unknown, and the generators then never
    accelerate the vehicle (cap 0.0), mirroring the registered R2 rule.

    ``leader_gap`` / ``leader_v`` describe the CURRENT same-lane leader
    (bumper gap in m, leader speed in m/s); ``None`` when the vehicle has no
    leader at this tick.

    ``present`` is False once the vehicle has left the network; members that
    are absent (or not present) at a later tick drop out of the response.
    """

    vehicle_id: str
    is_cav: bool
    lane_phys: int
    s_down: float
    v: float
    a: float
    leader_gap: float | None = None
    leader_v: float | None = None
    native_accel: float | None = None
    present: bool = True


@dataclass(frozen=True)
class ObstacleRecord:
    """The fixed stopped obstacle (INC1 registration section 3).

    It is a non-CAV pinned at constant ``s_down`` with ``v = 0`` and
    ``a = 0`` for the whole window; it is never a response member.
    """

    obstacle_id: str
    lane_phys: int
    s_down: float
    v: float = 0.0
    a: float = 0.0
    length_m: float = 5.0

    def distance_to(self, state: VehState) -> float:
        """``d = s_obs - s_down(CAV)`` (A2.1), positive when upstream."""
        return float(self.s_down) - float(state.s_down)


def distance_to_obstacle(state: VehState, obstacle_s_down: float) -> float:
    """Module-level form of :meth:`ObstacleRecord.distance_to`."""
    return float(obstacle_s_down) - float(state.s_down)
