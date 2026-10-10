"""R5/R6 SUMO-side schedule adapter (ADDENDUM_2 A2.2/A2.3).

The coordinated multi-CAV responses R5 and R6 live in ONE shared package,
``inc1_gen`` (gated by ``INC1_GENERATORS.json``, A2.6).  This module is the
SUMO call site: it renders the runner's :class:`~inc1.schedules.CavState`
records into the shared traci-like rows, asks the shared generator for the
per-tick commands, and wraps the result back into a
:class:`~inc1.schedules.ControlPlan` so the frozen ``CommandOwnership`` remains
the sole longitudinal writer (no private traci write, no lateral command).

Nothing here re-implements a rule: the speed/headway targets, the window and
the membership live only in ``inc1_gen.generators``.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from . import constants as C
    from .schedules import ControlPlan, ScheduleError, _validate_plan
except ImportError:  # script execution
    import constants as C  # type: ignore
    from schedules import (ControlPlan, ScheduleError,  # type: ignore
                           _validate_plan)

# the shared generator package lives beside ``inc1`` under ``output/wd01``
if str(C.WD01_ROOT) not in sys.path:
    sys.path.insert(0, str(C.WD01_ROOT))

from inc1_gen.adapters import states_from_traci_like  # noqa: E402
from inc1_gen.generators import make_generator  # noqa: E402
from inc1_gen.states import ObstacleRecord  # noqa: E402

SCRIPTED_ARMS = ("R5", "R6")


def to_traci_rows(states) -> dict:
    """Render ``CavState`` records into the shared SUMO-side row shape.

    The adapter is the only place call-site shapes are interpreted; the
    generator sees :class:`inc1_gen.states.VehState` records.
    """
    rows = {}
    for state in states:
        row = {
            "speed": float(state.speed_mps),
            "lane_position": float(state.s_down_m),
            "lane_index": int(state.lane_phys),
            "acceleration": float(getattr(state, "accel_mps2", 0.0) or 0.0),
            "type": "cav_passenger",
            "is_cav": True,
            "present": bool(getattr(state, "present", True)),
            "native_accel": getattr(state, "native_accel_mps2", None),
        }
        gap = getattr(state, "leader_gap_m", None)
        leader_v = getattr(state, "leader_v_mps", None)
        if gap is not None:
            row["leader"] = (float(gap),
                             None if leader_v is None else float(leader_v))
        rows[str(state.vehicle_id)] = row
    return rows


class ScriptedSchedule:
    """A ControlPlan schedule backed by one shared R5/R6 generator.

    Membership is latched at construction from the t0 ``states``; the
    feasibility verdict and the infeasibility reason are exposed for the
    receipt (``arm_feasibility``).  Every issued command is a longitudinal
    acceleration; ``native`` names the live CAVs released to native.
    """

    def __init__(self, arm: str, states, obstacle_s_down: float, *,
                 t0_s: float = C.T0_S, control_dt_s: float = C.CONTROL_DT_S):
        if arm not in SCRIPTED_ARMS:
            raise ScheduleError(f"{arm!r} is not a scripted arm")
        self.arm = arm
        self.obstacle_s_down = float(obstacle_s_down)
        self.t0_s = float(t0_s)
        self.control_dt_s = float(control_dt_s)
        obstacle = ObstacleRecord(
            obstacle_id="incident.obstacle", lane_phys=C.INCIDENT_LANE_PHYS,
            s_down=float(obstacle_s_down))
        self._generator = make_generator(
            arm, states_from_traci_like(to_traci_rows(states)), obstacle,
            t0_s=self.t0_s, control_dt_s=self.control_dt_s)
        self.members = tuple(getattr(self._generator, "_members", ()))
        self.feasible = bool(self._generator.feasible)
        self.infeasible_reason = self._generator.infeasible_reason
        self.events = self._generator.events

    def step(self, now_s: float, states) -> ControlPlan:
        step = self._generator.step(
            float(now_s), states_from_traci_like(to_traci_rows(states)))
        plan = ControlPlan(wall=dict(step.commands), native=tuple(step.native))
        _validate_plan(states, plan)
        return plan


def make_scripted_schedule(arm: str, states, obstacle_s_down: float, *,
                           t0_s: float = C.T0_S,
                           control_dt_s: float = C.CONTROL_DT_S
                           ) -> ScriptedSchedule:
    return ScriptedSchedule(arm, states, obstacle_s_down, t0_s=t0_s,
                            control_dt_s=control_dt_s)
