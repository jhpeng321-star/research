"""Drive the real frozen mixed dispatcher stack on the fake api (no SUMO).

Wiring mirrors the chunk-1 branch driver's production components:
MixedAssignmentSession + PISession + CommandOwnership + SumoLocalSensor,
all from the research-dense tree (wdpaths.RESEARCH_DENSE_ROOT), stepping
per 1 s control tick with strict
full-roster plans (commit_horizon_s=0).
"""
from __future__ import annotations

import refload

refload.load_reference()

from traffic_shepherd.config import load_config  # noqa: E402
from traffic_shepherd.damping import CirclesController  # noqa: E402
from traffic_shepherd.mixed_assignment import AssignmentPlan, MixedAssignmentSession  # noqa: E402
from traffic_shepherd.ownership import CommandOwnership  # noqa: E402
from traffic_shepherd.pi_session import PISession  # noqa: E402
from traffic_shepherd.sensing import SumoLocalSensor  # noqa: E402


class ReferenceStack:
    def __init__(self, api, cav_ids):
        self.config = load_config()
        self.circles = CirclesController(self.config)
        self.pi_session = PISession(config=self.config, cav_ids=list(cav_ids),
                                    desired_speed_mps=31.29)
        self.session = MixedAssignmentSession(
            circles_controller=self.circles, pi_session=self.pi_session,
            cav_ids=set(cav_ids))
        self.roster = set(cav_ids)
        self.sensor = SumoLocalSensor(api, self.config)
        self.ownership = CommandOwnership(api, self.roster,
                                          controller=self.circles,
                                          cav_speed_mode=31)
        self.api = api

    def apply_plan(self, assignments: dict, plan_id: str, *, now_s: float):
        self.session.apply_plan(
            AssignmentPlan(dict(assignments), plan_id,
                           commit_horizon_s=0.0, strict_roster=True),
            now_s=now_s)

    def step(self, *, now_s: float):
        """Capture frames for live roster CAVs, step the session once."""
        live = sorted(set(self.api.vehicles) & self.roster)
        frames = {cav: self.sensor.capture(cav, now_s=now_s) for cav in live}
        records = self.session.step(self.ownership, frames, now_s=now_s,
                                    live_ids=sorted(self.api.vehicles))
        telemetry = (self.session.assignment_records[-len(live):]
                     if live else [])
        return frames, records, telemetry
