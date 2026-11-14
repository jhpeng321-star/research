"""Mocked TraCI API + scripted damping controller for the WM-43 loop tests.

The fake implements exactly the TraCI surface the mechanics loop, the
ownership path and the sensor consume: speedMode31 validation, slowDown /
setSpeed command semantics (a later command overrides), 0.5 s physics
integration on an endless two-lane mainline (vehicles never leave), empty
collision/teleport channels and the 200 m leader query for TTC exposure.

No SUMO is started; the REAL CommandOwnership, MixedAssignmentSession,
wall pacing (direct_target) and SumoLocalSensor run against this fake, so
the loop test is an integration test of the dispatcher's physical path.
"""
from __future__ import annotations


from traffic_shepherd.damping import DampingDecision

MAINLINE_EDGE = "1191554636"
LANE_LENGTH_M = 2644.22
LANE_LIMIT_MPS = 31.29
VEHICLE_LENGTH_M = 5.0
MIN_GAP_M = 2.5


class FakeVehicle:
    def __init__(self, vehicle_id: str, lane_index: int, position_m: float,
                 speed_mps: float, insert_at_s: float = 0.0):
        self.id = vehicle_id
        self.lane_index = lane_index
        self.position_m = position_m
        self.speed_mps = speed_mps
        self.insert_at_s = insert_at_s
        self._previous_speed = speed_mps
        self._acceleration = 0.0
        self._pin: float | None = None
        self._ramp: tuple[float, float, float, float] | None = None  # v0, t0, dur, target

    def inserted(self) -> bool:
        """Real-SUMO insertion semantics: a depart=t vehicle appears in
        TraCI only from the first step at/after t, never before."""
        return _CLOCK[0] >= self.insert_at_s - 1e-9

    # -- command surface (real ownership semantics) ----------------------
    def slow_down(self, target_mps: float, duration_s: float) -> None:
        self._pin = None
        self._ramp = (self.speed_mps, _CLOCK[0], duration_s, float(target_mps))

    def set_speed(self, speed_mps: float) -> None:
        # Real SUMO semantics: setSpeed(v, -1) releases the vehicle to its
        # automatic (driver) speed control.  The fake resumes constant speed.
        if speed_mps < 0:
            self._pin = None
            self._ramp = None
        else:
            self._pin = float(speed_mps)
            self._ramp = None

    # -- physics ----------------------------------------------------------
    def step(self, dt_s: float, now_s: float) -> None:
        if self._pin is not None:
            new_speed = self._pin
        elif self._ramp is not None:
            v0, t0, dur, target = self._ramp
            fraction = min(1.0, max(0.0, (now_s - t0) / dur)) if dur > 0 else 1.0
            new_speed = v0 + (target - v0) * fraction
            if fraction >= 1.0:
                self._ramp = None
        else:
            new_speed = self.speed_mps
        self._acceleration = (new_speed - self.speed_mps) / dt_s
        self.speed_mps = new_speed
        self.position_m += new_speed * dt_s


_CLOCK = [0.0]


class FakeVehicles:
    def __init__(self, world: dict[str, FakeVehicle]):
        self.world = world

    def getIDList(self):
        return sorted(vid for vid, v in self.world.items() if v.inserted())

    def getSpeedMode(self, vehicle_id):
        return 31

    def getLaneID(self, vehicle_id):
        return f"{MAINLINE_EDGE}_{self.world[vehicle_id].lane_index}"

    def getLaneIndex(self, vehicle_id):
        return self.world[vehicle_id].lane_index

    def getLanePosition(self, vehicle_id):
        return self.world[vehicle_id].position_m

    def getLength(self, vehicle_id):
        return VEHICLE_LENGTH_M

    def getSpeed(self, vehicle_id):
        return self.world[vehicle_id].speed_mps

    def getAcceleration(self, vehicle_id):
        return self.world[vehicle_id]._acceleration

    def getAllowedSpeed(self, vehicle_id):
        return LANE_LIMIT_MPS

    def getMinGap(self, vehicle_id):
        return MIN_GAP_M

    def getLeader(self, vehicle_id, dist=100.0):
        """Real traci.vehicle.getLeader(vehID, dist=100.) shape: (id, gap) where
        gap excludes the vehicle's minGap, or None under the legacy default
        (setLegacyGetLeader(True), which is the 1.27.1 behavior)."""
        ego = self.world[vehicle_id]
        best = None
        for other in self.world.values():
            if other.id == vehicle_id or other.lane_index != ego.lane_index:
                continue
            if not other.inserted():
                continue
            gap = other.position_m - VEHICLE_LENGTH_M - ego.position_m
            if gap >= 0 and (best is None or gap < best[1]):
                best = (other.id, gap)
        return best

    def slowDown(self, vehicle_id, target_mps, duration_s):
        self.world[vehicle_id].slow_down(target_mps, duration_s)

    def setSpeed(self, vehicle_id, speed_mps):
        self.world[vehicle_id].set_speed(speed_mps)


class FakeLane:
    def __init__(self, world: dict[str, FakeVehicle]):
        self.world = world

    def getLength(self, lane_id):
        return LANE_LENGTH_M

    def getLastStepVehicleIDs(self, lane_id):
        return sorted(v.id for v in self.world.values()
                      if v.inserted()
                      and f"{MAINLINE_EDGE}_{v.lane_index}" == lane_id)


class FakeEdge:
    def getLaneNumber(self, edge_id):
        return {MAINLINE_EDGE: 2, "1487606179": 3, "1487606178": 2, "11553048#1": 1}.get(
            edge_id, 1)


class FakeSimulation:
    """Shapes mirror the installed traci.simulation domain (SUMO 1.27.1):
    getTime/getArrivedIDList/getCollidingVehiclesIDList/getStartingTeleportNumber/
    getIDList/getDepartedIDList/step.  NOTE: the real domain has NO
    simulationStep and NO getVersion - stepping is top-level
    traci.simulationStep(step=0) and the version is top-level
    traci.getVersion() -> (22, 'SUMO 1.27.1')."""

    def __init__(self, world):
        self.world = world

    def getTime(self):
        return _CLOCK[0]

    def step(self, step=0):
        _CLOCK[0] += 0.5
        for vehicle in self.world.values():
            vehicle.step(0.5, _CLOCK[0])

    def getIDList(self):
        return sorted(vid for vid, v in self.world.items() if v.inserted())

    def getDepartedIDList(self):
        return []

    def getArrivedIDList(self):
        return []

    def getCollidingVehiclesIDList(self):
        return []

    def getStartingTeleportNumber(self):
        return 0


class FakeSumo:
    """Injectable ``api`` for run_mechanics; freshly clocked per instance.

    The lifecycle/version surface mirrors the installed module-level API:
    start(cmd, port=...), isLoaded(), close(wait=True), simulationStep(step=0)
    and getVersion() -> (22, 'SUMO 1.27.1')-shaped (int, str) tuple."""

    VERSION = (22, "SUMO 1.27.1 (wm43-fake: installed-API shape)")

    def isLoaded(self):
        return getattr(self, "_loaded", False)

    def __init__(self, roster, insertion_delay_s: float = 0.0):
        _CLOCK[0] = 0.0
        self._loaded = False
        self.started_commands: list = []
        self.insertion_delay_s = float(insertion_delay_s)
        self.world: dict[str, FakeVehicle] = {}
        for entry in roster:
            self.world[entry["vehicle_id"]] = FakeVehicle(
                entry["vehicle_id"], int(entry["lane_index"]),
                float(entry["depart_pos_m"]), float(entry["depart_speed_mps"]),
                insert_at_s=float(insertion_delay_s))
        self.vehicle = FakeVehicles(self.world)
        self.lane = FakeLane(self.world)
        self.edge = FakeEdge()
        self.simulation = FakeSimulation(self.world)

    def start(self, command, port=None, numRetries=60, label="wm43-fake",
              verbose=False, **kwargs):
        # Real traci.start(cmd, port=None, numRetries=60, label=...): the
        # fixture loop always calls api.start(command, port=port).
        self._loaded = True
        self.started_commands.append(list(command))

    def close(self, wait=True):
        self._loaded = False

    def simulationStep(self, step=0):
        self.simulation.step(step)

    def getVersion(self):
        return self.VERSION


class ScriptedDampingController:
    """Comfort-follow damping stand-in for the A2 follower path.

    a = clip(0.5*(v_leader - v) + 0.05*(gap - 30), -3, +1.5) with the free-
    road cruise target 8 m/s; stateless per decide like the frozen
    controller contract requires.
    """

    def __init__(self, config):
        self.config = config

    def decide(self, frame, *, now_s: float) -> DampingDecision:
        if frame is None:
            return DampingDecision(fallback_reason="missing_observation")
        if frame.front is not None:
            raw = 0.5 * frame.front.relative_speed_mps + 0.05 * (frame.front.gap_m - 30.0)
        else:
            raw = 8.0 - frame.speed_mps
        a = min(max(raw, -3.0), 1.5)
        return DampingDecision(command_acceleration_mps2=a,
                               target_speed_mps=max(0.0, frame.speed_mps + a))


def build_fake_session(run_dir, world, controller, config,
                       insertion_delay_s: float = 0.0):
    """Build the FakeSumo for a run's roster (derive only; no files written).

    ``insertion_delay_s`` reproduces the real-SUMO behavior the r2 mock
    missed: depart=0 vehicles become TraCI-visible only from the first
    step at/after depart, so the loop's FIRST tick observes an empty
    network (the r2 smoke aborted its wall with member_frame_missing at
    t=0.0 for exactly this reason).
    """
    from dataclasses import asdict
    from wm43_scenario import derive_roster
    roster = derive_roster(world["arm"], int(world["seed"]))
    return (FakeSumo([asdict(v) for v in roster],
                     insertion_delay_s=insertion_delay_s), roster)
