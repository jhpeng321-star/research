"""Shared v2.2 fakes: api, frames, sessions (zero SUMO, zero network).

Same two-level pattern as the WM-43 fixture and the road_end_fix tests:
the REAL frozen ``MixedAssignmentSession`` (+ the patched v2.2
``WallGroupSession`` via the shim) with fake api/ownership/sensor
issuing scripted REPORT frames.  Single process, no bytecode.
"""
from __future__ import annotations

from types import SimpleNamespace

from traffic_shepherd.sensing import SensorFrame

CAVS = ("cav.0", "cav.1", "cav.2", "cav.3")
MEMBERS = ("cav.1", "cav.3")          # lane 0 + lane 1 on the entry edge
ENTRY_EDGE = "1191554636"
MERGE_EDGE = "1487606179"
EXIT_EDGE = "1487606178"
ENTRY_LEN_M = 2644.2
MERGE_LEN_M = 75.4
EXIT_LEN_M = 477.9
MERGE_START_S = ENTRY_LEN_M            # 2644.2
MERGE_END_S = MERGE_START_S + MERGE_LEN_M
NETWORK_END_S = MERGE_END_S + EXIT_LEN_M
T0 = 300.0


def frame_at_s(vid, s_road, *, lane_index=0, speed=20.0, now=T0):
    """A SensorFrame-like REPORT frame locating the vehicle at the
    cumulative downstream coordinate ``s_road`` (0 = entry edge origin):
    edge + lane-local position + the merge-distance coordinate, exactly
    what the edge-aware ``s_road`` reads."""
    if s_road <= ENTRY_LEN_M:
        edge, position = ENTRY_EDGE, float(s_road)
    elif s_road <= MERGE_END_S:
        edge, position = MERGE_EDGE, float(s_road) - ENTRY_LEN_M
    else:
        edge, position = EXIT_EDGE, float(s_road) - MERGE_END_S
    return SensorFrame(
        vehicle_id=vid, timestamp_s=float(now), speed_mps=float(speed),
        acceleration_mps2=0.0, allowed_speed_mps=30.0,
        lane_index=int(lane_index), distance_to_merge_m=ENTRY_LEN_M - float(s_road),
        lane_id=f"{edge}_{int(lane_index)}", position_m=float(position))


def entry_frames(distance_to_merge, *, speeds=(20.0, 20.0), now=T0,
                 span=4.0):
    """A member pair on the entry edge, one per lane, ``span`` m apart,
    whose LEAD is ``distance_to_merge`` upstream of the merge start."""
    lead_s = ENTRY_LEN_M - float(distance_to_merge)
    return {
        MEMBERS[0]: frame_at_s(MEMBERS[0], lead_s, lane_index=0,
                               speed=speeds[0], now=now),
        MEMBERS[1]: frame_at_s(MEMBERS[1], lead_s - float(span), lane_index=1,
                               speed=speeds[1], now=now),
    }


def converging_frames(t0, end, *, feasible_from=None, wall_speed=17.0,
                      start_distance=2000.0):
    """Scripted frames: pair feasible at ``feasible_from``, assembly
    completes two ticks later (speed within tolerance, span 3 m <= 6),
    then the wall cruises DOWNSTREAM at ``wall_speed`` (s_road grows)."""
    feasible_from = t0 if feasible_from is None else feasible_from
    frames = {}
    lead_s = ENTRY_LEN_M - float(start_distance)
    for t in range(int(t0), int(end)):
        now = float(t)
        if t < feasible_from:
            frames[now] = {
                m: frame_at_s(m, 100.0, lane_index=0, speed=20.0, now=now)
                for m in MEMBERS}
            continue
        dt = t - feasible_from
        if dt == 0:
            pair = entry_frames(start_distance, speeds=(20.0, 20.0), now=now)
        elif dt == 1:
            pair = entry_frames(start_distance - 20.0, speeds=(18.5, 18.5),
                                now=now)
        else:
            lead_s += float(wall_speed)
            pair = {
                MEMBERS[0]: frame_at_s(MEMBERS[0], lead_s, lane_index=0,
                                       speed=wall_speed, now=now),
                MEMBERS[1]: frame_at_s(MEMBERS[1], lead_s - 3.0, lane_index=1,
                                       speed=wall_speed, now=now)}
        frames[now] = pair
    return frames


class FakeApi:
    """Steps 0.5 s per simulationStep; real-named getters; loadState
    counted (never called by the driver-level tests)."""

    def __init__(self, t0=T0, live=MEMBERS):
        self.time = float(t0)
        self.t0 = float(t0)
        self.load_state_calls = 0
        self._live = tuple(live)
        self._wire()

    def _wire(self):
        api = self
        live = self._live

        class _Vehicle:
            @staticmethod
            def getIDList():
                return sorted(live)

            @staticmethod
            def getRoadID(v):
                return ENTRY_EDGE

            @staticmethod
            def getLaneID(v):
                return f"{ENTRY_EDGE}_0"

            @staticmethod
            def getLanePosition(v):
                return 100.0

            @staticmethod
            def getSpeed(v):
                return 20.0

            @staticmethod
            def getAcceleration(v):
                return 0.0

            @staticmethod
            def getTypeID(v):
                return "cautious_car"

            @staticmethod
            def getSpeedMode(v):
                return 27 if api.time > api.t0 + 1e-9 else 31

            @staticmethod
            def slowDown(v, s, d):
                pass

            @staticmethod
            def setSpeed(v, s):
                pass

            @staticmethod
            def setSpeedMode(v, m):
                pass

        class _Simulation:
            @staticmethod
            def getTime():
                return api.time

            @staticmethod
            def simulationStep():
                api.time += 0.5

            @staticmethod
            def loadState(path):
                api.load_state_calls += 1
                api.time = api.t0

            @staticmethod
            def getDepartedIDList():
                return []

            @staticmethod
            def getArrivedIDList():
                return []

            @staticmethod
            def getCollidingVehiclesIDList():
                return []

            @staticmethod
            def getStartingTeleportIDList():
                return ()

            @staticmethod
            def getEndingTeleportIDList():
                return ()

        api.vehicle = _Vehicle()
        api.simulation = _Simulation()

    def simulationStep(self):
        self.time += 0.5

    def getVersion(self):  # noqa: N802 - REAL traci shape
        return (22, "SUMO 1.27.1")

    def close(self):
        pass


class FakeSensor:
    """Scripted report frames per control tick (time -> {cav: frame})."""

    def __init__(self, frames_by_time: dict):
        self.frames_by_time = dict(frames_by_time)

    def capture(self, cav, now_s=None):
        frames = self.frames_by_time.get(float(now_s), {})
        if cav not in frames:
            raise ValueError(f"no observation for {cav} at {now_s}")
        return frames[cav]


class FakeOwnership:
    """Issues one record per live designated CAV (native/wall/damping)."""

    cav_speed_mode = 27

    def __init__(self, cav_ids=CAVS):
        self.cav_ids = frozenset(cav_ids)
        self.controller = _SentinelController()
        self._last = {}

    def step(self, frames, *, now_s, live_ids, native_ids=(),
             wall_commands=None):
        wall = dict(wall_commands or {})
        records = []
        for v in sorted(self.cav_ids & frozenset(live_ids)):
            if v in wall:
                owner, target = "wall", 20.0
            else:
                owner, target = "native", None
            records.append(SimpleNamespace(
                vehicle_id=v, owner=owner, issued_target_speed_mps=target,
                fallback_reason=None))
        return records


class _SentinelController:
    """Identity anchor for the real MixedAssignmentSession owner check."""


def real_session(cav_ids=CAVS, owner=None):
    """The REAL frozen dispatcher session (which drives the PATCHED v2.2
    WallGroupSession through the shim) + the fake ownership."""
    from traffic_shepherd.mixed_assignment import MixedAssignmentSession

    owner = owner or FakeOwnership(cav_ids)
    session = MixedAssignmentSession(
        circles_controller=owner.controller, pi_session=None,
        cav_ids=sorted(cav_ids))
    return session, owner


def restore_entry(cav_ids=CAVS, t0=T0):
    """A minimal restored-state dict (the driver reads roster + t0)."""
    return {"manifest": {"harness_state": {"roster": {
        "cav_ids": sorted(cav_ids), "boundary_edges": {}}},
        "physics_row_at_t0": None},
        "saved_state_id": "d" * 64}


def wall_arm(run_name="w300p10__w-mid-early", t0=T0, hold_s=6.0,
             arm="W-mid-early"):
    return {"arm": arm, "run": run_name, "mode": "restore_wall",
            "native_sibling_run": f"{run_name.rsplit('__', 1)[0]}__native",
            "branch_seed": 123456789,
            "restores_state": {"from_run": "w__base", "at_s": t0},
            "wall": {"speed_factor": 0.85, "timing": "early",
                     "target_time_s": t0, "request_window_s": 120.0,
                     "deadline_s": t0 + 120.0, "speed_floor_mps": 2.5,
                     "hold_s": hold_s}}
