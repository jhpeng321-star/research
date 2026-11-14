"""INC1 R4 adapter tests (fake traci only): patched-driver path + feasibility.

Zero SUMO.  Uses the REAL ``MixedAssignmentSession`` against the PATCHED v2.2
``wall_group`` (``r4_driver.install_shims``) and a fake api/sensor/ownership,
mirroring the frozen-driver fixtures.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from inc1 import r4_driver as R4

# install the read-only v2.2 shims BEFORE importing the dispatcher so it binds
# the patched ``WallGroupSession``
R4.install_shims()

from traffic_shepherd.mixed_assignment import MixedAssignmentSession  # noqa: E402
from traffic_shepherd.sensing import SensorFrame  # noqa: E402

CAVS = ("cav.0", "cav.1", "cav.2", "cav.3")
MEMBERS = ("cav.1", "cav.3")
ENTRY_EDGE = "1191554636"
MERGE_EDGE = "1487606179"
EXIT_EDGE = "1487606178"
ENTRY_LEN_M = 2644.2
MERGE_END_S = ENTRY_LEN_M + 75.4
T0 = 300.0


def frame_at_s(vid, s_road, *, lane_index=0, speed=20.0, now=T0):
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


def entry_frames(distance_to_merge, *, speeds=(20.0, 20.0), now=T0, span=4.0):
    lead_s = ENTRY_LEN_M - float(distance_to_merge)
    return {
        MEMBERS[0]: frame_at_s(MEMBERS[0], lead_s, lane_index=0,
                               speed=speeds[0], now=now),
        MEMBERS[1]: frame_at_s(MEMBERS[1], lead_s - float(span), lane_index=1,
                               speed=speeds[1], now=now),
    }


def merge_crossing_frames(t0, end, *, wall_speed=17.0, lead_start_s=1000.0):
    frames = {}
    lead_s = float(lead_start_s)
    for t in range(int(t0), int(end)):
        now = float(t)
        if t == int(t0):
            pair = entry_frames(ENTRY_LEN_M - lead_s, speeds=(20.0, 20.0),
                                now=now)
        elif t == int(t0) + 1:
            pair = entry_frames(ENTRY_LEN_M - (lead_s + 20.0),
                                speeds=(18.5, 18.5), now=now)
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
    def __init__(self, t0=T0, live=MEMBERS):
        self.time = float(t0)
        self.t0 = float(t0)
        self.load_state_calls = 0
        self._live = tuple(live)
        api = self

        class _Vehicle:
            @staticmethod
            def getIDList():
                return sorted(api._live)

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

    def getVersion(self):
        return (22, "SUMO 1.27.1")

    def simulationStep(self):
        self.time += 0.5

    def close(self):
        pass


class FakeSensor:
    def __init__(self, frames_by_time):
        self.frames_by_time = dict(frames_by_time)

    def capture(self, cav, now_s=None):
        frames = self.frames_by_time.get(float(now_s), {})
        if cav not in frames:
            raise ValueError(f"no observation for {cav} at {now_s}")
        return frames[cav]


class _SentinelController:
    pass


class FakeOwnership:
    cav_speed_mode = 27

    def __init__(self, cav_ids=CAVS):
        self.cav_ids = frozenset(cav_ids)
        self.controller = _SentinelController()
        self._last = {}

    def step(self, frames, *, now_s, live_ids, native_ids=(), wall_commands=None):
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


def _stack():
    owner = FakeOwnership(CAVS)
    session = MixedAssignmentSession(
        circles_controller=owner.controller, pi_session=None,
        cav_ids=sorted(CAVS))
    return session, owner


def _world(name="w9601p10", seed=9601, warmup="w9601p10__warmup"):
    return {"world": name, "seed": seed, "warmup_run": warmup,
            "incident_s_down_m": 40.0}


def _obstacle_hook(calls):
    def hook(api, t_s):
        calls.append(float(t_s))
        return {"obstacle_id": "incident.obstacle", "inserted_at_s": float(t_s)}
    return hook


def test_r4_feasible_merge_release(tmp_path):
    world = _world()
    frames = merge_crossing_frames(T0, T0 + 120)
    session, owner = _stack()
    calls = []
    result = R4.run_r4_branch(
        FakeApi(), world, tmp_path, {"saved_state_digest": "d" * 64},
        session=session, ownership=owner, sensor=FakeSensor(frames),
        cav_ids=CAVS, obstacle_hook=_obstacle_hook(calls), hold_s=600.0)
    assert result["feasible"] is True
    assert result["infeasible_reason"] is None
    assert result["wall_outcome"] == "merge_released"
    assert result["wall_request_at_s"] == pytest.approx(T0)
    assert result["speed_factor"] == pytest.approx(0.85)
    assert result["branch_length_s_override"] == pytest.approx(300.0)
    assert result["horizon_s"] == pytest.approx(T0 + 300.0)
    assert calls == [T0]                      # obstacle inserted once at t0
    record = json.loads(
        (tmp_path / "branch_record.json").read_text(encoding="utf-8"))
    assert record["post_restore_hook"]["obstacle_id"] == "incident.obstacle"
    assert record["branch_type"] == R4.R4_ARM_NAME
    # the adapter wrote its own flagged result
    assert json.loads((tmp_path / "r4_result.json").read_text(
        encoding="utf-8"))["feasible"] is True


def test_r4_infeasible_proposer_refusal(tmp_path):
    world = _world()
    frames = {float(t): entry_frames(500.0, now=float(t))
              for t in range(int(T0), int(T0) + 130)}
    session, owner = _stack()
    result = R4.run_r4_branch(
        FakeApi(), world, tmp_path, {"saved_state_digest": "d" * 64},
        session=session, ownership=owner, sensor=FakeSensor(frames),
        cav_ids=CAVS, obstacle_hook=_obstacle_hook([]), hold_s=6.0)
    assert result["feasible"] is False
    assert result["wall_outcome"] == "not_feasible"
    assert result["infeasible_reason"].startswith("proposer_refusal:")
    assert "dmin_unreachable" in result["infeasible_reason"]
    # the run completed and the flagged infeasible result was written
    assert (tmp_path / "r4_result.json").is_file()


def test_r4_infeasible_no_qualifying_pair(tmp_path):
    world = _world()
    session, owner = _stack()
    result = R4.run_r4_branch(
        FakeApi(live=CAVS), world, tmp_path, {"saved_state_digest": "d" * 64},
        session=session, ownership=owner, sensor=_EmptySensor(), cav_ids=CAVS,
        obstacle_hook=_obstacle_hook([]), hold_s=6.0)
    assert result["feasible"] is False
    assert result["wall_outcome"] == "not_feasible"


class _EmptySensor:
    def capture(self, cav, now_s=None):
        from traffic_shepherd.sensing import SensorFrame
        return SensorFrame(
            vehicle_id=cav, timestamp_s=float(now_s), speed_mps=20.0,
            acceleration_mps2=0.0, allowed_speed_mps=30.0, lane_index=0,
            distance_to_merge_m=0.0, lane_id="1487606179_0", position_m=0.0)


def test_classify_branch_aborted_assembly_is_infeasible(tmp_path):
    feasible, reason = R4.classify_branch(
        {"wall_outcome": "aborted_in_assembly", "abort_reason": "span",
         "assembled_at_s": None}, tmp_path)
    assert feasible is False
    assert reason == "assembly_not_completed:span"
    feasible, reason = R4.classify_branch(
        {"wall_outcome": "aborted_in_hold", "abort_reason": "x",
         "assembled_at_s": T0 + 2}, tmp_path)
    assert feasible is True and reason is None
