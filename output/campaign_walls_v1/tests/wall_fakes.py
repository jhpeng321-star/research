"""Shared WALL-PILOT fakes: api, frames, sessions, synthetic SOURCE plans.

ZERO SUMO and zero network: the dispatcher machinery is exercised either
through the REAL frozen MixedAssignmentSession (with a fake ownership
issuing the per-tick records) or through a recording FakeSession - the
same two-level pattern the WM-43 fixture and the v3 chunk tests used.
Single-process pytest (the CPU is busy with the running campaign).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace


OUTPUT_ROOT = Path(__file__).resolve().parent.parent.parent      # <repo>/output
REPO_ROOT = OUTPUT_ROOT.parent                                   # <repo>
for _root in (REPO_ROOT, OUTPUT_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from traffic_shepherd.sensing import SensorFrame  # noqa: E402

CAVS = ("cav.0", "cav.1", "cav.2", "cav.3")
MEMBERS = ("cav.1", "cav.3")          # lane 0 + lane 1 on the mainline edge
EDGE = "1191554636"
T0 = 300.0

#: the synthetic SOURCE availability mirrors the real chunk-2/3 plans:
#: per penetration block of 14 per chunk ->
#: p10 TRAIN 10 CAL 1 HOLD 3; p20 TRAIN 9 CAL 2 HOLD 3; p40 TRAIN 10 CAL 2 HOLD 2
SOURCE_SHAPE = {
    0.1: {"TRAIN": 10, "CALIBRATION": 1, "HOLDOUT": 3},
    0.2: {"TRAIN": 9, "CALIBRATION": 2, "HOLDOUT": 3},
    0.4: {"TRAIN": 10, "CALIBRATION": 2, "HOLDOUT": 2},
}


def make_frame(vid, speed, lane_index, distance_to_merge_m, lane_id, now):
    """A SensorFrame-like REPORT frame (report content only, no truth)."""
    return SensorFrame(
        vehicle_id=vid, timestamp_s=now, speed_mps=float(speed),
        acceleration_mps2=0.0, allowed_speed_mps=30.0, lane_index=int(lane_index),
        distance_to_merge_m=float(distance_to_merge_m), lane_id=str(lane_id))


def feasible_frames(now=T0, speeds=(20.0, 20.0), edge=EDGE):
    """Two live members on the SAME 2-lane mainline edge, different lanes,
    span 40 m (< 75) - a feasible pair nearest the merge."""
    return {
        MEMBERS[0]: make_frame(MEMBERS[0], speeds[0], 0, 120.0,
                               f"{edge}_0", now),
        MEMBERS[1]: make_frame(MEMBERS[1], speeds[1], 1, 80.0,
                               f"{edge}_1", now),
        # non-members: one on another edge, one single-lane on the edge
        "cav.0": make_frame("cav.0", 22.0, 0, 60.0, "777_other_0", now),
    }


class FakeApi:
    """Steps 0.5 s per simulationStep; teleport getters real-named;
    loadState rewinds time to the branch point and is counted."""

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
                return EDGE

            @staticmethod
            def getLaneID(v):
                return f"{EDGE}_0"

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
                # the real CommandOwnership configures 27 on FIRST
                # observation; from the next tick every CAV reports 27
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
        """The branch loop steps the api DIRECTLY (traci module shape)."""
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
        self._last = {}          # the real session telemetry reads this

    def step(self, frames, *, now_s, live_ids, native_ids=(),
             wall_commands=None):
        wall = dict(wall_commands or {})
        native = frozenset(native_ids)
        records = []
        for v in sorted(self.cav_ids & frozenset(live_ids)):
            if v in wall:
                owner, target = "wall", 20.0
            elif v in native:
                owner, target = "native", None
            else:
                owner, target = "damping", 20.0
            records.append(SimpleNamespace(
                vehicle_id=v, owner=owner, issued_target_speed_mps=target,
                fallback_reason=None))
        return records


class _SentinelController:
    """Identity anchor for the real MixedAssignmentSession owner check."""


class FakeSession:
    """Recording session: applies plans, emits per-tick dispatcher
    telemetry with wall fields once a wall plan is applied."""

    def __init__(self, cav_ids=CAVS):
        self.cav_ids = frozenset(cav_ids)
        self.circles_controller = None
        self.pi_session = None
        self.plans = []
        self.sink = None
        self.last_plan_id = None

    def bind_emit(self, sink):
        self.sink = sink

    def apply_plan(self, plan, *, now_s):
        self.plans.append({"now_s": now_s, "plan": plan})
        self.last_plan_id = plan.plan_id
        self.last_plan = plan

    def step(self, ownership, frames, *, now_s, live_ids):
        plan = self.plans[-1]["plan"] if self.plans else None
        wall_members, wall_id = set(), None
        if plan is not None and getattr(plan, "walls", ()):
            wall = plan.walls[0]
            wall_members, wall_id = set(wall.members), wall.wall_id
        records = []
        for v in sorted(self.cav_ids & frozenset(live_ids)):
            owner = "wall" if v in wall_members else "native"
            records.append(SimpleNamespace(
                vehicle_id=v, owner=owner, issued_target_speed_mps=20.0,
                fallback_reason=None))
        if self.sink is not None:
            for v in sorted(self.cav_ids):
                member = v in wall_members
                self.sink("mixed_assignment", {
                    "plan_id": self.last_plan_id,
                    "vehicle_id": v,
                    "action": "wall-v2" if member else "native",
                    "wall_id": wall_id if member else None,
                    "wall_phase": "assembly" if member else None,
                    "time_in_phase_s": 0.0 if member else None,
                    "wall_speed_mps": 12.0 if member else None,
                    "locked": member,
                    "controller_state": {"damping_target_speed_mps": None,
                                         "pi_running_target_mps": None}})
        return records


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


def base_saved_state(base_dir: Path, t0: float, cav_ids=CAVS) -> str:
    """A fake base <run>/saved_state/{state.xml, STATE_MANIFEST.json};
    returns the manifest digest-checked state sha."""
    state_dir = base_dir / "saved_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_xml = state_dir / "state.xml"
    state_xml.write_text(
        f'<snapshot version="1.27.1" time="{t0:.2f}"/></snapshot>',
        encoding="utf-8")
    state_sha = hashlib.sha256(state_xml.read_bytes()).hexdigest()
    manifest = {
        "schema": "wm41_saved_state/1",
        "branch_point_t0_s": float(t0),
        "sumo_version": [1, 27, 1],
        "sumo_state_sha256": state_sha,
        "harness_state": {"roster": {"cav_ids": sorted(cav_ids),
                                     "boundary_edges": {}}},
        "scenario_digests": {},
    }
    (state_dir / "STATE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return state_sha


def make_source_world(seed, pen, split, t0, bases_root: Path):
    """One synthetic SOURCE world (chunk-plan schema subset)."""
    name = f"w{seed}m2700p{int(pen * 100)}t{int(t0)}"
    base_run = f"{name}__base"
    base_saved_state(bases_root / base_run, t0, CAVS)
    return {
        "world": name, "seed": seed, "loading_veh_h": 2700,
        "penetration": pen, "split": split, "role": "exposed",
        "t0_s": float(t0),
        "base_arm": {"arm": "base", "run": base_run,
                     "saves_state": {"at_s": float(t0)}},
        "branches": [{"arm": "native", "run": f"{name}__native",
                      "mode": "restore_suffix", "native_sibling_run": f"{name}__native",
                      "restores_state": {"from_run": base_run,
                                         "at_s": float(t0)},
                      "branch_seed": 1}],
    }


def make_source_plan(first_seed: int, bases_root: Path) -> dict:
    """A synthetic SOURCE chunk plan with the REAL chunk availability:
    14 worlds per penetration (per-chunk split shape from SOURCE_SHAPE)."""
    worlds = []
    seed = first_seed
    for pen in (0.1, 0.2, 0.4):
        t0_values = [240.0, 300.0, 360.0, 420.0, 480.0, 540.0]
        for split, count in SOURCE_SHAPE[pen].items():
            for _ in range(count):
                t0 = t0_values[(seed - first_seed) % len(t0_values)]
                worlds.append(make_source_world(seed, pen, split, t0,
                                                bases_root))
                seed += 1
    return {"schema": "wm47_chunk2_world_plan/1", "plan_seed": 20261009,
            "worlds": worlds}


def make_source_plan_chunk3(first_seed: int, bases_root: Path) -> dict:
    plan = make_source_plan(first_seed, bases_root)
    plan["schema"] = "wm54_chunk3_world_plan/1"
    plan["plan_seed"] = 20261010
    return plan




def converging_frames(t0, end, feasible_from=None, wall_speed=17.0):
    """Scripted report frames for a member pair that is feasible at
    ``feasible_from`` and assembles two ticks later (speed within the
    tolerance, span 3 m <= 6) - what the REAL dispatcher needs to see."""
    feasible_from = t0 if feasible_from is None else feasible_from
    frames = {}
    for t in range(int(t0), int(end)):
        if t < feasible_from:
            frames[float(t)] = {
                m: make_frame(m, 20.0, 0, 400.0, "777_other_0", t)
                for m in MEMBERS}
            continue
        dt = t - feasible_from
        if dt == 0:
            speeds, d0, d1 = (20.0, 20.0), 120.0, 80.0
        elif dt == 1:
            speeds, d0, d1 = (18.5, 18.5), 112.0, 80.0
        else:
            speeds, d0, d1 = (wall_speed, wall_speed), 100.0, 97.0
        frames[float(t)] = {
            MEMBERS[0]: make_frame(MEMBERS[0], speeds[0], 0, d0,
                                   f"{EDGE}_0", t),
            MEMBERS[1]: make_frame(MEMBERS[1], speeds[1], 1, d1,
                                   f"{EDGE}_1", t)}
    return frames


def real_session(cav_ids=CAVS, owner=None):
    """The REAL frozen dispatcher session + the fake ownership (identity
    anchor satisfied; records issued per live designated CAV)."""
    from traffic_shepherd.mixed_assignment import MixedAssignmentSession

    owner = owner or FakeOwnership(cav_ids)
    session = MixedAssignmentSession(
        circles_controller=owner.controller, pi_session=None,
        cav_ids=sorted(cav_ids))
    return session, owner
