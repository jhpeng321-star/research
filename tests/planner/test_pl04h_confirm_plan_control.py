"""PL-04h REAL-CONTRACT test: the confirm smoke's plan branch commands.

The confirm_1 INERT-branch regression, pinned as a real contract: the plan
branch's control write (:meth:`ConfirmLauncher.plan_control`) must
materialize a REAL ``mixed_assignment/3`` ``AssignmentPlan`` (damping for
every roster CAV; ``strict_roster=True``, ``commit_horizon_s=0``) through
the planner's ONE shared conversion and commit it via
``MixedAssignmentSession.apply_plan`` BEFORE the first control tick --
otherwise the session constructs all-native and every command row falls
back (240/240 in confirm_1).

Everything downstream of the plan application is REAL: the REAL
``MixedAssignmentSession``, the REAL ``CommandOwnership``, the REAL
``CirclesController`` (the adopted adapter config + ONNX model), the REAL
``SumoLocalSensor`` and the REAL ``record_suffix`` tick loop. Only traci
(the connection surface) and the lane->merge-distance observation map are
faked; zero SUMO, zero launches. The roster mirrors the confirm_1 evidence
pair: ``mainline.6`` with a covered leader (35 m, inside the 100 m sensor
range) and ``mainline.10`` with no leader within coverage (it fell back on
every tick of the real run).
"""

import importlib.util
import sys
from pathlib import Path

PLANNER_ROOT = Path(__file__).resolve().parents[2]


def _load_wrapper(name, relative):
    spec = importlib.util.spec_from_file_location(name, PLANNER_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_confirm = _load_wrapper(
    "pl04h_run_confirm", "output/planner_v1/resmoke/run_confirm.py"
)

T0_S = 30.0
LANE_ID = "E0_0"
LANE_LENGTH_M = 500.0
#: positions on the lane: the covered-leader pair 35 m apart inside the
#: 100 m sensor range, and an uncovered CAV 160 m behind the pair
POSITIONS = {"mainline.6": 100.0, "hdv.0": 140.0, "mainline.10": 300.0}
ROSTER = ("mainline.10", "mainline.6")


class _ConfirmFakeConn:
    """A fake traci connection with the surface plan_control touches:
    0.5 s physics steps from t0, speedMode 31, recorded slowDown/setSpeed
    writes, and the same-lane neighbor surface the REAL SumoLocalSensor
    reads (every vehicle at a fixed position, speed 20)."""

    def __init__(self, t0_s: float = T0_S):
        self.time = float(t0_s)
        self.slowdowns = []  # (vehicle_id, target_speed_mps, duration_s, conn_time)
        self.setspeeds = []  # (vehicle_id, speed, conn_time)
        self.simulation = self._Sim(self)
        self.vehicle = self._Vehicle(self)
        self.lane = self._Lane(self)

    def simulationStep(self):
        self.simulation.simulationStep()  # the real Connection's top-level step

    class _Sim:
        def __init__(self, conn):
            self._conn = conn

        def getTime(self):
            return self._conn.time

        def simulationStep(self):
            self._conn.time = round(self._conn.time + 0.5, 10)

        def getArrivedIDList(self):
            return []  # T-B: the control write reads arrivals (run_suffix lifecycle)

        def getStartingTeleportIDList(self):
            return ()

        def getCollidingVehiclesIDList(self):
            return ()

    class _Vehicle:
        def __init__(self, conn):
            self._conn = conn

        def getIDList(self):
            return tuple(sorted(POSITIONS))

        def getSpeedMode(self, vid):
            return 31

        def slowDown(self, vid, target_speed_mps, duration_s):
            self._conn.slowdowns.append(
                (str(vid), float(target_speed_mps), float(duration_s), self._conn.time)
            )

        def setSpeed(self, vid, speed):
            self._conn.setspeeds.append((str(vid), float(speed), self._conn.time))

        def getLaneID(self, vid):
            return LANE_ID

        def getLanePosition(self, vid):
            return POSITIONS[str(vid)]

        def getLength(self, vid):
            return 5.0

        def getSpeed(self, vid):
            return 20.0

        def getAcceleration(self, vid):
            return 0.0

        def getAllowedSpeed(self, vid):
            return 30.0

        def getLaneIndex(self, vid):
            return 0

    class _Lane:
        def __init__(self, conn):
            self._conn = conn

        def getLength(self, lane_id):
            assert lane_id == LANE_ID
            return LANE_LENGTH_M

        def getLastStepVehicleIDs(self, lane_id):
            assert lane_id == LANE_ID
            return tuple(sorted(POSITIONS))


def _plan_control(conn, tmp_path):
    launcher = run_confirm.ConfirmLauncher(
        sumo_binary="<unused: plan_control never launches>",
        cfg="<unused>",
        seed=8468,
        diag_dir=tmp_path,
    )
    return launcher.plan_control(conn, ROSTER)


def test_plan_branch_issues_nonfallback_damping_to_a_covered_leader(
    tmp_path, monkeypatch
):
    """THE PL-04h gate: over the REAL session/ownership/controller/sensor, the
    plan branch commits its damping plan BEFORE the first control tick and
    issues >= 1 NON-fallback damping command to the covered-leader roster
    CAV (mainline.6), while the uncovered CAV (mainline.10) is visible in
    the fallback histogram."""
    monkeypatch.setattr(
        "traffic_shepherd.scenarios.observation_map", lambda api, config=None: {}
    )
    conn = _ConfirmFakeConn()
    control = _plan_control(conn, tmp_path)

    outcome = run_confirm.record_suffix(conn, 12.0, control=control)
    del outcome
    stats = control.stats
    # 12 control ticks x 2 roster CAVs, every row accounted for
    assert stats["control_ticks"] == 12
    assert stats["command_rows"] == 24
    assert (
        stats.get("non_fallback_rows", 0) + stats.get("fallbacks", 0)
        == stats["command_rows"]
    )
    # THE REGRESSION: at least one NON-fallback damping command was issued
    assert stats.get("non_fallback_rows", 0) >= 1
    assert stats["rows_by_owner"].get("damping", 0) == stats["non_fallback_rows"]
    assert stats["fallback_reason_histogram"].get("(none)", 0) == (
        stats["non_fallback_rows"]
    )
    # the applied plan committed DAMPING for the whole roster from the first
    # tick (rows_by_mode is finalized from the session's own assignment rows)
    assert stats["rows_by_mode"] == {"damping": 24}
    # the real write: only a non-fallback damping decision calls slowDown;
    # the covered leader was written FROM THE FIRST control tick (the plan
    # was applied BEFORE it), the uncovered CAV never was
    mainline6 = [(t, target) for v, target, _d, t in conn.slowdowns if v == "mainline.6"]
    mainline10 = [t for v, _target, _d, t in conn.slowdowns if v == "mainline.10"]
    assert len(mainline6) >= 1
    assert mainline6[0][0] == T0_S
    assert mainline10 == []
    # the uncovered CAV's every-tick fallback is diagnosable
    assert stats["fallback_reason_histogram"].get("missing_or_uncovered_leader", 0) >= 1
    # the finally-release handed every owned CAV back (setSpeed(-1))
    released = {v for v, speed, _t in conn.setspeeds if speed < 0}
    assert released == {"mainline.6"}


def test_plan_control_stats_carry_the_branch_diagnostics_shape(tmp_path, monkeypatch):
    """The stats the verdict embeds carry the PL-04h diagnostic keys (owner /
    mode counts + fallback histogram) with the legacy keys intact."""
    monkeypatch.setattr(
        "traffic_shepherd.scenarios.observation_map", lambda api, config=None: {}
    )
    conn = _ConfirmFakeConn()
    control = _plan_control(conn, tmp_path)
    run_confirm.record_suffix(conn, 2.0, control=control)
    stats = control.stats
    for key in (
        "control_ticks",
        "command_rows",
        "fallbacks",
        "non_fallback_rows",
        "rows_by_owner",
        "rows_by_mode",
        "fallback_reason_histogram",
    ):
        assert key in stats
    assert set(stats["rows_by_owner"]) <= {"native", "damping", "wall"}
    assert set(stats["rows_by_mode"]) <= {"native", "damping", "pi-external-u", "wall-v2"}
    assert sum(stats["fallback_reason_histogram"].values()) == stats["command_rows"]
