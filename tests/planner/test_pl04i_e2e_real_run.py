"""PL-04i: the END-TO-END closed-loop test over the REAL component wiring.

The PL-04 chain's step-11 failure (planner_cl_smoke_1.log) was a startup-
ordering defect that crashed BEFORE any SUMO started: ClosedLoopRunner.run
builds the world model immediately after the harness, and
build_real_components.make_world_model dereferenced
``harness.master_api`` -- an attribute RealMasterHarness only sets in
``start()`` -- so ``--components real`` died with
``AttributeError: 'RealMasterHarness' object has no attribute '_traci'``.
Every prior test faked Popen or faked the harness, so the ordering bug hid
behind the fakes.

This module drives ``ClosedLoopRunner.run`` through the REAL
``build_real_components`` -> REAL ``RealMasterHarness`` /
``MasterBoundSnapshotter`` / ``RealBranchProcessFactory`` /
``TraciBranchRunner`` / ``SessionControlWrite`` wiring. ONLY two seams are
stubbed, so ordering and lazy-init bugs cannot hide:

- ``sys.modules['traci']`` is a tiny fake world (>=2 live roster CAVs;
  saveState/loadState to REAL temp files that pass the precision-17/RNG
  exactness guards; speedMode/slowDown/getVersion surfaces for the REAL
  SumoLocalSensor, CommandOwnership and CirclesController);
- the SUMO binary is a harmless ``python`` child (the PL-04g pattern: the
  REAL Popen path runs; only the launchers' OWN children are ever waited;
  no SUMO, no signalling by image name or pattern).

The run starts from a campaign-format ``--base-run`` fixture under
tmp_path (resolve_base_run's guards pass on it), runs >=1 decision point
with >=1 non-native lookahead branch, produces RUN_RECEIPT.json, and the
run_cl_smoke wrapper's evaluation reads the produced rows and computes
the a/b/c/receipt_complete verdict (PASS or INCONCLUSIVE, never a crash).
"""

import importlib.util
import inspect
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

PLANNER_ROOT = Path(__file__).resolve().parents[2]

from traffic_shepherd.planner.closed_loop import (  # noqa: E402
    BuildContext,
    ClosedLoopConfig,
    ClosedLoopRunner,
    build_real_components,
    initial_plan_for_roster,
    resolve_base_run,
)

#: the tiny world's seed == the wrapper's SMOKE_SEED, so the wrapper's
#: criterion (c) (world seed everywhere) reads the same value everywhere
SEED = 8468
BASE_T0_S = 240.0
CAVS = ("mainline.6", "mainline.7")
HDV = "hdv.9"
RUN_ID = "w_pl04i_e2e"

#: the adopted adapter's real edge ids (baselines/circles/adapter.json
#: edge_ids) with their REAL lane counts from the preserved network: the
#: fake traci exposes this surface so observation_map (PL-04j) resolves the
#: same 12-lane merge map on the fakes as on the live connection.
EDGE_LANE_COUNTS = {
    "1191554636": 2,  # pre_merge
    "1487606179": 3,  # merge
    "1487606178": 2,  # downstream
    "11553048#1": 1,  # ramp
}
#: the adapter's pre_merge edge, lane 0: the PL-04j variant world's lane
#: (a roster CAV near its end has a frame whose coverage reaches the merge)
PRE_MERGE_LANE = "1191554636_0"


class _EdgeDomain:
    """The edge surface ``observation_map`` reads (PL-04j): the adapter's
    edge_ids -> lane counts (the real network's counts, so the resolved map
    has the registered 12 entries). PL-04l: bound to ITS connection, so
    every read is recorded (and refused after close) like every other op."""

    def __init__(self, connection):
        self._c = connection

    def getLaneNumber(self, edge_id):
        self._c._touch("edge.getLaneNumber")
        if edge_id not in EDGE_LANE_COUNTS:
            raise KeyError(f"fake traci: unknown edge {edge_id!r}")
        return EDGE_LANE_COUNTS[edge_id]


def _load_wrapper(name, relative):
    spec = importlib.util.spec_from_file_location(name, PLANNER_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_cl_smoke = _load_wrapper(
    "pl04i_run_cl_smoke", "output/planner_v1/closed_loop_smoke/run_cl_smoke.py"
)


# --------------------------------------------------- the campaign base run
def make_base_run_pl04i(tmp_path, hdv_speed=22.0):
    """A campaign-format base run (<world>__base) that passes
    ``closed_loop.resolve_base_run`` AND carries both roster CAVs (+ one
    HDV) in the saved state, so the restored tiny world has >=2 live CAVs
    at t0. Same guards as tests.planner._helpers.make_base_run; the state
    file here holds every restored vehicle with 17-decimal positions and
    an <rngState> (the exactness guard reads those bytes). The PL-04m
    roster source is present: run/physics_steps.jsonl rows carry the
    constant cav_ids list (the two CAVs)."""
    base = tmp_path / "w_pl04i_e2e__base"
    scenario_dir = base / "run" / "scenario"
    state_dir = base / "saved_state"
    scenario_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)

    (scenario_dir / "requests.rou.xml").write_text(
        "<routes>\n"
        '  <vType id="normal_car" carFollowModel="IDM" speedDev="0"/>\n'
        '  <route id="mainline_route" edges="e0 e1"/>\n'
        + "".join(
            f'  <vehicle id="v{i}" type="normal_car" route="mainline_route" depart="{i}.0"/>\n'
            for i in range(3)
        )
        + "</routes>\n",
        encoding="utf-8",
    )
    (scenario_dir / "scenario.sumocfg").write_text(
        "<configuration>\n  <input>\n"
        '    <net-file value="legacy-mount/assets/net.xml.gz"/>\n'
        '    <route-files value="requests.rou.xml"/>\n'
        "  </input>\n</configuration>\n",
        encoding="utf-8",
    )
    import hashlib

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def veh_xml(vid, pos, speed):
        return f'<vehicle id="{vid}" pos="{pos:.17f}" speed="{speed:.17f}"/>'

    state_text = (
        "<state>"
        f'<time value="{BASE_T0_S:.17f}"/>'
        f'<rngState seed="{SEED}"/>'
        # spacing keeps every CAV within the 100 m sensor range of its
        # leader, so the REAL CirclesController issues real damping
        # commands instead of falling back
        + veh_xml("mainline.6", 200.0, 20.0)
        + veh_xml("mainline.7", 260.0, 20.0)
        + veh_xml(HDV, 330.0, float(hdv_speed))
        + "</state>"
    )
    (state_dir / "state.xml").write_text(state_text, encoding="utf-8")

    with (base / "run" / "telemetry.jsonl").open("w", encoding="utf-8") as stream:
        for cav in CAVS:  # damping command rows designate the CAV roster
            stream.write(
                json.dumps(
                    {
                        "kind": "command",
                        "vehicle_id": cav,
                        "timestamp_s": 10.0,
                        "owner": "damping",
                        "issued_target_speed_mps": 25.0,
                        "fallback_reason": None,
                        "released_to_native": False,
                    }
                )
                + "\n"
            )
    with (base / "run" / "insertions.jsonl").open("w", encoding="utf-8") as stream:
        for vehicle_id, inserted_s in ((CAVS[0], 5.0), (CAVS[1], 6.0), (HDV, 7.0)):
            stream.write(
                json.dumps(
                    {
                        "vehicle_id": vehicle_id,
                        "scheduled_depart_s": inserted_s,
                        "inserted_s": inserted_s,
                        "entrance": "mainline",
                        "is_cav": False,
                        "type_id": "normal_car",
                    }
                )
                + "\n"
            )

    # PL-04m: the AUTHORITATIVE roster source -- every physics row carries
    # the SAME constant cav_ids list (the campaign's full designated set)
    with (base / "run" / "physics_steps.jsonl").open("w", encoding="utf-8") as stream:
        for row_time in (0.5, float(BASE_T0_S)):
            stream.write(
                json.dumps(
                    {
                        "time_s": row_time,
                        "live_ids": [],
                        "departed_ids": [],
                        "arrived_ids": [],
                        "colliding_ids": [],
                        "cav_ids": list(CAVS),
                        "speed_modes": {},
                        "cav_actions": {},
                    }
                )
                + "\n"
            )

    manifest = {
        "schema": "wm41_saved_state/1",
        "seed": SEED,
        "branch_point_t0_s": float(BASE_T0_S),
        "state_options": {
            "save_state_precision_decimals": 17,
            "rng_state_in_file": True,
            "required_cli": ["--save-state.precision", "17", "--save-state.rng", "true"],
        },
        "sumo_version": [1, 27, 1],
        "sumo_state_file": "state.xml",
        "sumo_state_sha256": sha(state_dir / "state.xml"),
        "scenario_digests": {
            "scenario/requests.rou.xml": sha(scenario_dir / "requests.rou.xml"),
            "scenario/scenario.sumocfg": sha(scenario_dir / "scenario.sumocfg"),
        },
        "harness_components": ["ownership", "circles_controller"],
        "harness_state": {
            "roster": {"cav_ids": list(CAVS)},
            "ownership": {
                "cav_speed_mode": 27,
                "mode_configured": [],
                "owned": {},
                "last": {},
                "time_s": float(BASE_T0_S) - 1.0,
            },
            "loop_state": {"previous_live": [], "retired_cavs": [], "collision_previous": []},
        },
        "physics_row_at_t0": {"time_s": float(BASE_T0_S)},
    }
    (state_dir / "STATE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return base


def make_base_run_pl04j(tmp_path):
    """The PL-04j variant of the campaign base run: the SAME guards as
    ``make_base_run_pl04i``, but the roster CAVs sit on the adapter's
    pre_merge lane within 100 m (sensor range) of the lane end -- the
    merge point -- so a captured frame's coverage REACHES the merge region
    once the RSU carries the merge map (merge_observed True at t0)."""
    base = make_base_run_pl04i(tmp_path)
    state_path = base / "saved_state" / "state.xml"
    state_text = state_path.read_text(encoding="utf-8")
    # 450.0 is 50 m from the 500 m lane end; 470.0 is 30 m -- both inside
    # the 100 m front coverage, spaced 20 m (no overlap, leader in range)
    state_text = state_text.replace('id="mainline.6" pos="200.00000000000000000"',
                                    'id="mainline.6" pos="450.00000000000000000"')
    state_text = state_text.replace('id="mainline.7" pos="260.00000000000000000"',
                                    'id="mainline.7" pos="470.00000000000000000"')
    state_text = state_text.replace(f'id="{HDV}" pos="330.00000000000000000"',
                                    f'id="{HDV}" pos="490.00000000000000000"')
    state_path.write_text(state_text, encoding="utf-8")
    import hashlib

    manifest_path = base / "saved_state" / "STATE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sumo_state_sha256"] = hashlib.sha256(state_path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return base


# ------------------------------------------------------------ the fake traci
class _TinyWorld:
    """One fake SUMO process's world: a 1-lane road, 0.5 s steps, speeds
    that obey the last slowDown command. saveState/loadState go to REAL
    files in the precision-17 + rngState format the exactness guard reads;
    loadState lands on the saved clock (the guarded restore's t0 check).

    ``lane`` is the id EVERY vehicle drives on: the PL-04j variant world
    uses the adapter's pre_merge lane so a CAV near the lane end carries a
    merge-reaching sensor frame on the REAL SumoLocalSensor path."""

    LANE = "L0"

    def __init__(self, lane=None):
        self.LANE = lane or type(self).LANE
        self.time = 0.0
        self.vehicles = {}  # id -> {pos, speed, lane, target}
        self.speed_modes = {}

    def step(self, dt=0.5):
        for veh in self.vehicles.values():
            if veh["target"] is not None:
                veh["speed"] = veh["target"]
            veh["pos"] += veh["speed"] * dt
        self.time = round(self.time + dt, 6)

    def save(self, path):
        parts = [
            "<state>",
            f'<time value="{self.time:.17f}"/>',
            '<rngState seed="8468"/>',
        ]
        for vid in sorted(self.vehicles):
            veh = self.vehicles[vid]
            parts.append(
                f'<vehicle id="{vid}" pos="{veh["pos"]:.17f}" speed="{veh["speed"]:.17f}"/>'
            )
        parts.append("</state>")
        Path(path).write_text("".join(parts), encoding="utf-8")

    def load(self, path):
        root = ET.fromstring(Path(path).read_text(encoding="utf-8"))
        self.time = float(root.find("time").get("value"))
        self.vehicles = {}
        self.speed_modes = {}  # a fresh process: no traci-configured modes
        for element in root.iter("vehicle"):
            self.vehicles[str(element.get("id"))] = {
                "pos": float(element.get("pos")),
                "speed": float(element.get("speed")),
                "lane": self.LANE,
                "target": None,
            }


class _FatalTraCIError(RuntimeError):
    """``traci.exceptions.FatalTraCIError`` stand-in (PL-04l): any traci
    call on a CLOSED connection dies with this error -- the exact real-traci
    symptom of the cl_smoke_2 defect, where the master's saveState landed on
    a lookahead-branch connection that had already been closed."""


class _VehicleDomain:
    """One connection's vehicle surface. Every call is recorded on ITS OWN
    connection (the PL-04l no-cross-talk assertions read those records) and
    is refused with :class:`_FatalTraCIError` once that connection closed."""

    def __init__(self, connection):
        self._c = connection

    def getIDList(self):
        self._c._touch("vehicle.getIDList")
        return tuple(sorted(self._c.world.vehicles))

    def getLaneID(self, vid):
        self._c._touch("vehicle.getLaneID")
        return self._c.world.vehicles[vid]["lane"]

    def getLanePosition(self, vid):
        self._c._touch("vehicle.getLanePosition")
        return self._c.world.vehicles[vid]["pos"]

    def getSpeed(self, vid):
        self._c._touch("vehicle.getSpeed")
        return self._c.world.vehicles[vid]["speed"]

    def getAcceleration(self, vid):
        self._c._touch("vehicle.getAcceleration")
        return 0.0

    def getLaneIndex(self, vid):
        self._c._touch("vehicle.getLaneIndex")
        return 0

    # PL-04k: the campaign per-vehicle physics fields
    def getRoadID(self, vid):
        self._c._touch("vehicle.getRoadID")
        return self._c.world.vehicles[vid]["lane"].rsplit("_", 1)[0]

    def getTypeID(self, vid):
        self._c._touch("vehicle.getTypeID")
        return "normal_car"

    def getLeader(self, vid, dist):
        """The nearest same-lane vehicle ahead within ``dist`` (the real
        traci surface: ('', -1) when there is none)."""
        self._c._touch("vehicle.getLeader")
        veh = self._c.world.vehicles[vid]
        ahead = [
            (other["pos"] - veh["pos"], other_vid)
            for other_vid, other in self._c.world.vehicles.items()
            if other_vid != vid and other["pos"] > veh["pos"]
        ]
        if not ahead:
            return ("", -1)
        gap, leader = min(ahead)
        return (leader, gap) if gap <= float(dist) else ("", -1)

    def getLength(self, vid):
        self._c._touch("vehicle.getLength")
        return 5.0

    def getAllowedSpeed(self, vid):
        self._c._touch("vehicle.getAllowedSpeed")
        return 30.0

    def getSpeedMode(self, vid):
        self._c._touch("vehicle.getSpeedMode")
        return self._c.world.speed_modes.get(vid, 31)

    def setSpeedMode(self, vid, mode):
        self._c._touch("vehicle.setSpeedMode")
        self._c.world.speed_modes[vid] = int(mode)

    def slowDown(self, vid, speed, duration):
        self._c._touch("vehicle.slowDown")
        self._c.world.vehicles[vid]["target"] = float(speed)

    def setSpeed(self, vid, speed):
        self._c._touch("vehicle.setSpeed")
        veh = self._c.world.vehicles[vid]
        veh["target"] = None if float(speed) < 0 else float(speed)

    def add(self, rid, route, typeID="passenger", departSpeed=0.0, departLane="free"):
        self._c._touch("vehicle.add")
        self._c.world.vehicles[str(rid)] = {
            "pos": 10.0,
            "speed": float(departSpeed or 20.0),
            "lane": self._c.world.LANE,
            "target": None,
        }


class _LaneDomain:
    LENGTH = 500.0

    def __init__(self, connection):
        self._c = connection

    def getLength(self, lane_id):
        self._c._touch("lane.getLength")
        return self.LENGTH

    def getLastStepVehicleIDs(self, lane_id):
        self._c._touch("lane.getLastStepVehicleIDs")
        return tuple(
            sorted(
                vid for vid, v in self._c.world.vehicles.items() if v["lane"] == lane_id
            )
        )


class _SimDomain:
    def __init__(self, connection):
        self._c = connection

    def getTime(self):
        self._c._touch("simulation.getTime")
        return self._c.world.time

    def simulationStep(self):
        self._c._touch("simulation.simulationStep")
        self._c.world.step()

    def saveState(self, path):
        self._c._touch("simulation.saveState")
        self._c.world.save(path)

    def loadState(self, path):
        self._c._touch("simulation.loadState")
        self._c.world.load(path)

    def getStartingTeleportIDList(self):
        self._c._touch("simulation.getStartingTeleportIDList")
        return ()

    def getCollidingVehiclesIDList(self):
        self._c._touch("simulation.getCollidingVehiclesIDList")
        return ()

    def getDepartedIDList(self):
        self._c._touch("simulation.getDepartedIDList")
        return ()

    def getArrivedIDList(self):
        self._c._touch("simulation.getArrivedIDList")
        return ()

    def getEndingTeleportIDList(self):
        self._c._touch("simulation.getEndingTeleportIDList")
        return ()


class _TinyConnection:
    """One fake SUMO process's connection (its OWN fresh world).

    PL-04l: every domain call goes through :meth:`_touch`, which (a) records
    the op on THIS connection (``calls``) and in the shared ordered event
    log (``(label, kind, op)`` tuples on the owning :class:`_TinyTraci`),
    and (b) refuses with :class:`_FatalTraCIError` once the connection has
    been closed -- the real Connection socket behaviour the cl_smoke_2
    defect tripped over."""

    def __init__(self, label, lane=None, bus=None):
        self.label = str(label)
        self.world = _TinyWorld(lane=lane)
        self.closed_calls = []
        self.calls = []  # "domain.method" per call, in order
        self._closed = False
        self._bus = bus if bus is not None else []
        self.simulation = _SimDomain(self)
        self.vehicle = _VehicleDomain(self)
        self.lane = _LaneDomain(self)
        self.edge = _EdgeDomain(self)

    def _touch(self, op):
        if self._closed:
            raise _FatalTraCIError("Connection already closed")
        self.calls.append(op)
        self._bus.append((self.label, "op", op))

    def simulationStep(self):
        self._touch("simulationStep")
        self.world.step()

    def getVersion(self):
        self._touch("getVersion")
        return (27, " FakeSUMO 1.27.1")

    def close(self, wait=True):
        self.closed_calls.append(bool(wait))
        self._closed = True
        self._bus.append((self.label, "close", None))


class _TinyTraci:
    """The sys.modules['traci'] fake with REAL module semantics (PL-04l):

    - ``init(label, doSwitch=True)`` registers a fresh connection and, when
      ``doSwitch``, makes it the module-level CURRENT connection (real traci
      main.py: ``init`` calls ``switch(label)`` when ``doSwitch``);
    - every module-level surface (``simulation``/``vehicle``/``lane``/
      ``edge``, ``simulationStep``, ``getVersion``, ``close``) acts on the
      CURRENT connection (real traci: module functions call
      ``getConnection()``);
    - ``getConnection(label)`` returns the labelled connection;
    - module ``close()`` closes the CURRENT connection and deletes its
      registry entry (real traci module close).

    ``module_calls`` records every module-level surface access, so a test
    can prove production code never routed anything through the module;
    ``events`` is the global ordered ``(label, kind, op)`` log across all
    connections (``init`` entries included)."""

    def __init__(self, lane=None):
        self.connections = {}
        self.init_calls = []
        self.current_label = "default"  # real traci's module-level _defaultLabel
        self.module_calls = []
        self.events = []
        self._lane = lane  # every connection's world drives on this lane

    # -- connection lifecycle (the launchers' real traci calls) ----------
    def init(self, label="default", numRetries=10, port=0, host="localhost",
             proc=None, doSwitch=True, **kwargs):
        self.init_calls.append(
            {
                "label": str(label),
                "port": port,
                "pid": getattr(proc, "pid", None),
                "doSwitch": bool(doSwitch),
            }
        )
        self.connections[str(label)] = _TinyConnection(
            label, lane=self._lane, bus=self.events
        )
        if doSwitch:
            self.current_label = str(label)
        self.events.append((str(label), "init", None))
        return None  # real traci.init returns None; use getConnection(label)

    def getConnection(self, label=None):
        if label is None:
            label = self.current_label
        return self.connections[label]

    def _current(self, surface):
        self.module_calls.append(surface)
        try:
            return self.connections[self.current_label]
        except KeyError:
            raise KeyError(
                f"Connection with label {self.current_label!r} does not exist"
            ) from None

    def close(self):
        conn = self._current("close")
        conn.close()
        del self.connections[self.current_label]

    # -- the module-level surfaces (real traci: they act on the CURRENT
    #    connection -- NOT on any fixed master)
    @property
    def simulation(self):
        return self._current("simulation").simulation

    @property
    def vehicle(self):
        return self._current("vehicle").vehicle

    @property
    def lane(self):
        return self._current("lane").lane

    @property
    def edge(self):
        return self._current("edge").edge

    def simulationStep(self):
        self._current("simulationStep").simulationStep()

    def getVersion(self):
        return self._current("getVersion").getVersion()


# ------------------------------------------------------------- the E2E test
def _real_args(base):
    """The closed_loop CLI's real-run options, as main() would parse them:
    --components real --predictor perfect_sim --base-run <fixture> --seed
    8468 --walls off --max-decision-units 2 --decision-period-s 10
    --lookahead-horizon-s 5 --horizon-s 10 --initial-plan damping."""
    return SimpleNamespace(
        predictor="perfect_sim",
        seed=SEED,
        base_run=str(base),
        decision_period_s=10.0,
        lookahead_horizon_s=5.0,
        lookahead_commit_s=None,
    )


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class TestPL04iEndToEnd:
    """ClosedLoopRunner.run over the REAL wiring; only traci + the SUMO
    binary are stubbed. This is the regression test for the step-11
    AttributeError ('RealMasterHarness' object has no attribute '_traci')
    -- on the pre-PL-04i code it dies inside make_world_model before any
    SUMO would start."""

    @pytest.fixture()
    def e2e(self, tmp_path, monkeypatch):
        base = make_base_run_pl04i(tmp_path)
        resolved = resolve_base_run(base)
        # the SUMO binary is a harmless python child (PL-04g); the real
        # Popen path runs and only the launchers' OWN children are waited
        monkeypatch.setenv("SUMO_BINARY", sys.executable)
        monkeypatch.delenv("SUMO_HOME", raising=False)
        traci = _TinyTraci()
        monkeypatch.setitem(sys.modules, "traci", traci)  # the ONLY module stub
        args = _real_args(base)
        components = build_real_components(args)
        # --initial-plan damping (PL-04f): the pre-set non-native plan puts
        # a non-native lookahead branch through the REAL per-branch control
        # write at the decision point (the planner otherwise abstains on
        # this fixture: the real RSU path carries no merge-region inputs).
        components.initial_plan = initial_plan_for_roster(components.roster, "damping")
        config = ClosedLoopConfig(
            out_root=tmp_path,
            run_id=RUN_ID,
            world="w_pl04i",
            seed=SEED,
            horizon_s=10.0,
            warmup_s=0.0,
            decision_period_s=10.0,
            predictor="perfect_sim",
            lookahead_horizon_s=5.0,
            lookahead_commit_s=None,
            k_samples=1,
            max_sim_s=500.0,
            base_run=resolved.base_run,
            base_t0_s=resolved.t0_s,
            base_state_sha256=resolved.state_sha256,
            # T-B (#10): this fixture carries no RSU distances, so a CAP would
            # revert every CAV to native (no_decision_units); the control-path
            # fixture runs UNCAPPED to keep exercising the non-native path
            max_decision_units=None,
            walls="off",
            initial_plan_action="damping",
        )
        return SimpleNamespace(
            base=base,
            traci=traci,
            components=components,
            config=config,
            run_dir=tmp_path / RUN_ID,
        )

    def test_real_components_run_completes_with_receipt_and_verdict(self, e2e):
        receipt = ClosedLoopRunner(e2e.config, e2e.components).run()

        # the run COMPLETED: >=1 decision point, base-state start
        assert receipt["exit_status"] == "complete"
        assert receipt["decisions"] >= 1
        assert receipt["start_mode"] == "base_state"
        assert receipt["cav_roster"] == list(CAVS)
        assert receipt["seed"] == SEED
        assert (e2e.run_dir / "RUN_RECEIPT.json").is_file()

        # the REAL launch path ran: a REAL Popen child per process (the
        # harmless python children), each with its own port and the run seed
        kinds = [record["kind"] for record in receipt["sumo_launches"]]
        assert kinds[0] == "master" and kinds.count("branch") >= 1
        assert all(record["pid"] > 0 for record in receipt["sumo_launches"])
        assert all("--seed" in record["cmd"] for record in receipt["sumo_launches"])
        assert [call["label"] for call in e2e.traci.init_calls][0] == "pl04"
        assert len(e2e.traci.init_calls) == len(receipt["sumo_launches"])

        # the tiny world really ran: >=1 decision with a NON-NATIVE lookahead
        # branch whose REAL control write issued command rows on its own
        # fresh connection, and the master's control tick issued damping
        branch_rows = read_jsonl(e2e.run_dir / "branch_control_rows.jsonl")
        assert branch_rows, "the non-native lookahead branch issued no control rows"
        assert any("plan" in str(row["branch"]) for row in branch_rows)
        assert any(str(row["command"]) == "damping" for row in branch_rows)
        decision = read_jsonl(e2e.run_dir / "decisions.jsonl")[0]
        assert set(decision["actions"].values()) == {"damping"}
        control = [
            row
            for row in read_jsonl(e2e.run_dir / "dispatcher_rows.jsonl")
            if row.get("source") == "master_control_tick"
        ]
        assert control and any(row["command"] == "damping" for row in control)
        assert read_jsonl(e2e.run_dir / "physics_steps.jsonl")

        # the guarded snapshot protocol ran for real: the master's ONE base
        # loadState + every branch's own loadState landed on the saved t0
        master = e2e.traci.connections["pl04"]
        assert abs(master.world.time - 250.0) < 1e-9  # stepped to the horizon
        assert len(master.closed_calls) == 1  # closed exactly once at run end
        branch_labels = [
            call["label"] for call in e2e.traci.init_calls if call["label"] != "pl04"
        ]
        assert branch_labels
        for label in branch_labels:
            assert e2e.traci.connections[label].world.time == pytest.approx(245.0)

        # every branch process was retired on its own connection
        assert all(
            e2e.traci.connections[label].closed_calls == [False]
            for label in branch_labels
        )

        # the run_cl_smoke evaluation reads the REAL produced rows: the a/b/c
        # keys are computed and the fake world's verdict is PASS/INCONCLUSIVE
        code = run_cl_smoke.main(
            ["--out", str(e2e.run_dir.parent), "--base-run", str(e2e.base),
             "--run-id", RUN_ID],
            launcher=lambda argv, cwd: {"returncode": 0, "stdout": "", "stderr": ""},
        )
        verdict = json.loads(
            (e2e.run_dir.parent / "CL_SMOKE_VERDICT.json").read_text(encoding="utf-8")
        )
        for key in (
            "a_cav_command_differs_from_native",
            "b_branch_control_rows_nonempty_for_nonnative",
            "c_world_seed_everywhere",
            "receipt_complete",
        ):
            assert key in verdict  # all four criteria were actually computed
        assert verdict["verdict"] in ("PASS", "INCONCLUSIVE")
        assert code in (0, 1)
        assert verdict["a_cav_command_differs_from_native"] is True
        assert verdict["b_branch_control_rows_nonempty_for_nonnative"] is True
        assert verdict["c_world_seed_everywhere"] is True
        assert verdict["receipt_complete"] is True


class TestPL04jPlannerDecidesNearMerge:
    """PL-04j: the REAL master's RSU sensor carries the merge-region lane
    map, resolved from the LIVE connection via ``observation_map`` (nothing
    injected). With a roster CAV near the merge, ``merge_observed`` is True
    at the decision point and the planner DECIDES -- no ``--initial-plan``
    crutch, no abstain (the pre-PL-04j code abstained at every decision
    because the real path carried no merge-region inputs)."""

    @pytest.fixture()
    def e2e_planner(self, tmp_path, monkeypatch):
        base = make_base_run_pl04j(tmp_path)
        resolved = resolve_base_run(base)
        monkeypatch.setenv("SUMO_BINARY", sys.executable)
        monkeypatch.delenv("SUMO_HOME", raising=False)
        # the tiny world drives on the adapter's pre_merge lane: a mapped
        # lane, so the REAL SumoLocalSensor frames carry distance_to_merge_m
        traci = _TinyTraci(lane=PRE_MERGE_LANE)
        monkeypatch.setitem(sys.modules, "traci", traci)
        args = _real_args(base)
        components = build_real_components(args)
        # NO initial plan (default native): run_purpose='planner' -- the
        # planner decides everything from its own RSU observation
        config = ClosedLoopConfig(
            out_root=tmp_path,
            run_id=RUN_ID + "_planner",
            world="w_pl04j",
            seed=SEED,
            horizon_s=10.0,
            warmup_s=0.0,
            decision_period_s=10.0,
            predictor="perfect_sim",
            lookahead_horizon_s=5.0,
            lookahead_commit_s=None,
            k_samples=1,
            max_sim_s=500.0,
            base_run=resolved.base_run,
            base_t0_s=resolved.t0_s,
            base_state_sha256=resolved.state_sha256,
            max_decision_units=2,
            walls="off",
        )
        return SimpleNamespace(
            base=base,
            traci=traci,
            components=components,
            config=config,
            run_dir=tmp_path / (RUN_ID + "_planner"),
        )

    def test_planner_decides_when_merge_observed_without_initial_plan(
        self, e2e_planner
    ):
        receipt = ClosedLoopRunner(e2e_planner.config, e2e_planner.components).run()

        assert receipt["exit_status"] == "complete"
        assert receipt["decisions"] >= 1
        assert receipt["start_mode"] == "base_state"
        assert receipt["initial_plan"] == "native"
        assert receipt["run_purpose"] == "planner"

        # PL-04j receipt evidence: the master's merge map was resolved from
        # the live connection (source 'observation_map'), 12 mapped lanes,
        # never silently empty
        master_launch = receipt["sumo_launches"][0]
        assert master_launch["kind"] == "master"
        assert master_launch["merge_lane_map_source"] == "observation_map"
        assert master_launch["merge_lane_map_size"] == 12

        # THE defect's observable: at the decision point the RSU frames
        # coverage-reach the merge, so the planner does NOT abstain and the
        # lookahead branches were evaluated (>=2 candidate plans predicted)
        decision = read_jsonl(e2e_planner.run_dir / "decisions.jsonl")[0]
        assert decision["merge_observed"] == [True]
        assert decision["abstain"] is False
        assert decision["n_candidates_evaluated"] >= 2
        assert any(
            record["kind"] == "branch" for record in receipt["sumo_launches"]
        )


class TestPL04iStartupOrder:
    """The startup-order contract (PL-04i): the runner builds ALL components
    BEFORE the harness starts, so the build path must not need the master
    connection -- and anything that truly needs it gets the NAMED error."""

    def test_build_completes_before_start_no_traci_touched(self, tmp_path, monkeypatch):
        base = make_base_run_pl04i(tmp_path / "fixture")
        monkeypatch.delenv("SUMO_BINARY", raising=False)
        monkeypatch.setenv("SUMO_HOME", "E:/sumo_home_fake_for_test")
        # sys.modules['traci'] = None makes `import traci` raise ImportError:
        # if the build path touched traci at all, this test fails loudly.
        monkeypatch.setitem(sys.modules, "traci", None)

        from traffic_shepherd.planner.closed_loop import HarnessNotStartedError

        components = build_real_components(_real_args(base))
        context = BuildContext(run_dir=tmp_path / "b", resume=False, process_seq=1)
        harness = components.make_harness(context)
        # the build order make_harness -> make_world_model runs BEFORE
        # start() and must not dereference the master connection
        model = components.make_world_model(context)
        assert type(harness).__name__ == "RealMasterHarness"
        assert type(model).__name__ == "PerfectSimWorldModel"
        assert type(model.snapshotter).__name__ == "MasterBoundSnapshotter"
        assert type(model.branch_runner).__name__ == "TraciBranchRunner"
        assert (
            type(model.branch_runner.process_factory).__name__
            == "RealBranchProcessFactory"
        )
        # the symmetry-only branch-runner api is a LAZY provider or None --
        # never a build-time dereference of harness.master_api
        assert model.branch_runner.api is None or callable(model.branch_runner.api)
        # the master-bound snapshotter binds lazily (nothing bound yet)
        assert model.snapshotter._inner is None
        # anything that truly needs the master api before start() raises the
        # NAMED error, never AttributeError
        with pytest.raises(HarnessNotStartedError, match="harness not started"):
            harness.master_api
        with pytest.raises(HarnessNotStartedError, match="harness not started"):
            model.snapshotter._bound()

    def test_run_documents_the_build_then_start_order(self):
        source = inspect.getsource(ClosedLoopRunner.run)
        assert "build, then start" in source
        harness_at = source.index("harness = self.components.make_harness(context)")
        model_at = source.index("model = self.components.make_world_model(context)")
        # search AFTER the build lines: the comment above them mentions
        # start() too
        start_at = source.index("harness.start()", model_at)
        assert harness_at < model_at < start_at
