"""WM-43 pre-launch reality audit: fixture surface vs the INSTALLED traci/SUMO.

These tests import the installed traci module objects (SUMO 1.27.1 tools)
and read signatures/help text ONLY - no SUMO process is ever started.  They
pin the exact WM-41 crash classes out of the fixture:

1. mock-vs-real method shapes: every TraCI call the mechanics loop, the
   ownership path, the sensor and the metrics consume must exist on the
   installed module with the same positional arity/defaults as fake_sumo;
2. the version/step entry points: the installed simulation domain has NO
   getVersion and NO simulationStep - the version is top-level
   ``traci.getVersion()`` (2-tuple (int, 'SUMO x.y.z')) and stepping is
   top-level ``traci.simulationStep(step=0)``;
3. getLeader legacy default: (vehID, dist=100.0) -> (id, gap-without-minGap)
   or None (legacy setLegacyGetLeader(True) default in 1.27.1);
4. the SUMO command line options used by the fixture exist in 1.27.1
   (help text only);
5. the on-disk scenario layout: absolute preserved-network net-file, route
   file next to the sumocfg, no legacy D:/research paths (so
   sumo_input_overrides contributes nothing and resolve_research_path
   passes the fixture paths through unchanged).
"""
from __future__ import annotations

import inspect
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import traci

from conftest import FIXTURE_DIR, REPO_ROOT  # noqa: F401 - pytest path helper
from fake_sumo import FakeSumo, FakeVehicles


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def shape(fn) -> tuple[int, tuple]:
    """(positional arity, defaults tuple) of a bound method."""
    sig = inspect.signature(fn)
    parameters = list(sig.parameters.values())
    required = [p for p in parameters if p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    defaulted = [p for p in parameters if p.default is not inspect.Parameter.empty
                 and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return len(required) + len(defaulted), tuple(
        p.default for p in defaulted if p.default is not inspect.Parameter.empty)


def defaulted(fn) -> tuple:
    return tuple(p.default for p in inspect.signature(fn).parameters.values()
                 if p.default is not inspect.Parameter.empty)


#: every TraCI surface the fixture consumes, as (real callable, fake callable)
CONSUMED_SURFACE = [
    ("simulation.getTime", traci.simulation.getTime, FakeSumo({}).simulation.getTime),
    ("simulation.getArrivedIDList", traci.simulation.getArrivedIDList,
     FakeSumo({}).simulation.getArrivedIDList),
    ("simulation.getCollidingVehiclesIDList", traci.simulation.getCollidingVehiclesIDList,
     FakeSumo({}).simulation.getCollidingVehiclesIDList),
    ("simulation.getStartingTeleportNumber", traci.simulation.getStartingTeleportNumber,
     FakeSumo({}).simulation.getStartingTeleportNumber),
    ("vehicle.getIDList", traci.vehicle.getIDList, FakeVehicles({}).getIDList),
    ("vehicle.getSpeedMode", traci.vehicle.getSpeedMode, FakeVehicles({}).getSpeedMode),
    ("vehicle.getLaneID", traci.vehicle.getLaneID, FakeVehicles({}).getLaneID),
    ("vehicle.getLaneIndex", traci.vehicle.getLaneIndex, FakeVehicles({}).getLaneIndex),
    ("vehicle.getLanePosition", traci.vehicle.getLanePosition,
     FakeVehicles({}).getLanePosition),
    ("vehicle.getLength", traci.vehicle.getLength, FakeVehicles({}).getLength),
    ("vehicle.getSpeed", traci.vehicle.getSpeed, FakeVehicles({}).getSpeed),
    ("vehicle.getAcceleration", traci.vehicle.getAcceleration,
     FakeVehicles({}).getAcceleration),
    ("vehicle.getAllowedSpeed", traci.vehicle.getAllowedSpeed,
     FakeVehicles({}).getAllowedSpeed),
    ("vehicle.getMinGap", traci.vehicle.getMinGap, FakeVehicles({}).getMinGap),
    ("vehicle.getLeader", traci.vehicle.getLeader, FakeVehicles({}).getLeader),
    ("vehicle.slowDown", traci.vehicle.slowDown, FakeVehicles({}).slowDown),
    ("vehicle.setSpeed", traci.vehicle.setSpeed, FakeVehicles({}).setSpeed),
    ("lane.getLength", traci.lane.getLength, None),  # fake asserts below
    ("lane.getLastStepVehicleIDs", traci.lane.getLastStepVehicleIDs, None),
    ("edge.getLaneNumber", traci.edge.getLaneNumber, None),
]


# ---------------------------------------------------------------------------
# 1. consumed-surface parity (the WM-41 crash class: mock vs real shapes)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label", [row[0] for row in CONSUMED_SURFACE])
def test_consumed_surface_exists_on_installed_traci(label):
    for name, real_fn, _fake in CONSUMED_SURFACE:
        if name == label:
            assert callable(real_fn), f"installed traci lacks {name}"


def test_consumed_surface_arity_matches_fake():
    mismatches = []
    for name, real_fn, fake_fn in CONSUMED_SURFACE:
        if fake_fn is None:
            continue  # lane/edge fakes checked in their own test
        if shape(real_fn) != shape(fake_fn):
            mismatches.append((name, shape(real_fn), shape(fake_fn)))
    assert not mismatches, f"fake shapes differ from installed traci: {mismatches}"


def test_lane_edge_fake_arity():
    api = FakeSumo([])
    for real_fn, fake_fn in ((traci.lane.getLength, api.lane.getLength),
                             (traci.lane.getLastStepVehicleIDs,
                              api.lane.getLastStepVehicleIDs),
                             (traci.edge.getLaneNumber, api.edge.getLaneNumber)):
        assert shape(real_fn) == shape(fake_fn), (real_fn, fake_fn)


def test_vehicle_get_leader_shape_and_legacy_default():
    assert defaulted(traci.vehicle.getLeader) == (100.0,)
    doc = inspect.getdoc(traci.vehicle.getLeader) or ""
    assert "'None'" in doc or "None" in doc  # legacy default returns None
    assert callable(traci.setLegacyGetLeader)  # the legacy switch exists


def test_vehicle_command_shapes():
    # the ownership write path: slowDown(vehID, speed, duration); setSpeed(vehID, speed)
    assert [p.name for p in inspect.signature(traci.vehicle.slowDown).parameters.values()] \
        == ["vehID", "speed", "duration"]
    assert [p.name for p in inspect.signature(traci.vehicle.setSpeed).parameters.values()] \
        == ["vehID", "speed"]


# ---------------------------------------------------------------------------
# 2. version/step entry points (the exact WM-41 getVersion crash class)
# ---------------------------------------------------------------------------

def test_version_is_top_level_two_tuple_api():
    assert callable(traci.getVersion)
    # the installed simulation domain has NO getVersion (top-level only)
    assert not hasattr(traci.simulation, "getVersion")
    source = inspect.getsource(type(traci.simulation))
    assert "def getVersion" not in source


def test_step_is_top_level_not_domain_simulationStep():
    assert callable(traci.simulationStep)
    assert not hasattr(traci.simulation, "simulationStep")  # the 1.27.1 reality
    assert callable(traci.simulation.step)  # domain stepping is named step
    params = list(inspect.signature(traci.simulationStep).parameters.values())
    assert params[0].name == "step" and params[0].default == 0


def test_start_accepts_port_kwarg_like_the_fixture_calls_it():
    params = inspect.signature(traci.start).parameters
    assert list(params)[0] == "cmd"
    assert "port" in params and params["port"].default is None
    # the fake accepts the exact launch call simulation.py makes:
    # api.start(command, port=port)
    api = FakeSumo([])
    api.start(["sumo", "-c", "x.sumocfg"], port=54321)
    assert api.started_commands == [["sumo", "-c", "x.sumocfg"]]
    assert api.isLoaded() is True
    api.close()
    assert api.isLoaded() is False


def test_fake_version_shape_matches_installed_contract():
    version = FakeSumo([]).getVersion()
    assert isinstance(version, tuple) and len(version) == 2  # (major:int, str)
    assert isinstance(version[0], int) and isinstance(version[1], str)
    assert version[1].startswith("SUMO")


# ---------------------------------------------------------------------------
# 3. SUMO 1.27.1 command line options used by the fixture (help text only)
# ---------------------------------------------------------------------------

OPTIONS_USED = (
    "--seed",                 # start_sumo
    "--step-length",          # start_sumo (physics 0.5)
    "--end",                  # start_sumo (episode+1 cap)
    "--no-step-log",          # start_sumo
    "--duration-log.disable",  # start_sumo
    "--remote-port",          # added by traci.start itself
    "--net-file",             # sumocfg input element (build_scenario)
    "--route-files",          # sumocfg input element (build_scenario)
    "--time-to-teleport",     # sumocfg processing element (-1 = off)
    "--collision.action",     # sumocfg processing element (warn)
)


@pytest.mark.skipif(shutil.which("sumo") is None, reason="sumo not on PATH")
def test_sumo_1271_command_line_options_exist():
    help_text = subprocess.run(["sumo", "--help"], capture_output=True, text=True).stdout
    missing = [option for option in OPTIONS_USED
               if not any(token in line for line in help_text.splitlines()
                          for token in (option,))]
    assert not missing, f"options missing from installed sumo --help: {missing}"


# ---------------------------------------------------------------------------
# 4. on-disk scenario layout (WM-41 crash class: path layout)
# ---------------------------------------------------------------------------

def test_scenario_layout_and_legacy_passthrough(tmp_path):
    from traffic_shepherd.config import load_config
    from traffic_shepherd.research_assets import resolve_research_path
    from traffic_shepherd.research_assets import sumo_input_overrides
    from wm43_scenario import build_scenario

    config = load_config()
    world = {"run": "layout", "arm": "A1", "seed": 920249, "loading": 2000,
             "penetration": 1.0, "role": "exposed", "wall_speed_mps": 8.0}
    scene, roster, manifest = build_scenario(tmp_path, world, config)

    # run-dir tree: scenario/ contains exactly the three fixture files
    assert sorted(p.name for p in scene.parent.iterdir()) == \
        ["requests.rou.xml", "scenario.sumocfg", "scenario_manifest.json"]

    root = ET.parse(scene).getroot()
    net = root.find("./input/net-file")
    routes = root.find("./input/route-files")
    # net-file is the ABSOLUTE preserved-network path (bytes untouched)
    assert Path(net.get("value")) == config.network_path
    assert Path(net.get("value")).is_file()
    # route file is referenced RELATIVE and sits next to the sumocfg
    assert routes.get("value") == "requests.rou.xml"
    assert (scene.parent / routes.get("value")).is_file()
    # the registered processing flags are present
    processing = {n.tag: n.get("value") for n in root.find("./processing")}
    assert processing == {"time-to-teleport": "-1", "collision.action": "warn"}

    # no legacy D:/research spellings anywhere in the config: the resolver
    # passes the fixture paths through unchanged and adds NO CLI overrides,
    # so the real launch line is exactly the config + start_sumo's options
    assert resolve_research_path(scene) == Path(scene)
    assert sumo_input_overrides(scene) == []
    assert resolve_research_path(config.network_path) == config.network_path
