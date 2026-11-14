"""WM-43 r2 scenario-gate tests: static validator + the r1 failure regression.

The WM-43 r1 launch failed on all 15 runs because the generated route file defined
vType 'normal_car' (the template identity leaked through an attribute merge)
while every vehicle referenced 'wm43_cav'.  These tests pin the fix: the
builder writes the fixture identity, and the static validator rejects every
synthetic corruption class (the v1 recorded r1 evidence was deleted by MIG-04), while accepting all 25 regenerated scenarios.  No SUMO.
"""
from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from conftest import REPO_ROOT  # noqa: F401 - pytest path helper
from traffic_shepherd.config import load_config
from wm43_plan import ARMS, SEEDS_BY_ARM
from wm43_scenario import build_scenario
from wm43_validate import validate_scenario

WORLD = {"run": "gate", "arm": "A1", "seed": 920249, "loading": 2000,
         "penetration": 1.0, "role": "exposed", "wall_speed_mps": 8.0}


def _build(tmp_path: Path, world: dict | None = None) -> Path:
    scene, roster, manifest = build_scenario(tmp_path, world or WORLD,
                                             load_config())
    return scene.parent


def test_vtype_identity_regression_r1(tmp_path):
    """The r1 bug: the template 'id' overwrote 'wm43_cav' in the merge."""
    scenario = _build(tmp_path / "vtype")
    root = ET.parse(scenario / "requests.rou.xml").getroot()
    vtypes = root.findall("vType")
    assert [v.get("id") for v in vtypes] == ["wm43_cav"]
    assert {v.get("type") for v in root.findall("vehicle")} == {"wm43_cav"}
    # template-derived dynamics survive, the template identity does not
    assert vtypes[0].get("carFollowModel") == "IDM"
    assert vtypes[0].get("speedDev") == "0"
    assert "id" not in {k: v for k, v in vtypes[0].attrib.items()
                        if k != "id"} or vtypes[0].get("id") == "wm43_cav"


@pytest.mark.parametrize("arm,seed", [(a, sd) for a in ARMS for sd in SEEDS_BY_ARM[a]])
def test_validator_accepts_all_regenerated(tmp_path, arm, seed):
    scenario = _build(tmp_path / f"{arm}_{seed}",
                      {**WORLD, "arm": arm, "seed": seed})
    assert validate_scenario(scenario) == []


def _corrupt(tmp_path: Path, mutation) -> list[str]:
    name = mutation.__name__
    scenario = _build(tmp_path / f"good_{name}")  # fresh build per mutation
    broken = tmp_path / f"bad_{name}"
    shutil.copytree(scenario, broken)
    mutation(broken)
    return validate_scenario(broken)


def test_validator_rejects_synthetic_corruptions(tmp_path):
    routes_name = "requests.rou.xml"

    def dangling_type(d: Path):
        tree = ET.parse(d / routes_name)
        tree.getroot().find("vType").set("id", "normal_car")
        tree.write(d / routes_name, encoding="utf-8", xml_declaration=True)

    def unknown_route(d: Path):
        tree = ET.parse(d / routes_name)
        tree.getroot().find("vehicle").set("route", "ghost_route")
        tree.write(d / routes_name, encoding="utf-8", xml_declaration=True)

    def lane_out_of_range(d: Path):
        tree = ET.parse(d / routes_name)
        tree.getroot().find("vehicle").set("departLane", "5")
        tree.write(d / routes_name, encoding="utf-8", xml_declaration=True)

    def pos_beyond_length(d: Path):
        tree = ET.parse(d / routes_name)
        tree.getroot().find("vehicle").set("departPos", "99999")
        tree.write(d / routes_name, encoding="utf-8", xml_declaration=True)

    def speed_over_limit(d: Path):
        tree = ET.parse(d / routes_name)
        tree.getroot().find("vehicle").set("departSpeed", "99")
        tree.write(d / routes_name, encoding="utf-8", xml_declaration=True)

    def duplicate_vehicle(d: Path):
        tree = ET.parse(d / routes_name)
        root = tree.getroot()
        vehicle = root.find("vehicle")
        clone = ET.Element("vehicle", dict(vehicle.attrib))
        root.append(clone)
        tree.write(d / routes_name, encoding="utf-8", xml_declaration=True)

    def missing_net(d: Path):
        cfg = d / "scenario.sumocfg"
        cfg.write_text(cfg.read_text(encoding="utf-8").replace(
            "osm.net.xml.gz", "absent.net.xml.gz"), encoding="utf-8")

    cases = ((dangling_type, "V4"), (unknown_route, "V7"),
             (lane_out_of_range, "V8"), (pos_beyond_length, "V9"),
             (speed_over_limit, "V10"), (duplicate_vehicle, "V6"),
             (missing_net, "V1"))
    for mutation, rule in cases:
        problems = _corrupt(tmp_path, mutation)
        assert problems, f"{mutation.__name__} must be rejected"
        assert any(p.startswith(rule) for p in problems), \
            f"{mutation.__name__}: expected {rule}, got {problems}"


def test_loop_child_refuses_invalid_scenario_before_sumo(tmp_path, monkeypatch):
    """The run child validates its own freshly built scenario pre-start_sumo."""
    import wm43_loop
    import wm43_scenario

    started = []
    monkeypatch.setattr(wm43_loop, "start_sumo", lambda *a, **k: started.append(a))

    real_build = wm43_scenario.build_scenario

    def build_then_corrupt(run_dir, world, config):
        scene, roster, manifest = real_build(run_dir, world, config)
        routes = scene.parent / "requests.rou.xml"
        tree = ET.parse(routes)
        tree.getroot().find("vType").set("id", "normal_car")  # the r1 corruption
        tree.write(routes, encoding="utf-8", xml_declaration=True)
        return scene, roster, manifest

    monkeypatch.setattr(wm43_scenario, "build_scenario", build_then_corrupt)

    world = dict(WORLD, run="gate_child")
    with pytest.raises(RuntimeError, match="scenario validation refused"):
        wm43_loop.run_mechanics(world, tmp_path / "run", api=object(),
                                controller=object(), config=load_config())
    assert started == []  # zero SUMO interactions
