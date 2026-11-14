"""Mechanics runner allocation/fixture tests; never launches SUMO or netconvert."""
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from traffic_shepherd.formation_mechanics import (
    MEMBERS, apply_fixture_disturbance, build_fixture, case_catalog, lifecycle_window, plan,
    validate_registration,
)
from traffic_shepherd.verification import hash_file


def test_catalog_exact_bounded_one_factor_design():
    cases = case_catalog()
    by_id = {c.case_id:c for c in cases}
    assert len(cases) == len(by_id) == 3
    assert set(by_id) == {"member_lane_loss","member_loss","numeric_taper"}
    assert by_id["member_lane_loss"].disturbance_time_s == 2.
    assert by_id["member_loss"].disturbance_time_s == 8.
    assert by_id["numeric_taper"].road_speed_limit_mps == 15.
    assert plan()["maximum_simulated_seconds"] == 420
    assert plan()["horizon_s"] == 140 and plan()["automatic_retries"] == 0
    assert plan()["mechanical_not_traffic_efficacy"]


@pytest.mark.parametrize("case", case_catalog(),ids=lambda c:c.case_id)
def test_fixture_is_exact_two_lane_idm_without_hdv_speed_override(tmp_path,monkeypatch,case):
    import traffic_shepherd.formation_mechanics as module
    calls = []
    def fake_conversion(command,**kwargs):
        calls.append(command)
        Path(command[-1]).write_text("<net/>")
        return SimpleNamespace(stdout="fixture only",stderr="",check_returncode=lambda:None)
    monkeypatch.setattr(module.subprocess,"run",fake_conversion)
    scene,requests = build_fixture(tmp_path/case.case_id,case,"C:/SUMO/sumo.exe")
    assert scene.exists() and len(calls) == 1
    root = ET.parse(scene.parent/"routes.rou.xml").getroot()
    assert all(t.attrib["carFollowModel"] == "IDM" for t in root.findall("vType"))
    assert all(t.attrib["speedDev"] == "0" for t in root.findall("vType"))
    vehicles = {v.attrib["id"]:v.attrib for v in root.findall("vehicle")}
    assert set(MEMBERS) <= set(vehicles)
    assert vehicles[MEMBERS[0]]["departLane"] == "0" and vehicles[MEMBERS[1]]["departLane"] == "1"
    assert float(vehicles[MEMBERS[0]]["departPos"])-float(vehicles[MEMBERS[1]]["departPos"]) == case.initial_span_m
    assert float(vehicles[MEMBERS[0]]["departSpeed"])-float(vehicles[MEMBERS[1]]["departSpeed"]) == case.speed_difference_mps
    assert sum(r.is_cav for r in requests) == 2
    assert all(0 < float(v["departPos"]) < 3500 for v in vehicles.values())


def test_scripted_disturbance_boundary_uses_no_longitudinal_command_or_teleport():
    lane_requests,removals = [],[]
    vehicle = SimpleNamespace(getLanePosition=lambda v:1000.,
        changeLane=lambda *args:lane_requests.append(args),remove=lambda v:removals.append(v))
    api = SimpleNamespace(vehicle=vehicle)
    for case in case_catalog():
        if case.disturbance_time_s is not None:
            event = apply_fixture_disturbance(api,case,now_s=case.disturbance_time_s)
            if case.disturbance == "report_loss":
                assert event is None
            else:
                assert event["source"] == "registered_fixture_disturbance_not_controller"
    assert lane_requests == [(MEMBERS[1],0,1.)]
    assert removals == [MEMBERS[1]]


def test_registration_requires_exact_plan_and_implementation(tmp_path,monkeypatch):
    import traffic_shepherd.formation_mechanics as module
    monkeypatch.setattr(module,"ROOT",tmp_path)
    package = tmp_path/"traffic_shepherd"
    package.mkdir()
    for name in ("formation.py","formation_mechanics.py"):
        (package/name).write_text(name)
    adapter = tmp_path/"baselines/circles/adapter.json"
    adapter.parent.mkdir(parents=True)
    adapter.write_text("{}")
    registration = dict(plan(),registration_id="synthetic_unit_fixture_not_run",
        source_sha256={name:hash_file(package/name) for name in ("formation.py","formation_mechanics.py")},
        adapter_sha256=hash_file(adapter))
    validate_registration(registration)
    altered = dict(registration,horizon_s=141)
    with pytest.raises(ValueError,match="horizon_s"):
        validate_registration(altered)
    (package/"formation.py").write_text("changed")
    with pytest.raises(ValueError,match="source changed"):
        validate_registration(registration)
    assert json.loads(json.dumps(registration))["case_definitions"] == [asdict(c) for c in case_catalog()]


def test_plan_does_not_launch_a_process(monkeypatch):
    import traffic_shepherd.formation_mechanics as module
    def forbidden(*args,**kwargs):
        raise AssertionError("planning is read-only and must not launch a simulator")
    monkeypatch.setattr(module.subprocess,"run",forbidden)
    monkeypatch.setattr(module,"start_sumo",forbidden)
    assert module.plan()["schema"] == "formation_mechanics_stage/3"


def test_lifecycle_scoring_stops_at_terminal_without_inventing_lost_member_geometry():
    samples = [dict(time_s=1.,phase="assembly",members=[{"id":"a"},{"id":"b"}]),
        dict(time_s=2.,phase="abort",members=[{"id":"a"}],reason="missing_member_report"),
        dict(time_s=3.,phase="abort",members=[])]
    selected = lifecycle_window(samples)
    assert selected == samples[:2]
    assert selected[-1]["members"] == [{"id":"a"}]
    assert len(samples) == 3
