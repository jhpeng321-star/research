"""Contract tests only; these tests never launch SUMO."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

import traffic_shepherd.gentle_mechanics as module
from traffic_shepherd.ownership import CommandRecord
from traffic_shepherd.paired_experiments import PolicyOutput, digest


def test_fixed_two_case_budget_includes_warmup_and_keeps_law():
    plan = module.plan()
    assert plan["maximum_simulated_seconds"] == 240
    assert plan["maximum_attempts"] == 2 and plan["automatic_retries"] == 0
    assert plan["horizon_s"] == 120 and plan["warmup_included_s"] == plan["decision_s"] == 5
    assert plan["execution_order"] == ["gentle_complete", "gentle_report_loss"]
    assert plan["cases"][1]["packet_drop_from_s"] == 15
    assert plan["profile"]["hold_s"] == 20 and plan["profile"]["assembly_deadline_s"] == 50
    assert plan["profile"]["acceleration_min_mps2"] == -.3
    assert plan["profile"]["acceleration_change_limit_mps3"] == .15
    assert plan["mechanical_not_traffic_efficacy"]


def test_fixture_uses_inserted_idm_and_no_speed_commands(tmp_path, monkeypatch):
    calls = []
    def fake_conversion(command, **kwargs):
        calls.append(command)
        Path(command[-1]).write_text("<net/>")
        return SimpleNamespace(stdout="synthetic fixture test", stderr="", check_returncode=lambda: None)
    monkeypatch.setattr(module.subprocess, "run", fake_conversion)
    scene, requests = module.build_fixture(tmp_path/"fixture", "C:/SUMO/sumo.exe")
    assert len(calls) == 1 and "netconvert" in calls[0][0]
    root = ET.parse(scene.parent/"routes.rou.xml").getroot()
    assert root.find("vType").attrib["carFollowModel"] == "IDM"
    assert root.find("vType").attrib["maxSpeed"] == "24"
    vehicles = {v.attrib["id"]: v.attrib for v in root.findall("vehicle")}
    assert set(vehicles) == set((*module.MEMBERS, *module.FOLLOWERS))
    for member, follower in zip(module.MEMBERS, module.FOLLOWERS):
        assert float(vehicles[member]["departPos"])-float(vehicles[follower]["departPos"])-5 == 85
        assert vehicles[member]["departLane"] == vehicles[follower]["departLane"]
    assert all(v["departSpeed"] == "24" for v in vehicles.values())
    assert {r.vehicle_id for r in requests if r.is_cav} == set(module.MEMBERS)
    assert ET.parse(scene.parent/"edges.xml").getroot().find("edge").attrib["speed"] == "24"


def record(accel=-.15, **kwargs):
    return CommandRecord("member_a", 5., "wall", None, -.15, accel, 24.+accel,
                         None, False, **kwargs)


def test_actual_issued_saturation_and_slew_are_checked_independently():
    out = PolicyOutput({"member_a": -.15}, "assembly", member_ids=("member_a",))
    frames = {"member_a": SimpleNamespace(acceleration_mps2=0.)}
    assert module.command_failures(out, [record()], {}, frames) == []
    assert "actual_issued_wall_bound_or_identity" in module.command_failures(out, [record(-1.)], {}, frames)
    assert "actual_issued_wall_bound_or_identity" in module.command_failures(out, [record(-.2)], {}, frames)
    assert "actual_issued_entry_or_command_slew" in module.command_failures(out, [record()], {"member_a": .15}, frames)
    assert "actual_issued_entry_or_command_slew" in module.command_failures(out, [record()], {},
        {"member_a": SimpleNamespace(acceleration_mps2=-1.)})
    assert "actual_issued_wall_bound_or_identity" in module.command_failures(out, [replace(record(), owner="native")], {}, frames)


def test_command_checker_rejects_missing_duplicate_and_unrequested_owner():
    out = PolicyOutput({"member_a": -.15}, "assembly", member_ids=("member_a",))
    frames = {"member_a": SimpleNamespace(acceleration_mps2=0.)}
    assert module.command_failures(out, [], {}, frames)
    assert "duplicate_or_foreign_command_record" in module.command_failures(out, [record(), record()], {}, frames)
    assert "unrequested_wall_owner" in module.command_failures(PolicyOutput(), [record()], {}, frames)


def fixture_registration(tmp_path, monkeypatch):
    package = tmp_path/"traffic_shepherd"
    package.mkdir()
    (package/"gentle_wall.py").write_text("# mocked source; no simulator was run")
    adapter = tmp_path/"adapter.json"
    adapter.write_text("{}")
    identity = dict(active_source_sha256={"gentle_wall.py": module.hash_file(package/"gentle_wall.py")},
                    config_sha256=module.hash_file(adapter))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "CONFIG_PATH", adapter)
    monkeypatch.setattr(module, "_identity", lambda *args: (identity, "mocked_sumo_never_launched"))
    result = module.register(tmp_path/"registered")
    root, data = module.read_registration(result["registration"])
    return root, data, identity


def mock_completed_case(root, data, case_id):
    """Build synthetic integrity-test bytes; never represent physical evidence."""
    output = root/"jobs"/case_id
    output.mkdir(parents=True)
    (output/"scenario").mkdir()
    allocation = dict(case_id=case_id, allocated_simulated_seconds=120., retries=0,
                      registration_sha256=data["registration_sha256"])
    (root/"allocations").mkdir(exist_ok=True)
    module._write_json(root/"allocations"/f"{case_id}.json", allocation)
    for name in (*module.ARTIFACTS, "scenario/network.net.xml", "scenario/routes.rou.xml", "scenario/scenario.sumocfg"):
        (output/name).write_text("synthetic integrity fixture only\n")
    summary = dict(case_id=case_id, status="complete", passed=True, checks={"synthetic_integrity_test": True},
        registration_sha256=data["registration_sha256"], completed_simulated_seconds=120.)
    module._write_json(output/"summary.json", summary)
    manifest = dict(summary, artifact_sha256={name: module.hash_file(output/name) for name in
        (*module.ARTIFACTS, "summary.json", "scenario/network.net.xml", "scenario/routes.rou.xml", "scenario/scenario.sumocfg")})
    module._write_json(output/"manifest.json", manifest)
    return output


def test_registration_snapshots_and_hashes_without_launch(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("registration must not launch SUMO")
    monkeypatch.setattr(module, "start_sumo", forbidden)
    monkeypatch.setattr(module.subprocess, "run", forbidden)
    root, data, identity = fixture_registration(tmp_path, monkeypatch)
    assert data["identity"] == identity
    assert (root/"source_snapshot/gentle_wall.py").exists()
    assert not (root/"allocations").exists()
    assert data["registration_sha256"] == digest({k: v for k, v in data.items() if k != "registration_sha256"})
    with pytest.raises(FileExistsError):
        module.register(root)


def test_source_registration_or_snapshot_change_rejected(tmp_path, monkeypatch):
    root, data, identity = fixture_registration(tmp_path, monkeypatch)
    identity["config_sha256"] = "changed"
    with pytest.raises(ValueError, match="changed after"):
        module.read_registration(root/"registration.json")
    identity["config_sha256"] = data["identity"]["config_sha256"]
    (root/"source_snapshot/gentle_wall.py").write_text("changed")
    with pytest.raises(ValueError, match="snapshot changed"):
        module.read_registration(root/"registration.json")


def test_budget_mutation_rejected_even_with_recomputed_digest(tmp_path, monkeypatch):
    root, data, _ = fixture_registration(tmp_path, monkeypatch)
    data["maximum_simulated_seconds"] = 241
    data["registration_sha256"] = digest({k: v for k, v in data.items() if k != "registration_sha256"})
    module._write_json(root/"registration.json", data)
    with pytest.raises(ValueError, match="exact bounded plan"):
        module.read_registration(root/"registration.json")


def test_completed_verifier_requires_both_outputs_and_exact_bytes(tmp_path, monkeypatch):
    root, data, _ = fixture_registration(tmp_path, monkeypatch)
    first = mock_completed_case(root, data, module.CASE_IDS[0])
    with pytest.raises(ValueError, match="both exact"):
        module.verify_completed(root/"registration.json")
    mock_completed_case(root, data, module.CASE_IDS[1])
    result = module.verify_completed(root/"registration.json")
    assert result["passed"] and result["actual_simulated_seconds"] == 240
    assert not result["physical_traffic_benefit_established"]
    (first/"physical.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="evidence changed"):
        module.verify_completed(root/"registration.json")


def test_no_second_allocation_before_first_passes_and_no_retry(tmp_path, monkeypatch):
    root, data, _ = fixture_registration(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError):
        module.run_one(root/"registration.json", module.CASE_IDS[1])
    assert not (root/"allocations").exists()
    calls = []
    monkeypatch.setattr(module, "_run_allocated", lambda *args: calls.append(args) or {"passed": True})
    module.run_one(root/"registration.json", module.CASE_IDS[0])
    with pytest.raises(FileExistsError):
        module.run_one(root/"registration.json", module.CASE_IDS[0])
    assert len(calls) == 1


def test_failed_implementation_charged_and_stops_second(tmp_path, monkeypatch):
    root, data, _ = fixture_registration(tmp_path, monkeypatch)
    def fail(*args):
        raise RuntimeError("synthetic implementation failure before simulation")
    monkeypatch.setattr(module, "_run_allocated", fail)
    with pytest.raises(RuntimeError, match="synthetic implementation"):
        module.run_one(root/"registration.json", module.CASE_IDS[0])
    assert (root/"allocations"/f"{module.CASE_IDS[0]}.json").exists()
    assert (root/"STOP.json").exists()
    with pytest.raises(ValueError, match="stopped"):
        module.run_one(root/"registration.json", module.CASE_IDS[1])


def test_diagnostics_separate_members_followers_and_retain_mild_braking():
    rows = [dict(all_vehicles=[dict(id="member_a", acceleration_mps2=-.4),
                              dict(id="follower_a", acceleration_mps2=-1.)]),
            dict(all_vehicles=[dict(id="member_a", acceleration_mps2=.2),
                              dict(id="follower_a", acceleration_mps2=2.)])]
    result = module._physical_diagnostics(rows)
    assert result["members"]["peak_measured_deceleration_mps2"] == .4
    assert result["members"]["exposure_vehicle_seconds"]["-0.3"] == .5
    assert result["followers"]["maximum_halfsecond_acceleration_change_mps2"] == 3
    assert result["all"]["integrated_squared_acceleration_m2_per_s3"] == pytest.approx(2.6)
    assert result["all"]["severe_braking_vehicle_seconds"] == 0
