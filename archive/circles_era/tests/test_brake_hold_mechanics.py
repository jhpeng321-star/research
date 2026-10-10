"""No SUMO execution: bounded registration, issued commands and lifecycle score."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

import traffic_shepherd.brake_hold_mechanics as module
from traffic_shepherd.brake_hold_policy import BrakeHoldProfile
from traffic_shepherd.ownership import CommandRecord
from traffic_shepherd.paired_experiments import PolicyOutput


def test_exact_high_cap_endpoint_budget_and_unchanged_fixture():
    plan = module.plan()
    assert plan["maximum_simulated_seconds"] == 240 and plan["maximum_attempts"] == 2
    assert plan["horizon_s"] == 120 and plan["decision_s"] == plan["warmup_included_s"] == 5
    assert not plan["automatic_retries"] and not plan["automatic_successors"]
    assert plan["execution_order"] == ["strong_hold_zero","strong_hold_forty"]
    assert [c["profile"]["hold_s"] for c in plan["cases"]] == [0.,40.]
    assert all(c["profile"]["acceleration_min_mps2"] == -3. for c in plan["cases"])
    assert all(c["profile"]["acceleration_change_limit_mps3"] == .15 for c in plan["cases"])
    assert plan["fixture"]["source"] == "gentle_mechanics.build_fixture unchanged"


def samples(hold):
    end = 9+int(hold)
    rows = []
    for t in range(5,end+1):
        phase = "assembly" if t < 8 else "hold" if t < 8+hold else "taper" if t < end else "release"
        rows.append(dict(time_s=float(t),phase=phase,target_speed_mps=22.,
            assembly_completed_at_s=None if t < 8 else 8.,
            members=[dict(id=v,lane=lane,distance_m=1000.,speed_mps=24. if t == 5 else 22.)
                     for v,lane in zip(module.MEMBERS,module.LANES)],
            owners={v:"native" if phase == "release" else "wall" for v in module.MEMBERS},
            wall_command_ids=[] if phase == "release" else list(module.MEMBERS)))
    return rows


@pytest.mark.parametrize("hold",[0.,40.])
def test_independent_readiness_and_h0_h40_lifecycles(hold):
    out = module.lifecycle_checks(samples(hold),BrakeHoldProfile(acceleration_min_mps2=-3.,hold_s=hold))
    assert all(out["checks"].values()) and out["first_measured_ready_s"] == 8.
    assert out["hold_seconds"] == hold and out["taper_command_seconds"] == 1


def test_scorer_rejects_premature_readiness_missing_ticks_and_false_native_release():
    profile = BrakeHoldProfile(acceleration_min_mps2=-3.,hold_s=0.)
    missing = samples(0)
    missing.pop(1)
    assert not module.lifecycle_checks(missing,profile)["checks"]["measured_lifecycle_integrity"]
    unready = samples(0)
    unready[1]["members"][0]["speed_mps"] = 26.
    assert not module.lifecycle_checks(unready,profile)["checks"]["measured_lifecycle_integrity"]
    unreleased = samples(0)
    unreleased[-1]["owners"][module.MEMBERS[0]] = "wall"
    assert not module.lifecycle_checks(unreleased,profile)["checks"]["terminal_release_within_transition_budget"]


def test_scorer_does_not_treat_h0_as_damping_or_shortened_h40_as_complete():
    wrong = samples(0)
    wrong[0]["phase"] = "damping"
    assert not module.lifecycle_checks(wrong,BrakeHoldProfile(hold_s=0.))["checks"]["measured_lifecycle_integrity"]
    assert not module.lifecycle_checks(samples(0),BrakeHoldProfile(hold_s=40.))["checks"]["exact_registered_hold_duration"]


def test_actual_high_cap_command_and_measured_slew_checks():
    p = BrakeHoldProfile(acceleration_min_mps2=-3.,hold_s=0.)
    out = PolicyOutput({"member_a":-.45},"assembly",member_ids=("member_a",))
    record = CommandRecord("member_a",7.,"wall",None,-.45,-.45,23.,None,False)
    frames = {"member_a":SimpleNamespace(acceleration_mps2=-.3)}
    assert module.command_failures(out,[record],{"member_a":-.3},frames,p) == []
    assert module.command_failures(out,[replace(record,issued_acceleration_mps2=-3.1)],{},frames,p)
    assert module.command_failures(out,[replace(record,issued_acceleration_mps2=-.3)],{},frames,p)
    assert module.command_failures(out,[record],{"member_a":0.},frames,p)
    assert module.command_failures(out,[replace(record,owner="native")],{},frames,p)
    assert module.command_failures(out,[record,record],{},frames,p)


def registration(tmp_path,monkeypatch):
    package = tmp_path/"traffic_shepherd"
    package.mkdir()
    (package/"brake_hold_policy.py").write_text("# synthetic integrity test, never simulated")
    adapter = tmp_path/"adapter.json"
    adapter.write_text("{}")
    identity = dict(config_sha256=module.hash_file(adapter),active_source_sha256={
        "brake_hold_policy.py":module.hash_file(package/"brake_hold_policy.py")})
    monkeypatch.setattr(module,"ROOT",tmp_path)
    monkeypatch.setattr(module,"CONFIG_PATH",adapter)
    monkeypatch.setattr(module,"_identity",lambda *args:(identity,"mocked_sumo_never_executed"))
    result = module.register(tmp_path/"registered")
    return module.read_registration(result["registration"])


def test_registration_and_source_snapshot_mutation(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError("registration cannot launch a simulator")
    monkeypatch.setattr(module,"start_sumo",forbidden)
    root,data = registration(tmp_path,monkeypatch)
    assert not (root/"allocations").exists()
    assert data["schema"] == "brake_hold_mechanics/1"
    with pytest.raises(FileExistsError):
        module.register(root)
    (root/"source_snapshot/brake_hold_policy.py").write_text("changed")
    with pytest.raises(ValueError,match="snapshot changed"):
        module.read_registration(root/"registration.json")


def test_second_case_requires_first_proof_and_no_retry(tmp_path,monkeypatch):
    root,data = registration(tmp_path,monkeypatch)
    with pytest.raises(FileNotFoundError):
        module.run_one(root/"registration.json",module.CASE_IDS[1])
    assert not (root/"allocations").exists()
    monkeypatch.setattr(module,"_run_allocated",lambda *args:{"passed":True})
    module.run_one(root/"registration.json",module.CASE_IDS[0])
    with pytest.raises(FileExistsError):
        module.run_one(root/"registration.json",module.CASE_IDS[0])


def test_failure_is_charged_and_stops_stage(tmp_path,monkeypatch):
    root,data = registration(tmp_path,monkeypatch)
    def fail(*args):
        raise RuntimeError("synthetic prelaunch failure")
    monkeypatch.setattr(module,"_run_allocated",fail)
    with pytest.raises(RuntimeError):
        module.run_one(root/"registration.json",module.CASE_IDS[0])
    assert (root/"allocations"/f"{module.CASE_IDS[0]}.json").exists()
    assert (root/"STOP.json").exists()
    with pytest.raises(ValueError,match="stopped"):
        module.run_one(root/"registration.json",module.CASE_IDS[1])


def test_completed_verifier_requires_exact_allocations_and_hash_checks(tmp_path,monkeypatch):
    root,data = registration(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match="both"):
        module.verify_completed(root/"registration.json")
    (root/"allocations").mkdir()
    for name in module.CASE_IDS:
        (root/"allocations"/f"{name}.json").write_text("{}")
    calls = []
    monkeypatch.setattr(module,"_verify_case",lambda r,d,c: calls.append(c) or {"case_id":c,"passed":True})
    proof = module.verify_completed(root/"registration.json")
    assert calls == list(module.CASE_IDS) and proof["actual_simulated_seconds"] == 240
    assert not proof["physical_traffic_benefit_established"]
    (root/"allocations/extra.json").write_text("{}")
    with pytest.raises(ValueError,match="both"):
        module.verify_completed(root/"registration.json")
