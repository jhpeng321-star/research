"""Pure mechanics contracts. None of these tests launches SUMO or netconvert."""
from copy import deepcopy
import json
from dataclasses import replace
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

import traffic_shepherd.lifecycle_mechanics as module
from traffic_shepherd.lifecycle_wall import LifecycleProfile
from traffic_shepherd.damping import DampingDecision
from traffic_shepherd.ownership import CommandRecord
from traffic_shepherd.paired_experiments import PolicyOutput


def test_exact_single_new_fixture_budget_with_normal_reuse():
    plan = module.plan()
    assert plan["maximum_simulated_seconds"] == 120 and plan["maximum_attempts"] == 1
    assert plan["horizon_s"] == 120 and plan["warmup_included_s"] == plan["decision_s"] == 5
    assert not plan["automatic_retries"] and not plan["automatic_successors"]
    assert plan["execution_order"] == ["developing_constraint"]
    assert plan["reused_passing_simulated_seconds"] == 120
    assert plan["prior_closed_b2_simulated_seconds"] == 240
    assert [c["evidence_mode"] for c in plan["cases"]] == ["verified_b2_reuse","new_allocation"]
    assert [c["expected_early_exit"] for c in plan["cases"]] == [False,True]
    assert all(c["profile"]["acceleration_min_mps2"] == -.3 for c in plan["cases"])
    assert all(c["profile"]["acceleration_change_limit_mps3"] == .15 for c in plan["cases"])
    assert all(c["profile"]["hold_s"] == 20 for c in plan["cases"])


def test_fixture_xml_has_one_native_hdv_route_stop_without_runtime_brake_commands():
    normal = module.fixture_routes(module.CASE_IDS[0])
    changed = module.fixture_routes(module.CASE_IDS[1])
    assert len(normal.findall("vehicle")) == 4 and len(changed.findall("vehicle")) == 6
    assert len(normal.findall("vehicle/stop")) == 0 and len(changed.findall("vehicle/stop")) == 1
    by_id = {v.attrib["id"]:v for v in changed.findall("vehicle")}
    assert by_id[module.STOP_LEADER].find("stop").attrib == dict(lane="road_0",endPos="1665",duration="120")
    assert by_id[module.STOP_LEADER].attrib["departSpeed"] == "max"
    assert by_id[module.REPORTER].find("stop") is None
    assert {by_id[v].attrib["departSpeed"] for v in module.MEMBERS} == {"20.0"}
    assert all(n.attrib["carFollowModel"] == "IDM" for n in changed.findall("vType"))
    with pytest.raises(ValueError):
        module.fixture_routes("additional_unbudgeted_fixture")


def test_only_stop_leader_departure_semantics_changed_from_saved_b2_xml():
    def canonical(root):
        return ET.canonicalize(ET.tostring(root,encoding="unicode"),strip_text=True)
    for case in module.CASE_IDS:
        old = ET.parse(module.REUSE_REGISTRATION.parent/"jobs"/case/"scenario/routes.rou.xml").getroot()
        if case == module.CASE_IDS[1]:
            node = next(v for v in old.findall("vehicle") if v.attrib["id"] == module.STOP_LEADER)
            assert node.attrib["departSpeed"] == "24.0"
            node.attrib["departSpeed"] = "max"
        assert canonical(old) == canonical(module.fixture_routes(case))
    assert module.native_stop_insertion_speed() == pytest.approx(23.879051675553654)


def test_exact_first_halfsecond_insertion_prevents_absent_constraint_fixture():
    ids = (*module.MEMBERS,*module.FOLLOWERS,module.REPORTER,module.STOP_LEADER)
    sample = dict(time_s=.5,all_vehicles=[dict(id=v) for v in ids])
    assert module.fixture_insertion_verified([sample])
    assert not module.fixture_insertion_verified([])
    assert not module.fixture_insertion_verified([dict(sample,time_s=1.)])
    assert not module.fixture_insertion_verified([dict(sample,all_vehicles=sample["all_vehicles"][:-1])])


def test_reuse_allows_only_two_registered_module_changes():
    old = dict(config_sha256="config",active_source_sha256={"lifecycle_wall.py":"policy",
        "lifecycle_mechanics.py":"old_runner","lifecycle_study.py":"old_study"})
    current = deepcopy(old)
    current["active_source_sha256"].update({"lifecycle_mechanics.py":"new_runner","lifecycle_study.py":"new_study"})
    assert set(module.reuse_source_equivalence(old,current)) == module.ALLOWED_REUSE_SOURCE_CHANGES
    for bad in (dict(current,config_sha256="different"),
                dict(current,active_source_sha256=dict(current["active_source_sha256"],**{"lifecycle_wall.py":"different"})),
                dict(current,active_source_sha256=dict(current["active_source_sha256"],extra="added"))):
        with pytest.raises(ValueError,match="identical"):
            module.reuse_source_equivalence(old,bad)


def test_real_b2_normal_reuse_keeps_failed_whole_stage_closed():
    # Audit the completed B3 proof under its registered source identity. Later
    # additive studies cannot retroactively change that historical execution.
    registered = json.loads((module.ROOT/"runs/active/lifecycle_mechanics_v3/registration.json").read_text())
    proof = module.reused_normal_evidence(registered["identity"])
    assert proof["passed"] and proof["mode"] == "verified_reuse"
    assert proof["case_id"] == module.CASE_IDS[0]
    assert proof["newly_allocated_simulated_seconds"] == 0
    assert proof["preserved_closed_b2"]["actual_simulated_seconds"] == 240
    assert all(proof["measured_lifecycle_score"]["checks"].values())
    assert (module.REUSE_REGISTRATION.parent/"STOP.json").exists()


def test_pure_entry_and_stop_timing_forecast_is_nonvacuous_without_sumo(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError("pure preflight cannot execute SUMO")
    monkeypatch.setattr(module,"start_sumo",forbidden)
    monkeypatch.setattr(module.subprocess,"run",forbidden)
    entry,timing = module.entry_preflight(),module.constraint_timing_preflight()
    assert entry["passed"] and timing["passed"]
    assert entry["simulated_seconds"] == timing["simulated_seconds"] == 0
    assert entry["insertion_time_s"] == timing["insertion_time_s"] == .5
    assert entry["acceleration_measurement_interval_s"] == timing["acceleration_measurement_interval_s"] == .5
    assert entry["cases"][module.CASE_IDS[0]]["initial_acceleration_mps2"] == pytest.approx(.31855186227641497)
    assert entry["cases"][module.CASE_IDS[0]]["evidence_source"] == "saved_actual_acquired_startup"
    assert entry["saved_entry_provenance"]["prior_failed_simulated_seconds"] == 120.
    assert all(r["initial_acceleration_mps2"] > 0 for r in entry["cases"].values())
    assert all(r["lifecycle_plan"]["remaining_hold_s"] == 20 for r in entry["cases"].values())
    hold = [r for r in timing["timeline"] if r["phase"] == "hold"]
    trigger = next(r for r in timing["timeline"] if r["commitment_feasible"] is False)
    assert 0 < len(hold) < 20 and trigger["time_s"] > hold[0]["time_s"]
    assert trigger["commitment_traffic_margin_m"] < 0 < trigger["exit_now_traffic_margin_m"]
    assert timing["timeline"][-1]["phase"] == "release"


def samples(early=False,hold_entry=8):
    exit_at = hold_entry+(5 if early else 20)
    rows = []
    for t in range(5,exit_at+2):
        phase = "assembly" if t < hold_entry else "hold" if t < exit_at else "taper" if t == exit_at else "release"
        plans = dict(commitment=dict(feasible=True,maintainable_hold_ready_now=t >= hold_entry,
            predicted_hold_start_s=float(hold_entry),remaining_hold_s=20.,remaining_assembly_s=max(0,hold_entry-t)))
        trigger = early and t == exit_at
        if trigger:
            plans = dict(commitment=dict(feasible=False,failure_kind="traffic"),exit_now=dict(feasible=True))
        rows.append(dict(time_s=float(t),phase=phase,target_speed_mps=22.,
            assembly_completed_at_s=None if t < hold_entry else float(hold_entry),
            exit_started_at_s=None if t < exit_at else float(exit_at),
            early_exit_reason="reported_traffic_intersects_remaining_maneuver" if early and t >= exit_at else None,
            lifecycle_plan=plans,members=[dict(id=v,lane=lane,distance_m=1000.,
                speed_mps=24. if t == 5 else 22.,acceleration_mps2=.2 if t == 5 else 0.)
                for v,lane in zip(module.MEMBERS,module.LANES)],
            owners={v:"native" if phase == "release" else "wall" for v in module.MEMBERS},
            wall_command_ids=[] if phase == "release" else list(module.MEMBERS),
            wall_commands={} if phase == "release" else {v:0. if phase == "taper" else -.1 for v in module.MEMBERS}))
    return rows


@pytest.mark.parametrize("early",[False,True])
def test_measured_readiness_full_or_early_hold_and_coast_release(early):
    out = module.lifecycle_checks(samples(early),LifecycleProfile(),expected_early_exit=early)
    assert all(out["checks"].values())
    assert out["first_measured_ready_s"] == 8.
    assert out["hold_seconds"] == (5 if early else 20)
    assert out["taper_command_seconds"] == 1


def test_hold_entry_can_follow_earlier_geometric_readiness_but_needs_current_streak():
    rows = samples(hold_entry=10)
    out = module.lifecycle_checks(rows,LifecycleProfile())
    assert all(out["checks"].values())
    assert out["first_geometric_ready_at_s"] == 8. and out["actual_hold_started_at_s"] == 10.
    next(r for r in rows if r["time_s"] == 9.)["members"][0]["speed_mps"] = 26.
    out = module.lifecycle_checks(rows,LifecycleProfile())
    assert not out["checks"]["three_target_settled_readiness_samples"]
    assert not out["checks"]["maintainable_hold_entry_verified"]


@pytest.mark.parametrize("field,value",[("feasible",False),("maintainable_hold_ready_now",False),
    ("predicted_hold_start_s",9.),("remaining_hold_s",19.),("remaining_assembly_s",1.)])
def test_hold_entry_requires_complete_current_maintainability_proof(field,value):
    rows = samples()
    next(r for r in rows if r["phase"] == "hold")["lifecycle_plan"]["commitment"][field] = value
    assert not module.lifecycle_checks(rows,LifecycleProfile())["checks"]["maintainable_hold_entry_verified"]


def test_empty_commands_hold_and_already_native_comparison_are_not_delivery():
    assert not module.positive_entry_commands_verified([])
    assert not any(module.hold_physical_checks([],0.).values())
    native = [dict(vehicle_id=v,matched=True,prior_owner="native") for v in module.MEMBERS]
    assert not module.actual_damping_handover_verified(native,native)
    commands = [dict(vehicle_id=v,owner="wall",timestamp_s=5.,issued_acceleration_mps2=.1) for v in module.MEMBERS]
    assert module.positive_entry_commands_verified(commands)
    assert not module.positive_entry_commands_verified(commands[:1])
    handover = [dict(vehicle_id=v,matched=True,prior_owner="wall") for v in module.MEMBERS]
    assert module.actual_damping_handover_verified(handover,handover)
    hold = dict(all_vehicles=[dict(id=v,lane=lane,distance_m=1000.,speed_mps=22.)
                             for v,lane in zip(module.MEMBERS,module.LANES)])
    assert all(module.hold_physical_checks([hold,hold],1.).values())


def test_scorer_rejects_unready_missing_tick_and_unreleased_owner():
    for edit in (lambda rows:rows.pop(1),
                 lambda rows:rows[1]["members"][0].update(speed_mps=26.)):
        rows = samples()
        edit(rows)
        assert not module.lifecycle_checks(rows,LifecycleProfile())["checks"]["measured_lifecycle_integrity"]
    rows = samples()
    rows[-1]["owners"][module.MEMBERS[0]] = "wall"
    assert not module.lifecycle_checks(rows,LifecycleProfile())["checks"]["terminal_release_within_transition_budget"]


def test_scorer_rejects_fake_zero_native_and_out_of_bound_measured_exit():
    rows = samples()
    rows[-2]["wall_commands"][module.MEMBERS[0]] = -.1
    assert not module.lifecycle_checks(rows,LifecycleProfile())["checks"]["actual_measured_coast_terminal"]
    rows = samples()
    rows[-1]["members"][1]["acceleration_mps2"] = -.8
    assert not module.lifecycle_checks(rows,LifecycleProfile())["checks"]["actual_measured_coast_terminal"]


def test_scorer_requires_first_during_hold_traffic_failure_and_feasible_exit():
    original = samples(True)
    for field,value in (("failure_kind","map_boundary"),("feasible",True)):
        rows = deepcopy(original)
        rows[-2]["lifecycle_plan"]["commitment"][field] = value
        assert not module.lifecycle_checks(rows,LifecycleProfile(),expected_early_exit=True)["checks"]["timely_during_hold_constraint_exit"]
    rows = deepcopy(original)
    rows[-2]["lifecycle_plan"]["exit_now"]["feasible"] = False
    assert not module.lifecycle_checks(rows,LifecycleProfile(),expected_early_exit=True)["checks"]["timely_during_hold_constraint_exit"]
    rows = deepcopy(original)
    rows[1]["lifecycle_plan"] = deepcopy(rows[-2]["lifecycle_plan"])
    assert not module.lifecycle_checks(rows,LifecycleProfile(),expected_early_exit=True)["checks"]["timely_during_hold_constraint_exit"]
    assert not module.lifecycle_checks(original,LifecycleProfile())["checks"]["registered_actual_hold_duration"]


def test_actual_issued_gentle_bounds_ramp_and_nonmember_mask():
    profile = LifecycleProfile()
    tick = PolicyOutput({"member_a":-.3},"assembly",member_ids=("member_a",))
    record = CommandRecord("member_a",7.,"wall",None,-.3,-.3,23.,None,False)
    frames = {"member_a":SimpleNamespace(acceleration_mps2=-.15)}
    assert module.command_failures(tick,[record],{"member_a":-.15},frames,profile) == []
    assert module.command_failures(tick,[replace(record,issued_acceleration_mps2=-.31)],{},frames,profile)
    assert module.command_failures(tick,[record],{"member_a":0.},frames,profile)
    assert module.command_failures(tick,[record,record],{},frames,profile)
    assert module.command_failures(PolicyOutput({module.REPORTER:0.},"hold",member_ids=(module.REPORTER,)),[],{},frames,profile)


def test_measured_diagnostics_include_downstream_native_stop_separately():
    physical = [dict(all_vehicles=[dict(id="member_a",acceleration_mps2=-.2),
        dict(id=module.STOP_LEADER,acceleration_mps2=-4.)])]
    out = module.physical_diagnostics(physical)
    assert out["all"]["samples"] == 2
    assert out["all"]["minimum_measured_acceleration_mps2"] == -4.
    assert out["members"]["minimum_measured_acceleration_mps2"] == -.2
    assert out["native_downstream_fixture"]["minimum_measured_acceleration_mps2"] == -4.


def test_real_command_owner_distinguishes_coast_from_native_and_records_measured_response():
    writes = []
    api = SimpleNamespace(vehicle=SimpleNamespace(getSpeedMode=lambda v:31,
        slowDown=lambda v,s,t:writes.append(("slowDown",v,s,t)),
        setSpeed=lambda v,s:writes.append(("setSpeed",v,s))))
    controller = SimpleNamespace(config=module.load_config(),decide=lambda f,now_s:DampingDecision(
        fallback_reason="missing_or_uncovered_leader"))
    owner = module.CommandOwnership(api,module.MEMBERS,controller)
    frame = module.SensorFrame("member_a",5.,22.,0.,24.,0,2000.,front_coverage_m=100.,rear_coverage_m=100.)
    issued = owner.step({"member_a":frame},now_s=5.,live_ids={"member_a"},wall_commands={"member_a":0.})[0]
    assert issued.owner == "wall" and issued.issued_acceleration_mps2 == 0.
    assert writes[-1][0] == "slowDown" and not issued.released_to_native
    measured = owner.record_response("member_a",timestamp_s=5.5,speed_mps=21.6,acceleration_mps2=-.8)
    assert measured.owner == "wall" and measured.command_timestamp_s == 5. and measured.measured_acceleration_mps2 == -.8
    returned = owner.step({"member_a":replace(frame,timestamp_s=6.)},now_s=6.,live_ids={"member_a"})[0]
    assert returned.owner == "native" and returned.released_to_native
    assert writes[-1] == ("setSpeed","member_a",-1)
    assert owner.controller is controller


@pytest.mark.parametrize("rows,expected",[
    ([],False),
    ([dict(baseline_native=False,actual_override_release=False)],False),
    ([dict(baseline_native=True,actual_override_release=False)],False),
    ([dict(baseline_native=True,actual_override_release=True)],True),
    ([dict(baseline_native=True,actual_override_release=True),
      dict(baseline_native=False,actual_override_release=False)],True),
    ([dict(baseline_native=True,actual_override_release=True),
      dict(baseline_native=True,actual_override_release=False)],False),
])
def test_native_release_check_requires_nonvacuous_actual_release(rows,expected):
    assert module.actual_native_release_verified(rows) is expected


def test_constraint_trigger_requires_acquired_reporter_and_physical_slowing():
    rows = samples(True)
    rows[-2]["lifecycle_plan"]["commitment"]["fresh_downstream_reporter_ids"] = [module.REPORTER]
    physical = [dict(time_s=t,acquired_member_frames={module.REPORTER:dict(timestamp_s=t,
        speed_mps=24. if t == 5. else 9.,acceleration_mps2=0. if t == 5. else -2.)},
        all_vehicles=[dict(id=module.STOP_LEADER,acceleration_mps2=-1.)]) for t in (5.,13.)]
    assert all(module.developing_constraint_checks(rows,physical).values())
    physical[-1]["acquired_member_frames"][module.REPORTER]["timestamp_s"] = 12.
    assert not module.developing_constraint_checks(rows,physical)["trigger_uses_current_acquired_downstream_reporter"]
    physical[-1]["acquired_member_frames"][module.REPORTER]["speed_mps"] = 24.
    assert not module.developing_constraint_checks(rows,physical)["downstream_reporter_physically_slowed_since_entry"]


def registration(tmp_path,monkeypatch):
    package = tmp_path/"traffic_shepherd"
    package.mkdir()
    (package/"lifecycle_wall.py").write_text("# synthetic integrity test, never simulated")
    adapter = tmp_path/"adapter.json"
    adapter.write_text("{}")
    identity = dict(config_sha256=module.hash_file(adapter),active_source_sha256={
        "lifecycle_wall.py":module.hash_file(package/"lifecycle_wall.py")})
    monkeypatch.setattr(module,"ROOT",tmp_path)
    monkeypatch.setattr(module,"CONFIG_PATH",adapter)
    monkeypatch.setattr(module,"_identity",lambda *args:(identity,"mocked_sumo_never_executed"))
    monkeypatch.setattr(module,"reused_normal_evidence",lambda identity:dict(case_id=module.CASE_IDS[0],passed=True,
        mode="verified_reuse",actual_simulated_seconds=120,newly_allocated_simulated_seconds=0))
    result = module.register(tmp_path/"registered")
    return module.read_registration(result["registration"])


def test_registration_preserves_preflight_and_source_snapshot(tmp_path,monkeypatch):
    monkeypatch.setattr(module,"start_sumo",lambda *args:pytest.fail("registration cannot start SUMO"))
    root,data = registration(tmp_path,monkeypatch)
    assert not (root/"allocations").exists()
    assert data["entry_preflight"]["passed"] and data["constraint_timing_preflight"]["passed"]
    with pytest.raises(FileExistsError):
        module.register(root)
    (root/"source_snapshot/lifecycle_wall.py").write_text("changed")
    with pytest.raises(ValueError,match="snapshot changed"):
        module.read_registration(root/"registration.json")


def test_normal_is_reused_never_allocated_and_new_case_cannot_retry(tmp_path,monkeypatch):
    root,data = registration(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match="normal rerun"):
        module.run_one(root/"registration.json",module.CASE_IDS[0])
    assert not (root/"allocations").exists()
    monkeypatch.setattr(module,"_run_allocated",lambda *args:{"passed":True})
    module.run_one(root/"registration.json",module.CASE_IDS[1])
    assert {p.stem for p in (root/"allocations").glob("*.json")} == set(module.NEW_CASE_IDS)
    with pytest.raises(FileExistsError):
        module.run_one(root/"registration.json",module.CASE_IDS[1])


def test_failed_allocation_stops_stage(tmp_path,monkeypatch):
    root,data = registration(tmp_path,monkeypatch)
    def fail(*args):
        raise RuntimeError("synthetic prelaunch failure")
    monkeypatch.setattr(module,"_run_allocated",fail)
    with pytest.raises(RuntimeError):
        module.run_one(root/"registration.json",module.CASE_IDS[1])
    assert (root/"allocations"/f"{module.CASE_IDS[1]}.json").exists()
    assert (root/"STOP.json").exists()
    with pytest.raises(ValueError,match="stopped"):
        module.run_one(root/"registration.json",module.CASE_IDS[1])


def test_completed_verifier_requires_exact_allocations_and_hashes(tmp_path,monkeypatch):
    root,data = registration(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match="exactly"):
        module.verify_completed(root/"registration.json")
    (root/"allocations").mkdir()
    for name in module.NEW_CASE_IDS:
        (root/"allocations"/f"{name}.json").write_text("{}")
    calls = []
    monkeypatch.setattr(module,"_verify_case",lambda r,d,c:calls.append(c) or {"case_id":c,"passed":True})
    proof = module.verify_completed(root/"registration.json")
    assert calls == list(module.NEW_CASE_IDS) and proof["actual_simulated_seconds"] == 120
    assert proof["actual_new_simulated_seconds"] == proof["reused_simulated_seconds"] == 120
    assert proof["physical_evidence_simulated_seconds"] == 240
    assert [c["mode"] for c in proof["cases"]] == ["verified_reuse","new_allocation"]
    assert not proof["physical_traffic_benefit_established"]
    (root/"allocations/extra.json").write_text("{}")
    with pytest.raises(ValueError,match="exactly"):
        module.verify_completed(root/"registration.json")
