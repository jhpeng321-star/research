"""One fixture insertion correction plus verified reuse of passing mechanics.

Registration alone launches nothing. Native route stops are declared fixture
dynamics; all CAV longitudinal commands still pass through CommandOwnership.
"""
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from math import isfinite, sqrt
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from .lifecycle_wall import LifecycleProfile, VERSION, policy_factory
from .config import ROOT, CONFIG_PATH, load_config
from .damping import CirclesController
from .gentle_mechanics import (MEMBERS, FOLLOWERS, LANES, ARTIFACTS,
    _identity, _write_json, _verify_case)
from .mechanics import LoggedVehicle
from .metrics import RequestLedger, braking_effort, service_summary, ttc_exposure
from .ownership import CommandOwnership
from .paired_experiments import PolicyInput, digest
from .reporting import CAVReport, RSUReportBuffer
from .scenarios import TripRequest
from .sensing import NeighborMeasurement, SensorFrame, SumoLocalSensor
from .simulation import close_sumo, start_sumo
from .verification import hash_file


SCHEMA = "finite_lifecycle_mechanics/3"
CASE_IDS = ("accelerating_complete", "developing_constraint")
NEW_CASE_IDS = ("developing_constraint",)
REPORTER, STOP_LEADER = "downstream_reporter", "native_stop_leader"
HORIZON_S, DECISION_S = 120., 5.
PRIOR_REGISTRATION = ROOT/"runs/active/lifecycle_mechanics_v1/registration.json"
REUSE_REGISTRATION = ROOT/"runs/active/lifecycle_mechanics_v2/registration.json"
REPAIR_PROTOCOL = ROOT/"docs/LIFECYCLE_FIXTURE_INSERTION_CHECK.md"
ALLOWED_REUSE_SOURCE_CHANGES = frozenset({"lifecycle_mechanics.py","lifecycle_study.py"})


def plan():
    return dict(schema=SCHEMA, policy_version=VERSION, horizon_s=HORIZON_S, decision_s=DECISION_S,
        warmup_included_s=5., maximum_attempts=1, maximum_simulated_seconds=120, automatic_retries=0,
        repair_protocol_path=str(REPAIR_PROTOCOL),prior_failed_registration_path=str(PRIOR_REGISTRATION),
        prior_failed_simulated_seconds=120.,prior_closed_b2_simulated_seconds=240.,
        separately_authorized_repair_simulated_seconds=120.,reused_passing_simulated_seconds=120.,
        reused_registration_path=str(REUSE_REGISTRATION),
        automatic_successors=False, execution_order=list(NEW_CASE_IDS), simultaneous_runs=False,
        cases=[dict(case_id=case, profile=asdict(LifecycleProfile()), expected_early_exit=early,
                    evidence_mode="new_allocation" if early else "verified_b2_reuse")
               for case, early in zip(CASE_IDS, (False, True))],
        fixture=dict(source="lifecycle_mechanics.build_fixture", lanes=list(LANES),
            seed=1, seed_scope="same isolated deterministic mechanics fixture; not a corridor development seed",
            length_m=3500., speed_limit_mps=24., member_positions_m=[1000., 990.],
            follower_positions_m=[910., 900.], initial_member_follower_speed_mps=20., all_vehicle_model="IDM",
            fixed_lane_fixture_mode=512, physics_s=.5,
            constraint_case=dict(native_reporting_cav_id=REPORTER, reporter_position_m=1180.,
                reporter_lane=LANES[0], reporter_speed_mps=24., hdv_leader_id=STOP_LEADER,
                leader_position_m=1310., leader_depart_speed="max",leader_maximum_speed_mps=24.,
                native_route_stop_end_position_m=1665.,
                stop_duration_s=120., reporter_control="explicit native selection through CommandOwnership",
                disturbance="Registered HDV route stop; no runtime speed command, teleport or packet alteration")),
        criteria=dict(expected_terminal_phase="release", nominal_maximum_hold_s=20.,
            measured_target_readiness_samples=3, assembly_deadline_s=50., transition_deadline_s=10.,
            command_min_mps2=-.3, command_max_mps2=1.5, command_change_limit_mps2=.15,
            require_actual_positive_entry_acceleration=True, require_actual_native_release=True,
            complete_case_hold_s=20., changing_case_hold_s="strictly between zero and 20",
            require_all_six_fixture_vehicles_at_first_physics_s=.5,
            changing_case_trigger="first during-HOLD finite traffic commitment failure with feasible exit-now",
            zero_collisions=True, zero_teleports=True, require_full_horizon=True),
        stop_rules=["No new allocation without verified B2 normal-fixture reuse; never rerun that fixture.",
            "Stop immediately if any of six declared fixture vehicles is absent at .5 s.",
            "Stop on command, ownership, sensing or collision/teleport defect; preserve charged failure.",
            "No retries, fixture replacement or automatic campaign."], mechanical_not_traffic_efficacy=True)


def fixture_routes(case_id):
    """Pure XML construction makes every inserted object and stop reviewable."""
    if case_id not in CASE_IDS:
        raise ValueError("unknown lifecycle fixture")
    routes = ET.Element("routes")
    ET.SubElement(routes, "vType", dict(id="idm", carFollowModel="IDM", accel="2.6", decel="4.5",
        emergencyDecel="9", tau="1.1", minGap="2.5", sigma="0", speedFactor="1", speedDev="0",
        length="5", maxSpeed="24", lcStrategic="0", lcCooperative="0", lcSpeedGain="0", lcKeepRight="0"))
    ET.SubElement(routes, "route", dict(id="route", edges="road"))
    vehicles = []
    for i in range(2):
        vehicles.extend([(MEMBERS[i], 1000.-10*i, i, 20.), (FOLLOWERS[i], 910.-10*i, i, 20.)])
    if case_id == CASE_IDS[1]:
        vehicles.extend([(REPORTER,1180.,0,24.),(STOP_LEADER,1310.,0,24.)])
    for vehicle,position,lane,speed in vehicles:
        node = ET.SubElement(routes,"vehicle",dict(id=vehicle,type="idm",route="route",depart="0",
            departLane=str(lane),departPos=str(position),departSpeed="max" if vehicle == STOP_LEADER else str(speed)))
        if vehicle == STOP_LEADER:
            ET.SubElement(node,"stop",dict(lane=LANES[0],endPos="1665",duration="120"))
    ET.indent(routes)
    return routes


def build_fixture(output, sumo_binary, case_id):
    """Build one declared road, with no longitudinal fixture mutations."""
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    (output/"nodes.xml").write_text('<nodes><node id="start" x="0" y="0"/><node id="end" x="3500" y="0"/></nodes>')
    (output/"edges.xml").write_text('<edges><edge id="road" from="start" to="end" numLanes="2" speed="24"/></edges>')
    binary = Path(sumo_binary)
    converter = binary.parent/("netconvert.exe" if binary.suffix == ".exe" else "netconvert")
    result = subprocess.run([str(converter),"--node-files",str(output/"nodes.xml"),
        "--edge-files",str(output/"edges.xml"),"--output-file",str(output/"network.net.xml")],capture_output=True,text=True)
    (output/"netconvert.log").write_text(result.stdout+result.stderr)
    result.check_returncode()
    routes = fixture_routes(case_id)
    ET.ElementTree(routes).write(output/"routes.rou.xml",encoding="utf-8",xml_declaration=True)
    (output/"scenario.sumocfg").write_text('<configuration><input><net-file value="network.net.xml"/>'
        '<route-files value="routes.rou.xml"/></input><processing><time-to-teleport value="-1"/>'
        '<collision.action value="warn"/></processing></configuration>')
    requests = [TripRequest(n.attrib["id"],0.,"route","idm",n.attrib["id"] in (*MEMBERS,REPORTER))
                for n in routes.findall("vehicle")]
    return output/"scenario.sumocfg", requests


def saved_entry_evidence():
    """Reuse only the preserved failed mechanics' acquired startup evidence."""
    stage = PRIOR_REGISTRATION.parent
    job = stage/"jobs"/CASE_IDS[0]
    reg = json.loads(PRIOR_REGISTRATION.read_text(encoding="utf-8"))
    manifest = json.loads((job/"manifest.json").read_text(encoding="utf-8"))
    stop = json.loads((stage/"STOP.json").read_text(encoding="utf-8"))
    if (reg.get("registration_sha256") != digest({k:v for k,v in reg.items() if k != "registration_sha256"})
            or manifest.get("registration_sha256") != reg["registration_sha256"]
            or manifest.get("status") != "complete" or manifest.get("passed") is not False
            or manifest.get("completed_simulated_seconds") != 120.
            or stop.get("case_id") != CASE_IDS[0]):
        raise ValueError("prior failed mechanics lacks preserved evidence")
    for name,expected in manifest["artifact_sha256"].items():
        if hash_file(job/name) != expected:
            raise ValueError("prior failed mechanics evidence changed")
    selected = {}
    with (job/"physical.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["time_s"] in {.5,4.5,5.}:
                selected[row["time_s"]] = row["acquired_member_frames"]
            if row["time_s"] >= 5.:
                break
    frames = []
    for vid in MEMBERS:
        raw = dict(selected[5.][vid])
        measured = (raw["speed_mps"]-selected[4.5][vid]["speed_mps"])/.5
        if abs(measured-raw["acceleration_mps2"]) > 1e-8 or selected[.5][vid]["speed_mps"] != 20.:
            raise ValueError("saved insertion or acquired halfstep acceleration differs")
        for side in ("front","rear"):
            raw[side] = None if raw[side] is None else NeighborMeasurement(**raw[side])
        frames.append(SensorFrame(**raw))
    proof = dict(registration_path=str(PRIOR_REGISTRATION),registration_file_sha256=hash_file(PRIOR_REGISTRATION),
        prior_registration_sha256=reg["registration_sha256"],manifest_sha256=hash_file(job/"manifest.json"),
        physical_sha256=manifest["artifact_sha256"]["physical.jsonl"],stop_sha256=hash_file(stage/"STOP.json"),
        prior_failed_simulated_seconds=120.,entry_time_s=5.,actual_insertion_time_s=.5,
        acquired_acceleration_interval_s=[4.5,5.],offline_mechanics_only=True,
        acquired_entry_frames={f.vehicle_id:asdict(f) for f in frames})
    return frames,proof


def native_stop_insertion_speed():
    """SUMO 1.27.1 IDM stopSpeed arithmetic for this fixed insertion only.

    Same-lane stop admission uses _v(...respectMinGap=False), with two .25 s
    internal updates for the registered .5 s physics. departSpeed=max permits
    this value to replace 24; this calculation is not a simulator execution.
    """
    speed,gap = 24.,1665.-1310.
    for _ in range(2):
        desired = max(0.,speed*1.1+speed*speed/(2*sqrt(2.6*4.5)))
        acceleration = 2.6*(1-(speed/24.)**4-(desired/gap)**2)
        speed = max(0.,speed+.25*acceleration)
        gap -= max(0.,.25*speed)
    return speed


def approximate_native_history():
    """Fixed native-IDM approximation with insertion .5 and halfstep averages.

    This is an offline fixture timing estimate, never a prediction presented to
    the RSU. SUMO's native numerical integration and stop handling still differ.
    """
    def idm(speed,gap=None,front_speed=0.):
        desired = 2.5+max(0.,1.1*speed+speed*(speed-front_speed)/(2*sqrt(2.6*4.5)))
        interaction = 0. if gap is None else (desired/max(.01,gap))**2
        return max(-4.5,2.6*(1-(speed/24.)**4-interaction))
    positions = {MEMBERS[0]:1000.,MEMBERS[1]:990.,REPORTER:1180.,STOP_LEADER:1310.}
    speeds = {MEMBERS[0]:20.,MEMBERS[1]:20.,REPORTER:24.,STOP_LEADER:native_stop_insertion_speed()}
    previous_speeds = dict(speeds)
    state,dt = {},.005
    for n in range(1,9901):
        accelerations = {STOP_LEADER:idm(speeds[STOP_LEADER],1665.-positions[STOP_LEADER]),
            REPORTER:idm(speeds[REPORTER],positions[STOP_LEADER]-positions[REPORTER]-5.,speeds[STOP_LEADER])}
        if n <= 900:
            accelerations.update({MEMBERS[0]:idm(speeds[MEMBERS[0]],positions[REPORTER]-positions[MEMBERS[0]]-5.,speeds[REPORTER]),
                                 MEMBERS[1]:idm(speeds[MEMBERS[1]])})
        for vid,a in accelerations.items():
            positions[vid] += max(0.,speeds[vid]*dt+.5*a*dt*dt)
            speeds[vid] = max(0.,speeds[vid]+a*dt)
        if n % 100 == 0:
            now = .5+n*dt
            state[now] = {vid:dict(position_m=positions[vid],speed_mps=speeds[vid],
                acceleration_mps2=(speeds[vid]-previous_speeds[vid])/.5) for vid in positions}
            previous_speeds = dict(speeds)
    return state


def approximate_entry_frames(history):
    return [SensorFrame(vid,5.,history[5.][vid]["speed_mps"],history[5.][vid]["acceleration_mps2"],24.,lane,
        3500.-history[5.][vid]["position_m"],front_coverage_m=100.,rear_coverage_m=100.,
        lane_id=LANES[lane],position_m=history[5.][vid]["position_m"])
        for vid,lane in ((MEMBERS[0],0),(MEMBERS[1],1),(REPORTER,0))]


def reuse_source_equivalence(previous_identity, current_identity):
    """Only the named fixture/registration edits may differ from B2."""
    before = previous_identity["active_source_sha256"]
    after = current_identity["active_source_sha256"]
    if (set(before) != set(after) or any(before[n] != after[n] for n in before if n not in ALLOWED_REUSE_SOURCE_CHANGES)
            or {k:v for k,v in previous_identity.items() if k != "active_source_sha256"}
                != {k:v for k,v in current_identity.items() if k != "active_source_sha256"}):
        raise ValueError("B2 reuse requires identical policy, actuator, sensing, model and scenario")
    return {name:dict(previous=before[name],current=after[name]) for name in sorted(before)
            if before[name] != after[name]}


def reused_normal_evidence(current_identity):
    """Verify one passing B2 arm without treating its stopped stage as a pass."""
    stage = REUSE_REGISTRATION.parent
    data = json.loads(REUSE_REGISTRATION.read_text(encoding="utf-8"))
    if (data.get("schema") != "finite_lifecycle_mechanics/2"
            or data.get("registration_sha256") != digest({k:v for k,v in data.items() if k != "registration_sha256"})):
        raise ValueError("B2 reuse registration changed")
    differences = reuse_source_equivalence(data["identity"],current_identity)
    for name,expected in data["identity"]["active_source_sha256"].items():
        if hash_file(stage/"source_snapshot"/name) != expected:
            raise ValueError("B2 frozen source snapshot changed")
    if (hash_file(stage/"source_snapshot/adapter.json") != data["identity"]["config_sha256"]
            or hash_file(stage/"source_snapshot/repair_protocol.md") != data["repair_evidence"]["protocol_sha256"]):
        raise ValueError("B2 frozen configuration/protocol snapshot changed")
    if {p.stem for p in (stage/"allocations").glob("*.json")} != set(CASE_IDS):
        raise ValueError("B2 allocations changed")
    passed = _verify_case(stage,data,CASE_IDS[0])
    normal = stage/"jobs"/CASE_IDS[0]
    registered_profile = next(c["profile"] for c in data["cases"] if c["case_id"] == CASE_IDS[0])
    if registered_profile != asdict(LifecycleProfile()):
        raise ValueError("B2 normal policy profile changed")
    actual_routes = ET.canonicalize((normal/"scenario/routes.rou.xml").read_text(encoding="utf-8"),strip_text=True)
    expected_routes = ET.canonicalize(ET.tostring(fixture_routes(CASE_IDS[0]),encoding="unicode"),strip_text=True)
    if actual_routes != expected_routes:
        raise ValueError("normal fixture route semantics changed")
    samples = [json.loads(line) for line in (normal/"formation_samples.jsonl").read_text(encoding="utf-8").splitlines()]
    score = lifecycle_checks(samples,LifecycleProfile())
    if not all(score["checks"].values()):
        raise ValueError("B2 normal lacks measured readiness/maintainable HOLD/exit proof")
    stop = json.loads((stage/"STOP.json").read_text(encoding="utf-8"))
    negative = stage/"jobs"/CASE_IDS[1]
    negative_manifest = json.loads((negative/"manifest.json").read_text(encoding="utf-8"))
    allocation = json.loads((stage/"allocations"/f"{CASE_IDS[1]}.json").read_text(encoding="utf-8"))
    if (stop.get("case_id") != CASE_IDS[1] or negative_manifest.get("passed") is not False
            or negative_manifest.get("status") != "complete" or negative_manifest.get("completed_simulated_seconds") != 120.
            or negative_manifest.get("registration_sha256") != data["registration_sha256"]
            or allocation != dict(case_id=CASE_IDS[1],allocated_simulated_seconds=120.,retries=0,
                                  registration_sha256=data["registration_sha256"])):
        raise ValueError("B2 failed fixture and charged allocation must remain preserved")
    for name,expected in negative_manifest["artifact_sha256"].items():
        if hash_file(negative/name) != expected:
            raise ValueError("B2 negative evidence changed")
    normal_manifest = json.loads((normal/"manifest.json").read_text(encoding="utf-8"))
    return dict(passed,mode="verified_reuse",source_path=str(normal),
        registration_path=str(REUSE_REGISTRATION),registration_file_sha256=hash_file(REUSE_REGISTRATION),
        registration_sha256=data["registration_sha256"],source_sha256=data["identity"]["active_source_sha256"],
        artifact_sha256=normal_manifest["artifact_sha256"],current_source_equivalence_exceptions=differences,
        measured_lifecycle_score=score,normal_route_semantics_identical=True,newly_allocated_simulated_seconds=0.,
        preserved_closed_b2=dict(stop_sha256=hash_file(stage/"STOP.json"),negative_manifest_sha256=hash_file(negative/"manifest.json"),
            negative_artifact_sha256=negative_manifest["artifact_sha256"],actual_simulated_seconds=240.))


def repair_evidence(current_identity):
    _,proof = saved_entry_evidence()
    return dict(protocol_path=str(REPAIR_PROTOCOL),protocol_sha256=hash_file(REPAIR_PROTOCOL),
                preserved_failed_stage=proof,reused_normal=reused_normal_evidence(current_identity),
                separate_repair_budget_simulated_seconds=120.,automatic_retry=False)


def entry_preflight():
    """Saved actual normal startup plus corrected approximate dynamic startup."""
    actual,proof = saved_entry_evidence()
    approximate = approximate_entry_frames(approximate_native_history())
    result = {}
    for case,frames in zip(CASE_IDS,(actual,approximate)):
        reports = tuple(CAVReport(f,DECISION_S,"native") for f in frames)
        policy = policy_factory(dict(arm="wall",decision_time_s=DECISION_S,policy_spec=dict(
            member_ids=list(MEMBERS),required_lanes=list(LANES),profile=asdict(LifecycleProfile()))),{})
        tick = policy(PolicyInput(DECISION_S,reports,tuple((DECISION_S,r) for r in reports),{},
            frozenset(f.vehicle_id for f in frames)))
        commitment = tick.details.get("lifecycle_plan",{}).get("commitment",{})
        passed = (all(f.acceleration_mps2 > 0. for f in frames if f.vehicle_id in MEMBERS)
            and tick.phase == "assembly" and len(tick.wall_commands) == 2
            and commitment.get("feasible") is True and commitment.get("remaining_hold_s") == 20.)
        result[case] = dict(passed=passed,phase=tick.phase,reason=tick.reason,
            initial_speed_mps=frames[0].speed_mps,initial_acceleration_mps2=frames[0].acceleration_mps2,
            initial_frames={f.vehicle_id:asdict(f) for f in frames},wall_commands=dict(tick.wall_commands),
            evidence_source="saved_actual_acquired_startup" if case == CASE_IDS[0] else "approximate_native_startup_including_downstream_following",
            lifecycle_plan=commitment)
    return dict(passed=all(v["passed"] for v in result.values()),cases=result,
        saved_entry_provenance=proof,simulated_seconds=0,insertion_time_s=.5,acceleration_measurement_interval_s=.5,
        interpretation="Offline saved startup and corrected native-IDM approximation; no new SUMO execution or safety claim.")


def constraint_timing_preflight():
    """One fixed approximate-IDM timeline, not SUMO and never an online input.

    The HDV anticipates a stationary stop point; the reporter follows that HDV.
    This approximate continuous IDM calculation only checks the declared fixture
    timing. Native insertion, stop handling and response differ in SUMO; the
    registered physical pass criteria still require the actual causal sequence.
    """
    state = approximate_native_history()
    rows = [f for f in approximate_entry_frames(state) if f.vehicle_id in MEMBERS]
    policy = policy_factory(dict(arm="wall",decision_time_s=DECISION_S,policy_spec=dict(
        member_ids=list(MEMBERS),required_lanes=list(LANES),profile=asdict(LifecycleProfile()))),{})
    timeline = []
    for now in range(int(DECISION_S),50):
        observed = state[now][REPORTER]
        x,v,a = observed["position_m"],observed["speed_mps"],observed["acceleration_mps2"]
        reporter = SensorFrame(REPORTER,now,v,a,24.,0,3500.-x,front_coverage_m=100.,rear_coverage_m=100.,
            lane_id=LANES[0],position_m=x)
        reports = tuple(CAVReport(f,now,"native") for f in (*rows,reporter))
        tick = policy(PolicyInput(now,reports,(),{},frozenset(f.vehicle_id for f in (*rows,reporter))))
        plans = tick.details.get("lifecycle_plan",{})
        timeline.append(dict(time_s=now,phase=tick.phase,reason=tick.reason,reporter_speed_mps=v,
            reporter_bumper_gap_from_member_a_m=x-rows[0].position_m-5.,
            readiness_count=tick.details.get("readiness_count"),
            commitment_feasible=plans.get("commitment",{}).get("feasible"),
            commitment_traffic_margin_m=plans.get("commitment",{}).get("minimum_traffic_margin_m"),
            exit_now_feasible=plans.get("exit_now",{}).get("feasible"),
            exit_now_traffic_margin_m=plans.get("exit_now",{}).get("minimum_traffic_margin_m")))
        if tick.phase in {"refused","abort","release"}:
            break
        rows = [replace(f,timestamp_s=now+1,speed_mps=f.speed_mps+tick.wall_commands[f.vehicle_id],
            acceleration_mps2=tick.wall_commands[f.vehicle_id],
            distance_to_merge_m=f.distance_to_merge_m-f.speed_mps-.5*tick.wall_commands[f.vehicle_id],
            position_m=f.position_m+f.speed_mps+.5*tick.wall_commands[f.vehicle_id]) for f in rows]
    hold = [r for r in timeline if r["phase"] == "hold"]
    failed = [r for r in timeline if r["commitment_feasible"] is False]
    passed = (timeline[0]["phase"] == "assembly" and bool(hold) and len(hold) < 20
        and len(failed) == 1 and failed[0]["time_s"] > hold[0]["time_s"]
        and failed[0]["phase"] in {"taper","release"} and failed[0]["exit_now_feasible"] is True
        and timeline[-1]["phase"] == "release")
    return dict(passed=passed,simulated_seconds=0,timeline=timeline,insertion_time_s=.5,
        acceleration_measurement_interval_s=.5,
        native_stop_leader_initial_speed_estimate_mps=native_stop_insertion_speed(),
        interpretation="Approximate continuous native-IDM stop/following with .5 s insertion and measured halfstep acceleration; exact member-command forecast, no new native SUMO execution or efficacy evidence.")


def register(output):
    preflight = entry_preflight()
    timing = constraint_timing_preflight()
    if not preflight["passed"] or not timing["passed"]:
        raise ValueError("pure lifecycle fixture preflight failed")
    identity, binary = _identity()
    data = dict(plan(), identity=identity, sumo_binary=binary,entry_preflight=preflight,constraint_timing_preflight=timing,
                repair_evidence=repair_evidence(identity),
                registered_at_utc=datetime.now(timezone.utc).isoformat())
    data["registration_sha256"] = digest(data)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output/"source_snapshot"
    snapshot.mkdir()
    for name in identity["active_source_sha256"]:
        shutil.copy2(ROOT/"traffic_shepherd"/name, snapshot/name)
    shutil.copy2(CONFIG_PATH, snapshot/"adapter.json")
    shutil.copy2(REPAIR_PROTOCOL,snapshot/"repair_protocol.md")
    _write_json(output/"registration.json", data)
    return dict(registration=str(output/"registration.json"), case_ids=list(NEW_CASE_IDS),
                reused_case_ids=[CASE_IDS[0]],maximum_simulated_seconds=120, simulations_started=0)


def read_registration(path, *, require_current=True):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("registration_sha256") != digest({k:v for k,v in data.items() if k != "registration_sha256"}):
        raise ValueError("lifecycle mechanics registration changed")
    if any(data.get(k) != v for k,v in plan().items()):
        raise ValueError("lifecycle mechanics differs from one-allocation plus reuse plan")
    if require_current and data["identity"] != _identity(data["sumo_binary"])[0]:
        raise ValueError("source/configuration changed after mechanics registration")
    if data.get("repair_evidence") != repair_evidence(data["identity"]):
        raise ValueError("repair authorization or preserved failure changed")
    for name, expected in data["identity"]["active_source_sha256"].items():
        if hash_file(path.parent/"source_snapshot"/name) != expected:
            raise ValueError("mechanics source snapshot changed")
    if hash_file(path.parent/"source_snapshot/adapter.json") != data["identity"]["config_sha256"]:
        raise ValueError("mechanics adapter snapshot changed")
    if hash_file(path.parent/"source_snapshot/repair_protocol.md") != data["repair_evidence"]["protocol_sha256"]:
        raise ValueError("mechanics repair protocol snapshot changed")
    return path.parent, data


def verify_completed(registration):
    root, data = read_registration(registration)
    if (root/"STOP.json").exists():
        raise ValueError("lifecycle mechanics stage stopped")
    if {p.stem for p in (root/"allocations").glob("*.json")} != set(NEW_CASE_IDS):
        raise ValueError("exactly the new developing-constraint allocation must complete")
    cases = [data["repair_evidence"]["reused_normal"],dict(_verify_case(root,data,NEW_CASE_IDS[0]),mode="new_allocation")]
    return dict(schema="verified_lifecycle_mechanics/3", registration_path=str(root/"registration.json"),
        registration_sha256=data["registration_sha256"], registration_file_sha256=hash_file(root/"registration.json"),
        source_sha256=data["identity"]["active_source_sha256"], cases=cases, passed=True,
        actual_simulated_seconds=120,actual_new_simulated_seconds=120,reused_simulated_seconds=120,
        physical_evidence_simulated_seconds=240,physical_traffic_benefit_established=False)


def command_failures(result, records, previous, frames, profile):
    failures, by_id = [], {r.vehicle_id:r for r in records}
    if len(by_id) != len(records) or not set(by_id) <= set((*MEMBERS,REPORTER)):
        failures.append("duplicate_or_foreign_command_record")
    if not set(result.wall_commands) <= set(MEMBERS):
        failures.append("wall_attempted_nonmember_control")
    for vid, requested in result.wall_commands.items():
        row = by_id.get(vid)
        issued = None if row is None else row.issued_acceleration_mps2
        if (row is None or row.owner != "wall" or issued is None or not isfinite(issued)
                or abs(issued-requested) > 1e-8
                or not profile.acceleration_min_mps2-1e-8 <= issued <= profile.acceleration_max_mps2+1e-8):
            failures.append("actual_issued_bound_or_identity")
        elif (abs(issued-frames[vid].acceleration_mps2) > .15+1e-8
                or (vid in previous and abs(issued-previous[vid]) > .15+1e-8)):
            failures.append("actual_issued_entry_or_command_slew")
    if any(r.owner == "wall" and r.vehicle_id not in result.wall_commands for r in records):
        failures.append("unrequested_wall_owner")
    return failures


def physical_diagnostics(physical):
    groups = {"all":None,"members":set(MEMBERS),"followers":set(FOLLOWERS),
              "native_downstream_fixture":{REPORTER,STOP_LEADER}}
    answer = {}
    for name,ids in groups.items():
        accelerations,previous,changes = [],{},[]
        for sample in physical:
            for row in sample["all_vehicles"]:
                if ids is not None and row["id"] not in ids:
                    continue
                a = row["acceleration_mps2"]
                accelerations.append(a)
                if row["id"] in previous:
                    changes.append(abs(a-previous[row["id"]]))
                previous[row["id"]] = a
        answer[name] = dict(samples=len(accelerations),minimum_measured_acceleration_mps2=min(accelerations,default=None),
            maximum_measured_acceleration_mps2=max(accelerations,default=None),
            maximum_halfsecond_acceleration_change_mps2=max(changes,default=None),
            integrated_squared_acceleration_m2_per_s3=sum(a*a*.5 for a in accelerations),
            exposure_vehicle_seconds={str(t):.5*sum(a < t-1e-8 for a in accelerations) for t in (-.3,-.5,-1.,-2.,-3.)})
    return answer


def developing_constraint_checks(samples, physical):
    """Verify the trigger's acquired reporter corresponds to real slowing."""
    trigger = next((r for r in samples if r.get("lifecycle_plan",{}).get("commitment",{}).get("feasible") is False),None)
    initial = next((r for r in physical if r["time_s"] == DECISION_S),{})
    at_trigger = next((r for r in physical if trigger and r["time_s"] == trigger["time_s"]),{})
    first_frame = initial.get("acquired_member_frames",{}).get(REPORTER)
    trigger_frame = at_trigger.get("acquired_member_frames",{}).get(REPORTER)
    commitment = {} if trigger is None else trigger.get("lifecycle_plan",{}).get("commitment",{})
    return dict(trigger_uses_current_acquired_downstream_reporter=bool(trigger_frame) and bool(trigger)
        and trigger_frame["timestamp_s"] == trigger["time_s"]
        and REPORTER in commitment.get("fresh_downstream_reporter_ids",[]),
        downstream_reporter_physically_slowed_since_entry=bool(first_frame) and bool(trigger_frame)
        and trigger_frame["speed_mps"] < first_frame["speed_mps"]-1e-8
        and trigger_frame["acceleration_mps2"] < 0.,
        declared_native_hdv_stop_started_braking_before_exit=bool(trigger) and any(
            v["id"] == STOP_LEADER and v["acceleration_mps2"] < 0.
            for row in physical if row["time_s"] <= trigger["time_s"] for v in row["all_vehicles"]))


def actual_native_release_verified(first_handbacks):
    """At least one native fallback must actually release its prior override."""
    return (any(r["baseline_native"] and r["actual_override_release"] for r in first_handbacks)
        and all(not r["baseline_native"] or r["actual_override_release"] for r in first_handbacks))


def positive_entry_commands_verified(wall_records):
    entries = [r for r in wall_records if r["timestamp_s"] == DECISION_S and r["owner"] == "wall"]
    return (len(entries) == len(MEMBERS) and {r["vehicle_id"] for r in entries} == set(MEMBERS)
        and all(r["issued_acceleration_mps2"] is not None and r["issued_acceleration_mps2"] > 0. for r in entries))


def actual_damping_handover_verified(first_handbacks, handbacks):
    return (len(first_handbacks) == len(MEMBERS) and {r["vehicle_id"] for r in first_handbacks} == set(MEMBERS)
        and all(r.get("prior_owner") == "wall" for r in first_handbacks)
        and bool(handbacks) and all(r["matched"] for r in handbacks))


def hold_physical_checks(hold_rows, hold_seconds):
    compact = bool(hold_rows) and all(len(m := [v for v in row["all_vehicles"] if v["id"] in MEMBERS]) == 2
        and {v["lane"] for v in m} == set(LANES)
        and max(v["distance_m"] for v in m)-min(v["distance_m"] for v in m) <= 50.
        and max(v["speed_mps"] for v in m)-min(v["speed_mps"] for v in m) <= 1. for row in hold_rows)
    return dict(hold_compact_at_halfsecond_clock=compact,
        actual_hold_at_halfsecond_clock=bool(hold_rows) and hold_seconds > 0. and len(hold_rows)*.5 == hold_seconds)


def fixture_insertion_verified(physical):
    expected = set((*MEMBERS,*FOLLOWERS,REPORTER,STOP_LEADER))
    return bool(physical) and physical[0]["time_s"] == .5 and {r["id"] for r in physical[0]["all_vehicles"]} == expected


def lifecycle_checks(samples, profile, *, expected_early_exit=False):
    """Measured readiness, real phase duration, finite exit and causal trigger."""
    streak, first_ready, first_after_assembly = 0, None, None
    maintainable_entry = False
    hold_rows, transition_rows = [], []
    integrity = bool(samples) and samples[0]["phase"] == "assembly"
    previous_phase = None
    allowed = {"assembly": {"assembly","hold","taper","release"},
               "hold": {"hold","taper","release"}, "taper": {"taper","release"}, "release": set()}
    for index, row in enumerate(samples):
        integrity &= abs(row["time_s"]-DECISION_S-index) <= 1e-8
        if previous_phase is not None:
            integrity &= row["phase"] in allowed.get(previous_phase,set())
        previous_phase = row["phase"]
        members = row["members"]
        compact = (len(members) == 2 and {r["lane"] for r in members} == set(LANES)
            and max(r["distance_m"] for r in members)-min(r["distance_m"] for r in members) <= profile.formed_span_m
            and max(r["speed_mps"] for r in members)-min(r["speed_mps"] for r in members) <= profile.formed_speed_spread_mps)
        settled = compact and all(abs(r["speed_mps"]-row["target_speed_mps"]) <= profile.target_speed_tolerance_mps for r in members)
        if first_after_assembly is None:
            streak = streak+1 if settled else 0
            if streak >= profile.readiness_samples and first_ready is None:
                first_ready = row["time_s"]
            if row["phase"] != "assembly":
                first_after_assembly = row["time_s"] if row["phase"] == "hold" else None
                commitment = row.get("lifecycle_plan",{}).get("commitment",{})
                maintainable_entry = (streak >= profile.readiness_samples and row["assembly_completed_at_s"] == row["time_s"]
                    and row["phase"] == "hold" and commitment.get("feasible") is True
                    and commitment.get("maintainable_hold_ready_now") is True
                    and commitment.get("predicted_hold_start_s") == row["time_s"]
                    and commitment.get("remaining_hold_s") == profile.hold_s
                    and commitment.get("remaining_assembly_s") == 0)
                integrity &= maintainable_entry
        if row["phase"] == "hold":
            hold_rows.append(row)
            integrity &= compact
        elif row["phase"] == "taper":
            transition_rows.append(row)
        elif row["phase"] in {"abort", "refused"}:
            integrity = False
    hold_times = [r["time_s"] for r in hold_rows]
    terminal = samples[-1] if samples else {}
    terminal_time = terminal.get("time_s")
    exit_time = terminal.get("exit_started_at_s")
    expected_hold = ([] if first_after_assembly is None or exit_time is None else
                     [first_after_assembly+i for i in range(int(exit_time-first_after_assembly))])
    release_ok = (terminal.get("phase") == "release" and first_after_assembly is not None
        and exit_time is not None and exit_time <= terminal_time <= exit_time+profile.taper_s
        and not terminal.get("wall_command_ids")
        and set(terminal.get("owners", {})) == set(MEMBERS)
        and all(owner in {"native", "damping"} for owner in terminal["owners"].values()))
    failed_commitments = [r for r in samples if r.get("lifecycle_plan",{}).get("commitment",{}).get("feasible") is False]
    trigger = failed_commitments[0] if failed_commitments else None
    early_ok = (trigger is not None and first_after_assembly is not None and exit_time == trigger["time_s"]
        and first_after_assembly < exit_time < first_after_assembly+profile.hold_s
        and trigger["phase"] in {"taper","release"}
        and trigger.get("lifecycle_plan",{}).get("commitment",{}).get("failure_kind") == "traffic"
        and trigger.get("lifecycle_plan",{}).get("exit_now",{}).get("feasible") is True
        and trigger.get("early_exit_reason") == "reported_traffic_intersects_remaining_maneuver")
    hold_duration_ok = (0 < len(hold_rows) < profile.hold_s if expected_early_exit else len(hold_rows) == profile.hold_s)
    prior = samples[-2] if len(samples) >= 2 else {}
    coast_ok = (set(prior.get("wall_commands",{})) == set(MEMBERS)
        and all(abs(v) <= 1e-8 for v in prior.get("wall_commands",{}).values())
        and len(terminal.get("members",[])) == 2
        and all(abs(r["acceleration_mps2"]) <= profile.acceleration_change_limit_mps3+1e-8
                for r in terminal.get("members",[])))
    checks = dict(measured_lifecycle_integrity=bool(integrity),
        maintainable_hold_entry_verified=maintainable_entry,
        three_target_settled_readiness_samples=first_after_assembly is not None and streak >= profile.readiness_samples,
        assembly_before_original_deadline=first_after_assembly is not None and first_after_assembly-DECISION_S <= profile.assembly_deadline_s,
        registered_actual_hold_duration=hold_times == expected_hold and hold_duration_ok,
        entry_remaining_commitment_feasible=bool(samples) and samples[0].get("lifecycle_plan",{}).get("commitment",{}).get("feasible") is True,
        timely_during_hold_constraint_exit=early_ok if expected_early_exit else not failed_commitments,
        terminal_release_within_transition_budget=bool(release_ok), actual_measured_coast_terminal=bool(coast_ok))
    return dict(checks=checks, first_measured_ready_s=first_ready,first_geometric_ready_at_s=first_ready,
                actual_hold_started_at_s=first_after_assembly,hold_seconds=len(hold_rows),
                taper_command_seconds=len(transition_rows), terminal_time_s=terminal_time,
                exit_started_at_s=exit_time, first_commitment_failure_s=None if trigger is None else trigger["time_s"])


def run_one(registration, case_id):
    root, data = read_registration(registration)
    if case_id not in NEW_CASE_IDS or (root/"STOP.json").exists():
        raise ValueError("unbudgeted case, attempted normal rerun, or stopped lifecycle mechanics")
    (root/"allocations").mkdir(exist_ok=True)
    with (root/"allocations"/f"{case_id}.json").open("x",encoding="utf-8") as stream:
        json.dump(dict(case_id=case_id,allocated_simulated_seconds=HORIZON_S,retries=0,
                       registration_sha256=data["registration_sha256"]),stream,indent=2)
    try:
        result = _run_allocated(root,data,case_id)
        if not result["passed"]:
            _write_json(root/"STOP.json",dict(case_id=case_id,reason="mechanics_check_failed",checks=result["checks"]))
        return result
    except Exception as exc:
        _write_json(root/"STOP.json",dict(case_id=case_id,reason=type(exc).__name__,detail=str(exc)))
        raise


def _run_allocated(root,data,case_id):
    import traci
    output = root/"jobs"/case_id
    output.mkdir(parents=True,exist_ok=False)
    manifest = dict(schema=SCHEMA,case_id=case_id,registration_sha256=data["registration_sha256"],
        status="allocated",completed_simulated_seconds=0.,passed=False,mechanics_only=True)
    _write_json(output/"manifest.json",manifest)
    physical,telemetry,policies,samples,writes,handbacks = [],[],[],[],[],[]
    failures,previous_records,previous_wall,history,configured = [],{},{},[],set()
    terminal_time,proposal_record,last_phase = None,None,"damping"
    collisions = teleports = 0
    totals = dict(braking_vehicle_seconds=0.,acceleration_energy=0.,ttc_below_3_vehicle_seconds=0.)
    config = load_config()
    case = next(c for c in data["cases"] if c["case_id"] == case_id)
    profile = LifecycleProfile(**case["profile"])
    cav_ids = set(MEMBERS) | ({REPORTER} if case["expected_early_exit"] else set())
    fixture_natives = cav_ids-set(MEMBERS)
    try:
        scene,requests = build_fixture(output/"scenario",data["sumo_binary"],case_id)
        ledger = RequestLedger(requests)
        baseline = CirclesController(config)
        api = SimpleNamespace(vehicle=LoggedVehicle(traci.vehicle,writes),lane=traci.lane)
        owner = CommandOwnership(api,cav_ids,baseline)
        sensor = SumoLocalSensor(api,config,{lane:0. for lane in LANES})
        buffer = RSUReportBuffer(max_age_s=1.,history_s=20.)
        policy = policy_factory(dict(arm="wall",decision_time_s=DECISION_S,policy_spec=dict(
            member_ids=list(MEMBERS),required_lanes=list(LANES),profile=case["profile"])),data)
        start_sumo(data["sumo_binary"],scene,seed=1,config=config)
        for _ in range(240):
            traci.simulationStep()
            now = float(traci.simulation.getTime())
            manifest["completed_simulated_seconds"] = now
            live = set(traci.vehicle.getIDList())
            for vid in live-configured:
                traci.vehicle.setLaneChangeMode(vid,512)
                configured.add(vid)
            ledger.advance(now,list(traci.simulation.getDepartedIDList()),list(traci.simulation.getArrivedIDList()))
            collisions += int(traci.simulation.getCollidingVehiclesNumber())
            teleports += int(traci.simulation.getStartingTeleportNumber())
            vehicles = [dict(id=v,lane=traci.vehicle.getLaneID(v),position_m=float(traci.vehicle.getLanePosition(v)),
                distance_m=float(traci.lane.getLength(traci.vehicle.getLaneID(v)))-float(traci.vehicle.getLanePosition(v)),
                speed_mps=float(traci.vehicle.getSpeed(v)),acceleration_mps2=float(traci.vehicle.getAcceleration(v)),
                speed_mode=traci.vehicle.getSpeedMode(v),type_id=traci.vehicle.getTypeID(v)) for v in sorted(live)]
            for k,value in braking_effort([r["acceleration_mps2"] for r in vehicles],.5).items():
                totals[k] += value
            totals["ttc_below_3_vehicle_seconds"] += ttc_exposure(traci,live,.5)[0]
            frames = {v:sensor.capture(v,now_s=now) for v in sorted(live & cav_ids)}
            physical.append(dict(time_s=now,outcome_interval_phase=last_phase,all_vehicles=vehicles,
                acquired_member_frames={v:asdict(f) for v,f in frames.items()}))
            if now == .5 and not fixture_insertion_verified(physical):
                failures.append("declared_fixture_vehicle_missing_at_first_halfsecond")
                break
            for vid,f in frames.items():
                if vid in previous_records:
                    response = owner.record_response(vid,timestamp_s=now,speed_mps=f.speed_mps,acceleration_mps2=f.acceleration_mps2)
                    telemetry.append(dict(kind="measured_response",**response.as_dict()))
            if collisions or teleports:
                failures.append("collision_or_teleport")
                break
            if not now.is_integer() or now >= HORIZON_S:
                continue
            for vid,f in frames.items():
                prior = previous_records.get(vid)
                packet = CAVReport(f,now,prior.owner if prior else "native",prior.network_output_mps2 if prior else None,
                    prior.issued_acceleration_mps2 if prior else None,prior.fallback_reason if prior else None)
                buffer.receive(packet,received_at_s=now)
                history.append((now,packet))
                telemetry.append(dict(kind="rsu_received_report",received_at_s=now,**packet.payload()))
            history = [(t,r) for t,r in history if t >= now-20.]
            expected = {v:baseline.decide(f,now_s=now) for v,f in frames.items() if v in MEMBERS}
            result = policy(PolicyInput(now,tuple(buffer.reports(now_s=now)),tuple(history),
                {v:d.command_acceleration_mps2 for v,d in expected.items() if d.fallback_reason is None},
                frozenset(v for v,d in expected.items() if d.fallback_reason is not None) | frozenset(fixture_natives & live)))
            policies.append(dict(time_s=now,**result.payload()))
            if policy.proposal is not None and proposal_record is None:
                proposal_record = asdict(policy.proposal)
            count = len(writes)
            prior_owners = {vid:r.owner for vid,r in previous_records.items()}
            records = owner.step(frames,now_s=now,live_ids=live,wall_commands=result.wall_commands,
                native_ids=fixture_natives & live)
            failures.extend(command_failures(result,records,previous_wall,frames,profile))
            if len(writes)-count != sum(r.issued_acceleration_mps2 is not None or r.released_to_native for r in records):
                failures.append("competing_or_missing_command_writer")
            for record in records:
                telemetry.append(dict(kind="command",**record.as_dict()))
                if record.vehicle_id in expected:
                    telemetry.append(dict(kind="identical_damping_reference",vehicle_id=record.vehicle_id,
                        timestamp_s=now,**expected[record.vehicle_id].as_dict()))
            previous_records = {r.vehicle_id:r for r in records}
            previous_wall = {r.vehicle_id:r.issued_acceleration_mps2 for r in records if r.owner == "wall"}
            if now >= DECISION_S and terminal_time is None:
                samples.append(dict(time_s=now,phase=result.phase,reason=result.reason,
                    members=[{k:r[k] for k in ("id","lane","distance_m","speed_mps","acceleration_mps2")} for r in vehicles if r["id"] in MEMBERS],
                    owners={r.vehicle_id:r.owner for r in records if r.vehicle_id in MEMBERS},wall_command_ids=sorted(result.wall_commands),
                    wall_commands=dict(result.wall_commands),lifecycle_plan=result.details.get("lifecycle_plan",{}),
                    exit_started_at_s=result.details.get("exit_started_at_s"),early_exit_reason=result.details.get("early_exit_reason"),
                    target_speed_mps=result.details.get("target_speed_mps"),
                    assembly_completed_at_s=result.details.get("assembly_completed_at_s")))
            if result.phase in {"release","abort","refused"}:
                first = terminal_time is None
                terminal_time = now if first else terminal_time
                for r in records:
                    if r.vehicle_id not in MEMBERS:
                        continue
                    d = expected[r.vehicle_id]
                    matched = (r.owner == ("native" if d.fallback_reason else "damping")
                        and r.issued_acceleration_mps2 == d.command_acceleration_mps2
                        and r.issued_target_speed_mps == d.target_speed_mps and r.fallback_reason == d.fallback_reason)
                    handbacks.append(dict(time_s=now,vehicle_id=r.vehicle_id,matched=matched,terminal_tick=first,
                        actual_override_release=r.released_to_native,baseline_native=d.fallback_reason is not None,
                        prior_owner=prior_owners.get(r.vehicle_id)))
                    if not matched:
                        failures.append("handover_not_identical_damping")
            last_phase = result.phase
            if failures:
                break
        owner.release_all(live_ids=traci.vehicle.getIDList())
        delivery = lifecycle_checks(samples,profile,expected_early_exit=case["expected_early_exit"])
        wall = [r for r in telemetry if r["kind"] == "command" and r["owner"] == "wall"]
        first_handbacks = [r for r in handbacks if r["terminal_tick"]]
        hold_rows = [r for r in physical if r["outcome_interval_phase"] == "hold"]
        checks = dict(delivery["checks"], command_and_runtime_contract=not failures,
            all_six_fixture_vehicles_inserted_at_first_halfsecond=fixture_insertion_verified(physical),
            full_registered_horizon=manifest["completed_simulated_seconds"] == HORIZON_S,
            zero_collisions_teleports=collisions == teleports == 0,
            only_commandownership_cav_writes=all(w[1] in MEMBERS for w in writes),
            unchanged_circles_instance=owner.controller is baseline,
            all_idm_and_safety31=all(v["type_id"] == "idm" and v["speed_mode"] == 31 for r in physical for v in r["all_vehicles"]),
            acquired_sensing_contract=all(f["timestamp_s"] == row["time_s"]
                and 0. <= f["front_coverage_m"] <= 100. and 0. <= f["rear_coverage_m"] <= 100.
                for row in physical for f in row["acquired_member_frames"].values()),
            downstream_reporter_has_only_explicit_native_selection=all(r["owner"] == "native"
                and r["issued_acceleration_mps2"] is None and r["fallback_reason"] == "selected_native"
                for r in telemetry if r["kind"] == "command" and r["vehicle_id"] == REPORTER),
            actual_positive_acceleration_at_entry=bool(samples) and {r["id"] for r in samples[0]["members"]} == set(MEMBERS)
                and all(r["acceleration_mps2"] > 0. for r in samples[0]["members"]),
            positive_entry_wall_commands=positive_entry_commands_verified(wall),
            nonvacuous_wall_commands={r["vehicle_id"] for r in wall} == set(MEMBERS),
            actual_identical_handover=actual_damping_handover_verified(first_handbacks,handbacks),
            actual_native_override_release=actual_native_release_verified(first_handbacks),
            warmup_has_no_wall_commands=all(not r["wall_commands"] and r["phase"] == "damping" for r in policies if r["time_s"] < DECISION_S))
        checks.update(hold_physical_checks(hold_rows,delivery["hold_seconds"]))
        if case["expected_early_exit"]:
            checks.update(developing_constraint_checks(samples,physical))
        summary = dict(manifest,status="complete",passed=all(checks.values()),checks=checks,profile=asdict(profile),
            observed_lifecycle=delivery,failures=failures,collisions=collisions,teleports=teleports,
            physical_totals=totals,physical_diagnostics=physical_diagnostics(physical),
            service=service_summary(ledger,manifest["completed_simulated_seconds"]),
            handover_evidence=handbacks,wall_command_count=len(wall),
            issued_wall_min_mps2=min((r["issued_acceleration_mps2"] for r in wall),default=None),
            issued_wall_max_mps2=max((r["issued_acceleration_mps2"] for r in wall),default=None),
            interpretation="Two synthetic-road execution checks, not natural-traffic efficacy or a universal measured braking guarantee.")
        _write_json(output/"summary.json",summary)
        manifest.update(status="complete",passed=summary["passed"])
        return summary
    except Exception as exc:
        manifest.update(status="failed",error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        close_sumo()
        _write_json(output/"longitudinal_writes.json",writes)
        _write_json(output/"disturbances.json",[] if not case["expected_early_exit"] else [dict(
            kind="registered_native_idm_route_stop",vehicle_id=STOP_LEADER,lane=LANES[0],end_position_m=1665.,
            duration_s=120.,runtime_commands=False,reporter_control="selected_native",reporter_id=REPORTER)])
        _write_json(output/"proposal.json",proposal_record)
        for name,rows in (("physical.jsonl",physical),("telemetry.jsonl",telemetry),
                          ("policy.jsonl",policies),("formation_samples.jsonl",samples)):
            with (output/name).open("w",encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row,allow_nan=False)+"\n")
        manifest["artifact_sha256"] = {name:hash_file(output/name) for name in (*ARTIFACTS,"summary.json",
            "scenario/network.net.xml","scenario/routes.rou.xml","scenario/scenario.sumocfg") if (output/name).exists()}
        _write_json(output/"manifest.json",manifest)
