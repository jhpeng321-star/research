"""Two immutable SUMO mechanics allocations for the gentle report-only wall.

An isolated road and scripted packet drop exercise execution, not traffic
benefit. All longitudinal writes pass through the unchanged CommandOwnership.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from .config import ROOT, CONFIG_PATH, load_config
from .damping import CirclesController
from .escalation_evidence import formation_delivery
from .gentle_wall import GentleProfile, VERSION, policy_factory
from .mechanics import LoggedVehicle
from .metrics import RequestLedger, braking_effort, service_summary, ttc_exposure
from .ownership import CommandOwnership
from .paired_experiments import PolicyInput, digest
from .reporting import CAVReport, RSUReportBuffer
from .scenarios import TripRequest
from .sensing import SumoLocalSensor
from .simulation import close_sumo, start_sumo
from .verification import hash_file


SCHEMA = "gentle_wall_mechanics/1"
MEMBERS = ("member_a", "member_b")
FOLLOWERS = ("follower_a", "follower_b")
LANES = ("road_0", "road_1")
HORIZON_S = 120.
DECISION_S = 5.
CASE_IDS = ("gentle_complete", "gentle_report_loss")
ARTIFACTS = ("physical.jsonl", "telemetry.jsonl", "policy.jsonl", "formation_samples.jsonl",
             "longitudinal_writes.json", "disturbances.json", "proposal.json")


def plan():
    """One reasoned fixture geometry; never select a fixture from new outcomes."""
    return dict(schema=SCHEMA, policy_version=VERSION, profile=asdict(GentleProfile()),
        cases=[dict(case_id=CASE_IDS[0], expected="delivered", packet_drop_from_s=None),
               dict(case_id=CASE_IDS[1], expected="aborted", packet_drop_from_s=15.)],
        decision_s=DECISION_S, horizon_s=HORIZON_S, warmup_included_s=DECISION_S,
        maximum_attempts=2, maximum_simulated_seconds=240, automatic_retries=0,
        execution_order=list(CASE_IDS), simultaneous_runs=False, automatic_successors=False,
        fixture=dict(lanes=list(LANES), length_m=3500., road_speed_mps=24.,
            member_ids=list(MEMBERS), follower_ids=list(FOLLOWERS), member_positions_m=[1000., 990.],
            follower_positions_m=[910., 900.], initial_speed_mps=24., vehicle_length_m=5.,
            all_vehicle_model="IDM", fixed_lane_fixture_mode=512, physics_s=.5,
            disturbance="Drop member_b RSU packets from 15 s in second case only; no physics alteration"),
        criteria=dict(wall_acceleration_min_mps2=-.3, wall_acceleration_max_mps2=1.5,
            maximum_command_change_mps2=.15, maximum_entry_to_measured_change_mps2=.15,
            minimum_wall_command_count=2, formed_span_m=50., speed_spread_mps=1.,
            target_tolerance_mps=1., readiness_samples=3, maximum_assembly_s=50., hold_s=20.,
            require_actual_identical_damping_handover=True, require_complete_horizon=True,
            require_only_cav_writes=True, zero_collisions=True, zero_teleports=True),
        stop_rules=["Do not allocate the second fixture unless the first passes every implementation check.",
            "Any failed allocation remains charged; never retry or overwrite it.",
            "Stop a trajectory immediately on a command, ownership, sensing, collision or teleport defect."],
        mechanical_not_traffic_efficacy=True, actual_braking_not_universally_limited=True)


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def _identity(sumo_binary=None):
    config = load_config()
    binary = Path(sumo_binary or shutil.which("sumo") or
                  "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo.exe").resolve()
    converter = binary.parent / ("netconvert.exe" if binary.suffix == ".exe" else "netconvert")
    if hash_file(config.model_path) != config.model_sha256:
        raise ValueError("frozen CIRCLES model identity differs")
    return dict(config_sha256=hash_file(CONFIG_PATH), model_sha256=config.model_sha256,
        network_sha256=hash_file(config.network_path), scenario_template_sha256=hash_file(config.scenario_template_path),
        sumo_binary_sha256=hash_file(binary), netconvert_sha256=hash_file(converter),
        active_source_sha256={p.name: hash_file(p) for p in sorted((ROOT/"traffic_shepherd").glob("*.py"))}), str(binary)


def register(output):
    """Freeze both allocations and source copies; launch no simulator."""
    identity, binary = _identity()
    data = dict(plan(), identity=identity, sumo_binary=binary,
        registered_at_utc=datetime.now(timezone.utc).isoformat())
    data["registration_sha256"] = digest(data)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output/"source_snapshot"
    snapshot.mkdir()
    for name in identity["active_source_sha256"]:
        shutil.copy2(ROOT/"traffic_shepherd"/name, snapshot/name)
    shutil.copy2(CONFIG_PATH, snapshot/"adapter.json")
    _write_json(output/"registration.json", data)
    return dict(registration=str(output/"registration.json"), case_ids=list(CASE_IDS),
                maximum_simulated_seconds=240, simulations_started=0)


def read_registration(path, *, require_current=True):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("registration_sha256") != digest({k: v for k, v in data.items() if k != "registration_sha256"}):
        raise ValueError("mechanics registration changed")
    if any(data.get(k) != v for k, v in plan().items()):
        raise ValueError("mechanics registration differs from exact bounded plan")
    if require_current and data["identity"] != _identity(data["sumo_binary"])[0]:
        raise ValueError("source/configuration changed after mechanics registration")
    for name, expected in data["identity"]["active_source_sha256"].items():
        if hash_file(path.parent/"source_snapshot"/name) != expected:
            raise ValueError("registered mechanics source snapshot changed")
    if hash_file(path.parent/"source_snapshot/adapter.json") != data["identity"]["config_sha256"]:
        raise ValueError("registered mechanics adapter snapshot changed")
    return path.parent, data


def _verify_case(root, data, case_id):
    output = root/"jobs"/case_id
    manifest_path, summary_path = output/"manifest.json", output/"summary.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    allocation = json.loads((root/"allocations"/f"{case_id}.json").read_text(encoding="utf-8"))
    if allocation != dict(case_id=case_id, allocated_simulated_seconds=HORIZON_S, retries=0,
                          registration_sha256=data["registration_sha256"]):
        raise ValueError("mechanics allocation changed")
    if (manifest.get("status") != "complete" or manifest.get("passed") is not True
            or summary.get("passed") is not True or not summary.get("checks")
            or not all(value is True for value in summary["checks"].values())
            or manifest.get("case_id") != case_id or summary.get("case_id") != case_id
            or manifest.get("registration_sha256") != data["registration_sha256"]
            or summary.get("registration_sha256") != data["registration_sha256"]
            or manifest.get("completed_simulated_seconds") != HORIZON_S
            or summary.get("completed_simulated_seconds") != HORIZON_S):
        raise ValueError("mechanics case lacks complete passing evidence")
    expected_files = (*ARTIFACTS, "summary.json", "scenario/network.net.xml", "scenario/routes.rou.xml", "scenario/scenario.sumocfg")
    if set(manifest.get("artifact_sha256", {})) != set(expected_files):
        raise ValueError("mechanics evidence manifest is incomplete")
    for name, expected in manifest["artifact_sha256"].items():
        if hash_file(output/name) != expected:
            raise ValueError("completed mechanics evidence changed")
    return dict(case_id=case_id, passed=True, summary_sha256=hash_file(summary_path),
                manifest_sha256=hash_file(manifest_path), actual_simulated_seconds=HORIZON_S)


def verify_completed(registration):
    """Require both genuine, current-source, hash-bound physical allocations."""
    root, data = read_registration(registration)
    if (root/"STOP.json").exists():
        raise ValueError("mechanics stage stopped; full-horizon comparison is blocked")
    allocated = {p.stem for p in (root/"allocations").glob("*.json")}
    if allocated != set(CASE_IDS):
        raise ValueError("both exact mechanics allocations must complete")
    cases = [_verify_case(root, data, case) for case in CASE_IDS]
    return dict(schema="verified_gentle_mechanics/1", registration_path=str(root/"registration.json"),
        registration_sha256=data["registration_sha256"], registration_file_sha256=hash_file(root/"registration.json"),
        source_sha256=data["identity"]["active_source_sha256"], cases=cases,
        actual_simulated_seconds=240, passed=True, physical_traffic_benefit_established=False)


def build_fixture(output, sumo_binary):
    """Deterministic initial insertion; no longitudinal fixture commands."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/"nodes.xml").write_text('<nodes><node id="start" x="0" y="0"/><node id="end" x="3500" y="0"/></nodes>')
    (output/"edges.xml").write_text('<edges><edge id="road" from="start" to="end" numLanes="2" speed="24"/></edges>')
    binary = Path(sumo_binary)
    converter = binary.parent/("netconvert.exe" if binary.suffix == ".exe" else "netconvert")
    result = subprocess.run([str(converter), "--node-files", str(output/"nodes.xml"),
        "--edge-files", str(output/"edges.xml"), "--output-file", str(output/"network.net.xml")],
        capture_output=True, text=True)
    (output/"netconvert.log").write_text(result.stdout+result.stderr)
    result.check_returncode()
    routes = ET.Element("routes")
    ET.SubElement(routes, "vType", dict(id="idm", carFollowModel="IDM", accel="2.6", decel="4.5",
        emergencyDecel="9", tau="1.1", minGap="2.5", sigma="0", speedFactor="1", speedDev="0",
        length="5", maxSpeed="24", lcStrategic="0", lcCooperative="0", lcSpeedGain="0", lcKeepRight="0"))
    ET.SubElement(routes, "route", dict(id="route", edges="road"))
    for i in range(2):
        for vehicle, position in ((MEMBERS[i], 1000.-10*i), (FOLLOWERS[i], 910.-10*i)):
            ET.SubElement(routes, "vehicle", dict(id=vehicle, type="idm", route="route", depart="0",
                departLane=str(i), departPos=str(position), departSpeed="24"))
    ET.indent(routes)
    ET.ElementTree(routes).write(output/"routes.rou.xml", encoding="utf-8", xml_declaration=True)
    (output/"scenario.sumocfg").write_text('<configuration><input><net-file value="network.net.xml"/>'
        '<route-files value="routes.rou.xml"/></input><processing><time-to-teleport value="-1"/>'
        '<collision.action value="warn"/></processing></configuration>')
    requests = [TripRequest(v, 0., "route", "idm", v in MEMBERS) for v in (*MEMBERS, *FOLLOWERS)]
    return output/"scenario.sumocfg", requests


def command_failures(result, records, previous_wall, frames):
    """Independent check of actual issued commands, including adapter clipping."""
    failures = []
    by_id = {r.vehicle_id: r for r in records}
    wall = dict(result.wall_commands)
    if len(records) != len(by_id) or not set(by_id) <= set(MEMBERS):
        failures.append("duplicate_or_foreign_command_record")
    for vid, requested in wall.items():
        record = by_id.get(vid)
        issued = None if record is None else record.issued_acceleration_mps2
        if (record is None or record.owner != "wall" or issued is None or not isfinite(issued)
                or abs(issued-requested) > 1e-8 or not -.3-1e-8 <= issued <= 1.5+1e-8):
            failures.append("actual_issued_wall_bound_or_identity")
            continue
        if (abs(issued-frames[vid].acceleration_mps2) > .15+1e-8
                or (vid in previous_wall and abs(issued-previous_wall[vid]) > .15+1e-8)):
            failures.append("actual_issued_entry_or_command_slew")
    if any(r.owner == "wall" and r.vehicle_id not in wall for r in records):
        failures.append("unrequested_wall_owner")
    return failures


def _physical_diagnostics(physics):
    groups = {"all": set((*MEMBERS, *FOLLOWERS)), "members": set(MEMBERS), "followers": set(FOLLOWERS)}
    answer = {}
    for name, ids in groups.items():
        previous, changes, rows = {}, [], []
        for sample in physics:
            for row in sample["all_vehicles"]:
                if row["id"] not in ids:
                    continue
                a = row["acceleration_mps2"]
                rows.append(a)
                if row["id"] in previous:
                    changes.append(abs(a-previous[row["id"]]))
                previous[row["id"]] = a
        answer[name] = dict(samples=len(rows), minimum_measured_acceleration_mps2=min(rows, default=None),
            peak_measured_deceleration_mps2=max((max(0., -a) for a in rows), default=None),
            maximum_halfsecond_acceleration_change_mps2=max(changes, default=None),
            integrated_squared_acceleration_m2_per_s3=sum(a*a*.5 for a in rows),
            exposure_vehicle_seconds={str(threshold): .5*sum(a < threshold-1e-8 for a in rows)
                for threshold in (-.3, -.5, -1., -2.)},
            severe_braking_vehicle_seconds=.5*sum(a < -3. for a in rows))
    return answer


def run_one(registration, case_id):
    """Allocate exactly one case before launch; failures stop all new allocation."""
    root, data = read_registration(registration)
    if case_id not in CASE_IDS or (root/"STOP.json").exists():
        raise ValueError("unknown case or stopped mechanics stage")
    for previous in CASE_IDS[:CASE_IDS.index(case_id)]:
        _verify_case(root, data, previous)
    allocations = root/"allocations"
    allocations.mkdir(exist_ok=True)
    with (allocations/f"{case_id}.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(case_id=case_id, allocated_simulated_seconds=HORIZON_S, retries=0,
                       registration_sha256=data["registration_sha256"]), stream, indent=2)
    try:
        summary = _run_allocated(root, data, case_id)
        if not summary["passed"]:
            _write_json(root/"STOP.json", dict(case_id=case_id, reason="mechanics_check_failed", checks=summary["checks"]))
        return summary
    except Exception as exc:
        _write_json(root/"STOP.json", dict(case_id=case_id, reason=type(exc).__name__, detail=str(exc)))
        raise


def _run_allocated(root, data, case_id):
    import traci
    output = root/"jobs"/case_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema=SCHEMA, case_id=case_id, registration_sha256=data["registration_sha256"],
        status="allocated", completed_simulated_seconds=0., passed=False, mechanics_only=True)
    _write_json(output/"manifest.json", manifest)
    physical, telemetry, policies, samples, writes, disturbances = [], [], [], [], [], []
    failures, previous_records, previous_wall, handbacks = [], {}, {}, []
    collisions = teleports = 0
    totals = dict(braking_vehicle_seconds=0., acceleration_energy=0., ttc_below_3_vehicle_seconds=0.)
    minimum_ttc = None
    owner = baseline = policy = None
    proposal_record = None
    terminal_time = terminal_phase = terminal_reason = None
    last_phase = "damping"
    configured = set()
    config = load_config()
    case = next(c for c in data["cases"] if c["case_id"] == case_id)
    try:
        scene, requests = build_fixture(output/"scenario", data["sumo_binary"])
        ledger = RequestLedger(requests)
        baseline = CirclesController(config)
        api = SimpleNamespace(vehicle=LoggedVehicle(traci.vehicle, writes), lane=traci.lane)
        owner = CommandOwnership(api, MEMBERS, baseline)
        sensor = SumoLocalSensor(api, config, {lane: 0. for lane in LANES})
        buffer = RSUReportBuffer(max_age_s=1., history_s=20.)
        history = []
        job = dict(arm="wall", decision_time_s=DECISION_S, policy_spec=dict(
            member_ids=list(MEMBERS), required_lanes=list(LANES), profile=data["profile"]))
        policy = policy_factory(job, data)
        start_sumo(data["sumo_binary"], scene, seed=1, config=config)
        for _ in range(int(HORIZON_S/config.physics_dt_s)):
            traci.simulationStep()
            now = float(traci.simulation.getTime())
            manifest["completed_simulated_seconds"] = now
            live = set(traci.vehicle.getIDList())
            for vid in live-configured:
                traci.vehicle.setLaneChangeMode(vid, 512)
                configured.add(vid)
            ledger.advance(now, list(traci.simulation.getDepartedIDList()), list(traci.simulation.getArrivedIDList()))
            collisions += int(traci.simulation.getCollidingVehiclesNumber())
            teleports += int(traci.simulation.getStartingTeleportNumber())
            all_vehicles = [dict(id=v, lane=traci.vehicle.getLaneID(v), position_m=float(traci.vehicle.getLanePosition(v)),
                distance_m=float(traci.lane.getLength(traci.vehicle.getLaneID(v)))-float(traci.vehicle.getLanePosition(v)),
                speed_mps=float(traci.vehicle.getSpeed(v)), acceleration_mps2=float(traci.vehicle.getAcceleration(v)),
                speed_mode=traci.vehicle.getSpeedMode(v), type_id=traci.vehicle.getTypeID(v)) for v in sorted(live)]
            for key, value in braking_effort([r["acceleration_mps2"] for r in all_vehicles], .5).items():
                totals[key] += value
            exposure, ttc = ttc_exposure(traci, live, .5)
            totals["ttc_below_3_vehicle_seconds"] += exposure
            if ttc is not None:
                minimum_ttc = ttc if minimum_ttc is None else min(minimum_ttc, ttc)
            frames = {v: sensor.capture(v, now_s=now) for v in sorted(live & set(MEMBERS))}
            physical.append(dict(time_s=now, outcome_interval_phase=last_phase, all_vehicles=all_vehicles,
                acquired_member_frames={v: asdict(f) for v, f in frames.items()}))
            for vid, frame in frames.items():
                if vid in previous_records:
                    response = owner.record_response(vid, timestamp_s=now, speed_mps=frame.speed_mps,
                                                     acceleration_mps2=frame.acceleration_mps2)
                    telemetry.append(dict(kind="measured_response", **response.as_dict()))
            if collisions or teleports:
                failures.append("collision_or_teleport")
                break
            if not now.is_integer() or now >= HORIZON_S:
                continue
            for vid, frame in frames.items():
                dropping = case["packet_drop_from_s"] is not None and now >= case["packet_drop_from_s"] and vid == MEMBERS[1]
                if dropping:
                    if now == case["packet_drop_from_s"]:
                        disturbances.append(dict(time_s=now, vehicle_id=vid, kind="registered_rsu_packet_drop",
                            physics_unchanged=True, local_damping_available=True))
                    continue
                previous = previous_records.get(vid)
                packet = CAVReport(frame, now, previous.owner if previous else "native",
                    previous.network_output_mps2 if previous else None,
                    previous.issued_acceleration_mps2 if previous else None,
                    previous.fallback_reason if previous else None)
                buffer.receive(packet, received_at_s=now)
                history.append((now, packet))
                telemetry.append(dict(kind="rsu_received_report", received_at_s=now, **packet.payload()))
            history = [(receipt, packet) for receipt, packet in history if receipt >= now-20.]
            reports = tuple(buffer.reports(now_s=now))
            expected = {v: baseline.decide(f, now_s=now) for v, f in frames.items()}
            current_reporters = {r.frame.vehicle_id for r in reports if r.frame.timestamp_s == now}
            # Local decisions survive packet loss physically, but the RSU receives
            # no current execution assertion for the deliberately dropped CAV.
            references = {v: d.command_acceleration_mps2 for v, d in expected.items()
                          if d.fallback_reason is None and v in current_reporters}
            natives = frozenset(v for v, d in expected.items() if d.fallback_reason is not None and v in current_reporters)
            result = policy(PolicyInput(now, reports, tuple(history), references, natives))
            policies.append(dict(time_s=now, **result.payload()))
            if policy.proposal is not None and proposal_record is None:
                proposal_record = asdict(policy.proposal)
            if now < DECISION_S and (result.phase != "damping" or result.wall_commands):
                failures.append("acted_before_registered_decision")
            write_start = len(writes)
            records = owner.step(frames, now_s=now, live_ids=live, wall_commands=result.wall_commands)
            failures.extend(command_failures(result, records, previous_wall, frames))
            expected_writes = sum(r.issued_acceleration_mps2 is not None or r.released_to_native for r in records)
            if len(writes)-write_start != expected_writes:
                failures.append("competing_or_missing_longitudinal_writer")
            for record in records:
                telemetry.append(dict(kind="command", **record.as_dict()))
                telemetry.append(dict(kind="identical_damping_reference", vehicle_id=record.vehicle_id,
                                      timestamp_s=now, **expected[record.vehicle_id].as_dict()))
            previous_records = {r.vehicle_id: r for r in records}
            previous_wall = {r.vehicle_id: r.issued_acceleration_mps2 for r in records if r.owner == "wall"}
            measured_members = [{k: row[k] for k in ("id", "lane", "distance_m", "speed_mps")}
                                for row in all_vehicles if row["id"] in MEMBERS]
            if now >= DECISION_S and terminal_time is None:
                samples.append(dict(time_s=now, phase=result.phase, reason=result.reason, members=measured_members,
                    owners={r.vehicle_id: r.owner for r in records}, wall_command_ids=sorted(result.wall_commands),
                    target_speed_mps=result.details.get("target_speed_mps"),
                    assembly_completed_at_s=result.details.get("assembly_completed_at_s")))
            if result.phase in {"release", "abort", "refused"}:
                first_terminal = terminal_time is None
                if first_terminal:
                    terminal_time, terminal_phase, terminal_reason = now, result.phase, result.reason
                for record in records:
                    reference = expected[record.vehicle_id]
                    same = (record.owner == ("native" if reference.fallback_reason else "damping")
                        and record.issued_acceleration_mps2 == reference.command_acceleration_mps2
                        and record.issued_target_speed_mps == reference.target_speed_mps
                        and record.fallback_reason == reference.fallback_reason)
                    handbacks.append(dict(time_s=now, vehicle_id=record.vehicle_id, matched=same,
                        terminal_tick=first_terminal, actual_override_release=record.released_to_native,
                        baseline_native=reference.fallback_reason is not None))
                    if not same:
                        failures.append("handover_not_identical_damping")
            last_phase = result.phase
            if failures:
                break
        if owner is not None:
            owner.release_all(live_ids=traci.vehicle.getIDList())
        delivery = None
        if samples and samples[0]["phase"] == "assembly":
            delivery = formation_delivery(samples, required_lanes=LANES, assembly_deadline_s=50.,
                hold_duration_s=20., formed_span_m=50., speed_spread_mps=1., readiness_samples=3)
        readiness_streak, hold_entry_target_ready = 0, False
        for sample in samples:
            rows = sample["members"]
            settled = (len(rows) == 2 and sample["target_speed_mps"] is not None
                and max(r["distance_m"] for r in rows)-min(r["distance_m"] for r in rows) <= 50.
                and max(r["speed_mps"] for r in rows)-min(r["speed_mps"] for r in rows) <= 1.
                and all(abs(r["speed_mps"]-sample["target_speed_mps"]) <= 1. for r in rows))
            readiness_streak = readiness_streak+1 if settled else 0
            if sample["phase"] == "hold":
                hold_entry_target_ready = readiness_streak >= 3
                break
        wall_records = [r for r in telemetry if r["kind"] == "command" and r["owner"] == "wall"]
        first_handbacks = [r for r in handbacks if r["terminal_tick"]]
        report_loss_ok = (case["expected"] != "aborted" or (terminal_phase == "abort"
            and 15. <= terminal_time <= 17. and len(disturbances) == 1
            and any(p["wall_commands"] for p in policies if p["time_s"] < 15.)))
        observed = "delivered" if delivery and delivery["delivered"] and hold_entry_target_ready else "aborted" if terminal_phase == "abort" else terminal_phase
        hold_physics = [row for row in physical if row["outcome_interval_phase"] == "hold"]
        halfstep_hold_compact = all(len(members := [r for r in row["all_vehicles"] if r["id"] in MEMBERS]) == 2
            and {r["lane"] for r in members} == set(LANES)
            and max(r["distance_m"] for r in members)-min(r["distance_m"] for r in members) <= 50.
            and max(r["speed_mps"] for r in members)-min(r["speed_mps"] for r in members) <= 1.
            for row in hold_physics)
        checks = dict(expected_outcome=observed == case["expected"], command_and_runtime_contract=not failures,
            nonvacuous_wall_commands=len(wall_records) >= 2, complete_registered_horizon=manifest["completed_simulated_seconds"] == HORIZON_S,
            zero_collisions_teleports=collisions == teleports == 0,
            only_commandownership_cav_writes=all(w[1] in MEMBERS for w in writes),
            unchanged_circles_instance=owner.controller is baseline,
            all_vehicles_idm_and_safety31=all(r["type_id"] == "idm" and r["speed_mode"] == 31 for row in physical for r in row["all_vehicles"]),
            actual_identical_handover=bool(first_handbacks) and all(r["matched"] for r in handbacks),
            actual_native_override_release=bool(first_handbacks) and all(not r["baseline_native"] or r["actual_override_release"] for r in first_handbacks),
            hold_compact_at_physics_clock=halfstep_hold_compact,
            three_target_settled_readiness_samples=case["expected"] != "delivered" or hold_entry_target_ready,
            full_twenty_second_hold=case["expected"] != "delivered" or (bool(delivery) and delivery["compact_hold_delivered"]),
            report_drop_aborts_without_physics_change=report_loss_ok,
            warmup_has_no_wall_commands=all(not p["wall_commands"] for p in policies if p["time_s"] < DECISION_S))
        summary = dict(manifest, status="complete", passed=all(checks.values()), checks=checks,
            expected=case["expected"], observed=observed, terminal_time_s=terminal_time,
            terminal_phase=terminal_phase, terminal_reason=terminal_reason, formation_delivery=delivery,
            failures=failures, collisions=collisions, teleports=teleports, physical_totals=totals,
            minimum_ttc_s=minimum_ttc, physical_diagnostics=_physical_diagnostics(physical),
            service=service_summary(ledger, manifest["completed_simulated_seconds"]),
            handover_evidence=handbacks, wall_command_count=len(wall_records),
            interpretation="Actual isolated SUMO execution; no natural-traffic efficacy or universal braking/safety guarantee.")
        _write_json(output/"summary.json", summary)
        manifest.update(status="complete", passed=summary["passed"])
        return summary
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        close_sumo()
        _write_json(output/"longitudinal_writes.json", writes)
        _write_json(output/"disturbances.json", disturbances)
        _write_json(output/"proposal.json", proposal_record)
        for name, rows in (("physical.jsonl", physical), ("telemetry.jsonl", telemetry),
                           ("policy.jsonl", policies), ("formation_samples.jsonl", samples)):
            with (output/name).open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False)+"\n")
        manifest["artifact_sha256"] = {name: hash_file(output/name) for name in (*ARTIFACTS, "summary.json",
            "scenario/network.net.xml", "scenario/routes.rou.xml", "scenario/scenario.sumocfg") if (output/name).exists()}
        _write_json(output/"manifest.json", manifest)
