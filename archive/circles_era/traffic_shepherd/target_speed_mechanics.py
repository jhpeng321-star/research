"""One bounded mechanics fixture for the registered uncapped target rule.

Verifies, in an isolated 120 s SUMO fixture with the same policy code path as
the corridor (target_speed_wall.policy_factory), that the uncapped rule
commands mean-minus-2 (25 m/s from a 27 m/s cruise, i.e. above the old 24 m/s
cap), that the wall reaches and maintains it through measured readiness, and
that the unchanged coast exit and identical-damping handback still hold.
Registration alone launches nothing; this is mechanical verification, never
traffic efficacy.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from .config import ROOT, CONFIG_PATH, load_config
from .damping import CirclesController
from .gentle_mechanics import _identity, _write_json, command_failures, _physical_diagnostics
from .mechanics import LoggedVehicle
from .metrics import RequestLedger, braking_effort, service_summary, ttc_exposure
from .ownership import CommandOwnership
from .paired_experiments import PolicyInput, digest
from .reporting import CAVReport, RSUReportBuffer
from .scenarios import TripRequest
from .sensing import SumoLocalSensor
from .simulation import close_sumo, start_sumo
from .target_speed_wall import TargetSpeedProfile, VERSION as RULE_VERSION, policy_factory
from .verification import hash_file

SCHEMA = "target_speed_mechanics/1"
MEMBERS = ("member_a", "member_b")
FOLLOWERS = ("follower_a", "follower_b")
LANES = ("road_0", "road_1")
CASE_ID = "uncapped_target_complete"
ARTIFACTS = ("physical.jsonl", "telemetry.jsonl", "policy.jsonl", "formation_samples.jsonl",
             "longitudinal_writes.json", "disturbances.json", "proposal.json")
HORIZON_S, DECISION_S = 120., 5.
ROAD_SPEED_MPS, CRUISE_MPS = 33., 27.
EXPECTED_TARGET_MPS = CRUISE_MPS - 2.


def plan():
    return dict(schema=SCHEMA, policy_version=RULE_VERSION, profile=asdict(TargetSpeedProfile()),
        cases=[dict(case_id=CASE_ID, expected="delivered")],
        target_rule="reported_member_mean_speed_mps_minus_2_bounded_by_member_allowed_speed",
        expected_commanded_target_mps=EXPECTED_TARGET_MPS,
        decision_s=DECISION_S, horizon_s=HORIZON_S, warmup_included_s=DECISION_S,
        maximum_attempts=1, maximum_simulated_seconds=120, automatic_retries=0,
        execution_order=[CASE_ID], simultaneous_runs=False, automatic_successors=False,
        fixture=dict(lanes=list(LANES), length_m=3500., road_speed_mps=ROAD_SPEED_MPS,
            member_ids=list(MEMBERS), follower_ids=list(FOLLOWERS), member_positions_m=[1000., 990.],
            follower_positions_m=[910., 900.], initial_speed_mps=CRUISE_MPS, vehicle_length_m=5.,
            all_vehicle_model="IDM", fixed_lane_fixture_mode=512, physics_s=.5,
            vtype_max_speed_mps=CRUISE_MPS,
            note="Edge speed 33 m/s keeps the road permissive; vtype cruise 27 m/s reproduces "
                 "corridor-like ambient speed so the uncapped rule commands 25 m/s > old 24 cap"),
        criteria=dict(wall_acceleration_min_mps2=-.3, wall_acceleration_max_mps2=1.5,
            maximum_command_change_mps2=.15, maximum_entry_to_measured_change_mps2=.15,
            minimum_wall_command_count=2, formed_span_m=50., speed_spread_mps=1.,
            target_tolerance_mps=1., readiness_samples=3, maximum_assembly_s=50., hold_s=20.,
            commanded_target_equals_uncapped_rule=True, commanded_target_above_old_cap_mps=24.,
            require_actual_identical_damping_handover=True, require_complete_horizon=True,
            require_only_cav_writes=True, zero_collisions=True, zero_teleports=True),
        stop_rules=["Any failed allocation remains charged; never retry or overwrite it.",
            "Stop a trajectory immediately on a command, ownership, sensing, collision or teleport defect.",
            "No successor, sweep or corridor launch follows automatically from this fixture."],
        mechanical_not_traffic_efficacy=True, actual_braking_not_universally_limited=True)


def build_fixture(output, sumo_binary):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/"nodes.xml").write_text('<nodes><node id="start" x="0" y="0"/><node id="end" x="3500" y="0"/></nodes>')
    (output/"edges.xml").write_text(f'<edges><edge id="road" from="start" to="end" numLanes="2" speed="{ROAD_SPEED_MPS:g}"/></edges>')
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
        length="5", maxSpeed=f"{CRUISE_MPS:g}", lcStrategic="0", lcCooperative="0",
        lcSpeedGain="0", lcKeepRight="0"))
    ET.SubElement(routes, "route", dict(id="route", edges="road"))
    for i in range(2):
        for vehicle, position in ((MEMBERS[i], 1000.-10*i), (FOLLOWERS[i], 910.-10*i)):
            ET.SubElement(routes, "vehicle", dict(id=vehicle, type="idm", route="route", depart="0",
                departLane=str(i), departPos=str(position), departSpeed=f"{CRUISE_MPS:g}"))
    ET.indent(routes)
    ET.ElementTree(routes).write(output/"routes.rou.xml", encoding="utf-8", xml_declaration=True)
    (output/"scenario.sumocfg").write_text('<configuration><input><net-file value="network.net.xml"/>'
        '<route-files value="routes.rou.xml"/></input><processing><time-to-teleport value="-1"/>'
        '<collision.action value="warn"/></processing></configuration>')
    requests = [TripRequest(v, 0., "route", "idm", v in MEMBERS) for v in (*MEMBERS, *FOLLOWERS)]
    return output/"scenario.sumocfg", requests


def register(output):
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
    return dict(registration=str(output/"registration.json"), case_ids=[CASE_ID],
                maximum_simulated_seconds=120, simulations_started=0)


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
    allocation = json.loads((root/"allocations"/f"{case_id}.json").read_text(encoding="utf-8"))
    if allocation != dict(case_id=case_id, allocated_simulated_seconds=HORIZON_S, retries=0,
                          registration_sha256=data["registration_sha256"]):
        raise ValueError("mechanics allocation changed")
    manifest = json.loads((output/"manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output/"summary.json").read_text(encoding="utf-8"))
    if (manifest.get("registration_sha256") != data["registration_sha256"]
            or manifest.get("status") != "complete" or not manifest.get("passed")
            or manifest.get("completed_simulated_seconds") != HORIZON_S
            or summary.get("completed_simulated_seconds") != HORIZON_S or not summary.get("passed")):
        raise ValueError("mechanics case is not a completed passing run")
    for name, expected in manifest["artifact_sha256"].items():
        if hash_file(output/name) != expected:
            raise ValueError("mechanics artifact changed")
    return dict(manifest_sha256=hash_file(output/"manifest.json"),
                actual_simulated_seconds=HORIZON_S, passed=True)


def verify_completed(registration):
    root, data = read_registration(registration, require_current=False)
    if not (root/"jobs"/CASE_ID/"summary.json").exists():
        raise ValueError("target-speed mechanics case has not run")
    summary = json.loads((root/"jobs"/CASE_ID/"summary.json").read_text(encoding="utf-8"))
    if summary.get("passed"):
        proof = _verify_case(root, data, CASE_ID)
        return dict(registration_path=str(root/"registration.json"),
                    registration_sha256=data["registration_sha256"], case_id=CASE_ID,
                    checks=summary["checks"], commanded_target_mps=summary["commanded_target_mps"],
                    proof=proof)
    return mechanics_law_audit(registration)


def mechanics_law_audit(registration):
    """Offline verification of a completed-but-unpassed fixture, without rerunning.

    The first physical fixture simulated its full 120 s and passed every
    runtime check except commanded_target_equals_uncapped_rule, whose
    registered formulation compared the commanded target to an absolute
    25.0 m/s with 1e-9 tolerance. The target law is mean(reported member
    speeds at the decision tick) - 2 m/s, so this audit recomputes that one
    check from the hash-bound immutable artifacts. The charged failure and
    its STOP record stay preserved; no additional fixture is allocated.
    """
    path = Path(registration).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("registration_sha256") != digest({k: v for k, v in data.items() if k != "registration_sha256"}):
        raise ValueError("mechanics registration changed")
    root = path.parent
    if not (root/"STOP.json").exists():
        raise ValueError("law audit applies only to a preserved stopped fixture block")
    output = root/"jobs"/CASE_ID
    manifest = json.loads((output/"manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output/"summary.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("completed_simulated_seconds") != HORIZON_S:
        raise ValueError("law audit requires a completed full-horizon fixture")
    for name, expected in manifest["artifact_sha256"].items():
        if hash_file(output/name) != expected:
            raise ValueError("law audit artifact changed")
    checks = dict(summary["checks"])
    failed = [k for k, v in checks.items() if not v]
    if failed != ["commanded_target_equals_uncapped_rule"]:
        raise ValueError(f"law audit covers only the target-law formulation defect: {failed}")
    reported = None
    with (output/"physical.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["time_s"] == DECISION_S:
                reported = {v: f["speed_mps"] for v, f in row["acquired_member_frames"].items()}
                break
    if reported is None or set(reported) != set(MEMBERS):
        raise ValueError("decision-tick member reports missing for law audit")
    expected = max(0., sum(reported.values())/len(reported) - 2.)
    commanded = summary["commanded_target_mps"]
    if isinstance(commanded, list) or abs(commanded-expected) > 1e-6 or not commanded > 24.:
        raise ValueError("commanded target is not the uncapped report-based rule")
    checks["commanded_target_equals_uncapped_rule"] = True
    passed = all(checks.values())
    return dict(registration_path=str(path), registration_sha256=data["registration_sha256"],
        case_id=CASE_ID, checks=checks, passed=passed, audit="offline_law_recomputation/1",
        original_failed_checks=["commanded_target_equals_uncapped_rule"],
        original_check_formulation="absolute 25.0 m/s with 1e-9 tolerance",
        corrected_formulation="mean(reported member speeds at decision) - 2 m/s, above old 24 m/s cap",
        reported_member_speeds_mps_at_decision=reported, expected_uncapped_target_mps=expected,
        commanded_target_mps=commanded, stop_record_preserved=True, rerun_allocated=False,
        artifact_binding=manifest["artifact_sha256"], proof=dict(
            manifest_sha256=hash_file(output/"manifest.json"), actual_simulated_seconds=HORIZON_S,
            passed=passed))


def run_one(registration, case_id=CASE_ID):
    if case_id != CASE_ID:
        raise ValueError("unknown target-speed mechanics case")
    root, data = read_registration(registration)
    allocations = root/"allocations"
    allocations.mkdir(exist_ok=True)
    with (allocations/f"{case_id}.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(case_id=case_id, allocated_simulated_seconds=HORIZON_S, retries=0,
                       registration_sha256=data["registration_sha256"]), stream, indent=2)
    try:
        summary = _run_allocated(root, data, case_id)
        if not summary["passed"]:
            _write_json(root/"STOP.json", dict(case_id=case_id, reason="mechanics_check_failed",
                                               checks=summary["checks"]))
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
    physical, telemetry, policies, samples, writes = [], [], [], [], []
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
        from .escalation_evidence import formation_delivery
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
        commanded = sorted({p["target_speed_mps"] for p in policies if p.get("target_speed_mps") is not None})
        uncapped_law = (len(commanded) == 1 and abs(commanded[0]-EXPECTED_TARGET_MPS) <= 1e-9
            and commanded[0] > plan()["criteria"]["commanded_target_above_old_cap_mps"])
        observed = "delivered" if delivery and delivery["delivered"] and hold_entry_target_ready else terminal_phase
        hold_physics = [row for row in physical if row["outcome_interval_phase"] == "hold"]
        halfstep_hold_compact = all(len(members := [r for r in row["all_vehicles"] if r["id"] in MEMBERS]) == 2
            and {r["lane"] for r in members} == set(LANES)
            and max(r["distance_m"] for r in members)-min(r["distance_m"] for r in members) <= 50.
            and max(r["speed_mps"] for r in members)-min(r["speed_mps"] for r in members) <= 1.
            for row in hold_physics)
        checks = dict(expected_outcome=observed == "delivered", command_and_runtime_contract=not failures,
            nonvacuous_wall_commands=len(wall_records) >= 2, complete_registered_horizon=manifest["completed_simulated_seconds"] == HORIZON_S,
            zero_collisions_teleports=collisions == teleports == 0,
            only_commandownership_cav_writes=all(w[1] in MEMBERS for w in writes),
            unchanged_circles_instance=owner.controller is baseline,
            all_vehicles_idm_and_safety31=all(r["type_id"] == "idm" and r["speed_mode"] == 31 for row in physical for r in row["all_vehicles"]),
            actual_identical_handover=bool(first_handbacks) and all(r["matched"] for r in handbacks),
            actual_native_override_release=bool(first_handbacks) and all(not r["baseline_native"] or r["actual_override_release"] for r in first_handbacks),
            hold_compact_at_physics_clock=halfstep_hold_compact,
            three_target_settled_readiness_samples=hold_entry_target_ready,
            full_twenty_second_hold=bool(delivery) and delivery["compact_hold_delivered"],
            commanded_target_equals_uncapped_rule=uncapped_law,
            warmup_has_no_wall_commands=all(not p["wall_commands"] for p in policies if p["time_s"] < DECISION_S))
        summary = dict(manifest, status="complete", passed=all(checks.values()), checks=checks,
            expected="delivered", observed=observed, terminal_time_s=terminal_time,
            terminal_phase=terminal_phase, terminal_reason=terminal_reason, formation_delivery=delivery,
            commanded_target_mps=commanded[0] if len(commanded) == 1 else commanded,
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
        _write_json(output/"disturbances.json", [])
        _write_json(output/"proposal.json", proposal_record)
        for name, rows in (("physical.jsonl", physical), ("telemetry.jsonl", telemetry),
                           ("policy.jsonl", policies), ("formation_samples.jsonl", samples)):
            with (output/name).open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False)+"\n")
        manifest["artifact_sha256"] = {name: hash_file(output/name) for name in (*ARTIFACTS, "summary.json",
            "scenario/network.net.xml", "scenario/routes.rou.xml", "scenario/scenario.sumocfg") if (output/name).exists()}
        _write_json(output/"manifest.json", manifest)
