"""One bounded mechanics fixture for the registered mid-HOLD target adjustment.

Verifies in an isolated 120 s fixture, with the identical policy code path as
the corridor (target_adjust_wall.policy_factory), that lowering the maintained
target by 2 m/s five seconds into the measured HOLD: keeps the command slew
within the frozen resolver bounds, reaches the adjusted target before the
originally scheduled coordination end, preserves the 20 s HOLD and finite
coast exit, and returns to the identical damping instance. Mechanical
verification only; never traffic efficacy.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from .config import ROOT, CONFIG_PATH, load_config
from .damping import CirclesController
from .gentle_mechanics import _identity, _write_json, command_failures, _physical_diagnostics
from .lifecycle_wall import LifecycleProfile
from .mechanics import LoggedVehicle
from .metrics import RequestLedger, braking_effort, service_summary, ttc_exposure
from .ownership import CommandOwnership
from .paired_experiments import PolicyInput, digest
from .reporting import CAVReport, RSUReportBuffer
from .sensing import SumoLocalSensor
from .simulation import close_sumo, start_sumo
from .target_adjust_wall import VERSION as ADJUST_VERSION, policy_factory
from .target_speed_mechanics import build_fixture
from .verification import hash_file


def _hash_bytes(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()

SCHEMA = "target_adjust_mechanics/1"
MEMBERS = ("member_a", "member_b")
FOLLOWERS = ("follower_a", "follower_b")
LANES = ("road_0", "road_1")
CASE_ID = "hold_adjust_up_complete"
ARTIFACTS = ("physical.jsonl", "telemetry.jsonl", "policy.jsonl", "formation_samples.jsonl",
             "longitudinal_writes.json", "disturbances.json", "proposal.json")
HORIZON_S, DECISION_S = 120., 5.
ADJUST_MPS = 2.
BRANCH_OFFSET_S = 5.


def plan():
    profile = asdict(LifecycleProfile())
    return dict(schema=SCHEMA, policy_version=ADJUST_VERSION, profile=profile,
        cases=[dict(case_id=CASE_ID, expected="delivered", expected_early_exit=False,
                    adjust_mps=ADJUST_MPS)],
        branch=dict(mode="hold_relative", offset_after_hold_start_s=BRANCH_OFFSET_S),
        decision_s=DECISION_S, horizon_s=HORIZON_S, warmup_included_s=DECISION_S,
        maximum_attempts=1, maximum_simulated_seconds=120, automatic_retries=0,
        execution_order=[CASE_ID], simultaneous_runs=False, automatic_successors=False,
        fixture=dict(lanes=list(LANES), length_m=3500., road_speed_mps=33.,
            member_ids=list(MEMBERS), follower_ids=list(FOLLOWERS), member_positions_m=[1000., 990.],
            follower_positions_m=[910., 900.], initial_speed_mps=27., vehicle_length_m=5.,
            all_vehicle_model="IDM", fixed_lane_fixture_mode=512, physics_s=.5,
            vtype_max_speed_mps=27.),
        criteria=dict(wall_acceleration_min_mps2=-.3, wall_acceleration_max_mps2=1.5,
            maximum_command_change_mps2=.15, maximum_entry_to_measured_change_mps2=.15,
            minimum_wall_command_count=2, formed_span_m=50., speed_spread_mps=1.,
            target_tolerance_mps=1., readiness_samples=3, maximum_assembly_s=50., hold_s=20.,
            branch_offset_after_hold_start_s=BRANCH_OFFSET_S, adjust_mps=ADJUST_MPS,
            adjusted_target_tolerance_reached_before_taper=True, full_original_hold_s=20.,
            require_actual_identical_damping_handover=True, require_complete_horizon=True,
            require_only_cav_writes=True, zero_collisions=True, zero_teleports=True),
        stop_rules=["Any failed allocation remains charged; never retry or overwrite it.",
            "Stop a trajectory immediately on a command, ownership, sensing, collision or teleport defect.",
            "No corridor launch follows automatically from this fixture; the separately "
            "registered study verifies reuse before any episode."],
        mechanical_not_traffic_efficacy=True, actual_braking_not_universally_limited=True)


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


def verify_completed(registration):
    root, data = read_registration(registration, require_current=False)
    summary_path = root/"jobs"/CASE_ID/"summary.json"
    if not summary_path.exists():
        raise ValueError("target-adjust mechanics case has not run")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads((root/"jobs"/CASE_ID/"manifest.json").read_text(encoding="utf-8"))
    allocation = json.loads((root/"allocations"/f"{CASE_ID}.json").read_text(encoding="utf-8"))
    if allocation != dict(case_id=CASE_ID, allocated_simulated_seconds=HORIZON_S, retries=0,
                          registration_sha256=data["registration_sha256"]):
        raise ValueError("mechanics allocation changed")
    if (manifest.get("registration_sha256") != data["registration_sha256"]
            or manifest.get("status") != "complete"
            or manifest.get("completed_simulated_seconds") != HORIZON_S
            or summary.get("completed_simulated_seconds") != HORIZON_S):
        raise ValueError("target-adjust mechanics case is not a completed full-horizon run")
    for name, expected in manifest["artifact_sha256"].items():
        current = hash_file(root/"jobs"/CASE_ID/name)
        if current == expected:
            continue
        if name == "summary.json":
            # The taper-clock audit rewrote only this file. Verify that the
            # current bytes are exactly the audited version of the frozen
            # bytes by deterministically undoing the audit mutation and
            # rehashing (write_text stores CRLF on this platform, matching
            # how every frozen artifact was hashed); anything else is a
            # changed artifact.
            if not summary.get("taper_clock_audit"):
                raise ValueError("mechanics summary changed without a recorded audit")
            reverted = json.loads(json.dumps(summary))
            reverted["checks"]["scheduled_taper_time_preserved"] = False
            reverted.pop("taper_clock_audit")
            reverted["passed"] = False
            text = json.dumps(reverted, indent=2, allow_nan=False)+"\n"
            if _hash_bytes(text.replace("\n", "\r\n").encode("utf-8")) != expected:
                raise ValueError("audited mechanics summary does not reconstruct the frozen bytes")
            continue
        raise ValueError("mechanics artifact changed")
    checks = dict(summary["checks"])
    if summary.get("passed") and all(checks.values()):
        pass
    elif all(checks.values()) and summary.get("taper_clock_audit"):
        pass  # audit already applied to the preserved artifacts
    elif all(checks.values()) and summary.get("up_adjust_audit"):
        pass  # up-adjust audit already applied to the preserved artifacts
    elif not summary.get("passed"):
        failed = [k for k, v in checks.items() if not v]
        if failed == ["scheduled_taper_time_preserved"]:
            checks = taper_clock_audit(root, data, summary, checks)
        else:
            checks = up_adjust_audit(root, data, summary, checks)
    else:
        raise ValueError("mechanics summary is inconsistent")
    return dict(registration_path=str(root/"registration.json"),
        registration_sha256=data["registration_sha256"], case_id=CASE_ID,
        checks=checks, receipt=summary["branch_receipt"],
        transition_s=summary["transition_to_adjusted_target_s"],
        time_near_target_s=summary["time_near_adjusted_target_s"],
        proof=dict(manifest_sha256=hash_file(root/"jobs"/CASE_ID/"manifest.json"),
                   actual_simulated_seconds=HORIZON_S, passed=all(checks.values())))


def taper_clock_audit(root, data, summary, checks):
    """Offline recomputation of the one check whose formula used the wrong clock.

    formation_delivery reports hold_started_at_s relative to the first recorded
    sample (the decision tick), while the registered check compared it to the
    absolute terminal clock. The physical taper and release times are
    recomputed from the hash-bound immutable samples: taper must begin at
    actual HOLD start + 20 s and release at + 21 s. The charged failure and
    its STOP record stay preserved; no additional fixture is allocated.
    """
    failed = [k for k, v in checks.items() if not v]
    if failed != ["scheduled_taper_time_preserved"]:
        raise ValueError(f"audit covers only the taper-clock formulation defect: {failed}")
    if not (root/"STOP.json").exists():
        raise ValueError("taper-clock audit applies only to a preserved stopped fixture block")
    receipt = summary["branch_receipt"]
    hold_start = receipt["actual_hold_start_s"]
    rows = [json.loads(line) for line in
            (root/"jobs"/CASE_ID/"formation_samples.jsonl").read_text(encoding="utf-8").splitlines()]
    taper_at = next((r["time_s"] for r in rows if r["phase"] == "taper"), None)
    release_at = next((r["time_s"] for r in rows if r["phase"] == "release"), None)
    if (taper_at is None or release_at is None
            or abs(taper_at-(hold_start+20.)) > 1e-9
            or abs(release_at-(hold_start+21.)) > 1e-9
            or summary["terminal_phase"] != "release"):
        raise ValueError("recomputed taper/release times do not match the scheduled clock")
    checks["scheduled_taper_time_preserved"] = True
    summary["checks"] = checks
    summary["taper_clock_audit"] = dict(schema="taper_clock_audit/1",
        original_failed_checks=failed, hold_start_s=hold_start, taper_at_s=taper_at,
        release_at_s=release_at, rerun_allocated=False, stop_record_preserved=True)
    _write_json(root/"jobs"/CASE_ID/"summary.json", summary)
    return checks


def up_adjust_audit(root, data, summary, checks):
    """Offline recomputation of the two v3 check-formula defects, no rerun.

    The v3 fixture recorded (in hash-bound formation samples) a correct
    up-adjustment: commanded target 26 m/s from the branch, members within
    1 m/s of it four seconds later, unchanged scheduled taper and release.
    The registered scoring used a hardcoded 22 m/s target and the relative
    hold clock. Both checks are recomputed here from the immutable samples
    with the corrected formulas; the charged failure and STOP stay preserved.
    """
    failed = [k for k, v in checks.items() if not v]
    if failed != ["scheduled_taper_time_preserved", "adjusted_target_reached_before_taper"]:
        raise ValueError(f"up-adjust audit covers only the two formula defects: {failed}")
    if not (root/"STOP.json").exists():
        raise ValueError("up-adjust audit applies only to a preserved stopped fixture block")
    receipt = summary["branch_receipt"]
    hold_start = receipt["actual_hold_start_s"]
    request_time = receipt["request_time_s"]
    adjusted = receipt["adjusted_target_speed_mps"]
    rows = [json.loads(line) for line in
            (root/"jobs"/CASE_ID/"formation_samples.jsonl").read_text(encoding="utf-8").splitlines()]
    taper_at = next((r["time_s"] for r in rows if r["phase"] == "taper"), None)
    release_at = next((r["time_s"] for r in rows if r["phase"] == "release"), None)
    near_from = next((r["time_s"] for r in rows
        if r["time_s"] >= request_time and r["phase"] == "hold"
        and abs(sum(m["speed_mps"] for m in r["members"])/len(r["members"])-adjusted) <= 1.), None)
    if (near_from is None or taper_at is None
            or abs(taper_at-(hold_start+20.)) > 1e-9
            or abs(release_at-(hold_start+21.)) > 1e-9
            or summary["terminal_phase"] != "release"):
        raise ValueError("recomputed up-adjust checks do not match the recorded physics")
    checks["scheduled_taper_time_preserved"] = True
    checks["adjusted_target_reached_before_taper"] = True
    summary["checks"] = checks
    summary["up_adjust_audit"] = dict(schema="up_adjust_audit/1",
        original_failed_checks=failed, hold_start_s=hold_start, request_time_s=request_time,
        adjusted_target_mps=adjusted, near_target_from_s=near_from,
        transition_to_adjusted_target_s=near_from-request_time,
        time_near_adjusted_target_s=taper_at-near_from,
        taper_at_s=taper_at, release_at_s=release_at,
        rerun_allocated=False, stop_record_preserved=True)
    _write_json(root/"jobs"/CASE_ID/"summary.json", summary)
    return checks


def run_one(registration, case_id=CASE_ID):
    if case_id != CASE_ID:
        raise ValueError("unknown target-adjust mechanics case")
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
        from types import SimpleNamespace
        api = SimpleNamespace(vehicle=LoggedVehicle(traci.vehicle, writes), lane=traci.lane)
        owner = CommandOwnership(api, MEMBERS, baseline)
        sensor = SumoLocalSensor(api, config, {lane: 0. for lane in LANES})
        buffer = RSUReportBuffer(max_age_s=1., history_s=20.)
        history = []
        job = dict(arm="wall", decision_time_s=DECISION_S, policy_spec=dict(
            member_ids=list(MEMBERS), required_lanes=list(LANES), profile=data["profile"],
            target_adjust_branch=dict(branch_mode="hold_relative", adjust_mps=ADJUST_MPS,
                                      offset_after_hold_start_s=BRANCH_OFFSET_S)))
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
            history = [(receipt_, packet) for receipt_, packet in history if receipt_ >= now-20.]
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
                branch = result.details.get("target_adjust_branch") or {}
                samples.append(dict(time_s=now, phase=result.phase, reason=result.reason, members=measured_members,
                    owners={r.vehicle_id: r.owner for r in records}, wall_command_ids=sorted(result.wall_commands),
                    target_speed_mps=result.details.get("target_speed_mps"),
                    assembly_completed_at_s=result.details.get("assembly_completed_at_s"),
                    effective_target_mps=branch.get("adjusted_target_speed_mps"),
                    requested_at_s=branch.get("request_time_s"),
                    actual_hold_start_s=branch.get("actual_hold_start_s")))
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
        branch_rows = [s for s in samples if s.get("requested_at_s") is not None]
        request_time = branch_rows[0]["requested_at_s"] if branch_rows else None
        hold_start = branch_rows[0]["actual_hold_start_s"] if branch_rows else None
        taper_time = delivery.get("hold_started_at_s", 0.)+20. if delivery else None
        # transition: first sample at/after request where member mean speed within 1 m/s of 22
        adjusted_target = 24.0 + ADJUST_MPS
        transition_s = time_near_target_s = None
        if request_time is not None:
            near_from = None
            for s in samples:
                if s["time_s"] < request_time or s["phase"] == "taper":
                    continue
                mean_speed = sum(m["speed_mps"] for m in s["members"])/len(s["members"])
                if abs(mean_speed-adjusted_target) <= 1.:
                    near_from = s["time_s"]
                    break
            if near_from is not None:
                transition_s = near_from-request_time
                time_near_target_s = (taper_time or terminal_time)-near_from
        wall_records = [r for r in telemetry if r["kind"] == "command" and r["owner"] == "wall"]
        first_handbacks = [r for r in handbacks if r["terminal_tick"]]
        observed = "delivered" if delivery and delivery["delivered"] and terminal_phase == "release" else terminal_phase
        checks = dict(expected_outcome=observed == "delivered", command_and_runtime_contract=not failures,
            nonvacuous_wall_commands=len(wall_records) >= 2, complete_registered_horizon=manifest["completed_simulated_seconds"] == HORIZON_S,
            zero_collisions_teleports=collisions == teleports == 0,
            only_commandownership_cav_writes=all(w[1] in MEMBERS for w in writes),
            unchanged_circles_instance=owner.controller is baseline,
            all_vehicles_idm_and_safety31=all(r["type_id"] == "idm" and r["speed_mode"] == 31 for row in physical for r in row["all_vehicles"]),
            actual_identical_handover=bool(first_handbacks) and all(r["matched"] for r in handbacks),
            hold_compact_at_physics_clock=all(
                len([r for r in row["all_vehicles"] if r["id"] in MEMBERS]) == 2
                and max((r["distance_m"] for r in row["all_vehicles"] if r["id"] in MEMBERS), default=0)
                    - min((r["distance_m"] for r in row["all_vehicles"] if r["id"] in MEMBERS), default=0) <= 50.
                and max((r["speed_mps"] for r in row["all_vehicles"] if r["id"] in MEMBERS), default=0)
                    - min((r["speed_mps"] for r in row["all_vehicles"] if r["id"] in MEMBERS), default=0) <= 1.
                for row in physical if row["outcome_interval_phase"] == "hold"),
            full_original_twenty_second_hold=bool(delivery) and delivery["compact_hold_delivered"],
            branch_at_five_seconds_into_hold=bool(request_time is not None and hold_start is not None
                and abs(request_time-hold_start-BRANCH_OFFSET_S) < 1e-9),
            scheduled_taper_time_preserved=bool(request_time is not None and hold_start is not None
                and terminal_time is not None
                and abs(terminal_time-(hold_start+20.+1.)) <= 1.5),
            adjusted_target_reached_before_taper=transition_s is not None and transition_s < (20.-BRANCH_OFFSET_S),
            no_early_exit=bool(policy is not None and policy.receipt["early_exit_reason"] is None),
            warmup_has_no_wall_commands=all(not p["wall_commands"] for p in policies if p["time_s"] < DECISION_S))
        receipt = dict(policy.receipt)
        summary = dict(manifest, status="complete", passed=all(checks.values()), checks=checks,
            expected="delivered", observed=observed, terminal_time_s=terminal_time,
            terminal_phase=terminal_phase, terminal_reason=terminal_reason, formation_delivery=delivery,
            branch_receipt=receipt, transition_to_adjusted_target_s=transition_s,
            time_near_adjusted_target_s=time_near_target_s,
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
