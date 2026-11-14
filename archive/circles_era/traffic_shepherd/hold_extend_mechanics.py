"""One bounded mechanics fixture for the registered HOLD extension (v1).

Verifies in an isolated 120 s fixture, with the identical policy code path as
the corridor (hold_extend_wall.policy_factory), the full chain: entry
assembly, raise to 26 m/s at HOLD+5, extension at HOLD+20 (total 30 s
compact HOLD), unchanged coast exit and identical-CIRCLES handback.
Mechanical verification only; never traffic efficacy.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from .budget_guard import BudgetGuard, reserve_and_launch
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
from .target_adjust_mechanics import ADJUST_MPS as RAISE_MPS
from .target_adjust_mechanics import BRANCH_OFFSET_S, build_fixture
from .target_adjust_wall import VERSION as ADJUST_VERSION
from .hold_extend_wall import VERSION as EXTEND_VERSION, policy_factory
from .verification import hash_file

SCHEMA = "hold_extend_mechanics/1"
MEMBERS = ("member_a", "member_b")
FOLLOWERS = ("follower_a", "follower_b")
LANES = ("road_0", "road_1")
CASE_ID = "hold26_ext10_complete"
ARTIFACTS = ("physical.jsonl", "telemetry.jsonl", "policy.jsonl", "formation_samples.jsonl",
             "longitudinal_writes.json", "disturbances.json", "proposal.json")
HORIZON_S, DECISION_S = 120., 5.
EXTENSION_S = 10.

# Campaign budget enforcement (registered totals unchanged: 4920 s authorized,
# 6360 s historically charged, 1440 s overrun recorded as a deviation in the
# seeded ledger). Mechanics and corridor share one campaign ledger, cumulative
# across versioned output directories.
CAMPAIGN = "hold_extend"
LEDGER_PATH = ROOT / "runs/active/budget/hold_extend_campaign.json"
CEILINGS = dict(
    categories={"mechanics": {"max_allocations": 1, "max_seconds": 120.},
                "corridor": {"max_allocations": 2, "max_seconds": 4800.}},
    total_seconds=4920.)
PREREQUISITE_KIND = "hold_extend_mechanics_passed"


def plan():
    profile = asdict(LifecycleProfile())
    return dict(schema=SCHEMA, policy_version=EXTEND_VERSION, profile=profile,
        cases=[dict(case_id=CASE_ID, expected="delivered")],
        target_adjust=dict(version=ADJUST_VERSION, adjust_mps=RAISE_MPS,
                           branch_offset_after_hold_start_s=BRANCH_OFFSET_S),
        hold_extension=dict(extension_s=EXTENSION_S, branch_after_hold_start_s=20.),
        decision_s=DECISION_S, horizon_s=HORIZON_S, warmup_included_s=DECISION_S,
        maximum_attempts=1, maximum_simulated_seconds=120, automatic_retries=0,
        execution_order=[CASE_ID], simultaneous_runs=False, automatic_successors=False,
        budget=dict(campaign=CAMPAIGN, category="mechanics", amount_s=120.,
                    ledger_path=str(LEDGER_PATH), ceilings=CEILINGS),
        fixture=dict(lanes=list(LANES), length_m=3500., road_speed_mps=33.,
            member_ids=list(MEMBERS), follower_ids=list(FOLLOWERS), member_positions_m=[1000., 990.],
            follower_positions_m=[910., 900.], initial_speed_mps=27., vehicle_length_m=5.,
            all_vehicle_model="IDM", fixed_lane_fixture_mode=512, physics_s=.5,
            vtype_max_speed_mps=27.),
        criteria=dict(wall_acceleration_min_mps2=-.3, wall_acceleration_max_mps2=1.5,
            maximum_command_change_mps2=.15, maximum_entry_to_measured_change_mps2=.15,
            minimum_wall_command_count=2, formed_span_m=50., speed_spread_mps=1.,
            target_tolerance_mps=1., readiness_samples=3, maximum_assembly_s=50.,
            nominal_hold_s=20., adjust_mps=RAISE_MPS, adjust_offset_after_hold_s=BRANCH_OFFSET_S,
            extension_s=EXTENSION_S, extension_offset_after_hold_s=20.,
            total_hold_s=30., require_actual_identical_damping_handover=True,
            require_complete_horizon=True, require_only_cav_writes=True,
            zero_collisions=True, zero_teleports=True),
        stop_rules=["Any failed allocation remains charged; never retry or overwrite it.",
            "Stop a trajectory immediately on a command, ownership, sensing, collision or teleport defect.",
            "A failed fixture stops corridor execution; the verifier is never edited to pass it."],
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
        raise ValueError("hold-extend mechanics case has not run")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads((root/"jobs"/CASE_ID/"manifest.json").read_text(encoding="utf-8"))
    allocation = json.loads((root/"allocations"/f"{CASE_ID}.json").read_text(encoding="utf-8"))
    if allocation != dict(case_id=CASE_ID, allocated_simulated_seconds=HORIZON_S, retries=0,
                          registration_sha256=data["registration_sha256"]):
        raise ValueError("mechanics allocation changed")
    if (manifest.get("registration_sha256") != data["registration_sha256"]
            or manifest.get("status") != "complete" or not manifest.get("passed")
            or manifest.get("completed_simulated_seconds") != HORIZON_S
            or summary.get("completed_simulated_seconds") != HORIZON_S or not summary.get("passed")):
        raise ValueError("hold-extend mechanics case is not a completed passing run")
    for name, expected in manifest["artifact_sha256"].items():
        if hash_file(root/"jobs"/CASE_ID/name) != expected:
            raise ValueError("mechanics artifact changed")
    return dict(registration_path=str(root/"registration.json"),
        registration_sha256=data["registration_sha256"], case_id=CASE_ID,
        checks=summary["checks"], receipt=summary["branch_receipt"],
        proof=dict(manifest_sha256=hash_file(root/"jobs"/CASE_ID/"manifest.json"),
                   actual_simulated_seconds=HORIZON_S, passed=True))


def run_one(registration, case_id=CASE_ID):
    if case_id != CASE_ID:
        raise ValueError("unknown hold-extend mechanics case")
    root, data = read_registration(registration)
    budget = data.get("budget")
    if not isinstance(budget, dict) or budget.get("category") != "mechanics":
        raise ValueError("mechanics registration declares no budget enforcement")
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
        else:
            # A genuinely passing fixture is the corridor's recorded prerequisite.
            BudgetGuard(budget["ledger_path"]).record_prerequisite(
                budget["campaign"], PREREQUISITE_KIND, root/"registration.json",
                note="recorded by the passing mechanics run itself")
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
            member_ids=list(MEMBERS), required_lanes=list(LANES), profile=data["profile"],
            target_adjust_branch=dict(branch_mode="hold_relative", adjust_mps=RAISE_MPS,
                                      offset_after_hold_start_s=BRANCH_OFFSET_S),
            hold_extension=dict(branch_mode="hold_relative", extension_s=EXTENSION_S,
                                branch_after_hold_start_s=20.)))
        policy = policy_factory(job, data)
        # Budget reservation immediately precedes the simulator start: a
        # rejected reservation raises before SUMO is ever invoked, and a
        # crashed launch leaves its reservation charged in the ledger.
        reserve_and_launch(data["budget"], target_dir=root,
            launcher=lambda: start_sumo(data["sumo_binary"], scene, seed=1, config=config))
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
                ext = result.details.get("hold_extension_branch") or {}
                samples.append(dict(time_s=now, phase=result.phase, reason=result.reason, members=measured_members,
                    owners={r.vehicle_id: r.owner for r in records}, wall_command_ids=sorted(result.wall_commands),
                    target_speed_mps=result.details.get("target_speed_mps"),
                    assembly_completed_at_s=result.details.get("assembly_completed_at_s"),
                    ext_requested_at_s=ext.get("request_time_s"),
                    ext_applied=ext.get("applied"),
                    ext_actual_hold_start_s=ext.get("actual_hold_start_s")))
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
                hold_duration_s=30., formed_span_m=50., speed_spread_mps=1., readiness_samples=3)
        receipt = dict(policy.ext_receipt)
        adjust_receipt = dict(policy.receipt)
        hold_start = receipt.get("actual_hold_start_s")
        observed = "delivered" if delivery and delivery["delivered"] and terminal_phase == "release" else terminal_phase
        checks = dict(expected_outcome=observed == "delivered", command_and_runtime_contract=not failures,
            nonvacuous_wall_commands=sum(1 for r in telemetry if r["kind"] == "command" and r["owner"] == "wall") >= 2,
            complete_registered_horizon=manifest["completed_simulated_seconds"] == HORIZON_S,
            zero_collisions_teleports=collisions == teleports == 0,
            only_commandownership_cav_writes=all(w[1] in MEMBERS for w in writes),
            unchanged_circles_instance=owner.controller is baseline,
            all_vehicles_idm_and_safety31=all(r["type_id"] == "idm" and r["speed_mode"] == 31 for row in physical for r in row["all_vehicles"]),
            actual_identical_handover=bool([r for r in handbacks if r["terminal_tick"]]) and all(r["matched"] for r in handbacks),
            hold_compact_at_physics_clock=all(
                len([r for r in row["all_vehicles"] if r["id"] in MEMBERS]) == 2
                and max((r["distance_m"] for r in row["all_vehicles"] if r["id"] in MEMBERS), default=0)
                    - min((r["distance_m"] for r in row["all_vehicles"] if r["id"] in MEMBERS), default=0) <= 50.
                and max((r["speed_mps"] for r in row["all_vehicles"] if r["id"] in MEMBERS), default=0)
                    - min((r["speed_mps"] for r in row["all_vehicles"] if r["id"] in MEMBERS), default=0) <= 1.
                for row in physical if row["outcome_interval_phase"] == "hold"),
            raise_adjustment_delivered=bool(adjust_receipt["adjusted_target_speed_mps"] == 26.),
            extension_requested_at_hold20=bool(receipt["requested"] and hold_start is not None
                and abs(receipt["request_time_s"]-(hold_start+20.)) < 1e-9),
            extension_applied_once=bool(receipt["applied"]) and not receipt["declined_reason"],
            full_thirty_second_hold=bool(delivery) and delivery["compact_hold_delivered"],
            coast_release_after_hold30=bool(delivery and terminal_time is not None
                and abs(terminal_time-(hold_start+30.+1.)) <= 1.5),
            no_early_exit=bool(receipt["early_exit_reason"] is None),
            warmup_has_no_wall_commands=all(not p_["wall_commands"] for p_ in policies if p_["time_s"] < DECISION_S))
        summary = dict(manifest, status="complete", passed=all(checks.values()), checks=checks,
            expected="delivered", observed=observed, terminal_time_s=terminal_time,
            terminal_phase=terminal_phase, terminal_reason=terminal_reason, formation_delivery=delivery,
            branch_receipt=receipt,
            failures=failures, collisions=collisions, teleports=teleports, physical_totals=totals,
            minimum_ttc_s=minimum_ttc, physical_diagnostics=_physical_diagnostics(physical),
            service=service_summary(ledger, manifest["completed_simulated_seconds"]),
            handover_evidence=handbacks, wall_command_count=len([r for r in telemetry if r["kind"] == "command" and r["owner"] == "wall"]),
            interpretation="Actual isolated SUMO execution; no natural-traffic efficacy or universal braking/safety guarantee.")
        _write_json(output/"summary.json", summary)
        manifest.update(status="complete", passed=summary["passed"])
        return summary
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        close_sumo()
        if policy is not None:
            _write_json(output/"ext_receipt_debug.json", dict(policy.ext_receipt))
        _write_json(output/"longitudinal_writes.json", writes)
        _write_json(output/"disturbances.json", [])
        _write_json(output/"proposal.json", proposal_record)
        for name, rows_ in (("physical.jsonl", physical), ("telemetry.jsonl", telemetry),
                            ("policy.jsonl", policies), ("formation_samples.jsonl", samples)):
            with (output/name).open("w", encoding="utf-8") as stream:
                for row in rows_:
                    stream.write(json.dumps(row, allow_nan=False)+"\n")
        manifest["artifact_sha256"] = {name: hash_file(output/name) for name in (*ARTIFACTS, "summary.json",
            "scenario/network.net.xml", "scenario/routes.rou.xml", "scenario/scenario.sumocfg") if (output/name).exists()}
        _write_json(output/"manifest.json", manifest)
