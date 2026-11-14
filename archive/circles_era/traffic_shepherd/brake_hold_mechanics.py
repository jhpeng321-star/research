"""Two fixed SUMO fixture checks for the brake/HOLD factorial endpoints.

Reuses the isolated IDM fixture and evidence format, never patches the frozen
gentle runner or calls a second longitudinal writer. No efficacy claim.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path
import shutil
from types import SimpleNamespace

from .brake_hold_policy import BrakeHoldProfile, VERSION, policy_factory
from .config import ROOT, CONFIG_PATH, load_config
from .damping import CirclesController
from .gentle_mechanics import (MEMBERS, LANES, ARTIFACTS, build_fixture,
    _identity, _write_json, _verify_case, _physical_diagnostics)
from .mechanics import LoggedVehicle
from .metrics import RequestLedger, braking_effort, service_summary, ttc_exposure
from .ownership import CommandOwnership
from .paired_experiments import PolicyInput, digest
from .reporting import CAVReport, RSUReportBuffer
from .sensing import SumoLocalSensor
from .simulation import close_sumo, start_sumo
from .verification import hash_file


SCHEMA = "brake_hold_mechanics/1"
CASE_IDS = ("strong_hold_zero", "strong_hold_forty")
HORIZON_S, DECISION_S = 120., 5.


def plan():
    return dict(schema=SCHEMA, policy_version=VERSION, horizon_s=HORIZON_S, decision_s=DECISION_S,
        warmup_included_s=5., maximum_attempts=2, maximum_simulated_seconds=240, automatic_retries=0,
        automatic_successors=False, execution_order=list(CASE_IDS), simultaneous_runs=False,
        cases=[dict(case_id=case, profile=asdict(BrakeHoldProfile(acceleration_min_mps2=-3., hold_s=hold)))
               for case, hold in zip(CASE_IDS, (0., 40.))],
        fixture=dict(source="gentle_mechanics.build_fixture unchanged", lanes=list(LANES),
            seed=1, seed_scope="same isolated deterministic mechanics fixture; not a corridor development seed",
            length_m=3500., speed_limit_mps=24., member_positions_m=[1000., 990.],
            follower_positions_m=[910., 900.], initial_speed_mps=24., all_vehicle_model="IDM",
            fixed_lane_fixture_mode=512, physics_s=.5, disturbance=None),
        criteria=dict(expected_terminal_phase="release", exact_hold_seconds=[0.,40.],
            measured_target_readiness_samples=3, assembly_deadline_s=50., transition_deadline_s=10.,
            command_min_mps2=-3., command_max_mps2=1.5, command_change_limit_mps2=.15,
            require_actual_command_below_minus_0_3=True, require_actual_native_release=True,
            zero_collisions=True, zero_teleports=True, require_full_horizon=True),
        stop_rules=["No second allocation before the first hash-bound passing result.",
            "Stop on command, ownership, sensing or collision/teleport defect; preserve charged failure.",
            "No retries, fixture replacement or automatic campaign."], mechanical_not_traffic_efficacy=True)


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
    return dict(registration=str(output/"registration.json"), case_ids=list(CASE_IDS),
                maximum_simulated_seconds=240, simulations_started=0)


def read_registration(path, *, require_current=True):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("registration_sha256") != digest({k:v for k,v in data.items() if k != "registration_sha256"}):
        raise ValueError("brake/HOLD mechanics registration changed")
    if any(data.get(k) != v for k,v in plan().items()):
        raise ValueError("brake/HOLD mechanics differs from exact two-case plan")
    if require_current and data["identity"] != _identity(data["sumo_binary"])[0]:
        raise ValueError("source/configuration changed after mechanics registration")
    for name, expected in data["identity"]["active_source_sha256"].items():
        if hash_file(path.parent/"source_snapshot"/name) != expected:
            raise ValueError("mechanics source snapshot changed")
    if hash_file(path.parent/"source_snapshot/adapter.json") != data["identity"]["config_sha256"]:
        raise ValueError("mechanics adapter snapshot changed")
    return path.parent, data


def verify_completed(registration):
    root, data = read_registration(registration)
    if (root/"STOP.json").exists():
        raise ValueError("brake/HOLD mechanics stage stopped")
    if {p.stem for p in (root/"allocations").glob("*.json")} != set(CASE_IDS):
        raise ValueError("both brake/HOLD mechanics allocations must complete")
    cases = [_verify_case(root, data, case) for case in CASE_IDS]
    return dict(schema="verified_brake_hold_mechanics/1", registration_path=str(root/"registration.json"),
        registration_sha256=data["registration_sha256"], registration_file_sha256=hash_file(root/"registration.json"),
        source_sha256=data["identity"]["active_source_sha256"], cases=cases, passed=True,
        actual_simulated_seconds=240, physical_traffic_benefit_established=False)


def command_failures(result, records, previous, frames, profile):
    failures, by_id = [], {r.vehicle_id:r for r in records}
    if len(by_id) != len(records) or not set(by_id) <= set(MEMBERS):
        failures.append("duplicate_or_foreign_command_record")
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


def lifecycle_checks(samples, profile):
    """Independent measured scoring, including valid assembly-to-taper at H0."""
    streak, first_ready, first_after_assembly = 0, None, None
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
                first_after_assembly = row["time_s"]
                integrity &= (first_ready == row["time_s"] and row["assembly_completed_at_s"] == row["time_s"]
                    and row["phase"] in ({"taper", "release"} if profile.hold_s == 0 else {"hold"}))
        if row["phase"] == "hold":
            hold_rows.append(row)
            integrity &= compact
        elif row["phase"] == "taper":
            transition_rows.append(row)
        elif row["phase"] in {"abort", "refused"}:
            integrity = False
    hold_times = [r["time_s"] for r in hold_rows]
    expected_hold = ([] if first_after_assembly is None else
                     [first_after_assembly+i for i in range(int(profile.hold_s))])
    terminal = samples[-1] if samples else {}
    terminal_time = terminal.get("time_s")
    release_ok = (terminal.get("phase") == "release" and first_after_assembly is not None
        and first_after_assembly+profile.hold_s <= terminal_time <= first_after_assembly+profile.hold_s+profile.taper_s
        and not terminal.get("wall_command_ids")
        and set(terminal.get("owners", {})) == set(MEMBERS)
        and all(owner in {"native", "damping"} for owner in terminal["owners"].values()))
    checks = dict(measured_lifecycle_integrity=bool(integrity),
        three_target_settled_readiness_samples=first_ready is not None,
        assembly_before_original_deadline=first_ready is not None and first_ready-DECISION_S <= profile.assembly_deadline_s,
        exact_registered_hold_duration=hold_times == expected_hold and first_after_assembly is not None,
        terminal_release_within_transition_budget=bool(release_ok))
    return dict(checks=checks, first_measured_ready_s=first_ready, hold_seconds=len(hold_rows),
                taper_command_seconds=len(transition_rows), terminal_time_s=terminal_time)


def run_one(registration, case_id):
    root, data = read_registration(registration)
    if case_id not in CASE_IDS or (root/"STOP.json").exists():
        raise ValueError("unknown case or stopped brake/HOLD mechanics")
    for previous in CASE_IDS[:CASE_IDS.index(case_id)]:
        _verify_case(root,data,previous)
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
    profile = BrakeHoldProfile(**case["profile"])
    try:
        scene,requests = build_fixture(output/"scenario",data["sumo_binary"])
        ledger = RequestLedger(requests)
        baseline = CirclesController(config)
        api = SimpleNamespace(vehicle=LoggedVehicle(traci.vehicle,writes),lane=traci.lane)
        owner = CommandOwnership(api,MEMBERS,baseline)
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
            frames = {v:sensor.capture(v,now_s=now) for v in sorted(live & set(MEMBERS))}
            physical.append(dict(time_s=now,outcome_interval_phase=last_phase,all_vehicles=vehicles,
                acquired_member_frames={v:asdict(f) for v,f in frames.items()}))
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
            expected = {v:baseline.decide(f,now_s=now) for v,f in frames.items()}
            result = policy(PolicyInput(now,tuple(buffer.reports(now_s=now)),tuple(history),
                {v:d.command_acceleration_mps2 for v,d in expected.items() if d.fallback_reason is None},
                frozenset(v for v,d in expected.items() if d.fallback_reason is not None)))
            policies.append(dict(time_s=now,**result.payload()))
            if policy.proposal is not None and proposal_record is None:
                proposal_record = asdict(policy.proposal)
            count = len(writes)
            records = owner.step(frames,now_s=now,live_ids=live,wall_commands=result.wall_commands)
            failures.extend(command_failures(result,records,previous_wall,frames,profile))
            if len(writes)-count != sum(r.issued_acceleration_mps2 is not None or r.released_to_native for r in records):
                failures.append("competing_or_missing_command_writer")
            for record in records:
                telemetry.append(dict(kind="command",**record.as_dict()))
                telemetry.append(dict(kind="identical_damping_reference",vehicle_id=record.vehicle_id,
                    timestamp_s=now,**expected[record.vehicle_id].as_dict()))
            previous_records = {r.vehicle_id:r for r in records}
            previous_wall = {r.vehicle_id:r.issued_acceleration_mps2 for r in records if r.owner == "wall"}
            if now >= DECISION_S and terminal_time is None:
                samples.append(dict(time_s=now,phase=result.phase,reason=result.reason,
                    members=[{k:r[k] for k in ("id","lane","distance_m","speed_mps")} for r in vehicles if r["id"] in MEMBERS],
                    owners={r.vehicle_id:r.owner for r in records},wall_command_ids=sorted(result.wall_commands),
                    target_speed_mps=result.details.get("target_speed_mps"),
                    assembly_completed_at_s=result.details.get("assembly_completed_at_s")))
            if result.phase in {"release","abort","refused"}:
                first = terminal_time is None
                terminal_time = now if first else terminal_time
                for r in records:
                    d = expected[r.vehicle_id]
                    matched = (r.owner == ("native" if d.fallback_reason else "damping")
                        and r.issued_acceleration_mps2 == d.command_acceleration_mps2
                        and r.issued_target_speed_mps == d.target_speed_mps and r.fallback_reason == d.fallback_reason)
                    handbacks.append(dict(time_s=now,vehicle_id=r.vehicle_id,matched=matched,terminal_tick=first,
                        actual_override_release=r.released_to_native,baseline_native=d.fallback_reason is not None))
                    if not matched:
                        failures.append("handover_not_identical_damping")
            last_phase = result.phase
            if failures:
                break
        owner.release_all(live_ids=traci.vehicle.getIDList())
        delivery = lifecycle_checks(samples,profile)
        wall = [r for r in telemetry if r["kind"] == "command" and r["owner"] == "wall"]
        first_handbacks = [r for r in handbacks if r["terminal_tick"]]
        hold_rows = [r for r in physical if r["outcome_interval_phase"] == "hold"]
        compact_hold = all(len(m := [v for v in row["all_vehicles"] if v["id"] in MEMBERS]) == 2
            and {v["lane"] for v in m} == set(LANES)
            and max(v["distance_m"] for v in m)-min(v["distance_m"] for v in m) <= 50.
            and max(v["speed_mps"] for v in m)-min(v["speed_mps"] for v in m) <= 1. for row in hold_rows)
        checks = dict(delivery["checks"], command_and_runtime_contract=not failures,
            full_registered_horizon=manifest["completed_simulated_seconds"] == HORIZON_S,
            zero_collisions_teleports=collisions == teleports == 0,
            only_commandownership_cav_writes=all(w[1] in MEMBERS for w in writes),
            unchanged_circles_instance=owner.controller is baseline,
            all_idm_and_safety31=all(v["type_id"] == "idm" and v["speed_mode"] == 31 for r in physical for v in r["all_vehicles"]),
            actual_stronger_than_gentle_command=any(r["issued_acceleration_mps2"] < -.3-1e-8 for r in wall),
            nonvacuous_wall_commands=len(wall) >= 2, hold_compact_at_halfsecond_clock=compact_hold,
            exact_hold_at_halfsecond_clock=len(hold_rows)*.5 == profile.hold_s,
            actual_identical_handover=bool(first_handbacks) and all(r["matched"] for r in handbacks),
            actual_native_override_release=bool(first_handbacks) and all(not r["baseline_native"] or r["actual_override_release"] for r in first_handbacks),
            warmup_has_no_wall_commands=all(not r["wall_commands"] and r["phase"] == "damping" for r in policies if r["time_s"] < DECISION_S))
        summary = dict(manifest,status="complete",passed=all(checks.values()),checks=checks,profile=asdict(profile),
            observed_lifecycle=delivery,failures=failures,collisions=collisions,teleports=teleports,
            physical_totals=totals,physical_diagnostics=_physical_diagnostics(physical),
            service=service_summary(ledger,manifest["completed_simulated_seconds"]),
            handover_evidence=handbacks,wall_command_count=len(wall),
            interpretation="Two synthetic-road execution checks, not factorial efficacy or population evidence.")
        _write_json(output/"summary.json",summary)
        manifest.update(status="complete",passed=summary["passed"])
        return summary
    except Exception as exc:
        manifest.update(status="failed",error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        close_sumo()
        _write_json(output/"longitudinal_writes.json",writes)
        _write_json(output/"disturbances.json",[])
        _write_json(output/"proposal.json",proposal_record)
        for name,rows in (("physical.jsonl",physical),("telemetry.jsonl",telemetry),
                          ("policy.jsonl",policies),("formation_samples.jsonl",samples)):
            with (output/name).open("w",encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row,allow_nan=False)+"\n")
        manifest["artifact_sha256"] = {name:hash_file(output/name) for name in (*ARTIFACTS,"summary.json",
            "scenario/network.net.xml","scenario/routes.rou.xml","scenario/scenario.sumocfg") if (output/name).exists()}
        _write_json(output/"manifest.json",manifest)
