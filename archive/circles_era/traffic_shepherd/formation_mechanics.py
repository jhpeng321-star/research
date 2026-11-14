"""Registered formation fixture repair; no efficacy or predictor claims.

This module is staged until its source and exact case budget are registered.
Fixture disturbances are explicitly logged, and never become policy access to
the simulator. CommandOwnership remains the sole longitudinal command writer.
"""
import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from traffic_shepherd.config import load_config
from traffic_shepherd.damping import CirclesController
from traffic_shepherd.escalation_evidence import formation_delivery
from traffic_shepherd.formation import (
    FormationController, FormationProfile, assess_formation, authorize_experiment,
    make_proposal,
)
from traffic_shepherd.mechanics import LoggedVehicle
from traffic_shepherd.metrics import RequestLedger, braking_effort, service_summary, ttc_exposure
from traffic_shepherd.ownership import CommandOwnership
from traffic_shepherd.reporting import CAVReport, RSUReportBuffer
from traffic_shepherd.scenarios import TripRequest
from traffic_shepherd.sensing import SumoLocalSensor
from traffic_shepherd.simulation import close_sumo, start_sumo
from traffic_shepherd.verification import hash_file


ROOT = Path(__file__).resolve().parents[1]
MEMBERS = ("member_a", "member_b")
LANES = ("road_0", "road_1")
HORIZON_S = 140.
DECISION_S = 1.


@dataclass(frozen=True)
class MechanicsCase:
    case_id: str
    initial_span_m: float
    speed_difference_mps: float = 0.
    nearest_boundary_m: float = 2500.
    initial_member_mean_speed_mps: float = 15.
    disturbance: str = "none"
    disturbance_time_s: float | None = None
    expected: str = "delivered"
    road_speed_limit_mps: float = 30.
    reference_leaders: bool = False


def case_catalog():
    """One bounded integration check: current-member loss, lane loss and taper."""
    cases = [
        MechanicsCase("member_lane_loss",75.,disturbance="member_lane_loss",disturbance_time_s=2.,expected="aborted"),
        MechanicsCase("member_loss",75.,disturbance="member_loss",disturbance_time_s=8.,expected="aborted"),
        MechanicsCase("numeric_taper",10.,road_speed_limit_mps=15.,reference_leaders=True),
    ]
    assert len(cases) == 3
    return cases


def plan():
    return dict(schema="formation_mechanics_stage/3", case_definitions=[asdict(c) for c in case_catalog()],
        profile=asdict(FormationProfile()), decision_s=DECISION_S, horizon_s=HORIZON_S,
        maximum_simulated_seconds=len(case_catalog())*HORIZON_S, automatic_retries=0,
        model_sha256=load_config().model_sha256,
        purpose="One bounded integration repair for absent current local execution evidence; exercise actual safe member lane loss and previously unrun numeric taper.",
        network=dict(lanes=2,length_m=3500.,speed_limit_mps="per registered case: 30 except numeric_taper15",release_boundary="road_end"),
        repair=dict(formation_law_changed=False,failed_v1_v2_preserved=True,
            execution_guard="Current numerical-or-native local baseline reference required for every member; aged RSU packet cannot authorize a missing local executor",
            member_lane_loss="Same actual changeLane512 requested at2s with approximately68m bumpergap rather than8s/34m; no longitudinal reposition or safety-mode relaxation",
            prior_lane_loss_limit="The8s request was not delivered; safety blocking is consistent with acquired geometry, but v2 did not record internal lanechange-state bits",
            numeric_taper="Two native13 m/s leaders initially50 m bumper gap ahead on15 m/s road",
            unsafe_dynamic_cut_in_ttc="not physically validated by the safe cut-in fixture"),
        observation="100 m ideal same-lane sensing, zero nominal communication delay; only the registered report-loss case drops packets",
        criteria=dict(readiness_samples=3,formed_span_m=50.,formed_speed_spread_mps=1.,
            assembly_deadline_s=50.,compact_hold_s=60.,zero_collisions=True,zero_teleports=True,
            hdv_longitudinal_writes=0,expected_case_outcome_required=True),
        stopping_rules=["Stop physical stage on collision/teleport or ownership/model/information boundary defect.",
            "Preserve a failed positive case and diagnose before one newly registered focused repair; no hidden rerun.",
            "Expected initial refusals and scripted aborts are mechanism checks, not naturally occurring failure rates."],
        mechanical_not_traffic_efficacy=True)


def validate_registration(registration):
    required = plan()
    for key in required:
        if registration.get(key) != required[key]:
            raise ValueError(f"registration differs from frozen mechanics plan: {key}")
    if not registration.get("registration_id"):
        raise ValueError("explicit mechanics registration ID required before running")
    expected_sources = registration.get("source_sha256", {})
    for name in ("formation.py", "formation_mechanics.py"):
        path = ROOT / "traffic_shepherd" / name
        if expected_sources.get(name) != hash_file(path):
            raise ValueError(f"registered source changed: {name}")
    if registration.get("adapter_sha256") != hash_file(ROOT / "baselines/circles/adapter.json"):
        raise ValueError("registered adopted adapter changed")


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def build_fixture(output, case, sumo_binary):
    """Isolated synthetic assets only. No shared network asset is changed."""
    output.mkdir(parents=True, exist_ok=False)
    (output/"nodes.xml").write_text('<nodes><node id="start" x="0" y="0"/><node id="end" x="3500" y="0"/></nodes>')
    (output/"edges.xml").write_text(f'<edges><edge id="road" from="start" to="end" numLanes="2" speed="{case.road_speed_limit_mps}"/></edges>')
    binary = Path(sumo_binary)
    converter = binary.parent / ("netconvert.exe" if binary.suffix == ".exe" else "netconvert")
    converted = subprocess.run([str(converter),"--node-files",str(output/"nodes.xml"),
        "--edge-files",str(output/"edges.xml"),"--output-file",str(output/"network.net.xml")],
        capture_output=True,text=True)
    (output/"netconvert.log").write_text(converted.stdout+converted.stderr)
    converted.check_returncode()
    routes = ET.Element("routes")
    types = dict(idm=30.,hdv_front=18.,hdv_rear=10.,hdv_static=30.,hdv_cut_in=20.,hdv_reference=13.)
    for name, maximum in types.items():
        ET.SubElement(routes,"vType",dict(id=name,carFollowModel="IDM",accel="2.6",decel="4.5",
            emergencyDecel="9",tau="1.1",minGap="2.5",sigma="0",speedFactor="1",speedDev="0",length="5",
            maxSpeed=str(maximum),lcStrategic="0",lcCooperative="0",lcSpeedGain="0",lcKeepRight="0"))
    ET.SubElement(routes,"route",dict(id="route",edges="road"))
    leading_position = 3500.-case.nearest_boundary_m
    definitions = [dict(id=MEMBERS[i],type="idm",lane=i,
        position=leading_position-i*case.initial_span_m,
        speed=case.initial_member_mean_speed_mps+(1 if i == 0 else -1)*case.speed_difference_mps/2) for i in range(2)]
    disturbance = case.disturbance
    if disturbance != "none" and disturbance not in {"member_lane_loss","member_loss","report_loss"}:
        driver = {"front_safe":"hdv_front","rear_safe":"hdv_rear","front_unsafe":"hdv_static",
                  "rear_unsafe":"hdv_static","cut_in_safe":"hdv_cut_in"}[disturbance]
        offset = {"front_safe":45.,"rear_safe":-45.,"front_unsafe":8.5,
                  "rear_unsafe":-8.5,"cut_in_safe":10.}[disturbance]
        definitions.append(dict(id="disturbance_hdv",type=driver,lane=1 if disturbance == "cut_in_safe" else 0,
            position=leading_position+offset,speed=0. if "unsafe" in disturbance else types[driver]))
    if case.reference_leaders:
        definitions += [dict(id=f"reference_hdv_{i}",type="hdv_reference",lane=i,
            position=leading_position-i*case.initial_span_m+55.,speed=13.) for i in range(2)]
    for vehicle in sorted(definitions, key=lambda x:(x["lane"],-x["position"])):
        ET.SubElement(routes,"vehicle",dict(id=vehicle["id"],type=vehicle["type"],route="route",depart="0",
            departLane=str(vehicle["lane"]),departPos=str(vehicle["position"]),departSpeed=str(vehicle["speed"])))
    ET.indent(routes)
    ET.ElementTree(routes).write(output/"routes.rou.xml",encoding="utf-8",xml_declaration=True)
    (output/"scenario.sumocfg").write_text('<configuration><input><net-file value="network.net.xml"/>'
        '<route-files value="routes.rou.xml"/></input><processing><time-to-teleport value="-1"/>'
        '<collision.action value="warn"/></processing></configuration>')
    requests = [TripRequest(v["id"],0.,"route",v["type"],v["id"] in MEMBERS) for v in definitions]
    return output/"scenario.sumocfg", requests


def apply_fixture_disturbance(api, case, *, now_s):
    """Scripted intervention used only by the experiment, never by the policy."""
    if case.disturbance_time_s != now_s or case.disturbance == "report_loss":
        return None
    event = dict(timestamp_s=now_s,source="registered_fixture_disturbance_not_controller",kind=case.disturbance)
    if case.disturbance == "cut_in_safe":
        api.vehicle.changeLane("disturbance_hdv",0,1.)
        event.update(vehicle="disturbance_hdv",requested_lane="road_0",request_duration_s=1.)
    elif case.disturbance == "member_lane_loss":
        api.vehicle.changeLane(MEMBERS[1],0,1.)
        event.update(vehicle=MEMBERS[1],requested_lane="road_0",request_duration_s=1.)
    elif case.disturbance == "member_loss":
        api.vehicle.remove(MEMBERS[1])
        event["vehicle"] = MEMBERS[1]
    return event


def _observed_members(api, live):
    rows = []
    for member in MEMBERS:
        if member in live:
            lane = api.vehicle.getLaneID(member)
            position = float(api.vehicle.getLanePosition(member))
            rows.append(dict(id=member,lane=lane,position_m=position,
                distance_m=float(api.lane.getLength(lane))-position,
                speed_mps=float(api.vehicle.getSpeed(member)),
                acceleration_mps2=float(api.vehicle.getAcceleration(member))))
    return rows


def lifecycle_window(samples):
    """Score through first terminal handback; retain later physics separately.

    Missing members on an abort row stay missing. There are no invented states,
    padded HOLD intervals or post-exit ownership checks masquerading as delivery.
    """
    for index, sample in enumerate(samples):
        if sample["phase"] in {"abort","release"}:
            return samples[:index+1]
    return list(samples)


def run_case(registration_path, case_id, *, sumo_binary="C:/Program Files (x86)/Eclipse/Sumo/bin/sumo.exe"):
    """Charge one fixed allocation before any launch; never resume or auto-retry."""
    import traci
    registration_path = Path(registration_path).resolve()
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    validate_registration(registration)
    cases = {c.case_id:c for c in case_catalog()}
    if case_id not in cases:
        raise ValueError("case absent from fixed registered mechanics allocation")
    case = cases[case_id]
    allocations = registration_path.parent/"allocations"
    allocations.mkdir(parents=True,exist_ok=True)
    with (allocations/f"{case_id}.json").open("x",encoding="utf-8") as charge:
        json.dump(dict(case_id=case_id,charged_simulated_seconds=HORIZON_S,retries=0,
                       registration_sha256=hash_file(registration_path)),charge,indent=2)
    output = registration_path.parent/"jobs"/case_id
    output.mkdir(parents=True,exist_ok=False)
    config = load_config()
    manifest = dict(schema="formation_mechanics_run/1",case=asdict(case),status="allocated",
        model_sha256=config.model_sha256,registration_sha256=hash_file(registration_path),
        horizon_s=HORIZON_S,mechanics_only=True,seed=1,
        seed_scope="isolated deterministic fixture; no corridor or prospective seed",
        exact_controller_instance_retained=True,completed_simulated_seconds=0.)
    _write_json(output/"manifest.json",manifest)
    commands, disturbances, physics, samples, telemetry = [], [], [], [], []
    collisions = teleports = 0
    totals = dict(braking_vehicle_seconds=0.,acceleration_energy=0.,ttc_below_3_vehicle_seconds=0.)
    minimum_ttc = None
    proposal = assessment = controller = None
    initial_refusal = None
    terminal_reason = None
    observer_errors = []
    handback_checks = []
    scripted_removals = set()
    try:
        scene, requests = build_fixture(output/"scenario",case,sumo_binary)
        ledger = RequestLedger(requests)
        baseline = CirclesController(config)
        api = SimpleNamespace(vehicle=LoggedVehicle(traci.vehicle,commands),lane=traci.lane)
        owner = CommandOwnership(api,MEMBERS,baseline)
        sensor = SumoLocalSensor(api,config,{lane:0. for lane in LANES})
        buffer = RSUReportBuffer(max_age_s=1.,history_s=20.)
        previous = {}
        start_sumo(sumo_binary,scene,seed=1,config=config)
        configured = set()
        for _ in range(int(HORIZON_S/config.physics_dt_s)):
            traci.simulationStep()
            now = float(traci.simulation.getTime())
            manifest["completed_simulated_seconds"] = now
            live = set(traci.vehicle.getIDList())
            for vehicle in live-configured:
                # A fixed-lane experimental road isolates longitudinal formation.
                # This is not an online wall lane-change policy.
                traci.vehicle.setLaneChangeMode(vehicle,512)
                configured.add(vehicle)
            event = apply_fixture_disturbance(traci,case,now_s=now)
            if event is not None:
                disturbances.append(event)
                if event["kind"] == "member_loss":
                    scripted_removals.add(event["vehicle"])
                live = set(traci.vehicle.getIDList())
            ledger.advance(now,list(traci.simulation.getDepartedIDList()),
                [v for v in traci.simulation.getArrivedIDList() if v not in scripted_removals])
            collisions += int(traci.simulation.getCollidingVehiclesNumber())
            teleports += int(traci.simulation.getStartingTeleportNumber())
            all_accels = [float(traci.vehicle.getAcceleration(v)) for v in sorted(live)]
            effort = braking_effort(all_accels,config.physics_dt_s)
            for key,value in effort.items():
                totals[key] += value
            exposure, ttc = ttc_exposure(traci,live,config.physics_dt_s)
            totals["ttc_below_3_vehicle_seconds"] += exposure
            if ttc is not None:
                minimum_ttc = ttc if minimum_ttc is None else min(minimum_ttc,ttc)
            measured = _observed_members(traci,live)
            sensed_halfstep = {}
            for member in sorted(live & set(MEMBERS)):
                try:
                    sensed_halfstep[member] = asdict(sensor.capture(member,now_s=now))
                except ValueError as exc:
                    observer_errors.append(dict(time_s=now,vehicle_id=member,error=str(exc)))
            physics.append(dict(time_s=now,members=measured,
                acquired_member_frames=sensed_halfstep,
                lane_change_diagnostics={v:dict(mode=traci.vehicle.getLaneChangeMode(v),
                    state_right=traci.vehicle.getLaneChangeState(v,-1)) for v in sorted(live)},
                all_vehicles=[dict(id=v,lane=traci.vehicle.getLaneID(v),position_m=traci.vehicle.getLanePosition(v),
                    speed_mps=traci.vehicle.getSpeed(v),acceleration_mps2=traci.vehicle.getAcceleration(v)) for v in sorted(live)],
                speed_modes={v:traci.vehicle.getSpeedMode(v) for v in sorted(live)},
                vehicle_types={v:traci.vehicle.getTypeID(v) for v in sorted(live)}))
            for member in MEMBERS:
                if member in live and member in previous:
                    response = owner.record_response(member,timestamp_s=now,
                        speed_mps=traci.vehicle.getSpeed(member),acceleration_mps2=traci.vehicle.getAcceleration(member))
                    telemetry.append(dict(kind="measured_response",**response.as_dict()))
            if not now.is_integer():
                if collisions or teleports:
                    manifest["safety_stop"] = dict(time_s=now,collisions=collisions,teleports=teleports)
                    break
                continue
            frames = {}
            for member in sorted(live & set(MEMBERS)):
                try:
                    frames[member] = sensor.capture(member,now_s=now)
                except ValueError as exc:
                    observer_errors.append(dict(time_s=now,vehicle_id=member,error=str(exc)))
            for member,frame in frames.items():
                if case.disturbance == "report_loss" and now >= case.disturbance_time_s and member == MEMBERS[1]:
                    if now == case.disturbance_time_s:
                        disturbances.append(dict(timestamp_s=now,kind="report_loss",vehicle=member,
                            source="registered_packet_drop_not_physics_change"))
                    continue
                packet = CAVReport(frame,now,"damping" if member not in previous else previous[member].owner)
                buffer.receive(packet,received_at_s=now)
                telemetry.append(dict(kind="received_cav_report",received_at_s=now,**packet.payload()))
            reports = buffer.reports(now_s=now)
            if now == DECISION_S:
                if set(frames) != set(MEMBERS):
                    raise RuntimeError("registered members did not insert before the exact decision tick")
                proposal = make_proposal(reports,now_s=now,required_lanes=LANES,member_ids=MEMBERS)
                assessment = assess_formation(proposal,reports,now_s=now)
                _write_json(output/"proposal.json",asdict(proposal))
                _write_json(output/"initial_assessment.json",asdict(assessment))
                if assessment.approved:
                    authorization = authorize_experiment(proposal,assessment,
                        registration_id=registration["registration_id"])
                    controller = FormationController(proposal,authorization)
                else:
                    initial_refusal = assessment.reason
            baseline_decisions = {member:baseline.decide(frame,now_s=now) for member,frame in frames.items()}
            damping_commands = {v:d.command_acceleration_mps2 for v,d in baseline_decisions.items()
                                if d.fallback_reason is None}
            for member,decision in baseline_decisions.items():
                telemetry.append(dict(kind="identical_damping_reference",vehicle_id=member,
                                      timestamp_s=now,**decision.as_dict()))
            phase, wall = "damping", {}
            if controller is not None:
                native_references = tuple(v for v,d in baseline_decisions.items() if d.fallback_reason is not None)
                step = controller.step(reports,now_s=now,damping_commands=damping_commands,
                                       damping_native_ids=native_references)
                phase, wall = step.phase, step.commands
                telemetry.append(dict(kind="formation_step",time_s=now,**step.as_dict()))
                if phase in {"abort","release"} and terminal_reason is None:
                    terminal_reason = step.reason
            records = owner.step(frames,now_s=now,live_ids=live,wall_commands=wall)
            previous = {r.vehicle_id:r for r in records}
            telemetry.extend(dict(kind="issued_command",**r.as_dict()) for r in records)
            samples.append(dict(time_s=now,phase=phase,members=measured,
                reason=step.reason if controller else initial_refusal,
                owners={r.vehicle_id:r.owner for r in records},wall_command_ids=sorted(wall)))
            if phase in {"abort","release"}:
                handback_checks.append(all(r.owner == ("native" if baseline_decisions[r.vehicle_id].fallback_reason else "damping")
                    and r.issued_acceleration_mps2 == baseline_decisions[r.vehicle_id].command_acceleration_mps2
                    for r in records))
            if collisions or teleports:
                manifest["safety_stop"] = dict(time_s=now,collisions=collisions,teleports=teleports)
                break
        owner.release_all(live_ids=traci.vehicle.getIDList())
        delivery = formation_delivery(lifecycle_window(samples),required_lanes=LANES,
            assembly_deadline_s=50.,hold_duration_s=60.,formed_span_m=50.,
            speed_spread_mps=1.,readiness_samples=3) if controller else None
        observed = "refused" if initial_refusal else "delivered" if delivery and delivery["delivered"] else "aborted"
        requested_changes = [event for event in disturbances if "requested_lane" in event]
        lane_changes_observed = all(any(vehicle["id"] == event["vehicle"] and vehicle["lane"] == event["requested_lane"]
            for row in physics if event["timestamp_s"] < row["time_s"] <= event["timestamp_s"]+2.
            for vehicle in row["all_vehicles"]) for event in requested_changes)
        taper_samples = [row for row in samples if row["phase"] == "taper"]
        checks = dict(expected_outcome=observed == case.expected,
            zero_collisions_teleports=collisions == 0 and teleports == 0,
            only_commandownership_cav_writes=all(row[1] in MEMBERS for row in commands),
            speedmode31=all(set(row["speed_modes"].values()) <= {31} for row in physics),
            hdvs_idm=all(name.startswith("idm") or name.startswith("hdv_") for row in physics for name in row["vehicle_types"].values()),
            exact_controller_instance_retained=owner.controller is baseline,
            actual_handback_matches_identical_damping=all(handback_checks),
            delivered_release_ownership_verified=not (delivery and delivery["delivered"]) or delivery["release_ownership_verified"],
            scripted_lane_changes_physically_observed=lane_changes_observed,
            numeric_taper_10_seconds=not case.reference_leaders or len(taper_samples) == 10,
            complete_registered_horizon=manifest["completed_simulated_seconds"] == HORIZON_S,
            no_sensor_overlap_errors=not observer_errors)
        summary = dict(manifest,status="complete",case_outcome=observed,expected=case.expected,
            initial_refusal=initial_refusal,terminal_reason=terminal_reason,
            formation_delivery=delivery,initial_measured_span_m=assessment.initial_or_current_span_m,
            initial_measured_speed_spread_mps=assessment.speed_spread_mps,
            collisions=collisions,teleports=teleports,totals=totals,minimum_closing_ttc_s=minimum_ttc,
            service=service_summary(ledger,manifest["completed_simulated_seconds"]),
            command_owners=dict(Counter(r["owner"] for r in telemetry if r["kind"] == "issued_command")),
            disturbances=disturbances,sensor_errors=observer_errors,checks=checks,passed=all(checks.values()),
            scripted_removals_preserved_as_unfinished=sorted(scripted_removals),
            interpretation="Controlled synthetic-road SUMO mechanics, not natural mixed-traffic efficacy or policy generalization.")
        manifest["status"] = "complete"
        _write_json(output/"summary.json",summary)
        return summary
    except Exception as exc:
        manifest.update(status="failed",error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        close_sumo()
        _write_json(output/"manifest.json",manifest)
        _write_json(output/"longitudinal_writes.json",commands)
        _write_json(output/"disturbances.json",disturbances)
        for name,rows in (("physical.jsonl",physics),("telemetry.jsonl",telemetry),("formation_samples.jsonl",samples)):
            with (output/name).open("w",encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row,allow_nan=False)+"\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    sub.add_parser("plan")
    run = sub.add_parser("run")
    run.add_argument("--registration",required=True)
    run.add_argument("--case-id",required=True)
    run.add_argument("--sumo-binary",default="C:/Program Files (x86)/Eclipse/Sumo/bin/sumo.exe")
    args = parser.parse_args(argv)
    if args.command == "plan":
        print(json.dumps(plan(),indent=2))
        return 0
    result = run_case(args.registration,args.case_id,sumo_binary=args.sumo_binary)
    print(json.dumps({k:result[k] for k in ("case_outcome","expected","passed","checks","initial_refusal","terminal_reason")},indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
