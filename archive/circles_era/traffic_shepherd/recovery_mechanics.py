"""Single 120-second live recovery chain, using the production policy wrapper."""
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from .budget_guard import BudgetGuard, reserve_and_launch
from .config import load_config
from .damping import CirclesController
from .duration_study import make_branch_spec
from .lifecycle_wall import LifecycleProfile, LifecycleWallPolicy
from .ownership import CommandOwnership
from .paired_experiments import PolicyInput
from .recovery_policy import RecoveryReleasePolicy
from .reporting import CAVReport
from .sensing import SumoLocalSensor
from .sim_clock import ExactClock
from .simulation import close_sumo, start_sumo
from .target_speed_mechanics import build_fixture
from .workflow import write_json

MEMBERS = ("member_a", "member_b")
REPORTERS = (*MEMBERS, "reference_a", "reference_b")
LANES = ("road_0", "road_1")


def run(output, registration, budget):
    import traci
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = load_config()
    manifest = dict(status="preparing", actual_simulated_s=0., mechanics_only=True,
                    registration_sha256=registration["registration_sha256"])
    write_json(output/"manifest.json", manifest)
    history, prior, rows, physics, configured = [], {}, [], [], set()
    allocation = None
    clock = None
    try:
        scene, _ = build_fixture(output/"scenario", registration["sumo_binary"])
        route = ET.parse(scene.parent/"routes.rou.xml")
        for index, vid in enumerate(REPORTERS[2:]):
            ET.SubElement(route.getroot(), "vehicle", dict(id=vid, type="idm", route="route", depart="0",
                departLane=str(index), departPos=str(1150-10*index), departSpeed="27"))
        route.write(scene.parent/"routes.rou.xml", encoding="utf-8", xml_declaration=True)
        api = SimpleNamespace(vehicle=traci.vehicle, lane=traci.lane)
        controller = CirclesController(config)
        owner = CommandOwnership(api, REPORTERS, controller)
        sensor = SumoLocalSensor(api, config, {lane: 0. for lane in LANES})
        job = dict(arm="wall", duration_s=120, decision_time_s=5,
            policy_spec=dict(profile=asdict(LifecycleProfile()), member_ids=list(MEMBERS), required_lanes=list(LANES)))
        base = LifecycleWallPolicy(job, registration)
        policy = base
        wrapped = None
        command_record = []
        allocation, port = reserve_and_launch(budget, target_dir=budget["target_dir"],
            launcher=lambda: start_sumo(registration["sumo_binary"], scene, seed=1, config=config,
                command_record=command_record, extra_cli=["--message-log", str(output/"sumo_messages.log")]),
            campaign_stop_on_failure=True)
        manifest.update(status="running", allocation_id=allocation["id"], command_line=command_record,
                        port=port, sumo_version=list(traci.getVersion()), traci_module=traci.__file__)
        write_json(output/"manifest.json", manifest)
        clock = ExactClock(step_fn=traci.simulationStep, time_fn=lambda: float(traci.simulation.getTime()),
                           dt=.5, start=0., context="oe28_recovery_mechanics")
        for tick in range(120):
            now = clock.check_now(float(traci.simulation.getTime()), context=f"control {tick}")
            live = set(traci.vehicle.getIDList())
            for vid in live-configured:
                traci.vehicle.setLaneChangeMode(vid, 512)
                configured.add(vid)
            if tick:
                frames = {v: sensor.capture(v, now_s=now) for v in sorted(live & set(REPORTERS))}
                reports = []
                for vid, frame in frames.items():
                    prev = prior.get(vid)
                    r = CAVReport(frame, now, prev.owner if prev else "native",
                        prev.network_output_mps2 if prev else None, prev.issued_acceleration_mps2 if prev else None,
                        prev.fallback_reason if prev else None)
                    reports.append(r)
                    history.append((now, r))
                history = [(t, r) for t, r in history if t >= now-10]
                damping = {v: controller.decide(f, now_s=now) for v, f in frames.items()}
                numeric = {v: d.command_acceleration_mps2 for v, d in damping.items() if d.fallback_reason is None}
                native = frozenset(v for v, d in damping.items() if d.fallback_reason is not None)
                context = PolicyInput(now, tuple(reports), tuple(history), numeric, native)
                if wrapped is None and base.ready_at is not None and now == base.ready_at+5:
                    spec = make_branch_spec(context=context, expected_hold_start_s=base.ready_at,
                        previous_wall_commands=base.previous_commands, proposal_sha256=base.proposal.fingerprint,
                        source_manifest_sha256=registration["registration_sha256"])
                    job["policy_spec"]["duration_branch"] = spec
                    wrapped = RecoveryReleasePolicy(job, registration)
                    wrapped.base, wrapped.last_output_phase = base, "hold"
                    policy = wrapped
                result = policy(context)
                records = owner.step(frames, now_s=now, live_ids=live, wall_commands=result.wall_commands)
                prior.update({r.vehicle_id: r for r in records})
                rows.append(dict(time_s=now, policy=result.payload(), reports=[r.payload() for r in reports],
                                 ownership=[r.as_dict() for r in records]))
                if result.phase in {"refused", "abort"}:
                    raise RuntimeError("mechanics protective failure: "+str(result.reason))
                for record in records:
                    if record.wall_request_mps2 is not None and record.owner != "wall":
                        raise RuntimeError("wall ownership request not issued")
            for _ in range(2):
                after = clock.advance()
                live_after = sorted(traci.vehicle.getIDList())
                physics.append(dict(time_s=after, collisions=list(traci.simulation.getCollidingVehiclesIDList()),
                    teleports=list(traci.simulation.getStartingTeleportIDList()),
                    responses=[dict(id=v, speed=traci.vehicle.getSpeed(v), acceleration=traci.vehicle.getAcceleration(v))
                               for v in live_after]))
                if physics[-1]["collisions"] or physics[-1]["teleports"]:
                    raise RuntimeError("mechanics collision or teleport")
            manifest["actual_simulated_s"] = clock.last
        recovery = [r for r in rows if r["policy"].get("recovery_release")]
        active = [r for r in recovery if r["policy"]["wall_commands"]
                  and r["policy"]["recovery_release"]["latest"]["status"] == "active"]
        release = next((r for r in recovery if r["policy"]["phase"] == "release"), None)
        receipt = wrapped.recovery_receipt if wrapped else {}
        checks = dict(clock_exact=clock.calls == 240 and clock.last == 120,
            recovery_exercised=len(active) >= 2,
            upward_recovery=bool(active) and max(r["policy"]["recovery_release"]["latest"]["details"].get("target_speed_mps", 0)
                                               for r in active) > 24.5,
            normal_measured_handback=release is not None and receipt.get("fallback_at_s") is None,
            actual_handback=release is not None and all(x["owner"] != "wall" for x in release["ownership"]),
            zero_collision_teleport=all(not x["collisions"] and not x["teleports"] for x in physics))
        summary = dict(passed=all(checks.values()), checks=checks, actual_simulated_s=clock.last,
                       receipt=receipt, clock_history=list(clock.history), mechanics_only=True)
        write_json(output/"summary.json", summary)
        manifest.update(status="complete", passed=summary["passed"])
        return summary
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        if clock is not None:
            manifest["actual_simulated_s"] = clock.last
        if allocation:
            BudgetGuard(budget["ledger_path"]).mark_progress(allocation["id"], manifest["actual_simulated_s"])
        write_json(output/"manifest.json", manifest)
        for name, values in (("policy", rows), ("physics", physics)):
            (output/(name+".jsonl")).write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in values), encoding="utf-8")
        close_sumo()
