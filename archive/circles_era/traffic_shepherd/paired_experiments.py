"""Registered paired CAV-report experiments with one longitudinal command owner.

Policy callbacks receive only timestamped CAV packets. Simulator measurements,
scenario designation and outcome accounting live in separate offline channels.
Experimental assignment is not a learned incremental-benefit approval.
"""
import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from math import isfinite
from pathlib import Path
import shutil
import time
from types import MappingProxyType

from . import budget_guard
from .config import CONFIG_PATH, ROOT, load_config
from .damping import CirclesController
from .escalation_evidence import BenefitCriteria, benefit_label, paired_effect
from .escalation_observations import OfflineTrafficObserver, report_from_payload
from .interaction_diagnostics import (FILENAME as INTERACTION_FILENAME, SCHEMA as INTERACTION_SCHEMA,
    OfflineInteractionObserver, file_sha256, offline_read_cache)
from .metrics import RequestLedger, braking_effort, service_summary, ttc_exposure
from .ownership import CommandOwnership
from .reporting import CAVReport, RSUReportBuffer
from .scenarios import build_scenario, observation_map, scenario_identity
from .seeds import exclusion_provenance, require_development_case
from .sensor_batch import capture_batch
from .sensing import SumoLocalSensor
from .simulation import close_sumo, start_sumo
from .workflow import sha, write_json


ARMS = frozenset({"damping", "wall", "sham", "assembly_only"})
PHASES = frozenset({"damping", "assembly", "hold", "taper", "release", "abort", "refused"})


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class PolicyInput:
    """All callback inputs are acquired data; no simulator or case metadata."""
    now_s: float
    reports: tuple
    history: tuple  # (receipt time, CAVReport), retained for a declared window
    damping_commands: object
    damping_native_ids: frozenset = frozenset()

    def __post_init__(self):
        if any(not isinstance(r, CAVReport) or r.sent_at_s > self.now_s for r in self.reports):
            raise ValueError("only already received CAV reports may enter policy")
        if any(receipt > self.now_s or not isinstance(r, CAVReport)
               or receipt < r.sent_at_s for receipt, r in self.history):
            raise ValueError("invalid or future history packet")
        object.__setattr__(self, "damping_commands", MappingProxyType(dict(self.damping_commands)))
        native = frozenset(self.damping_native_ids)
        current_reporters = {r.frame.vehicle_id for r in self.reports if r.frame.timestamp_s == self.now_s}
        if not native <= current_reporters or native & set(self.damping_commands):
            raise ValueError("native references require current reports and no numeric reference")
        object.__setattr__(self, "damping_native_ids", native)


@dataclass(frozen=True)
class PolicyOutput:
    wall_commands: object = field(default_factory=dict)
    phase: str = "damping"
    reason: str | None = None
    member_ids: tuple = ()
    proposal_sha256: str | None = None
    evidence_sha256: str | None = None
    details: object = field(default_factory=dict)

    def __post_init__(self):
        if self.phase not in PHASES or len(self.member_ids) != len(set(self.member_ids)):
            raise ValueError("invalid single-wall lifecycle output")
        commands = dict(self.wall_commands)
        if not set(commands) <= set(self.member_ids) or any(not isfinite(a) for a in commands.values()):
            raise ValueError("commands must be finite and confined to declared members")
        if commands and self.phase not in {"assembly", "hold", "taper"}:
            raise ValueError("release, abort and refusal cannot retain wall commands")
        object.__setattr__(self, "wall_commands", MappingProxyType(commands))
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))

    def payload(self):
        return dict(phase=self.phase, reason=self.reason, member_ids=list(self.member_ids),
                    proposal_sha256=self.proposal_sha256, evidence_sha256=self.evidence_sha256,
                    wall_commands=dict(self.wall_commands), **dict(self.details))


def current_identity():
    config = load_config()
    binary = shutil.which("sumo")
    if binary is None:
        raise RuntimeError("SUMO must be on PATH")
    if sha(config.model_path) != config.model_sha256:
        raise ValueError("frozen model identity differs")
    return dict(config_sha256=sha(CONFIG_PATH), model_sha256=config.model_sha256,
                network_sha256=sha(config.network_path), sumo_binary_sha256=sha(binary),
                scenario_template_sha256=sha(config.scenario_template_path),
                active_source_sha256={p.name: sha(p) for p in sorted((ROOT / "traffic_shepherd").glob("*.py"))})


def validate_job(job):
    allowed = {"id", "case_id", "seed", "loading", "penetration", "role", "arm", "duration_s",
               "decision_time_s", "policy_spec"}
    if not allowed <= set(job) or set(job)-allowed-{"snapshot_times_s", "reference_job_id"} or job["arm"] not in ARMS:
        raise ValueError("job fields or arm are not the paired contract")
    if not isinstance(job["id"], str) or not job["id"] or any(c not in
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in job["id"]):
        raise ValueError("job ID must be a plain path-safe name")
    if not isinstance(job["case_id"], str) or not job["case_id"]:
        raise ValueError("case identity required")
    require_development_case(job["seed"], job["loading"], job["role"])
    if job["role"] not in {"exposed", "development"} or job["penetration"] not in {.05, .1, .2, .3}:
        raise ValueError("only registered development and 5-30% penetration supported")
    if type(job["duration_s"]) is not int or not 1 <= job["duration_s"] <= 2400:
        raise ValueError("positive bounded horizon required")
    if type(job["decision_time_s"]) is not int or not 0 <= job["decision_time_s"] < job["duration_s"]:
        raise ValueError("decision must be on the control grid before the service horizon")
    snapshots = job.get("snapshot_times_s", [job["decision_time_s"]])
    if (not snapshots or job["decision_time_s"] not in snapshots or len(set(snapshots)) != len(snapshots)
            or any(type(t) is not int or not 0 <= t < job["duration_s"] for t in snapshots)):
        raise ValueError("distinct registered snapshot times must include the decision")
    digest(job["policy_spec"])


def register(output, *, jobs, hypotheses, criteria, stopping_rules, policy_version,
             max_report_age_s=1., history_s=10., interaction_diagnostics=False):
    """Freeze exact cases/arms and a zero-retry budget before any physical launch."""
    if not jobs or not hypotheses or not stopping_rules or not policy_version:
        raise ValueError("declare jobs, hypotheses, stopping rules and exact policy version")
    if type(interaction_diagnostics) is not bool:
        raise ValueError("interaction diagnostics requires an explicit boolean")
    for job in jobs:
        validate_job(job)
    if len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("duplicate job allocation")
    for job in jobs:
        if job["arm"] == "damping":
            continue
        references = [j for j in jobs if j["arm"] == "damping" and
            (j["id"] == job["reference_job_id"] if "reference_job_id" in job else j["case_id"] == job["case_id"])]
        if len(references) != 1:
            raise ValueError("each treatment needs exactly one explicit damping reference")
        reference = references[0]
        if any(job[k] != reference[k] for k in ("seed", "loading", "penetration", "role", "duration_s")):
            raise ValueError("paired scenario/horizon fields differ")
        if job["decision_time_s"] not in reference.get("snapshot_times_s", [reference["decision_time_s"]]):
            raise ValueError("reference did not register this decision snapshot")
    if criteria is None:
        if any(j["duration_s"] >= 2400 or j["arm"] not in {"damping", "sham"} for j in jobs):
            raise ValueError("physical wall/full-horizon studies require explicit benefit criteria")
    else:
        BenefitCriteria(**criteria)
    RSUReportBuffer(max_age_s=max_report_age_s, history_s=history_s)
    registration = dict(schema="paired_escalation_experiment/1", registered_at_utc=datetime.now(timezone.utc).isoformat(),
        identity=current_identity(), jobs=jobs, hypotheses=hypotheses, criteria=criteria,
        stopping_rules=stopping_rules, policy_version=policy_version, automatic_retries=0,
        maximum_attempts=len(jobs), maximum_simulated_seconds=sum(j["duration_s"] for j in jobs),
        assignment="registered_experimental_assignment_not_trained_benefit_gate",
        communication=dict(delay_s=0., max_report_age_s=max_report_age_s, history_s=history_s),
        seed_exclusions=exclusion_provenance(), automatic_successors=False)
    if interaction_diagnostics:
        registration["interaction_diagnostics"] = True
    registration["registration_sha256"] = digest(registration)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "registration.json", registration)
    return registration


def read_registration(path, *, require_current=True):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "paired_escalation_experiment/1" or data.get("registration_sha256") != digest(
            {k: v for k, v in data.items() if k != "registration_sha256"}):
        raise ValueError("registration was modified or is unsupported")
    for job in data["jobs"]:
        validate_job(job)
    if data["maximum_attempts"] != len(data["jobs"]) or data["maximum_simulated_seconds"] != sum(
            j["duration_s"] for j in data["jobs"]) or data["automatic_retries"] != 0:
        raise ValueError("invalid bounded allocation")
    if require_current and data["identity"] != current_identity():
        raise ValueError("source or configuration changed; register a new version")
    if data["communication"]["delay_s"] != 0.:
        raise ValueError("this runner version implements only the registered zero-delay receipt")
    if type(data.get("interaction_diagnostics", False)) is not bool:
        raise ValueError("invalid registered interaction diagnostics flag")
    return path.parent, data


def run_one(registration, job_id, *, policy_factory=None):
    root, data = read_registration(registration)
    matches = [j for j in data["jobs"] if j["id"] == job_id]
    if len(matches) != 1:
        raise ValueError("unknown or duplicate registered job")
    if (root / "STOP.json").exists():
        raise RuntimeError("stage is stopped; preserve it and register a justified new stage")
    job = matches[0]
    allocations = root / "allocations"
    allocations.mkdir(exist_ok=True)
    # Atomic exclusive creation is both the budget charge and the no-retry lock.
    with (allocations / f"{job_id}.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(job_id=job_id, allocated_simulated_seconds=job["duration_s"],
                       registration_sha256=data["registration_sha256"], retries=0), stream, indent=2)
    factory = policy_factory or fixed_formation_policy
    return evaluate_arm(job=job, registration=data, output=root / "jobs" / job_id,
                        policy=factory(job, data))


def full_measured_state(api, live):
    """Offline exact kinematic snapshot; never passed to the policy callback.

This is not represented as a complete dump of SUMO's private RNG/controller state.
Matched requests, seed, binary and deterministic command prefix bind replay too.
"""
    return dict(timestamp_s=float(api.simulation.getTime()),
                channel="offline_evaluation_only", vehicles=[dict(
                    id=v, lane=api.vehicle.getLaneID(v), position_m=float(api.vehicle.getLanePosition(v)),
                    speed_mps=float(api.vehicle.getSpeed(v)), acceleration_mps2=float(api.vehicle.getAcceleration(v)),
                    route=api.vehicle.getRouteID(v), route_index=int(api.vehicle.getRouteIndex(v))) for v in sorted(live)])


def _member_measurements(sensor, members, live, now_s):
    rows = []
    for member in members:
        row = dict(id=member, present=member in live)
        if member in live:
            try:
                f = sensor.capture(member, now_s=now_s)
                row.update(lane=f.lane_id, distance_m=f.distance_to_merge_m, position_m=f.position_m,
                    speed_mps=f.speed_mps, acceleration_mps2=f.acceleration_mps2,
                    front_gap_m=f.front.gap_m if f.front else None,
                    rear_gap_m=f.rear.gap_m if f.rear else None,
                    front_relative_speed_mps=f.front.relative_speed_mps if f.front else None,
                    rear_relative_speed_mps=f.rear.relative_speed_mps if f.rear else None,
                    front_coverage_m=f.front_coverage_m, rear_coverage_m=f.rear_coverage_m)
            except ValueError as exc:
                row["measurement_unavailable"] = str(exc)
        rows.append(row)
    return rows


def evaluate_arm(*, job, registration, output, policy, budget=None):
    """One charged arm, with no training, retries, automatic successor or global features.

``budget`` (registration-declared enforcement block, plus ``target_dir``) is
required from registered campaign runners: the arm's budget is reserved in
the persistent campaign ledger immediately before SUMO starts, and a
rejected reservation raises before the simulator is invoked. Callers that
pass no budget are direct, explicitly bounded tool invocations.
"""
    validate_job(job)
    cav_speed_mode = registration.get("cav_speed_mode", 31)
    if type(cav_speed_mode) is not int or cav_speed_mode not in (31, 27):
        raise ValueError("registered CAV command interface must explicitly select31 or27")
    diagnostics_enabled = registration.get("interaction_diagnostics", False)
    if type(diagnostics_enabled) is not bool:
        raise ValueError("invalid registered interaction diagnostics flag")
    config = load_config()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("preserving existing evidence; never resume or retry implicitly")
    import traci
    if traci.isLoaded():
        raise RuntimeError("refusing to reuse or close a pre-existing TraCI connection")
    binary = shutil.which("sumo")
    if binary is None:
        raise RuntimeError("SUMO must be on PATH")
    controller = CirclesController(config)
    horizon, decision = job["duration_s"], job["decision_time_s"]
    snapshot_times = set(job.get("snapshot_times_s", [decision]))
    last_snapshot = max(snapshot_times)
    snapshots = {}
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema="paired_escalation_arm/1", status="preparing", **registration["identity"],
        cav_speed_mode=cav_speed_mode,
        command_interface="circles_emergency_braking/1" if cav_speed_mode == 27 else "circles_legacy31/1",
        seed=job["seed"], mainline_vph=job["loading"], penetration=job["penetration"], role=job["role"],
        controller=job["arm"], duration_s=horizon, mechanics_only=horizon != config.episode_seconds,
        decision_time_s=decision, job_id=job["id"], case_id=job["case_id"],
        reference_job_id=job.get("reference_job_id"), snapshot_times_s=sorted(snapshot_times),
        registration_sha256=registration["registration_sha256"], policy_spec=job["policy_spec"],
        assignment=registration["assignment"], allocated_simulated_seconds=horizon,
        scientifically_accepted=False, offline_traffic_observations=True,
        communication=registration["communication"], formation_samples_are_lifecycle_scoped=True)
    if diagnostics_enabled:
        manifest["interaction_diagnostics"] = dict(enabled=True, schema=INTERACTION_SCHEMA,
            file=INTERACTION_FILENAME, online_input=False)
    write_json(output / "manifest.json", manifest)
    api, owner, diagnostic, diagnostic_record, started = None, None, None, None, time.monotonic()
    try:
        scene, requests, scenario = build_scenario(output / "scenario", job["seed"], job["loading"],
            penetration=job["penetration"], role=job["role"], config=config)
        manifest.update(scenario=scenario, scenario_identity=scenario_identity(scene, requests,
                        network_path=config.network_path), status="running")
        manifest["exogenous_future_sha256"] = digest(dict(requests=[asdict(r) for r in requests],
            sumo_seed=job["seed"], binary_sha256=manifest["sumo_binary_sha256"],
            scenario_identity=manifest["scenario_identity"]))
        write_json(output / "manifest.json", manifest)
        if budget is None:
            start_sumo(binary, scene, seed=job["seed"], config=config, api=traci)
        else:
            budget_guard.reserve_and_launch(
                budget, target_dir=Path(budget["target_dir"]),
                launcher=lambda: start_sumo(binary, scene, seed=job["seed"],
                                            config=config, api=traci))
        api = traci
        cav_ids = frozenset(r.vehicle_id for r in requests if r.is_cav)
        owner = CommandOwnership(api, cav_ids, controller, cav_speed_mode=cav_speed_mode) if cav_speed_mode == 27 else CommandOwnership(api, cav_ids, controller)
        lane_map = observation_map(api, config)
        sensor = SumoLocalSensor(api, config, lane_map)
        observer = OfflineTrafficObserver(api, config)
        comm = registration["communication"]
        buffer = RSUReportBuffer(max_age_s=comm["max_report_age_s"], history_s=comm["history_s"])
        ledger, totals, command_counts, fallback_counts = RequestLedger(requests), Counter(), Counter(), Counter()
        prior_commands, history, collisions_previous = {}, [], set()
        prefix_reports, prefix_commands = hashlib.sha256(), hashlib.sha256()
        minimum_ttc, policy_counts, invariant_counts = None, Counter(), Counter()
        terminal_geometry_logged = set()
        filenames = ("telemetry", "intervals", "traffic", "policy", "formation", "offline_state")
        streams = {name: (output / f"{name}.jsonl").open("x", encoding="utf-8") for name in filenames}
        def emit(name, row):
            streams[name].write(json.dumps(row, allow_nan=False) + "\n")
        try:
            if diagnostics_enabled:
                diagnostic = OfflineInteractionObserver(output / INTERACTION_FILENAME,
                    requests=requests, horizon_s=horizon,
                    edge_regions={edge: region for region, edge in config.edge_ids.items()})
                diagnostic.sample(offline_read_cache(api), now_s=float(api.simulation.getTime()),
                    live_ids=api.vehicle.getIDList())
            for tick in range(horizon):
                now = float(api.simulation.getTime())
                if abs(now-tick) > 1e-6:
                    raise RuntimeError("SUMO/control time grid mismatch")
                live = set(api.vehicle.getIDList())
                emit("traffic", observer.sample(now_s=now))
                frames, errors, cache_stats = capture_batch(api, live & cav_ids, now_s=now,
                    config=config, lane_end_distance_to_merge=lane_map)
                invariant_counts.update(cache_stats)
                for cav, error in errors.items():
                    emit("telemetry", dict(kind="observation_unavailable", vehicle_id=cav,
                                           timestamp_s=now, reason=error))
                # Self reports contain the most recent already issued command, not
                # future wall action or a privileged prediction of its consequences.
                received_packets = []
                for cav, frame in sorted(frames.items()):
                    previous = prior_commands.get(cav)
                    previous_row = previous.as_dict() if previous else {}
                    report = CAVReport(frame, now, previous.owner if previous else "native",
                        previous_row.get("network_output_mps2"),
                        previous.issued_acceleration_mps2 if previous else None,
                        previous.fallback_reason if previous else None)
                    buffer.receive(report, received_at_s=now)
                    history.append((now, report))
                    received_packets.append(dict(received_at_s=now, **report.payload()))
                    emit("telemetry", dict(kind="rsu_received_report", received_at_s=now, **report.payload()))
                history = [(receipt, r) for receipt, r in history if receipt >= now-comm["history_s"]]
                if now <= last_snapshot:
                    # The full received-packet stream binds history without hashing
                    # each retained old packet repeatedly at every new tick.
                    prefix_reports.update(digest(dict(now_s=now, received_packets=received_packets)).encode())
                    if now in snapshot_times:
                        evidence = dict(now_s=now, reports=[r.payload() for r in buffer.reports(now_s=now)],
                            history=[dict(received_at_s=t, **r.payload()) for t, r in history])
                        offline = full_measured_state(api, live)
                        emit("offline_state", offline)
                        snapshot = dict(decision_time_s=now, evidence=evidence,
                            predecision_state_sha256=digest(offline),
                            predecision_report_evidence_sha256=digest(evidence),
                            predecision_report_prefix_sha256=prefix_reports.hexdigest(),
                            predecision_command_prefix_sha256=prefix_commands.hexdigest())
                        snapshots[str(int(now))] = snapshot
                        if now == decision:
                            manifest.update({k: v for k, v in snapshot.items() if k != "evidence"})
                # Reference inference is pure and uses the same immutable actor instance.
                expected = {v: controller.decide(f, now_s=now) for v, f in frames.items()}
                references = {v: d.command_acceleration_mps2 for v, d in expected.items()
                              if d.fallback_reason is None}
                native_references = frozenset(v for v, d in expected.items() if d.fallback_reason is not None)
                policy_input = PolicyInput(now, tuple(buffer.reports(now_s=now)), tuple(history),
                                           references, native_references)
                result = policy(policy_input)
                if not isinstance(result, PolicyOutput):
                    raise TypeError("policy must return the typed PolicyOutput boundary")
                if now < decision and (result.wall_commands or result.phase != "damping"):
                    raise ValueError("registered experiment acted before its decision time")
                if job["arm"] in {"damping", "sham"} and result.wall_commands:
                    raise ValueError("damping and sham arms cannot issue wall commands")
                if now in snapshot_times:
                    snapshots[str(int(now))]["proposal_sha256"] = result.proposal_sha256
                    if now == decision:
                        manifest["proposal_sha256"] = result.proposal_sha256
                    write_json(output / "decision_snapshots.json", snapshots)
                    write_json(output / "manifest.json", manifest)
                emit("policy", dict(time_s=now, assignment=registration["assignment"], **result.payload()))
                policy_counts[result.phase] += 1
                records = owner.step(frames, now_s=now, live_ids=live, wall_commands=result.wall_commands)
                if cav_speed_mode == 27:
                    for change in owner.last_mode_changes:
                        emit("telemetry", dict(kind="command_mode", **change))
                commanded = {r.vehicle_id for r in records}
                records_by_id = {r.vehicle_id: r for r in records}
                for record in records:
                    if record.vehicle_id not in result.wall_commands and record.vehicle_id in expected:
                        d, row = expected[record.vehicle_id].as_dict(), record.as_dict()
                        if tuple(row.get(k) for k in ("network_output_mps2", "issued_acceleration_mps2",
                            "issued_target_speed_mps", "fallback_reason", "network_output_mps2_nonfinite")) != tuple(
                                d.get(k) for k in ("network_output_mps2", "command_acceleration_mps2",
                                "target_speed_mps", "fallback_reason", "network_output_mps2_nonfinite")):
                            raise RuntimeError("nonmember/released vehicle differs from identical damping")
                        invariant_counts["exact_damping_reference_decisions"] += 1
                    invariant_counts["hdv_commands"] += int(record.vehicle_id not in cav_ids)
                    prior_commands[record.vehicle_id] = record
                    row = record.as_dict()
                    emit("telemetry", dict(kind="command", **row))
                    command_counts[record.owner] += 1
                    if record.fallback_reason:
                        fallback_counts[record.fallback_reason] += 1
                    if now < last_snapshot:
                        prefix_commands.update(digest(row).encode())
                    if record.vehicle_id in frames:
                        report = CAVReport(frames[record.vehicle_id], now, record.owner,
                            row["network_output_mps2"], row["issued_acceleration_mps2"], record.fallback_reason)
                        emit("telemetry", dict(kind="cav_report", **report.payload()))
                prior_commands = {v: r for v, r in prior_commands.items() if v in live}
                interval = Counter()
                proposal_key = result.proposal_sha256
                terminal = result.phase in {"release", "abort", "refused"}
                log_formation = bool(result.member_ids and now >= decision and (
                    result.phase in {"assembly", "hold", "taper"}
                    or terminal and proposal_key not in terminal_geometry_logged
                    or now in snapshot_times))
                for substep in range(2):
                    # Start-of-interval measured geometry at every 0.5 s physics tick.
                    if log_formation:
                        emit("formation", dict(time_s=float(api.simulation.getTime()),
                            phase=result.phase, reason=result.reason,
                            proposal_sha256=proposal_key, evidence_sha256=result.evidence_sha256,
                            readiness_count=result.details.get("readiness_count"),
                            readiness_control_dt_s=config.control_dt_s,
                            target_speed_mps=result.details.get("target_speed_mps"),
                            compact_reported=result.details.get("compact"),
                            readiness_samples_required=result.details.get("readiness_samples_required"),
                            formed_span_m=result.details.get("formed_span_m"),
                            formed_speed_spread_mps=result.details.get("formed_speed_spread_mps"),
                            target_speed_tolerance_mps=result.details.get("target_speed_tolerance_mps"),
                            assembly_completed_at_s=result.details.get("assembly_completed_at_s"),
                            wall_command_ids=sorted(result.wall_commands),
                            owner_by_member={v: records_by_id[v].owner if v in records_by_id else None
                                             for v in result.member_ids},
                            command_time_s=now, members=_member_measurements(sensor, result.member_ids,
                                live, float(api.simulation.getTime()))))
                    api.simulationStep()
                    after = float(api.simulation.getTime())
                    if abs(after-(now+(substep+1)*.5)) > 1e-6:
                        raise RuntimeError("SUMO physics time grid mismatch")
                    live = set(api.vehicle.getIDList())
                    # Cache only immutable physical reads at this exact instant.
                    # Never share this facade with control, sensing or PolicyInput.
                    physical_api = offline_read_cache(api) if diagnostic is not None else api
                    interval.update(braking_effort([float(physical_api.vehicle.getAcceleration(v)) for v in sorted(live)], .5))
                    exposure, closing_minimum = ttc_exposure(physical_api, sorted(live), .5)
                    interval["ttc_below_3_vehicle_seconds"] += exposure
                    if closing_minimum is not None:
                        minimum_ttc = closing_minimum if minimum_ttc is None else min(minimum_ttc, closing_minimum)
                    departed_ids = list(api.simulation.getDepartedIDList())
                    arrived_ids = list(api.simulation.getArrivedIDList())
                    interval["requested_not_arrived_vehicle_seconds"] += ledger.advance(after, departed_ids, arrived_ids)
                    colliding = set(api.simulation.getCollidingVehiclesIDList())
                    new_colliding = colliding-collisions_previous
                    interval["collision_involvements"] += len(new_colliding)
                    collisions_previous = colliding
                    starting_teleports = api.simulation.getStartingTeleportNumber()
                    interval["starting_teleports"] += starting_teleports
                    for cav in sorted(commanded & live):
                        emit("telemetry", dict(kind="response", **owner.record_response(cav,
                            timestamp_s=after, speed_mps=float(api.vehicle.getSpeed(cav)),
                            acceleration_mps2=float(api.vehicle.getAcceleration(cav))).as_dict()))
                    if diagnostic is not None:
                        diagnostic.sample(physical_api, now_s=after, live_ids=live,
                            departed_ids=departed_ids, arrived_ids=arrived_ids, colliding_ids=colliding,
                            new_colliding_ids=new_colliding, starting_teleports=starting_teleports,
                            starting_teleport_ids=(api.simulation.getStartingTeleportIDList()
                                if hasattr(api.simulation, "getStartingTeleportIDList") else None))
                if log_formation and terminal:
                    terminal_geometry_logged.add(proposal_key)
                totals.update(interval)
                emit("intervals", dict(start_s=now, end_s=after, **interval))
                for stream in streams.values():
                    stream.flush()
        finally:
            for stream in streams.values():
                stream.close()
            if diagnostic is not None:
                diagnostic_record = diagnostic.close()
                manifest["interaction_diagnostics"] = diagnostic_record
        summary = dict(schema="paired_escalation_outcomes/1", seed=job["seed"], controller=job["arm"],
            duration_s=horizon, mechanics_only=horizon != config.episode_seconds,
            complete_fixed_horizon=horizon == config.episode_seconds, **service_summary(ledger, horizon),
            requested_due_at_horizon=ledger.due(horizon), totals=dict(totals), minimum_closing_ttc_s=minimum_ttc,
            command_counts=dict(command_counts), fallback_counts=dict(fallback_counts),
            phase_seconds=dict(policy_counts), ownership_invariants=dict(invariant_counts),
            scenario_identity=manifest["scenario_identity"], elapsed_wall_seconds=time.monotonic()-started,
            sumo_version=list(api.getVersion()), assignment=registration["assignment"], scientifically_accepted=False)
        summary["all_requested_service_completed"] = (summary["requested_due_at_horizon"] == summary["population"]
            and summary["arrived"] == summary["population"] and summary["unfinished_requested"] == 0)
        summary["censored_service"] = not summary["all_requested_service_completed"]
        summary["future_requests_at_horizon"] = summary["population"]-summary["requested_due_at_horizon"]
        owner.release_all(live_ids=set(api.vehicle.getIDList()))
        owner = None
        close_sumo(api)
        api = None
        write_json(output / "summary.json", summary)
        manifest.update(status="complete", summary_sha256=sha(output / "summary.json"),
                        decision_snapshots_sha256=sha(output / "decision_snapshots.json"))
        manifest.update({f"{name}_sha256": sha(output / f"{name}.jsonl") for name in filenames})
        write_json(output / "manifest.json", manifest)
        return summary
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc))
        write_json(output / "manifest.json", manifest)
        raise
    finally:
        if api is not None:
            try:
                if owner is not None:
                    owner.release_all(live_ids=set(api.vehicle.getIDList()))
            finally:
                close_sumo(api)


def verified_arm(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("arm incomplete; retain its allocation and failure in comparison")
    diagnostic = manifest.get("interaction_diagnostics")
    if diagnostic is not None:
        if (diagnostic.get("enabled") is not True or diagnostic.get("schema") != INTERACTION_SCHEMA
                or diagnostic.get("file") != INTERACTION_FILENAME or diagnostic.get("online_input") is not False
                or diagnostic.get("complete") is not True or diagnostic.get("final_time_s") != manifest["duration_s"]
                or diagnostic.get("sample_count") != 2*manifest["duration_s"]+1):
            raise ValueError("incomplete or invalid offline interaction diagnostic stream")
        if file_sha256(path / INTERACTION_FILENAME) != diagnostic["sha256"]:
            raise ValueError("changed offline interaction diagnostic stream")
    for name in ("summary", "telemetry", "intervals", "traffic", "policy", "formation", "offline_state"):
        suffix = "json" if name == "summary" else "jsonl"
        if sha(path / f"{name}.{suffix}") != manifest[f"{name}_sha256"]:
            raise ValueError(f"changed arm artifact: {name}")
    if sha(path / "decision_snapshots.json") != manifest["decision_snapshots_sha256"]:
        raise ValueError("changed predecision snapshots")
    verify_snapshots(path, manifest)
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    for key in ("seed", "controller", "duration_s", "scenario_identity", "mechanics_only"):
        if summary.get(key) != manifest.get(key):
            raise ValueError(f"manifest/summary mismatch: {key}")
    previous, totals = 0., Counter()
    for line in (path / "intervals.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.pop("start_s") != previous or row["end_s"] != previous+1.:
            raise ValueError("missing physical outcome interval")
        previous = row.pop("end_s")
        if any(not isfinite(v) or v < 0 for v in row.values()):
            raise ValueError("invalid physical interval")
        totals.update(row)
    if previous != manifest["duration_s"] or dict(totals) != summary["totals"]:
        raise ValueError("measured intervals do not reproduce summary")
    if abs(totals["requested_not_arrived_vehicle_seconds"]-
           summary["system_seconds_per_requested_vehicle"]*summary["population"]) > 1e-6:
        raise ValueError("service summary disagrees with measured intervals")
    return manifest, summary


def verify_snapshots(path, manifest):
    """Rebuild proposal/evidence/physical snapshot hashes from separate raw logs.

Hashes in a snapshot or manifest are assertions to check, never their own proof.
This reconstruction uses actual receipt rows and predecision issued commands;
offline live-vehicle kinematics are independently bound to their own raw channel.
"""
    path = Path(path)
    snapshots = json.loads((path / "decision_snapshots.json").read_text(encoding="utf-8"))
    expected_keys = {str(t) for t in manifest["snapshot_times_s"]}
    if set(snapshots) != expected_keys:
        raise ValueError("snapshot times differ from registered manifest")
    last = max(manifest["snapshot_times_s"])
    received, commands = {}, {}
    with (path / "telemetry.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["kind"] == "rsu_received_report":
                receipt = row.pop("received_at_s")
                row.pop("kind")
                if not isfinite(receipt) or receipt < 0 or receipt % 1.:
                    raise ValueError("receipt outside registered control grid")
                if receipt <= last:
                    report = report_from_payload(row)
                    received.setdefault(receipt, []).append((report, row))
            elif row["kind"] == "command" and row["timestamp_s"] < last:
                row.pop("kind")
                commands.setdefault(row["timestamp_s"], []).append(row)
    offline = {}
    with (path / "offline_state.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = str(int(row["timestamp_s"]))
            if row["timestamp_s"] != int(row["timestamp_s"]) or key in offline:
                raise ValueError("duplicate or non-grid offline decision state")
            offline[key] = row
    if set(offline) != expected_keys:
        raise ValueError("missing or extra offline decision state")
    decisions = {}
    with (path / "policy.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["time_s"] in manifest["snapshot_times_s"]:
                key = str(int(row["time_s"]))
                if key in decisions:
                    raise ValueError("duplicate policy decision time")
                decisions[key] = row["proposal_sha256"]
    buffer = RSUReportBuffer(max_age_s=manifest["communication"]["max_report_age_s"],
                             history_s=manifest["communication"]["history_s"])
    history, prefix_reports, prefix_commands = [], hashlib.sha256(), hashlib.sha256()
    for tick in range(last+1):
        now = float(tick)
        rows = received.get(now, ())
        if len({r.frame.vehicle_id for r, _ in rows}) != len(rows):
            raise ValueError("duplicate received packet in one control tick")
        packets = []
        for report, row in rows:
            buffer.receive(report, received_at_s=now)
            history.append((now, report))
            packets.append(dict(received_at_s=now, **row))
        history = [(t, r) for t, r in history if t >= now-buffer.history_s]
        prefix_reports.update(digest(dict(now_s=now, received_packets=packets)).encode())
        key = str(tick)
        if key in snapshots:
            evidence = dict(now_s=now, reports=[r.payload() for r in buffer.reports(now_s=now)],
                history=[dict(received_at_s=t, **r.payload()) for t, r in history])
            expected = dict(decision_time_s=now, evidence=evidence,
                predecision_state_sha256=digest(offline[key]),
                predecision_report_evidence_sha256=digest(evidence),
                predecision_report_prefix_sha256=prefix_reports.hexdigest(),
                predecision_command_prefix_sha256=prefix_commands.hexdigest(),
                proposal_sha256=decisions.get(key))
            if snapshots[key] != expected:
                raise ValueError("predecision snapshot does not reconstruct from raw received/command/offline logs")
            if tick == manifest["decision_time_s"] and any(manifest.get(k) != v for k, v in expected.items() if k != "evidence"):
                raise ValueError("manifest primary snapshot disagrees with independent reconstruction")
        for row in commands.get(now, ()):
            prefix_commands.update(digest(row).encode())
    return dict(snapshots_verified=len(snapshots), independently_reconstructed=True)


def verify_pair(reference_path, treatment_path):
    control_manifest, control = verified_arm(reference_path)
    treatment_manifest, treatment = verified_arm(treatment_path)
    if control_manifest["controller"] != "damping":
        raise ValueError("paired reference must be continued frozen damping")
    decision_key = str(int(treatment_manifest["decision_time_s"]))
    left = json.loads((Path(reference_path) / "decision_snapshots.json").read_text(encoding="utf-8"))[decision_key]
    right = json.loads((Path(treatment_path) / "decision_snapshots.json").read_text(encoding="utf-8"))[decision_key]
    # Derived views bind a shared immutable damping arm to this exact opportunity;
    # neither original manifest nor summary is rewritten to manufacture a pair.
    control_manifest = {**control_manifest, **{k: v for k, v in left.items() if k != "evidence"}}
    treatment_manifest = {**treatment_manifest, **{k: v for k, v in right.items() if k != "evidence"}}
    for key in ("registration_sha256", "proposal_sha256", "decision_time_s", "exogenous_future_sha256",
                "predecision_state_sha256", "predecision_report_evidence_sha256",
                "predecision_report_prefix_sha256", "predecision_command_prefix_sha256"):
        if key not in control_manifest or control_manifest[key] != treatment_manifest.get(key):
            raise ValueError(f"unmatched paired decision: {key}")
    mechanics = control_manifest["mechanics_only"]
    if mechanics:
        for key in ("seed", "mainline_vph", "penetration", "duration_s", "scenario_identity", "mechanics_only",
                    "config_sha256", "model_sha256", "sumo_binary_sha256", "active_source_sha256"):
            if control_manifest[key] != treatment_manifest.get(key):
                raise ValueError(f"unmatched mechanics pair: {key}")
        result = None  # Short trajectories cannot become wall-benefit labels.
    else:
        result = paired_effect(control_manifest, control, treatment_manifest, treatment)
    if treatment_manifest["controller"] == "sham":
        for name in ("telemetry_sha256", "intervals_sha256", "traffic_sha256"):
            if control_manifest[name] != treatment_manifest[name]:
                raise ValueError("sham is not exact damping telemetry and physical equivalence")
    return dict(effect=result, prefix_verified=True, sham_exact=treatment_manifest["controller"] == "sham",
                mechanics_only=mechanics,
                reference_manifest_sha256=sha(Path(reference_path) / "manifest.json"),
                treatment_manifest_sha256=sha(Path(treatment_path) / "manifest.json"),
                decision_snapshot_sha256=digest(left),
                censored_service=bool(control.get("censored_service") or treatment.get("censored_service")))


def analyze(registration):
    root, data = read_registration(registration, require_current=False)
    completed, jobs = {}, []
    for job in data["jobs"]:
        path = root / "jobs" / job["id"]
        row = dict(job_id=job["id"], case_id=job["case_id"], arm=job["arm"], status="pending")
        if (root / "allocations" / f"{job['id']}.json").exists():
            row["status"] = "allocated_incomplete"
        if path.exists():
            try:
                manifest, summary = verified_arm(path)
                if manifest["registration_sha256"] != data["registration_sha256"]:
                    raise ValueError("arm belongs to another registration")
                if any(manifest.get(k) != v for k, v in data["identity"].items()):
                    raise ValueError("arm source/configuration differs from its registration")
                for key, field in (("seed", "seed"), ("mainline_vph", "loading"),
                                   ("penetration", "penetration"), ("controller", "arm"),
                                   ("duration_s", "duration_s"), ("decision_time_s", "decision_time_s")):
                    if manifest[key] != job[field]:
                        raise ValueError(f"arm differs from its registered job: {key}")
                row.update(status="complete", phase_seconds=summary["phase_seconds"],
                           ownership_invariants=summary["ownership_invariants"])
                completed[job["id"]] = path
            except (ValueError, KeyError, OSError) as exc:
                row.update(status="failed_or_invalid", error=str(exc))
        jobs.append(row)
    effects = []
    for job in data["jobs"]:
        if job["arm"] == "damping" or job["id"] not in completed:
            continue
        references = [j for j in data["jobs"] if j["arm"] == "damping" and
            (j["id"] == job["reference_job_id"] if "reference_job_id" in job else j["case_id"] == job["case_id"])]
        if len(references) != 1 or references[0]["id"] not in completed:
            continue
        result = verify_pair(completed[references[0]["id"]], completed[job["id"]])
        if job["arm"] == "wall" and result["effect"] is not None:
            label = benefit_label(result["effect"], BenefitCriteria(**data["criteria"]))
            if result["censored_service"] and label["label"] == "beneficial":
                label.update(label="inconclusive_censored_service", material_improvements=[])
            result["physical_label"] = label
        effects.append(dict(case_id=job["case_id"], arm=job["arm"], **result))
    return dict(jobs=jobs, effects=effects, completed=sum(j["status"] == "complete" for j in jobs),
                allocated=sum((root / "allocations" / f"{j['id']}.json").exists() for j in data["jobs"]),
                maximum_simulated_seconds=data["maximum_simulated_seconds"],
                failed_and_refused_attempts_excluded=False, scientifically_accepted=False)


def register_prefix_bridge(output, *, reference):
    """Freeze exactly two 30 s no-wall runs, matched to saved S02 prefix evidence."""
    from .escalation_study import verified_run
    reference = Path(reference).resolve()
    old, _ = verified_run(reference)
    if (old["seed"], old["mainline_vph"], old["penetration"], old["controller"], old["duration_s"]) != (
            4107, 2000, .2, "damping", 2400):
        raise ValueError("bridge is restricted to the declared exposed S02 reference")
    current = current_identity()
    for key in ("config_sha256", "model_sha256", "sumo_binary_sha256"):
        if current[key] != old[key]:
            raise ValueError(f"reference differs in frozen runtime identity: {key}")
    if current["network_sha256"] != old["scenario"]["network_sha256"]:
        raise ValueError("reference network differs")
    dependencies = ("config.py", "damping.py", "ownership.py", "reporting.py", "sensing.py", "scenarios.py",
                    "seeds.py", "simulation.py", "metrics.py", "escalation_observations.py")
    if any(current["active_source_sha256"][name] != old["active_source_sha256"][name] for name in dependencies):
        raise ValueError("a shared runtime dependency changed; this bridge must be assessed and versioned anew")
    jobs = [dict(id=f"prefix_{arm}", case_id="s4107_m2000_p20_prefix30", seed=4107, loading=2000,
        penetration=.2, role="exposed", arm=arm, duration_s=30, decision_time_s=10,
        policy_spec=dict(required_lanes=["1191554636_0", "1191554636_1"], member_ids=[], profile={}))
        for arm in ("damping", "sham")]
    data = register(output, jobs=jobs, hypotheses=[
        "Memoized CAV sensor reads and paired callback instrumentation preserve saved damping physics exactly.",
        "A sham assignment with no selected wall members has exactly the same commands and measured trajectory as damping."],
        criteria=None, stopping_rules=["Any source, scenario, frame, command or measured-interval mismatch stops the bridge.",
            "At most two 30-second trajectories, zero retries, no efficacy or wall-formation claim."],
        policy_version="paired_prefix_equivalence_bridge/1")
    bridge = dict(schema="paired_prefix_bridge/1", registered_at_utc=datetime.now(timezone.utc).isoformat(),
        paired_registration_sha256=data["registration_sha256"], reference_path=str(reference),
        reference_manifest_sha256=sha(reference / "manifest.json"),
        reference_artifact_sha256={name: old[f"{name}_sha256"] for name in ("telemetry", "intervals", "traffic", "summary")},
        unchanged_shared_dependencies={name: current["active_source_sha256"][name] for name in dependencies},
        exact_criteria=dict(sensor_frames=True, model_outputs=True, issued_commands=True, measured_responses=True,
            physical_intervals=True, offline_region_measurements=True, floating_tolerance=0.),
        scope=dict(duration_s=30, maximum_attempts=2, maximum_simulated_seconds=60, automatic_retries=0,
            decision_time_s=10, no_wall_members=True, full_horizon_reuse_qualified=False),
        limitation="A 30 s prefix validates this integration prefix, not all 2400 s or original full-state wall pairing.")
    bridge["bridge_sha256"] = digest(bridge)
    write_json(Path(output) / "bridge_registration.json", bridge)
    return dict(registration=str(Path(output).resolve() / "registration.json"),
        bridge_registration=str(Path(output).resolve() / "bridge_registration.json"),
        simulations_started=0, maximum_simulated_seconds=60)


def compare_saved_prefix(reference, replay, *, duration_s):
    """Exact read-only comparison; can later audit a complete 2400 s replay bridge."""
    reference, replay = Path(reference), Path(replay)
    old = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))
    new, _ = verified_arm(replay)
    if type(duration_s) is not int or duration_s <= 0 or duration_s > min(old["duration_s"], new["duration_s"]):
        raise ValueError("prefix duration exceeds recorded trajectories")
    for key in ("seed", "mainline_vph", "penetration", "scenario_identity", "config_sha256", "model_sha256", "sumo_binary_sha256"):
        if old[key] != new[key]:
            raise ValueError(f"reference/replay identity differs: {key}")
    def rows(path, name, *, original=False):
        if original and sha(path / f"{name}.jsonl") != old[f"{name}_sha256"]:
            raise ValueError("saved reference artifact changed")
        result = []
        with (path / f"{name}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if name == "telemetry":
                    kind = row["kind"]
                    if kind not in {"command", "response", "cav_report", "observation_unavailable"}:
                        continue
                    now = row["frame"]["timestamp_s"] if kind == "cav_report" else row["timestamp_s"]
                    include = now <= duration_s if kind == "response" else now < duration_s
                elif name == "intervals":
                    include = row["end_s"] <= duration_s
                else:
                    include = row["timestamp_s"] < duration_s
                if include:
                    result.append(row)
        return result
    results = {}
    for name in ("telemetry", "intervals", "traffic"):
        left, right = rows(reference, name, original=True), rows(replay, name)
        if left != right:
            mismatch = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))
            raise ValueError(f"saved {name} prefix differs at row {mismatch}; rows={len(left)}/{len(right)}")
        results[name] = dict(exact=True, rows=len(left), canonical_sha256=digest(left))
    return dict(duration_s=duration_s, comparisons=results, numerical_tolerance=0.,
        complete_horizon_replay=duration_s == old["duration_s"] == new["duration_s"] == 2400,
        source_wait_prefix_independently_logged=False,
        note="Old prefix source waiting has no separate interval series; all-request system intervals remain exact.")


def analyze_prefix_bridge(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "paired_prefix_bridge/1" or data.get("bridge_sha256") != digest(
            {k: v for k, v in data.items() if k != "bridge_sha256"}):
        raise ValueError("bridge registration changed")
    root, registration = read_registration(path.parent / "registration.json", require_current=False)
    if registration["registration_sha256"] != data["paired_registration_sha256"]:
        raise ValueError("bridge and paired allocation differ")
    reference = Path(data["reference_path"])
    if sha(reference / "manifest.json") != data["reference_manifest_sha256"]:
        raise ValueError("reference manifest changed")
    arms = {}
    for name in ("damping", "sham"):
        arms[name] = compare_saved_prefix(reference, root / "jobs" / f"prefix_{name}", duration_s=30)
    return dict(status="mechanics_prefix_verified", arms=arms,
        sham_pair=verify_pair(root / "jobs/prefix_damping", root / "jobs/prefix_sham"),
        budget=dict(attempts=2, simulated_seconds=60, retries=0), scientifically_accepted=False,
        full_horizon_reuse_qualified=False)


def fixed_formation_policy(job, registration):
    # Kept lazy so callback-boundary tests do not need to instantiate treatment.
    from .formation import (FormationController, FormationProfile, assess_formation,
                            authorize_experiment, make_proposal)
    spec, proposal, controller, passive = job["policy_spec"], None, None, PolicyOutput()
    opportunities = spec.get("opportunities", [dict(time_s=job["decision_time_s"], **spec)])
    by_time = {row["time_s"]: row for row in opportunities}
    if len(by_time) != len(opportunities):
        raise ValueError("duplicate proposal decision time")
    if job["arm"] not in {"damping", "sham"} and set(by_time) != {job["decision_time_s"]}:
        raise ValueError("experimental wall arm has one registered opportunity")
    def policy(context):
        nonlocal proposal, controller, passive
        if context.now_s < job["decision_time_s"]:
            return PolicyOutput()
        if context.now_s in by_time:
            row = by_time[context.now_s]
            profile = FormationProfile(**row["profile"])
            try:
                proposal = make_proposal(context.reports, now_s=context.now_s,
                    required_lanes=tuple(row["required_lanes"]), member_ids=tuple(row["member_ids"]), profile=profile)
            except ValueError as exc:
                # Missing reports are a retained refusal, not a simulator failure or
                # invented observed free space. The proposal request remains bound.
                passive = PolicyOutput({}, "damping" if job["arm"] in {"damping", "sham"} else "refused",
                    f"proposal_unavailable:{exc}", tuple(row["member_ids"]), digest(row),
                    digest([r.payload() for r in context.reports]), dict(proposal_materialized=False))
                return passive
            assessment = assess_formation(proposal, context.reports, now_s=context.now_s)
            passive = PolicyOutput({}, "damping" if job["arm"] in {"damping", "sham"} else "refused",
                f"{assessment.status}:{assessment.reason}", proposal.member_ids, proposal.fingerprint,
                assessment.evidence_fingerprint, dict(proposal_materialized=True,
                    initial_assessment=asdict(assessment)))
            if job["arm"] in {"damping", "sham"} or not assessment.approved:
                return passive
            authorization = authorize_experiment(proposal, assessment,
                registration_id=registration["registration_sha256"], arm=job["arm"])
            controller = FormationController(proposal, authorization)
        if controller is None or job["arm"] in {"damping", "sham"}:
            return passive
        members = frozenset(proposal.member_ids)
        step = controller.step(context.reports, now_s=context.now_s,
            damping_commands={v: a for v, a in context.damping_commands.items() if v in members},
            damping_native_ids=context.damping_native_ids & members)
        phase, commands, reason = step.phase, dict(step.commands), step.reason
        return PolicyOutput(commands, phase, reason, tuple(step.member_ids), step.proposal_fingerprint,
            step.evidence_fingerprint, dict(readiness_count=step.readiness_count, span_m=step.span_m,
                speed_spread_mps=step.speed_spread_mps, target_speed_mps=step.target_speed_mps,
                compact=step.compact, assembly_completed_at_s=step.assembly_completed_at_s,
                readiness_samples_required=proposal.profile.readiness_samples,
                formed_span_m=proposal.profile.formed_span_m,
                formed_speed_spread_mps=proposal.profile.formed_speed_spread_mps,
                target_speed_tolerance_mps=proposal.profile.target_speed_tolerance_mps))
    return policy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    reg = sub.add_parser("register", help="freeze a fully specified job JSON without simulation")
    reg.add_argument("--spec", required=True)
    reg.add_argument("--output", required=True)
    bridge = sub.add_parser("register-bridge", help="freeze two 30 s mechanics runs against the saved S02 prefix")
    bridge.add_argument("--reference", required=True)
    bridge.add_argument("--output", required=True)
    bridge_analysis = sub.add_parser("analyze-bridge")
    bridge_analysis.add_argument("--bridge-registration", required=True)
    run = sub.add_parser("run")
    run.add_argument("--registration", required=True)
    run.add_argument("--job-id", required=True)
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--registration", required=True)
    args = parser.parse_args(argv)
    if args.command == "register":
        result = register(args.output, **json.loads(Path(args.spec).read_text(encoding="utf-8")))
    elif args.command == "register-bridge":
        result = register_prefix_bridge(args.output, reference=args.reference)
    elif args.command == "analyze-bridge":
        result = analyze_prefix_bridge(args.bridge_registration)
    elif args.command == "run":
        result = run_one(args.registration, args.job_id)
    else:
        result = analyze(args.registration)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
