"""Offline Part C evidence: complete same-action reuse and immutable comparisons.

No simulator import, launch, state facade, or physical outcome enters a policy.
This staged module is activated only with the separately registered campaign.
"""
from collections import Counter
import importlib
import json
from math import isfinite
from pathlib import Path

from traffic_shepherd import paired_experiments as paired
from traffic_shepherd.brake_hold_study import command_contract, rows, source_bridge
from traffic_shepherd.damping import CirclesController
from traffic_shepherd.dissolution_study import reference_proof
from traffic_shepherd.escalation_evidence import BenefitCriteria, benefit_label, paired_effect
from traffic_shepherd.escalation_observations import report_from_payload
from traffic_shepherd.lifecycle_study import joint_non_deterioration, score_lifecycle
from traffic_shepherd.lifecycle_wall import LifecycleProfile, policy_factory as lifecycle_factory
from traffic_shepherd.ownership import CommandOwnership
from traffic_shepherd.reporting import RSUReportBuffer
from traffic_shepherd.workflow import sha

SCHEMA = "report_cohort_evidence/1"
SELECTOR_DETAILS = frozenset({"cohort_selector", "cohort_selection_verified",
    "cohort_selection_sha256", "cohort_selection"})
PREFIX_KEYS = ("decision_time_s", "predecision_state_sha256",
    "predecision_report_evidence_sha256", "predecision_report_prefix_sha256",
    "predecision_command_prefix_sha256")
JOB_FIELDS = {"job_id": "id", "case_id": "case_id", "seed": "seed",
    "mainline_vph": "loading", "penetration": "penetration", "role": "role",
    "decision_time_s": "decision_time_s", "policy_spec": "policy_spec",
    "duration_s": "duration_s", "controller": "arm", "reference_job_id": "reference_job_id"}


def _cohort():
    return importlib.import_module("traffic_shepherd.cohort_study")


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _same(left, right, message):
    if paired.digest(left) != paired.digest(right):
        raise ValueError(message)


def _registration(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("registration_sha256") != paired.digest({k: v for k, v in data.items()
                                                          if k != "registration_sha256"}):
        raise ValueError("source registration digest changed")
    for name, checksum in data["identity"]["active_source_sha256"].items():
        if sha(path.parent / "source_snapshot" / name) != checksum:
            raise ValueError("source registration snapshot changed: " + name)
    return data


def _registered_arm(path, job, registration):
    manifest, summary = paired.verified_arm(path)
    if manifest["registration_sha256"] != registration["registration_sha256"]:
        raise ValueError("physical arm registration changed")
    if job not in registration["jobs"]:
        raise ValueError("job does not belong to source registration")
    if any(manifest.get(k) != v for k, v in registration["identity"].items()):
        raise ValueError("physical arm source identity changed")
    for key, field in JOB_FIELDS.items():
        if manifest.get(key) != job.get(field):
            raise ValueError("physical arm differs from registered job: " + key)
    if (manifest["duration_s"] != 2400 or manifest["mechanics_only"] is not False
            or manifest["communication"] != registration["communication"]):
        raise ValueError("full-horizon physical/communication contract required")
    return manifest, summary


def _snapshot(path, now=600):
    return json.loads((Path(path) / "decision_snapshots.json").read_text())[str(now)]


def _prefix(left, right):
    for key in PREFIX_KEYS:
        if left[key] != right[key]:
            raise ValueError("physical/report/command predecision prefix changed: " + key)


class _ReplayVehicle:
    """Auditable no-op writer. It cannot read traffic state or launch SUMO."""
    def __init__(self):
        self.calls = []

    def getSpeedMode(self, vehicle_id):
        return 31

    def slowDown(self, vehicle_id, target, duration):
        self.calls.append(["slowDown", vehicle_id, target, duration])

    def setSpeed(self, vehicle_id, speed):
        self.calls.append(["setSpeed", vehicle_id, speed])


class _ReplayAPI:
    def __init__(self):
        self.vehicle = _ReplayVehicle()


def _telemetry_ticks(path, duration):
    """Read only acquired packets and executed command records, in log order."""
    tick, packets, commands = 0, [], []
    with (Path(path) / "telemetry.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if '"rsu_received_report"' not in line and '"command"' not in line:
                continue
            row = json.loads(line)
            if row["kind"] not in {"rsu_received_report", "command"}:
                continue
            now = row.get("received_at_s") if row["kind"] == "rsu_received_report" else row["timestamp_s"]
            if not isfinite(now) or now != int(now) or now < tick or not 0 <= now < duration:
                raise ValueError("invalid or reordered acquired/command clock")
            while tick < now:
                yield tick, packets, commands
                tick, packets, commands = tick+1, [], []
            if row["kind"] == "rsu_received_report":
                payload = {k: v for k, v in row.items() if k not in {"kind", "received_at_s"}}
                report = report_from_payload(payload)
                # This runner's immutable acquisition has zero delay and current frames.
                if report.sent_at_s != now or report.frame.timestamp_s != now:
                    raise ValueError("replay requires the registered current zero-delay acquisition")
                packets.append(report)
            else:
                commands.append({k: v for k, v in row.items() if k != "kind"})
    while tick < duration:
        yield tick, packets, commands
        tick, packets, commands = tick+1, [], []


def _replay_stream(path, source_policy, selected_policy, communication, *, duration=2400):
    """Actual pure policy + CommandOwnership replay; injectable policies aid unit tests."""
    if communication["delay_s"] != 0.:
        raise ValueError("zero-delay saved acquisition only")
    controller, api = CirclesController(), _ReplayAPI()
    # The command-origin live mask is an execution input only, never a policy feature.
    cav_ids = {row["vehicle_id"] for row in rows(Path(path) / "telemetry.jsonl")
               if row["kind"] == "command"}
    owner = CommandOwnership(api, cav_ids, controller)
    buffer = RSUReportBuffer(max_age_s=communication["max_report_age_s"], history_s=communication["history_s"])
    history, previous, counts = [], {}, Counter()
    saved_policy = iter(rows(Path(path) / "policy.jsonl"))
    terminal, selected_at_entry = None, None
    for tick, packets, saved_commands in _telemetry_ticks(path, duration):
        now = float(tick)
        saved = next(saved_policy, None)
        if saved is None or saved["time_s"] != now:
            raise ValueError("incomplete or noncontiguous saved policy clock")
        frames = {packet.frame.vehicle_id: packet.frame for packet in packets}
        live = {record["vehicle_id"] for record in saved_commands}
        if len(frames) != len(packets) or len(live) != len(saved_commands) or not set(frames) <= live:
            raise ValueError("duplicate packet/command or packet outside live CAV command mask")
        for packet in packets:
            prior = previous.get(packet.frame.vehicle_id)
            expected = dict(owner="native" if prior is None else prior["owner"],
                network_output_mps2=None if prior is None else prior["network_output_mps2"],
                issued_acceleration_mps2=None if prior is None else prior["issued_acceleration_mps2"],
                fallback_reason=None if prior is None else prior["fallback_reason"])
            _same({key: getattr(packet, key) for key in expected}, expected,
                  "received packet's prior ownership does not follow exact replay")
            buffer.receive(packet, received_at_s=now)
            history.append((now, packet))
        history = [(t, packet) for t, packet in history if t >= now-communication["history_s"]]
        damping = {vid: controller.decide(frame, now_s=now) for vid, frame in frames.items()}
        numeric = {vid: d.command_acceleration_mps2 for vid, d in damping.items() if d.fallback_reason is None}
        natives = frozenset(vid for vid, d in damping.items() if d.fallback_reason is not None)
        context = paired.PolicyInput(now, tuple(buffer.reports(now_s=now)), tuple(history), numeric, natives)
        original, selected = source_policy(context).payload(), selected_policy(context).payload()
        saved_core = {k: v for k, v in saved.items() if k not in {"time_s", "assignment"}}
        _same(original, saved_core, f"current preceding-B policy differs from physical policy at {now}")
        selected_core = {k: v for k, v in selected.items() if k not in SELECTOR_DETAILS}
        _same(selected_core, saved_core, f"selected policy core differs at {now}")
        if selected.get("cohort_selection") is not None:
            selected_at_entry = selected["cohort_selection"]
        api.vehicle.calls.clear()
        actual = [_json(record.as_dict()) for record in owner.step(frames, now_s=now,
            live_ids=live, wall_commands=selected["wall_commands"])]
        _same(actual, saved_commands, f"actual ownership/command record differs at {now}")
        expected_calls = []
        for record in actual:
            if record["issued_acceleration_mps2"] is not None:
                expected_calls.append(["slowDown", record["vehicle_id"], record["issued_target_speed_mps"],
                                       controller.config.control_dt_s])
            elif record["released_to_native"]:
                expected_calls.append(["setSpeed", record["vehicle_id"], -1])
            counts[record["owner"]+"_records"] += 1
            counts["actual_native_release_writes"] += int(record["released_to_native"])
        _same(api.vehicle.calls, expected_calls, f"native release/actuator write protocol differs at {now}")
        previous = {record["vehicle_id"]: record for record in actual}
        if terminal is None and selected["phase"] in {"release", "abort", "refused"}:
            terminal = dict(time_s=now, phase=selected["phase"], reason=selected["reason"])
        counts["decision_count"] += 1
        counts["command_records"] += len(actual)
        counts["received_packets"] += len(packets)
        counts["missing_observation_live_decisions"] += len(live-set(frames))
    if next(saved_policy, None) is not None or counts["decision_count"] != duration or terminal is None:
        raise ValueError("full policy horizon and terminal path required")
    return dict(counts, first_decision_s=0., last_decision_s=float(duration-1), terminal=terminal,
        selected_at_entry=selected_at_entry)


def replay_reuse(path, b_job, c_job, b_registration, c_registration):
    """Prove complete 4107 reuse before C registration, without a circular digest."""
    path = Path(path).resolve()
    disk_b = _registration(path.parent.parent / "registration.json")
    _same(disk_b, b_registration, "preceding B registration changed")
    bm, _ = _registered_arm(path, b_job, disk_b)
    bridge = source_bridge(bm, c_registration["identity"])
    if (b_job["seed"] != 4107 or c_job["seed"] != 4107
            or b_job["policy_spec"]["member_ids"] != c_job["policy_spec"]["member_ids"]
            or b_job["policy_spec"]["profile"] != c_job["policy_spec"]["profile"]
            or b_registration["communication"] != c_registration["communication"]):
        raise ValueError("only identical 4107 cohort/law/communication can reuse")
    for field in ("case_id", "seed", "loading", "penetration", "role", "arm", "duration_s",
                  "decision_time_s", "reference_job_id", "snapshot_times_s"):
        if b_job.get(field) != c_job.get(field):
            raise ValueError("reuse exogenous job field changed: " + field)
    selection = _cohort().snapshot_selection(_snapshot(path))
    _same(selection, c_registration["selections"][c_job["id"]], "registered selection differs from acquired B evidence")
    actual = _replay_stream(path, lifecycle_factory(b_job, b_registration),
        _cohort().policy_factory(c_job, c_registration), b_registration["communication"])
    _same(actual.pop("selected_at_entry"), selection, "runtime selection did not reproduce registered choice")
    proof = dict(schema=SCHEMA, source_path=str(path), source_job_id=b_job["id"],
        source_manifest_sha256=sha(path / "manifest.json"), source_registration_sha256=disk_b["registration_sha256"],
        source_registration_file_sha256=sha(path.parent.parent / "registration.json"),
        source_artifact_sha256={k: v for k, v in bm.items() if k.endswith("_sha256")}, source_bridge=bridge,
        selection_sha256=selection["selection_sha256"], target_job_sha256=paired.digest(c_job),
        target_profile_sha256=paired.digest(c_job["policy_spec"]["profile"]),
        target_source_sha256=c_registration["identity"]["active_source_sha256"],
        replay_implementation_sha256=sha(Path(__file__)), **actual,
        all_phases_and_members_equal=True, all_wall_commands_equal=True,
        all_core_policy_payload_fields_equal=True,
        all_actual_actuator_commands_and_native_release_flags_equal=True,
        actual_noop_writer_protocol_verified=True, no_simulator_rollout=True, complete=True,
        ignored_policy_fields=sorted(SELECTOR_DETAILS),
        command_record_comparison="every CommandRecord.as_dict field; exact canonical JSON equality",
        no_online_global_or_future_inputs=True, simulations_started=0,
        inference="same-action induction over unchanged deterministic execution and exogenous future; one physical trace")
    _cohort().validate_reuse_proof(proof, selection=selection,
        source_manifest_sha256=proof["source_manifest_sha256"], source_job=b_job)
    return proof


def _preceding(data, job):
    proof = data["preceding_b"][job["id"]]
    path = Path(proof["path"]).resolve()
    registration_path = Path(proof["registration_path"])
    if registration_path.resolve() != path.parent.parent / "registration.json":
        raise ValueError("preceding B registration path changed")
    registration = _registration(registration_path)
    if registration["registration_sha256"] != proof["registration_sha256"]:
        raise ValueError("preceding B registration binding changed")
    manifest, summary = _registered_arm(path, proof["job"], registration)
    for field, name in (("manifest_sha256", "manifest.json"), ("summary_sha256", "summary.json"),
                        ("snapshot_sha256", "decision_snapshots.json")):
        if sha(path / name) != proof[field]:
            raise ValueError("preceding B physical binding changed: " + field)
    _same(source_bridge(manifest, data["identity"]), proof["source_bridge"], "preceding B source bridge changed")
    return path, manifest, summary, registration, proof


def compare_case(root, data, job):
    """C versus frozen D and preceding B, preserving physical source provenance."""
    root = Path(root).resolve()
    if job not in data["jobs"]:
        raise ValueError("unregistered C job")
    bp, bm, bs, br, bproof = _preceding(data, job)
    if data["criteria"] != br["criteria"] or data["communication"] != br["communication"]:
        raise ValueError("C changed original physical criteria or communication")
    reference = data["references"][job["id"]]
    dp = Path(reference["path"]).resolve()
    dm, ds, dproof = reference_proof(dp, data["identity"])
    if dproof["manifest_sha256"] != reference["manifest_sha256"]:
        raise ValueError("damping reference binding changed")
    left, previous = _snapshot(dp), _snapshot(bp)
    _prefix(left, previous)
    _same(previous, data["bound_snapshots"][job["id"]], "bound B decision snapshot changed")
    selection = _cohort().snapshot_selection(previous)
    _same(selection, data["selections"][job["id"]], "C selection no longer follows acquired evidence")
    bound_job, _ = _cohort().bind_selected_job(bproof["job"], previous)
    _same(job, bound_job, "registered C job differs from acquired assignment")
    binding = data["bindings"][job["id"]]
    if (binding["source_job_id"] != bproof["job"]["id"]
            or binding["source_manifest_sha256"] != bproof["manifest_sha256"]):
        raise ValueError("C binding preceding B source changed")
    if binding["mode"] == "verified_reuse":
        if Path(binding["path"]).resolve() != bp:
            raise ValueError("reuse path is not preceding B physical source")
        replay = replay_reuse(bp, bproof["job"], job, br, data)
        _same(replay, binding["reuse_proof"], "full acquired/ownership replay proof changed")
        path, tm, ts = bp, bm, bs
    elif binding["mode"] == "simulate":
        path = root / "jobs" / job["id"]
        if Path(binding["path"]).resolve() != path:
            raise ValueError("new physical path differs from exact registered allocation")
        tm, ts = _registered_arm(path, job, data)
        decision = next((r for r in rows(path / "policy.jsonl") if r["time_s"] == 600.), None)
        if (decision is None or decision.get("cohort_selection_verified") is not True
                or decision.get("cohort_selection_sha256") != selection["selection_sha256"]):
            raise ValueError("actual runtime acquired selection was not verified")
        _same(decision.get("cohort_selection"), selection, "runtime candidate inventory differs")
        replay = None
    else:
        raise ValueError("unknown C physical binding mode")
    right = _snapshot(path)
    _prefix(left, right)
    _prefix(previous, right)
    if right["proposal_sha256"] != selection["selected"]["proposal_sha256"]:
        raise ValueError("physical proposal differs from the acquired selected proposal")
    if len({m["exogenous_future_sha256"] for m in (dm, bm, tm)}) != 1:
        raise ValueError("D/B/C exogenous requested future differs")
    # Only now construct in-memory comparison views; physical manifests remain intact.
    identity = data["identity"]["active_source_sha256"]
    dview = dict(dm, active_source_sha256=identity, proposal_sha256=right["proposal_sha256"],
                 decision_time_s=job["decision_time_s"], predecision_state_sha256=left["predecision_state_sha256"])
    tview = dict(tm, active_source_sha256=identity)
    bview = dict(bm, active_source_sha256=identity)
    preceding_dview = dict(dview, proposal_sha256=previous["proposal_sha256"])
    effect = paired_effect(dview, ds, tview, ts)
    preceding_effect = paired_effect(preceding_dview, ds, bview, bs)
    delta = {key: effect["costs_treatment"][key]-preceding_effect["costs_treatment"][key]
             for key in effect["costs_treatment"]}
    if replay is not None and any(value != 0. for value in delta.values()):
        raise ValueError("reused physical trace cannot have nonzero C-minus-B costs")
    criteria = BenefitCriteria(**data["criteria"])
    contract = command_contract(path, LifecycleProfile(**job["policy_spec"]["profile"]))
    life = score_lifecycle(path, job)
    transitions = []
    for row in rows(path / "policy.jsonl"):
        if row["phase"] == "damping":
            continue
        if not transitions or row["phase"] != transitions[-1]["phase"]:
            transitions.append(dict(time_s=row["time_s"], phase=row["phase"], reason=row["reason"],
                                    lifecycle_plan=row.get("lifecycle_plan")))
    terminal = next((r for r in transitions if r["phase"] in {"release", "abort", "refused"}), None)
    if terminal is None or not contract["passed"]:
        raise ValueError("complete terminal and actual command contract required")
    return dict(schema=SCHEMA, job_id=job["id"], seed=job["seed"], status="complete",
        mode=binding["mode"], physical_source_path=str(path), physical_manifest_sha256=sha(path / "manifest.json"),
        new_simulated_seconds=0 if replay is not None else 2400, reused_trace_is_not_independent=True,
        effect=effect, benefit=benefit_label(effect, criteria),
        joint_non_deterioration=joint_non_deterioration(effect, criteria), lifecycle=life,
        command_contract=contract, phase_transitions=transitions,
        terminal_diagnostics=terminal["lifecycle_plan"], selection=selection, reference_proof=dproof,
        preceding_b=dict(path=str(bp), manifest_sha256=sha(bp / "manifest.json"), source_bridge=bproof["source_bridge"],
            effect=preceding_effect, cohort_minus_preceding_cost_per_request=delta,
            benefit=benefit_label(preceding_effect, criteria), lifecycle=score_lifecycle(bp, bproof["job"])),
        reuse_proof=replay, predecision_prefix_exact=True, exogenous_future_exact=True,
        derived_view_fields=["active_source_sha256", "proposal_sha256", "decision_time_s", "predecision_state_sha256"],
        changed_existing_manifests=False, qualified_gate=False, original_criteria_unchanged=True)
