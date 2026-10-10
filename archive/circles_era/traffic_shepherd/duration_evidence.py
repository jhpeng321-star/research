"""Offline evidence for one registered HOLD+5 continuation decision per seed."""
from collections import Counter
import gzip
import hashlib
import importlib
import json
from pathlib import Path

from traffic_shepherd import paired_experiments as paired
from traffic_shepherd import cohort_evidence as cohort
from traffic_shepherd.brake_hold_study import command_contract, rows, source_bridge
from traffic_shepherd.damping import CirclesController
from traffic_shepherd.dissolution_study import reference_proof
from traffic_shepherd.escalation_evidence import BenefitCriteria, benefit_label, outcome_costs, paired_effect
from traffic_shepherd.formation import make_proposal
from traffic_shepherd.lifecycle_study import joint_non_deterioration, score_lifecycle
from traffic_shepherd.lifecycle_wall import LifecycleProfile, policy_factory as lifecycle_factory, remaining_plan
from traffic_shepherd.ownership import CommandOwnership
from traffic_shepherd.reporting import RSUReportBuffer
from traffic_shepherd.workflow import sha

SCHEMA = "duration_continuation_evidence/1"
OFFSET_S = 5


def _duration():
    return importlib.import_module("traffic_shepherd.duration_study")


def source_policy(job, registration):
    if "cohort_selection" in job["policy_spec"]:
        return importlib.import_module("traffic_shepherd.cohort_study").policy_factory(job, registration)
    return lifecycle_factory(job, registration)


def contexts(path, communication, controller):
    """Current acquired frames only; live command mask never enters PolicyInput."""
    if communication["delay_s"] != 0.:
        raise ValueError("registered zero-delay acquisition required")
    buffer = RSUReportBuffer(max_age_s=communication["max_report_age_s"], history_s=communication["history_s"])
    history = []
    for tick, packets, commands in cohort._telemetry_ticks(path, 2400):
        now = float(tick)
        frames = {p.frame.vehicle_id: p.frame for p in packets}
        live = {r["vehicle_id"] for r in commands}
        if len(frames) != len(packets) or len(live) != len(commands) or not set(frames) <= live:
            raise ValueError("duplicate acquired/command record or invalid current mask")
        for packet in packets:
            buffer.receive(packet, received_at_s=now)
            history.append((now, packet))
        history = [(receipt, packet) for receipt, packet in history if receipt >= now-communication["history_s"]]
        decisions = {v: controller.decide(f, now_s=now) for v, f in frames.items()}
        numeric = {v: d.command_acceleration_mps2 for v, d in decisions.items() if d.fallback_reason is None}
        native = frozenset(v for v, d in decisions.items() if d.fallback_reason is not None)
        context = paired.PolicyInput(now, tuple(buffer.reports(now_s=now)), tuple(history), numeric, native)
        yield context, frames, live, commands, packets


def acquired(context):
    return dict(now_s=context.now_s, reports=[r.payload() for r in context.reports],
                history=[dict(received_at_s=t, **r.payload()) for t, r in context.history])


def _hash_rows(values):
    value, count = hashlib.sha256(), 0
    for row in values:
        value.update(paired.digest(row).encode())
        count += 1
    return dict(sha256=value.hexdigest(), count=count)


def branch_prefix(path, branch_s):
    """Measured-kinematic prefix; no fabricated route-index or private RNG state."""
    path = Path(path)
    count, value, branch, header = 0, hashlib.sha256(), None, None
    with gzip.open(path / "interactions.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["kind"] == "header":
                if header is not None or row.get("sample_timing") != "initial_then_post_physics_before_next_control_or_shutdown":
                    raise ValueError("unsupported diagnostic sample timing")
                header = row
            elif row["kind"] == "sample":
                if header is None or row["time_s"] != count*.5:
                    raise ValueError("noncontiguous offline measured prefix")
                if row["time_s"] > branch_s:
                    break
                value.update(paired.digest(row).encode())
                count += 1
                if row["time_s"] == branch_s:
                    branch = row
    if branch is None or count != 2*branch_s+1:
        raise ValueError("exact pre-control branch kinematics unavailable")
    report_rows, command_rows = [], []
    for tick, packets, commands in cohort._telemetry_ticks(path, 2400):
        if tick > branch_s:
            break
        report_rows.append(dict(now_s=float(tick), received_packets=[dict(received_at_s=float(tick),
                                                                          **p.payload()) for p in packets]))
        if tick < branch_s:
            command_rows.extend(commands)
    intervals = _hash_rows(r for r in rows(path / "intervals.jsonl") if r["end_s"] <= branch_s)
    return dict(schema="offline_measured_kinematics_branch/1", branch_time_s=branch_s,
        diagnostic_sample_prefix_sha256=value.hexdigest(), diagnostic_sample_count=count,
        offline_measured_kinematics_branch_sha256=paired.digest(branch),
        diagnostic_header_sha256=paired.digest(header), acquired_reports_inclusive_branch=_hash_rows(report_rows),
        issued_commands_strictly_before_branch=_hash_rows(command_rows), outcome_intervals_before_branch=intervals,
        private_simulator_state_reconstructed=False, original_full_measured_state_reconstructed=False,
        branch_sample_timing="post-physics/pre-next-control", online_input=False)


def opportunity(path, job, registration):
    """Eligibility depends on delivered HOLD and acquired feasibility, never benefit."""
    path = Path(path)
    life = score_lifecycle(path, job)
    hold = life.get("committed_hold_at_s")
    if hold is None or not life.get("assembly_complete"):
        return dict(eligible=False, reason="no_verified_committed_hold", lifecycle=life)
    if not life.get("registered_maneuver_delivered"):
        return dict(eligible=False, reason="source_maneuver_not_delivered", lifecycle=life)
    branch_s = hold+OFFSET_S
    policies = {r["time_s"]: r for r in rows(path / "policy.jsonl")}
    row, prior = policies.get(branch_s), policies.get(branch_s-1)
    if row is None or row["phase"] != "hold" or prior is None:
        return dict(eligible=False, reason="hold_not_active_at_fixed_offset", branch_time_s=branch_s, lifecycle=life)
    context = next(context for context, *_ in contexts(path, registration["communication"], CirclesController())
                   if context.now_s == branch_s)
    snapshot = cohort._snapshot(path)
    from traffic_shepherd.escalation_observations import report_from_payload
    reports = tuple(report_from_payload(p) for p in snapshot["evidence"]["reports"])
    profile = LifecycleProfile(**job["policy_spec"]["profile"])
    proposal = make_proposal(reports, now_s=float(job["decision_time_s"]),
        member_ids=tuple(job["policy_spec"]["member_ids"]),
        required_lanes=tuple(job["policy_spec"]["required_lanes"]), profile=profile)
    if proposal.fingerprint != row["proposal_sha256"]:
        raise ValueError("acquired source proposal reconstruction changed")
    commit = remaining_plan(proposal, context.reports, now_s=branch_s,
        previous_commands=prior["wall_commands"], readiness_count=row["readiness_count"], ready_at_s=hold)
    exit_now = remaining_plan(proposal, context.reports, now_s=branch_s,
        previous_commands=prior["wall_commands"], readiness_count=row["readiness_count"], ready_at_s=hold, exit_now=True)
    cohort._same(commit, row["lifecycle_plan"]["commitment"], "saved branch continuation differs from acquired planner")
    eligible = bool(commit["feasible"] and exit_now["feasible"])
    spec = _duration().make_branch_spec(context=context, expected_hold_start_s=hold,
        previous_wall_commands=prior["wall_commands"], proposal_sha256=proposal.fingerprint,
        source_manifest_sha256=sha(path / "manifest.json"))
    return dict(eligible=eligible, reason="current_hold_and_both_acquired_plans_feasible" if eligible else "branch_plan_infeasible",
        expected_hold_start_s=hold, branch_time_s=branch_s, hold_offset_s=OFFSET_S,
        branch_evidence_sha256=paired.digest(acquired(context)),
        previous_wall_commands_sha256=paired.digest(prior["wall_commands"]),
        proposal_sha256=proposal.fingerprint, source_manifest_sha256=sha(path / "manifest.json"),
        commitment=commit, exit_now=exit_now, lifecycle=life,
        continued_to_nominal_end=bool(life.get("fixed_maximum_hold_maneuver_delivered")),
        finite_policy_triggered_after_branch=any(r.get("early_exit_reason") for t, r in policies.items() if t >= branch_s),
        acquired_only_eligibility=True, branch_spec=spec, branch_prefix=branch_prefix(path, branch_s))


def continuation_proof(path, source_job, source_registration, identity, *, release_job=None, release_registration=None):
    """Every source action and actual owner, plus optional unaltered release prefix."""
    path = Path(path).resolve()
    disk = cohort._registration(path.parent.parent / "registration.json")
    cohort._same(disk, source_registration, "continuation source registration changed")
    manifest, _ = cohort._registered_arm(path, source_job, disk)
    bridge = source_bridge(manifest, identity)
    base = source_policy(source_job, disk)
    branch_policy = _duration().policy_factory(release_job, release_registration) if release_job else None
    branch_s = release_job["policy_spec"]["duration_branch"]["branch_time_s"] if release_job else None
    controller, api = CirclesController(), cohort._ReplayAPI()
    cavs = {r["vehicle_id"] for r in rows(path / "telemetry.jsonl") if r["kind"] == "command"}
    owner = CommandOwnership(api, cavs, controller)
    release_api = cohort._ReplayAPI()
    release_owner = CommandOwnership(release_api, cavs, controller)
    policies = iter(rows(path / "policy.jsonl"))
    counts, previous, terminal, branch_output = Counter(), {}, None, None
    for context, frames, live, saved_commands, packets in contexts(path, disk["communication"], controller):
        now = context.now_s
        saved = next(policies, None)
        if saved is None or saved["time_s"] != now:
            raise ValueError("continuation policy clock incomplete")
        for packet in packets:
            prior = previous.get(packet.frame.vehicle_id)
            expected = dict(owner="native" if prior is None else prior["owner"],
                network_output_mps2=None if prior is None else prior["network_output_mps2"],
                issued_acceleration_mps2=None if prior is None else prior["issued_acceleration_mps2"],
                fallback_reason=None if prior is None else prior["fallback_reason"])
            cohort._same({k: getattr(packet, k) for k in expected}, expected, "acquired prior ownership mismatch")
        output = base(context)
        core = {k: v for k, v in saved.items() if k not in {"time_s", "assignment"}}
        cohort._same(output.payload(), core, f"continuation policy mismatch at {now}")
        api.vehicle.calls.clear()
        actual = [cohort._json(r.as_dict()) for r in owner.step(frames, now_s=now, live_ids=live,
                                                              wall_commands=output.wall_commands)]
        cohort._same(actual, saved_commands, f"continuation actual command mismatch at {now}")
        expected_calls = []
        for r in actual:
            if r["issued_acceleration_mps2"] is not None:
                expected_calls.append(["slowDown", r["vehicle_id"], r["issued_target_speed_mps"], 1.])
            elif r["released_to_native"]:
                expected_calls.append(["setSpeed", r["vehicle_id"], -1])
            counts["actual_native_release_writes"] += int(r["released_to_native"])
        cohort._same(api.vehicle.calls, expected_calls, "actual continuation release writer mismatch")
        if branch_policy is not None and now <= branch_s:
            released = branch_policy(context)
            if now < branch_s:
                cohort._same({k: v for k, v in released.payload().items() if k != "duration_branch"}, core,
                             f"release policy changes prebranch continuation at {now}")
                release_api.vehicle.calls.clear()
                commands = [cohort._json(r.as_dict()) for r in release_owner.step(frames, now_s=now,
                    live_ids=live, wall_commands=released.wall_commands)]
                cohort._same(commands, saved_commands, "release prebranch actual command differs")
                cohort._same(release_api.vehicle.calls, expected_calls, "release prebranch native writer differs")
                counts["release_prefix_decisions"] += 1
            else:
                branch_output = released.payload()
                if released.phase not in {"taper", "release"}:
                    raise ValueError("registered branch did not invoke finite exit")
        previous = {r["vehicle_id"]: r for r in actual}
        counts["decision_count"] += 1
        counts["command_records"] += len(actual)
        counts["missing_observation_live_decisions"] += len(live-set(frames))
        if terminal is None and output.phase in {"release", "refused", "abort"}:
            terminal = dict(time_s=now, phase=output.phase, reason=output.reason)
    if next(policies, None) is not None or counts["decision_count"] != 2400 or terminal is None:
        raise ValueError("full 2400-source horizon and terminal required")
    if branch_policy is not None and (branch_output is None or counts["release_prefix_decisions"] != branch_s):
        raise ValueError("complete strictly prebranch release replay required")
    return dict(schema=SCHEMA, complete=True, **counts, first_decision_s=0., last_decision_s=2399.,
        source_path=str(path), source_job_sha256=paired.digest(source_job),
        source_manifest_sha256=sha(path / "manifest.json"), source_registration_sha256=disk["registration_sha256"],
        target_source_sha256=identity["active_source_sha256"], replay_implementation_sha256=sha(Path(__file__)),
        source_bridge=bridge, every_core_policy_field_equal=True, every_actual_command_record_field_equal=True,
        terminal=terminal, release_prefix_exact=branch_policy is not None, branch_output=branch_output,
        branch_time_s=branch_s, no_simulator_rollout=True, simulations_started=0,
        release_job_sha256=None if release_job is None else paired.digest(release_job),
        source_trace_is_not_a_new_replication=True)


def validate_branch_output(row, bound):
    _duration().validate_branch_spec(bound)
    proof = row.get("duration_branch", {})
    expected = dict(version=bound["version"], hold_offset_s=OFFSET_S,
        registered_hold_start_s=bound["expected_hold_start_s"], registered_branch_time_s=bound["branch_time_s"],
        original_profile_hold_s=20., deliberate_release=True, feasibility_triggered=False,
        requested=True, verified=True, request_time_s=bound["branch_time_s"],
        actual_hold_start_s=bound["expected_hold_start_s"], exit_started_at_s=bound["branch_time_s"],
        continuation_feasible=True, exit_now_feasible=True,
        proposal_sha256=bound["proposal_sha256"], branch_evidence_sha256=bound["branch_evidence_sha256"],
        previous_wall_commands_sha256=bound["previous_wall_commands_sha256"],
        source_manifest_sha256=bound["source_manifest_sha256"])
    if any(proof.get(k) != v for k, v in expected.items()):
        return False
    if (row.get("time_s") != bound["branch_time_s"] or row.get("phase") not in {"taper", "release"}
            or row.get("early_exit_reason") is not None or row.get("proposal_sha256") != bound["proposal_sha256"]
            or paired.digest(proof.get("previous_wall_commands")) != bound["previous_wall_commands_sha256"]):
        return False
    continuation, exit_now = proof.get("continuation_plan", {}), proof.get("exit_now_plan", {})
    return bool(continuation.get("feasible") is True and exit_now.get("feasible") is True
        and continuation.get("remaining_hold_s") == 15 and exit_now.get("remaining_hold_s") == 0
        and continuation.get("remaining_assembly_s") == 0 and exit_now.get("remaining_assembly_s") == 0)


def score_release(path, job):
    """Deliberate H+5 release is not a fabricated finite-constraint trigger."""
    base = score_lifecycle(path, job)
    bound = job["policy_spec"]["duration_branch"]
    policy = list(rows(Path(path) / "policy.jsonl"))
    at = next((r for r in policy if r["time_s"] == bound["branch_time_s"]), {})
    proof = at.get("duration_branch", {})
    verified = validate_branch_output(at, bound)
    prefix_hold = base.get("committed_hold_at_s") == bound["expected_hold_start_s"]
    deliberate = bool(verified and prefix_hold and base.get("assembly_complete")
        and base.get("valid_sampled_hold") and base.get("controlled_exit_verified")
        and base.get("hold_seconds") == OFFSET_S and not at.get("early_exit_reason"))
    return dict(base, registered_maneuver_delivered=deliberate,
        duration_deliberate_release_verified=bool(verified), intentional_hold_offset_s=OFFSET_S,
        original_full_nominal_hold_required=False, finite_policy_early_exit=False,
        fabricated_dynamic_constraint_reason=False, branch_diagnostics=proof)


def compare_case(root, data, job):
    root = Path(root).resolve()
    path = root / "jobs" / job["id"]
    tm, ts = cohort._registered_arm(path, job, data)
    source = data["continuations"][job["id"]]
    sp = Path(source["path"])
    registration = cohort._registration(source["registration_path"])
    sm, ss = cohort._registered_arm(sp, source["job"], registration)
    if sha(sp / "manifest.json") != source["manifest_sha256"]:
        raise ValueError("registered continuation physical manifest changed")
    cohort._same(source_bridge(sm, data["identity"]), source["source_bridge"], "continuation source bridge changed")
    if data["criteria"] != registration["criteria"] or data["communication"] != registration["communication"]:
        raise ValueError("duration criteria or communication changed")
    branch_s = job["policy_spec"]["duration_branch"]["branch_time_s"]
    before, after = branch_prefix(sp, branch_s), branch_prefix(path, branch_s)
    cohort._same(before, source["opportunity"]["branch_prefix"], "bound source branch prefix changed")
    cohort._same(before, after, "actual full-traffic/report/command prebranch histories differ")
    cohort._prefix(cohort._snapshot(sp), cohort._snapshot(path))
    if sm["proposal_sha256"] != tm["proposal_sha256"] or sm["exogenous_future_sha256"] != tm["exogenous_future_sha256"]:
        raise ValueError("continuation/release proposal or requested future differs")
    for key in ("seed", "mainline_vph", "penetration", "duration_s", "scenario_identity"):
        if sm[key] != tm[key]:
            raise ValueError("duration physical scenario differs: " + key)
    if sm["controller"] != "wall" or tm["controller"] != "wall":
        raise ValueError("duration contrast requires two genuine wall manifests")
    if ss["population"] != ts["population"]:
        raise ValueError("duration requested populations differ")
    life = score_release(path, job)
    if not life["duration_deliberate_release_verified"]:
        raise ValueError("actual registered duration branch assignment failed")
    contract = command_contract(path, LifecycleProfile(**job["policy_spec"]["profile"]))
    if not contract["passed"]:
        raise ValueError("actual finite-exit command contract failed")
    reference = data["references"][job["id"]]
    dp = Path(reference["path"])
    dm, ds, dproof = reference_proof(dp, data["identity"])
    if dproof["manifest_sha256"] != reference["manifest_sha256"]:
        raise ValueError("true damping binding changed")
    cohort._prefix(cohort._snapshot(dp), cohort._snapshot(path))
    if dm["exogenous_future_sha256"] != tm["exogenous_future_sha256"]:
        raise ValueError("damping requested future differs")
    identity = data["identity"]["active_source_sha256"]
    dv = dict(dm, active_source_sha256=identity, proposal_sha256=tm["proposal_sha256"],
              predecision_state_sha256=tm["predecision_state_sha256"], decision_time_s=job["decision_time_s"])
    effect = paired_effect(dv, ds, dict(tm, active_source_sha256=identity), ts)
    continuation_effect = paired_effect(dv, ds, dict(sm, active_source_sha256=identity), ss)
    left, right = outcome_costs(ts), outcome_costs(ss)
    criteria = BenefitCriteria(**data["criteria"])
    absolute = {label:dict(population=s["population"], unfinished_requested=s["unfinished_requested"],
        collision_involvements=s["totals"]["collision_involvements"], starting_teleports=s["totals"]["starting_teleports"])
        for label, s in (("release", ts), ("continuation", ss), ("damping", ds))}
    return dict(schema=SCHEMA, job_id=job["id"], seed=job["seed"], status="complete",
        effect=effect, benefit=benefit_label(effect, criteria),
        joint_non_deterioration=joint_non_deterioration(effect, criteria), lifecycle=life, command_contract=contract,
        duration_contrast=dict(cost_release_minus_cost_continue={k:left[k]-right[k] for k in left},
            positive_means="continuing_coordination_reduced_this_cost", costs_release=left, costs_continue=right,
            release_controller=tm["controller"], continuation_controller=sm["controller"],
            wall_was_not_relabelled_as_damping=True, absolute_outcomes=absolute,
            common_horizon_s=2400, prebranch_costs_retained_in_absolute_totals=True),
        continuation=dict(path=str(sp), manifest_sha256=sha(sp / "manifest.json"), effect=continuation_effect,
            benefit=benefit_label(continuation_effect, criteria),
            joint_non_deterioration=joint_non_deterioration(continuation_effect, criteria),
            lifecycle=score_lifecycle(sp, source["job"]),
            nominal_and_finite_are_same_saved_trace=source["opportunity"]["continued_to_nominal_end"]),
        branch_prefix=after, reference_proof=dproof, physical_manifest_sha256=sha(path / "manifest.json"),
        new_simulated_seconds=2400, original_criteria_unchanged=True, changed_existing_manifests=False,
        optimal_duration_established=False, qualified_gate=False)
