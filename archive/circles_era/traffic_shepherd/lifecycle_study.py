"""Two registered full-episode checks of a finite acquired-report wall lifecycle."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from . import paired_experiments as paired
from .brake_hold_study import command_contract, lifecycle, rows, source_bridge
from .config import ROOT
from .dissolution_study import reference_proof, REFERENCES
from .escalation_evidence import BenefitCriteria, benefit_label, paired_effect
from .formation_evidence import _geometry
from .workflow import sha, write_json


SCHEMA = "finite_lifecycle_study/2"
PROTOCOL = "docs/LIFECYCLE_FIXTURE_INSERTION_CHECK.md"
OLD = ROOT / "runs/active/gentle_wall_v1_logging_repair"
PART_A = ROOT / "output/lifecycle_repair_v1/PART_A.json"


def verify_part_a(identity):
    part = json.loads(PART_A.read_text())
    if (part.get("decision") != "retain_original_network_and_current_control_region_for_part_B"
            or part.get("extension_required_for_part_B") is not False
            or part.get("simulations_started") != 0
            or part.get("source_sha256", {}).get("D:/research/highway_merge_shepherd_long/osm.net.xml.gz")
            != identity["network_sha256"]):
        raise ValueError("Part A does not authorize this unchanged-network check")
    return part


def jobs():
    from .lifecycle_wall import LifecycleProfile, VERSION
    original = json.loads((OLD / "registration.json").read_text())
    result = []
    for seed in (4107, 4108):
        job = deepcopy(next(j for j in original["jobs"] if j["seed"] == seed))
        job["id"] = f"s{seed}_m2700_p20_t600_lifecycle_v2_wall"
        job["policy_spec"]["profile"].update(version=VERSION)
        LifecycleProfile(**job["policy_spec"]["profile"])
        paired.validate_job(job)
        result.append(job)
    return result


def register(output, mechanics_registration):
    from .lifecycle_mechanics import verify_completed
    from .lifecycle_wall import VERSION
    mechanics = verify_completed(mechanics_registration)
    if not PART_A.exists():
        raise ValueError("Part A scenario decision required before registration")
    identity = paired.current_identity()
    verify_part_a(identity)
    original = json.loads((OLD / "registration.json").read_text())
    all_jobs = jobs()
    references = {j["id"]: reference_proof(REFERENCES / "jobs" / j["reference_job_id"], identity)[2]
                  for j in all_jobs}
    data = dict(schema=SCHEMA, registered_at_utc=datetime.now(timezone.utc).isoformat(),
        identity=identity, jobs=all_jobs, references=references,
        communication=original["communication"], criteria=original["criteria"],
        assignment="registered_experimental_assignment_not_trained_benefit_gate",
        interaction_diagnostics=True, policy_version=VERSION,
        maximum_attempts=2, maximum_simulated_seconds=4800, automatic_retries=0,
        automatic_successors=False, mechanics_evidence=mechanics,
        part_a_path=str(PART_A), part_a_sha256=sha(PART_A),
        protocol_path=PROTOCOL, protocol_sha256=sha(ROOT / PROTOCOL))
    data["registration_sha256"] = paired.digest(data)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for name in identity["active_source_sha256"]:
        shutil.copy2(ROOT / "traffic_shepherd" / name, snapshot / name)
    write_json(output / "registration.json", data)
    return dict(registration=str(output / "registration.json"), jobs=[j["id"] for j in all_jobs],
        maximum_simulated_seconds=4800, simulations_started=0, reused_damping_references=2)


def read_registration(path, *, require_current=True):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if (data.get("schema") != SCHEMA or data.get("registration_sha256") != paired.digest(
            {k: v for k, v in data.items() if k != "registration_sha256"})
            or data.get("jobs") != jobs() or data.get("maximum_attempts") != 2
            or data.get("maximum_simulated_seconds") != 4800 or data.get("automatic_retries") != 0
            or data.get("automatic_successors") is not False):
        raise ValueError("finite lifecycle registration or exact budget changed")
    if require_current and (data["identity"] != paired.current_identity()
            or data["protocol_sha256"] != sha(ROOT / PROTOCOL)):
        raise ValueError("source or protocol changed after registration")
    if sha(data["part_a_path"]) != data["part_a_sha256"]:
        raise ValueError("registered scenario decision changed")
    if Path(data["part_a_path"]).resolve() != PART_A.resolve():
        raise ValueError("registered scenario evidence path changed")
    verify_part_a(data["identity"])
    original = json.loads((OLD / "registration.json").read_text())
    if (data["criteria"] != original["criteria"] or data["communication"] != original["communication"]
            or set(data["references"]) != {j["id"] for j in data["jobs"]}
            or data["assignment"] != "registered_experimental_assignment_not_trained_benefit_gate"
            or data["interaction_diagnostics"] is not True):
        raise ValueError("registered scientific criteria, communication or reference scope changed")
    for job in data["jobs"]:
        if Path(data["references"][job["id"]]["path"]).resolve() != (REFERENCES / "jobs" / job["reference_job_id"]).resolve():
            raise ValueError("registered reference path changed")
    for name, expected in data["identity"]["active_source_sha256"].items():
        if sha(path.parent / "source_snapshot" / name) != expected:
            raise ValueError("registered source snapshot changed")
    return path.parent, data


def run_one(registration, job_id):
    from .lifecycle_wall import LifecycleProfile, policy_factory
    root, data = read_registration(registration)
    job = next((j for j in data["jobs"] if j["id"] == job_id), None)
    if job is None or (root / "STOP.json").exists():
        raise ValueError("unknown job or stopped lifecycle study")
    for proof in data["references"].values():
        if sha(Path(proof["path"]) / "manifest.json") != proof["manifest_sha256"]:
            raise ValueError("registered reference changed")
    (root / "allocations").mkdir(exist_ok=True)
    with (root / "allocations" / f"{job_id}.json").open("x") as stream:
        json.dump(dict(job_id=job_id, allocated_simulated_seconds=2400, retries=0,
            registration_sha256=data["registration_sha256"]), stream)
    try:
        path = root / "jobs" / job_id
        result = paired.evaluate_arm(job=job, registration=data, output=path, policy=policy_factory(job, data))
        contract = command_contract(path, LifecycleProfile(**job["policy_spec"]["profile"]))
        write_json(path / "lifecycle_command_contract.json", contract)
        if not contract["passed"]:
            raise ValueError("actual command contract failed")
        return result
    except Exception as exc:
        write_json(root / "STOP.json", dict(job_id=job_id, reason=str(exc), automatic_retries=0))
        raise


def joint_non_deterioration(effect, criteria):
    """Additional descriptive question; never changes the original benefit label."""
    nonworse = {k: v >= -1e-9 for k, v in effect["improvements"].items()}
    absolute = (effect["complete_requested_service_control"] and effect["complete_requested_service_treatment"]
                and effect["treatment_collision_involvements"] == 0 and effect["treatment_starting_teleports"] == 0)
    original = benefit_label(effect, criteria)
    material = bool(original["material_improvements"])
    return dict(nonworsening_by_metric=nonworse, all_continuous_costs_nonworsening=all(nonworse.values()),
        absolute_safety_and_service=absolute, material_primary_improvement=material,
        joint_material_success=bool(all(nonworse.values()) and absolute and material),
        original_label_unchanged=True, comparison_tolerance=1e-9)


def score_lifecycle(path, job):
    """Distinguish planned-max HOLD, valid early exit, refusal and failed assembly."""
    base = lifecycle(path, job)
    policy = list(rows(Path(path) / "policy.jsonl"))
    terminal = next(r for r in policy if r["phase"] in {"release", "abort", "refused"})
    time = terminal["time_s"]
    samples = [r for r in rows(Path(path) / "formation.jsonl") if r["time_s"] <= time]
    profile = job["policy_spec"]["profile"]
    assigned = dict(zip(job["policy_spec"]["member_ids"], job["policy_spec"]["required_lanes"]))
    target = samples[0]["target_speed_mps"] if samples else None
    indexed = {r["time_s"]: r for r in samples}
    hold = [r for r in samples if r["phase"] == "hold"]
    first_hold = hold[0]["time_s"] if hold else None
    streak = 0
    for row in samples:
        if first_hold is None or row["time_s"] > first_hold:
            break
        if float(row["time_s"]).is_integer():
            streak = streak+1 if _geometry(row, assigned, profile, target)["settled"] else 0
    entry = next((r for r in policy if r["time_s"] == first_hold), {})
    entry_plan = entry.get("lifecycle_plan", {}).get("commitment", {})
    first_transition = next((r for r in samples if r["phase"] != "assembly"), {})
    entry_valid = bool(first_hold is not None and streak >= profile["readiness_samples"]
        and first_transition.get("phase") == "hold" and first_transition.get("time_s") == first_hold
        and first_hold-job["decision_time_s"] <= profile["assembly_deadline_s"]
        and entry.get("assembly_completed_at_s") == first_hold
        and entry_plan.get("feasible") is True and entry_plan.get("remaining_assembly_s") == 0
        and entry_plan.get("maintainable_hold_ready_now") is True
        and entry_plan.get("predicted_hold_start_s") == first_hold
        and entry_plan.get("remaining_hold_s") == profile["hold_s"])
    valid_hold = entry_valid
    for row in hold:
        endpoint = indexed.get(row["time_s"]+.5)
        valid_hold &= bool(endpoint and _geometry(row, assigned, profile, target)["compact"]
                           and _geometry(endpoint, assigned, profile, target)["compact"]
                           and all(row["owner_by_member"].get(v) == "wall" for v in assigned))
    if hold:
        valid_hold &= all(b["time_s"] == a["time_s"]+.5 for a, b in zip(hold, hold[1:]))
    last_commands = {r["vehicle_id"]: r for r in rows(Path(path) / "telemetry.jsonl")
                     if r["kind"] == "command" and r["timestamp_s"] == time-1
                     and r["vehicle_id"] in assigned}
    terminal_rows = {r["id"]: r for r in indexed.get(time, {}).get("members", [])}
    coast = bool(terminal["phase"] == "release" and all(
        v in last_commands and last_commands[v]["owner"] == "wall"
        and last_commands[v].get("issued_acceleration_mps2") is not None
        and abs(last_commands[v]["issued_acceleration_mps2"]) <= 1e-8
        and terminal_rows.get(v, {}).get("present")
        and terminal_rows[v].get("acceleration_mps2") is not None
        and abs(terminal_rows[v]["acceleration_mps2"]) <= .15+1e-8 for v in assigned))
    taper = next((r["time_s"] for r in policy if r["phase"] == "taper"), None)
    exit_duration = 0. if taper is None else time-taper
    early = 0 < base.get("hold_seconds", 0.) < profile["hold_s"]
    boundary = next((r for r in policy if r["time_s"] >= job["decision_time_s"]
                     and r["phase"] in {"taper", "release"}), None)
    early_proof = bool(boundary and boundary.get("early_exit_reason")
                       and boundary.get("lifecycle_plan", {}).get("commitment", {}).get("feasible") is False
                       and boundary.get("lifecycle_plan", {}).get("exit_now", {}).get("feasible") is True)
    handback = base.get("handback", {}).get("verified", False)
    controlled_exit = bool(coast and handback and exit_duration <= profile["taper_s"]
                           and not base.get("wall_reengaged_after_terminal", False))
    delivered = bool(valid_hold and controlled_exit and base["hold_seconds"] <= profile["hold_s"]
                     and (not early or early_proof))
    full_hold = bool(valid_hold and base["hold_seconds"] == profile["hold_s"])
    return dict(base, legacy_first_ready_entry_matches=base.get("phase_entry_matches_measured_readiness"),
        first_geometric_ready_at_s=base.get("measured_ready_at_s"),
        assembly_complete=entry_valid, committed_hold_at_s=first_hold,
        phase_entry_matches_measured_readiness=entry_valid,
        assembly_seconds=None if first_hold is None else first_hold-job["decision_time_s"],
        full_nominal_hold_delivered=full_hold,
        fixed_maximum_hold_maneuver_delivered=bool(full_hold and controlled_exit),
        registered_maneuver_delivered=delivered, measured_coast_terminal=coast,
        controlled_exit_verified=controlled_exit, measured_exit_seconds=exit_duration,
        valid_sampled_hold=valid_hold, early_exit_evidence_verified=early_proof,
        finite_policy_early_exit=bool(delivered and early),
        assembly_abandoned_before_readiness=bool(controlled_exit and not entry_valid),
        full_hold_is_not_required_for_valid_declared_early_exit=True)


def compare_job(root, data, job):
    from .lifecycle_wall import LifecycleProfile
    path = root / "jobs" / job["id"]
    tm, ts = paired.verified_arm(path)
    if tm["registration_sha256"] != data["registration_sha256"] or any(
            tm.get(k) != v for k, v in data["identity"].items()):
        raise ValueError("treatment source/registration mismatch")
    for key, field in {"job_id": "id", "case_id": "case_id", "seed": "seed", "mainline_vph": "loading",
                       "decision_time_s": "decision_time_s", "policy_spec": "policy_spec",
                       "duration_s": "duration_s", "controller": "arm"}.items():
        if tm.get(key) != job[field]:
            raise ValueError("treatment metadata differs from registered job")
    reference = Path(data["references"][job["id"]]["path"])
    cm, cs, proof = reference_proof(reference, data["identity"])
    if proof["manifest_sha256"] != data["references"][job["id"]]["manifest_sha256"]:
        raise ValueError("reference binding changed")
    left = json.loads((reference / "decision_snapshots.json").read_text())["600"]
    right = json.loads((path / "decision_snapshots.json").read_text())["600"]
    for key in ("decision_time_s", "predecision_state_sha256", "predecision_report_evidence_sha256",
                "predecision_report_prefix_sha256", "predecision_command_prefix_sha256"):
        if left[key] != right[key]:
            raise ValueError("matched prefix failed: " + key)
    if cm["exogenous_future_sha256"] != tm["exogenous_future_sha256"]:
        raise ValueError("matched requested future differs")
    view = dict(cm, active_source_sha256=data["identity"]["active_source_sha256"],
        proposal_sha256=right["proposal_sha256"], decision_time_s=600.,
        predecision_state_sha256=left["predecision_state_sha256"])
    effect = paired_effect(view, cs, tm, ts)
    criteria = BenefitCriteria(**data["criteria"])
    life = score_lifecycle(path, job)
    boundaries = [r for r in rows(path / "policy.jsonl") if r["phase"] != "damping"]
    terminal = next(r for r in boundaries if r["phase"] in {"release", "abort", "refused"})
    transitions = []
    for row in boundaries:
        if not transitions or transitions[-1]["phase"] != row["phase"]:
            transitions.append(dict(time_s=row["time_s"], phase=row["phase"], reason=row["reason"],
                lifecycle_plan=row.get("lifecycle_plan")))
    contract = command_contract(path, LifecycleProfile(**job["policy_spec"]["profile"]))
    old_path = OLD / "jobs" / f"s{job['seed']}_m2700_p20_t600_gentle_wall"
    om, os = paired.verified_arm(old_path)
    incumbent_bridge = source_bridge(om, data["identity"])
    old_prefix = json.loads((old_path / "decision_snapshots.json").read_text())["600"]
    for key in ("decision_time_s", "predecision_state_sha256", "predecision_report_evidence_sha256",
                "predecision_report_prefix_sha256", "predecision_command_prefix_sha256"):
        if old_prefix[key] != left[key]:
            raise ValueError("incumbent matched prefix failed: " + key)
    old_view = dict(om, active_source_sha256=data["identity"]["active_source_sha256"],
        proposal_sha256=right["proposal_sha256"])
    old_effect = paired_effect(view, cs, old_view, os)
    return dict(job_id=job["id"], seed=job["seed"], status="complete", effect=effect,
        benefit=benefit_label(effect, criteria), joint_non_deterioration=joint_non_deterioration(effect, criteria),
        lifecycle=life, phase_transitions=transitions, terminal_diagnostics=terminal.get("lifecycle_plan"),
        command_contract=contract, reference_proof=proof, predecision_prefix_exact=True,
        changed_existing_manifests=False, qualified_gate=False,
        incumbent_gentle=dict(path=str(old_path), manifest_sha256=sha(old_path / "manifest.json"),
            source_bridge=incumbent_bridge, predecision_prefix_exact=True,
            effect=old_effect, lifecycle_repair_minus_incumbent_cost_per_request={
                k: effect["costs_treatment"][k]-old_effect["costs_treatment"][k] for k in effect["costs_treatment"]}))


def analyze(registration):
    root, data = read_registration(registration, require_current=False)
    result, pending = [], []
    for job in data["jobs"]:
        path = root / "jobs" / job["id"] / "manifest.json"
        if not path.exists() or json.loads(path.read_text()).get("status") != "complete":
            pending.append(job["id"])
            continue
        result.append(compare_job(root, data, job))
    answer = dict(schema="finite_lifecycle_results/1", results=result, pending=pending,
        complete=len(result) == 2 and not pending, independent_seed_count=2,
        maximum_new_simulated_seconds=4800, original_criteria_unchanged=True,
        population_generalization_established=False)
    write_json(root / "analysis.json", answer)
    return dict(analysis_path=str(root / "analysis.json"), complete=answer["complete"],
        results=[dict(job_id=r["job_id"], benefit=r["benefit"], lifecycle=r["lifecycle"],
                      improvements=r["effect"]["improvements"]) for r in result], pending=pending)
