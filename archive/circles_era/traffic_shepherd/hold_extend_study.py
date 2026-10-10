"""Registered bounded comparison: raise-26 HOLD extension 20 -> 30 seconds.

Two 2400 s corridor episodes branch at the scheduled coordination end
(measured HOLD start + 20 s = 632/638 s) of the verified raise-26 histories,
requesting exactly one +10 s extension before the original taper. The
raise-26 H20 arms are the primary same-prefix references; verified
observer-replay damping arms are the whole-policy references. Original
benefit criteria unchanged; strict-joint diagnostic separate. Budget:
4920 s (one mechanics fixture + two corridor runs). No retries, extra seeds,
further duration variants or successor campaign.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from . import paired_experiments as paired
from . import hold_extend_mechanics
from .brake_hold_study import command_contract, rows
from .config import ROOT
from .dissolution_study import REFERENCES, reference_proof
from .escalation_evidence import BenefitCriteria, paired_effect
from .hold_extend_mechanics import EXTENSION_S, verify_completed
from .hold_extend_wall import VERSION, policy_factory
from .lifecycle_wall import LifecycleProfile
from .target_speed_study import wall_vs_wall_contrast
from .workflow import sha, write_json

SCHEMA = "registered_hold_extend_campaign/1"
PROTOCOL = "docs/HOLD_EXTEND_CHECK.md"
SOURCE_REGISTRATION = ROOT / "runs/active/target_adjust_v3/registration.json"
SOURCE_JOBS = {
    4107: ("target_adjust_v3", "s4107_m2700_p20_t600_hold_adjust_up_wall"),
    4108: ("target_adjust_v3", "s4108_m2700_p20_t600_hold_adjust_up_wall"),
}
EXPECTED_HOLD_START_S = {4107: 612.0, 4108: 618.0}
EXTENSION_BRANCH_S = {4107: 632.0, 4108: 638.0}


def _measured_hold_start(seed):
    block, job = SOURCE_JOBS[seed]
    source = ROOT / "runs/active" / block / "jobs" / job
    with open(source / "formation.jsonl", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["phase"] == "hold":
                return float(row["time_s"])
    raise ValueError(f"source {seed} shows no measured HOLD")


def jobs():
    result = []
    for seed in (4107, 4108):
        block, job_id = SOURCE_JOBS[seed]
        source_root = ROOT / "runs/active" / block
        source = json.loads((source_root / "registration.json").read_text(encoding="utf-8"))
        base = deepcopy(next(j for j in source["jobs"] if j["id"] == job_id))
        hold_start = _measured_hold_start(seed)
        if hold_start != EXPECTED_HOLD_START_S[seed]:
            raise ValueError(f"seed {seed} source HOLD start {hold_start} differs")
        job = base
        job["id"] = f"s{seed}_m2700_p20_t600_hold26_ext10_wall"
        job["policy_spec"]["profile"] = asdict(LifecycleProfile())
        job["policy_spec"]["target_adjust_branch"] = dict(
            branch_mode="absolute", adjust_mps=2.0,
            expected_hold_start_s=hold_start, branch_time_s=hold_start + 5.0)
        job["policy_spec"]["hold_extension"] = dict(
            branch_mode="absolute", extension_s=EXTENSION_S,
            expected_hold_start_s=hold_start, branch_time_s=hold_start + 20.0)
        paired.validate_job(job)
        result.append(job)
    return result


def register(output, mechanics_registration):
    mechanics = verify_completed(mechanics_registration)
    if not mechanics["proof"]["passed"] or not all(mechanics["checks"].values()):
        raise ValueError("hold-extend mechanics fixture is not a completed passing check")
    identity = paired.current_identity()
    all_jobs = jobs()
    references, sources = {}, {}
    for job in all_jobs:
        seed = job["seed"]
        references[job["id"]] = reference_proof(
            REFERENCES / "jobs" / job["reference_job_id"], identity)[2]
        block, job_id = SOURCE_JOBS[seed]
        source_root = ROOT / "runs/active" / block
        sources[job["id"]] = dict(
            path=str(source_root / "jobs" / job_id),
            manifest_sha256=sha(source_root / "jobs" / job_id / "manifest.json"),
            registration_path=str(source_root / "registration.json"),
            registration_file_sha256=sha(source_root / "registration.json"),
            measured_hold_start_s=EXPECTED_HOLD_START_S[seed],
            extension_branch_s=EXTENSION_BRANCH_S[seed])
    original = json.loads(SOURCE_REGISTRATION.read_text(encoding="utf-8"))
    mechanics_path = Path(mechanics_registration).resolve() / "registration.json"
    data = dict(schema=SCHEMA, registered_at_utc=datetime.now(timezone.utc).isoformat(),
        identity=identity, jobs=all_jobs, references=references, sources=sources,
        extension_s=EXTENSION_S, criteria=original["criteria"],
        communication=original["communication"],
        assignment="registered_experimental_assignment_not_trained_benefit_gate",
        interaction_diagnostics=True, policy_version=VERSION,
        maximum_attempts=1, maximum_simulated_seconds=4800, automatic_retries=0,
        automatic_successors=False, duration_variants=False,
        new_mechanics_simulated_seconds=120.,
        budget=dict(campaign=hold_extend_mechanics.CAMPAIGN, category="corridor",
                    amount_s=2400., ledger_path=str(hold_extend_mechanics.LEDGER_PATH),
                    ceilings=hold_extend_mechanics.CEILINGS,
                    prerequisite=[hold_extend_mechanics.PREREQUISITE_KIND,
                                  str(mechanics_path)]),
        mechanics_evidence=mechanics, protocol_path=PROTOCOL,
        protocol_sha256=sha(ROOT / PROTOCOL))
    data["registration_sha256"] = paired.digest(data)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for name in identity["active_source_sha256"]:
        shutil.copy2(ROOT / "traffic_shepherd" / name, snapshot / name)
    write_json(output / "registration.json", data)
    return dict(registration=str(output / "registration.json"),
        jobs=[j["id"] for j in all_jobs], maximum_simulated_seconds=4800,
        simulations_started=0, reused_raise_references=2, reused_damping_references=2)


def read_registration(path, *, require_current=True):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if (data.get("schema") != SCHEMA or data.get("registration_sha256") != paired.digest(
            {k: v for k, v in data.items() if k != "registration_sha256"})
            or data.get("jobs") != jobs() or data.get("maximum_attempts") != 1
            or data.get("maximum_simulated_seconds") != 4800
            or data.get("automatic_retries") != 0
            or data.get("automatic_successors") is not False
            or data.get("duration_variants") is not False
            or data.get("extension_s") != EXTENSION_S):
        raise ValueError("hold-extend registration or exact budget changed")
    if require_current and (data["identity"] != paired.current_identity()
                            or data["protocol_sha256"] != sha(ROOT / PROTOCOL)):
        raise ValueError("source or protocol changed after registration")
    original = json.loads(SOURCE_REGISTRATION.read_text(encoding="utf-8"))
    if (data["criteria"] != original["criteria"]
            or data["communication"] != original["communication"]
            or set(data["references"]) != {j["id"] for j in data["jobs"]}
            or set(data["sources"]) != {j["id"] for j in data["jobs"]}
            or data["assignment"] != "registered_experimental_assignment_not_trained_benefit_gate"):
        raise ValueError("registered criteria, communication or reference scope changed")
    for job in data["jobs"]:
        seed = job["seed"]
        block, job_id = SOURCE_JOBS[seed]
        if Path(data["sources"][job["id"]]["path"]).resolve() != (
                ROOT / "runs/active" / block / "jobs" / job_id).resolve():
            raise ValueError("registered raise-26 reference path changed")
        if Path(data["references"][job["id"]]["path"]).resolve() != (
                REFERENCES / "jobs" / job["reference_job_id"]).resolve():
            raise ValueError("registered damping reference path changed")
    for name, expected in data["identity"]["active_source_sha256"].items():
        if sha(path.parent / "source_snapshot" / name) != expected:
            raise ValueError("registered source snapshot changed")
    return path.parent, data


def verify_shared_prefix(new_path, source_path, before_s, seed):
    from .target_speed_study import _prefix_rows
    from .target_speed_wall import predecision_identity_keys
    selector_bookkeeping = ("cohort_selection", "cohort_selection_sha256",
                            "cohort_selection_verified", "cohort_selector")
    mismatched = []

    def policy_rows(path):
        selected = []
        for line in open(path / "policy.jsonl", encoding="utf-8"):
            row = json.loads(line)
            if row["time_s"] < before_s:
                for key in selector_bookkeeping:
                    row.pop(key, None)
                selected.append(json.dumps(row, sort_keys=True, allow_nan=False))
        return selected

    if policy_rows(new_path) != policy_rows(source_path):
        mismatched.append("policy.jsonl")
    for name in ("intervals.jsonl", "traffic.jsonl", "telemetry.jsonl"):
        if _prefix_rows(new_path, name, before_s) != _prefix_rows(source_path, name, before_s):
            mismatched.append(name)
    left = json.loads((new_path / "decision_snapshots.json").read_text(encoding="utf-8"))["600"]
    right = json.loads((source_path / "decision_snapshots.json").read_text(encoding="utf-8"))["600"]
    for key in predecision_identity_keys():
        if left[key] != right[key]:
            mismatched.append(f"decision_snapshot.{key}")
    if mismatched:
        raise ValueError(f"seed {seed} differs from its raise-26 source before "
                         f"{before_s} s: {sorted(mismatched)}")
    return dict(rows_before_s=before_s,
        streams=("intervals.jsonl", "traffic.jsonl", "policy.jsonl", "telemetry.jsonl"),
        policy_comparison_excludes=sorted(selector_bookkeeping),
        decision_snapshot_keys=list(predecision_identity_keys()), exact=True)


def verify_extension(path, job):
    spec = job["policy_spec"]["hold_extension"]
    branch = spec["branch_time_s"]
    adjust = job["policy_spec"]["target_adjust_branch"]["branch_time_s"]
    formation = list(rows(path / "formation.jsonl"))
    # "Pre-branch" means after the raise adjustment (617/618) and before the
    # extension branch (632/638); the assembly window legitimately targets 24.
    pre = sorted({row["target_speed_mps"] for row in formation
                  if adjust <= row["time_s"] < branch})
    post = sorted({row["target_speed_mps"] for row in formation
                   if row["time_s"] >= branch})
    samples = []
    for row in rows(path / "formation.jsonl"):
        if row["time_s"] < branch:
            continue
        mean_speed = sum(m["speed_mps"] for m in row["members"]) / len(row["members"])
        samples.append((row["time_s"], row["phase"], mean_speed))
    taper_at = next((t for t, phase, _ in samples if phase == "taper"), None)
    release_at = next((t for t, phase, _ in samples if phase == "release"), None)
    abort_at = next((t for t, phase, _ in samples if phase == "abort"), None)
    # A protective abort during the extension is a physical outcome: report
    # the actual delivered HOLD instead of the nominal 30 s.
    delivered_end = min(t for t, phase, _ in samples
                        if phase in ("taper", "release", "abort"))         if any(phase in ("taper", "release", "abort") for _, phase, _ in samples) else None
    return dict(schema="hold_extend_verification/2", pre_branch_targets_mps=pre,
        post_branch_targets_mps=post, expected_total_hold_s=30.0,
        pre_branch_all_26=bool(pre) and all(abs(t - 26.0) < 1e-9 for t in pre),
        post_branch_all_26=bool(post) and all(abs(t - 26.0) < 1e-9 for t in post),
        taper_at_s=taper_at, release_at_s=release_at, abort_at_s=abort_at,
        delivered_hold_end_s=delivered_end)


def run_one(registration, job_id):
    root, data = read_registration(registration)
    job = next((j for j in data["jobs"] if j["id"] == job_id), None)
    if job is None or (root / "STOP.json").exists():
        raise ValueError("unknown job or stopped hold-extend study")
    budget = data.get("budget")
    if not isinstance(budget, dict) or budget.get("category") != "corridor":
        raise ValueError("hold-extend registration declares no budget enforcement")
    source = data["sources"][job_id]
    if sha(Path(source["path"]) / "manifest.json") != source["manifest_sha256"]:
        raise ValueError("registered raise-26 wall reference changed")
    for proof in data["references"].values():
        if sha(Path(proof["path"]) / "manifest.json") != proof["manifest_sha256"]:
            raise ValueError("registered damping reference changed")
    (root / "allocations").mkdir(exist_ok=True)
    with (root / "allocations" / f"{job_id}.json").open("x") as stream:
        json.dump(dict(job_id=job_id, allocated_simulated_seconds=2400, retries=0,
                       registration_sha256=data["registration_sha256"]), stream)
    policy = policy_factory(job, data)
    try:
        path = root / "jobs" / job_id
        result = paired.evaluate_arm(job=job, registration=data, output=path, policy=policy,
                                     budget=dict(budget, target_dir=str(root)))
        receipt = policy.ext_receipt
        prefix = verify_shared_prefix(path, Path(source["path"]),
                                      EXTENSION_BRANCH_S[job["seed"]], job["seed"])
        extension = verify_extension(path, job)
        life = score_extension(path, job)
        contract = command_contract(path, LifecycleProfile(**job["policy_spec"]["profile"]))
        protective_abort = life.get("terminal_phase") == "abort"
        scheduled_end_kept = (
            receipt["early_exit_reason"] is not None or protective_abort or (
                abs(receipt["actual_hold_start_s"] + 30.0
                    - extension["taper_at_s"]) < 1e-9))
        checks = dict(receipt_requested=receipt["requested"] and receipt["verified"],
            receipt_matches_source=receipt["actual_hold_start_s"] == source["measured_hold_start_s"],
            applied_once=bool(receipt["applied"]),
            extension_feasible_at_branch=bool(receipt["feasibility"]["feasible"]),
            targets_26_throughout=bool(extension["pre_branch_all_26"] and extension["post_branch_all_26"]),
            prefix_exact=prefix["exact"],
            end_clock_consistent=bool(scheduled_end_kept),
            terminal_recorded=bool(life.get("terminal_phase") in ("release", "abort")),
            command_contract=contract["passed"])
        write_json(path / "hold_extend_contract.json", dict(checks=checks, receipt=receipt,
            extension=extension, prefix=prefix, lifecycle=dict(
                assembly_seconds=life.get("assembly_seconds"),
                hold_seconds=life.get("hold_seconds"),
                terminal_phase=life.get("terminal_phase"),
                terminal_time_s=life.get("terminal_time_s")),
            command_contract=contract))
        if not all(checks.values()):
            raise ValueError("hold-extend assignment or contract verification failed: "
                             + ",".join(k for k, v in checks.items() if not v))
        return result
    except Exception as exc:
        write_json(root / "STOP.json", dict(job_id=job_id, reason=str(exc),
                                            automatic_retries=0))
        raise


def score_extension(path, job):
    """Extension-aware lifecycle scoring: measured HOLD must equal 30 s."""
    from .lifecycle_study import score_lifecycle
    extended_job = deepcopy(job)
    extended_job["policy_spec"]["profile"]["hold_s"] = 30.0
    return score_lifecycle(path, extended_job)


def resolve_verification(registration):
    """Re-run post-run verification after a verifier-defect STOP; no simulation."""
    root, data = read_registration(registration, require_current=False)
    stop = root / "STOP.json"
    if not stop.exists():
        raise ValueError("no stopped hold-extend block to resolve")
    original_stop = json.loads(stop.read_text(encoding="utf-8"))
    current = paired.current_identity()
    source_edits = {name: dict(registered=data["identity"]["active_source_sha256"][name],
                               current=current["active_source_sha256"][name])
        for name in data["identity"]["active_source_sha256"]
        if data["identity"]["active_source_sha256"][name] != current["active_source_sha256"].get(name)}
    unexpected = [n for n in source_edits
                  if not (n.startswith("hold_extend") or n.startswith("target_")
                          or n.startswith("continuation_"))
                  and n != "escalation_study.py"]
    if unexpected:
        raise ValueError(f"unexpected post-registration source edit: {unexpected}")
    resolutions = []
    for job in data["jobs"]:
        path = root / "jobs" / job["id"]
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            resolutions.append(dict(job_id=job["id"], status="not_started"))
            continue
        if json.loads(manifest_path.read_text()).get("status") != "complete":
            raise ValueError(f"cannot resolve: {job['id']} is incomplete")
        source = data["sources"][job["id"]]
        if sha(Path(source["path"]) / "manifest.json") != source["manifest_sha256"]:
            raise ValueError("registered raise-26 wall reference changed")
        prefix = verify_shared_prefix(path, Path(source["path"]),
                                      EXTENSION_BRANCH_S[job["seed"]], job["seed"])
        extension = verify_extension(path, job)
        receipt = None
        for row in rows(path / "policy.jsonl"):
            if isinstance(row.get("hold_extension_branch"), dict)                     and row["hold_extension_branch"].get("requested"):
                receipt = row["hold_extension_branch"]
        if receipt is None:
            raise ValueError(f"no hold-extension receipt found for {job['id']}")
        life = score_extension(path, job)
        contract = command_contract(path, LifecycleProfile(**job["policy_spec"]["profile"]))
        checks = dict(receipt_requested=receipt["requested"] and receipt["verified"],
            receipt_matches_source=receipt["actual_hold_start_s"] == source["measured_hold_start_s"],
            applied_once=bool(receipt["applied"]),
            prefix_exact=prefix["exact"],
            targets_26=bool(extension["pre_branch_all_26"] and extension["post_branch_all_26"]),
            terminal_recorded=bool(life.get("terminal_phase") in ("release", "abort")),
            command_contract=contract["passed"])
        if not all(checks.values()):
            raise ValueError(f"resolution checks failed for {job['id']}: "
                             + ",".join(k for k, v in checks.items() if not v))
        write_json(path / "hold_extend_contract.json",
                   dict(checks=checks, receipt=receipt, prefix=prefix,
                        extension=extension, command_contract=contract,
                        lifecycle=dict(assembly_seconds=life.get("assembly_seconds"),
                            hold_seconds=life.get("hold_seconds"),
                            terminal_phase=life.get("terminal_phase"),
                            terminal_time_s=life.get("terminal_time_s"),
                            early_exit_reason=life.get("early_exit_reason")),
                        verifier_resolution=dict(original_stop=original_stop,
                            defect="verifier required a taper; a protective abort "
                                   "during the extension is a physical outcome",
                            new_simulated_seconds=0, source_edits=sorted(source_edits))))
        resolutions.append(dict(job_id=job["id"], status="resolved", checks=checks))
    receipt_doc = dict(schema="hold_extend_verifier_resolution/1",
        original_stop=original_stop, new_simulated_seconds=0,
        resolutions=resolutions, post_registration_source_edits=sorted(source_edits),
        control_behavior_changed=False)
    write_json(root / "verification_resolution.json", receipt_doc)
    stop.unlink()
    return receipt_doc


def analyze(registration):
    from .escalation_evidence import benefit_label
    from .lifecycle_study import joint_non_deterioration
    root, data = read_registration(registration, require_current=False)
    if (root / "STOP.json").exists():
        raise ValueError("stopped hold-extend block cannot become an accepted complete comparison")
    results, pending = [], []
    for job in data["jobs"]:
        manifest_path = root / "jobs" / job["id"] / "manifest.json"
        if not manifest_path.exists() or json.loads(manifest_path.read_text()).get("status") != "complete":
            pending.append(job["id"])
            continue
        path = root / "jobs" / job["id"]
        tm, ts = paired.verified_arm(path)
        if tm["registration_sha256"] != data["registration_sha256"] or any(
                tm.get(k) != v for k, v in data["identity"].items()):
            raise ValueError("treatment source/registration mismatch")
        reference = Path(data["references"][job["id"]]["path"])
        cm, cs, proof = reference_proof(reference, data["identity"])
        from .target_speed_wall import predecision_identity_keys
        left = json.loads((reference / "decision_snapshots.json").read_text())["600"]
        right = json.loads((path / "decision_snapshots.json").read_text())["600"]
        for key in predecision_identity_keys():
            if left[key] != right[key]:
                raise ValueError("matched damping prefix failed: " + key)
        if cm["exogenous_future_sha256"] != tm["exogenous_future_sha256"]:
            raise ValueError("matched requested future differs")
        view = dict(cm, active_source_sha256=data["identity"]["active_source_sha256"],
            proposal_sha256=right["proposal_sha256"], decision_time_s=600.,
            predecision_state_sha256=left["predecision_state_sha256"])
        effect = paired_effect(view, cs, tm, ts)
        criteria = BenefitCriteria(**data["criteria"])
        benefit = benefit_label(effect, criteria)
        life = score_extension(path, job)
        contract = json.loads((path / "hold_extend_contract.json").read_text())
        source_path = Path(data["sources"][job["id"]]["path"])
        source_summary = json.loads((source_path / "summary.json").read_text())
        contrast = wall_vs_wall_contrast(path, source_path, ts["population"])
        results.append(dict(job_id=job["id"], seed=job["seed"],
            effect=effect, benefit=benefit,
            joint_non_deterioration=joint_non_deterioration(effect, criteria),
            lifecycle=life, contract_checks=contract["checks"], receipt=contract["receipt"],
            command_contract_passed=contract["command_contract"]["passed"],
            prefix=contract["prefix"], wall_vs_wall=contrast,
            source_raise_h20=dict(job_id=SOURCE_JOBS[job["seed"]][1],
                system_seconds_per_request=source_summary["system_seconds_per_requested_vehicle"],
                totals=source_summary["totals"],
                lifecycle_phases=source_summary["phase_seconds"])))
    answer = dict(schema="registered_hold_extend_results/1", results=results, pending=pending,
        complete=len(results) == len(data["jobs"]) and not pending,
        independent_exposed_seed_count=len({r["seed"] for r in results}),
        extension_s=EXTENSION_S,
        maximum_new_simulated_seconds=data["maximum_simulated_seconds"],
        actual_new_complete_simulated_seconds=2400 * len(results),
        new_mechanics_simulated_seconds=data["new_mechanics_simulated_seconds"],
        original_criteria_unchanged=True, duration_variants=False,
        escalation_established=False, optimal_duration_established=False,
        population_generalization_established=False, automatic_successors=False)
    write_json(root / "analysis.json", answer)
    return dict(analysis_path=str(root / "analysis.json"), complete=answer["complete"],
        results=[dict(job_id=r["job_id"], benefit=r["benefit"]["label"],
                      improvements=r["effect"]["improvements"],
                      joint=r["joint_non_deterioration"]["all_continuous_costs_nonworsening"],
                      service_vs_raise_h20=r["wall_vs_wall"]["metrics"]["requested_not_arrived_vehicle_seconds"]["per_request"])
                 for r in results], pending=pending)
