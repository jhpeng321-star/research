"""Registered same-prefix continuation contrast: lower, keep, or raise the target.

Four 2400 s episodes (seeds 4107/4108; adjustments -2/+2 m/s) branch five
seconds after the measured HOLD start of the already-verified H20 wall traces
and keep the original 20 s HOLD and finite exit law. The unchanged-target
continuations are the reused wall references; verified observer-replay damping
arms are the damping references. Original benefit criteria unchanged; the
strict joint diagnostic stays separate. This is the registered smaller
allocation (one 120 s mechanics fixture plus four episodes = 9720 s of the
9840 s ceiling). No retries, no extra seeds, no target grid beyond the two
registered adjustments, no successor campaign.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from . import paired_experiments as paired
from .brake_hold_study import command_contract, rows
from .config import ROOT
from .dissolution_study import REFERENCES, reference_proof
from .escalation_evidence import BenefitCriteria, paired_effect
from .lifecycle_study import score_lifecycle
from .lifecycle_wall import LifecycleProfile
from .target_adjust_mechanics import verify_completed
from .target_adjust_wall import VERSION, policy_factory
from .target_speed_study import (SOURCE_JOB_IDS, SOURCE_REGISTRATIONS,
                                 wall_vs_wall_contrast)
from .workflow import sha, write_json

SCHEMA = "registered_hold_target_adjust_campaign/1"
PROTOCOL = "docs/TARGET_ADJUST_CHECK.md"
ADJUST_MPS = (-2., 2.)
EXPECTED_HOLD_START_S = {4107: 612.0, 4108: 618.0}
PREFIX_END_S = {4107: 617.0, 4108: 623.0}


def _source_hold_start(seed):
    """Measured HOLD start from the immutable source formation file."""
    source = SOURCE_REGISTRATIONS[seed].parent/"jobs"/SOURCE_JOB_IDS[seed]
    with open(source/"formation.jsonl", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["phase"] == "hold":
                return float(row["time_s"])
    raise ValueError(f"source {seed} shows no measured HOLD")


def jobs():
    result = []
    for seed in (4107, 4108):
        registration = json.loads(SOURCE_REGISTRATIONS[seed].read_text(encoding="utf-8"))
        base = next(j for j in registration["jobs"] if j["id"] == SOURCE_JOB_IDS[seed])
        hold_start = _source_hold_start(seed)
        if hold_start != EXPECTED_HOLD_START_S[seed]:
            raise ValueError(f"seed {seed} source HOLD start {hold_start} differs from registered")
        for adjust in ADJUST_MPS:
            job = deepcopy(base)
            sign = "down" if adjust < 0 else "up"
            job["id"] = f"s{seed}_m2700_p20_t600_hold_adjust_{sign}_wall"
            job["policy_spec"]["profile"] = asdict(LifecycleProfile())
            job["policy_spec"]["target_adjust_branch"] = dict(
                branch_mode="absolute", adjust_mps=float(adjust),
                expected_hold_start_s=hold_start, branch_time_s=hold_start+5.)
            paired.validate_job(job)
            result.append(job)
    return result


def register(output, mechanics_registration):
    mechanics = verify_completed(mechanics_registration)
    if not mechanics["proof"]["passed"] or not all(mechanics["checks"].values()):
        raise ValueError("target-adjust mechanics fixture is not a completed passing check")
    identity = paired.current_identity()
    all_jobs = jobs()
    references, sources = {}, {}
    for job in all_jobs:
        seed = job["seed"]
        references[job["id"]] = reference_proof(
            REFERENCES/"jobs"/job["reference_job_id"], identity)[2]
        root = SOURCE_REGISTRATIONS[seed].parent
        sources[job["id"]] = dict(
            path=str(root/"jobs"/SOURCE_JOB_IDS[seed]),
            manifest_sha256=sha(root/"jobs"/SOURCE_JOB_IDS[seed]/"manifest.json"),
            registration_path=str(SOURCE_REGISTRATIONS[seed]),
            registration_file_sha256=sha(SOURCE_REGISTRATIONS[seed]),
            measured_hold_start_s=EXPECTED_HOLD_START_S[seed])
    original = json.loads(SOURCE_REGISTRATIONS[4107].read_text(encoding="utf-8"))
    data = dict(schema=SCHEMA, registered_at_utc=datetime.now(timezone.utc).isoformat(),
        identity=identity, jobs=all_jobs, references=references, sources=sources,
        adjust_mps=list(ADJUST_MPS), criteria=original["criteria"],
        communication=original["communication"],
        assignment="registered_experimental_assignment_not_trained_benefit_gate",
        interaction_diagnostics=True, policy_version=VERSION,
        maximum_attempts=1, maximum_simulated_seconds=9600, automatic_retries=0,
        automatic_successors=False, target_grid=False, new_mechanics_simulated_seconds=120.,
        mechanics_evidence=mechanics, protocol_path=PROTOCOL, protocol_sha256=sha(ROOT/PROTOCOL))
    data["registration_sha256"] = paired.digest(data)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output/"source_snapshot"
    snapshot.mkdir()
    for name in identity["active_source_sha256"]:
        shutil.copy2(ROOT/"traffic_shepherd"/name, snapshot/name)
    write_json(output/"registration.json", data)
    return dict(registration=str(output/"registration.json"), jobs=[j["id"] for j in all_jobs],
        maximum_simulated_seconds=9600, simulations_started=0, reused_wall_references=2,
        reused_damping_references=2)


def read_registration(path, *, require_current=True):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if (data.get("schema") != SCHEMA or data.get("registration_sha256") != paired.digest(
            {k: v for k, v in data.items() if k != "registration_sha256"})
            or data.get("jobs") != jobs() or data.get("maximum_attempts") != 1
            or data.get("maximum_simulated_seconds") != 9600 or data.get("automatic_retries") != 0
            or data.get("automatic_successors") is not False or data.get("target_grid") is not False
            or data.get("adjust_mps") != [-2., 2.]):
        raise ValueError("target-adjust registration or exact budget changed")
    if require_current and (data["identity"] != paired.current_identity()
            or data["protocol_sha256"] != sha(ROOT/PROTOCOL)):
        raise ValueError("source or protocol changed after registration")
    original = json.loads(SOURCE_REGISTRATIONS[4107].read_text(encoding="utf-8"))
    if (data["criteria"] != original["criteria"] or data["communication"] != original["communication"]
            or set(data["references"]) != {j["id"] for j in data["jobs"]}
            or set(data["sources"]) != {j["id"] for j in data["jobs"]}
            or data["assignment"] != "registered_experimental_assignment_not_trained_benefit_gate"):
        raise ValueError("registered criteria, communication or reference scope changed")
    for job in data["jobs"]:
        seed = job["seed"]
        if Path(data["sources"][job["id"]]["path"]).resolve() != (
                SOURCE_REGISTRATIONS[seed].parent/"jobs"/SOURCE_JOB_IDS[seed]).resolve():
            raise ValueError("registered wall reference path changed")
        if Path(data["references"][job["id"]]["path"]).resolve() != (
                REFERENCES/"jobs"/job["reference_job_id"]).resolve():
            raise ValueError("registered damping reference path changed")
    for name, expected in data["identity"]["active_source_sha256"].items():
        if sha(path.parent/"source_snapshot"/name) != expected:
            raise ValueError("registered source snapshot changed")
    return path.parent, data


def _target_profile(path):
    """(time, commanded target) pairs from the immutable formation file."""
    return [(row["time_s"], row["target_speed_mps"]) for row in rows(path/"formation.jsonl")]


def verify_adjustment(path, job):
    spec = job["policy_spec"]["target_adjust_branch"]
    branch = spec["branch_time_s"]
    profile_rows = _target_profile(path)
    pre = sorted({tgt for t, tgt in profile_rows if t < branch})
    post = sorted({tgt for t, tgt in profile_rows if t >= branch})
    # transition and time near the adjusted target from member mean speeds
    samples = []
    for row in rows(path/"formation.jsonl"):
        if row["time_s"] < branch:
            continue
        mean_speed = sum(m["speed_mps"] for m in row["members"])/len(row["members"])
        samples.append((row["time_s"], row["phase"], mean_speed))
    adjusted = spec["adjust_mps"] + 24.
    near_from = next((t for t, phase, spd in samples
                      if phase == "hold" and abs(spd-adjusted) <= 1.), None)
    taper_at = next((t for t, phase, _ in samples if phase == "taper"), None)
    transition_s = near_from-branch if near_from is not None else None
    time_near_s = (taper_at-near_from) if near_from is not None and taper_at is not None else None
    return dict(schema="hold_adjust_verification/1", pre_branch_targets_mps=pre,
        post_branch_targets_mps=post, adjusted_target_mps=adjusted,
        pre_branch_all_24=bool(pre) and all(abs(t-24.) < 1e-9 for t in pre),
        post_branch_all_adjusted=bool(post) and all(abs(t-adjusted) < 1e-9 for t in post),
        near_target_from_s=near_from, taper_at_s=taper_at,
        transition_to_adjusted_target_s=transition_s, time_near_adjusted_target_s=time_near_s)


def run_one(registration, job_id):
    root, data = read_registration(registration)
    job = next((j for j in data["jobs"] if j["id"] == job_id), None)
    if job is None or (root/"STOP.json").exists():
        raise ValueError("unknown job or stopped target-adjust study")
    source = data["sources"][job_id]
    if sha(Path(source["path"])/"manifest.json") != source["manifest_sha256"]:
        raise ValueError("registered H20 wall reference changed")
    for proof in data["references"].values():
        if sha(Path(proof["path"])/"manifest.json") != proof["manifest_sha256"]:
            raise ValueError("registered damping reference changed")
    (root/"allocations").mkdir(exist_ok=True)
    with (root/"allocations"/f"{job_id}.json").open("x") as stream:
        json.dump(dict(job_id=job_id, allocated_simulated_seconds=2400, retries=0,
            registration_sha256=data["registration_sha256"]), stream)
    policy = policy_factory(job, data)
    try:
        path = root/"jobs"/job_id
        result = paired.evaluate_arm(job=job, registration=data, output=path, policy=policy)
        prefix = verify_shared_prefix(path, Path(source["path"]),
                                      PREFIX_END_S[job["seed"]], job["seed"])
        adjustment = verify_adjustment(path, job)
        receipt = policy.receipt
        life = score_lifecycle(path, job)
        contract = command_contract(path, LifecycleProfile(**job["policy_spec"]["profile"]))
        scheduled_taper_kept = (receipt["early_exit_reason"] is None
            and life.get("hold_seconds") == 20.
            and abs(receipt["actual_hold_start_s"]+20.-adjustment["taper_at_s"]) < 1e-9)
        checks = dict(receipt_requested=receipt["requested"] and receipt["verified"],
            receipt_matches_source=receipt["actual_hold_start_s"] == source["measured_hold_start_s"],
            original_target_24=abs(receipt["original_target_speed_mps"]-24.) < 1e-9,
            adjusted_target=receipt["adjusted_target_speed_mps"] == 22. or receipt["adjusted_target_speed_mps"] == 26.,
            prefix_exact=prefix["exact"], targets_verified=adjustment["pre_branch_all_24"]
            and adjustment["post_branch_all_adjusted"],
            adjustment_reached=bool(adjustment["transition_to_adjusted_target_s"] is not None),
            scheduled_taper_kept=bool(scheduled_taper_kept) or receipt["early_exit_reason"] is not None,
            command_contract=contract["passed"])
        write_json(path/"target_adjust_contract.json", dict(checks=checks, receipt=receipt,
            adjustment=adjustment, prefix=prefix, lifecycle=dict(
                assembly_seconds=life.get("assembly_seconds"), hold_seconds=life.get("hold_seconds"),
                terminal_phase=life.get("terminal_phase"), terminal_time_s=life.get("terminal_time_s")),
            command_contract=contract))
        if not all(checks.values()):
            raise ValueError("target-adjust assignment or contract verification failed: "
                             + ",".join(k for k, v in checks.items() if not v))
        return result
    except Exception as exc:
        write_json(root/"STOP.json", dict(job_id=job_id, reason=str(exc), automatic_retries=0))
        raise


def verify_shared_prefix(new_path, source_path, before_s, seed):
    """Exact measured/report/command identity strictly before the branch tick.

    policy.jsonl rows are compared after removing source-side cohort-selector
    bookkeeping fields (cohort_selection*, cohort_selector): a fixed-cohort
    reuse legitimately omits the selector record, which is provenance about
    selection, never a measurement, command or report. Membership and measured
    HOLD start are verified separately against the registered source values.
    """
    from .target_speed_study import _prefix_rows
    from .target_speed_wall import predecision_identity_keys
    selector_bookkeeping = ("cohort_selection", "cohort_selection_sha256",
                            "cohort_selection_verified", "cohort_selector")

    def policy_rows(path):
        selected = []
        for line in open(path/"policy.jsonl", encoding="utf-8"):
            row = json.loads(line)
            if row["time_s"] < before_s:
                for key in selector_bookkeeping:
                    row.pop(key, None)
                selected.append(json.dumps(row, sort_keys=True, allow_nan=False))
        return selected

    mismatched = []
    if policy_rows(new_path) != policy_rows(source_path):
        mismatched.append("policy.jsonl")
    for name in ("intervals.jsonl", "traffic.jsonl", "telemetry.jsonl"):
        if _prefix_rows(new_path, name, before_s) != _prefix_rows(source_path, name, before_s):
            mismatched.append(name)
    left = json.loads((new_path/"decision_snapshots.json").read_text(encoding="utf-8"))["600"]
    right = json.loads((source_path/"decision_snapshots.json").read_text(encoding="utf-8"))["600"]
    for key in predecision_identity_keys():
        if left[key] != right[key]:
            mismatched.append(f"decision_snapshot.{key}")
    if mismatched:
        raise ValueError(f"seed {seed} differs from its H20 wall source before {before_s} s: {sorted(mismatched)}")
    return dict(rows_before_s=before_s,
        streams=("intervals.jsonl", "traffic.jsonl", "policy.jsonl", "telemetry.jsonl"),
        policy_comparison_excludes=sorted(selector_bookkeeping),
        decision_snapshot_keys=list(predecision_identity_keys()), exact=True)


def resolve_verification(registration):
    """Re-run post-run verification after a verifier-defect STOP; no simulation.

    Same contract as the target-speed resolution: the completed episodes are
    immutable; a STOP caused by the offline verifier itself is superseded only
    when every registered check passes on the preserved artifacts. The
    original STOP text stays inside the receipt. Post-registration source
    edits are disclosed and must be confined to the verifier modules.
    """
    root, data = read_registration(registration, require_current=False)
    stop = root/"STOP.json"
    if not stop.exists():
        raise ValueError("no stopped target-adjust block to resolve")
    original_stop = json.loads(stop.read_text(encoding="utf-8"))
    current = paired.current_identity()
    source_edits = {name: dict(registered=data["identity"]["active_source_sha256"][name],
                               current=current["active_source_sha256"][name])
        for name in data["identity"]["active_source_sha256"]
        if data["identity"]["active_source_sha256"][name] != current["active_source_sha256"].get(name)}
    unexpected = [n for n in source_edits
                  if not n.startswith("target_adjust") and not n.startswith("target_speed")
                  and n != "escalation_study.py"]
    if unexpected:
        raise ValueError(f"unexpected post-registration source edit: {unexpected}")
    resolutions = []
    for job in data["jobs"]:
        path = root/"jobs"/job["id"]
        manifest_path = path/"manifest.json"
        if not manifest_path.exists():
            resolutions.append(dict(job_id=job["id"], status="not_started"))
            continue
        if json.loads(manifest_path.read_text()).get("status") != "complete":
            raise ValueError(f"cannot resolve: {job['id']} is incomplete")
        source = data["sources"][job["id"]]
        if sha(Path(source["path"])/"manifest.json") != source["manifest_sha256"]:
            raise ValueError("registered H20 wall reference changed")
        prefix = verify_shared_prefix(path, Path(source["path"]),
                                      PREFIX_END_S[job["seed"]], job["seed"])
        adjustment = verify_adjustment(path, job)
        contract_path = path/"target_adjust_contract.json"
        if contract_path.exists():
            previous = json.loads(contract_path.read_text())
        else:
            # The episode completed but its run-time verification hit the same
            # verifier defect before writing the contract. Rebuild the receipt
            # from the immutable policy rows (last tick carries the compact state).
            receipt = None
            for row in rows(path/"policy.jsonl"):
                if isinstance(row.get("target_adjust_branch"), dict) \
                        and row["target_adjust_branch"].get("requested"):
                    receipt = row["target_adjust_branch"]
            if receipt is None:
                raise ValueError(f"no target-adjust receipt found for {job['id']}")
            previous = dict(checks={}, receipt=receipt, adjustment=None, prefix=None,
                            command_contract=None, lifecycle=None)
        receipt = previous["receipt"]
        life = score_lifecycle(path, job)
        contract = command_contract(path, LifecycleProfile(**job["policy_spec"]["profile"]))
        scheduled_taper_kept = (receipt.get("early_exit_reason") is None
            and life.get("hold_seconds") == 20.
            and abs(receipt["actual_hold_start_s"]+20.-adjustment["taper_at_s"]) < 1e-9)
        checks = dict(receipt_requested=receipt["requested"] and receipt["verified"],
            receipt_matches_source=receipt["actual_hold_start_s"] == source["measured_hold_start_s"],
            original_target_24=abs(receipt["original_target_speed_mps"]-24.) < 1e-9,
            adjusted_target=receipt["adjusted_target_speed_mps"] in (22., 26.),
            prefix_exact=prefix["exact"], targets_verified=adjustment["pre_branch_all_24"]
            and adjustment["post_branch_all_adjusted"],
            adjustment_reached=bool(adjustment["transition_to_adjusted_target_s"] is not None),
            scheduled_taper_kept=bool(scheduled_taper_kept),
            command_contract=contract["passed"])
        if not all(checks.values()):
            raise ValueError(f"resolution checks failed for {job['id']}: "
                             + ",".join(k for k, v in checks.items() if not v))
        previous.update(checks=checks, receipt=receipt, adjustment=adjustment, prefix=prefix,
            command_contract=contract,
            lifecycle=dict(assembly_seconds=life.get("assembly_seconds"),
                hold_seconds=life.get("hold_seconds"), terminal_phase=life.get("terminal_phase"),
                terminal_time_s=life.get("terminal_time_s")),
            verifier_resolution=dict(original_stop=original_stop,
                defect="offline verifier compared branch-time values where target values were "
                       "intended, and stripped no source-side cohort-selector bookkeeping",
                new_simulated_seconds=0, source_edits=sorted(source_edits)))
        write_json(path/"target_adjust_contract.json", previous)
        resolutions.append(dict(job_id=job["id"], status="resolved",
            checks=checks, prefix=prefix, adjustment=adjustment))
    receipt = dict(schema="target_adjust_verifier_resolution/1", original_stop=original_stop,
        new_simulated_seconds=0, resolutions=resolutions,
        post_registration_source_edits=sorted(source_edits), control_behavior_changed=False)
    write_json(root/"verification_resolution.json", receipt)
    stop.unlink()
    return receipt


def analyze(registration):
    from .escalation_evidence import benefit_label
    from .lifecycle_study import joint_non_deterioration
    root, data = read_registration(registration, require_current=False)
    if (root/"STOP.json").exists():
        raise ValueError("stopped target-adjust block cannot become an accepted complete comparison")
    results, pending = [], []
    for job in data["jobs"]:
        manifest_path = root/"jobs"/job["id"]/"manifest.json"
        if not manifest_path.exists() or json.loads(manifest_path.read_text()).get("status") != "complete":
            pending.append(job["id"])
            continue
        path = root/"jobs"/job["id"]
        tm, ts = paired.verified_arm(path)
        if tm["registration_sha256"] != data["registration_sha256"] or any(
                tm.get(k) != v for k, v in data["identity"].items()):
            raise ValueError("treatment source/registration mismatch")
        reference = Path(data["references"][job["id"]]["path"])
        cm, cs, proof = reference_proof(reference, data["identity"])
        left = json.loads((reference/"decision_snapshots.json").read_text())["600"]
        right = json.loads((path/"decision_snapshots.json").read_text())["600"]
        from .target_speed_wall import predecision_identity_keys
        for key in predecision_identity_keys():
            if left[key] != right[key]:
                raise ValueError("matched damping prefix failed: "+key)
        if cm["exogenous_future_sha256"] != tm["exogenous_future_sha256"]:
            raise ValueError("matched requested future differs")
        view = dict(cm, active_source_sha256=data["identity"]["active_source_sha256"],
            proposal_sha256=right["proposal_sha256"], decision_time_s=600.,
            predecision_state_sha256=left["predecision_state_sha256"])
        effect = paired_effect(view, cs, tm, ts)
        criteria = BenefitCriteria(**data["criteria"])
        benefit = benefit_label(effect, criteria)
        life = score_lifecycle(path, job)
        contract = json.loads((path/"target_adjust_contract.json").read_text())
        source_path = Path(data["sources"][job["id"]]["path"])
        source_summary = json.loads((source_path/"summary.json").read_text())
        contrast = wall_vs_wall_contrast(path, source_path, ts["population"])
        results.append(dict(job_id=job["id"], seed=job["seed"], adjust_mps=job["policy_spec"]["target_adjust_branch"]["adjust_mps"],
            effect=effect, benefit=benefit,
            joint_non_deterioration=joint_non_deterioration(effect, criteria),
            lifecycle=life, contract_checks=contract["checks"], receipt=contract["receipt"],
            adjustment=contract["adjustment"], command_contract_passed=contract["command_contract"]["passed"],
            prefix=contract["prefix"], wall_vs_wall=contrast,
            source_h20=dict(job_id=SOURCE_JOB_IDS[job["seed"]],
                system_seconds_per_request=source_summary["system_seconds_per_requested_vehicle"],
                totals=source_summary["totals"], lifecycle_phases=source_summary["phase_seconds"])))
    answer = dict(schema="registered_hold_target_adjust_results/1", results=results, pending=pending,
        complete=len(results) == len(data["jobs"]) and not pending,
        independent_exposed_seed_count=len({r["seed"] for r in results}),
        continuation_actions_per_seed=2, reused_unchanged_continuation_source=True,
        adjust_mps=list(ADJUST_MPS),
        maximum_new_simulated_seconds=data["maximum_simulated_seconds"],
        actual_new_complete_simulated_seconds=2400*len(results),
        new_mechanics_simulated_seconds=data["new_mechanics_simulated_seconds"],
        original_criteria_unchanged=True, target_grid=False,
        escalation_established=False, optimal_target_established=False,
        population_generalization_established=False, automatic_successors=False)
    write_json(root/"analysis.json", answer)
    return dict(analysis_path=str(root/"analysis.json"), complete=answer["complete"],
        results=[dict(job_id=r["job_id"], benefit=r["benefit"]["label"],
                      improvements=r["effect"]["improvements"],
                      joint=r["joint_non_deterioration"]["all_continuous_costs_nonworsening"],
                      service_vs_h20=r["wall_vs_wall"]["metrics"]["requested_not_arrived_vehicle_seconds"]["per_request"])
                 for r in results], pending=pending)
