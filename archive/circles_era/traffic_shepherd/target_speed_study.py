"""Registered bounded comparison of the uncapped target rule on exposed seeds.

Two 2400 s corridor episodes (seeds 4107/4108, m2700, p20, t600) run the
unchanged finite lifecycle wall with the single registered profile change
target_speed_wall.TargetSpeedProfile. The H20 wall traces that already exist
are reused as wall-versus-wall references and the verified observer-replay
damping arms as wall-versus-damping references. Original benefit criteria are
unchanged. This is one fixed alternative rule, never a speed sweep; no
successor, sweep or replication follows automatically.
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
from .target_speed_mechanics import verify_completed
from .target_speed_wall import (TARGET_RULE, TargetSpeedProfile, VERSION,
                                expected_uncapped_target, predecision_identity_keys)
from .workflow import sha, write_json

SCHEMA = "registered_target_speed_campaign/1"
PROTOCOL = "docs/TARGET_SPEED_CHECK.md"
SOURCE_REGISTRATIONS = {
    4107: ROOT/"runs/active/lifecycle_repair_v2/registration.json",
    4108: ROOT/"runs/active/cohort_eligibility_v1/registration.json",
}
SOURCE_JOB_IDS = {
    4107: "s4107_m2700_p20_t600_lifecycle_v2_wall",
    4108: "s4108_m2700_p20_t600_cohort_v1_wall",
}
PREFIX_END_S = 600.0


def jobs():
    result = []
    for seed, path in SOURCE_REGISTRATIONS.items():
        registration = json.loads(path.read_text(encoding="utf-8"))
        job = deepcopy(next(j for j in registration["jobs"] if j["id"] == SOURCE_JOB_IDS[seed]))
        job["id"] = f"s{seed}_m2700_p20_t600_target_v1_wall"
        job["policy_spec"]["profile"] = asdict(TargetSpeedProfile())
        paired.validate_job(job)
        result.append(job)
    return result


def source_job(seed):
    registration = json.loads(SOURCE_REGISTRATIONS[seed].read_text(encoding="utf-8"))
    job = next(j for j in registration["jobs"] if j["id"] == SOURCE_JOB_IDS[seed])
    root = SOURCE_REGISTRATIONS[seed].parent
    return root, registration, job


def register(output, mechanics_registration):
    mechanics = verify_completed(mechanics_registration)
    if not mechanics["proof"]["passed"] or not all(mechanics["checks"].values()):
        raise ValueError("target-speed mechanics fixture is not a completed passing check")
    identity = paired.current_identity()
    all_jobs = jobs()
    references, sources = {}, {}
    for job in all_jobs:
        seed = job["seed"]
        references[job["id"]] = reference_proof(
            REFERENCES/"jobs"/job["reference_job_id"], identity)[2]
        root, registration, source_job_dict = source_job(seed)
        sources[job["id"]] = dict(
            path=str(root/"jobs"/SOURCE_JOB_IDS[seed]),
            manifest_sha256=sha(root/"jobs"/SOURCE_JOB_IDS[seed]/"manifest.json"),
            registration_path=str(SOURCE_REGISTRATIONS[seed]),
            registration_file_sha256=sha(SOURCE_REGISTRATIONS[seed]),
            original_profile_law_version=source_job_dict["policy_spec"]["profile"]["version"],
            original_maximum_target_speed_mps=source_job_dict["policy_spec"]["profile"]["maximum_target_speed_mps"])
    original = json.loads(SOURCE_REGISTRATIONS[4107].read_text(encoding="utf-8"))
    data = dict(schema=SCHEMA, registered_at_utc=datetime.now(timezone.utc).isoformat(),
        identity=identity, jobs=all_jobs, references=references, sources=sources,
        target_rule=TARGET_RULE, criteria=original["criteria"], communication=original["communication"],
        assignment="registered_experimental_assignment_not_trained_benefit_gate",
        interaction_diagnostics=True, policy_version=VERSION,
        maximum_attempts=1, maximum_simulated_seconds=4800, automatic_retries=0,
        automatic_successors=False, speed_sweep=False, new_mechanics_simulated_seconds=120.,
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
        maximum_simulated_seconds=4800, simulations_started=0, reused_wall_references=2,
        reused_damping_references=2)


def read_registration(path, *, require_current=True):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if (data.get("schema") != SCHEMA or data.get("registration_sha256") != paired.digest(
            {k: v for k, v in data.items() if k != "registration_sha256"})
            or data.get("jobs") != jobs() or data.get("maximum_attempts") != 1
            or data.get("maximum_simulated_seconds") != 4800 or data.get("automatic_retries") != 0
            or data.get("automatic_successors") is not False or data.get("speed_sweep") is not False
            or data.get("target_rule") != TARGET_RULE):
        raise ValueError("target-speed registration or exact budget changed")
    if require_current and (data["identity"] != paired.current_identity()
            or data["protocol_sha256"] != sha(ROOT/PROTOCOL)):
        raise ValueError("source or protocol changed after registration")
    original = json.loads(SOURCE_REGISTRATIONS[4107].read_text(encoding="utf-8"))
    if (data["criteria"] != original["criteria"] or data["communication"] != original["communication"]
            or set(data["references"]) != {j["id"] for j in data["jobs"]}
            or set(data["sources"]) != {j["id"] for j in data["jobs"]}
            or data["assignment"] != "registered_experimental_assignment_not_trained_benefit_gate"
            or data["interaction_diagnostics"] is not True):
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


def _prefix_rows(path, name, before_s):
    """Canonical rows of one stream strictly before the branch time."""
    if name == "telemetry.jsonl":
        def time_of(row):
            kind = row.get("kind")
            if kind in {"command", "response", "measured_response", "observation_unavailable"}:
                return row.get("timestamp_s")
            if kind == "rsu_received_report":
                return row.get("received_at_s")
            if kind == "cav_report":
                return row.get("frame", {}).get("timestamp_s")
            raise ValueError(f"unknown telemetry kind {kind}")
    elif name == "policy.jsonl":
        def time_of(row):
            return row["time_s"]
    elif name == "intervals.jsonl":
        def time_of(row):
            return row["start_s"]
    elif name == "traffic.jsonl":
        def time_of(row):
            return row["timestamp_s"]
    else:
        raise ValueError(f"unknown stream {name}")
    selected = []
    with open(path/name, encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            t = time_of(row)
            if t is not None and t < before_s:
                selected.append(json.dumps(row, sort_keys=True, allow_nan=False))
    return selected


def verify_shared_prefix(new_path, source_path, seed):
    """Exact measured/report/command identity before the 600 s engagement."""
    mismatched = []
    for name in ("intervals.jsonl", "traffic.jsonl", "policy.jsonl", "telemetry.jsonl"):
        if _prefix_rows(new_path, name, PREFIX_END_S) != _prefix_rows(source_path, name, PREFIX_END_S):
            mismatched.append(name)
    left = json.loads((new_path/"decision_snapshots.json").read_text(encoding="utf-8"))["600"]
    right = json.loads((source_path/"decision_snapshots.json").read_text(encoding="utf-8"))["600"]
    for key in predecision_identity_keys():
        if left[key] != right[key]:
            mismatched.append(f"decision_snapshot.{key}")
    if mismatched:
        raise ValueError(f"seed {seed} differs from its H20 wall source before 600 s: {sorted(mismatched)}")
    return dict(rows_before_s=PREFIX_END_S,
        streams=("intervals.jsonl", "traffic.jsonl", "policy.jsonl", "telemetry.jsonl"),
        decision_snapshot_keys=list(predecision_identity_keys()), exact=True)


def run_one(registration, job_id):
    from .target_speed_wall import policy_factory
    root, data = read_registration(registration)
    job = next((j for j in data["jobs"] if j["id"] == job_id), None)
    if job is None or (root/"STOP.json").exists():
        raise ValueError("unknown job or stopped target-speed study")
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
    try:
        path = root/"jobs"/job_id
        result = paired.evaluate_arm(job=job, registration=data, output=path,
                                     policy=policy_factory(job, data))
        prefix = verify_shared_prefix(path, Path(source["path"]), job["seed"])
        law = verify_target_law(path, job)
        contract = command_contract(path, TargetSpeedProfile(**job["policy_spec"]["profile"]))
        write_json(path/"target_speed_contract.json",
                   dict(prefix=prefix, target_law=law, command_contract=contract))
        if not contract["passed"] or not law["uncapped_rule_held"]:
            raise ValueError("target-speed assignment, target law or command contract failed")
        return result
    except Exception as exc:
        write_json(root/"STOP.json", dict(job_id=job_id, reason=str(exc), automatic_retries=0))
        raise


def verify_target_law(path, job):
    """Every commanded target must equal the uncapped rule and exceed the old cap."""
    formation = [r for r in rows(path/"formation.jsonl")]
    if not formation:
        raise ValueError("no formation rows for target-law audit")
    first = formation[0]
    speeds = [m["speed_mps"] for m in first["members"]]
    allowed = [1000.]*len(speeds)  # member allowed speeds are not in formation rows
    entry_mean = sum(speeds)/len(speeds)
    expected = max(0., entry_mean-2.)
    distinct = sorted({r["target_speed_mps"] for r in formation})
    ok = (len(distinct) == 1 and abs(distinct[0]-expected) <= 1e-9
          and distinct[0] > 24. and expected <= TargetSpeedProfile().maximum_target_speed_mps
          and expected_uncapped_target(speeds, allowed) == distinct[0])
    return dict(schema="target_law_audit/1", entry_member_speeds_mps=speeds,
        entry_mean_speed_mps=entry_mean, expected_uncapped_target_mps=expected,
        distinct_commanded_targets_mps=distinct, uncapped_rule_held=ok,
        exceeds_old_cap_mps=bool(distinct[0] > 24.) if len(distinct) == 1 else False)


def wall_vs_wall_contrast(new_path, source_path, population):
    """Paired H20-minus-new differences from immutable interval files."""
    metrics = ("braking_vehicle_seconds", "acceleration_energy", "ttc_below_3_vehicle_seconds",
               "requested_not_arrived_vehicle_seconds", "collision_involvements", "starting_teleports")
    windows = ((0, 600), (600, 650), (650, 900), (900, 1500), (1500, 2400))
    def series(path):
        out = {m: [] for m in metrics}
        for r in rows(path/"intervals.jsonl"):
            for m in metrics:
                out[m].append(r[m])
        return out
    new, old = series(new_path), series(source_path)
    return dict(population=population, direction="new_target_minus_h20_source",
        metrics={m: dict(full_difference=sum(new[m])-sum(old[m]),
                         per_request=(sum(new[m])-sum(old[m]))/population,
                         windows={f"{a}-{b}": sum(new[m][a:b])-sum(old[m][a:b]) for a, b in windows})
                 for m in metrics})


def resolve_verification(registration):
    """Re-run post-run verification after a verifier-defect STOP; no simulation.

    The physical episodes are immutable and complete. A STOP written by the
    verification helper itself (here: an unhandled offline telemetry kind in
    the source file) is superseded only when every registered check passes on
    the preserved artifacts. The original STOP text is preserved inside the
    resolution receipt; the STOP file is then cleared so analyze can proceed.
    Any post-registration source edit is disclosed by hash diff.
    """
    root, data = read_registration(registration, require_current=False)
    stop = root/"STOP.json"
    if not stop.exists():
        raise ValueError("no stopped target-speed block to resolve")
    original_stop = json.loads(stop.read_text(encoding="utf-8"))
    current = paired.current_identity()
    source_edits = {name: dict(registered=data["identity"]["active_source_sha256"][name],
                               current=current["active_source_sha256"][name])
        for name in data["identity"]["active_source_sha256"]
        if data["identity"]["active_source_sha256"][name] != current["active_source_sha256"].get(name)}
    if any("target_speed_" not in name and name != "escalation_study.py" for name in source_edits):
        raise ValueError(f"unexpected post-registration source edit: {sorted(source_edits)}")
    resolutions = []
    for job in data["jobs"]:
        path = root/"jobs"/job["id"]
        manifest = json.loads((path/"manifest.json").read_text())
        if manifest.get("status") != "complete":
            raise ValueError(f"cannot resolve: {job['id']} is not complete")
        source = data["sources"][job["id"]]
        if sha(Path(source["path"])/"manifest.json") != source["manifest_sha256"]:
            raise ValueError("registered H20 wall reference changed")
        prefix = verify_shared_prefix(path, Path(source["path"]), job["seed"])
        law = verify_target_law(path, job)
        contract = command_contract(path, TargetSpeedProfile(**job["policy_spec"]["profile"]))
        if not contract["passed"] or not law["uncapped_rule_held"]:
            raise ValueError(f"resolution checks failed for {job['id']}")
        write_json(path/"target_speed_contract.json",
                   dict(prefix=prefix, target_law=law, command_contract=contract))
        resolutions.append(dict(job_id=job["id"], prefix=prefix, target_law=law,
                                command_contract_passed=contract["passed"]))
    receipt = dict(schema="target_speed_verifier_resolution/1",
        original_stop=original_stop, defect="offline verifier did not classify the "
            "observation_unavailable telemetry kind; the physical episodes were complete",
        new_simulated_seconds=0, resolutions=resolutions,
        post_registration_source_edits=source_edits,
        control_behavior_changed=False)
    write_json(root/"verification_resolution.json", receipt)
    stop.unlink()
    return receipt


def analyze(registration):
    from .escalation_evidence import benefit_label
    from .lifecycle_study import joint_non_deterioration
    root, data = read_registration(registration, require_current=False)
    if (root/"STOP.json").exists():
        raise ValueError("stopped target-speed block cannot become an accepted complete comparison")
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
        contract = json.loads((path/"target_speed_contract.json").read_text())
        source_path = Path(data["sources"][job["id"]]["path"])
        source_summary = json.loads((source_path/"summary.json").read_text())
        contrast = wall_vs_wall_contrast(path, source_path, ts["population"])
        results.append(dict(job_id=job["id"], seed=job["seed"],
            effect=effect, benefit=benefit,
            joint_non_deterioration=joint_non_deterioration(effect, criteria),
            lifecycle=life, target_law=contract["target_law"], command_contract_passed=contract["command_contract"]["passed"],
            prefix=contract["prefix"],
            assembly_seconds=life.get("assembly_seconds"),
            wall_vs_wall=contrast,
            source_h20=dict(job_id=SOURCE_JOB_IDS[job["seed"]],
                system_seconds_per_request=source_summary["system_seconds_per_requested_vehicle"],
                totals=source_summary["totals"],
                lifecycle_phases=source_summary["phase_seconds"])))
    answer = dict(schema="registered_target_speed_results/1", results=results, pending=pending,
        complete=len(results) == len(data["jobs"]) and not pending,
        independent_exposed_seed_count=len(results), target_rule=TARGET_RULE,
        maximum_new_simulated_seconds=data["maximum_simulated_seconds"],
        actual_new_complete_simulated_seconds=2400*len(results),
        new_mechanics_simulated_seconds=data["new_mechanics_simulated_seconds"],
        original_criteria_unchanged=True, speed_sweep=False, optimal_target_established=False,
        population_generalization_established=False, automatic_successors=False)
    write_json(root/"analysis.json", answer)
    return dict(analysis_path=str(root/"analysis.json"), complete=answer["complete"],
        results=[dict(job_id=r["job_id"], benefit=r["benefit"],
                      improvements=r["effect"]["improvements"],
                      wall_vs_wall=r["wall_vs_wall"]["metrics"]["requested_not_arrived_vehicle_seconds"]["per_request"])
                 for r in results], pending=pending)
