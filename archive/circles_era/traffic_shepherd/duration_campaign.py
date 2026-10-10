"""Bounded ordinary full-episode HOLD+5 release checks after completed B and C."""
from copy import deepcopy
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import shutil

from traffic_shepherd import paired_experiments as paired
from traffic_shepherd import cohort_campaign as cohort_campaign
from traffic_shepherd import cohort_evidence as cohort
from traffic_shepherd.brake_hold_study import command_contract, source_bridge
from traffic_shepherd.config import ROOT
from traffic_shepherd.dissolution_study import reference_proof
from traffic_shepherd.lifecycle_wall import LifecycleProfile
from traffic_shepherd.workflow import sha, write_json

SCHEMA = "registered_duration_branch_campaign/1"
PROTOCOL = "docs/DURATION_RELEASE_CHECK.md"


def _evidence():
    return importlib.import_module("traffic_shepherd.duration_evidence")


def _policy():
    return importlib.import_module("traffic_shepherd.duration_study")


def budget(jobs):
    if (not 1 <= len(jobs) <= 2 or len({j["id"] for j in jobs}) != len(jobs)
            or {j["seed"] for j in jobs} not in ({4107}, {4107, 4108})
            or any(j["duration_s"] != 2400 for j in jobs)):
        raise ValueError("one eligible 4107 and at most one eligible 4108 full episode")
    return dict(maximum_attempts=len(jobs), maximum_simulated_seconds=2400*len(jobs),
        new_job_ids=[j["id"] for j in jobs], automatic_retries=0, automatic_successors=False,
        new_mechanics_seconds=0, prefix_is_charged_in_full_episode=True)


def verified_prerequisites(b_registration, c_registration, identity):
    b_root, b_data, b_proof = cohort_campaign.verified_part_b(b_registration, identity)
    c_root, c_data = cohort_campaign.read_registration(c_registration, require_current=False)
    if (c_root / "STOP.json").exists():
        raise ValueError("stopped Part C cannot authorize duration comparison")
    if Path(c_data["part_b"]["registration_path"]).resolve() != (b_root / "registration.json").resolve():
        raise ValueError("Part C is not the registered successor of this Part B")
    analysis_path = c_root / "analysis.json"
    analysis = json.loads(analysis_path.read_text())
    if not analysis.get("complete") or analysis.get("pending"):
        raise ValueError("complete verified Part C physical verdict required before registration")
    actual = [cohort.compare_case(c_root, c_data, j) for j in c_data["jobs"]]
    cohort._same(actual, analysis["results"], "Part C final verdict differs from verified physical records")
    sources = []
    for seed, root, data in ((4107, b_root, b_data), (4108, c_root, c_data)):
        source_job = next(j for j in data["jobs"] if j["seed"] == seed)
        path = root / "jobs" / source_job["id"]
        manifest, _ = cohort._registered_arm(path, source_job, data)
        bridge = source_bridge(manifest, identity)
        sources.append(dict(seed=seed, path=path, job=source_job, registration=data,
            registration_path=root / "registration.json", manifest=manifest, source_bridge=bridge))
    proof = dict(part_b=b_proof, part_c=dict(registration_path=str(c_root / "registration.json"),
        registration_file_sha256=sha(c_root / "registration.json"),
        registration_sha256=c_data["registration_sha256"], analysis_path=str(analysis_path),
        analysis_sha256=sha(analysis_path), verified_comparisons_sha256=paired.digest(actual)))
    return sources, proof


def _job(source_job, opportunity):
    job = deepcopy(source_job)
    job["id"] = f"s{job['seed']}_m2700_p20_t600_release_h5_wall"
    job["policy_spec"]["duration_branch"] = opportunity["branch_spec"]
    job["snapshot_times_s"] = sorted({job["decision_time_s"], int(opportunity["branch_time_s"])})
    paired.validate_job(job)
    _policy().validate_branch_spec(job["policy_spec"]["duration_branch"])
    if (job["loading"] != 2700 or job["penetration"] != .2 or job["arm"] != "wall"
            or job["decision_time_s"] != 600 or LifecycleProfile(**job["policy_spec"]["profile"]).hold_s != 20):
        raise ValueError("unchanged H20 approach proposal required")
    return job


def register(output, b_registration, c_registration):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("preserve previous duration registration")
    identity = paired.current_identity()
    sources, prerequisite = verified_prerequisites(b_registration, c_registration, identity)
    jobs, continuations, excluded, references = [], {}, [], {}
    original = sources[0]["registration"]
    for source in sources:
        path, source_job = source["path"], source["job"]
        if (source["registration"]["criteria"] != original["criteria"]
                or source["registration"]["communication"] != original["communication"]):
            raise ValueError("source arms differ in original criteria or communication")
        eligible = _evidence().opportunity(path, source_job, source["registration"])
        if not eligible["eligible"]:
            excluded.append(dict(seed=source["seed"], source_path=str(path), source_job=source_job,
                source_manifest_sha256=sha(path / "manifest.json"), registration_path=str(source["registration_path"]),
                opportunity=eligible, allocated_simulated_seconds=0, replacement_authorized=False))
            if source["seed"] == 4107:
                raise ValueError("primary registered 4107 branch is no longer eligible")
            continue
        job = _job(source_job, eligible)
        jobs.append(job)
        reference_path = source["registration"]["references"][source_job["id"]]["path"]
        dm, _, reference = reference_proof(reference_path, identity)
        cohort._prefix(cohort._snapshot(reference_path), cohort._snapshot(path))
        if dm["exogenous_future_sha256"] != source["manifest"]["exogenous_future_sha256"]:
            raise ValueError("duration reference requested future differs")
        references[job["id"]] = reference
        continuations[job["id"]] = dict(path=str(path), job=source_job,
            registration_path=str(source["registration_path"]), registration_file_sha256=sha(source["registration_path"]),
            registration_sha256=source["registration"]["registration_sha256"],
            manifest_sha256=sha(path / "manifest.json"), summary_sha256=sha(path / "summary.json"),
            source_bridge=source["source_bridge"], opportunity=eligible)
    data = dict(schema=SCHEMA, registered_at_utc=datetime.now(timezone.utc).isoformat(), identity=identity,
        jobs=jobs, continuations=continuations, ineligible_opportunities=excluded, references=references,
        prerequisites=prerequisite, criteria=original["criteria"], communication=original["communication"],
        assignment="registered_experimental_assignment_not_trained_benefit_gate", interaction_diagnostics=True,
        policy_version=_policy().VERSION, hold_offset_s=5., original_profile_hold_s=20.,
        protocol_path=PROTOCOL, protocol_sha256=sha(ROOT / PROTOCOL), **budget(jobs))
    draft = dict(data, registration_sha256=paired.digest(data))
    for job in jobs:
        source = continuations[job["id"]]
        source_registration = cohort._registration(source["registration_path"])
        proof = _evidence().continuation_proof(source["path"], source["job"], source_registration,
            identity, release_job=job, release_registration=draft)
        if not _evidence().validate_branch_output(dict(time_s=source["opportunity"]["branch_time_s"],
                **proof["branch_output"]), job["policy_spec"]["duration_branch"]):
            raise ValueError("registered branch replay did not prove deliberate acquired exit")
        source["replay_proof"], source["replay_proof_sha256"] = proof, paired.digest(proof)
    data["registration_sha256"] = paired.digest(data)
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for name in identity["active_source_sha256"]:
        shutil.copy2(ROOT / "traffic_shepherd" / name, snapshot / name)
    shutil.copy2(ROOT / PROTOCOL, output / "protocol.md")
    write_json(output / "registration.json", data)
    return dict(registration=str(output / "registration.json"), new_jobs=data["new_job_ids"],
        maximum_simulated_seconds=data["maximum_simulated_seconds"], simulations_started=0,
        ineligible_opportunities=excluded, reused_continuation_traces=len(jobs))


def read_registration(path, *, require_current=True):
    path = Path(path).resolve()
    data = cohort._registration(path)
    if data.get("schema") != SCHEMA or any(data.get(k) != v for k, v in budget(data["jobs"]).items()):
        raise ValueError("duration registration schema or exact budget changed")
    if (data.get("policy_version") != _policy().VERSION or data.get("hold_offset_s") != 5.
            or data.get("original_profile_hold_s") != 20. or data.get("protocol_path") != PROTOCOL
            or data.get("interaction_diagnostics") is not True
            or data.get("assignment") != "registered_experimental_assignment_not_trained_benefit_gate"):
        raise ValueError("duration scientific assignment changed")
    if sha(path.parent / "protocol.md") != data["protocol_sha256"]:
        raise ValueError("registered duration protocol snapshot changed")
    if require_current and (data["identity"] != paired.current_identity()
                            or sha(ROOT / PROTOCOL) != data["protocol_sha256"]):
        raise ValueError("duration execution source or protocol changed")
    for stage in (data["prerequisites"]["part_b"], data["prerequisites"]["part_c"]):
        if (sha(stage["registration_path"]) != stage["registration_file_sha256"]
                or sha(stage["analysis_path"]) != stage["analysis_sha256"]
                or (Path(stage["registration_path"]).parent / "STOP.json").exists()):
            raise ValueError("duration prerequisite changed or stopped")
        original = cohort._registration(stage["registration_path"])
        analysis = json.loads(Path(stage["analysis_path"]).read_text())
        if (original["registration_sha256"] != stage["registration_sha256"]
                or not analysis.get("complete") or analysis.get("pending")
                or paired.digest(analysis["results"]) != stage["verified_comparisons_sha256"]):
            raise ValueError("duration prerequisite verdict or source proof changed")
    ids = {j["id"] for j in data["jobs"]}
    if set(data["continuations"]) != ids or set(data["references"]) != ids:
        raise ValueError("duration source scope changed")
    source_seeds = {j["seed"] for j in data["jobs"]}
    for job in data["jobs"]:
        source = data["continuations"][job["id"]]
        expected_stage = data["prerequisites"]["part_b" if job["seed"] == 4107 else "part_c"]
        if Path(source["registration_path"]).resolve() != Path(expected_stage["registration_path"]).resolve():
            raise ValueError("duration source is not exact B4107 or C4108")
        source_registration = cohort._registration(source["registration_path"])
        if (sha(source["registration_path"]) != source["registration_file_sha256"]
                or source_registration["registration_sha256"] != source["registration_sha256"]):
            raise ValueError("continuation registration binding changed")
        sp = Path(source["path"]).resolve()
        if sp != Path(source["registration_path"]).parent / "jobs" / source["job"]["id"]:
            raise ValueError("continuation path is not its exact registered source")
        sm, _ = cohort._registered_arm(sp, source["job"], source_registration)
        if (sha(sp / "manifest.json") != source["manifest_sha256"]
                or sha(sp / "summary.json") != source["summary_sha256"]):
            raise ValueError("continuation raw binding changed")
        cohort._same(source_bridge(sm, data["identity"]), source["source_bridge"], "continuation source bridge changed")
        if (source_registration["criteria"] != data["criteria"]
                or source_registration["communication"] != data["communication"]):
            raise ValueError("duration original criteria/communication changed")
        opportunity = _evidence().opportunity(sp, source["job"], source_registration)
        cohort._same(opportunity, source["opportunity"], "acquired branch eligibility changed")
        cohort._same(_job(source["job"], opportunity), job, "registered release job differs from original H20 branch")
        proof = source["replay_proof"]
        if (paired.digest(proof) != source["replay_proof_sha256"] or proof.get("complete") is not True
                or proof.get("decision_count") != 2400 or proof.get("first_decision_s") != 0.
                or proof.get("last_decision_s") != 2399. or proof.get("release_prefix_exact") is not True
                or proof.get("release_prefix_decisions") != opportunity["branch_time_s"]
                or proof.get("source_manifest_sha256") != source["manifest_sha256"]
                or proof.get("source_registration_sha256") != source["registration_sha256"]
                or proof.get("source_job_sha256") != paired.digest(source["job"])
                or proof.get("target_source_sha256") != data["identity"]["active_source_sha256"]
                or proof.get("replay_implementation_sha256") != data["identity"]["active_source_sha256"].get("duration_evidence.py")
                or proof.get("release_job_sha256") != paired.digest(job)
                or proof.get("every_core_policy_field_equal") is not True
                or proof.get("every_actual_command_record_field_equal") is not True
                or proof.get("no_simulator_rollout") is not True
                or proof.get("simulations_started") != 0):
            raise ValueError("complete source-bound continuation/prefix replay proof changed")
        if not _evidence().validate_branch_output(dict(time_s=opportunity["branch_time_s"],
                **proof["branch_output"]), job["policy_spec"]["duration_branch"]):
            raise ValueError("acquired replay branch proof changed")
        reference = data["references"][job["id"]]
        dm, _, observed = reference_proof(reference["path"], data["identity"])
        if observed["manifest_sha256"] != reference["manifest_sha256"]:
            raise ValueError("duration damping reference changed")
        cohort._prefix(cohort._snapshot(reference["path"]), cohort._snapshot(sp))
        if dm["exogenous_future_sha256"] != sm["exogenous_future_sha256"]:
            raise ValueError("duration exogenous requested future changed")
    exclusions = data["ineligible_opportunities"]
    if (len(exclusions) != 2-len(source_seeds)
            or any(r["seed"] != 4108 or r["seed"] in source_seeds or r["allocated_simulated_seconds"] != 0
                   or r["replacement_authorized"] is not False for r in exclusions)):
        raise ValueError("conditional zero-allocation accounting changed")
    for row in exclusions:
        if Path(row["registration_path"]).resolve() != Path(data["prerequisites"]["part_c"]["registration_path"]).resolve():
            raise ValueError("ineligible source is not the exact completed C source")
        reg = cohort._registration(row["registration_path"])
        cohort._registered_arm(row["source_path"], row["source_job"], reg)
        if sha(Path(row["source_path"]) / "manifest.json") != row["source_manifest_sha256"]:
            raise ValueError("ineligible source changed")
        observed = _evidence().opportunity(row["source_path"], row["source_job"], reg)
        cohort._same(observed, row["opportunity"], "conditional eligibility changed")
        if observed["eligible"]:
            raise ValueError("eligible branch was silently excluded")
    return path.parent, data


def run_one(registration, job_id):
    root, data = read_registration(registration)
    job = next((j for j in data["jobs"] if j["id"] == job_id), None)
    if job is None or (root / "STOP.json").exists():
        raise ValueError("unknown or stopped duration job")
    allocations = root / "allocations"
    allocations.mkdir(exist_ok=True)
    if {p.stem for p in allocations.glob("*.json")}-set(data["new_job_ids"]):
        raise ValueError("unregistered allocation changes bounded duration budget")
    with (allocations / f"{job_id}.json").open("x") as stream:
        json.dump(dict(job_id=job_id, allocated_simulated_seconds=2400, automatic_retries=0,
            registration_sha256=data["registration_sha256"]), stream)
    try:
        path = root / "jobs" / job_id
        result = paired.evaluate_arm(job=job, registration=data, output=path, policy=_policy().policy_factory(job, data))
        source = data["continuations"][job_id]
        branch = job["policy_spec"]["duration_branch"]["branch_time_s"]
        cohort._same(_evidence().branch_prefix(path, branch), source["opportunity"]["branch_prefix"],
                     "actual release history differs before registered branch")
        life = _evidence().score_release(path, job)
        contract = command_contract(path, LifecycleProfile(**job["policy_spec"]["profile"]))
        if not life["duration_deliberate_release_verified"] or not contract["passed"]:
            raise ValueError("duration branch assignment or actual command contract failed")
        write_json(path / "duration_contract.json", dict(lifecycle=life, command_contract=contract))
        return result
    except Exception as exc:
        write_json(root / "STOP.json", dict(job_id=job_id, reason=str(exc), automatic_retries=0))
        raise


def analyze(registration):
    root, data = read_registration(registration, require_current=False)
    if (root / "STOP.json").exists():
        raise ValueError("stopped duration block cannot become an accepted complete comparison")
    results, pending = [], []
    for job in data["jobs"]:
        manifest = root / "jobs" / job["id"] / "manifest.json"
        if not manifest.exists() or json.loads(manifest.read_text()).get("status") != "complete":
            pending.append(job["id"])
            continue
        results.append(_evidence().compare_case(root, data, job))
    answer = dict(schema="registered_duration_branch_results/1", complete=len(results) == len(data["jobs"]) and not pending,
        results=results, pending=pending, independent_exposed_seed_count=len(results),
        maximum_new_simulated_seconds=data["maximum_simulated_seconds"],
        actual_new_complete_simulated_seconds=2400*len(results), reused_continuations_are_not_replications=True,
        original_criteria_unchanged=True, optimal_duration_established=False, population_generalization_established=False,
        ineligible_opportunities=data["ineligible_opportunities"], automatic_successors=False)
    write_json(root / "analysis.json", answer)
    return dict(analysis_path=str(root / "analysis.json"), complete=answer["complete"], pending=pending,
        results=[dict(job_id=r["job_id"], benefit=r["benefit"], duration_contrast=r["duration_contrast"]) for r in results])
