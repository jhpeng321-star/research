"""Five existing-arm observer replays, with no new treatment or retry budget."""
from pathlib import Path
import json

from . import paired_experiments as paired
from .config import ROOT
from .workflow import sha, write_json


SOURCE = ROOT / "runs/active/natural_wall_effects_v1/registration.json"
JOB_IDS = (
    "s4107_m2700_p20_damping",
    "s4107_m2700_p20_t600_wall",
    "s4107_m2700_p20_t600_assembly_only",
    "s4108_m2700_p20_damping",
    "s4108_m2700_p20_t600_wall",
)
CONTRACT = "observer_replay_contract.json"


def selected_jobs(source):
    by_id = {j["id"]: j for j in source["jobs"]}
    jobs = [by_id[name] for name in JOB_IDS]
    if any(j["duration_s"] != 2400 or j["loading"] != 2700 or j["penetration"] != .2
           or j["decision_time_s"] != 600 for j in jobs):
        raise ValueError("existing explanatory case identity changed")
    return jobs


def register(output):
    """Register exactly the five previously exposed arms after observer checks."""
    original_root, original = paired.read_registration(SOURCE, require_current=False)
    jobs = selected_jobs(original)
    sources = {}
    for job in jobs:
        directory = original_root / "jobs" / job["id"]
        manifest, _ = paired.verified_arm(directory)
        sources[job["id"]] = dict(path=str(directory), manifest_sha256=sha(directory / "manifest.json"),
                                    scenario_identity=manifest["scenario_identity"])
    original_identity = original["identity"]
    identity = paired.current_identity()
    for key in ("config_sha256", "model_sha256", "network_sha256", "sumo_binary_sha256",
                "scenario_template_sha256"):
        if identity[key] != original_identity[key]:
            raise ValueError(f"observer replay cannot change {key}")
    for name in ("damping.py", "formation.py", "ownership.py", "sensing.py", "sensor_batch.py",
                 "scenarios.py", "metrics.py", "simulation.py", "reporting.py"):
        if identity["active_source_sha256"][name] != original_identity["active_source_sha256"][name]:
            raise ValueError(f"observer replay cannot change frozen execution module {name}")
    registration = paired.register(output, jobs=jobs, criteria=original["criteria"],
        policy_version=original["policy_version"],
        max_report_age_s=original["communication"]["max_report_age_s"],
        history_s=original["communication"]["history_s"], interaction_diagnostics=True,
        hypotheses=[
            "Observer instrumentation preserves every saved command, acquired report and physical outcome.",
            "Offline interaction identities resolve delayed cost contributors and the time at which permitted CAV reports could reveal them.",
            "Assembly-only versus HOLD isolates continuation effects after identical initial assembly; two seeds are explanatory cases, not independent validation.",
        ], stopping_rules=[
            "Five existing arms, 12000 simulated seconds maximum; no automatic retry or successor.",
            "First damping replay must reproduce exactly before remaining jobs can launch.",
            "Any replay mismatch or failed run stops remaining launches; preserve allocation and all output.",
            "No controller changes, fitted predictor, acceptance-data inspection, or efficacy claims from this observer block.",
        ])
    contract = dict(schema="observer_replay_contract/1", source_registration=str(SOURCE),
        source_registration_sha256=sha(SOURCE), sources=sources,
        paired_registration_sha256=registration["registration_sha256"],
        maximum_simulated_seconds=12000, maximum_attempts=5, automatic_retries=0,
        policy_history_s=original["communication"]["history_s"], regional_offline_history_s=60,
        first_integrity_job=JOB_IDS[0],
        diagnostic_information_boundary="Full state and identity crosswalks are offline only; gate candidates use acquired anonymous CAV reports at or before each decision.")
    contract["contract_sha256"] = paired.digest(contract)
    write_json(Path(output) / CONTRACT, contract)
    return dict(registration=str(Path(output).resolve() / "registration.json"),
                jobs=list(JOB_IDS), maximum_simulated_seconds=12000, simulations_started=0)


def read_contract(registration, *, require_current=True):
    root, data = paired.read_registration(registration, require_current=require_current)
    contract = json.loads((root / CONTRACT).read_text(encoding="utf-8"))
    if contract.get("contract_sha256") != paired.digest({k: v for k, v in contract.items() if k != "contract_sha256"}):
        raise ValueError("observer contract changed")
    if (contract.get("schema") != "observer_replay_contract/1"
            or contract["paired_registration_sha256"] != data["registration_sha256"]
            or not data.get("interaction_diagnostics")
            or [j["id"] for j in data["jobs"]] != list(JOB_IDS)
            or data["maximum_simulated_seconds"] != 12000):
        raise ValueError("observer block differs from its five-arm budget")
    source_path = Path(contract["source_registration"])
    if sha(source_path) != contract["source_registration_sha256"]:
        raise ValueError("source registration changed")
    _, source = paired.read_registration(source_path, require_current=False)
    if selected_jobs(source) != data["jobs"]:
        raise ValueError("observer replay changed an original job")
    for item in contract["sources"].values():
        if sha(Path(item["path"]) / "manifest.json") != item["manifest_sha256"]:
            raise ValueError("source evidence manifest changed")
    return root, data, contract


def run_one(registration, job_id):
    from .observer_replay_evidence import compare_replay, validate_diagnostics

    root, _, contract = read_contract(registration)
    if job_id not in JOB_IDS:
        raise ValueError("job is outside the five-arm replay block")
    first = root / "comparisons" / f"{JOB_IDS[0]}.json"
    if job_id != JOB_IDS[0] and not first.exists():
        raise ValueError("first damping replay must pass exact comparison before other launches")
    if job_id != JOB_IDS[0]:
        passed = json.loads(first.read_text(encoding="utf-8"))
        if (not passed.get("exact_replay")
                or not passed.get("diagnostic_channel_validated")
                or passed.get("original_manifest_file_sha256") != contract["sources"][JOB_IDS[0]]["manifest_sha256"]
                or passed.get("replay_manifest_file_sha256") != sha(root / "jobs" / JOB_IDS[0] / "manifest.json")):
            raise ValueError("first replay integrity evidence is absent or changed")
    try:
        result = paired.run_one(registration, job_id)
        comparison = compare_replay(contract["sources"][job_id]["path"], root / "jobs" / job_id)
        if not comparison.get("exact_replay", False):
            raise ValueError(f"observer replay mismatch: {comparison}")
        diagnostic = validate_diagnostics(root / "jobs" / job_id)
        if not diagnostic.get("diagnostic_channel_validated"):
            raise ValueError("new diagnostic evidence failed independent reconstruction")
        comparison.update(diagnostic_channel_validated=True, diagnostic_validation=diagnostic)
        (root / "comparisons").mkdir(exist_ok=True)
        write_json(root / "comparisons" / f"{job_id}.json", comparison)
        return dict(job_id=job_id, comparison=comparison, physical_outcomes=result)
    except Exception as exc:
        write_json(root / "STOP.json", dict(job_id=job_id, reason=str(exc), automatic_retry=False))
        raise


def analyze(registration):
    root, _, contract = read_contract(registration, require_current=False)
    completed, pending, failed = {}, [], []
    for name in JOB_IDS:
        path = root / "jobs" / name
        if not (root / "allocations" / f"{name}.json").exists():
            pending.append(name)
            continue
        try:
            paired.verified_arm(path)
            comparison = json.loads((root / "comparisons" / f"{name}.json").read_text(encoding="utf-8"))
            if (not comparison.get("exact_replay") or not comparison.get("diagnostic_channel_validated")
                    or comparison["original_manifest_file_sha256"] != contract["sources"][name]["manifest_sha256"]
                    or comparison["replay_manifest_file_sha256"] != sha(path / "manifest.json")):
                raise ValueError("saved comparison does not bind current complete replay artifacts")
            completed[name] = comparison
        except (OSError, ValueError, KeyError) as exc:
            failed.append(dict(job_id=name, reason=str(exc)))
    return dict(completed=completed, pending=pending, failed=failed,
                stopped=(root / "STOP.json").exists(), maximum_simulated_seconds=12000,
                efficacy_evidence_is_unchanged=True, new_independent_cases=0)
