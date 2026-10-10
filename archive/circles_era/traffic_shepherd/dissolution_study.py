"""Two treatment-only development checks against immutable verified damping."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

from .config import ROOT
from . import paired_experiments as paired
from .report_dissolution import VERSION, policy_factory
from .escalation_evidence import BenefitCriteria, benefit_label, paired_effect
from .formation import FormationProfile, make_proposal
from .formation_evidence import score_formation_files
from .escalation_observations import report_from_payload
from .workflow import sha, write_json


REFERENCES = ROOT / "runs/active/observer_replay_v1"
ORIGINAL = ROOT / "runs/active/natural_wall_effects_v1/registration.json"


def reference_proof(path, identity):
    manifest, summary = paired.verified_arm(path)
    for key in ("config_sha256", "model_sha256", "network_sha256", "sumo_binary_sha256", "scenario_template_sha256"):
        if manifest[key] != identity[key]:
            raise ValueError(f"changed damping reference identity {key}")
    # Existing baseline modules must be byte-identical. Only the public command
    # dispatcher gains a new explicit subcommand; it is not imported by control.
    old = manifest["active_source_sha256"]
    changed = [name for name, checksum in old.items()
               if name != "escalation_study.py" and identity["active_source_sha256"].get(name) != checksum]
    if changed:
        raise ValueError(f"reference reuse cannot bridge changed baseline execution: {changed}")
    checked = json.loads((REFERENCES / "comparisons" / f"{manifest['job_id']}.json").read_text())
    if (not checked["exact_replay"] or not checked["diagnostic_channel_validated"]
            or checked["replay_manifest_file_sha256"] != sha(Path(path) / "manifest.json")):
        raise ValueError("damping reference is not the verified observer replay")
    return manifest, summary, dict(path=str(path), manifest_sha256=sha(Path(path) / "manifest.json"),
        actual_reference_source_sha256=old, identical_existing_execution_modules=True,
        dispatcher_exception="escalation_study.py adds an explicit research command, no control behavior",
        new_source_modules=sorted(set(identity["active_source_sha256"])-set(old)),
        baseline_rerun_required=False, original_outcomes_modified=False)


def register(output):
    identity = paired.current_identity()
    original = json.loads(ORIGINAL.read_text())
    jobs, references = [], {}
    for seed in (4107, 4108):
        job = deepcopy(next(j for j in original["jobs"] if j["id"] == f"s{seed}_m2700_p20_t600_wall"))
        job["id"] = f"s{seed}_m2700_p20_t600_report_dissolution"
        job["policy_spec"]["profile"]["version"] = VERSION
        paired.validate_job(job)
        _, _, proof = reference_proof(REFERENCES / "jobs" / job["reference_job_id"], identity)
        references[job["id"]] = proof
        jobs.append(job)
    data = dict(schema="report_dissolution_study/1", registered_at_utc=datetime.now(timezone.utc).isoformat(),
        identity=identity, jobs=jobs, references=references, communication=original["communication"],
        assignment="registered_experimental_assignment_not_trained_benefit_gate", interaction_diagnostics=True,
        criteria=original["criteria"], policy_version=VERSION, maximum_attempts=2,
        maximum_simulated_seconds=4800, automatic_retries=0, automatic_successors=False,
        protocol_path="docs/RSU_DISSOLUTION_CHECK.md", protocol_sha256=sha(ROOT / "docs/RSU_DISSOLUTION_CHECK.md"),
        purpose="test_report_visible_native_handback_sensing_gap_hypothesis_not_a_parameter_search")
    data["registration_sha256"] = paired.digest(data)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "registration.json", data)
    return dict(registration=str(output / "registration.json"), jobs=[j["id"] for j in jobs],
                maximum_simulated_seconds=4800, reused_damping_arms=2, simulations_started=0)


def read_registration(path, *, require_current=True):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data.get("schema") != "report_dissolution_study/1" or data.get("registration_sha256") != paired.digest(
            {k: v for k, v in data.items() if k != "registration_sha256"}):
        raise ValueError("dissolution registration changed")
    if (data["maximum_attempts"] != 2 or len(data["jobs"]) != 2 or data["maximum_simulated_seconds"] != 4800
            or data["automatic_retries"] != 0 or data["automatic_successors"]):
        raise ValueError("dissolution budget changed")
    if require_current and data["identity"] != paired.current_identity():
        raise ValueError("source changed after dissolution registration")
    for job in data["jobs"]:
        paired.validate_job(job)
        if job["duration_s"] != 2400 or job["policy_spec"]["profile"]["version"] != VERSION:
            raise ValueError("dissolution job/horizon changed")
    return path.parent, data


def run_one(registration, job_id):
    root, data = read_registration(registration)
    jobs = [j for j in data["jobs"] if j["id"] == job_id]
    if len(jobs) != 1 or (root / "STOP.json").exists():
        raise ValueError("unknown job or stopped study")
    proof = data["references"][job_id]
    if sha(Path(proof["path"]) / "manifest.json") != proof["manifest_sha256"]:
        raise ValueError("immutable reference changed")
    (root / "allocations").mkdir(exist_ok=True)
    with (root / "allocations" / f"{job_id}.json").open("x") as stream:
        json.dump(dict(job_id=job_id, allocated_simulated_seconds=2400, retries=0,
                       registration_sha256=data["registration_sha256"]), stream, indent=2)
    try:
        return paired.evaluate_arm(job=jobs[0], registration=data, output=root / "jobs" / job_id,
                                   policy=policy_factory(jobs[0], data))
    except Exception as exc:
        write_json(root / "STOP.json", dict(job_id=job_id, reason=str(exc), retries=0))
        raise


def compare_job(root, data, job):
    reference = Path(data["references"][job["id"]]["path"])
    control_m, control, proof = reference_proof(reference, data["identity"])
    treatment_path = root / "jobs" / job["id"]
    treatment_m, treatment = paired.verified_arm(treatment_path)
    if treatment_m["registration_sha256"] != data["registration_sha256"] or any(
            treatment_m.get(k) != v for k, v in data["identity"].items()):
        raise ValueError("treatment execution does not bind the registered source and identity")
    left = json.loads((reference / "decision_snapshots.json").read_text())["600"]
    right = json.loads((treatment_path / "decision_snapshots.json").read_text())["600"]
    for key in ("decision_time_s", "predecision_state_sha256", "predecision_report_evidence_sha256",
                "predecision_report_prefix_sha256", "predecision_command_prefix_sha256"):
        if left[key] != right[key]:
            raise ValueError(f"reused reference prefix mismatch {key}")
    if control_m["exogenous_future_sha256"] != treatment_m["exogenous_future_sha256"]:
        raise ValueError("reused reference exogenous future mismatch")
    # This explicit derived comparison view binds unchanged damping outcomes to
    # the new treatment proposal. It is saved separately; original bytes stay put.
    derived = dict(control_m, active_source_sha256=data["identity"]["active_source_sha256"],
        proposal_sha256=right["proposal_sha256"], decision_time_s=600.,
        predecision_state_sha256=left["predecision_state_sha256"])
    effect = paired_effect(derived, control, treatment_m, treatment)
    label = benefit_label(effect, BenefitCriteria(**data["criteria"]))
    transitions = [json.loads(line) for line in (treatment_path / "policy.jsonl").open()]
    bridge = [r for r in transitions if r.get("reason") == "report_bridge_to_local_feedback"]
    proposal = make_proposal([report_from_payload(r) for r in right["evidence"]["reports"]], now_s=600.,
        required_lanes=tuple(job["policy_spec"]["required_lanes"]), member_ids=tuple(job["policy_spec"]["member_ids"]),
        profile=FormationProfile(**job["policy_spec"]["profile"]))
    delivery = score_formation_files(treatment_path / "formation.jsonl", treatment_path / "telemetry.jsonl", proposal=proposal)
    # Preserve the legacy scorer and its fixed-duration convention verbatim.
    # The new registered contract permits early release to numeric local
    # feedback, so assess that distinct transition explicitly rather than
    # weakening the historical formation/taper contract.
    numeric_bridge_release = bool(bridge and delivery["terminal_reason"] == "report_bridge_local_numeric_feedback_available"
        and 0 < delivery["taper_seconds"] <= proposal.profile.taper_s
        and not delivery["terminal_ownership"]["native_members"]
        and set(delivery["failure_reasons"]) <= {"taper_incomplete_without_explicit_native_handback"})
    mechanism = dict(version=VERSION, legacy_fixed_taper_score=delivery,
        registered_early_numeric_release=numeric_bridge_release,
        mechanical_contract_passed=bool(delivery["delivered"] or (numeric_bridge_release
            and delivery["compact_hold_delivered"] and delivery["release_ownership_verified"])),
        physical_benefit_not_implied=True)
    return dict(job_id=job["id"], effect=effect, benefit=label, reference_reuse_proof=proof,
        mechanical_delivery=mechanism,
        derived_reference_view=derived, derived_view_not_original_execution_manifest=True,
        predecision_prefix_exact=True, transition_command_seconds=len(bridge),
        transition_start_s=bridge[0]["time_s"] if bridge else None,
        transition_end_s=bridge[-1]["time_s"]+1 if bridge else None,
        treatment_manifest_sha256=sha(treatment_path / "manifest.json"),
        new_independent_seed=False, qualified_gate=False)


def analyze(registration):
    root, data = read_registration(registration, require_current=False)
    results = [compare_job(root, data, j) for j in data["jobs"]]
    write_json(root / "analysis.json", dict(schema="report_dissolution_result/1", results=results,
        maximum_simulated_seconds=4800, automatic_successors=False, new_independent_seeds=0,
        analysis_source_sha256=sha(__file__)))
    return results
