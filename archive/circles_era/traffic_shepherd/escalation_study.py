"""Bounded escalation test plan, one-job execution, and verified result analysis.

Registering a panel starts no simulation. Execution requires one explicit job ID,
charges its full allocation before launch, and never retries or starts a successor.
Future wall/predictor stages are specified in the plan, not claimed implemented.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from math import isclose, isfinite
from pathlib import Path
import shutil

from .config import CONFIG_PATH, ROOT, load_config
from .escalation_evidence import paired_effect, formation_delivery, probability_metrics, seed_cluster_summary
from .escalation_observations import candidate_geometry, report_from_payload
from .reporting import RSUReportBuffer
from .seeds import exclusion_provenance, require_development_case
from .workflow import evaluate, sha, write_json


PLAN_PATH = ROOT / "protocols/escalation_test_plan_v1.json"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def load_plan(path=PLAN_PATH):
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if plan["schema"] != "escalation_test_plan/1":
        raise ValueError("unsupported escalation plan")
    if plan["scope"]["congestion_required"] is not False or plan["scope"]["damping_must_outperform_idm"] is not False:
        raise ValueError("escalation must allow incremental benefit in any traffic state")
    if plan["scope"]["penetrations"] != [.05, .1, .2, .3] or plan["scope"]["maximum_active_walls"] != 1:
        raise ValueError("this registered test family covers 5-30% and one active wall")
    seen = set()
    for stage in plan["stages"]:
        if stage["id"] in seen or not set(stage["depends_on"]) <= seen:
            raise ValueError("invalid stage dependency graph")
        if not stage["tests"] or not stage["deliverable"] or not stage["advance_rule"]:
            raise ValueError("every stage needs concrete tests, evidence and an advance rule")
        seen.add(stage["id"])
    return plan


def panel_jobs(plan, panel):
    """Matched seeds across loads, with paired native runs for every assignment."""
    if panel not in {"pilot", "penetration_extension"}:
        raise ValueError("choose a declared bounded development panel")
    penetrations = [.2] if panel == "pilot" else [.05, .1, .3]
    jobs = []
    for seed in plan["characterization"]["seeds"]:
        for loading in plan["characterization"]["mainline_vph"]:
            require_development_case(seed, loading, "exposed")
            for p in penetrations:
                for arm in ("native", "damping"):
                    jobs.append(dict(id=f"s{seed}_m{loading}_p{round(100*p):02d}_{arm}",
                                     seed=seed, loading=loading, penetration=p, controller_name=arm,
                                     role="exposed", duration=2400))
    return jobs


def current_identity():
    config = load_config()
    if sha(config.model_path) != config.model_sha256:
        raise ValueError("adopted model identity changed")
    binary = shutil.which("sumo")
    if binary is None:
        raise RuntimeError("SUMO must be on PATH before freezing an executable panel")
    return dict(config_sha256=sha(CONFIG_PATH), model_sha256=config.model_sha256,
                sumo_binary_sha256=sha(binary),
                network_sha256=sha(config.network_path), scenario_template_sha256=sha(config.scenario_template_path),
                active_source_sha256={p.name: sha(p) for p in sorted((ROOT / "traffic_shepherd").glob("*.py"))})


def register(output, *, panel="pilot"):
    plan = load_plan()
    jobs = panel_jobs(plan, panel)
    identity = current_identity()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("registration requires a new directory; existing evidence is preserved")
    registration = dict(schema="escalation_panel/1", panel=panel, plan_sha256=digest(plan), plan=plan,
                        registered_at_utc=datetime.now(timezone.utc).isoformat(), identity=identity,
                        jobs=jobs, max_attempts=len(jobs), max_simulated_seconds=sum(j["duration"] for j in jobs),
                        automatic_retries=0, automatic_successors=False, wall_runs=0, training_runs=0,
                        seed_exclusions=exclusion_provenance(), purpose="characterize_fixed_damping_and_capture_opportunities",
                        baseline_performance_pass_required=False, congestion_required=False)
    registration["registration_sha256"] = digest(registration)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "registration.json", registration)
    return dict(registration=str(output / "registration.json"), jobs=len(jobs),
                allocated_simulated_seconds=registration["max_simulated_seconds"], simulations_started=0)


def read_registration(path, *, require_current=True):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    content = {k: v for k,v in data.items() if k != "registration_sha256"}
    if data.get("schema") != "escalation_panel/1" or digest(content) != data.get("registration_sha256"):
        raise ValueError("registration content changed or schema unsupported")
    plan = data["plan"]
    if data["plan_sha256"] != digest(plan) or data["jobs"] != panel_jobs(plan, data["panel"]):
        raise ValueError("registered jobs differ from the fixed plan")
    if data["max_attempts"] != len(data["jobs"]) or data["max_simulated_seconds"] != sum(j["duration"] for j in data["jobs"]):
        raise ValueError("registration budget mismatch")
    if require_current and (data["identity"] != current_identity() or data["plan_sha256"] != digest(load_plan())):
        raise ValueError("source/model/scenario changed after registration; register a new version")
    return path.parent, data


def run_one(registration, job_id):
    root, data = read_registration(registration)
    matches = [j for j in data["jobs"] if j["id"] == job_id]
    if len(matches) != 1:
        raise ValueError("unknown registered job")
    job = matches[0]
    allocation = root / "allocations"
    allocation.mkdir(exist_ok=True)
    # Exclusive creation is the retry/budget lock, retained even if SUMO fails.
    with (allocation / f"{job_id}.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(job_id=job_id, allocated_simulated_seconds=job["duration"],
                       registration_sha256=data["registration_sha256"], retries=0), stream, indent=2)
    return evaluate(**{k:v for k,v in job.items() if k != "id"}, output=root / "jobs" / job_id,
                    collect_traffic=True)


def verified_run(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("run is incomplete or failed")
    for name, key in (("summary.json", "summary_sha256"), ("intervals.jsonl", "intervals_sha256"),
                      ("telemetry.jsonl", "telemetry_sha256")):
        if sha(root / name) != manifest.get(key):
            raise ValueError(f"changed or unverified artifact: {name}")
    if manifest.get("offline_traffic_observations") and sha(root / "traffic.jsonl") != manifest.get("traffic_sha256"):
        raise ValueError("changed offline traffic observations")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    keys = ("braking_vehicle_seconds", "acceleration_energy", "ttc_below_3_vehicle_seconds",
            "collision_involvements", "starting_teleports", "requested_not_arrived_vehicle_seconds")
    sums = {key: 0. for key in keys}
    previous = 0.
    with (root / "intervals.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["start_s"] != previous or row["end_s"] != previous + 1.:
                raise ValueError("missing or irregular physical outcome interval")
            previous = row["end_s"]
            for key in keys:
                value = row[key]
                if not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
                    raise ValueError("invalid measured outcome interval")
                sums[key] += value
    if previous != manifest["duration_s"]:
        raise ValueError("incomplete physical outcome horizon")
    for key in keys:
        expected = (summary["system_seconds_per_requested_vehicle"] * summary["population"]
                    if key == "requested_not_arrived_vehicle_seconds" else summary["totals"][key])
        if not isclose(sums[key], expected, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError(f"summary disagrees with measured intervals: {key}")
    return manifest, summary


def analyze(registration):
    root, data = read_registration(registration, require_current=False)
    ready, pending, failed = {}, [], []
    for job in data["jobs"]:
        path = root / "jobs" / job["id"]
        if not path.exists():
            (failed if (root / "allocations" / f"{job['id']}.json").exists() else pending).append(job["id"])
            continue
        try:
            manifest, summary = verified_run(path)
            for key in ("config_sha256", "model_sha256", "active_source_sha256", "sumo_binary_sha256"):
                if manifest[key] != data["identity"][key]:
                    raise ValueError(f"run does not match registration: {key}")
            if manifest["scenario"]["network_sha256"] != data["identity"]["network_sha256"]:
                raise ValueError("run network differs from registration")
            for key, field in (("seed", "seed"), ("mainline_vph", "loading"), ("penetration", "penetration"),
                               ("controller", "controller_name"), ("duration_s", "duration")):
                if manifest[key] != job[field]:
                    raise ValueError(f"wrong job identity: {key}")
            ready[job["id"]] = (manifest, summary)
        except (ValueError, KeyError, OSError) as exc:
            failed.append(dict(job=job["id"], error=str(exc)))
    effects = []
    for job in data["jobs"]:
        if job["controller_name"] != "damping":
            continue
        reference = job["id"].removesuffix("damping") + "native"
        if job["id"] in ready and reference in ready:
            effects.append(paired_effect(*ready[reference], *ready[job["id"]]))
    return dict(status="characterized" if not pending and not failed else "incomplete",
                completed=len(ready), planned=len(data["jobs"]), pending=pending, failures=failed,
                paired_effects=effects, baseline_performance_pass_required=False,
                interpretation="performance characterizes the baseline; it does not authorize retraining or prove wall benefit")


def capture_geometry(run, *, times, max_groups=256):
    manifest, _ = verified_run(run)
    if manifest["controller"] != "damping" or manifest.get("mechanics_only") is not False:
        raise ValueError("capture from a complete damping-only trajectory")
    if not times or len(times) != len(set(times)) or any(type(t) is not int or not 0 <= t < manifest["duration_s"] for t in times):
        raise ValueError("distinct predeclared control times within the recorded horizon required")
    plan = load_plan()
    definition = plan["geometry_capture"]
    buffer = RSUReportBuffer(max_age_s=definition["max_report_age_s"], history_s=definition["history_s"])
    packets = defaultdict(list)
    # Read only CAV messages; full-state traffic.jsonl is deliberately never read.
    with (Path(run) / "telemetry.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("kind") == "cav_report":
                report = report_from_payload(row)
                packets[report.sent_at_s].append(report)
    results = []
    requested = set(times)
    for now in sorted(set(packets) | requested):
        for report in packets.get(now, ()):
            # The registered initial collector assumes zero communication delay.
            buffer.receive(report, received_at_s=now)
        if now in requested:
            results.append(dict(time_s=now, **candidate_geometry(buffer.reports(now_s=now), now_s=now,
                required_lanes=definition["required_lanes"], recruitment_band_m=definition["recruitment_band_m"],
                max_report_age_s=definition["max_report_age_s"], max_groups=max_groups)))
    return dict(source_telemetry_sha256=manifest["telemetry_sha256"], communication="zero_delay_diagnostic",
                observations=results, physical_assembly_tested=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan", help="all stages, evidence requirements and stopping rules; no simulation")
    lifecycle_mreg = commands.add_parser("register-lifecycle-mechanics")
    lifecycle_mreg.add_argument("--output", required=True)
    lifecycle_mrun = commands.add_parser("run-lifecycle-mechanics")
    lifecycle_mrun.add_argument("--registration", required=True)
    lifecycle_mrun.add_argument("--case-id", required=True)
    lifecycle_reg = commands.add_parser("register-lifecycle-wall")
    lifecycle_reg.add_argument("--output", required=True)
    lifecycle_reg.add_argument("--mechanics-registration", required=True)
    for name in ("run-lifecycle-wall", "analyze-lifecycle-wall"):
        lifecycle_cmd = commands.add_parser(name)
        lifecycle_cmd.add_argument("--registration", required=True)
        if name == "run-lifecycle-wall":
            lifecycle_cmd.add_argument("--job-id", required=True)
    reg = commands.add_parser("register", help="write a bounded job allocation; starts no runs")
    reg.add_argument("--output", required=True)
    reg.add_argument("--panel", choices=("pilot", "penetration_extension"), default="pilot")
    observer_reg = commands.add_parser("register-observer-replays", help="freeze five existing explanatory arms; no simulation")
    observer_reg.add_argument("--output", required=True)
    ts_mech = commands.add_parser("register-target-speed-mechanics")
    ts_mech.add_argument("--output", required=True)
    ts_mech_run = commands.add_parser("run-target-speed-mechanics")
    ts_mech_run.add_argument("--registration", required=True)
    ts_mech_run.add_argument("--case-id", default="uncapped_target_complete")
    ts_reg = commands.add_parser("register-target-speed")
    ts_reg.add_argument("--output", required=True)
    ts_reg.add_argument("--mechanics-registration", required=True)
    for name in ("run-target-speed", "analyze-target-speed", "resolve-target-speed"):
        ts_cmd = commands.add_parser(name)
        ts_cmd.add_argument("--registration", required=True)
        if name == "run-target-speed":
            ts_cmd.add_argument("--job-id", required=True)
    ta_mech = commands.add_parser("register-target-adjust-mechanics")
    ta_mech.add_argument("--output", required=True)
    ta_mech_run = commands.add_parser("run-target-adjust-mechanics")
    ta_mech_run.add_argument("--registration", required=True)
    ta_mech_run.add_argument("--case-id", default="hold_adjust_down_complete")
    ta_reg = commands.add_parser("register-target-adjust")
    ta_reg.add_argument("--output", required=True)
    ta_reg.add_argument("--mechanics-registration", required=True)
    for name in ("run-target-adjust", "analyze-target-adjust", "resolve-target-adjust"):
        ta_cmd = commands.add_parser(name)
        ta_cmd.add_argument("--registration", required=True)
        if name == "run-target-adjust":
            ta_cmd.add_argument("--job-id", required=True)
    he_mech = commands.add_parser("register-hold-extend-mechanics")
    he_mech.add_argument("--output", required=True)
    he_mech_run = commands.add_parser("run-hold-extend-mechanics")
    he_mech_run.add_argument("--registration", required=True)
    he_mech_run.add_argument("--case-id", default="hold26_ext10_complete")
    he_reg = commands.add_parser("register-hold-extend")
    he_reg.add_argument("--output", required=True)
    he_reg.add_argument("--mechanics-registration", required=True)
    for name in ("run-hold-extend", "analyze-hold-extend", "resolve-hold-extend"):
        he_cmd = commands.add_parser(name)
        he_cmd.add_argument("--registration", required=True)
        if name == "run-hold-extend":
            he_cmd.add_argument("--job-id", required=True)
    dissolution_reg = commands.add_parser("register-report-dissolution", help="freeze two focused treatment arms with verified reused damping")
    dissolution_reg.add_argument("--output", required=True)
    gentle_mechanics_reg = commands.add_parser("register-gentle-mechanics")
    gentle_mechanics_reg.add_argument("--output", required=True)
    gentle_mechanics_run = commands.add_parser("run-gentle-mechanics")
    gentle_mechanics_run.add_argument("--registration", required=True)
    gentle_mechanics_run.add_argument("--case-id", required=True)
    gentle_reg = commands.add_parser("register-gentle-wall")
    gentle_reg.add_argument("--output", required=True)
    gentle_reg.add_argument("--mechanics-registration", required=True)
    gentle_reg.add_argument("--replaces-failed-registration", help="explicit separate replacement budget after authorization; never automatic")
    brake_pre = commands.add_parser("preflight-brake-hold", help="prove existing policy reuse without simulation")
    brake_pre.add_argument("--output", required=True)
    brake_mech_reg = commands.add_parser("register-brake-hold-mechanics")
    brake_mech_reg.add_argument("--output", required=True)
    brake_mech_run = commands.add_parser("run-brake-hold-mechanics")
    brake_mech_run.add_argument("--registration", required=True)
    brake_mech_run.add_argument("--case-id", required=True)
    duration_reg = commands.add_parser("register-duration")
    duration_reg.add_argument("--output", required=True)
    duration_reg.add_argument("--b-registration", required=True)
    duration_reg.add_argument("--c-registration", required=True)
    for name in ("run-duration", "analyze-duration"):
        duration_cmd = commands.add_parser(name)
        duration_cmd.add_argument("--registration", required=True)
        if name == "run-duration":
            duration_cmd.add_argument("--job-id", required=True)
    cohort_reg = commands.add_parser("register-cohort")
    cohort_reg.add_argument("--output", required=True)
    cohort_reg.add_argument("--b-registration", required=True)
    for name in ("run-cohort", "analyze-cohort"):
        cohort_cmd = commands.add_parser(name)
        cohort_cmd.add_argument("--registration", required=True)
        if name == "run-cohort":
            cohort_cmd.add_argument("--job-id", required=True)
    brake_reg = commands.add_parser("register-brake-hold")
    brake_reg.add_argument("--output", required=True)
    brake_reg.add_argument("--mechanics-registration", required=True)
    brake_reg.add_argument("--preflight", required=True)
    for name in ("run-brake-hold", "analyze-brake-hold"):
        brake_cmd = commands.add_parser(name)
        brake_cmd.add_argument("--registration", required=True)
        if name == "run-brake-hold":
            brake_cmd.add_argument("--job-id", required=True)
    for name in ("run-gentle-wall", "analyze-gentle-wall"):
        gentle_cmd = commands.add_parser(name)
        gentle_cmd.add_argument("--registration", required=True)
        if name == "run-gentle-wall":
            gentle_cmd.add_argument("--job-id", required=True)
    for name in ("run-report-dissolution", "analyze-report-dissolution"):
        dissolution_cmd = commands.add_parser(name)
        dissolution_cmd.add_argument("--registration", required=True)
        if name == "run-report-dissolution":
            dissolution_cmd.add_argument("--job-id", required=True)
    for name in ("run-observer-replay", "analyze-observer-replays"):
        observer_cmd = commands.add_parser(name)
        observer_cmd.add_argument("--registration", required=True)
        if name == "run-observer-replay":
            observer_cmd.add_argument("--job-id", required=True)
    for name in ("run", "analyze"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--registration", required=True)
        if name == "run":
            cmd.add_argument("--job-id", required=True)
    geometry = commands.add_parser("geometry")
    geometry.add_argument("--run", required=True)
    geometry.add_argument("--times", nargs="+", type=int, required=True)
    score = commands.add_parser("score", help="offline measurements only, never simulator or learner execution")
    score.add_argument("--kind", choices=("formation", "probabilities", "seed-summary"), required=True)
    score.add_argument("--input", required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        result = load_plan()
    elif args.command in {"register-duration", "run-duration", "analyze-duration"}:
        from . import duration_campaign
        if args.command == "register-duration":
            result = duration_campaign.register(args.output, args.b_registration, args.c_registration)
        elif args.command == "run-duration":
            result = duration_campaign.run_one(args.registration, args.job_id)
        else:
            result = duration_campaign.analyze(args.registration)
    elif args.command in {"register-cohort", "run-cohort", "analyze-cohort"}:
        from . import cohort_campaign
        if args.command == "register-cohort":
            result = cohort_campaign.register(args.output, args.b_registration)
        elif args.command == "run-cohort":
            result = cohort_campaign.run_one(args.registration, args.job_id)
        else:
            result = cohort_campaign.analyze(args.registration)
    elif args.command in {"register-lifecycle-mechanics", "run-lifecycle-mechanics"}:
        from . import lifecycle_mechanics
        result = (lifecycle_mechanics.register(args.output) if args.command == "register-lifecycle-mechanics"
                  else lifecycle_mechanics.run_one(args.registration, args.case_id))
    elif args.command in {"register-lifecycle-wall", "run-lifecycle-wall", "analyze-lifecycle-wall"}:
        from . import lifecycle_study
        if args.command == "register-lifecycle-wall":
            result = lifecycle_study.register(args.output, args.mechanics_registration)
        elif args.command == "run-lifecycle-wall":
            result = lifecycle_study.run_one(args.registration, args.job_id)
        else:
            result = lifecycle_study.analyze(args.registration)
    elif args.command in {"register-brake-hold-mechanics", "run-brake-hold-mechanics"}:
        from . import brake_hold_mechanics
        result = (brake_hold_mechanics.register(args.output) if args.command == "register-brake-hold-mechanics"
                  else brake_hold_mechanics.run_one(args.registration, args.case_id))
    elif args.command in {"preflight-brake-hold", "register-brake-hold", "run-brake-hold", "analyze-brake-hold"}:
        from . import brake_hold_study
        if args.command == "preflight-brake-hold":
            result = brake_hold_study.preflight(args.output)
        elif args.command == "register-brake-hold":
            result = brake_hold_study.register(args.output, args.mechanics_registration, args.preflight)
        elif args.command == "run-brake-hold":
            result = brake_hold_study.run_one(args.registration, args.job_id)
        else:
            result = brake_hold_study.analyze(args.registration)
    elif args.command in {"register-gentle-mechanics", "run-gentle-mechanics"}:
        from . import gentle_mechanics
        result = (gentle_mechanics.register(args.output) if args.command == "register-gentle-mechanics"
                  else gentle_mechanics.run_one(args.registration, args.case_id))
    elif args.command in {"register-gentle-wall", "run-gentle-wall", "analyze-gentle-wall"}:
        from . import gentle_study
        if args.command == "register-gentle-wall":
            result = gentle_study.register(args.output, args.mechanics_registration,
                replaces_failed_registration=args.replaces_failed_registration)
        elif args.command == "run-gentle-wall":
            result = gentle_study.run_one(args.registration, args.job_id)
        else:
            result = gentle_study.analyze(args.registration)
    elif args.command in {"register-report-dissolution", "run-report-dissolution", "analyze-report-dissolution"}:
        from . import dissolution_study
        if args.command == "register-report-dissolution":
            result = dissolution_study.register(args.output)
        elif args.command == "run-report-dissolution":
            result = dissolution_study.run_one(args.registration, args.job_id)
        else:
            result = dissolution_study.analyze(args.registration)
    elif args.command in {"register-observer-replays", "run-observer-replay", "analyze-observer-replays"}:
        from . import observer_replays
        if args.command == "register-observer-replays":
            result = observer_replays.register(args.output)
        elif args.command == "run-observer-replay":
            result = observer_replays.run_one(args.registration, args.job_id)
        else:
            result = observer_replays.analyze(args.registration)
    elif args.command in {"register-target-speed-mechanics", "run-target-speed-mechanics"}:
        from . import target_speed_mechanics
        result = (target_speed_mechanics.register(args.output)
                  if args.command == "register-target-speed-mechanics"
                  else target_speed_mechanics.run_one(args.registration, args.case_id))
    elif args.command in {"register-target-speed", "run-target-speed", "analyze-target-speed",
                          "resolve-target-speed"}:
        from . import target_speed_study
        if args.command == "register-target-speed":
            result = target_speed_study.register(args.output, args.mechanics_registration)
        elif args.command == "run-target-speed":
            result = target_speed_study.run_one(args.registration, args.job_id)
        elif args.command == "resolve-target-speed":
            result = target_speed_study.resolve_verification(args.registration)
        else:
            result = target_speed_study.analyze(args.registration)
    elif args.command in {"register-target-adjust-mechanics", "run-target-adjust-mechanics"}:
        from . import target_adjust_mechanics
        result = (target_adjust_mechanics.register(args.output)
                  if args.command == "register-target-adjust-mechanics"
                  else target_adjust_mechanics.run_one(args.registration, args.case_id))
    elif args.command in {"register-hold-extend-mechanics", "run-hold-extend-mechanics"}:
        from . import hold_extend_mechanics
        result = (hold_extend_mechanics.register(args.output)
                  if args.command == "register-hold-extend-mechanics"
                  else hold_extend_mechanics.run_one(args.registration, args.case_id))
    elif args.command in {"register-hold-extend", "run-hold-extend", "analyze-hold-extend",
                          "resolve-hold-extend"}:
        from . import hold_extend_study
        if args.command == "register-hold-extend":
            result = hold_extend_study.register(args.output, args.mechanics_registration)
        elif args.command == "run-hold-extend":
            result = hold_extend_study.run_one(args.registration, args.job_id)
        elif args.command == "resolve-hold-extend":
            result = hold_extend_study.resolve_verification(args.registration)
        else:
            result = hold_extend_study.analyze(args.registration)
    elif args.command in {"register-target-adjust", "run-target-adjust", "analyze-target-adjust",
                          "resolve-target-adjust"}:
        from . import target_adjust_study
        if args.command == "register-target-adjust":
            result = target_adjust_study.register(args.output, args.mechanics_registration)
        elif args.command == "run-target-adjust":
            result = target_adjust_study.run_one(args.registration, args.job_id)
        elif args.command == "resolve-target-adjust":
            result = target_adjust_study.resolve_verification(args.registration)
        else:
            result = target_adjust_study.analyze(args.registration)
    elif args.command == "register":
        result = register(args.output, panel=args.panel)
    elif args.command == "run":
        result = run_one(args.registration, args.job_id)
    elif args.command == "analyze":
        result = analyze(args.registration)
    elif args.command == "geometry":
        result = capture_geometry(args.run, times=args.times)
    else:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        fn = {"formation": formation_delivery, "probabilities": probability_metrics, "seed-summary": seed_cluster_summary}[args.kind]
        result = fn(**payload)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
