"""Bounded cap-by-HOLD experiment, with exact saved-policy reuse proofs."""
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from . import paired_experiments as paired
from .brake_hold_policy import BrakeHoldProfile, VERSION, policy_factory
from .config import ROOT
from .damping import CirclesController
from .dissolution_study import reference_proof, REFERENCES
from .escalation_evidence import BenefitCriteria, benefit_label, paired_effect
from .escalation_observations import report_from_payload
from .formation_evidence import _geometry, _handback
from .reporting import RSUReportBuffer
from .workflow import sha, write_json

SCHEMA = "brake_hold_study/1"
OLD = ROOT / "runs/active/gentle_wall_v1_logging_repair"
PROTOCOL = "docs/BRAKE_HOLD_TEST.md"
CAPS = (-.3, -3.)
HOLDS = (0, 20, 40)


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def cell_id(seed, cap, hold):
    return f"s{seed}_b{'03' if cap == -.3 else '30'}_h{hold}"


def binding_mode(job):
    p = job["policy_spec"]["profile"]
    if p["acceleration_min_mps2"] == -.3:
        if p["hold_s"] == 20:
            return "reused"
        if job["seed"] == 4108:
            return "trace_equivalent"
    return "simulate"


def jobs():
    original = json.loads((OLD / "registration.json").read_text())
    result = []
    for seed in (4107, 4108):
        template = next(j for j in original["jobs"] if j["seed"] == seed)
        for cap in CAPS:
            for hold in HOLDS:
                job = deepcopy(template)
                job["id"] = cell_id(seed, cap, hold)
                job["policy_spec"]["profile"].update(version=VERSION,
                    acceleration_min_mps2=cap, hold_s=hold)
                BrakeHoldProfile(**job["policy_spec"]["profile"])
                paired.validate_job(job)
                result.append(job)
    return result


def source_bridge(manifest, identity):
    """Existing execution modules must be byte-identical; new modules are additive."""
    for key in ("config_sha256", "model_sha256", "network_sha256", "sumo_binary_sha256",
                "scenario_template_sha256"):
        if manifest[key] != identity[key]:
            raise ValueError("reused physical identity changed: " + key)
    changed = [name for name, value in manifest["active_source_sha256"].items()
        if name != "escalation_study.py" and identity["active_source_sha256"].get(name) != value]
    if changed:
        raise ValueError("reused execution module changed: " + str(changed))
    return dict(existing_modules_identical=True, dispatcher_only_exception=True,
        new_modules=sorted(set(identity["active_source_sha256"]) - set(manifest["active_source_sha256"])))


def replay_saved(seed, target_jobs, identity):
    """Same-action induction over every saved acquired-context decision; no SUMO."""
    path = OLD / "jobs" / f"s{seed}_m2700_p20_t600_gentle_wall"
    manifest, _ = paired.verified_arm(path)
    bridge = source_bridge(manifest, identity)
    old_reg = json.loads((OLD / "registration.json").read_text())
    policies = {j["id"]: policy_factory(j, old_reg) for j in target_jobs}
    packets = defaultdict(list)
    # Filter before JSON parsing; measured/global state never enters replay.
    with (path / "telemetry.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if '"rsu_received_report"' in line:
                row = json.loads(line)
                receipt = row.pop("received_at_s")
                row.pop("kind")
                packets[receipt].append(report_from_payload(row))
    controller = CirclesController()
    comm = old_reg["communication"]
    buffer = RSUReportBuffer(max_age_s=comm["max_report_age_s"], history_s=comm["history_s"])
    history, count, terminal = [], 0, None
    keys = ("phase", "reason", "member_ids", "wall_commands")
    for saved in rows(path / "policy.jsonl"):
        now = saved["time_s"]
        if now != count:
            raise ValueError("saved replay clock is incomplete")
        for packet in packets[now]:
            buffer.receive(packet, received_at_s=now)
            history.append((now, packet))
        history = [(t, p) for t, p in history if t >= now - comm["history_s"]]
        current = {p.frame.vehicle_id: controller.decide(p.frame, now_s=now) for p in packets[now]}
        numeric = {v: d.command_acceleration_mps2 for v, d in current.items() if d.fallback_reason is None}
        natives = frozenset(v for v, d in current.items() if d.fallback_reason is not None)
        context = paired.PolicyInput(now, tuple(buffer.reports(now_s=now)), tuple(history), numeric, natives)
        for name, policy in policies.items():
            output = policy(context).payload()
            if any(output[k] != saved[k] for k in keys):
                raise ValueError(f"saved policy physical-output mismatch {name} at {now}")
        if terminal is None and saved["phase"] in {"abort", "refused", "release"}:
            terminal = dict(time_s=now, phase=saved["phase"], reason=saved["reason"])
        count += 1
    if count != 2400 or terminal is None:
        raise ValueError("equivalence requires full horizon and terminal ownership path")
    return dict(source_path=str(path), source_manifest_sha256=sha(path / "manifest.json"),
        telemetry_sha256=manifest["telemetry_sha256"], policy_sha256=manifest["policy_sha256"],
        source_bridge=bridge, cell_ids=sorted(policies), decisions_replayed=count,
        compared_fields=list(keys), all_physical_outputs_exact=True, terminal=terminal,
        simulations_started=0, inference="same actions, unchanged deterministic execution and exogenous future; derived cells are not independent runs")


def preflight(output):
    identity = paired.current_identity()
    all_jobs = jobs()
    proof = {str(seed): replay_saved(seed, [j for j in all_jobs if j["seed"] == seed
        and j["policy_spec"]["profile"]["acceleration_min_mps2"] == -.3
        and (seed == 4108 or j["policy_spec"]["profile"]["hold_s"] == 20)], identity)
        for seed in (4107, 4108)}
    data = dict(schema="brake_hold_preflight/1", identity=identity, passed=True,
        protocol_sha256=sha(ROOT / PROTOCOL), reused_policy_proofs=proof,
        new_simulations=0, maximum_new_corridor_simulated_seconds=19200)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("preserve the previous preflight")
    write_json(output, data)
    return dict(path=str(output), passed=True, decisions_replayed=4800, simulations_started=0)


def register(output, mechanics_registration, preflight_path):
    from .brake_hold_mechanics import verify_completed
    mechanics = verify_completed(mechanics_registration)
    identity = paired.current_identity()
    preflight_path = Path(preflight_path).resolve()
    proof = json.loads(preflight_path.read_text())
    if (proof.get("passed") is not True or proof["identity"] != identity
            or proof["protocol_sha256"] != sha(ROOT / PROTOCOL)):
        raise ValueError("current exact replay preflight required")
    all_jobs, bindings, references, reference_cache = jobs(), {}, {}, {}
    for job in all_jobs:
        seed = job["seed"]
        profile = job["policy_spec"]["profile"]
        reuse = binding_mode(job) != "simulate"
        if reuse:
            rp = proof["reused_policy_proofs"][str(seed)]
            if job["id"] not in rp["cell_ids"] or not rp["all_physical_outputs_exact"]:
                raise ValueError("missing cell equivalence proof")
            path = Path(rp["source_path"])
            if sha(path / "manifest.json") != rp["source_manifest_sha256"]:
                raise ValueError("replay source changed")
            mode = "reused" if profile["hold_s"] == 20 else "trace_equivalent"
            bindings[job["id"]] = dict(mode=mode, source_path=str(path),
                manifest_sha256=rp["source_manifest_sha256"], evidence_cluster=f"s{seed}_gentle_h20")
        else:
            bindings[job["id"]] = dict(mode="simulate", evidence_cluster=job["id"])
        if seed not in reference_cache:
            reference_cache[seed] = reference_proof(REFERENCES / "jobs" / job["reference_job_id"], identity)[2]
        ref = reference_cache[seed]
        references[job["id"]] = ref
    original = json.loads((OLD / "registration.json").read_text())
    data = dict(schema=SCHEMA, registered_at_utc=datetime.now(timezone.utc).isoformat(),
        identity=identity, jobs=all_jobs, bindings=bindings, references=references,
        communication=original["communication"], criteria=original["criteria"],
        assignment="registered_experimental_assignment_not_trained_benefit_gate",
        interaction_diagnostics=True, policy_version=VERSION,
        maximum_attempts=8, maximum_simulated_seconds=19200, automatic_retries=0,
        automatic_successors=False, mechanics_evidence=mechanics,
        preflight_path=str(preflight_path), preflight_sha256=sha(preflight_path),
        protocol_path=PROTOCOL, protocol_sha256=sha(ROOT / PROTOCOL))
    data["registration_sha256"] = paired.digest(data)
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "source_snapshot").mkdir()
    for name in identity["active_source_sha256"]:
        shutil.copy2(ROOT / "traffic_shepherd" / name, root / "source_snapshot" / name)
    write_json(root / "registration.json", data)
    return dict(registration=str(root / "registration.json"), simulated_jobs=[j["id"] for j in all_jobs
        if bindings[j["id"]]["mode"] == "simulate"], maximum_simulated_seconds=19200,
        reused_cells=2, trace_equivalent_cells=2, simulations_started=0)


def read_registration(path, require_current=True):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if (data.get("schema") != SCHEMA or data.get("registration_sha256") != paired.digest(
            {k: v for k, v in data.items() if k != "registration_sha256"})
            or data["jobs"] != jobs() or data["maximum_attempts"] != 8
            or data["maximum_simulated_seconds"] != 19200 or data["automatic_retries"] != 0
            or data["automatic_successors"]):
        raise ValueError("factorial registration or exact budget changed")
    if require_current and (data["identity"] != paired.current_identity()
            or data["protocol_sha256"] != sha(ROOT / PROTOCOL)):
        raise ValueError("source or protocol changed after registration")
    if sha(data["preflight_path"]) != data["preflight_sha256"]:
        raise ValueError("replay proof changed")
    if set(data["bindings"]) != {j["id"] for j in data["jobs"]}:
        raise ValueError("factorial binding scope changed")
    for job in data["jobs"]:
        binding = data["bindings"][job["id"]]
        if binding["mode"] != binding_mode(job):
            raise ValueError("factorial allocation mode changed")
        if binding["mode"] != "simulate":
            expected = OLD / "jobs" / f"s{job['seed']}_m2700_p20_t600_gentle_wall"
            if Path(binding["source_path"]).resolve() != expected.resolve():
                raise ValueError("factorial reuse source changed")
    for name, value in data["identity"]["active_source_sha256"].items():
        if sha(path.parent / "source_snapshot" / name) != value:
            raise ValueError("registered source snapshot changed")
    return path.parent, data


def command_contract(path, profile):
    requests, previous, failures, measured = {}, {}, [], []
    for row in rows(Path(path) / "policy.jsonl"):
        for vid, value in row["wall_commands"].items():
            proof = row["command_resolution"][vid]
            if (row["time_s"], vid) in requests:
                failures.append([row["time_s"], vid, "duplicate_policy_command"])
            requests[row["time_s"], vid] = value
            if (not profile.acceleration_min_mps2-1e-8 <= value <= 1.5+1e-8
                    or abs(value-proof["measured_entry_or_current_acceleration_mps2"]) > .15+1e-8
                    or vid in previous and abs(value-previous[vid]) > .15+1e-8
                    or abs(value-proof["resolved_wall_acceleration_mps2"]) > 1e-8):
                failures.append([row["time_s"], vid, "resolver"])
            previous[vid] = value
    matched = set()
    issued_values = []
    for row in rows(Path(path) / "telemetry.jsonl"):
        if row["kind"] == "response" and row["owner"] == "wall":
            measured.append(row["measured_acceleration_mps2"])
        if row["kind"] != "command":
            continue
        key = row["timestamp_s"], row["vehicle_id"]
        if row["owner"] == "wall" or key in requests:
            if key in matched:
                failures.append([*key, "duplicate_actual_command"])
            matched.add(key)
            value = row["issued_acceleration_mps2"]
            if row["owner"] != "wall" or key not in requests or value is None or abs(value-requests[key]) > 1e-8:
                failures.append([*key, "actual_owner_or_command"])
            else:
                issued_values.append(value)
    if matched != set(requests):
        failures.append(["missing_actual_command"])
    return dict(passed=not failures, failures=failures, wall_member_commands=len(requests),
        minimum_issued_mps2=min(issued_values, default=None),
        minimum_measured_wall_response_mps2=min(measured, default=None),
        actual_member_exposure_below_minus_03_s=.5*sum(a < -.3-1e-8 for a in measured),
        command_contract_is_not_physical_gentleness=True)


def lifecycle(path, job):
    """Independent measured readiness/HOLD and actual owner checks, including H0."""
    profile = job["policy_spec"]["profile"]
    policy = list(rows(Path(path) / "policy.jsonl"))
    terminal = next((r for r in policy if r["phase"] in {"release", "abort", "refused"}), None)
    if terminal is None:
        raise ValueError("terminal phase missing")
    samples = [r for r in rows(Path(path) / "formation.jsonl") if r["time_s"] <= terminal["time_s"]]
    if not samples:
        return dict(assembly_complete=False, hold_seconds=0., registered_maneuver_delivered=False,
                    terminal_phase=terminal["phase"], terminal_time_s=terminal["time_s"])
    first = samples[0]
    if first["time_s"] != job["decision_time_s"]:
        raise ValueError("formation physics clock missing registered start")
    assigned = dict(zip(job["policy_spec"]["member_ids"], job["policy_spec"]["required_lanes"]))
    if len(assigned) != len(job["policy_spec"]["member_ids"]):
        raise ValueError("registered cohort mapping is not unique")
    target = first["target_speed_mps"]
    streak, ready, compact_hold, hold_rows, last = 0, None, True, [], None
    for row in samples:
        if last is not None and row["time_s"] != last+.5:
            raise ValueError("formation physics clock incomplete")
        last = row["time_s"]
        g = _geometry(row, assigned, profile, target)
        if float(row["time_s"]).is_integer() and ready is None:
            streak = streak+1 if g["settled"] else 0
            if streak >= profile["readiness_samples"]:
                ready = row["time_s"]
        if row["phase"] == "hold":
            hold_rows.append(row)
            compact_hold &= g["compact"] and all(row["owner_by_member"].get(v) == "wall" for v in assigned)
    # Terminal sample is a boundary, not another HOLD interval.
    hold_seconds = .5*len(hold_rows)
    indexed = {r["time_s"]: r for r in samples}
    for row in hold_rows:
        endpoint = indexed.get(row["time_s"]+.5)
        compact_hold &= endpoint is not None and _geometry(endpoint, assigned, profile, target)["compact"]
    count_matches = hold_seconds == profile["hold_s"]
    hold_clock = not hold_rows or (ready == hold_rows[0]["time_s"]
        and all(b["time_s"] == a["time_s"]+.5 for a, b in zip(hold_rows, hold_rows[1:])))
    first_transition = next((r for r in samples if r["phase"] != "assembly"), None)
    entry_clock = (ready is not None and first_transition is not None
        and first_transition["time_s"] == ready)
    commands = {(r["timestamp_s"], r["vehicle_id"]): r for r in rows(Path(path) / "telemetry.jsonl")
        if r["kind"] == "command" and r["vehicle_id"] in assigned}
    terminal_geometry = _geometry(samples[-1], assigned, profile, target)
    handback = _handback(terminal["time_s"], terminal_geometry, commands, assigned)
    after = any(t > terminal["time_s"] and r["owner"] == "wall" for (t, _), r in commands.items())
    assembly = ready is not None and ready-job["decision_time_s"] <= profile["assembly_deadline_s"]
    phases = []
    previous = None
    for row in policy:
        if row["phase"] != previous:
            phases.append(dict(time_s=row["time_s"], phase=row["phase"], reason=row["reason"]))
            previous = row["phase"]
    return dict(assembly_complete=assembly, measured_ready_at_s=ready,
        assembly_seconds=None if ready is None else ready-job["decision_time_s"],
        nominal_hold_seconds=profile["hold_s"], hold_seconds=hold_seconds,
        phase_entry_matches_measured_readiness=entry_clock,
        full_nominal_hold_delivered=bool(assembly and count_matches and compact_hold and hold_clock and entry_clock),
        zero_hold_means_no_hold_interval=profile["hold_s"] != 0 or not hold_rows,
        handback=handback, wall_reengaged_after_terminal=after, phases=phases,
        terminal_phase=terminal["phase"], terminal_time_s=terminal["time_s"], terminal_reason=terminal["reason"],
        registered_maneuver_delivered=bool(assembly and count_matches and compact_hold and hold_clock and entry_clock
            and handback["verified"] and not after and terminal["phase"] == "release"), safety_established=False)


def run_one(registration, job_id):
    root, data = read_registration(registration)
    job = next((j for j in data["jobs"] if j["id"] == job_id), None)
    if job is None or data["bindings"][job_id]["mode"] != "simulate" or (root / "STOP.json").exists():
        raise ValueError("unknown, reused or stopped job; no duplicate simulation permitted")
    for proof in data["references"].values():
        if sha(Path(proof["path"]) / "manifest.json") != proof["manifest_sha256"]:
            raise ValueError("registered damping manifest changed before allocation")
    for binding in data["bindings"].values():
        if binding["mode"] != "simulate" and sha(Path(binding["source_path"]) / "manifest.json") != binding["manifest_sha256"]:
            raise ValueError("registered reuse manifest changed before allocation")
    (root / "allocations").mkdir(exist_ok=True)
    with (root / "allocations" / f"{job_id}.json").open("x") as stream:
        json.dump(dict(job_id=job_id, allocated_simulated_seconds=2400,
            retries=0, registration_sha256=data["registration_sha256"]), stream)
    try:
        output = root / "jobs" / job_id
        result = paired.evaluate_arm(job=job, registration=data, output=output,
            policy=policy_factory(job, data))
        check = command_contract(output, BrakeHoldProfile(**job["policy_spec"]["profile"]))
        write_json(output / "brake_hold_command_contract.json", check)
        if not check["passed"]:
            raise ValueError("actual factorial command contract failed")
        return result
    except Exception as exc:
        write_json(root / "STOP.json", dict(job_id=job_id, reason=str(exc), automatic_retries=0))
        raise


def analyze(registration):
    root, data = read_registration(registration, require_current=False)
    result, pending, verified_cache, reference_cache = [], [], {}, {}
    for job in data["jobs"]:
        binding = data["bindings"][job["id"]]
        path = root / "jobs" / job["id"] if binding["mode"] == "simulate" else Path(binding["source_path"])
        if not (path / "manifest.json").exists():
            pending.append(job["id"])
            continue
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest.get("status") != "complete":
            pending.append(dict(job_id=job["id"], status=manifest.get("status")))
            continue
        if str(path) not in verified_cache:
            verified_cache[str(path)] = paired.verified_arm(path)
        tm, ts = verified_cache[str(path)]
        if binding["mode"] == "simulate":
            if tm["registration_sha256"] != data["registration_sha256"] or any(
                    tm.get(k) != v for k, v in data["identity"].items()):
                raise ValueError("factorial treatment identity changed")
            fields = {"job_id": "id", "case_id": "case_id", "seed": "seed", "mainline_vph": "loading",
                "penetration": "penetration", "duration_s": "duration_s", "decision_time_s": "decision_time_s",
                "policy_spec": "policy_spec", "controller": "arm", "role": "role"}
            if any(tm.get(key) != job[field] for key, field in fields.items()):
                raise ValueError("factorial cell metadata differs from registered job")
        elif sha(path / "manifest.json") != binding["manifest_sha256"]:
            raise ValueError("reused cell changed")
        reference = Path(data["references"][job["id"]]["path"])
        if str(reference) not in reference_cache:
            reference_cache[str(reference)] = reference_proof(reference, data["identity"])
        cm, cs, proof = reference_cache[str(reference)]
        if proof["manifest_sha256"] != data["references"][job["id"]]["manifest_sha256"]:
            raise ValueError("registered damping manifest identity changed")
        left = json.loads((reference / "decision_snapshots.json").read_text())["600"]
        right = json.loads((path / "decision_snapshots.json").read_text())["600"]
        for key in ("decision_time_s", "predecision_state_sha256", "predecision_report_evidence_sha256",
                    "predecision_report_prefix_sha256", "predecision_command_prefix_sha256"):
            if left[key] != right[key]:
                raise ValueError("mismatched factorial prefix: " + key)
        if cm["exogenous_future_sha256"] != tm["exogenous_future_sha256"]:
            raise ValueError("mismatched factorial future")
        control_view = dict(cm, active_source_sha256=data["identity"]["active_source_sha256"],
            proposal_sha256=right["proposal_sha256"], decision_time_s=600.,
            predecision_state_sha256=left["predecision_state_sha256"])
        treatment_view = dict(tm, active_source_sha256=data["identity"]["active_source_sha256"])
        effect = paired_effect(control_view, cs, treatment_view, ts)
        profile = BrakeHoldProfile(**job["policy_spec"]["profile"])
        life = lifecycle(path, job)
        result.append(dict(job_id=job["id"], seed=job["seed"], factors=profile.factors(),
            binding=binding, physical_source=str(path), manifest_sha256=sha(path / "manifest.json"),
            predecision_prefix_exact=True, reference_proof=proof, effect=effect,
            benefit=benefit_label(effect, BenefitCriteria(**data["criteria"])), lifecycle=life,
            command_contract=command_contract(path, profile), qualified_gate=False))
    lookup = {r["job_id"]: r for r in result}
    contrasts = []

    def contrast(name, seed, left_id, right_id):
        if left_id not in lookup or right_id not in lookup:
            return
        left, right = lookup[left_id], lookup[right_id]
        lc, rc = left["effect"]["costs_treatment"], right["effect"]["costs_treatment"]
        contrasts.append(dict(name=name, seed=seed, left=left_id, right=right_id,
            left_minus_right_cost_per_request={k: lc[k]-rc[k] for k in lc},
            left_delivered=left["lifecycle"]["registered_maneuver_delivered"],
            right_delivered=right["lifecycle"]["registered_maneuver_delivered"],
            same_realized_trajectory=left["physical_source"] == right["physical_source"]))
    for seed in (4107, 4108):
        for hold in HOLDS:
            contrast(f"gentle_minus_strong_H{hold}", seed, cell_id(seed, -.3, hold), cell_id(seed, -3., hold))
        for cap in CAPS:
            for lo, hi in ((0, 20), (20, 40)):
                contrast(f"H{hi}_minus_H{lo}_cap{cap}", seed, cell_id(seed, cap, hi), cell_id(seed, cap, lo))
    interactions = []
    for seed in (4107, 4108):
        for lo, hi in ((0, 20), (20, 40)):
            ids = [cell_id(seed, cap, hold) for cap, hold in ((-.3, hi), (-.3, lo), (-3., hi), (-3., lo))]
            if not all(name in lookup for name in ids):
                continue
            costs = [lookup[name]["effect"]["costs_treatment"] for name in ids]
            interactions.append(dict(seed=seed, contrast=f"(H{hi}-H{lo})gentle_minus_strong",
                cell_ids=ids, difference_of_cost_differences_per_request={k: costs[0][k]-costs[1][k]-costs[2][k]+costs[3][k]
                    for k in costs[0]}, inferential_p_value=None))
    answer = dict(schema="brake_hold_analysis/1", results=result, contrasts=contrasts, interactions=interactions, pending=pending,
        complete=not pending and len(result) == 12, independent_seed_count=2,
        population_statistical_significance_established=False, optimal_duration_established=False,
        maximum_new_simulated_seconds=19200, original_criteria_unchanged=True)
    write_json(root / "analysis.json", answer)
    return dict(analysis_path=str(root / "analysis.json"), complete=answer["complete"],
        completed_cells=len(result), pending=pending, contrasts=contrasts)
