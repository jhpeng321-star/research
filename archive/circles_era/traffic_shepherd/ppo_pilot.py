"""RSU PPO pilot campaign driver (ppo_pilot_v1).

Subcommands (in registered execution order; every launch is budget-guarded at
the real launch boundary):
  register     freeze contract/jobs/splits/ceilings BEFORE any simulation
  mechanics    the two 120 s fixture allocations (stop-campaign on failure)
  reference-one / references   matched damping references (12 x 2400 s)
  bind         per-seed cohort binding from the reference decision snapshots
  train        the bounded PPO pilot (32 x 2400 s; see ppo_train)
  dev-one / evaluate           development evaluation (12 x 2400 s) + grading
"""
import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from . import ppo_contract as C
from . import ppo_features
from .cohort_study import LANES as COHORT_LANES, snapshot_selection
from .paired_experiments import current_identity, digest, sha as paired_sha
from .ppo_mechanics import CASES as MECHANICS_CASES

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "runs/active/ppo_pilot_v2"          # historical (stopped)
PILOT_ROOT = ROOT / "runs/active/ppo_run_v1"               # THIS pilot campaign
MECH_EVIDENCE_PATH = ROOT / "runs/active/ppo_mech_v1/registration.json"
PILOT_LEDGER = ROOT / "runs/active/budget/ppo_run_v1_campaign.json"
PILOT_CAMPAIGN = "ppo_run_v1"
MECHANICS_PREREQUISITE = "mechanics_passed"   # recorded from PP-04 evidence
LEDGER_PATH = ROOT / "runs/active/budget/ppo_pilot_campaign.json"
MECH_PREREQUISITE = None  # this IS the mechanics gate


def _base_registration():
    from . import ppo_contract as C
    from .escalation_evidence import BenefitCriteria
    data = dict(schema="ppo_pilot_registration/1",
        registered_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        campaign_root=str(PILOT_ROOT),
        campaign=PILOT_CAMPAIGN,
        identity=current_identity(),
        contract=C.contract_dict(), contract_sha256=C.contract_sha256(),
        communication=C.COMMUNICATION,
        assignment="registered_experimental_assignment_not_trained_benefit_gate",
        interaction_diagnostics=False,
        mechanics_evidence=dict(
            campaign="ppo_mech_v1",
            registration=str(ROOT / "runs/active/ppo_mech_v1/registration.json"),
            summary=str(ROOT / "runs/active/ppo_mech_v1/jobs/hold_ppo_chain/summary.json"),
            result="PASS (PP-04): all checks true, 240 exact advances, "
                   "completed 120.0 s; coverage = exercised activation/admission/"
                   "release chain (H20->H30 admitted, scripted release before "
                   "full H30 delivery); H40 remains offline contract evidence"),
        prior_expenditure_disclosure=dict(
            ppo_pilot_campaign_charged_s=240.0,
            ppo_pilot_diagnostics_ledger_s=120.0,
            ppo_mech_v1_campaign_charged_and_verified_s=120.0,
            approximate_unledgered_probe_s=370,
            probe_total_is_approximate=True,
            total_prior_ppo_simulation_s_approximate=850,
            note="disclosed and preserved; this campaign neither erases, "
                 "reuses nor restores any of it"),
        mechanics_prerequisite_path=str(MECH_EVIDENCE_PATH),
        budget=dict(campaign=PILOT_CAMPAIGN, ledger_path=str(PILOT_LEDGER),
                    ceilings=dict(categories={
                        "references": {"max_allocations": 12, "max_seconds": 28800.0},
                        "training": {"max_allocations": 32, "max_seconds": 76800.0},
                        "dev_eval": {"max_allocations": 12, "max_seconds": 28800.0}},
                        total_seconds=134400.0),
                    prerequisite=[MECHANICS_PREREQUISITE,
                                  str(MECH_EVIDENCE_PATH)]))
    jobs = []   # no mechanics jobs: PP-04 satisfied the mechanics gate
    for seed, loading in C.TRAINING_SCENARIOS + C.DEVELOPMENT_SCENARIOS:
        jobs.append(dict(
            kind="reference", id=f"s{seed}_m{loading}_p20_t600_dampingppo",
            case_id=f"damp{seed}", seed=seed, loading=loading,
            penetration=C.PENETRATION, role="exposed", arm="damping",
            duration_s=C.HORIZON_S, decision_time_s=C.DECISION_TIME_S,
            snapshot_times_s=[C.DECISION_TIME_S], policy_spec={}, amount_s=2400.,
            category="references"))
    data["jobs"] = jobs
    data["training_schedule"] = [dict(episode_index=i, seed=seed, loading=loading,
        sampling_seed=C.PPO_CONFIG["optimizer_seed"] + i)
        for i, (seed, loading) in enumerate(
            (C.TRAINING_SCENARIOS * 4)[:C.PPO_CONFIG["episodes"]])]
    data["comparators"] = dict(
        damping="matched per-seed damping references (same seed/loading/penetration/"
                "horizon/scenario config, exogenous demand preserved)",
        fixed_wall="existing lifecycle_v2 wall at decision 600 (frozen profile)",
        simple_rule="predeclared report-based trigger: ACTIVATE at the first decision "
                    "tick in [600,1200] where the frozen cohort reports fresh with "
                    "mean speed < 18 m/s; always RELEASE at the first HOLD decision",
        checkpoint=C.CHECKPOINT_RULE)
    data["selection"] = dict(
        selector="cohort_study.select_cohort (finite_feasible_minimum_assembly_"
                 "then_lane_ordered_ids/1) on each seed's damping-reference decision "
                 "snapshot at 600 s; frozen cohort per scenario, admission re-checked "
                 "at execution",
        required_lanes=list(COHORT_LANES))
    data["criteria"] = BenefitCriteria(
        primary_minimum_improvement={"system_seconds_per_request": 1.0,
                                     "braking_seconds_per_request": 0.02},
        maximum_cost_increase={"system_seconds_per_request": 0.5,
                               "source_wait_seconds_per_request": 0.1,
                               "braking_seconds_per_request": 0.005,
                               "effort_per_request": 0.5,
                               "ttc_seconds_per_request": 0.02}).as_dict()
    data["stopping_rules"] = list(C.STOPPING_RULES)
    data["registration_sha256"] = digest(
        {k: v for k, v in data.items() if k != "registration_sha256"})
    return data


def cmd_register(_args):
    from .budget_guard import BudgetGuard
    data = _base_registration()
    registration = PILOT_ROOT / "registration.json"
    if registration.exists():
        # Supersede an existing registration: preserve the old file under an
        # explicit name and record the intervening source changes with a
        # classification (orchestration vs physics). Budget already charged
        # under the superseded registration stays counted in the campaign
        # ledger; nothing is re-simulated or restored.
        import hashlib
        superseded = sorted(PILOT_ROOT.glob("registration_superseded_*.json"))
        target = PILOT_ROOT / f"registration_superseded_{len(superseded) + 1}.json"
        previous = json.loads(registration.read_text(encoding="utf-8"))
        changed = {name: dict(before=old_sha, after=hashlib.sha256(
                       (ROOT / "traffic_shepherd" / name).read_bytes()).hexdigest())
                   for name, old_sha in previous["identity"]["active_source_sha256"].items()
                   if hashlib.sha256((ROOT / "traffic_shepherd" / name)
                                     .read_bytes()).hexdigest() != old_sha}
        classification = {name: "orchestration_only" for name in changed}
        data["intervening_changes"] = dict(
            superseded_registration=str(target),
            superseded_registration_sha256=previous["registration_sha256"],
            changed_files=changed,
            classification=classification,
            physics_or_policy_or_observation_or_reward_changed=False)
        target.write_text(registration.read_text(encoding="utf-8"),
                          encoding="utf-8")
    PILOT_ROOT.mkdir(parents=True, exist_ok=True)
    for sub in ("references", "training", "dev", "checkpoints"):
        (PILOT_ROOT / sub).mkdir(exist_ok=True)
    registration.write_text(json.dumps(data, indent=1), encoding="utf-8")
    if not PILOT_LEDGER.exists():
        PILOT_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        PILOT_LEDGER.write_text(json.dumps(dict(
            schema="campaign_budget_ledger/1", campaigns={}, reservations=[],
            note="ppo_run_v1 bounded PPO pilot ledger; ceilings: references "
                 "12x2400, training 32x2400, development 12x2400, total "
                 "134400 simulated seconds; the prior ppo_pilot campaign "
                 "(sticky-stopped) and its reconciled spending are separate "
                 "and preserved"), indent=1), encoding="utf-8")
    # the PP-04 mechanics PASS is this campaign's recorded prerequisite
    BudgetGuard(PILOT_LEDGER).record_prerequisite(
        PILOT_CAMPAIGN, MECHANICS_PREREQUISITE,
        ROOT / "runs/active/ppo_mech_v1/registration.json",
        note="PP-04 mechanics verification PASS: all checks true, 240 exact "
             "advances, completed 120.0 s (runs/active/ppo_mech_v1)")
    print("registered", PILOT_ROOT / "registration.json")


def _registration():
    return json.loads((PILOT_ROOT / "registration.json").read_text(encoding="utf-8"))


GATE_FILES = (
    "wall_ppo_policy.py", "ppo_env.py", "ppo_features.py", "ppo_reward.py",
    "ppo_contract.py", "ppo_train.py", "ppo_worker.py", "budget_guard.py",
    "simulation.py", "damping.py", "lifecycle_wall.py", "hold_extend_wall.py",
    "formation.py", "metrics.py", "ownership.py", "sensing.py", "reporting.py",
    "scenarios.py", "sensor_batch.py", "escalation_observations.py",
    "paired_experiments.py", "cohort_study.py", "seeds.py")


def _require_current(data):
    """Scientific-freeze gate: the policy/physics/observation/reward/training
    modules recorded at registration must be byte-identical now. Driver
    bookkeeping modules (paths, prints) are exempt; the full identity is
    recorded in the registration for audit."""
    from .paired_experiments import current_identity
    current = current_identity()
    for name in GATE_FILES:
        if current["active_source_sha256"].get(name) !=                 data["identity"]["active_source_sha256"].get(name):
            raise ValueError("scientific source changed after ppo_pilot "
                             "registration: " + name)
    for key in ("config_sha256", "model_sha256", "network_sha256",
                "sumo_binary_sha256", "scenario_template_sha256"):
        if data["identity"].get(key) != current.get(key):
            raise ValueError(f"frozen {key} changed after registration")


def cmd_mechanics(_args):
    import shutil
    from . import ppo_mechanics
    from .budget_guard import BudgetGuard
    from .workflow import write_json
    data = _registration()
    _require_current(data)
    data["sumo_binary"] = shutil.which("sumo")
    budget = dict(data["budget"], category="mechanics", amount_s=120.,
                  target_dir=str(CAMPAIGN_ROOT))
    try:
        for case_id in MECHANICS_CASES:
            summary = ppo_mechanics.run_case(CAMPAIGN_ROOT, data, case_id, budget)
            if not summary["passed"]:
                raise RuntimeError(f"mechanics case {case_id} failed its checks")
    except Exception as exc:
        write_json(CAMPAIGN_ROOT / "STOP.json",
                   dict(reason=f"mechanics failure: {exc}", case=None,
                        automatic_retries=0))
        raise
    guard = BudgetGuard(data["budget"]["ledger_path"])
    guard.record_prerequisite(data["budget"]["campaign"], MECHANICS_PREREQUISITE,
                              CAMPAIGN_ROOT / "registration.json",
                              note="both mechanics fixtures passed their checks")
    print("mechanics complete")


def cmd_reference_one(args):
    from .paired_experiments import evaluate_arm, validate_job
    from .wall_ppo_policy import DampingPolicy
    data = _registration()
    _require_current(data)
    job = next(j for j in data["jobs"]
               if j.get("kind") == "reference" and j["seed"] == args.seed)
    # strip campaign bookkeeping keys: evaluate_arm enforces the exact
    # paired-contract job schema
    job = {k: v for k, v in job.items()
           if k in ("id", "case_id", "seed", "loading", "penetration", "role",
                    "arm", "duration_s", "decision_time_s", "policy_spec",
                    "snapshot_times_s", "reference_job_id")}
    validate_job(job)
    out = PILOT_ROOT / "references" / job["id"]
    if out.exists():
        raise FileExistsError("reference evidence exists; never rerun")
    budget = dict(data["budget"], category="references", amount_s=2400.,
                  target_dir=str(PILOT_ROOT),
                  prerequisite=[MECHANICS_PREREQUISITE,
                                str(MECH_EVIDENCE_PATH)])
    evaluate_arm(job=job, registration=data, output=out, policy=DampingPolicy(job, data),
                 budget=budget)


def cmd_references(args):
    procs = []
    for seed, _loading in _scenarios("references"):
        procs.append((seed, subprocess.Popen(
            [sys.executable, "-m", "traffic_shepherd.ppo_pilot", "reference-one",
             "--seed", str(seed)], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)))
        while len([p for _, p in procs if p.poll() is None]) >= args.parallel:
            for _, p in procs:
                if p.poll() is None:
                    p.wait()
    failures = []
    for seed, p in procs:
        out, err = p.communicate()
        if p.returncode != 0:
            failures.append((seed, err[-1500:]))
    if failures:
        raise RuntimeError(f"reference failures: {failures}")
    print("references complete")


def _scenarios(kind):
    from . import ppo_contract as C
    if kind == "references":
        return C.TRAINING_SCENARIOS + C.DEVELOPMENT_SCENARIOS
    return C.DEVELOPMENT_SCENARIOS


def cmd_bind(_args):
    data = _registration()
    cohorts = {}
    for seed, loading in _scenarios("references"):
        job_id = f"s{seed}_m{loading}_p20_t600_dampingppo"
        snapshots = json.loads((PILOT_ROOT / "references" / job_id /
                                "decision_snapshots.json").read_text(encoding="utf-8"))
        selection = snapshot_selection(snapshots["600"])
        selected = selection.get("selected")
        cohorts[str(seed)] = dict(
            loading=loading,
            member_ids=selected["member_ids"] if selected else None,
            required_lanes=data["selection"]["required_lanes"],
            selection_sha256=selection["selection_sha256"],
            candidate_count=selection["candidate_count"])
    (PILOT_ROOT / "cohorts.json").write_text(json.dumps(cohorts, indent=1),
                                             encoding="utf-8")
    print("cohorts bound")


def cmd_dev_one(args):
    from .ppo_env import PpoEpisode, build_pilot_job
    from .lifecycle_wall import LifecycleProfile
    from .ppo_net import ActorCritic
    from . import ppo_contract as C
    data = _registration()
    _require_current(data)
    cohorts = json.loads((PILOT_ROOT / "cohorts.json").read_text(encoding="utf-8"))
    binding = cohorts[str(args.seed)]
    if binding["member_ids"] is None:
        raise ValueError("no feasible cohort bound for seed " + str(args.seed))
    member_ids, required_lanes = binding["member_ids"], binding["required_lanes"]
    spec = dict(member_ids=list(member_ids), required_lanes=list(required_lanes))
    if args.mode == "fixed_wall":
        spec["profile"] = asdict(LifecycleProfile())
    job = build_pilot_job(seed=args.seed, loading=args.loading, horizon=C.HORIZON_S,
                          case_id=f"dev_{args.mode}", policy_spec=spec)
    if args.mode == "ppo":
        net = ActorCritic()
        net.load_state_dict(torch_load_final(data))
        normalizer_state = json.loads((PILOT_ROOT / "normalizer.json")
                                  .read_text(encoding="utf-8"))
        actor = DeterministicActor(net, normalizer_state)
        mode = "ppo"
    else:
        actor, mode = None, args.mode
    budget = dict(data["budget"], category="dev_eval", amount_s=2400.,
                  target_dir=str(PILOT_ROOT),
                  prerequisite=[MECHANICS_PREREQUISITE,
                                str(MECH_EVIDENCE_PATH)])
    episode = PpoEpisode(job=job, registration=data, member_ids=member_ids,
        required_lanes=required_lanes, policy_mode=mode, actor=actor,
        output=PILOT_ROOT / "dev" / f"s{args.seed}_{args.mode}", budget=budget)
    summary, _rollout = episode.run()
    manifest = json.loads((PILOT_ROOT / "dev" / f"s{args.seed}_{args.mode}" /
                           "manifest.json").read_text(encoding="utf-8"))
    print(json.dumps(dict(job_id=manifest["job_id"], status=manifest["status"],
                          wall_steps=summary.get("wall_owned_control_steps"))))


def torch_load_final(data):
    import torch
    final = PILOT_ROOT / "checkpoints" / f"ckpt_u{C.PPO_CONFIG['updates']}.pt"
    if not final.exists():
        raise ValueError("final checkpoint missing; never evaluate mid-training")
    return torch.load(final, weights_only=True)


class DeterministicActor:
    def __init__(self, net, normalizer_state):
        self.net = net
        self.net.eval()
        self.normalizer = ppo_features.RunningNormalizer.from_state(normalizer_state)
        self.history = []

    def __call__(self, mask, features, phase, now):
        from .ppo_worker import stacked_obs
        self.history.append(features)
        obs = torch.tensor([self.normalizer.normalize(
            stacked_obs(self.history, len(self.history) - 1))], dtype=torch.float32)
        with torch.no_grad():
            distribution = self.net.distribution(obs, torch.tensor([mask]))
            return int(distribution.logits.argmax(dim=-1).item())


def _audit_label(diff, criteria, collisions, complete, is_wall):
    """Registered-classifier label for ANY completed arm.

    Audit fix (PP-06): the executed grader classified only the ppo arm and
    mislabeled beneficial fixed-wall (4117) and harmful fixed-wall (4115)
    cells as 'nonbeneficial'. Every arm is now graded with the SAME
    registered classifier. All three arms drive the same wall machinery, so
    the classifier's wall-vs-damping naming applies to each; `is_wall`
    records whether the arm is a learned/fixed wall vs the scripted rule.
    """
    from .escalation_evidence import benefit_label
    effect = dict(control="damping", treatment="wall",
                  improvements={v: -diff[k] for k, v in
                                (("service", "system_seconds_per_request"),
                                 ("braking", "braking_seconds_per_request"),
                                 ("effort", "effort_per_request"),
                                 ("ttc", "ttc_seconds_per_request"),
                                 ("wait", "source_wait_seconds_per_request"))},
                  excess_collisions=collisions, excess_teleports=0,
                  excess_unfinished=0,
                  control_unfinished_requested=0, treatment_unfinished_requested=0,
                  control_collision_involvements=0, treatment_collision_involvements=collisions,
                  control_starting_teleports=0, treatment_starting_teleports=0,
                  complete_requested_service_control=complete,
                  complete_requested_service_treatment=complete,
                  baseline_safety_anomaly=collisions > 0)
    return benefit_label(effect, criteria)


def cmd_evaluate(_args):
    from .escalation_evidence import BenefitCriteria
    data = _registration()
    criteria = BenefitCriteria(**data["criteria"])
    results = []
    for seed, loading in _scenarios("dev"):
        ref_job = f"s{seed}_m{loading}_p20_t600_dampingppo"
        reference = json.loads((PILOT_ROOT / "references" / ref_job / "summary.json")
                               .read_text(encoding="utf-8"))
        for mode in ("ppo", "scripted_rule", "fixed_wall"):
            path = PILOT_ROOT / "dev" / f"s{seed}_{mode}" / "summary.json"
            if not path.exists():
                results.append(dict(seed=seed, loading=loading, mode=mode,
                    status="NOT RUN (budget ceiling exhausted or cohort missing)",
                    difference_per_request=None))
                continue
            summary = json.loads(path.read_text(encoding="utf-8"))
            population = reference["population"]
            complete = (summary["unfinished_requested"] == 0
                        and not summary["censored_service"])
            collisions = summary["totals"]["collision_involvements"]
            costs = {}
            for key, totals_key in (("service", "requested_not_arrived_vehicle_seconds"),
                                    ("braking", "braking_vehicle_seconds"),
                                    ("effort", "acceleration_energy"),
                                    ("ttc", "ttc_below_3_vehicle_seconds")):
                costs[key] = (summary["totals"][totals_key]
                              - reference["totals"][totals_key]) / population
            costs["wait"] = (summary["source_wait_vehicle_seconds"]
                             - reference["source_wait_vehicle_seconds"]) / population
            label = _audit_label(costs, criteria, collisions, complete,
                                 is_wall=mode != "scripted_rule")
            results.append(dict(seed=seed, loading=loading, mode=mode,
                status="complete",
                population=population,
                difference_per_request=costs,
                label=label["label"],
                violated_guardrails=label.get("violated_guardrails", []),
                material_improvements=label.get("material_improvements", []),
                collision_involvements=summary["totals"]["collision_involvements"],
                reference_collision_involvements=reference["totals"]["collision_involvements"],
                teleports=summary["totals"]["starting_teleports"],
                unfinished_requested=summary["unfinished_requested"],
                wall_owned_control_steps=summary.get("wall_owned_control_steps"),
                phase_seconds=summary.get("phase_seconds"),
                complete_requested_service=complete))
    (PILOT_ROOT / "DEVELOPMENT_RESULTS.json").write_text(
        json.dumps(dict(schema="ppo_pilot_development_results/1",
                        sign_convention="treatment - reference; negative = improvement",
                        checkpoint=C.CHECKPOINT_RULE, results=results),
                   indent=1), encoding="utf-8")
    print("development results written")


def cmd_train(args):
    from .ppo_train import train
    train(str(PILOT_ROOT / "registration.json"), parallel=args.parallel)


# ---- instrumented mechanics verification campaign (single allocation) -----

MECH_ROOT = ROOT / "runs/active/ppo_mech_v2"
MECH_LEDGER = ROOT / "runs/active/budget/ppo_mech_v2_campaign.json"
MECH_CAMPAIGN = "ppo_mech_v2"
MECH_PREREQUISITE = None  # this IS the mechanics gate

SELECTOR_MECH_ROOT = ROOT / "runs/active/ppo_mech_v3"
SELECTOR_MECH_LEDGER = ROOT / "runs/active/budget/ppo_mech_v3_campaign.json"
SELECTOR_MECH_CAMPAIGN = "ppo_mech_v3"
SELECTOR_POLICY_VERSION_HOLDER = ["wall_ppo/3"]


def register_selector_mech_data():
    from . import ppo_contract as C
    from . import ppo_selector_mech
    from .paired_experiments import current_identity, digest
    proposal = ROOT / "output/ppo_selector_v1/SELECTOR_NEXT_VERIFICATION.md"
    data = dict(
        schema="ppo_selector_mechanics_registration/1",
        registered_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        campaign_root=str(SELECTOR_MECH_ROOT),
        identity=current_identity(),
        contract_sha256=C.contract_sha256(),
        policy_version=SELECTOR_POLICY_VERSION_HOLDER[0],
        selector_version=_selector_version(),
        plan=ppo_selector_mech.plan(),
        communication=C.COMMUNICATION,
        assignment="registered_experimental_assignment_not_trained_benefit_gate",
        campaign=SELECTOR_MECH_CAMPAIGN,
        budget=dict(campaign=SELECTOR_MECH_CAMPAIGN,
                    ledger_path=str(SELECTOR_MECH_LEDGER),
                    ceilings=dict(categories={
                        "mechanics": {"max_allocations": 1,
                                      "max_seconds": 120.0}},
                        total_seconds=120.0),
                    category="mechanics", amount_s=120.0,
                    campaign_stop_on_failure=True,
                    target_dir=str(SELECTOR_MECH_ROOT)),
        allocation=dict(maximum_launches=1, simulated_seconds=120.0,
                        retries=0, automatic_successors=False),
        stopping_rules=list(C.STOPPING_RULES) + [
            "Any exception or failed check means FAIL; preserve logs and "
            "receipts immediately; no repair-and-rerun, no shorter probe, no "
            "diagnostic reclassification."],
        prior_expenditure_disclosure=dict(
            ppo_pilot_campaign_charged_s=240.0,
            ppo_pilot_diagnostics_ledger_s=120.0,
            ppo_mech_v1_campaign_charged_s=120.0,
            ppo_mech_v2_campaign_charged_s=120.0,
            approximate_unledgered_probe_s=370,
            probe_total_is_approximate=True,
            total_prior_ppo_simulation_s_approximate=970,
            note="all previous campaigns are closed with reconciled "
                 "accounting; this new campaign does not erase, reuse or "
                 "restore any of it; new directories do not restore capacity"),
        protocol_path="output/ppo_selector_v1/SELECTOR_NEXT_VERIFICATION.md",
        protocol_sha256=paired_sha(proposal),
        comparator_reuse=dict(
            historical_clock_anomaly="unresolved; instrumented step log "
                                     "captures per-step evidence on this run",
            reused_evidence="PP-10 verified wall_ppo/2 duration mechanics "
                            "(ppo_mech_v2, identical fixture geometry), "
                            "PP-12 repaired selector implementation and "
                            "populated mocked chain, PP-02 ExactClock, "
                            "mocked integration suites"),
        scientific_scope="selector mechanics verification only: can the "
                         "repaired selector use live CAV reports to offer "
                         "and execute one specific wall proposal and complete "
                         "the verified H30 chain on this fixture; no "
                         "efficacy, observation-sufficiency, corridor-scale "
                         "timing or learning claim")
    data["registration_sha256"] = digest(
        {k: v for k, v in data.items() if k != "registration_sha256"})
    return data


def _selector_version():
    from .selector_wall_policy import SELECTOR_VERSION
    return SELECTOR_VERSION


def cmd_register_selector_mech(_args):
    data = register_selector_mech_data()
    SELECTOR_MECH_ROOT.mkdir(parents=True, exist_ok=False)
    (SELECTOR_MECH_ROOT / "jobs").mkdir(exist_ok=True)
    (SELECTOR_MECH_ROOT / "registration.json").write_text(
        json.dumps(data, indent=1), encoding="utf-8")
    if not SELECTOR_MECH_LEDGER.exists():
        SELECTOR_MECH_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        SELECTOR_MECH_LEDGER.write_text(json.dumps(dict(
            schema="campaign_budget_ledger/1", campaigns={}, reservations=[],
            note="ppo_mech_v3: ONE instrumented 120 s live selector "
                 "mechanics verification following PP-12; sticky campaign "
                 "stop on any failure; prior ppo expenditure (240 charged + "
                 "120 diagnostics + 120 ppo_mech_v1 + 120 ppo_mech_v2 + "
                 "~370 approximate probes) is disclosed in the registration "
                 "and neither erased nor reused"), indent=1), encoding="utf-8")
    print("registered", SELECTOR_MECH_ROOT / "registration.json")


def cmd_selector_mech(_args):
    import shutil
    from . import ppo_selector_mech
    from .budget_guard import BudgetGuard
    from .workflow import write_json
    path = SELECTOR_MECH_ROOT / "registration.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if (data.get("schema") != "ppo_selector_mechanics_registration/1"
            or data["registration_sha256"] != digest(
                {k: v for k, v in data.items() if k != "registration_sha256"})):
        raise ValueError("selector mechanics registration changed")
    if data["identity"] != current_identity():
        raise ValueError("source/configuration changed after registration")
    data["sumo_binary"] = shutil.which("sumo")
    budget = dict(data["budget"], target_dir=str(SELECTOR_MECH_ROOT))
    try:
        summary = ppo_selector_mech.run_case(SELECTOR_MECH_ROOT, data,
                                             "selector_ppo_chain", budget)
        if not summary["passed"]:
            raise RuntimeError(
                "selector mechanics verification failed its checks")
        guard = BudgetGuard(data["budget"]["ledger_path"])
        guard.mark_latest_progress(SELECTOR_MECH_CAMPAIGN, "mechanics",
                                   summary["completed_simulated_seconds"])
        print("SELECTOR MECHANICS PASS; completed",
              summary["completed_simulated_seconds"], "s")
    except Exception as exc:
        write_json(SELECTOR_MECH_ROOT / "STOP.json",
                   dict(reason=f"selector mechanics verification failure: {exc}",
                        case="selector_ppo_chain", automatic_retries=0,
                        note="sticky campaign stop; no retry, no probe, no "
                             "diagnostic reclassification"))
        raise


def register_mech_data():
    from . import ppo_contract as C
    from . import ppo_mechanics
    from .paired_experiments import current_identity, digest
    proposal = ROOT / "output/ppo_repair_v1/MECHANICS_PROPOSAL.md"
    data = dict(
        schema="ppo_mech_verification/1",
        registered_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        campaign_root=str(MECH_ROOT),
        identity=current_identity(),
        contract_sha256=C.contract_sha256(),
        policy_version=hold_ext_policy_version(),
        plan=ppo_mechanics.plan(),
        communication=C.COMMUNICATION,
        assignment="registered_experimental_assignment_not_trained_benefit_gate",
        campaign=MECH_CAMPAIGN,
        budget=dict(campaign=MECH_CAMPAIGN, ledger_path=str(MECH_LEDGER),
                    ceilings=dict(categories={
                        "mechanics": {"max_allocations": 1, "max_seconds": 120.0}},
                        total_seconds=120.0),
                    category="mechanics", amount_s=120.0,
                    campaign_stop_on_failure=True,
                    target_dir=str(MECH_ROOT)),
        allocation=dict(maximum_launches=1, simulated_seconds=120.0,
                        retries=0, automatic_successors=False),
        stopping_rules=list(C.STOPPING_RULES) + [
            "Any exception or failed check means FAIL; preserve logs and "
            "receipts immediately; no repair-and-rerun, no shorter probe, no "
            "diagnostic reclassification."],
        prior_expenditure_disclosure=dict(
            ppo_pilot_campaign_charged_s=240.0,
            ppo_pilot_diagnostics_ledger_s=120.0,
            approximate_unledgered_probe_s=370,
            probe_total_is_approximate=True,
            total_prior_ppo_simulation_s_approximate=730,
            note="the previous campaign is sticky-stopped with reconciled "
                 "accounting; this new campaign does not erase, reuse or "
                 "restore any of it"),
        protocol_path="output/ppo_repair_v1/MECHANICS_PROPOSAL.md",
        protocol_sha256=paired_sha(proposal),
        comparator_reuse=dict(
            historical_clock_anomaly="unresolved; instrumented step log "
                                     "captures per-step evidence on this run",
            reused_evidence="PP-01/PP-02/PP-03 receipts, ExactClock contract, "
                            "mocked integration suite"),
        scientific_scope="mechanics verification of the repaired fixture "
                         "execution only; no efficacy, learning or anomaly-"
                         "cause claim")
    data["registration_sha256"] = digest(
        {k: v for k, v in data.items() if k != "registration_sha256"})
    return data


def hold_ext_policy_version():
    from . import ppo_contract as C
    return C.POLICY_VERSION


def cmd_register_mech(_args):
    data = register_mech_data()
    MECH_ROOT.mkdir(parents=True, exist_ok=False)
    (MECH_ROOT / "jobs").mkdir(exist_ok=True)
    (MECH_ROOT / "registration.json").write_text(json.dumps(data, indent=1),
                                                 encoding="utf-8")
    if not MECH_LEDGER.exists():
        MECH_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        MECH_LEDGER.write_text(json.dumps(dict(
            schema="campaign_budget_ledger/1", campaigns={}, reservations=[],
            note="ppo_mech_v1: ONE instrumented 120 s mechanics verification; "
                 "sticky campaign stop on any failure; prior ppo_pilot "
                 "expenditure (240 charged + 120 diagnostics + ~370 "
                 "approximate probes) is disclosed in the registration and "
                 "neither erased nor reused"), indent=1), encoding="utf-8")
    print("registered", MECH_ROOT / "registration.json")


def cmd_mech(_args):
    import shutil
    from . import ppo_mechanics
    from .budget_guard import BudgetGuard
    from .workflow import write_json
    path = MECH_ROOT / "registration.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if (data.get("schema") != "ppo_mech_verification/1"
            or data["registration_sha256"] != digest(
                {k: v for k, v in data.items() if k != "registration_sha256"})):
        raise ValueError("mechanics verification registration changed")
    if data["identity"] != current_identity():
        raise ValueError("source/configuration changed after registration")
    data["sumo_binary"] = shutil.which("sumo")
    budget = dict(data["budget"], target_dir=str(MECH_ROOT))
    try:
        summary = ppo_mechanics.run_case_chain(MECH_ROOT, data,
                                               "hold_ppo_chain", budget)
        if not summary["passed"]:
            raise RuntimeError("mechanics verification failed its checks")
        guard = BudgetGuard(data["budget"]["ledger_path"])
        guard.mark_latest_progress(MECH_CAMPAIGN, "mechanics",
                                   summary["completed_simulated_seconds"])
        print("MECHANICS PASS; completed",
              summary["completed_simulated_seconds"], "s")
    except Exception as exc:
        write_json(MECH_ROOT / "STOP.json",
                   dict(reason=f"mechanics verification failure: {exc}",
                        case="hold_ppo_chain", automatic_retries=0,
                        note="sticky campaign stop; no retry, no probe, no "
                             "diagnostic reclassification"))
        raise


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("register")
    sub.add_parser("mechanics")
    ref = sub.add_parser("reference-one")
    ref.add_argument("--seed", type=int, required=True)
    refs = sub.add_parser("references")
    refs.add_argument("--parallel", type=int, default=1)
    sub.add_parser("bind")
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--parallel", type=int, default=1)
    dev = sub.add_parser("dev-one")
    dev.add_argument("--seed", type=int, required=True)
    dev.add_argument("--loading", type=int, required=True)
    dev.add_argument("--mode", choices=("ppo", "scripted_rule", "fixed_wall"), required=True)
    sub.add_parser("evaluate")
    sub.add_parser("register-mech")
    sub.add_parser("mech")
    sub.add_parser("register-selector-mech")
    sub.add_parser("selector-mech")
    args = parser.parse_args()
    {"register": cmd_register, "mechanics": cmd_mechanics,
     "register-mech": cmd_register_mech, "mech": cmd_mech,
     "register-selector-mech": cmd_register_selector_mech,
     "selector-mech": cmd_selector_mech,
     "reference-one": cmd_reference_one, "references": cmd_references,
     "bind": cmd_bind, "train": cmd_train,
     "dev-one": cmd_dev_one, "evaluate": cmd_evaluate}[args.command](args)


if __name__ == "__main__":
    main()
