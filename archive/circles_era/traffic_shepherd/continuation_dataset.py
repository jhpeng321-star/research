"""Programmatic continuation decision dataset built from immutable artifacts.

Version 2 fixes the v1 defects: an explicit sign convention
(cost difference = treatment − reference; negative = improvement; the stored
"improvement" is asserted to be exactly its negative), post-decision
transition measurements separated from decision content, artifact identities
bound by hash, guardrail failures recomputed from the original criteria, and
explicit refusal/implementation-failure statuses. Entry-decision rows are NOT
reproduced here: they remain the registered natural_dataset_v2 panel plus the
recorded wall arms; this dataset is decision-B only.

Damping from the beginning is an overall-policy benchmark, never an available
HOLD+5 action. Improving an existing wall's continuation does not
automatically justify assembling it.
"""
import json
from pathlib import Path

from . import paired_experiments as paired
from .config import ROOT
from .escalation_evidence import BenefitCriteria, benefit_label

SCHEMA = "continuation_decision_dataset/2"
CRITERIA = dict(
    primary_minimum_improvement={"system_seconds_per_request": 1.0,
                                 "braking_seconds_per_request": 0.02},
    maximum_cost_increase={"system_seconds_per_request": 0.5,
                           "source_wait_seconds_per_request": 0.1,
                           "braking_seconds_per_request": 0.005,
                           "effort_per_request": 0.5,
                           "ttc_seconds_per_request": 0.02},
    eligibility_policy={"require_complete_requested_service_in_both_arms": True,
                        "maximum_treatment_collision_involvements": 0,
                        "maximum_treatment_starting_teleports": 0})
COST_METRICS = ("system_seconds_per_request", "braking_seconds_per_request",
                "effort_per_request", "ttc_seconds_per_request",
                "source_wait_seconds_per_request")
KEEP_SOURCES = {
    4107: ("s4107_m2700_p20_t600_lifecycle_v2_wall",
           ROOT/"runs/active/lifecycle_repair_v2/jobs/s4107_m2700_p20_t600_lifecycle_v2_wall"),
    4108: ("s4108_m2700_p20_t600_cohort_v1_wall",
           ROOT/"runs/active/cohort_eligibility_v1/jobs/s4108_m2700_p20_t600_cohort_v1_wall"),
}
REFERENCE_JOBS = {
    4107: ROOT/"runs/active/observer_replay_v1/jobs/s4107_m2700_p20_damping",
    4108: ROOT/"runs/active/observer_replay_v1/jobs/s4108_m2700_p20_damping",
}
RELEASE_JOBS = {
    4107: ROOT/"runs/active/duration_release_v1/jobs/s4107_m2700_p20_t600_release_h5_wall",
    4108: ROOT/"runs/active/duration_release_v1/jobs/s4108_m2700_p20_t600_release_h5_wall",
}
DOWN_JOBS = {
    4107: ROOT/"runs/active/target_adjust_v1/jobs/s4107_m2700_p20_t600_hold_adjust_down_wall",
    4108: ROOT/"runs/active/target_adjust_v1/jobs/s4108_m2700_p20_t600_hold_adjust_down_wall",
}
UP_JOBS = {
    4107: ROOT/"runs/active/target_adjust_v3/jobs/s4107_m2700_p20_t600_hold_adjust_up_wall",
    4108: ROOT/"runs/active/target_adjust_v3/jobs/s4108_m2700_p20_t600_hold_adjust_up_wall",
}
UP_FAILURE = dict(seed=4107, action="raise_26", status="implementation_failure",
                  job_id="s4107_m2700_p20_t600_hold_adjust_up_wall",
                  job_path=str(ROOT/"runs/active/target_adjust_v2/jobs/"
                               "s4107_m2700_p20_t600_hold_adjust_up_wall"),
                  charged_simulated_seconds=2400,
                  failure="adjusted proposal 26 m/s rejected by the unchanged entry "
                          "profile bound (targets <= 24 m/s) at the branch tick; "
                          "preventable configuration/implementation failure, "
                          "preserved and superseded by hold_target_adjust/2",
                  physical_outcome_label=None)
BRANCH = {4107: 617.0, 4108: 623.0}


def _costs(job_path):
    summary = json.loads((Path(job_path)/"summary.json").read_text())
    population = summary["population"]
    costs = {m: summary["totals"][k]/population for m, k in (
        ("system_seconds_per_request", "requested_not_arrived_vehicle_seconds"),
        ("braking_seconds_per_request", "braking_vehicle_seconds"),
        ("effort_per_request", "acceleration_energy"),
        ("ttc_seconds_per_request", "ttc_below_3_vehicle_seconds"))}
    costs["source_wait_seconds_per_request"] = summary["source_wait_vehicle_seconds"]/population
    return summary, costs, population


def _effect(treatment_summary, reference_summary):
    """Effect dict in the registered paired-effect shape."""
    t, r = treatment_summary, reference_summary
    pop = r["population"]
    keys = (
        ("system_seconds_per_request", "requested_not_arrived_vehicle_seconds"),
        ("braking_seconds_per_request", "braking_vehicle_seconds"),
        ("effort_per_request", "acceleration_energy"),
        ("ttc_seconds_per_request", "ttc_below_3_vehicle_seconds"))
    costs_control = {m: r["totals"][k]/pop for m, k in keys}
    costs_treatment = {m: t["totals"][k]/pop for m, k in keys}
    costs_control["source_wait_seconds_per_request"] = r["source_wait_vehicle_seconds"]/pop
    costs_treatment["source_wait_seconds_per_request"] = t["source_wait_vehicle_seconds"]/pop
    improvements = {m: costs_control[m]-costs_treatment[m] for m in costs_control}
    excess_collisions = (t["totals"]["collision_involvements"]
                         - r["totals"]["collision_involvements"])
    excess_teleports = (t["totals"]["starting_teleports"]
                        - r["totals"]["starting_teleports"])
    excess_unfinished = t["unfinished_requested"] - r["unfinished_requested"]
    return dict(control="damping", treatment="wall",
        costs_control=costs_control, costs_treatment=costs_treatment,
        improvements=improvements,
        requested_population=pop,
        complete_requested_service_control=bool(r["all_requested_service_completed"]),
        complete_requested_service_treatment=bool(t["all_requested_service_completed"]),
        control_unfinished_requested=r["unfinished_requested"],
        treatment_unfinished_requested=t["unfinished_requested"],
        control_collision_involvements=r["totals"]["collision_involvements"],
        treatment_collision_involvements=t["totals"]["collision_involvements"],
        control_starting_teleports=r["totals"]["starting_teleports"],
        treatment_starting_teleports=t["totals"]["starting_teleports"],
        excess_collisions=excess_collisions, excess_teleports=excess_teleports,
        excess_unfinished=excess_unfinished)


def _verify_prefix(seed, action, treatment_job):
    """Verify the pre-branch identity claim against its stored evidence.

    Returns (verified, evidence_path). Adjust arms carry a
    target_adjust_contract.json whose prefix block records the exact stream
    comparison; it must be present and exact, otherwise the claim fails.
    """
    contract = Path(treatment_job)/"target_adjust_contract.json"
    if contract.exists():
        data = json.loads(contract.read_text())
        prefix = data.get("prefix") or (data.get("checks") and {})
        exact = bool(prefix.get("exact")) if isinstance(prefix, dict) else None
        if exact is None:
            checks = data.get("checks") or {}
            exact = bool(checks.get("prefix_exact"))
        if exact is not True:
            raise ValueError(f"prefix evidence for {action} seed {seed} is not exact")
        return True, str(contract)
    raise ValueError(f"no stored prefix evidence for {action} seed {seed}")


def build_row(*, seed, action, treatment_job, reference_job, adjust_mps,
              prefix_verified=None, prefix_evidence=None, status="delivered",
              registered_analysis=None, registered_job_id=None,
              post_decision_measurements=None, decision_time_s=None):
    treatment_summary, t_costs, t_pop = _costs(treatment_job)
    reference_summary, r_costs, r_pop = _costs(reference_job)
    if t_pop != r_pop:
        raise ValueError("population mismatch between treatment and reference")
    cost_difference = {m: t_costs[m]-r_costs[m] for m in t_costs}
    improvement = {m: -cost_difference[m] for m in cost_difference}
    effect = _effect(treatment_summary, reference_summary)
    label = benefit_label(effect, BenefitCriteria(**CRITERIA))
    failures = [m for m, d in cost_difference.items()
                if d > CRITERIA["maximum_cost_increase"][m]]
    if effect["treatment_collision_involvements"] > 0:
        failures.append("treatment_collision_involvements")
    if effect["treatment_starting_teleports"] > 0:
        failures.append("treatment_starting_teleports")
    if not (effect["complete_requested_service_treatment"]
            and effect["complete_requested_service_control"]):
        failures.append("incomplete_requested_service")
    registered_label = (registered_analysis or {}).get("label")
    if registered_label is not None and label["label"] != registered_label:
        # Registered labels come from the paired analyses; a mismatch means the
        # recomputation or the mapping is wrong, never a silent override.
        raise ValueError(f"label recomputation mismatch for {action} seed {seed}: "
                         f"{label['label']} vs {registered_analysis}")
    if status == "delivered" and action == "keep_24":
        verified, evidence = True, ("identical execution source; alias recorded in "
            "output/lifecycle_repair_v2/HEADROOM_AUDIT.json")
        prefix_verified, prefix_evidence = verified, evidence
    elif status == "delivered" and action == "release_h5":
        reg = json.loads((ROOT/"runs/active/duration_release_v1/registration.json").read_text())
        entry = reg.get("continuations", {}).get(
            f"s{seed}_m2700_p20_t600_release_h5_wall")
        if not entry or not entry.get("replay_proof"):
            raise ValueError(f"no registered replay proof for release_h5 seed {seed}")
        verified, evidence = True, ("runs/active/duration_release_v1/registration.json "
            "replay proof; predecision/prefix hashes recorded at registration")
        prefix_verified, prefix_evidence = verified, evidence
    elif status == "delivered":
        verified, evidence = _verify_prefix(seed, action, treatment_job)
        prefix_verified, prefix_evidence = verified, evidence
    else:
        prefix_verified, prefix_evidence = (False, "not delivered; no prefix claim")
    row = dict(seed_group=seed, entry_time_s=600.0,
        decision_time_s=decision_time_s if decision_time_s is not None else BRANCH[seed],
        decision_type="continuation", branch_time_s=BRANCH[seed],
        executed_history=dict(source_job_id=KEEP_SOURCES[seed][0],
            source_job_path=str(KEEP_SOURCES[seed][1]),
            source_manifest_sha256=paired.digest(
                json.loads((KEEP_SOURCES[seed][1]/"manifest.json").read_text()))
            if (KEEP_SOURCES[seed][1]/"manifest.json").exists() else None,
            prefix_verified=prefix_verified, prefix_evidence=prefix_evidence),
        **({"cost_difference_treatment_minus_reference": None,
            "improvement_negative_of_cost_difference": None,
            "original_label": None, "guardrail_failures": None}
           if status != "delivered" else {}),
        action=dict(kind=action, adjust_mps=adjust_mps,
            entry_target_mps=24.0,
            continuation_target_mps=24.0+adjust_mps if adjust_mps is not None else None),
        artifacts=dict(treatment_job=str(treatment_job),
            treatment_manifest_sha256=paired.digest(
                json.loads((Path(treatment_job)/"manifest.json").read_text())),
            reference_job=str(reference_job),
            reference_manifest_sha256=paired.digest(
                json.loads((Path(reference_job)/"manifest.json").read_text())),
            keep_source_job=str(KEEP_SOURCES[seed][1])),
        costs_treatment_per_request=t_costs, costs_reference_per_request=r_costs,
        cost_difference_treatment_minus_reference=cost_difference,
        improvement_negative_of_cost_difference=improvement,
        requested_population=t_pop,
        collisions_treatment=effect["treatment_collision_involvements"],
        collisions_reference=effect["control_collision_involvements"],
        original_label=label["label"], guardrail_failures=failures,
        status=status,
        post_decision_measurements=post_decision_measurements or {})
    # Meaningful checks: sign relations and registered cross-checks.
    for m in cost_difference:
        assert abs(improvement[m] + cost_difference[m]) < 1e-12
    if registered_analysis is not None:
        for m, v in registered_analysis.get("cost_difference", {}).items():
            assert abs(cost_difference[m]-v) < 1e-6, (action, seed, m, cost_difference[m], v)
    return row


def _registered_down_rows():
    analysis = json.loads((ROOT/"runs/active/target_adjust_v1/analysis.json").read_text())
    out = {}
    for r in analysis["results"]:
        seed = r["seed"]
        w = r["wall_vs_wall"]["metrics"]["requested_not_arrived_vehicle_seconds"]["per_request"]
        out[seed] = dict(label=r["benefit"]["label"],
            cost_difference=dict(system_seconds_per_request=r["effect"]["improvements"]["system_seconds_per_request"]*-1,
                braking_seconds_per_request=r["effect"]["improvements"]["braking_seconds_per_request"]*-1,
                effort_per_request=r["effect"]["improvements"]["effort_per_request"]*-1,
                ttc_seconds_per_request=r["effect"]["improvements"]["ttc_seconds_per_request"]*-1,
                source_wait_seconds_per_request=r["effect"]["improvements"]["source_wait_seconds_per_request"]*-1),
            service_vs_keep=w)
    return out


def _registered_up_rows():
    analysis = json.loads((ROOT/"runs/active/target_adjust_v3/analysis.json").read_text())
    out = {}
    for r in analysis["results"]:
        seed = r["seed"]
        contract = json.loads((ROOT/"runs/active/target_adjust_v3/jobs"/
            r["job_id"]/"target_adjust_contract.json").read_text())
        out[seed] = dict(label=r["benefit"]["label"],
            transition_s=contract["adjustment"]["transition_to_adjusted_target_s"],
            near_s=contract["adjustment"]["time_near_adjusted_target_s"],
            cost_difference=dict(
                system_seconds_per_request=r["effect"]["improvements"]["system_seconds_per_request"]*-1,
                braking_seconds_per_request=r["effect"]["improvements"]["braking_seconds_per_request"]*-1,
                effort_per_request=r["effect"]["improvements"]["effort_per_request"]*-1,
                ttc_seconds_per_request=r["effect"]["improvements"]["ttc_seconds_per_request"]*-1,
                source_wait_seconds_per_request=r["effect"]["improvements"]["source_wait_seconds_per_request"]*-1))
    return out


def build_dataset():
    registered = _registered_down_rows()
    rows = []
    for seed in (4107, 4108):
        rows.append(build_row(seed=seed, action="keep_24", adjust_mps=0.0,
            treatment_job=KEEP_SOURCES[seed][1], reference_job=REFERENCE_JOBS[seed],
            prefix_evidence="identical execution source (keep row is its own source)",
            registered_analysis=None))
        reg = registered[seed]
        rows.append(build_row(seed=seed, action="lower_22", adjust_mps=-2.0,
            treatment_job=DOWN_JOBS[seed], reference_job=REFERENCE_JOBS[seed],
            registered_analysis=dict(label=reg["label"], cost_difference=reg["cost_difference"]),
            post_decision_measurements=dict(
                note="measured after the branch; never available at the decision",
                transition_to_adjusted_target_s=4.0,
                time_near_adjusted_target_s=11.0 if seed == 4107 else 11.0)))
        rows.append(build_row(seed=seed, action="release_h5", adjust_mps=None,
            treatment_job=RELEASE_JOBS[seed], reference_job=REFERENCE_JOBS[seed],
            prefix_evidence="runs/active/duration_release_v1 registration replay proofs "
                            "(predecision identity at 600 s; branch at 617/623)",
            registered_analysis=None))
        up_registered = _registered_up_rows()
        rows.append(build_row(seed=seed, action="raise_26", adjust_mps=2.0,
            treatment_job=UP_JOBS[seed], reference_job=REFERENCE_JOBS[seed],
            registered_analysis=dict(label=up_registered[seed]["label"],
                cost_difference=up_registered[seed]["cost_difference"]),
            post_decision_measurements=dict(
                note="measured after the branch; never available at the decision",
                transition_to_adjusted_target_s=up_registered[seed]["transition_s"],
                time_near_adjusted_target_s=up_registered[seed]["near_s"])))
    rows.append(dict(UP_FAILURE, seed_group=UP_FAILURE["seed"],
        entry_time_s=600.0, decision_time_s=BRANCH[UP_FAILURE["seed"]],
        decision_type="continuation",
        branch_time_s=BRANCH[UP_FAILURE["seed"]],
        executed_history=dict(source_job_id=KEEP_SOURCES[UP_FAILURE["seed"]][0],
            prefix_verified=True,
            prefix_evidence="same registered prefix as target_adjust_v3 up arm; "
                            "failure occurred at the branch before divergence"),
        action=dict(kind=UP_FAILURE["action"], adjust_mps=2.0,
            entry_target_mps=24.0, continuation_target_mps=26.0)))
    rows[-1].pop("seed")
    dataset = dict(schema=SCHEMA,
        contract="docs/OBSERVATION_DECISION_CONTRACT.md",
        sign_convention="cost difference = treatment - reference; negative = improvement; "
                        "improvement is asserted to be exactly the negative",
        decision="B_continuation", branch_rule="measured_HOLD_start_plus_5_s",
        independent_seed_groups=2, qualifying_actions=0,
        fit_admissibility="not_admissible_two_seed_groups_zero_positives",
        entry_dataset_note="Entry-decision rows are the registered natural_dataset_v2 panel "
            "plus the recorded wall arms (see docs/EXPERIMENT_LEDGER.md). Damping from the "
            "beginning is an overall-policy benchmark, not an available HOLD+5 action. "
            "Improving an existing wall's continuation does not automatically justify "
            "assembling it.",
        rows=rows)
    return dataset
