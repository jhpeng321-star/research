"""Registered continuation comparators for the hold-extension campaign.

Builds the three explicitly named comparisons from registered artifacts
only: (A) extension versus frozen damping, (B) extension versus the
same-prefix raise-26 H20 arm, (C) extension versus keep-24. C never
isolates the extension alone: it confounds the 24->26 m/s raise and the
+10 s hold; it is reported for continuity with earlier keep-24 tables
only.

Every comparison carries explicit reference identity, population,
scenario and (for B) prefix checks. Reference identity is declared, never
inferred from a shared seed: the module rejects a reference whose job id
differs from the registered one, and cross-checks computed differences
against the registered wall_vs_wall blocks (which pin the reference
artifacts). Sign convention: cost difference = treatment - reference;
negative = improvement. No simulation, no new criteria.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "hold_extend_comparator/1"
SIGN_CONVENTION = "cost difference = treatment - reference; negative = improvement"

SEEDS = (4107, 4108)

# Registered arm identities, one per seed. keep-24 4107 is the executed
# lifecycle_v2 wall; keep-24 4108 is the executed cohort_v1 wall.
EXTENSION_JOB = {4107: "s4107_m2700_p20_t600_hold26_ext10_wall",
                 4108: "s4108_m2700_p20_t600_hold26_ext10_wall"}
EXTENSION_BLOCK = {4107: "hold_extend_v1", 4108: "hold_extend_v2"}
RAISE_H20_JOB = {4107: "s4107_m2700_p20_t600_hold_adjust_up_wall",
                 4108: "s4108_m2700_p20_t600_hold_adjust_up_wall"}
KEEP24_JOB = {4107: "s4107_m2700_p20_t600_lifecycle_v2_wall",
              4108: "s4108_m2700_p20_t600_cohort_v1_wall"}
DAMPING_JOB = {4107: "s4107_m2700_p20_damping", 4108: "s4108_m2700_p20_damping"}
EXTENSION_BRANCH_S = {4107: 632.0, 4108: 638.0}

# metric -> per-request cost key in registered effect blocks
COST_KEYS = {
    "service": "system_seconds_per_request",
    "braking": "braking_seconds_per_request",
    "effort": "effort_per_request",
    "ttc": "ttc_seconds_per_request",
    "wait": "source_wait_seconds_per_request",
}
METRICS = tuple(COST_KEYS)

# Total-key names in paired_escalation_outcomes/1 summaries.
TOTAL_KEYS = {
    "service": "requested_not_arrived_vehicle_seconds",
    "braking": "braking_vehicle_seconds",
    "effort": "acceleration_energy",
    "ttc": "ttc_below_3_vehicle_seconds",
}

# Original registered criteria, unchanged (identical values in every
# registered analysis block; cross-checked by tests against the artifacts).
CRITERIA = dict(
    primary_minimum_improvement={"service": 1.0, "braking": 0.02},
    maximum_cost_increase={"service": 0.5, "wait": 0.1, "braking": 0.005,
                           "effort": 0.5, "ttc": 0.02},
    eligibility="complete requested service in both arms; zero treatment "
                "collisions/teleports/unfinished")

EFFORT_FORMULA = (
    "effort = acceleration_energy = sum over physics steps (dt = 0.5 s) of "
    "sum over all live vehicles of measured acceleration^2 "
    "(traffic_shepherd.metrics.braking_effort: a*a*dt). Units: m^2/s^3 in "
    "total over the 2400 s horizon; effort_per_request = total / requested "
    "population, so the criterion and guardrail are m^2/s^3 per request. "
    "It is an integral of squared acceleration, not fuel energy. Braking is "
    "vehicle time with measured acceleration below -3 m/s^2 (s per request).")


class ReferenceMismatch(ValueError):
    pass


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def per_request_costs(costs_block):
    """Re-key and validate a registered per-request cost block."""
    out = {}
    for metric, key in COST_KEYS.items():
        value = costs_block[key]
        if not _number(value) or value < 0:
            raise ValueError(f"invalid registered cost {key}: {value!r}")
        out[metric] = float(value)
    return out


def per_request_from_summary(summary):
    """Costs from a paired_escalation_outcomes/1 summary (registered totals)."""
    population = summary["population"]
    totals = summary["totals"]
    costs = {"service": float(summary["system_seconds_per_requested_vehicle"])}
    for metric, key in TOTAL_KEYS.items():
        if metric == "service":
            continue
        if not _number(totals[key]) or totals[key] < 0:
            raise ValueError(f"invalid registered total {key}")
        costs[metric] = float(totals[key]) / population
    costs["wait"] = float(summary["source_wait_vehicle_seconds"]) / population
    return costs


def cost_difference(treatment, reference):
    """treatment - reference per metric; negative = improvement."""
    if set(treatment) != set(METRICS) or set(reference) != set(METRICS):
        raise ValueError("cost dictionaries must carry exactly the registered metrics")
    return {m: treatment[m] - reference[m] for m in METRICS}


def require_same_conditions(treatment_meta, reference_meta):
    """Reject any pair that differs beyond the declared reference identity."""
    for key in ("seed", "requested_population", "mainline_vph", "penetration",
                "duration_s"):
        if treatment_meta.get(key) != reference_meta.get(key):
            raise ReferenceMismatch(
                f"unmatched comparison: {key} {treatment_meta.get(key)!r} vs "
                f"{reference_meta.get(key)!r}")
    expected = reference_meta.get("expected_job_id")
    if expected is not None and reference_meta.get("job_id") != expected:
        raise ReferenceMismatch(
            f"reference identity mismatch: got {reference_meta.get('job_id')!r}, "
            f"registered reference is {expected!r}; a shared seed never proves "
            "an identical reference arm")


def assert_total_consistent(per_request, total, population, metric, tolerance=1e-6):
    """Totals and per-request values are distinct quantities that must agree."""
    if abs(per_request * population - total) > tolerance:
        raise ReferenceMismatch(
            f"{metric}: per-request {per_request} x population {population} "
            f"does not reproduce registered total {total}")


def classify_difference(diff, criteria=CRITERIA):
    """Guardrail classification of one difference vector (no labeling)."""
    violated = sorted(m for m in METRICS if diff[m] > criteria["maximum_cost_increase"][m])
    within = sorted(m for m in METRICS
                    if 0 < diff[m] <= criteria["maximum_cost_increase"][m])
    improvements = sorted(m for m in METRICS if diff[m] < 0)
    material_primary = sorted(
        m for m in ("service", "braking")
        if diff[m] <= -criteria["primary_minimum_improvement"][m])
    return dict(violated_guardrails=violated,
                increases_within_allowance=within,
                improvements=improvements,
                material_primary_improvement=material_primary)


# ---- registered-artifact loading (paths relative to the repository root) ----

ANALYSIS_PATHS = {
    "extension": {s: f"runs/active/{EXTENSION_BLOCK[s]}/analysis.json" for s in SEEDS},
    "raise_h20": "runs/active/target_adjust_v3/analysis.json",
    "damping": {s: f"runs/active/observer_replay_v1/jobs/{DAMPING_JOB[s]}/summary.json"
                for s in SEEDS},
    "keep24": {s: (f"runs/active/lifecycle_repair_v2/jobs/{KEEP24_JOB[s]}/summary.json"
                   if s == 4107 else
                   f"runs/active/cohort_eligibility_v1/jobs/{KEEP24_JOB[s]}/summary.json")
               for s in SEEDS},
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _result_for(analysis, job_id):
    for result in analysis["results"]:
        if result["job_id"] == job_id:
            return result
    raise ReferenceMismatch(f"{job_id} absent from registered analysis")


def _extension_result(root, seed):
    analysis = json.loads((root / ANALYSIS_PATHS["extension"][seed]).read_text(encoding="utf-8"))
    return analysis, _result_for(analysis, EXTENSION_JOB[seed])


def _raise_result(root, seed):
    analysis = json.loads((root / ANALYSIS_PATHS["raise_h20"]).read_text(encoding="utf-8"))
    return analysis, _result_for(analysis, RAISE_H20_JOB[seed])


def _prefix_evidence(extension_result, raise_result, seed):
    """Same-prefix evidence for comparison B, from registered blocks only.

    A shared seed alone proves nothing: the registered prefix block of the
    extension arm records the exact stream match against the raise-26 H20
    source through the branch tick, and the raise arm's phases show its
    first divergence (taper) at that same tick.
    """
    prefix = extension_result["prefix"]
    branch = EXTENSION_BRANCH_S[seed]
    if not prefix.get("exact") or prefix.get("rows_before_s") != branch:
        raise ReferenceMismatch(
            f"seed {seed}: extension prefix evidence does not establish identity "
            f"through {branch} s")
    phases = {p["phase"]: p["time_s"] for p in raise_result["lifecycle"]["phases"]}
    if phases.get("taper") != branch:
        raise ReferenceMismatch(
            f"seed {seed}: raise-26 H20 source tapers at {phases.get('taper')} s, "
            f"not at the extension branch {branch} s")
    return dict(shared_prefix_through_s=branch,
                extension_prefix_exact=prefix["exact"],
                extension_rows_before_branch_s=prefix["rows_before_s"],
                raise_h20_taper_at_s=phases["taper"],
                raise_h20_release_at_s=phases.get("release"),
                streams=prefix["streams"],
                basis="registered extension-arm prefix block (exact stream match "
                      "against the raise-26 H20 source) plus the source's phase "
                      "schedule; not inferred from the shared seed")


def _cross_check_wall_vs_wall(diff, wall_vs_wall, population, seed):
    """The registered wall_vs_wall block pins the reference artifacts.

    Its direction string names new_target_minus_h20_source and its
    source_raise_h20.job_id names the raise arm; a comparison computed
    against any other arm cannot reproduce it.
    """
    if wall_vs_wall.get("direction") != "new_target_minus_h20_source":
        raise ReferenceMismatch("unexpected wall_vs_wall direction")
    key_map = {"braking": "braking_vehicle_seconds",
               "effort": "acceleration_energy",
               "ttc": "ttc_below_3_vehicle_seconds",
               "service": "requested_not_arrived_vehicle_seconds"}
    for metric, key in key_map.items():
        block = wall_vs_wall["metrics"][key]
        if abs(block["per_request"] - diff[metric]) > 1e-9:
            raise ReferenceMismatch(
                f"seed {seed} {metric}: computed difference {diff[metric]!r} does "
                f"not reproduce the registered wall_vs_wall per-request value "
                f"{block['per_request']!r} (wrong reference or arithmetic)")
        assert_total_consistent(diff[metric], block["full_difference"],
                                population, metric)
    collisions = wall_vs_wall["metrics"]["collision_involvements"]["full_difference"]
    if not isinstance(collisions, int):
        raise ValueError("collision difference must be an integer count")
    return collisions


def build_comparison_record(root=ROOT):
    root = Path(root)
    inputs = {}

    def register_input(name, path):
        inputs[name] = dict(path=str(path), sha256=_sha256(root / path))

    for seed in SEEDS:
        register_input(f"extension_analysis_{seed}", ANALYSIS_PATHS["extension"][seed])
        register_input(f"damping_summary_{seed}", ANALYSIS_PATHS["damping"][seed])
        register_input(f"keep24_summary_{seed}", ANALYSIS_PATHS["keep24"][seed])
    register_input("raise_h20_analysis", ANALYSIS_PATHS["raise_h20"])

    seeds = {}
    for seed in SEEDS:
        ext_analysis, ext = _extension_result(root, seed)
        raise_analysis, raise_result = _raise_result(root, seed)
        damping = json.loads((root / ANALYSIS_PATHS["damping"][seed]).read_text(encoding="utf-8"))
        keep24 = json.loads((root / ANALYSIS_PATHS["keep24"][seed]).read_text(encoding="utf-8"))
        effect = ext["effect"]
        population = effect["requested_population"]
        meta = dict(seed=seed, requested_population=population,
                    mainline_vph=effect["mainline_vph"],
                    penetration=effect["penetration"], duration_s=2400)

        # (A) extension vs frozen damping.
        damping_costs = per_request_from_summary(damping)
        treatment_costs = per_request_costs(effect["costs_treatment"])
        require_same_conditions(meta, dict(meta, job_id=DAMPING_JOB[seed],
                                           expected_job_id=DAMPING_JOB[seed]))
        # The effect block's registered control costs must be the damping arm's.
        registered_control = per_request_costs(effect["costs_control"])
        for metric in METRICS:
            if abs(registered_control[metric] - damping_costs[metric]) > 1e-9:
                raise ReferenceMismatch(
                    f"seed {seed}: registered damping control cost {metric} does "
                    "not match the reused damping summary")
        diff_a = cost_difference(treatment_costs, damping_costs)
        class_a = classify_difference(diff_a)
        improvement_cross = {m: -effect["improvements"][COST_KEYS[m]] for m in METRICS}
        for metric in METRICS:
            if abs(diff_a[metric] - improvement_cross[metric]) > 1e-9:
                raise ReferenceMismatch(f"seed {seed}: comparison A disagrees with "
                                        f"the registered improvements block ({metric})")

        # (B) extension vs the same-prefix raise-26 H20 arm.
        raise_costs = per_request_costs(raise_result["effect"]["costs_treatment"])
        require_same_conditions(meta, dict(meta, job_id=RAISE_H20_JOB[seed],
                                           expected_job_id=RAISE_H20_JOB[seed]))
        diff_b = cost_difference(treatment_costs, raise_costs)
        wvw = ext["wall_vs_wall"]
        if wvw["population"] != population:
            raise ReferenceMismatch("wall_vs_wall population mismatch")
        collision_diff_b = _cross_check_wall_vs_wall(diff_b, wvw, population, seed)
        if collision_diff_b != (effect["treatment_collision_involvements"]
                                - raise_result["effect"]["treatment_collision_involvements"]):
            raise ReferenceMismatch("collision difference disagrees with registered counts")
        prefix = _prefix_evidence(ext, raise_result, seed)

        # (C) extension vs keep-24 (never "the extension alone": it confounds
        # the raise and the extension).
        keep_costs = per_request_from_summary(keep24)
        require_same_conditions(meta, dict(meta, job_id=KEEP24_JOB[seed],
                                           expected_job_id=KEEP24_JOB[seed]))
        # Identity of the keep-24 arm is pinned by the registered source_h20
        # blocks of the raise analysis (job id + totals), not by this module.
        source_block = raise_result.get("source_h20") or raise_result.get("source")
        if source_block is None or source_block["job_id"] != KEEP24_JOB[seed]:
            raise ReferenceMismatch(
                f"seed {seed}: raise analysis does not pin keep-24 job "
                f"{KEEP24_JOB[seed]}")
        for metric, key in TOTAL_KEYS.items():
            assert_total_consistent(keep_costs[metric], source_block["totals"][key],
                                    population, metric)
        diff_c = cost_difference(treatment_costs, keep_costs)
        class_c = classify_difference(diff_c)

        seeds[str(seed)] = dict(
            population=population, meta=meta,
            A_extension_vs_frozen_damping=dict(
                reference=dict(job_id=DAMPING_JOB[seed],
                               controller="frozen CIRCLES damping (reused observer-replay arm)"),
                difference_per_request=diff_a,
                classification=class_a,
                absolute=dict(
                    treatment_collision_involvements=effect["treatment_collision_involvements"],
                    damping_collision_involvements=effect["control_collision_involvements"],
                    collision_difference=effect["treatment_collision_involvements"]
                    - effect["control_collision_involvements"],
                    treatment_teleports=effect["treatment_starting_teleports"],
                    treatment_unfinished=effect["treatment_unfinished_requested"]),
                eligibility=dict(
                    complete_requested_service_both_arms=(
                        effect["complete_requested_service_control"]
                        and effect["complete_requested_service_treatment"]),
                    note="the beneficial-label eligibility rule applies to this "
                         "damping comparison only"),
                registered_label=effect.get("label") or ext["benefit"]["label"]),
            B_extension_vs_same_prefix_raise26_H20=dict(
                reference=dict(job_id=RAISE_H20_JOB[seed],
                               controller="raise-26 H20 wall arm (same-prefix source)"),
                difference_per_request=diff_b,
                classification=classify_difference(diff_b),
                totals=dict(
                    ttc_vehicle_seconds=wvw["metrics"]["ttc_below_3_vehicle_seconds"]["full_difference"],
                    service_vehicle_seconds=wvw["metrics"]["requested_not_arrived_vehicle_seconds"]["full_difference"],
                    braking_vehicle_seconds=wvw["metrics"]["braking_vehicle_seconds"]["full_difference"],
                    effort_m2_s3=wvw["metrics"]["acceleration_energy"]["full_difference"]),
                collision_difference=collision_diff_b,
                wait_total_vehicle_seconds=diff_b["wait"] * population,
                prefix_evidence=prefix,
                interpretation_note="wall-vs-wall diagnostic only; the registered "
                                    "benefit label never applies to this comparison"),
            C_extension_vs_keep24_H20=dict(
                reference=dict(job_id=KEEP24_JOB[seed],
                               controller="keep-24 H20 wall arm (executed source)"),
                difference_per_request=diff_c,
                classification=class_c,
                collision_difference=(effect["treatment_collision_involvements"]
                                      - keep24["totals"]["collision_involvements"]),
                interpretation_note="confounds the 24->26 m/s raise with the +10 s "
                                    "extension; NEVER describes the extension alone"),
        )

    return dict(
        schema=SCHEMA,
        sign_convention=SIGN_CONVENTION,
        metric_units=dict(
            service="s/request (restricted system time until all requested trips arrive)",
            braking="veh-s/request (measured acceleration < -3 m/s^2)",
            effort=EFFORT_FORMULA,
            ttc="veh-s/request of TTC < 3 s exposure (0.5 veh-s per below-3 s sample at 0.5 s physics)",
            wait="s/request (source waiting)"),
        criteria=CRITERIA,
        inputs=inputs,
        superseded_claims=[
            dict(claim="REPORT_V2 'Question A' 4107 row compared extension with "
                       "raise-26 H20",
                 correction="its service/TTC/wait cells were computed against "
                            "keep-24 (lifecycle_v2 wall) and its braking/effort "
                            "cells match no registered pair; comparison B here is "
                            "the correct extension-minus-raise-H20 vector",
                 evidence="runs/active/hold_extend_v1/analysis.json wall_vs_wall "
                          "(direction new_target_minus_h20_source) vs "
                          "target_adjust_v3 costs"),
            dict(claim="4108 'all five fail; harmful' (REPORT_V2)",
                 correction="five continuous costs increase versus damping, but "
                            "only four violate their guardrails; source waiting "
                            "(+0.005738 s/request) stays within its 0.1 allowance",
                 evidence="runs/active/hold_extend_v2/analysis.json benefit block"),
            dict(claim="'closest any arm has come' (REPORT_V2)",
                 correction="unsupported: no collision-aware ranking of arms is "
                            "defined; withdrawn",
                 evidence="criteria make treatment collisions an eligibility "
                          "requirement, not a ranked metric"),
            dict(claim="'first continuation action to improve TTC' (REPORT_V2)",
                 correction="false and withdrawn: the raise-26 adjustment itself "
                            "improved TTC versus keep-24 in 4107 "
                            "(-103.0 veh-s, -0.0728/request; registered "
                            "target_adjust_v3 wall_vs_wall) before the extension "
                            "existed",
                 evidence="runs/active/target_adjust_v3/analysis.json"),
            dict(claim="4108 extension 'inert' / exact no-op",
                 correction="4108 shows small measured differences versus damping "
                            "and versus raise-26 H20 (e.g. +1.5 veh-s TTC, -1.0 "
                            "veh-s service versus raise-H20); source waiting is "
                            "identical only to registered float precision",
                 evidence="runs/active/hold_extend_v2/analysis.json wall_vs_wall"),
        ],
        seeds=seeds,
        generated_by="traffic_shepherd.hold_extend_comparisons (offline, zero simulation)")


def main():
    record = build_comparison_record(ROOT)
    out = ROOT / "output/hold_extend_patch_v1/COMPARISONS_V3.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, sort_keys=False), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
