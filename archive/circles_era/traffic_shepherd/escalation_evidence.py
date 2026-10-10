"""Offline tests of escalation evidence, independent of traffic-state labels.

These functions score supplied measurements. Passing a synthetic regression is
not physical evidence that a wall works. Nothing here runs a simulator, selects
a damping checkpoint, or supplies global outcomes to an online decision maker.
"""
from dataclasses import dataclass, field
from math import isfinite
from statistics import mean
from types import MappingProxyType

import numpy as np


COSTS = (
    "braking_seconds_per_request", "effort_per_request", "system_seconds_per_request",
    "ttc_seconds_per_request", "source_wait_seconds_per_request",
)

BENEFIT_ELIGIBILITY_POLICY = MappingProxyType(dict(
    require_complete_requested_service_in_both_arms=True,
    maximum_treatment_collision_involvements=0,
    maximum_treatment_starting_teleports=0,
))


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def outcome_costs(summary):
    """Fixed-horizon costs, with every requested vehicle in the denominator.

    Completing the registered simulation horizon does not imply that every
    requested trip arrived. With unfinished service, the time integral is
    restricted system time through that horizon. Such outcomes remain valid
    characterization evidence; paired_effect records their service eligibility.
    """
    if summary.get("mechanics_only") is not False or summary.get("complete_fixed_horizon") is not True:
        raise ValueError("full-horizon physical outcomes required; mechanics cannot label benefit")
    population = summary["population"]
    if type(population) is not int or population <= 0:
        raise ValueError("positive requested population required")
    totals = summary["totals"]
    costs = dict(
        braking_seconds_per_request=totals["braking_vehicle_seconds"] / population,
        effort_per_request=totals["acceleration_energy"] / population,
        system_seconds_per_request=summary["system_seconds_per_requested_vehicle"],
        ttc_seconds_per_request=totals["ttc_below_3_vehicle_seconds"] / population,
        source_wait_seconds_per_request=summary["source_wait_vehicle_seconds"] / population,
    )
    if any(not finite(x) or x < 0 for x in costs.values()):
        raise ValueError("invalid or missing measured costs")
    for key in ("collision_involvements", "starting_teleports"):
        if not finite(totals[key]) or totals[key] < 0 or int(totals[key]) != totals[key]:
            raise ValueError("invalid safety counts")
    if type(summary["unfinished_requested"]) is not int or not 0 <= summary["unfinished_requested"] <= population:
        raise ValueError("invalid unfinished service count")
    due = summary.get("requested_due_at_horizon", population)
    if type(due) is not int or not 0 <= summary["unfinished_requested"] <= due <= population:
        raise ValueError("invalid requested service horizon")
    if "arrived" in summary and (type(summary["arrived"]) is not int or
                                 summary["arrived"] != due-summary["unfinished_requested"]):
        raise ValueError("arrival and unfinished service counts disagree")
    return costs


def paired_effect(control_manifest, control, treatment_manifest, treatment):
    """Reject mismatched pairs rather than interpreting them as treatment effects.

Equal scenario identity binds the request/type/CAV assignment, network and
scenario. Identical predecision state must additionally be tested by the wall
runner; summary-level matching alone cannot certify counterfactual replay.
"""
    if control_manifest.get("cav_speed_mode", 31) != treatment_manifest.get("cav_speed_mode", 31):
        raise ValueError("unmatched pair: CAV braking interfaces differ")
    for manifest, summary in ((control_manifest, control), (treatment_manifest, treatment)):
        if manifest.get("status") != "complete":
            raise ValueError("incomplete or failed run cannot form an outcome pair")
        for key in ("seed", "controller", "duration_s", "scenario_identity"):
            if key not in manifest or summary.get(key) != manifest[key]:
                raise ValueError(f"manifest/summary mismatch: {key}")
        if manifest.get("mechanics_only") is not False:
            raise ValueError("mechanics manifest cannot form an outcome pair")
    for key in ("seed", "mainline_vph", "penetration", "duration_s", "scenario_identity",
                "config_sha256", "model_sha256", "sumo_binary_sha256", "active_source_sha256"):
        if key not in control_manifest or control_manifest[key] != treatment_manifest.get(key):
            raise ValueError(f"unmatched pair: {key}")
    if control["population"] != treatment["population"]:
        raise ValueError("requested populations differ")
    if treatment_manifest["controller"] == "wall":
        if control_manifest["controller"] != "damping":
            raise ValueError("wall effect requires the identical damping control")
        for key in ("proposal_sha256", "predecision_state_sha256", "exogenous_future_sha256", "decision_time_s"):
            if key not in control_manifest or control_manifest[key] != treatment_manifest.get(key):
                raise ValueError(f"unmatched wall counterfactual: {key}")
        decision = control_manifest["decision_time_s"]
        if not finite(decision) or not 0 <= decision < control_manifest["duration_s"]:
            raise ValueError("invalid wall decision time")
    left, right = outcome_costs(control), outcome_costs(treatment)
    return dict(
        seed=control_manifest["seed"], mainline_vph=control_manifest["mainline_vph"],
        penetration=control_manifest["penetration"],
        control=control_manifest["controller"], treatment=treatment_manifest["controller"],
        costs_control=left, costs_treatment=right,
        requested_population=control["population"],
        complete_requested_service_control=(control["unfinished_requested"] == 0 and
            control.get("requested_due_at_horizon", control["population"]) == control["population"]),
        complete_requested_service_treatment=(treatment["unfinished_requested"] == 0 and
            treatment.get("requested_due_at_horizon", treatment["population"]) == treatment["population"]),
        control_unfinished_requested=control["unfinished_requested"],
        treatment_unfinished_requested=treatment["unfinished_requested"],
        control_collision_involvements=control["totals"]["collision_involvements"],
        treatment_collision_involvements=treatment["totals"]["collision_involvements"],
        control_starting_teleports=control["totals"]["starting_teleports"],
        treatment_starting_teleports=treatment["totals"]["starting_teleports"],
        service_interpretation="restricted_system_time_until_all_requested_trips_arrive",
        # Positive improvement always means lower physical cost.
        improvements={key: left[key] - right[key] for key in COSTS},
        excess_collisions=treatment["totals"]["collision_involvements"] - control["totals"]["collision_involvements"],
        excess_teleports=treatment["totals"]["starting_teleports"] - control["totals"]["starting_teleports"],
        excess_unfinished=treatment["unfinished_requested"] - control["unfinished_requested"],
        interpretation="measured_effect_not_baseline_acceptance",
    )


@dataclass(frozen=True)
class BenefitCriteria:
    """Explicit, preregistered physical margins; there are no accepted defaults.

At least one named primary outcome must improve materially, while every cost
stays within its declared maximum increase. Margins are in COSTS units, never
percentages with unstable denominators. Neither congestion nor density is a gate.
Complete requested service in both arms and zero treatment collisions/teleports
are eligibility requirements, independent of the configurable cost margins.
"""
    primary_minimum_improvement: dict[str, float]
    maximum_cost_increase: dict[str, float]
    eligibility_policy: dict[str, bool | int] = field(default_factory=lambda: dict(BENEFIT_ELIGIBILITY_POLICY))

    def __post_init__(self):
        if not self.primary_minimum_improvement or not set(self.primary_minimum_improvement) <= set(COSTS):
            raise ValueError("declare at least one recognized primary physical outcome")
        if set(self.maximum_cost_increase) != set(COSTS):
            raise ValueError("declare a guardrail for every physical cost")
        if any(not finite(v) or v <= 0 for v in self.primary_minimum_improvement.values()):
            raise ValueError("primary improvements must be finite and strictly positive")
        if any(not finite(v) or v < 0 for v in self.maximum_cost_increase.values()):
            raise ValueError("cost tolerances must be finite and nonnegative")
        if self.eligibility_policy != BENEFIT_ELIGIBILITY_POLICY or any(
            type(self.eligibility_policy[key]) is not type(value) for key, value in BENEFIT_ELIGIBILITY_POLICY.items()
        ):
            raise ValueError("complete service and zero treatment events cannot be waived")
        object.__setattr__(self, "primary_minimum_improvement", MappingProxyType(dict(self.primary_minimum_improvement)))
        object.__setattr__(self, "maximum_cost_increase", MappingProxyType(dict(self.maximum_cost_increase)))
        object.__setattr__(self, "eligibility_policy", MappingProxyType(dict(self.eligibility_policy)))

    def as_dict(self):
        """Serialize margins together with the fixed scientific eligibility rule."""
        return dict(primary_minimum_improvement=dict(self.primary_minimum_improvement),
                    maximum_cost_increase=dict(self.maximum_cost_increase),
                    eligibility_policy=dict(self.eligibility_policy))


def benefit_label(effect, criteria):
    """Score wall versus identical damping, including harmful and failed attempts.

Do not filter on formation success before computing intent-to-treat effects.
Formation delivery is a separate label. A costly failed assembly stays in data.
Measured fixed-horizon harms remain visible when trips are unfinished, but a
truncated service comparison cannot receive a beneficial efficacy label.
"""
    if effect.get("control") != "damping" or effect.get("treatment") != "wall":
        raise ValueError("wall benefit must compare wall with the identical damping arm")
    gains = effect["improvements"]
    if set(gains) != set(COSTS) or any(not finite(v) for v in gains.values()):
        raise ValueError("complete finite paired cost differences required")
    safety = ("excess_collisions", "excess_teleports", "excess_unfinished")
    if any(not finite(effect.get(key)) for key in safety):
        raise ValueError("explicit safety and completion differences required")
    violations = [key for key in safety if effect[key] > 0]
    violations += [key for key in COSTS if gains[key] < -criteria.maximum_cost_increase[key]]
    benefits = [key for key, margin in criteria.primary_minimum_improvement.items() if gains[key] >= margin]

    # Old saved effect records contain only excess counts. Preserve their known
    # harms, but never assume missing absolute counts or service flags were zero.
    counts = ("control_unfinished_requested", "treatment_unfinished_requested",
              "control_collision_involvements", "treatment_collision_involvements",
              "control_starting_teleports", "treatment_starting_teleports")
    flags = ("complete_requested_service_control", "complete_requested_service_treatment")
    missing = [key for key in (*counts, *flags) if key not in effect]
    for key in counts:
        if key in effect and (not finite(effect[key]) or effect[key] < 0 or int(effect[key]) != effect[key]):
            raise ValueError(f"invalid absolute measured count: {key}")
    for key in flags:
        if key in effect and type(effect[key]) is not bool:
            raise ValueError(f"explicit complete-service boolean required: {key}")
    for key in ("treatment_collision_involvements", "treatment_starting_teleports"):
        if effect.get(key, 0) > 0:
            violations.append(key)
    ineligible_service = [key for key in flags if key in effect and not effect[key]]
    ineligible_service += [key for key in counts[:2] if effect.get(key, 0) > 0]
    evidence_complete = not missing
    service_complete = evidence_complete and not ineligible_service
    label = ("harmful" if violations else "ineligible_evidence" if missing else
             "ineligible_service" if not service_complete else
             "beneficial" if benefits else "nonbeneficial")
    return dict(label=label,
                material_improvements=benefits, violated_guardrails=violations,
                eligible_for_beneficial_label=service_complete and not violations,
                ineligible_service_reasons=ineligible_service,
                missing_absolute_evidence=missing,
                baseline_safety_anomaly=any(effect.get(key, 0) > 0 for key in
                    ("control_collision_involvements", "control_starting_teleports")),
                criteria=criteria.as_dict(), congestion_required=False)


def two_gate_decision(*, feasibility, benefit_probability, probability_threshold,
                      evidence_current, wall_active=False, cooldown=False):
    """Decision-composition test interface; not a trained or calibrated predictor.

The production wall API must still bind both approvals to identical proposal
and report hashes. A high score cannot override missing information or members.
"""
    if not finite(probability_threshold) or not 0 <= probability_threshold <= 1:
        raise ValueError("invalid probability threshold")
    if any(type(v) is not bool for v in (evidence_current, wall_active, cooldown)):
        raise ValueError("explicit boolean observation and lifecycle status required")
    if wall_active or cooldown:
        return "continue_current_control"
    if not evidence_current or feasibility != "feasible":
        return "damping"
    if benefit_probability is None:
        return "damping"
    if not finite(benefit_probability) or not 0 <= benefit_probability <= 1:
        raise ValueError("invalid benefit probability")
    return "assemble" if benefit_probability >= probability_threshold else "damping"


def split_audit(partitions, *, development_excluded=()):
    """Episode-level separation: all windows/proposals of a seed stay together."""
    if set(partitions) != {"training", "calibration", "prospective"}:
        raise ValueError("explicit training/calibration/prospective seed partitions required")
    seen = set()
    for name, seeds in partitions.items():
        if not seeds or any(type(s) is not int for s in seeds) or len(seeds) != len(set(seeds)):
            raise ValueError(f"invalid seed partition: {name}")
        if seen & set(seeds):
            raise ValueError("seed leakage across episode partitions")
        seen.update(seeds)
    if set(partitions["prospective"]) & set(development_excluded):
        raise ValueError("development-exposed seed in prospective partition")
    return {"disjoint": True, "freshness_established": False,
            "note": "A separate archive exposure audit is still required."}


def probability_metrics(labels, probabilities, *, threshold, bins=5):
    """Discrimination operating point plus Brier/reliability, including abstention.

Call separately for feasibility, benefit and penetration strata. Model selection
uses development/calibration episodes; prospective results cannot retune a gate.
"""
    if not labels or len(labels) != len(probabilities) or any(type(y) is not int or y not in (0, 1) for y in labels):
        raise ValueError("aligned nonempty binary labels required")
    if not finite(threshold) or not 0 <= threshold <= 1 or type(bins) is not int or bins <= 0:
        raise ValueError("invalid operating point or calibration bins")
    if any(p is not None and (not finite(p) or not 0 <= p <= 1) for p in probabilities):
        raise ValueError("invalid predicted probability")
    available = [(y, p) for y, p in zip(labels, probabilities) if p is not None]
    tp = sum(y == 1 and p is not None and p >= threshold for y, p in zip(labels, probabilities))
    fp = sum(y == 0 and p is not None and p >= threshold for y, p in zip(labels, probabilities))
    fn = sum(labels) - tp  # Abstentions do not inflate overall recall.
    tn = len(labels) - tp - fp - fn
    reliability = []
    for i in range(bins):
        rows = [(y, p) for y, p in available if min(int(p * bins), bins - 1) == i]
        if rows:
            reliability.append(dict(lower=i/bins, upper=(i+1)/bins, n=len(rows),
                                    mean_probability=mean(p for _, p in rows), observed_rate=mean(y for y, _ in rows)))
    return dict(n=len(labels), available=len(available), abstentions=len(labels)-len(available),
                tp=tp, fp=fp, fn=fn, tn=tn, precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                false_positive_rate=fp/(fp+tn) if fp+tn else None,
                brier=mean((p-y)**2 for y,p in available) if available else None,
                reliability=reliability, scientifically_accepted=False)


def seed_cluster_summary(rows, metric, *, bootstrap_samples=2000, rng_seed=0):
    """Equal-seed average and descriptive seed bootstrap; never window-level CI."""
    if not rows or type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError("nonempty evidence and positive bootstrap count required")
    groups = {}
    for row in rows:
        if type(row["seed"]) is not int or not finite(row[metric]):
            raise ValueError("finite effects and exact seed IDs required")
        groups.setdefault(row["seed"], []).append(row[metric])
    values = np.array([mean(groups[s]) for s in sorted(groups)])
    ci = None
    if len(values) >= 2:
        rng = np.random.default_rng(rng_seed)
        estimates = [float(np.mean(rng.choice(values, len(values), replace=True))) for _ in range(bootstrap_samples)]
        ci = [float(x) for x in np.quantile(estimates, [.025, .975])]
    return dict(seeds=len(values), windows=len(rows), equal_seed_mean=float(values.mean()),
                descriptive_95pct_interval=ci, population_generalization_established=False)


def formation_delivery(samples, *, required_lanes, assembly_deadline_s,
                       hold_duration_s, formed_span_m, speed_spread_mps,
                       readiness_samples, dt_s=1.0):
    """Measure recorded trajectories, never infer success from a programmed cutoff.

Each row is the start-of-interval measured state: time_s, phase and members
[{id,lane,distance_m,speed_mps}]. The first HOLD boundary must have a current
consecutive readiness streak, and a terminal release/abort sample is required.
Candidate span during assembly has no 50 m veto. A recorded release phase is
not actuator proof: optional release-row ``owners`` and ``wall_command_ids``
verify ownership separately. Front/rear safety and delayed release effects
remain separate physical outcomes, never inferred from compactness alone.
"""
    if not samples or not required_lanes or len(set(required_lanes)) != len(required_lanes):
        raise ValueError("nonempty samples and explicit distinct required lanes needed")
    if any(not finite(v) or v <= 0 for v in (assembly_deadline_s, hold_duration_s, formed_span_m, dt_s)):
        raise ValueError("positive finite physical durations/span needed")
    if not finite(speed_spread_mps) or speed_spread_mps < 0 or type(readiness_samples) is not int or readiness_samples < 1:
        raise ValueError("invalid readiness definition")
    start = samples[0]["time_s"]
    if not finite(start) or start < 0:
        raise ValueError("invalid initial clock")
    spans, readiness, readiness_at, first_readiness_at = [], 0, None, None
    hold_seconds, continuous_hold, longest_hold = 0., 0., 0.
    hold_entry_valid, hold_started_at, hold_integrity = False, None, True
    release_observed, release_valid, release_ownership_verified = False, False, False
    terminal_observed, aborted, abort_reasons = False, False, []
    violations, first_members, previous_phase = [], None, None
    transitions = {"assembly": {"assembly","hold","abort"}, "hold": {"hold","taper","release","abort"},
                   "taper": {"taper","release","abort"}, "release": {"release","damping"},
                   "abort": {"abort","damping"}, "damping": {"damping"}}
    for index, row in enumerate(samples):
        if not finite(row["time_s"]) or abs(row["time_s"] - start - index*dt_s) > 1e-6:
            raise ValueError("trajectory has missing, duplicated or irregular samples")
        phase = row["phase"]
        if phase not in {"assembly", "hold", "taper", "release", "abort", "damping"}:
            raise ValueError("unknown lifecycle phase")
        transition_valid = not ((previous_phase is None and phase != "assembly") or
            (previous_phase is not None and phase not in transitions[previous_phase]))
        if not transition_valid:
            violations.append("invalid_lifecycle_transition")
            if phase in {"assembly", "hold"}:
                hold_integrity = False
        members = row["members"]
        ids = [m["id"] for m in members]
        lanes = [m["lane"] for m in members]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate member identity")
        if any(not finite(m["distance_m"]) or not finite(m["speed_mps"]) or m["speed_mps"] < 0 for m in members):
            raise ValueError("invalid member measurements")
        if first_members is None:
            first_members = set(ids)
        if phase in {"assembly", "hold", "taper"} and set(ids) != first_members:
            violations.append("membership_changed_without_registered_handoff")
            if phase in {"assembly", "hold"}:
                hold_integrity = False
        span = max(m["distance_m"] for m in members)-min(m["distance_m"] for m in members) if members else None
        spread = max(m["speed_mps"] for m in members)-min(m["speed_mps"] for m in members) if members else None
        spans.append(span)
        coverage = len(lanes) == len(required_lanes) and set(lanes) == set(required_lanes)
        compact = coverage and span is not None and span <= formed_span_m and spread <= speed_spread_mps
        entering_hold = phase == "hold" and previous_phase != "hold"
        # Readiness must be CURRENT at the first HOLD boundary. Include that
        # boundary's measurement, which the controller observes before choosing
        # the next interval's command. Later HOLD samples cannot repair an
        # invalid entry retroactively.
        if phase == "assembly" or entering_hold:
            readiness = readiness + 1 if compact else 0
            if not compact:
                readiness_at = None
            elif readiness >= readiness_samples and readiness_at is None:
                readiness_at = row["time_s"]-start
                if first_readiness_at is None:
                    first_readiness_at = readiness_at
        if entering_hold:
            hold_started_at = row["time_s"]-start
            hold_entry_valid = (previous_phase == "assembly" and readiness >= readiness_samples and
                readiness_at is not None and readiness_at <= assembly_deadline_s and
                hold_started_at <= assembly_deadline_s)
            if not hold_entry_valid:
                hold_integrity = False
                violations.append("hold_without_timely_readiness")
        if phase == "hold":
            if compact:
                # The next sampled boundary establishes this interval's end.
                # A final HOLD row alone cannot invent another dt of delivery.
                if index + 1 < len(samples):
                    hold_seconds += dt_s
                    continuous_hold += dt_s
                    longest_hold = max(longest_hold, continuous_hold)
            else:
                hold_integrity = False
                continuous_hold = 0.
                violations.append("noncompact_or_incomplete_hold")
        elif previous_phase == "hold":
            continuous_hold = 0.
        if phase == "abort":
            terminal_observed, aborted = True, True
            if row.get("reason"):
                abort_reasons.append(str(row["reason"]))
        if phase == "release":
            terminal_observed, release_observed = True, True
            if previous_phase != "release":
                release_valid = (transition_valid and previous_phase in {"hold", "taper"} and
                    hold_entry_valid and longest_hold >= hold_duration_s and not aborted)
                if not release_valid:
                    violations.append("release_without_completed_hold")
            if "owners" in row or "wall_command_ids" in row:
                owners = row.get("owners", {})
                wall_commands = row.get("wall_command_ids")
                correct_ownership = (bool(first_members) and isinstance(owners, dict) and
                    all(owners.get(member) in {"damping", "native"} for member in first_members) and
                    isinstance(wall_commands, (list, tuple)) and not wall_commands)
                if not correct_ownership:
                    violations.append("release_ownership_not_returned")
                release_ownership_verified = release_ownership_verified or correct_ownership
        previous_phase = phase
    assembled = first_readiness_at is not None and first_readiness_at <= assembly_deadline_s
    hold_delivered = hold_entry_valid and longest_hold >= hold_duration_s and hold_integrity
    delivered = assembled and hold_delivered and release_valid and not aborted and not violations
    reasons = set(violations)
    if not assembled:
        reasons.add("no_timely_readiness")
    if longest_hold < hold_duration_s:
        reasons.add("insufficient_compact_hold")
    if not release_observed:
        reasons.add("release_not_observed")
    if aborted:
        reasons.add("aborted")
    return dict(initial_span_m=spans[0], maximum_observed_span_m=max((s for s in spans if s is not None), default=None),
                assembly_complete=assembled, readiness_time_s=readiness_at,
                first_readiness_time_s=first_readiness_at, hold_started_at_s=hold_started_at,
                compact_hold_seconds=hold_seconds, longest_contiguous_compact_hold_seconds=longest_hold,
                compact_hold_delivered=hold_delivered, complete_observation_horizon=terminal_observed,
                release_observed=release_observed, release_completed=release_valid and not violations,
                release_ownership_verified=(release_ownership_verified and
                    "release_ownership_not_returned" not in violations),
                delivered=delivered, failure_reasons=sorted(reasons), abort_reasons=sorted(set(abort_reasons)),
                violations=sorted(set(violations)), safety_established=False)
