"""Scientific interpretation regressions on synthetic evidence, not SUMO results."""
from copy import deepcopy

import pytest

from traffic_shepherd.escalation_evidence import (
    COSTS, BenefitCriteria, benefit_label, formation_delivery, outcome_costs,
    paired_effect, probability_metrics, seed_cluster_summary, split_audit, two_gate_decision,
)


def run(arm="damping", *, system=20., braking=10., population=100):
    manifest = dict(status="complete", seed=4107, mainline_vph=2000, penetration=.2,
                    duration_s=2400, controller=arm, scenario_identity="same-exogenous-scene",
                    config_sha256="config", model_sha256="frozen-model", sumo_binary_sha256="sumo",
                    active_source_sha256={"source.py":"fixed"}, mechanics_only=False)
    manifest.update(proposal_sha256="same-proposed-wall",predecision_state_sha256="same-prefix",
                    exogenous_future_sha256="same-future",decision_time_s=600.)
    summary = dict(seed=4107, controller=arm, duration_s=2400,
                   scenario_identity="same-exogenous-scene", mechanics_only=False, complete_fixed_horizon=True,
                   population=population, unfinished_requested=0, source_wait_vehicle_seconds=10.,
                   system_seconds_per_requested_vehicle=system,
                   totals=dict(braking_vehicle_seconds=braking, acceleration_energy=100.,
                               ttc_below_3_vehicle_seconds=10., collision_involvements=0, starting_teleports=0))
    return manifest, summary


def criteria():
    # Test-only margins, not proposed acceptance tolerances for the paper.
    return BenefitCriteria({"system_seconds_per_request": 1., "braking_seconds_per_request": .02},
                           {key: 0. for key in COSTS})


def test_uncongested_service_gain_qualifies_without_damping_failure():
    effect = paired_effect(*run(), *run("wall", system=18.))
    effect["congested"] = False
    assert benefit_label(effect, criteria())["label"] == "beneficial"
    effect["congested"] = True
    assert benefit_label(effect, criteria())["label"] == "beneficial"


def test_smoothness_gain_can_qualify_when_service_is_already_good():
    effect = paired_effect(*run(system=8.), *run("wall", system=8., braking=7.))
    assert benefit_label(effect, criteria())["material_improvements"] == ["braking_seconds_per_request"]


def test_weak_damping_is_characterization_not_an_automatic_rejection():
    effect = paired_effect(*run("native", system=10., braking=5.), *run(system=20., braking=10.))
    assert effect["improvements"]["system_seconds_per_request"] == -10.
    assert effect["interpretation"] == "measured_effect_not_baseline_acceptance"
    with pytest.raises(ValueError, match="identical damping"):
        benefit_label(effect, criteria())


def test_benefit_cannot_hide_ttc_harm_or_failed_assembly_cost():
    wall = run("wall", system=18.)
    wall[1]["totals"]["ttc_below_3_vehicle_seconds"] = 20.
    result = benefit_label(paired_effect(*run(), *wall), criteria())
    assert result["label"] == "harmful" and result["violated_guardrails"] == ["ttc_seconds_per_request"]
    failed = paired_effect(*run(), *run("wall", system=22.))
    failed["assembly_completed"] = False
    assert benefit_label(failed, criteria())["label"] == "harmful"


def test_neutral_case_and_explicit_bounded_tradeoff():
    assert benefit_label(paired_effect(*run(), *run("wall")), criteria())["label"] == "nonbeneficial"
    guardrails = {key: 0. for key in COSTS}
    guardrails["system_seconds_per_request"] = .2
    tradeoff = BenefitCriteria({"braking_seconds_per_request": .02}, guardrails)
    assert benefit_label(paired_effect(*run(), *run("wall", system=20.1, braking=7.)), tradeoff)["label"] == "beneficial"


@pytest.mark.parametrize("field", ["seed", "mainline_vph", "penetration", "scenario_identity", "model_sha256", "active_source_sha256", "sumo_binary_sha256"])
def test_unmatched_comparisons_cannot_create_labels(field):
    left, right = run(), run("wall")
    right[0][field] = "different"
    with pytest.raises(ValueError):
        paired_effect(*left, *right)


def test_failed_truncated_and_missing_outcomes_are_not_neutral():
    right = run("wall")
    right[0]["status"] = "failed"
    with pytest.raises(ValueError, match="incomplete"):
        paired_effect(*run(), *right)
    right = run("wall")
    right[1]["mechanics_only"] = True
    with pytest.raises(ValueError, match="full-horizon"):
        paired_effect(*run(), *right)
    right = run("wall")
    del right[1]["totals"]["ttc_below_3_vehicle_seconds"]
    with pytest.raises(KeyError):
        paired_effect(*run(), *right)


def test_full_requested_denominator_and_source_wait_are_preserved():
    row = run()[1]
    row["arrived"] = 90
    row["unfinished_requested"] = 10
    assert outcome_costs(row)["braking_seconds_per_request"] == .1
    assert outcome_costs(row)["source_wait_seconds_per_request"] == .1
    bad = deepcopy(row)
    bad["source_wait_vehicle_seconds"] = float("nan")
    with pytest.raises(ValueError):
        outcome_costs(bad)


def test_safety_and_unfinished_service_cannot_be_bought_by_benefit():
    for name in ("excess_collisions", "excess_teleports", "excess_unfinished"):
        effect = paired_effect(*run(), *run("wall", system=1.))
        effect[name] = 1
        assert benefit_label(effect, criteria())["label"] == "harmful"


def test_no_implicit_acceptance_thresholds():
    with pytest.raises(ValueError):
        BenefitCriteria({}, {key:0. for key in COSTS})
    with pytest.raises(ValueError):
        BenefitCriteria({"system_seconds_per_request":0.}, {key:0. for key in COSTS})
    with pytest.raises(ValueError):
        BenefitCriteria({"system_seconds_per_request":1.}, {})
    with pytest.raises(TypeError):
        criteria().maximum_cost_increase["system_seconds_per_request"] = 100.


@pytest.mark.parametrize("field", ["proposal_sha256", "predecision_state_sha256", "exogenous_future_sha256", "decision_time_s"])
def test_wall_labels_require_same_proposal_prefix_and_future(field):
    wall = run("wall",system=18.)
    del wall[0][field]
    with pytest.raises(ValueError,match="counterfactual"):
        paired_effect(*run(),*wall)


@pytest.mark.parametrize("feasibility", ["insufficient_members", "unsafe_geometry", "insufficient_information"])
def test_benefit_cannot_override_a_feasibility_refusal(feasibility):
    assert two_gate_decision(feasibility=feasibility, benefit_probability=1., probability_threshold=.8,
                             evidence_current=True) == "damping"


def test_two_gates_current_evidence_and_single_wall_state():
    args = dict(feasibility="feasible", benefit_probability=.9, probability_threshold=.8, evidence_current=True)
    assert two_gate_decision(**args) == "assemble"
    assert two_gate_decision(**{**args,"benefit_probability":None}) == "damping"
    assert two_gate_decision(**{**args,"evidence_current":False}) == "damping"
    assert two_gate_decision(**args, wall_active=True) == "continue_current_control"
    assert two_gate_decision(**args, cooldown=True) == "continue_current_control"
    with pytest.raises(ValueError,match="boolean"):
        two_gate_decision(**{**args,"evidence_current":"false"})


def test_no_adjacent_window_leakage_or_development_reuse():
    good = dict(training=[1,2], calibration=[3], prospective=[4])
    assert split_audit(good)["disjoint"] and not split_audit(good)["freshness_established"]
    with pytest.raises(ValueError, match="leakage"):
        split_audit({**good,"prospective":[2]})
    with pytest.raises(ValueError, match="development-exposed"):
        split_audit(good, development_excluded=[4])


def test_probability_scores_count_abstention_and_false_escalation():
    result = probability_metrics([1,1,0,0], [.9,None,.8,.1], threshold=.8)
    assert (result["tp"],result["fp"],result["fn"],result["tn"]) == (1,1,1,1)
    assert result["recall"] == .5 and result["precision"] == .5
    assert result["abstentions"] == 1
    assert probability_metrics([1], [None], threshold=.8)["brier"] is None


def test_seed_cluster_not_window_count_controls_uncertainty():
    rows = [{"seed":1,"gain":10.} for _ in range(100)] + [{"seed":2,"gain":-10.}]
    result = seed_cluster_summary(rows,"gain",bootstrap_samples=100)
    assert result["equal_seed_mean"] == 0. and result["seeds"] == 2
    assert seed_cluster_summary(rows[:100],"gain")["descriptive_95pct_interval"] is None


def trajectory(spans=(80.,65.,50.,45.,45.,45.,45.,45.)):
    return [dict(time_s=float(t), phase="assembly" if t < 4 else "hold" if t < 6 else "release",
                 members=[dict(id="a",lane="L0",distance_m=500.-20*t,speed_mps=20.),
                          dict(id="b",lane="L1",distance_m=500.-20*t+span,speed_mps=20.)])
            for t,span in enumerate(spans)]


def delivery(rows):
    return formation_delivery(rows,required_lanes=["L0","L1"],assembly_deadline_s=4.,hold_duration_s=2.,
                              formed_span_m=50.,speed_spread_mps=2.,readiness_samples=2)


def test_initial_separation_is_distinct_from_formed_wall_compactness():
    result = delivery(trajectory())
    assert result["initial_span_m"] == 80. and result["delivered"]
    assert result["readiness_time_s"] == 3. and not result["safety_established"]


def test_initially_close_does_not_prove_deliverability():
    result = delivery(trajectory((25.,30.,40.,45.,60.,70.,75.,75.)))
    assert result["assembly_complete"] and not result["delivered"]
    assert "noncompact_or_incomplete_hold" in result["violations"]


def test_deadline_missing_lane_member_change_and_missing_sample_are_visible():
    assert not delivery(trajectory((80.,80.,80.,80.,80.,80.,40.,40.)))["assembly_complete"]
    rows = trajectory()
    rows[4]["members"][1]["lane"] = "L0"
    assert not delivery(rows)["delivered"]
    rows = trajectory()
    rows[4]["members"][1]["id"] = "replacement"
    assert "membership_changed_without_registered_handoff" in delivery(rows)["violations"]
    with pytest.raises(ValueError, match="missing"):
        delivery(trajectory()[::2])
    assert not delivery(trajectory()[:4])["delivered"]


def test_aborted_assembly_cannot_restart_and_accumulate_a_successful_hold():
    rows = trajectory()
    rows[1]["phase"] = "abort"
    result = delivery(rows)
    assert not result["delivered"] and "invalid_lifecycle_transition" in result["violations"]


def test_past_readiness_does_not_authorize_hold_after_compactness_loss():
    result = delivery(trajectory((40.,40.,80.,80.,40.,40.,40.,40.)))
    assert result["assembly_complete"]  # A partial earlier formation remains visible.
    assert not result["delivered"]
    assert "hold_without_timely_readiness" in result["failure_reasons"]
    assert not result["compact_hold_delivered"]


def test_readiness_can_recover_before_the_deadline_without_past_latch():
    result = delivery(trajectory((40.,80.,40.,40.,40.,40.,40.,40.)))
    assert result["delivered"] and result["readiness_time_s"] == 3.


def test_hold_boundary_measurement_can_complete_the_current_readiness_streak():
    result = delivery(trajectory((80.,80.,80.,40.,40.,40.,40.,40.)))
    assert result["delivered"] and result["readiness_time_s"] == 4.


def test_past_readiness_does_not_allow_unbounded_assembly_delay():
    rows = trajectory((40.,)*8)
    for index, row in enumerate(rows):
        row["phase"] = "assembly" if index < 5 else "hold" if index < 7 else "release"
    result = delivery(rows)
    assert not result["delivered"]
    assert "hold_without_timely_readiness" in result["failure_reasons"]


def test_release_sample_completes_an_early_wall_without_waiting_for_max_deadline():
    rows = trajectory((40.,)*8)[:5]
    for index, row in enumerate(rows):
        row["phase"] = "assembly" if index < 2 else "hold" if index < 4 else "release"
    result = delivery(rows)
    assert result["delivered"] and result["complete_observation_horizon"]
    assert result["compact_hold_seconds"] == 2.
    assert result["release_completed"] and not result["release_ownership_verified"]


def test_ending_on_hold_does_not_invent_release_or_an_unobserved_final_interval():
    rows = trajectory((40.,)*8)[:6]
    for index, row in enumerate(rows):
        row["phase"] = "assembly" if index < 2 else "hold"
    result = delivery(rows)
    assert result["compact_hold_seconds"] == 3.  # Only intervals ending at another recorded sample.
    assert result["compact_hold_delivered"]
    assert not result["delivered"] and not result["complete_observation_horizon"]
    assert "release_not_observed" in result["failure_reasons"]


def test_separated_compact_hold_fragments_do_not_create_continuous_delivery():
    rows = trajectory((40.,40.,40.,80.,40.,40.,40.,40.))
    for index, row in enumerate(rows):
        row["phase"] = "assembly" if index < 2 else "hold" if index < 6 else "release"
    result = delivery(rows)
    assert result["compact_hold_seconds"] == 3.
    assert result["longest_contiguous_compact_hold_seconds"] == 2.
    assert not result["compact_hold_delivered"] and not result["delivered"]


def test_release_ownership_has_separate_explicit_evidence():
    rows = trajectory()
    for row in rows[6:]:
        row.update(owners={"a":"damping", "b":"native"}, wall_command_ids=[])
    result = delivery(rows)
    assert result["delivered"] and result["release_ownership_verified"]
    rows[6]["owners"]["a"] = "wall"
    result = delivery(rows)
    assert result["compact_hold_delivered"]  # Release failure does not erase delivered HOLD.
    assert not result["delivered"]
    assert "release_ownership_not_returned" in result["failure_reasons"]


def test_release_phase_with_a_live_wall_command_is_not_a_valid_handback():
    rows = trajectory()
    rows[6].update(owners={"a":"damping", "b":"damping"}, wall_command_ids=["a"])
    assert "release_ownership_not_returned" in delivery(rows)["violations"]


def test_abort_after_hold_stays_an_abort_with_reason_and_observed_delivery():
    rows = trajectory()
    for row in rows[6:]:
        row.update(phase="abort", reason="rear clearance lost")
    result = delivery(rows)
    assert result["compact_hold_delivered"] and not result["delivered"]
    assert result["complete_observation_horizon"]
    assert result["abort_reasons"] == ["rear clearance lost"]
    assert "aborted" in result["failure_reasons"]


def test_equal_unfinished_service_retains_costs_but_cannot_receive_positive_label():
    control, wall = run(), run("wall", system=18.)
    for _, summary in (control, wall):
        summary.update(unfinished_requested=50, arrived=50)
    effect = paired_effect(*control, *wall)
    assert effect["improvements"]["system_seconds_per_request"] == 2.
    assert effect["excess_unfinished"] == 0
    assert not effect["complete_requested_service_control"]
    assert effect["treatment_unfinished_requested"] == 50
    label = benefit_label(effect, criteria())
    assert label["label"] == "ineligible_service"
    assert not label["eligible_for_beneficial_label"]
    assert label["material_improvements"] == ["system_seconds_per_request"]


def test_native_damping_characterization_keeps_unfinished_service_effects():
    native, damping = run("native"), run("damping", system=22.)
    native[1]["unfinished_requested"] = 10
    damping[1]["unfinished_requested"] = 20
    effect = paired_effect(*native, *damping)
    assert effect["improvements"]["system_seconds_per_request"] == -2.
    assert effect["excess_unfinished"] == 10
    assert effect["control_unfinished_requested"] == 10
    assert effect["treatment_unfinished_requested"] == 20
    assert effect["interpretation"] == "measured_effect_not_baseline_acceptance"


def test_additional_unfinished_service_preserves_harmful_intent_to_treat_label():
    wall = run("wall", system=18.)
    wall[1]["unfinished_requested"] = 1
    label = benefit_label(paired_effect(*run(), *wall), criteria())
    assert label["label"] == "harmful" and not label["eligible_for_beneficial_label"]
    assert "excess_unfinished" in label["violated_guardrails"]
    assert label["ineligible_service_reasons"]


def test_not_yet_due_requests_do_not_count_as_complete_requested_service():
    control, wall = run(), run("wall", system=18.)
    for _, summary in (control, wall):
        summary.update(requested_due_at_horizon=90, arrived=90)
    effect = paired_effect(*control, *wall)
    assert effect["control_unfinished_requested"] == 0
    assert not effect["complete_requested_service_control"]
    assert benefit_label(effect, criteria())["label"] == "ineligible_service"


@pytest.mark.parametrize("metric,field", [
    ("collision_involvements", "treatment_collision_involvements"),
    ("starting_teleports", "treatment_starting_teleports"),
])
def test_equal_nonzero_safety_events_cannot_be_bought_by_service_gain(metric, field):
    control, wall = run(), run("wall", system=18.)
    for _, summary in (control, wall):
        summary["totals"][metric] = 1
    effect = paired_effect(*control, *wall)
    label = benefit_label(effect, criteria())
    assert label["label"] == "harmful" and not label["eligible_for_beneficial_label"]
    assert field in label["violated_guardrails"] and label["baseline_safety_anomaly"]


def test_missing_absolute_evidence_is_unknown_and_never_assumed_zero():
    effect = paired_effect(*run(), *run("wall", system=18.))
    del effect["treatment_collision_involvements"]
    label = benefit_label(effect, criteria())
    assert label["label"] == "ineligible_evidence"
    assert label["missing_absolute_evidence"] == ["treatment_collision_involvements"]
    effect["excess_teleports"] = 1
    assert benefit_label(effect, criteria())["label"] == "harmful"


@pytest.mark.parametrize("bad", [-1, True, .5, float("nan")])
def test_absolute_event_fields_cannot_contain_invalid_counts(bad):
    effect = paired_effect(*run(), *run("wall", system=18.))
    effect["treatment_collision_involvements"] = bad
    with pytest.raises(ValueError, match="absolute measured count"):
        benefit_label(effect, criteria())


def test_complete_service_flag_cannot_override_observed_unfinished_requests():
    effect = paired_effect(*run(), *run("wall", system=18.))
    effect["treatment_unfinished_requested"] = 1
    assert benefit_label(effect, criteria())["label"] == "ineligible_service"


def test_criterion_serialization_preserves_nonwaivable_service_and_safety_policy():
    registered = criteria().as_dict()
    assert BenefitCriteria(**registered).as_dict() == registered
    assert registered["eligibility_policy"]["require_complete_requested_service_in_both_arms"] is True
    registered["eligibility_policy"]["maximum_treatment_collision_involvements"] = 1
    with pytest.raises(ValueError, match="cannot be waived"):
        BenefitCriteria(**registered)
    with pytest.raises(TypeError):
        criteria().eligibility_policy["require_complete_requested_service_in_both_arms"] = False


def test_inconsistent_service_summary_is_rejected_without_relabeling_censoring():
    summary = run()[1]
    summary.update(arrived=100, unfinished_requested=10)
    with pytest.raises(ValueError, match="counts disagree"):
        outcome_costs(summary)
