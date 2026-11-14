from counterfactual.headroom import (
    HeadroomMargins,
    TreatmentRepairCriteria,
    classify_case,
    summarize_headroom,
    summarize_treatment_repair,
)


def _row(
    seed,
    step,
    *,
    delay=1000,
    wall_delay=700,
    discharge=20,
    wall_discharge=24,
    engaged=True,
    hard_brake=10,
    wall_hard_brake=11,
    low_ttc=2,
    wall_low_ttc=3,
    collisions=0,
    wall_collisions=0,
    feature=1.0,
    control_collapse=False,
    wall_collapse=False,
    predecision_collapse=False,
):
    def outcomes(d, q, hb, lt, c, collapse):
        return {
            "incremental_delay_s": d,
            "queue_area_vehicle_s": d,
            "merge_discharge_count": q,
            "hard_brake_vehicle_steps": hb,
            "low_ttc_exposure_steps": lt,
            "collision_unique_vehicle_count": c,
            "starting_teleport_unique_vehicle_count": 0,
            "sustained_congestion": collapse,
            "hard_brake_attribution": {
                "event_start_count": hb,
                "unique_vehicle_count": hb,
            },
        }

    return {
        "eligible": True,
        "seed": seed,
        "decision_step": step,
        "features": {"observable": feature},
        "predecision_congestion": {
            "sustained_congestion": predecision_collapse,
        },
        "treatment_dose": {
            "activation_feasible": engaged,
            "realized_duration_fraction": 1.0 if engaged else 0.0,
            "lane_coverage_fraction": 1.0 if engaged else 0.0,
        },
        "arms": {
            "0": {
                "outcomes": outcomes(
                    delay,
                    discharge,
                    hard_brake,
                    low_ttc,
                    collisions,
                    control_collapse,
                )
            },
            "1": {
                "wall_engaged": engaged,
                "outcomes": outcomes(
                    wall_delay,
                    wall_discharge,
                    wall_hard_brake,
                    wall_low_ttc,
                    wall_collisions,
                    wall_collapse,
                ),
            },
        },
    }


def test_safe_meaningful_wall_is_selected():
    result = classify_case(_row(1, 120))
    assert result["wall_selected"] is True
    assert result["safety_admissible"] is True
    assert result["traffic_noninferior"] is True


def test_collision_or_large_service_loss_forces_damping():
    collision = classify_case(_row(1, 120, wall_collisions=1))
    service = classify_case(_row(1, 120, wall_delay=1600, wall_discharge=15))
    assert collision["wall_selected"] is False
    assert collision["safety_admissible"] is False
    assert service["wall_selected"] is False
    assert service["traffic_noninferior"] is False


def test_queue_area_is_diagnostic_not_a_rejection_criterion():
    row = _row(1, 120)
    row["arms"]["1"]["outcomes"]["queue_area_vehicle_s"] = 99999
    assert classify_case(row)["wall_selected"] is True


def test_headroom_requires_both_actions_and_multiple_beneficial_seeds():
    rows = [
        _row(1, 120, feature=2.0),
        _row(2, 120, feature=2.2),
        _row(3, 120, feature=2.4),
        _row(4, 120, wall_delay=1400, wall_discharge=18, feature=-1.0),
        _row(5, 120, wall_delay=1400, wall_discharge=18, feature=-1.2),
        _row(6, 120, wall_delay=1400, wall_discharge=18, feature=-1.4),
    ]
    summary = summarize_headroom(rows, HeadroomMargins())
    assert summary["passed"] is True
    assert summary["wall_selected_case_count"] == 3
    assert summary["damping_selected_case_count"] == 3


def test_treatment_repair_gate_requires_completed_safe_outcome_diversity():
    rows = [
        _row(1, 120, control_collapse=True),
        _row(2, 120, control_collapse=True),
        _row(3, 120, control_collapse=True),
        _row(
            4,
            120,
            wall_delay=1400,
            wall_discharge=18,
            control_collapse=True,
            wall_collapse=True,
        ),
        _row(
            5,
            120,
            wall_delay=1400,
            wall_discharge=18,
            control_collapse=True,
            wall_collapse=True,
        ),
        _row(
            6,
            120,
            wall_delay=1400,
            wall_discharge=18,
            control_collapse=True,
            wall_collapse=True,
        ),
    ]

    summary = summarize_treatment_repair(rows, effects_exceed_repeat_noise=True)

    assert summary["passed"] is True
    assert summary["dose_qualified_fraction"] == 1.0
    assert summary["beneficial_completed_seed_count"] == 3
    assert summary["nonbeneficial_completed_seed_count"] == 3
    assert summary["collapse_prevention_seed_count"] == 3
    assert summary["collapse_nonprevention_seed_count"] == 3


def test_safety_service_gate_reports_but_does_not_require_collapse_diversity():
    rows = [
        _row(1, 120),
        _row(2, 120),
        _row(3, 120),
        _row(4, 120, wall_delay=1400, wall_discharge=18),
        _row(5, 120, wall_delay=1400, wall_discharge=18),
        _row(6, 120, wall_delay=1400, wall_discharge=18),
    ]

    summary = summarize_treatment_repair(
        rows,
        TreatmentRepairCriteria(require_collapse_prevention_diversity=False),
        effects_exceed_repeat_noise=True,
    )

    assert summary["passed"] is True
    assert summary["collapse_prevention_seed_count"] == 0
    assert summary["collapse_checks"]["collapse_prevention_spans_seeds"] is False
    assert "collapse_prevention_spans_seeds" not in summary["checks"]


def test_treatment_repair_gate_rejects_compact_override_without_hold_or_geometry():
    rows = [_row(seed, 120, control_collapse=True) for seed in range(1, 7)]
    for row in rows:
        row["treatment_dose"].update(
            {
                "assembly_completed": False,
                "holding_fraction_of_override": 0.0,
                "wall_geometry": {
                    "geometry_mode": "compact_moving",
                    "fraction_within_configured_span": 0.25,
                },
            }
        )

    summary = summarize_treatment_repair(rows, effects_exceed_repeat_noise=True)

    assert summary["passed"] is False
    assert summary["dose_qualified_case_count"] == 0


def test_treatment_repair_gate_rejects_short_or_uncovered_dose():
    rows = [_row(seed, 120) for seed in range(1, 6)]
    for row in rows[:2]:
        row["treatment_dose"]["realized_duration_fraction"] = 0.25
        row["treatment_dose"]["lane_coverage_fraction"] = 0.25

    summary = summarize_treatment_repair(rows, effects_exceed_repeat_noise=True)

    assert summary["passed"] is False
    assert summary["dose_qualified_fraction"] == 0.6
    assert summary["checks"]["most_feasible_activations_receive_80pct_dose"] is False


def test_treatment_repair_counts_started_assembly_even_if_case_becomes_ineligible():
    rows = [_row(seed, 120) for seed in range(1, 7)]
    failed = rows[-1]
    failed["eligible"] = False
    failed["treatment_dose"]["realized_duration_fraction"] = 0.5
    failed["treatment_dose"]["lane_coverage_fraction"] = 0.5

    summary = summarize_treatment_repair(rows, effects_exceed_repeat_noise=True)

    assert summary["eligible_case_count"] == 5
    assert summary["feasible_activation_count"] == 6
    assert summary["dose_qualified_case_count"] == 5
    assert summary["dose_qualified_fraction"] == 5 / 6


def test_treatment_repair_excludes_predecision_supervisor_refusals():
    rows = [_row(seed, 120) for seed in range(1, 7)]
    refused = rows[-1]
    refused["predecision_supervisor"] = {
        "profile": "local_support_density_coherence_v1",
        "admitted": False,
    }
    refused["treatment_dose"]["realized_duration_fraction"] = 0.0
    refused["treatment_dose"]["lane_coverage_fraction"] = 0.0

    summary = summarize_treatment_repair(
        rows,
        TreatmentRepairCriteria(require_collapse_prevention_diversity=False),
        effects_exceed_repeat_noise=True,
    )

    assert summary["supervisor_admitted_case_count"] == 5
    assert summary["feasible_activation_count"] == 5
    assert summary["dose_qualified_fraction"] == 1.0


def test_treatment_repair_fails_closed_without_repeat_noise_evidence():
    rows = [
        _row(seed, 120, control_collapse=seed <= 3)
        if seed <= 3
        else _row(seed, 120, wall_delay=1400, wall_discharge=18)
        for seed in range(1, 7)
    ]

    summary = summarize_treatment_repair(
        rows,
        TreatmentRepairCriteria(require_collapse_prevention_diversity=False),
    )

    assert summary["passed"] is False
    assert summary["repeat_noise_evidence_provided"] is False
    assert summary["checks"]["meaningful_effects_exceed_repeat_noise"] is False
