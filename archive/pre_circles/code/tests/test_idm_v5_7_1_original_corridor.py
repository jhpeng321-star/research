from __future__ import annotations

import json

from experiments.idm.run_idm_v5_7_1_original_corridor import (
    _baseline_progress_path, _congestion_runs, _material_benefit,
    _safety_service, _summary, _write_baseline_progress,
)
from experiments.idm.wall_v5_7_1_original_corridor_contract import (
    ASSEMBLY_MAX_S, ASSEMBLY_MIN_STEPS, DECISION_ABSOLUTE_S, HOLD_S,
    MATERIAL_EFFECT_S, MAX_DECELERATION_MPS2, OUTCOME_HORIZON_S, SEEDS,
    TAPER_S, TOTAL_TREATMENT_S, WALL_BAND_HI_M, WALL_BAND_LO_M,
    overlay_overrides, profile_manifest,
)
from idm.contract import SCENARIO_ROOT, transition_scenario_config


def test_original_corridor_contract_has_no_derived_network_or_timing_scan():
    manifest = profile_manifest()
    assert manifest["scenario"]["original_verified_long_idm_corridor"] is True
    assert manifest["scenario"]["derived_network_used"] is False
    assert manifest["decision_absolute_s"] == DECISION_ABSOLUTE_S == 880
    assert "extension" not in json.dumps(manifest).lower()
    assert transition_scenario_config(2700).parent == SCENARIO_ROOT


def test_expanded_band_and_lifecycle_are_exact():
    overrides = overlay_overrides()
    assert overrides["wall_params"]["d_lo"] == WALL_BAND_LO_M == 200.0
    assert overrides["wall_params"]["d_hi"] == WALL_BAND_HI_M == 1900.0
    assert overrides["compact_assembly_a_max_dec_mps2"] == MAX_DECELERATION_MPS2
    assert overrides["compact_servo_a_max_dec_mps2"] == MAX_DECELERATION_MPS2
    assert overrides["compact_min_assembly_steps"] == ASSEMBLY_MIN_STEPS == 4
    assert overrides["compact_assembly_duration_s"] == ASSEMBLY_MAX_S == 10.0
    assert overrides["compact_requested_hold_duration_s"] == HOLD_S == 40.0
    assert overrides["taper_duration_s"] == TAPER_S == 8.0
    assert overrides["planned_treatment_duration_s"] == TOTAL_TREATMENT_S == 58.0
    assert OUTCOME_HORIZON_S == 178


def test_congestion_runs_retains_middle_multiple_terminal_and_empty_intervals():
    assert _congestion_runs([False, True, True, False]) == [2]
    assert _congestion_runs([True, False, True, True, False, True]) == [1, 2, 1]
    assert _congestion_runs([False, True, True]) == [2]
    assert _congestion_runs([False, False]) == []
    assert _congestion_runs([True] * 60) == [60]


def _case(seed: int, *, safe: bool, benefit: bool) -> dict:
    return {
        "seed": seed, "wall_outcome_run": True, "derived_network_used": False,
        "case_safe_complete": safe,
        "effect": {"material_benefit": benefit, "predicate_reduction_s": MATERIAL_EFFECT_S if benefit else 0, "longest_run_reduction_s": MATERIAL_EFFECT_S if benefit else 0, "replay_noise_s": 0},
    }


def test_unsafe_or_service_loss_case_cannot_pass_summary(tmp_path):
    (tmp_path / "cases").mkdir()
    rows = [_case(seed, safe=True, benefit=index < 2) for index, seed in enumerate(SEEDS)]
    rows[-1]["case_safe_complete"] = False  # represents any frozen safety/service failure
    for row in rows:
        (tmp_path / "cases" / f"seed{row['seed']}.json").write_text(json.dumps(row))
    summary = _summary(tmp_path)
    assert summary["gates"]["all_identity"] is False
    assert summary["four_case_passed"] is False


def _safety_inputs():
    outcomes = {"collision_unique_vehicle_count": 0, "starting_teleport_unique_vehicle_count": 0, "hard_brake_vehicle_steps": 1, "min_ttc_s": 2.0, "incremental_delay_s": 10.0, "merge_discharge_count": 4}
    control = {"outcomes": dict(outcomes)}; treatment = {"outcomes": dict(outcomes)}
    audit = {"checks": {"override_only_during_treatment": True, "out_of_band_damping_preserved": True, "release_returns_to_damping": True}}
    dose = {"abort_reasons": [], "per_lane_handoff_counts": {"a": 1}, "duration_fraction": .9, "lane_coverage": {"minimum": 1.0}, "realized": {"assembly_s": 4, "hold_s": 40, "taper_s": 8}, "maximum_span_by_phase_m": {"ASSEMBLING": 50.0, "HOLDING": 50.0}, "maximum_authority_mps2": .3}
    return control, treatment, audit, dose, {"passed": True}


def test_each_machine_readable_safety_or_integrity_rejection_fails_closed():
    control, treatment, audit, dose, prefix = _safety_inputs()
    assert all(_safety_service(control, treatment, audit, dose, prefix, True, True).values())
    for key, mutate in (
        ("predecision", lambda: prefix.update(passed=False)),
        ("inband", lambda: audit["checks"].update(override_only_during_treatment=False)),
        ("outband", lambda: audit["checks"].update(out_of_band_damping_preserved=False)),
        ("release", lambda: audit["checks"].update(release_returns_to_damping=False)),
        ("hard_brake", lambda: treatment["outcomes"].update(hard_brake_vehicle_steps=2)),
        ("collision", lambda: treatment["outcomes"].update(collision_unique_vehicle_count=1)),
        ("teleport", lambda: treatment["outcomes"].update(starting_teleport_unique_vehicle_count=1)),
        ("ttc", lambda: treatment["outcomes"].update(min_ttc_s=1.4)),
        ("service", lambda: treatment["outcomes"].update(incremental_delay_s=11.0)),
        ("dose", lambda: dose.update(duration_fraction=.79)),
        ("span", lambda: dose["maximum_span_by_phase_m"].update(HOLDING=50.1)),
        ("handoff", lambda: dose["per_lane_handoff_counts"].update(a=2)),
        ("authority", lambda: dose.update(maximum_authority_mps2=.31)),
    ):
        control, treatment, audit, dose, prefix = _safety_inputs()
        mutate()
        assert not all(_safety_service(control, treatment, audit, dose, prefix, True, True).values()), key
    control, treatment, audit, dose, prefix = _safety_inputs()
    assert not all(_safety_service(control, treatment, audit, dose, prefix, False, True).values())
    assert not all(_safety_service(control, treatment, audit, dose, prefix, True, False).values())


def test_replay_noise_cannot_count_as_material_benefit():
    assert _material_benefit(10, 10, 0)
    assert not _material_benefit(10, 10, 10)
    assert not _material_benefit(9, 12, 0)


def test_baseline_progress_is_treatment_blind_and_reusable(tmp_path):
    _write_baseline_progress(tmp_path, 4401, "scout_complete", decision=432, elapsed_s=12.5)
    path = _baseline_progress_path(tmp_path, 4401)
    row = json.loads(path.read_text())
    assert row["schema"] == "idm_v5_7_1_baseline_progress/1"
    assert row["phase"] == "scout_complete"
    assert row["treatment_blind"] is True
    assert row["decision_relative_step"] == 432
    assert row["decision_absolute_s"] == 880
    assert row["phase_elapsed_s"] == 12.5
