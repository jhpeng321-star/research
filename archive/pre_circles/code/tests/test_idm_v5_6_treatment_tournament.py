from experiments.idm.wall_v5_6_a_low_authority_contract import (
    DECELERATION_CAP_MPS2, overlay_overrides, profile_manifest,
)
from experiments.idm.run_idm_v5_6_treatment_tournament import CANDIDATE_ID, _phase_split_integrity, build_summary


def test_candidate_a_changes_both_active_servo_limits_only():
    overrides = overlay_overrides()
    assert overrides["compact_assembly_a_max_dec_mps2"] == DECELERATION_CAP_MPS2 == 0.3
    assert overrides["compact_servo_a_max_dec_mps2"] == DECELERATION_CAP_MPS2
    assert overrides["wall_params"]["a_max_dec"] == 0.3
    assert overrides["geometry_mode"] == "compact_moving"
    assert overrides["planned_treatment_duration_s"] == 20.0


def test_candidate_a_profile_is_fingerprinted_and_identified():
    manifest = profile_manifest()
    assert manifest["profile_id"].endswith("low_authority")
    assert len(manifest["profile_sha256"]) == 64
    assert CANDIDATE_ID == "A_low_authority"


def test_summary_requires_every_gate(tmp_path):
    root = tmp_path; cases = root / "cases"; cases.mkdir()
    for seed in (4401, 4402, 4404, 4405):
        (cases / f"seed{seed}.json").write_text(__import__("json").dumps({
            "seed": seed, "integrity": {"x": True},
            "safety_service_guardrail": {"safety_admissible": True, "traffic_noninferior": True},
            "primary_physical_effect": {"primary_physical_benefit": seed in (4402, 4405), "induces_sustained_congestion": False},
            "treatment_dose": {"controlled_cav_seconds": 2.0}, "follower_compression": {},
        }), encoding="utf-8")
    assert build_summary(root)["four_case_passed"] is True


def test_phase_split_integrity_rejects_high_authority_leak_and_window_extension():
    rows = [
        {"treatment_state": "ASSEMBLING", "control_deltas": {"cav.0": {"wall_a_max_dec_mps2": 1.0}}, "lane_coverage": {"coverage_fraction": 1.0}, "current_member_span_m": 10.0, "active_treatment_steps": 1},
        {"treatment_state": "SETTLING_TO_LOW_AUTHORITY", "control_deltas": {"cav.0": {"wall_a_max_dec_mps2": 0.5}}, "lane_coverage": {"coverage_fraction": 1.0}, "current_member_span_m": 10.0, "settling_max_jerk_mps3": 0.5, "settling_tau_frozen": True, "active_treatment_steps": 2},
        {"treatment_state": "HOLDING", "control_deltas": {"cav.0": {"wall_a_max_dec_mps2": 0.3}}, "lane_coverage": {"coverage_fraction": 1.0}, "current_member_span_m": 10.0, "active_treatment_steps": 3},
    ]
    assert all(_phase_split_integrity(rows).values())
    rows[-1]["control_deltas"]["cav.0"]["wall_a_max_dec_mps2"] = 0.4
    rows[-1]["active_treatment_steps"] = 21
    gates = _phase_split_integrity(rows)
    assert gates["hold_and_taper_within_0_3"] is False
    assert gates["treatment_window_not_extended"] is False
