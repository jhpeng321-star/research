from experiments.idm.run_idm_v5_5_1_four_case_repair import _adaptive_dose


def _row(step, state, *, override=True, blend=None, reason=None, abort=None):
    return {
        "relative_step": step,
        "wall_override_ids": ("cav.1", "cav.2") if override else (),
        "treatment_state": state,
        "lane_coverage": {"coverage_fraction": 1.0},
        "taper_blend_fraction": blend,
        "taper_reason": reason,
        "abort_reason": abort,
    }


def test_adaptive_taper_is_policy_complete_but_not_mislabeled_as_twenty_second_hold():
    rows = [
        _row(11, "ASSEMBLING"),
        _row(12, "HOLDING"),
        _row(13, "TAPERING", blend=0.8, reason="adaptive_soft_compression:risk"),
        _row(14, "TAPERING", blend=0.6, reason="adaptive_soft_compression:risk"),
        _row(15, "TAPERING", blend=0.4, reason="adaptive_soft_compression:risk"),
        _row(16, "TAPERING", blend=0.2, reason="adaptive_soft_compression:risk"),
        _row(17, "TAPERING", blend=0.0, reason="adaptive_soft_compression:risk"),
        _row(31, "RELEASED", override=False),
    ]
    audit = {"treatment_audit_rows": rows, "treatment_dose": {"controlled_cav_seconds": 14}, "checks": {"release_returns_to_damping": True}}
    result = _adaptive_dose(audit, 10)
    assert result["policy_completed"] is True
    assert result["completion_reason"] == "adaptive_release_complete"
    assert result["realized_holding_duration_s"] == 1
    assert result["requested_intervention_window_s"] == 20
    assert result["requested_and_realized_are_distinct"] is True


def test_hard_abort_is_not_policy_completion():
    rows = [_row(11, "HOLDING", abort="controlled_follower_hard_brake"), _row(31, "RELEASED", override=False)]
    audit = {"treatment_audit_rows": rows, "treatment_dose": {"controlled_cav_seconds": 2}, "checks": {"release_returns_to_damping": True}}
    result = _adaptive_dose(audit, 10)
    assert result["policy_completed"] is False
    assert result["completion_reason"] == "hard_abort"
