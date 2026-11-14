import pytest

from tools.analyze_idm_p30_p50_development_screen import (
    _command_trace_contrast,
    _physical_predecision_identity,
)


def _discovery(action_set, percentile, mask):
    return {
        "decision": {
            "decision_step": 400,
            "lane_id": "lane_0",
            "candidates": [{"vehicle_id": "cav", "lane_id": "lane_0"}],
            "candidate_observations": {"cav": [0.25] * 134 + list(mask)},
            "pattern_context": {"idm_severe_trend": 0.2},
            "action_set": action_set,
            "terminal_reference_percentile": percentile,
        }
    }


def _branch(values):
    return {
        "target_vehicle_id": "cav",
        "command_rows": [
            {
                "relative_step": step,
                "vehicle_id": "cav",
                "projected_acceleration_mps2": value,
            }
            for step, value in enumerate(values, start=1)
        ],
    }


def test_physical_identity_ignores_treatment_identity_and_action_mask_bits():
    result = _physical_predecision_identity(
        _discovery("activation_speed_ceiling_v1", "p30", (1, 1)),
        _discovery("activation_speed_ceiling_p50_v1", "p50", (1, 0)),
    )
    assert all(result.values())


def test_command_trace_contract_requires_0p05_delta_on_30pct_of_shared_steps():
    result = _command_trace_contrast(
        _branch((0.0, 0.0, 0.0)),
        _branch((0.05, 0.0, 0.10)),
    )
    assert result["shared_command_steps"] == 3
    assert result["distinct_command_steps"] == 2
    assert result["distinct_step_fraction"] == pytest.approx(2.0 / 3.0)
    assert result["case_trace_distinct"] is True
