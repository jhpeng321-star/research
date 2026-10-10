import numpy as np
import pytest

from tools.analyze_idm_structured_authority_screen import (
    _paired_command_dose,
    _spatial_effects,
)
from tools.run_idm_structured_authority_screen import _arms, _parse_cell


def test_cells_and_arms_are_deterministic_and_share_nominal_control():
    assert _parse_cell("2200:4107,4109") == (2200, (4107, 4109))
    arms = _arms(
        {"gain_values": [0.75, 1.0, 1.25], "reference_offsets_mps": [-1.0, 0.0, 1.0]}
    )
    assert [row["arm"] for row in arms[:2]] == ["nominal", "nominal_replay"]
    assert len(arms) == 6
    assert sum(row["is_control"] for row in arms) == 2


def test_paired_command_dose_excludes_analytical_baseline():
    control = {
        "command_rows": [
            {
                "relative_step": 1,
                "vehicle_id": "cav",
                "desired_acceleration_mps2": -0.4,
                "projected_acceleration_mps2": -0.3,
            }
        ]
    }
    action = {
        "command_rows": [
            {
                "relative_step": 1,
                "vehicle_id": "cav",
                "desired_acceleration_mps2": -0.6,
                "projected_acceleration_mps2": -0.4,
            }
        ]
    }
    result = _paired_command_dose(control, action)
    assert result["paired_preprojection_acceleration_seconds"] == pytest.approx(0.2)
    assert result["paired_executed_acceleration_seconds"] == pytest.approx(0.1)
    assert result["projection_survival_fraction"] == pytest.approx(0.5)


def test_spatial_effects_use_control_minus_action_for_costs():
    key = "lane.acceleration_energy_m2ps4"
    result = _spatial_effects(
        {key: np.asarray([4.0, 3.0])},
        {key: np.asarray([2.0, 2.0])},
    )
    assert result[f"{key}_improvement_sum"] == 3.0
