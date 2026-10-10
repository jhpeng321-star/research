import numpy as np

from experiments.counterfactual.train_counterfactual_merge_service_gate import (
    merge_service_gate_actions,
)


def test_merge_service_gate_requires_both_nonworsening_and_one_strict_gain():
    predictions = {
        "merge_speed_gain_mps": np.asarray([0.2, 0.2, -0.1, 0.0]),
        "merge_discharge_gain": np.asarray([0.0, -1.0, 1.0, 0.0]),
        "soft_hard_brake_rate_vehicle_steps_per_s": np.asarray([0.1] * 4),
        "soft_low_ttc_exposure_fraction": np.asarray([0.02] * 4),
    }

    actions = merge_service_gate_actions(predictions, hard_brake_limit=0.2, low_ttc_limit=0.05)

    assert actions.tolist() == [1, 0, 0, 0]


def test_merge_service_gate_keeps_safety_as_hard_veto():
    predictions = {
        "merge_speed_gain_mps": np.asarray([0.2, 0.2]),
        "merge_discharge_gain": np.asarray([1.0, 1.0]),
        "soft_hard_brake_rate_vehicle_steps_per_s": np.asarray([0.21, 0.1]),
        "soft_low_ttc_exposure_fraction": np.asarray([0.02, 0.06]),
    }

    actions = merge_service_gate_actions(predictions, hard_brake_limit=0.2, low_ttc_limit=0.05)

    assert actions.tolist() == [0, 0]
