import numpy as np

from idm.paired_option_dataset import OPTION_CONTEXT_NAMES, option_context_from_candidate


def test_option_context_is_fixed_causal_and_finite():
    candidate = {
        "lane_id": "edge_1",
        "predicted_dose_s": 18.0,
        "runway_m": 500.0,
        "distance_to_merge_m": 900.0,
        "disturbance_distance_to_merge_m": 250.0,
        "front_gap_m": 50.0,
        "front_secure_gap_m": 30.0,
        "rear_gap_m": 80.0,
        "rear_secure_gap_m": 40.0,
        "prospective_rear_headroom_m": 25.0,
        "front_ttc_s": None,
        "rear_ttc_s": 15.0,
        "preview_reference_speed_mps": 28.0,
        "preview_nominal_acceleration_mps2": -0.6,
    }
    values = option_context_from_candidate(candidate, 0.15)
    assert len(values) == len(OPTION_CONTEXT_NAMES) == 13
    assert np.all(np.isfinite(values))
    assert values[0] == 0.5
    assert values[1] == 0.9
    assert values[5] == 0.2
    assert values[8] == 2.0
    assert values[-1] == 0.5
