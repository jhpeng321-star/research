from tools.diagnose_idm_decentralized_decision_observability import _candidate_features


def test_candidate_features_are_predecision_and_include_recurrence():
    decision = {
        "decision_index": 3,
        "decision_step": 400,
        "pattern_context": {
            "idm_severe_level": 1.0,
            "idm_wave_energy_level": 2.0,
            "idm_severe_short_ema": 0.5,
            "idm_wave_short_ema": 0.6,
            "idm_severe_trend": 0.1,
            "idm_wave_energy_trend": 0.2,
        },
    }
    candidate = {
        "lane_id": "edge_1",
        "runway_m": 500.0,
        "distance_to_merge_m": 700.0,
        "disturbance_distance_to_merge_m": 200.0,
        "predicted_dose_s": 20.0,
        "preview_reference_speed_mps": 20.0,
        "prospective_rear_headroom_m": 10.0,
        "preview_nominal_acceleration_mps2": -0.2,
        "front_gap_m": 50.0,
        "front_secure_gap_m": 10.0,
        "rear_gap_m": 40.0,
        "rear_secure_gap_m": 12.0,
        "front_ttc_s": None,
        "rear_ttc_s": None,
    }
    observation = [0.0, 30.0 / 35.0]
    values = _candidate_features(decision, candidate, observation)
    assert len(values) == 22
    assert values[6] == 10.0
    assert values[14] == 3.0
    assert values[15] == 400.0
