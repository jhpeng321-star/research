from idm.action_local import (
    LandmarkContract,
    classify_effect,
    collapse_condition,
    landmark_signal,
    replay_deadband,
    safety_composite,
)


def test_landmarks_use_frozen_level_and_trend_thresholds():
    contract = LandmarkContract()
    context = {
        "idm_wave_energy_level": 1.0,
        "idm_wave_energy_trend": 0.1,
        "idm_severe_level": 0.99,
        "idm_severe_trend": 0.5,
    }
    assert landmark_signal("wave_onset", context, contract)
    assert not landmark_signal("severe_onset", context, contract)


def test_collapse_condition_matches_frozen_step_definition():
    contract = LandmarkContract()
    assert collapse_condition([10.0, 11.0, 12.0, 13.0, 14.0], contract)
    assert not collapse_condition([10.0, 11.0, 12.0, 13.0, 16.0], contract)


def test_replay_deadband_is_symmetric_and_noise_aware():
    assert replay_deadband(0.02, [0.0, 0.001]) == 0.02
    assert replay_deadband(0.02, [0.0, 0.03]) == 0.06


def test_effect_classification_keeps_neutral_distinct_from_harmful():
    assert classify_effect(0.021, deadband=0.02, constraints_pass=True) == "helpful"
    assert classify_effect(-0.02, deadband=0.02, constraints_pass=True) == "neutral"
    assert classify_effect(-0.021, deadband=0.02, constraints_pass=True) == "harmful"
    assert classify_effect(0.5, deadband=0.02, constraints_pass=False) == "harmful"


def test_safety_composite_preserves_separate_physical_effects():
    result = safety_composite(
        zero_severe=10,
        action_severe=8,
        severe_reference=0.5,
        zero_wave=100,
        action_wave=80,
        wave_reference=5,
        horizon_steps=20,
    )
    assert result["severe_braking_effect_reference_units"] == 0.2
    assert result["wave_effect_reference_units"] == 0.2
    assert result["safety_composite_reference_units"] == 0.2
