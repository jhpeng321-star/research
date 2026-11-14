"""Focused tests for the observation-repair screen v2 feature assembly."""

from __future__ import annotations

from tools.screen_idm_timing_observation_value_v2 import (
    _a_features,
    _observation_features,
)


def test_a_features_use_frozen_occurrence_semantics():
    occurrence = {
        "landmark": "wave_onset",
        "persistence_steps": 3,
        "runway_m": 500.0,
        "predicted_dose_s": 20.0,
        "lane_id": "1191554636_1",
    }
    vector = _a_features(occurrence, 2, 220)
    assert len(vector) == 8
    assert vector[0] == 2.0
    assert vector[1] == 1.0  # wave_onset
    assert vector[2] == 3.0
    assert vector[3] == 500.0
    assert vector[4] == 20.0
    assert vector[5] == 220.0
    assert vector[6] == 1.0 and vector[7] == 0.0  # lane_1 one-hot
    # severe landmark flag is zero for wave_onset (single flag in A: landmark_kind_wave)
    assert 4121 not in vector


def test_observation_family_vectors_follow_frozen_order():
    observation = {
        "features": {
            "B": {"ego_speed": 21.0, "ego_accel": -0.5, "leader_gap": 30.0, "leader_rel_speed": 1.0, "follower_gap": 40.0, "follower_closing": 0.5},
            "C": {"ramp_count": 2.0, "merge_zone_density": 20.0},
            "D": {"corridor_mean_speed": 25.0, "idm_wave_energy_level": 9.0},
        }
    }
    ab = _observation_features(observation, "AB")
    assert len(ab) == 14
    assert ab[0] == 21.0 and ab[5] == 0.5
    assert len(_observation_features(observation, "A")) == 0
    assert len(_observation_features(observation, "ABC")) == 24
    assert len(_observation_features(observation, "ABCD")) == 32
    # B fields outrank C/D in the frozen vector order
    assert ab[0] == 21.0  # ego_speed first
    abcd = _observation_features(observation, "ABCD")
    assert abcd[14 + 5] == 2.0  # C ramp_count (C starts with lane speeds)
    assert abcd[24] == 25.0  # D corridor_mean_speed


def test_missing_fields_default_to_zero():
    observation = {"features": {"B": {}, "C": {}, "D": {}}}
    assert _observation_features(observation, "ABCD") == [0.0] * 32
