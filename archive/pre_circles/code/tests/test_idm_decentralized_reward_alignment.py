import pytest

from tools.diagnose_idm_decentralized_reward_alignment import (
    _metrics,
    _trajectory_metrics,
)


def test_safety_only_reward_tracks_frozen_effect_sign():
    records = [
        {
            "classification": "helpful",
            "effect": 0.2,
            "raw_component_deltas": {
                "efficiency": -50.0,
                "delay_budget": -20.0,
                "corridor_ttc": -1.0,
                "corridor_severe_braking": 0.1,
                "corridor_wave_energy": 0.2,
            },
        },
        {
            "classification": "harmful",
            "effect": -0.1,
            "raw_component_deltas": {
                "efficiency": 50.0,
                "delay_budget": 20.0,
                "corridor_ttc": 1.0,
                "corridor_severe_braking": -0.05,
                "corridor_wave_energy": -0.1,
            },
        },
    ]
    result = _metrics(
        records,
        {
            "efficiency": 0.0,
            "delay_budget": 0.0,
            "corridor_ttc": 0.0,
            "corridor_severe_braking": 1.0,
            "corridor_wave_energy": 1.0,
        },
    )
    assert result["nonneutral_sign_match_fraction"] == 1.0
    assert result["reward_effect_correlation"] == 1.0


def test_trajectory_metrics_exposes_agent_lifetime_return_mismatch():
    records = [
        {
            "classification": "helpful",
            "effect": 0.2,
            "trajectory_return_deltas": {"selected_role": -1.0},
        },
        {
            "classification": "harmful",
            "effect": -0.2,
            "trajectory_return_deltas": {"selected_role": 1.0},
        },
    ]
    result = _trajectory_metrics(records, "selected_role")
    assert result["nonneutral_sign_match_fraction"] == 0.0
    assert result["reward_effect_correlation"] == pytest.approx(-1.0)
