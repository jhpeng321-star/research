"""Unit tests for Selfish CAV features and encoder."""

import numpy as np
import torch

from env.features import (
    RAW_OBS_DIM,
    RSU_CONTRACT_LEGACY_CONFIGURED_RANGE_V1,
    RSU_CONTRACT_LIVE_2000_DIAGNOSTIC_V1,
    VEH_SIGNAL_BLINKER_LEFT,
    VEH_SIGNAL_BLINKER_RIGHT,
    NeighborFeatures,
    TrustTracker,
    padded_zone_features,
    raw_obs_dim,
    resolve_rsu_communication_range,
    rsu_observation_metadata,
)
from unittest.mock import patch
from models import FeatureEncoder, SelfishPPOModel


def test_padded_zone_features():
    z = padded_zone_features(35.0)
    assert z.shape == (12,)
    assert z[0] == 0.0 and z[2] == 0.0 and z[4] == 0.0
    assert z[1] == 1.0 and z[3] == 1.0 and z[5] == 1.0  # normalized free flow
    assert z[6] == 0.0
    assert z[7] == 1.0  # merge_outflow neutral/healthy
    assert np.allclose(z[8:], 0.0)  # delay slots neutral when padded


def test_versioned_rsu_contract_keeps_legacy_range_and_freezes_diagnostic_range():
    assert resolve_rsu_communication_range(
        RSU_CONTRACT_LEGACY_CONFIGURED_RANGE_V1, 500.0
    ) == 500.0
    assert resolve_rsu_communication_range(
        RSU_CONTRACT_LEGACY_CONFIGURED_RANGE_V1, 750.0
    ) == 750.0
    assert resolve_rsu_communication_range(
        RSU_CONTRACT_LIVE_2000_DIAGNOSTIC_V1, 2000.0
    ) == 2000.0
    with np.testing.assert_raises(ValueError):
        resolve_rsu_communication_range(
            RSU_CONTRACT_LIVE_2000_DIAGNOSTIC_V1, 500.0
        )


def test_rsu_metadata_records_live_and_padded_zone_blocks():
    with patch("env.features.distance_to_merge", return_value=1600.0):
        legacy = rsu_observation_metadata(
            "cav",
            contract=RSU_CONTRACT_LEGACY_CONFIGURED_RANGE_V1,
            communication_range_m=500.0,
            observation_timestamp_s=42.0,
        )
        diagnostic = rsu_observation_metadata(
            "cav",
            contract=RSU_CONTRACT_LIVE_2000_DIAGNOSTIC_V1,
            communication_range_m=2000.0,
            observation_timestamp_s=42.0,
        )
    assert legacy["rsu_available"] is False
    assert legacy["observation_age_s"] is None
    assert set(legacy["zone_blocks"].values()) == {"padded", "live"}
    assert diagnostic["rsu_available"] is True
    assert diagnostic["observation_age_s"] == 0.0
    assert set(diagnostic["zone_blocks"].values()) == {"live"}


def test_delay_zone_slots():
    from env.features import delay_zone_slots

    # Missing refs → excesses 0; rates still normalize
    z = delay_zone_slots(ema_main_delay=0.25, ema_ramp_delay=0.10, zone_delay_abs_scale=0.50)
    assert z.shape == (4,)
    assert abs(z[0]) < 1e-9 and abs(z[1]) < 1e-9
    assert abs(z[2] - 0.5) < 1e-6 and abs(z[3] - 0.2) < 1e-6

    # Beating delay ref → negative signed excess
    z2 = delay_zone_slots(
        corridor_delay_rate=0.20,
        delay_rate_ref=0.30,
        delay_budget_scale=0.20,
    )
    assert abs(z2[0] - (-0.5)) < 1e-6

    # Fair gap excess only above ref
    z3 = delay_zone_slots(
        stream_delay_gap=0.30,
        stream_delay_gap_ref=0.10,
        stream_fairness_gap_scale=0.40,
    )
    assert abs(z3[1] - 0.5) < 1e-6  # excess 0.20 / 0.40


def test_turn_signal_encoding_bits():
    # Pure unit test of bit logic without TraCI: mirror the encoding.
    def encode(signals: int) -> float:
        left = bool(signals & VEH_SIGNAL_BLINKER_LEFT)
        right = bool(signals & VEH_SIGNAL_BLINKER_RIGHT)
        if left and not right:
            return -1.0
        if right and not left:
            return 1.0
        return 0.0

    assert encode(VEH_SIGNAL_BLINKER_LEFT) == -1.0
    assert encode(VEH_SIGNAL_BLINKER_RIGHT) == 1.0
    assert encode(0) == 0.0
    assert encode(VEH_SIGNAL_BLINKER_LEFT | VEH_SIGNAL_BLINKER_RIGHT) == 0.0


def test_trust_ema_and_weights():
    tracker = TrustTracker(lambda_ema=0.5)
    nbr = NeighborFeatures(
        vehicle_id="v1",
        rel_distance=40.0,
        rel_speed=0.0,
        rel_lane=0.0,
        turn_signal=0.0,
        abs_distance=40.0,
    )
    t0 = tracker.update_pair("ego", nbr, other_is_cav=False, vtype="normal_car")
    assert 0.0 <= t0 <= 1.0
    t1 = tracker.update_pair("ego", nbr, other_is_cav=False, vtype="normal_car")
    assert 0.0 <= t1 <= 1.0

    weights = tracker.update_and_weights("ego", [nbr], cav_ids=set())
    assert abs(sum(weights.values()) - 1.0) < 1e-6

    empty = tracker.update_and_weights("ego", [], cav_ids=set())
    assert empty == {}


def test_encoder_output_shape():
    dim = raw_obs_dim(20)
    assert dim == RAW_OBS_DIM
    enc = FeatureEncoder(max_neighbors=20)
    x = torch.zeros(4, dim)
    y = enc(x)
    assert y.shape == (4, 128)


def test_local_ppo_model_forward():
    import gymnasium as gym

    dim = raw_obs_dim(20)
    model = SelfishPPOModel(
        obs_space=gym.spaces.Box(-np.inf, np.inf, (dim,), np.float32),
        action_space=gym.spaces.Box(-3.0, 3.0, (1,), np.float32),
        # RLlib's Gaussian Box distribution requests mean + log std.
        num_outputs=2,
        model_config={"custom_model_config": {"max_neighbors": 20, "hidden": 64}},
        name="selfish_local_ppo_test",
    )
    logits, _ = model({"obs": torch.randn(3, dim)}, [], None)
    assert logits.shape == (3, 2)
    assert model.value_function().shape == (3,)


if __name__ == "__main__":
    test_padded_zone_features()
    test_turn_signal_encoding_bits()
    test_trust_ema_and_weights()
    test_encoder_output_shape()
    test_local_ppo_model_forward()
    print("all unit tests passed")
