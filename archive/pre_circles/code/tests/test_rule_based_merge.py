"""Tests for rule-based merge baseline and TraCI control helpers."""

from unittest.mock import MagicMock, patch

import numpy as np

from policies.baselines import RuleBasedMergePolicy
from policies.traci_control import (
    LCA_LEFT,
    compute_a_req,
    find_merge_intent,
    reset_configured,
)


class _FakeEnv:
    simulation_step = 1.0


def test_compute_a_req():
    # 30 m/s -> 20 m/s over 100 m: a = (400 - 900) / 200 = -2.5
    a = compute_a_req(30.0, 20.0, 100.0)
    assert abs(a - (-2.5)) < 1e-6
    print("test_compute_a_req passed")


def test_merge_intent_from_lane_change_state():
    reset_configured()

    cav_id = "cav_0"
    other_id = "veh_ramp"

    mock_vehicle = MagicMock()
    mock_vehicle.getIDList.return_value = [cav_id, other_id]
    mock_vehicle.getRoadID.side_effect = lambda vid: "1487606179"
    mock_vehicle.getLaneIndex.side_effect = lambda vid: 1 if vid == cav_id else 0
    mock_vehicle.getLanePosition.side_effect = lambda vid: 40.0 if vid == cav_id else 35.0
    mock_vehicle.getSpeed.side_effect = lambda vid: 25.0
    mock_vehicle.getLaneID.side_effect = lambda vid: f"{vid}_lane"
    mock_vehicle.getLaneChangeState.side_effect = lambda vid: LCA_LEFT if vid == other_id else 0
    mock_vehicle.getBestLanes.return_value = []
    mock_vehicle.getLeftLeaders.return_value = []
    mock_vehicle.getRightLeaders.return_value = []
    mock_vehicle.getLeftFollowers.return_value = []
    mock_vehicle.getRightFollowers.return_value = []

    with patch("policies.traci_control.traci") as mock_traci:
        mock_traci.vehicle = mock_vehicle
        intent = find_merge_intent(cav_id, lateral_radius=50.0)

    assert intent is not None
    assert intent.vehicle_id == other_id
    assert intent.delta_x == 5.0
    assert intent.v_merging == 25.0
    print("test_merge_intent_from_lane_change_state passed")


def test_merge_intent_from_adjacent_leader():
    reset_configured()

    cav_id = "cav_0"
    other_id = "veh_adj"

    mock_vehicle = MagicMock()
    mock_vehicle.getIDList.return_value = [cav_id, other_id]
    mock_vehicle.getRoadID.side_effect = lambda vid: "edge_a"
    mock_vehicle.getLaneIndex.side_effect = lambda vid: 1 if vid == cav_id else 0
    mock_vehicle.getLanePosition.side_effect = lambda vid: 50.0
    mock_vehicle.getSpeed.side_effect = lambda vid: 22.0
    mock_vehicle.getLaneID.side_effect = lambda vid: "lane_1" if vid == cav_id else "lane_0"
    mock_vehicle.getLaneChangeState.return_value = 0
    mock_vehicle.getBestLanes.side_effect = lambda vid: (
        [("lane_1", 100.0, 0.0, 0)] if vid == other_id else []
    )
    mock_vehicle.getLeftLeaders.return_value = []
    mock_vehicle.getRightLeaders.return_value = [(other_id, 8.0, 22.0)]
    mock_vehicle.getLeftFollowers.return_value = []
    mock_vehicle.getRightFollowers.return_value = []

    with patch("policies.traci_control.traci") as mock_traci:
        mock_traci.vehicle = mock_vehicle
        intent = find_merge_intent(cav_id, lateral_radius=50.0)

    assert intent is not None
    assert intent.vehicle_id == other_id
    assert intent.delta_x == 8.0
    print("test_merge_intent_from_adjacent_leader passed")


def test_merge_yield_applies_kinematic_speed():
    reset_configured()

    policy = RuleBasedMergePolicy()
    policy.reset(_FakeEnv())

    cav_id = "cav_0"
    obs = {cav_id: np.zeros(24, dtype=np.float32)}

    from policies.traci_control import MergeIntent

    mock_vehicle = MagicMock()
    mock_vehicle.getIDList.return_value = [cav_id]
    mock_vehicle.getSpeed.return_value = 30.0

    with (
        patch("policies.baselines.traci") as mock_traci,
        patch(
            "policies.baselines.find_merge_intent",
            return_value=MergeIntent(vehicle_id="veh_ramp", delta_x=20.0, v_merging=20.0),
        ),
        patch("policies.baselines.apply_kinematic_speed") as mock_apply,
        patch("policies.baselines.configure_cav"),
    ):
        mock_traci.vehicle = mock_vehicle
        actions = policy.act(obs)

    assert actions == {}
    mock_apply.assert_called_once()
    print("test_merge_yield_applies_kinematic_speed passed")


def test_no_intent_releases_to_sumo():
    reset_configured()

    policy = RuleBasedMergePolicy()
    policy.reset(_FakeEnv())

    cav_id = "cav_0"
    obs = {cav_id: np.zeros(24, dtype=np.float32)}

    mock_vehicle = MagicMock()
    mock_vehicle.getIDList.return_value = [cav_id]
    mock_vehicle.getSpeed.return_value = 28.0

    with (
        patch("policies.baselines.traci") as mock_traci,
        patch("policies.baselines.find_merge_intent", return_value=None),
        patch("policies.baselines.release_to_sumo") as mock_release,
        patch("policies.baselines.configure_cav"),
    ):
        mock_traci.vehicle = mock_vehicle
        actions = policy.act(obs)

    assert actions == {}
    mock_release.assert_called_once_with(cav_id, tau=1.0)
    print("test_no_intent_releases_to_sumo passed")


def test_act_returns_empty_dict():
    reset_configured()

    policy = RuleBasedMergePolicy()
    policy.reset(_FakeEnv())

    with (
        patch("policies.baselines.traci") as mock_traci,
        patch("policies.baselines.configure_cav"),
        patch("policies.baselines.find_merge_intent", return_value=None),
        patch("policies.baselines.release_to_sumo"),
    ):
        mock_traci.vehicle.getIDList.return_value = ["cav_0"]
        mock_traci.vehicle.getSpeed.return_value = 25.0
        actions = policy.act({"cav_0": np.zeros(24)})

    assert actions == {}
    print("test_act_returns_empty_dict passed")


if __name__ == "__main__":
    test_compute_a_req()
    test_merge_intent_from_lane_change_state()
    test_merge_intent_from_adjacent_leader()
    test_merge_yield_applies_kinematic_speed()
    test_no_intent_releases_to_sumo()
    test_act_returns_empty_dict()
