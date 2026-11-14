from config.constants import PRE_MERGE_EDGE, PRE_MERGE_LANE_LENGTH
from counterfactual.outcomes import OutcomeDefinition
from counterfactual.pilot import local_traffic_features
from counterfactual.twin_replay import StepSnapshot


def _vehicle(vehicle_id, lane_index, distance_to_merge, speed, acceleration=0.0):
    return (
        vehicle_id,
        PRE_MERGE_EDGE,
        f"{PRE_MERGE_EDGE}_{lane_index}",
        PRE_MERGE_LANE_LENGTH - distance_to_merge,
        speed,
        acceleration,
        0.0,
        0,
        "cav" if vehicle_id.startswith("cav") else "hdv",
    )


def test_local_features_capture_shock_and_soft_wall_neighbor_gaps():
    snapshot = StepSnapshot(
        relative_step=10,
        sim_time=5.0,
        action=0,
        supervisor_observation=tuple([0.0] * 16),
        vehicles=(
            _vehicle("hdv_lead", 0, 230.0, 7.0),
            _vehicle("cav0", 0, 250.0, 10.0),
            _vehicle("hdv_follow", 0, 265.0, 12.0, -5.0),
            _vehicle("cav1", 1, 300.0, 9.0),
            _vehicle("hdv_near", 1, 50.0, 4.0),
        ),
        active_cavs=("cav0", "cav1"),
        collision_vehicle_ids=(),
        starting_teleport_vehicle_ids=(),
        digest="test",
        vehicle_count=5,
        mean_speed=8.4,
        active_time_loss_sum=0.0,
    )
    features = local_traffic_features(
        snapshot,
        OutcomeDefinition(
            queue_speed_threshold_mps=10.0,
            hard_brake_threshold_mps2=-4.5,
        ),
    )

    assert features["local_bin_200_400m_vehicle_count"] == 4
    assert features["formation_soft_cav_count"] == 2
    assert features["formation_leader_gap_min_m"] == 15.0
    assert features["formation_follower_gap_min_m"] == 10.0
    assert features["decision_hard_brake_vehicle_count"] == 1
    assert features["local_downstream_minus_upstream_speed_mps"] < 0.0
