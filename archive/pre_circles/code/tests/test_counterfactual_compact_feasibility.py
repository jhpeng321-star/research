from config.constants import PRE_MERGE_EDGE, PRE_MERGE_LANE_LENGTH
from counterfactual.twin_replay import StepSnapshot
from experiments.counterfactual.audit_counterfactual_compact_feasibility import (
    audit_snapshot_scenario,
)


def _snapshot(rows, active):
    vehicles = tuple(
        (
            vehicle_id,
            PRE_MERGE_EDGE,
            f"{PRE_MERGE_EDGE}_{lane}",
            PRE_MERGE_LANE_LENGTH - distance,
            speed,
            0.0,
            0.0,
            0,
            "passenger",
            1.0,
        )
        for vehicle_id, lane, distance, speed in rows
    )
    return StepSnapshot(
        relative_step=120,
        sim_time=60.0,
        action=0,
        supervisor_observation=(0.0,),
        vehicles=vehicles,
        active_cavs=tuple(active),
        collision_vehicle_ids=(),
        starting_teleport_vehicle_ids=(),
        digest="decision",
        vehicle_count=len(vehicles),
        mean_speed=15.0,
        active_time_loss_sum=0.0,
    )


def _audit(snapshot, boundary):
    return audit_snapshot_scenario(
        snapshot,
        d_lo=200.0,
        d_hi=800.0,
        release_boundary_m=boundary,
        duration_s=30.0,
        span_cap_m=50.0,
        wall_speed_mps=10.0,
        a_max_dec_mps2=0.3,
    )


def test_release_boundary_changes_headroom_without_changing_recruitment():
    snapshot = _snapshot(
        [
            ("cav.0", 0, 500.0, 15.0),
            ("cav.1", 1, 520.0, 15.0),
        ],
        ["cav.0", "cav.1"],
    )

    assert _audit(snapshot, 100.0)["feasible"] is True
    assert _audit(snapshot, 200.0)["reason"] == "insufficient_kinematic_headroom"


def test_occupied_lane_without_cav_is_a_formation_veto():
    snapshot = _snapshot(
        [
            ("cav.0", 0, 500.0, 15.0),
            ("hdv.1", 1, 520.0, 15.0),
        ],
        ["cav.0"],
    )

    assert _audit(snapshot, 100.0)["reason"] == "uncovered_occupied_lane"


def test_trajectory_coherent_mode_prefers_speed_matched_pair():
    snapshot = _snapshot(
        [
            ("cav.fast", 0, 800.0, 30.0),
            ("cav.matched", 0, 760.0, 20.0),
            ("cav.lane1", 1, 790.0, 20.0),
        ],
        ["cav.fast", "cav.matched", "cav.lane1"],
    )
    common = {
        "d_lo": 200.0,
        "d_hi": 1000.0,
        "release_boundary_m": 50.0,
        "duration_s": 20.0,
        "span_cap_m": 50.0,
        "wall_speed_mps": 10.0,
        "a_max_dec_mps2": 0.3,
    }

    initial = audit_snapshot_scenario(snapshot, selection_mode="initial_compact", **common)
    coherent = audit_snapshot_scenario(snapshot, selection_mode="trajectory_coherent", **common)

    assert "cav.fast" in initial["selected_cav_ids"]
    assert set(coherent["selected_cav_ids"]) == {"cav.matched", "cav.lane1"}
    assert coherent["feasible"] is True
    assert coherent["predicted_max_span_m"] <= 50.0
