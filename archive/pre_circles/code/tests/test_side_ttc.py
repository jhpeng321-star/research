"""Tests for side-conflict TTC (reward helper + eval metric) and the
vehicle-length fix in evaluation._compute_ttc."""

from unittest.mock import patch

from env.traci_query import merge_arrival_pet, side_ttc
from env.rewards import _safety
from evaluation import _compute_ttc, evaluate_side_ttc

MERGE = "1487606179"
MAINLINE = "1191554636"
POS_END = 75.44


def _states(ego, nbrs):
    """Patch _vehicle_state / vehicles_within_radius for side_ttc."""
    state_map = dict(nbrs)
    state_map["ego"] = ego
    return patch("env.traci_query._vehicle_state", side_effect=lambda v: state_map.get(v)), patch(
        "env.traci_query.vehicles_within_radius", return_value=[v for v, _ in nbrs]
    )


def test_side_ttc_converging_pair_fires():
    # ego on dropped lane 0 behind nbr on surviving lane 1: gap 7-5=2, closing 2 -> ttc 1.0
    p1, p2 = _states((MERGE, 0, 100.0, 10.0), [("v2", (MERGE, 1, 107.0, 8.0))])
    with p1, p2:
        assert side_ttc("ego") == 1.0


def test_side_ttc_parallel_lanes_silent():
    # lanes 1 and 2 both continue downstream -> not a lane-drop race
    p1, p2 = _states((MERGE, 1, 100.0, 10.0), [("v2", (MERGE, 2, 107.0, 8.0))])
    with p1, p2:
        assert side_ttc("ego") == float("inf")


def test_side_ttc_ignores_non_merge_edge():
    p1, p2 = _states((MAINLINE, 0, 100.0, 10.0), [("v2", (MAINLINE, 1, 107.0, 8.0))])
    with p1, p2:
        assert side_ttc("ego") == float("inf")


def test_side_ttc_overlap_is_zero():
    # longitudinally overlapped on converging lanes -> side collision in progress
    p1, p2 = _states((MERGE, 0, 100.0, 10.0), [("v2", (MERGE, 1, 103.0, 8.0))])
    with p1, p2:
        assert side_ttc("ego") == 0.0


def test_side_ttc_not_closing_is_silent():
    # ego behind but slower -> gap grows, no race
    p1, p2 = _states((MERGE, 0, 100.0, 5.0), [("v2", (MERGE, 1, 107.0, 9.0))])
    with p1, p2:
        assert side_ttc("ego") == float("inf")


def test_side_ttc_far_pair_skipped():
    p1, p2 = _states((MERGE, 0, 100.0, 10.0), [("v2", (MERGE, 1, 130.0, 8.0))])
    with p1, p2:
        assert side_ttc("ego") == float("inf")


def test_safety_includes_side_ttc():
    with (
        patch("env.rewards.traci") as mock_traci,
        patch("env.rewards._local_vehicles", return_value=[]),
        patch("env.rewards.leader_ttc", return_value=float("inf")),
        patch("env.rewards.side_ttc", return_value=0.5),
        patch("env.rewards.merge_arrival_pet", return_value=float("inf")),
    ):
        mock_traci.vehicle.getAcceleration.return_value = 0.0
        mock_traci.simulation.getCollidingVehiclesIDList.return_value = []
        assert _safety("cav1", 200.0) == -1.0  # low side-TTC only


def test_safety_includes_merge_pet():
    with (
        patch("env.rewards.traci") as mock_traci,
        patch("env.rewards._local_vehicles", return_value=[]),
        patch("env.rewards.leader_ttc", return_value=float("inf")),
        patch("env.rewards.side_ttc", return_value=float("inf")),
        patch("env.rewards.merge_arrival_pet", return_value=0.8),
    ):
        mock_traci.vehicle.getAcceleration.return_value = 0.0
        mock_traci.simulation.getCollidingVehiclesIDList.return_value = []
        assert _safety("cav1", 200.0) == -1.0  # low merge-arrival-PET only


def test_merge_pet_converging_arrival_fires():
    # ego on dropped lane 0 at pos 40 doing 10 -> ETA 3.54; nbr on surviving
    # lane 1 at pos 50 doing 8 -> ETA 3.18; PET = |3.54-3.18| = 0.36
    p1, p2 = _states((MERGE, 0, 40.0, 10.0), [("v2", (MERGE, 1, 50.0, 8.0))])
    with p1, p2:
        pet = merge_arrival_pet("ego", pos_end=POS_END)
        assert abs(pet - 0.36) < 0.01


def test_merge_pet_parallel_lanes_silent():
    # lanes 1 and 2 are both surviving/continuing -> no lane-drop race
    p1, p2 = _states((MERGE, 1, 40.0, 10.0), [("v2", (MERGE, 2, 50.0, 8.0))])
    with p1, p2:
        assert merge_arrival_pet("ego", pos_end=POS_END) == float("inf")


def test_merge_pet_ignores_non_merge_edge():
    p1, p2 = _states((MAINLINE, 0, 40.0, 10.0), [("v2", (MAINLINE, 1, 50.0, 8.0))])
    with p1, p2:
        assert merge_arrival_pet("ego", pos_end=POS_END) == float("inf")


def test_merge_pet_well_sequenced_is_large():
    # ego 70 m out at 10 (ETA 7.0), nbr 10 m out at 8 (ETA 1.25) -> PET 5.75
    p1, p2 = _states((MERGE, 0, 5.44, 10.0), [("v2", (MERGE, 1, 65.44, 8.0))])
    with p1, p2:
        pet = merge_arrival_pet("ego", pos_end=POS_END)
        assert pet > 5.0


def test_compute_ttc_subtracts_vehicle_length():
    follower = {"lane": "e_0", "pos": 100.0, "speed": 10.0}
    leader = {"lane": "e_0", "pos": 107.0, "speed": 8.0}
    # raw pos diff 7; bumper gap = 7 - 5 = 2; closing 2 -> 1.0 s
    # (pre-fix metric would have reported 3.5 s and missed the event)
    assert _compute_ttc(follower, leader) == 1.0


def test_compute_ttc_overlap_while_closing_is_zero():
    follower = {"lane": "e_0", "pos": 100.0, "speed": 10.0}
    leader = {"lane": "e_0", "pos": 103.0, "speed": 8.0}
    assert _compute_ttc(follower, leader) == 0.0


FCD_FIXTURE = """<?xml version="1.0"?>
<fcd-export>
  <timestep time="0.00">
    <vehicle id="a" x="0" y="0" speed="10" lane="1487606179_0" pos="100"/>
    <vehicle id="b" x="3" y="0" speed="8" lane="1487606179_1" pos="107"/>
    <vehicle id="c" x="6" y="0" speed="10" lane="1487606179_2" pos="100"/>
    <vehicle id="d" x="9" y="0" speed="5" lane="1191554636_0" pos="50"/>
  </timestep>
  <timestep time="1.00">
    <vehicle id="a" x="0" y="0" speed="8" lane="1487606179_0" pos="110"/>
    <vehicle id="b" x="3" y="0" speed="8" lane="1487606179_1" pos="115"/>
  </timestep>
</fcd-export>
"""


def test_evaluate_side_ttc_counts_converging_only(tmp_path):
    f = tmp_path / "fcd.xml"
    f.write_text(FCD_FIXTURE)
    out = evaluate_side_ttc(f)
    # t=0: pair (a,b) converging, gap 7-5=2, closing 2 -> ttc 1.0 < 1.5 -> 1 event.
    # pairs (a,c)/(b,c) not converging lanes; d not on merge edge.
    # t=1: gap 10-5=5, closing 0 -> no event.
    assert out["total_events"] == 1
    assert out["events_by_timestep"] == {0.0: 1}
