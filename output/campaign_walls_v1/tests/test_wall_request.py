"""Wall REQUEST rule: feasibility, member choice, speed, timing."""
from __future__ import annotations

import pytest

from campaign_walls_v1 import wall_request
from campaign_walls_v1.common import (
    REQUEST_WINDOW_S,
    SPEED_FLOOR_MPS,
    TIMING_OFFSET_S,
)
from wall_fakes import EDGE, MEMBERS, T0, feasible_frames, make_frame


def test_feasible_pair_same_two_lane_edge_distinct_lanes():
    pairs = wall_request.feasible_pairs(feasible_frames())
    assert len(pairs) == 1
    pair = pairs[0]
    assert set(pair["members"]) == set(MEMBERS)
    assert pair["lanes"] == [0, 1]              # one member per lane
    assert pair["edge"] == EDGE                 # a registered 2-lane edge
    assert pair["span_m"] == pytest.approx(40.0)


def test_other_edges_and_single_lane_edges_are_infeasible():
    frames = feasible_frames()
    # same geometry but on an edge that is NOT a 2-lane mainline edge
    other = {v: make_frame(v, 20.0, 1 if i else 0, 120.0 - 40 * i,
                           f"888_notmain_{i}", T0)
             for i, v in enumerate(MEMBERS)}
    assert wall_request.feasible_pairs(other) == []
    # same edge but BOTH members in lane 0 -> never a cross-section
    frames[MEMBERS[1]] = make_frame(MEMBERS[1], 20.0, 0, 100.0,
                                    f"{EDGE}_0", T0)
    assert wall_request.feasible_pairs(frames) == []


def test_span_cap_75_m():
    frames = feasible_frames()
    frames[MEMBERS[1]] = make_frame(MEMBERS[1], 20.0, 1, 120.0 - 75.0,
                                    f"{EDGE}_1", T0)
    assert len(wall_request.feasible_pairs(frames)) == 1   # span exactly 75
    frames[MEMBERS[1]] = make_frame(MEMBERS[1], 20.0, 1, 120.0 - 75.5,
                                    f"{EDGE}_1", T0)
    assert wall_request.feasible_pairs(frames) == []       # 75.5 > cap


def test_member_choice_nearest_the_merge():
    frames = feasible_frames()
    # a SECOND feasible pair further upstream (mean 200 m vs 100 m)
    frames["cav.2"] = make_frame("cav.2", 20.0, 0, 220.0, f"{EDGE}_0", T0)
    frames["cav.0"] = make_frame("cav.0", 20.0, 1, 180.0, f"{EDGE}_1", T0)
    pair = wall_request.choose_pair(frames)
    assert set(pair["members"]) == set(MEMBERS)   # the nearer pair wins
    assert pair["mean_distance_to_merge_m"] == pytest.approx(100.0)


def test_frames_without_report_coordinates_are_never_candidates():
    import dataclasses

    frames = feasible_frames()
    frames["cav.2"] = dataclasses.replace(
        make_frame("cav.2", 20.0, 0, 130.0, f"{EDGE}_0", T0),
        distance_to_merge_m=None)    # no coordinate: never a candidate
    pair = wall_request.choose_pair(frames)
    assert set(pair["members"]) == set(MEMBERS)


def test_speed_factor_and_floor():
    frames = feasible_frames()                    # both members at 20 m/s
    pair = wall_request.choose_pair(frames)
    assert wall_request.wall_speed_for(0.60, pair["member_speeds"]) == 12.0
    assert wall_request.wall_speed_for(0.85, pair["member_speeds"]) == 17.0
    # the floor: slow members (3 m/s -> 0.6x = 1.8) still request 2.5 m/s,
    # always strictly above wall_group.SAFE_SPEED_MIN_MPS = 2.0
    slow = {v: 3.0 for v in pair["member_speeds"]}
    assert wall_request.wall_speed_for(0.60, slow) == SPEED_FLOOR_MPS


def test_early_and_late_target_and_deadline():
    for timing, offset in (("early", 0.0), ("late", 60.0)):
        target, deadline = wall_request.target_and_deadline(T0, timing)
        assert target == pytest.approx(T0 + TIMING_OFFSET_S[timing])
        assert deadline == pytest.approx(target + REQUEST_WINDOW_S)
    with pytest.raises(wall_request.FeasibilityError):
        wall_request.target_and_deadline(T0, "whenever")


def test_not_feasible_only_after_the_full_window():
    from campaign_walls_v1.wall_driver import _WallRequest

    empty: dict = {}
    logged = []
    request = _WallRequest(run_name="w__w-lo-early", arm="W-lo-early",
                           t0_s=T0, hold_s=60.0, cav_ids=("a", "b"))
    # a feasible tick at the target fires the request immediately
    class _Session:
        def apply_plan(self, plan, *, now_s):
            pass

    session = _Session()
    request.maybe_request(session, empty, T0, logged.append)
    assert request.state == "waiting" and not logged
    request.maybe_request(session, empty, T0 + 60.0, logged.append)
    assert request.state == "waiting" and not logged          # inside window
    request.maybe_request(session, empty, T0 + 120.0 + 0.5, logged.append)
    assert request.state == "not_feasible"
    assert logged[0]["kind"] == "not_feasible"
    assert logged[0]["deadline_s"] == pytest.approx(T0 + 120.0)


def test_request_row_carries_the_declared_fields():
    from campaign_walls_v1.wall_driver import _WallRequest

    logged = []

    class _Session:
        def apply_plan(self, plan, *, now_s):
            self.plan = plan

    session = _Session()
    request = _WallRequest(run_name="w__w-mid-early", arm="W-mid-early",
                           t0_s=T0, hold_s=120.0, cav_ids=("cav.0", *MEMBERS))
    request.maybe_request(session, feasible_frames(), T0, logged.append)
    assert request.state == "requested"
    row = logged[0]
    assert row["kind"] == "wall_request"
    assert row["source"] == "cav_report_frames"
    assert set(row["members"]) == set(MEMBERS)
    assert row["lanes"] == [0, 1]
    assert row["span_m"] == pytest.approx(40.0)
    assert row["speed_factor"] == 0.85
    assert row["wall_speed_mps"] == pytest.approx(17.0)
    assert row["timestamp_s"] == pytest.approx(T0)          # early: at t0
    # the applied plan: members wall-v2, EVERY roster CAV named, hold set
    assert set(session.plan.assignments) == {"cav.0", *MEMBERS}
    assert {v for v, a in session.plan.assignments.items() if a == "wall-v2"} \
        == set(MEMBERS)
    assert session.plan.assignments["cav.0"] == "native"    # non-member native
    assert session.plan.strict_roster is True
    wall = session.plan.walls[0]
    assert wall.hold_s == 120.0 and wall.phase == "assemble"
