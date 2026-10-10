"""v2.2 proposer tests: s_road per edge, D_min, hold cap, road_left."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from campaign_walls_v1 import wall_request
from campaign_walls_v1.common import (
    ASSEMBLY_TIME_ASSUMED_S,
    MERGE_RELEASE_POINT_S,
    MERGE_RELEASE_X_M,
    MIN_MEANINGFUL_HOLD_S,
    NETWORK_END_S,
)
from traffic_shepherd import wall_group
from v22_fakes import (
    ENTRY_LEN_M,
    MEMBERS,
    MERGE_START_S,
    T0,
    entry_frames,
    frame_at_s,
)


# ------------------------------------------------------- s_road per edge
def test_s_road_per_edge():
    entry = frame_at_s("a", 1000.0)
    merge = frame_at_s("a", ENTRY_LEN_M + 10.0)
    exit_ = frame_at_s("a", ENTRY_LEN_M + 75.4 + 300.0)
    assert wall_request.s_road(entry) == pytest.approx(1000.0)
    assert wall_request.s_road(merge) == pytest.approx(MERGE_START_S + 10.0)
    assert wall_request.s_road(exit_) == pytest.approx(
        MERGE_START_S + 75.4 + 300.0)


def test_s_road_exit_edge_not_read_as_entry_edge():
    """The regression class v2.2 kills: 100 m into the exit edge is
    s_road 2819.6, NOT ~2544 (the entry-edge assumption) — the merge
    release point is reachable only through the edge-aware coordinate."""
    frame = frame_at_s("a", ENTRY_LEN_M + 75.4 + 100.0)
    assert wall_request.s_road(frame) == pytest.approx(2819.6)
    assert wall_request.s_road(frame) > MERGE_RELEASE_POINT_S


def test_s_road_session_helper_agrees():
    frame = frame_at_s("a", ENTRY_LEN_M + 20.0)
    assert wall_group.frame_s_road(frame) == pytest.approx(
        wall_request.s_road(frame))


# ------------------------------------------------------------ D_min rule
def test_dmin_formula_and_values():
    # D_min = X + v_wall * (t_assemble + H_min), X = 0, t = 10, H = 60
    assert MERGE_RELEASE_X_M == 0.0
    assert ASSEMBLY_TIME_ASSUMED_S == 10.0
    assert MIN_MEANINGFUL_HOLD_S == 60.0
    assert wall_request.min_proposal_distance_m(15.0) == pytest.approx(1050.0)
    assert wall_request.min_proposal_distance_m(10.0) == pytest.approx(700.0)
    with pytest.raises(wall_request.FeasibilityError):
        wall_request.min_proposal_distance_m(0.0)


def test_pair_far_upstream_is_feasible_under_dmin():
    # lead 1500 m upstream, v_wall = 0.85 x 20 = 17 -> D_min = 1190 m
    pair = wall_request.choose_pair(entry_frames(1500.0), 0.85)
    assert pair is not None
    assert pair["lead_distance_to_merge_m"] == pytest.approx(1500.0)
    assert pair["dmin_m"] == pytest.approx(17.0 * 70.0)
    assert pair["lead_distance_to_merge_m"] >= pair["dmin_m"]


def test_pair_that_cannot_assemble_and_hold_before_merge_refused():
    # lead 500 m upstream, v_wall 17 -> D_min 1190 m: refused AT PROPOSAL
    frames = entry_frames(500.0)
    assert wall_request.choose_pair(frames, 0.85) is None
    refusal = wall_request.dmin_refusal(frames, 0.85)
    assert refusal is not None
    assert refusal["reason"] == "dmin_unreachable"
    assert refusal["lead_distance_to_merge_m"] == pytest.approx(500.0)
    assert refusal["dmin_m"] == pytest.approx(1190.0)


def test_no_pair_at_all_is_not_a_dmin_refusal():
    # both members in the SAME lane: no lane-distinct pair exists
    frames = entry_frames(1500.0)
    frames[MEMBERS[1]] = frame_at_s(MEMBERS[1], ENTRY_LEN_M - 1502.0,
                                    lane_index=0, speed=20.0)
    assert wall_request.choose_pair(frames, 0.85) is None
    assert wall_request.dmin_refusal(frames, 0.85) is None


def test_dmin_gate_supersedes_the_v21_road_left_rule():
    """The v2.1 A4 rule (road_left >= 80 v) would ADMIT a pair whose lead
    is 1000 m from the merge at v_wall 17 (road_left 1553 >= 1360) but
    which cannot assemble + hold 60 s before the merge (D_min 1190)."""
    frames = entry_frames(1000.0)
    assert wall_request.choose_pair(frames, 0.85) is None  # D_min binds


# ------------------------------------------------------------- hold cap
def test_hold_cap_capped_at_the_merge_start():
    # 240 s draw, lead 1000 m upstream at v 20 -> 50 s to the release pt
    assert wall_request.hold_cap_s(240.0, MERGE_START_S - 1000.0, 20.0) \
        == pytest.approx(50.0)
    # far upstream: the draw is NOT capped
    assert wall_request.hold_cap_s(240.0, MERGE_START_S - 20000.0, 20.0) \
        == pytest.approx(240.0)
    # already at/past the point: zero
    assert wall_request.hold_cap_s(240.0, MERGE_RELEASE_POINT_S, 20.0) == 0.0
    with pytest.raises(wall_request.FeasibilityError):
        wall_request.hold_cap_s(60.0, 0.0, 0.0)


# ------------------------------------------------------------ road_left
def test_road_left_edge_aware_supersedes_entry_edge_assumption():
    frame = frame_at_s("a", ENTRY_LEN_M + 75.4 + 100.0)   # exit edge
    assert wall_request.road_left_m(frame) == pytest.approx(
        NETWORK_END_S - (ENTRY_LEN_M + 75.4 + 100.0))
    assert wall_request.road_left_m(frame_at_s("a", 100.0)) == pytest.approx(
        NETWORK_END_S - 100.0)
    assert wall_request.road_left_m(SimpleNamespace(lane_id=None)) is None


# ---------------------------------------------------------- pair shapes
def test_pair_requires_two_lanes_and_span_cap():
    frames = entry_frames(1500.0)
    frames[MEMBERS[1]] = frame_at_s(MEMBERS[1], ENTRY_LEN_M - 1502.0,
                                    lane_index=0, speed=20.0)
    assert wall_request.choose_pair(frames, 0.85) is None
    wide = entry_frames(1500.0, span=80.0)
    assert wall_request.choose_pair(wide, 0.85) is None      # > 75 m cap
    ok = entry_frames(1500.0, span=74.0)
    assert wall_request.choose_pair(ok, 0.85)["span_m"] == pytest.approx(74.0)


def test_build_wall_plan_records_the_drawn_hold_not_a_capped_one():
    frames = entry_frames(1500.0)
    pair = wall_request.choose_pair(frames, 0.85)
    plans = []
    plan, wall_id, speed = wall_request.build_wall_plan(
        run_name="w__w-mid-early", cav_ids=("cav.0", "cav.1", "cav.2",
                                            "cav.3"),
        pair=pair, factor=0.85, hold_s=240.0, now_s=T0,
        plan_factory=lambda **kw: plans.append(kw)
        or SimpleNamespace(**kw))
    assert wall_id == "w__w-mid-early:wall1"
    assert speed == pytest.approx(17.0)
    wall = plan.walls[0]
    assert wall.hold_s == 240.0          # AS DRAWN; the cap is at run time
    assert plan.assignments[MEMBERS[0]] == "wall-v2"
    assert plan.assignments["cav.0"] == "native"
