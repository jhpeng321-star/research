"""v2.2 wall-session tests: merge_release semantics on WallGroupSession."""
from __future__ import annotations

import pytest

from traffic_shepherd import wall_group
from traffic_shepherd.wall_group import (
    MERGE_RELEASE_POINT_S,
    MERGE_START_S,
    REASON_ASSEMBLY_PAST_RELEASE,
    WallGroupSession,
)
from v22_fakes import ENTRY_LEN_M, MEMBERS, frame_at_s

WALL_ID = "w__w-mid-early:wall1"


def _session(wall_speed=17.0):
    return WallGroupSession(WALL_ID, members=frozenset(MEMBERS),
                            wall_speed_mps=wall_speed)


def _events(session, name):
    return [e for e in session.events if e.get("event") == name]


def test_law_version_is_v22_merge_release():
    assert wall_group.WALL_LAW_VERSION == "wall-v2.2-merge-release"


def _assemble(session, now=300.0):
    """One hold-entry step: both members at speed, 3 m apart, far
    upstream -> the session flips to HOLD."""
    frames = {
        MEMBERS[0]: frame_at_s(MEMBERS[0], 1000.0, lane_index=0,
                               speed=17.0, now=now),
        MEMBERS[1]: frame_at_s(MEMBERS[1], 997.0, lane_index=1,
                               speed=17.0, now=now)}
    commands = session.tick(frames, now)
    assert session.phase == wall_group.PHASE_HOLD
    assert not commands or all(abs(a) <= 1.5 for a in commands.values())
    return frames


def test_merge_release_fires_exactly_at_the_point_in_hold_only_once():
    session = _session()
    _assemble(session)
    # still upstream: no release
    frames = {m: frame_at_s(m, MERGE_RELEASE_POINT_S - 30.0,
                            lane_index=i, speed=17.0, now=301.0)
              for i, m in enumerate(MEMBERS)}
    session.tick(frames, 301.0)
    assert session.phase == wall_group.PHASE_HOLD
    assert not _events(session, "merge_release")
    # the LEAD member reaches the release point: exactly-once handback
    frames = {m: frame_at_s(m, MERGE_RELEASE_POINT_S + 8.0 - 3.0 * i,
                            lane_index=i, speed=17.0, now=302.0)
              for i, m in enumerate(MEMBERS)}
    commands = session.tick(frames, 302.0)
    assert commands == {}                     # handback: no wall commands
    assert session.phase == wall_group.PHASE_HANDBACK
    releases = _events(session, "merge_release")
    assert len(releases) == 1
    event = releases[0]
    assert event["release_kind"] == "merge_release"
    assert event["release_point_s"] == pytest.approx(MERGE_RELEASE_POINT_S)
    assert event["merge_start_s"] == pytest.approx(MERGE_START_S)
    assert event["handback"] == "native"
    # a later tick fires nothing further (once)
    session.tick(frames, 303.0)
    assert len(_events(session, "merge_release")) == 1


def test_merge_release_never_in_assembly_is_an_abort_instead():
    session = _session()
    # still ASSEMBLING (speeds not aligned) with the lead past the point
    frames = {
        MEMBERS[0]: frame_at_s(MEMBERS[0], MERGE_RELEASE_POINT_S + 5.0,
                               lane_index=0, speed=25.0),
        MEMBERS[1]: frame_at_s(MEMBERS[1], MERGE_RELEASE_POINT_S - 50.0,
                               lane_index=1, speed=17.0)}
    commands = session.tick(frames, 300.0)
    assert commands == {}
    assert session.phase == wall_group.PHASE_HANDBACK
    aborts = _events(session, "safety_abort")
    assert len(aborts) == 1
    assert aborts[0]["reason"] == REASON_ASSEMBLY_PAST_RELEASE
    assert not _events(session, "merge_release")


def test_merge_release_method_refused_outside_hold():
    session = _session()
    assert session.merge_release(300.0) is False      # assembly: refused
    assert session.phase == wall_group.PHASE_ASSEMBLY
    assert not session.events


def test_planner_release_before_the_merge_still_works():
    session = _session()
    _assemble(session)
    assert session.release(301.0) is True
    assert session.phase == wall_group.PHASE_HANDBACK
    assert len(_events(session, "released")) == 1
    assert not _events(session, "merge_release")
    # a merge release can no longer follow a handback
    assert session.merge_release(302.0) is False


def test_span_across_edges_on_s_road_no_false_abort():
    """HOLD with the front member 40 m INTO the merge edge and the rear
    still on the entry edge: the s_road span is 44 m (< the 150 m abort
    cap) -> the wall keeps pacing, NO false member_span_exceeded.  The
    old edge-blind coordinate read the merge-edge position as an
    entry-edge position and saw a phantom span."""
    session = _session()
    _assemble(session)
    frames = {
        MEMBERS[0]: frame_at_s(MEMBERS[0], ENTRY_LEN_M + 40.0, lane_index=0,
                               speed=17.0, now=301.0),
        MEMBERS[1]: frame_at_s(MEMBERS[1], ENTRY_LEN_M - 4.0, lane_index=1,
                               speed=17.0, now=301.0)}
    assert session._safety_abort(frames) is None
    positions, err = session._member_positions(frames)
    assert err is None
    span = max(positions.values()) - min(positions.values())
    assert span == pytest.approx(44.0)       # true cross-edge span on s_road
    commands = session.tick(frames, 301.0)
    # still HOLDING (the release point is only reached when the LEAD
    # member's s_road crosses 2644.2 - here it is 2684.2 -> this tick
    # fires the merge release; before that crossing no abort, no span
    # error: covered by test_merge_release_...), so assert the exact
    # semantics: lead past the point in HOLD -> merge_release, and the
    # span check itself never aborted.
    assert session.phase == wall_group.PHASE_HANDBACK
    assert len(_events(session, "merge_release")) == 1
    assert session.abort_reason is None
    assert commands == {}


def test_hold_crossing_the_merge_never_sees_member_left_network():
    """Synthetic hold across the whole merge zone: the session hands back
    AT the merge start; past the point every tick is inert (no commands,
    no aborts, no member_left_network - the v2.1 road-end behaviour)."""
    session = _session()
    _assemble(session)
    s = MERGE_RELEASE_POINT_S - 20.0
    fired_at = None
    for tick in range(300, 340):
        frames = {m: frame_at_s(m, s - 3.0 * i, lane_index=i, speed=17.0,
                                now=float(tick))
                  for i, m in enumerate(MEMBERS)}
        commands = session.tick(frames, float(tick))
        if session.phase == wall_group.PHASE_HANDBACK:
            if fired_at is None:
                fired_at = tick
                assert commands == {}
        else:
            assert set(commands) == set(MEMBERS)
        s += 17.0
        # continue far past the merge and the network half-way: the
        # session is already handed back and stays silent
        assert session.abort_reason is None
        assert not any(e.get("reason") == "member_left_network"
                       for e in session.events)
    assert fired_at is not None and fired_at <= 305
