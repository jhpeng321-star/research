"""v2.2 DRIVER end-to-end on the REAL frozen dispatcher (fake api)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from campaign_walls_v1 import wall_driver
from v22_fakes import (
    CAVS,
    ENTRY_LEN_M,
    MEMBERS,
    T0,
    FakeApi,
    FakeSensor,
    entry_frames,
    frame_at_s,
    real_session,
    restore_entry,
    wall_arm,
)


def _world(duration_s: float):
    return {"world": "w300p10", "seed": 300, "branch_length_s": duration_s}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _drive(world, arm, frames, tmp_path, *, cav_ids=CAVS, live=MEMBERS):
    api = FakeApi(t0=T0, live=live)
    session, owner = real_session(cav_ids)
    sensor = FakeSensor(frames)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary = wall_driver.drive_wall_branch(
        api, world, arm, run_dir, restore_entry(cav_ids), session=session,
        ownership=owner, sensor=sensor)
    return summary, run_dir, session


def _frame(vid, s_road, lane_index, speed, now):
    return frame_at_s(vid, s_road, lane_index=lane_index, speed=speed,
                      now=now)


def _merge_crossing_frames(t0, end, *, wall_speed=17.0, lead_start_s=1000.0):
    """Pair feasible at t0 (lead 1644.2 m upstream of the merge start >
    D_min 1190 m), assembled two ticks later at ``lead_start_s``, then
    cruising at ``wall_speed`` THROUGH the merge start."""
    frames = {}
    lead_s = float(lead_start_s)
    for t in range(int(t0), int(end)):
        now = float(t)
        if t == int(t0):
            pair = entry_frames(ENTRY_LEN_M - lead_s,
                                speeds=(20.0, 20.0), now=now)
        elif t == int(t0) + 1:
            pair = entry_frames(ENTRY_LEN_M - (lead_s + 20.0),
                                speeds=(18.5, 18.5), now=now)
        else:
            lead_s += float(wall_speed)
            pair = {
                MEMBERS[0]: _frame(MEMBERS[0], lead_s, 0, wall_speed, now),
                MEMBERS[1]: _frame(MEMBERS[1], lead_s - 3.0, 1,
                                   wall_speed, now)}
        frames[now] = pair
    return frames


def test_hold_crossing_merge_released_exactly_once_at_the_merge_start(
        tmp_path):
    """The owner binding: a planner-held wall (huge draw) is released at
    the merge start AT THE LATEST - via ``merge_release`` (a release
    kind, never an abort), exactly once, at s_road >= 2644.2."""
    hold_s = 600.0                                   # never expires first
    arm = wall_arm(hold_s=hold_s)
    frames = _merge_crossing_frames(T0, T0 + 120, lead_start_s=1000.0)
    summary, run_dir, session = _drive(_world(120.0), arm, frames, tmp_path)
    assert summary["wall_outcome"] == "merge_released"
    events = _jsonl(run_dir / "wall_events.jsonl")
    releases = [e for e in events
                if e.get("kind") == "wall_event"
                and e.get("event") == "merge_release"]
    assert len(releases) == 1                        # exactly once
    release = releases[0]
    assert release["release_kind"] == "merge_release"
    assert release["release_point_s"] == pytest.approx(2644.2)
    # tick-quantized crossing: the scripted lead is s = 1000 + 17*(t-T0-1),
    # so the first hold tick with s >= 2644.2 is t = T0 + 98
    assert release["timestamp_s"] == pytest.approx(T0 + 98.0)
    # fired in HOLD only (assembled before it), before any hold expiry
    assert release["timestamp_s"] < T0 + hold_s
    # the cap row: requested 600 s draw, effective ~38 s -> capped
    caps = [e for e in events if e["kind"] == "hold_cap"]
    assert len(caps) == 1
    cap = caps[0]
    assert cap["requested_hold_s"] == pytest.approx(hold_s)
    assert cap["capped_at_merge"] is True
    assert cap["cap_point_s"] == pytest.approx(2644.2)
    assert cap["effective_hold_s"] == pytest.approx(
        release["timestamp_s"] - (T0 + 2.0))
    assert cap["lead_s_road_at_release_m"] >= 2644.2 - 1e-6
    assert cap["lead_s_road_at_release_m"] < 2644.2 + 17.0 + 1e-6
    # outcome + record fields
    record = json.loads(
        (run_dir / "branch_record.json").read_text(encoding="utf-8"))
    assert record["outcome"] == "merge_released"
    assert record["merge_release_s"] == pytest.approx(release["timestamp_s"])
    assert record["hold_capped_at_merge"] is True
    assert record["released_at_s"] == pytest.approx(release["timestamp_s"])
    assert record["aborted_at_s"] is None
    handback = [e for e in events if e["kind"] == "wall_handback"][0]
    assert handback["how"] == "merge_release"
    assert handback["release_kind"] == "merge_release"
    assert handback["capped_at_merge"] is True
    assert handback["handback"] == "native"
    # NO member_left_network anywhere; members never reach the merge
    # edge while held past the release (the release IS the merge start)
    assert not any(e.get("reason") == "member_left_network"
                   for e in events)
    rows = _jsonl(run_dir / "dispatcher_rows.jsonl")
    member_rows = [r for r in rows if r["vehicle_id"] == MEMBERS[0]]
    locked = [r for r in member_rows if r["locked"]]
    assert all(r["wall_phase"] in ("assembly", "hold") for r in locked)
    after = [r for r in member_rows
             if r["timestamp_s"] > release["timestamp_s"] + 1e-9]
    assert after and all(r["action"] == "native" and not r["locked"]
                         for r in after)


def test_hold_cap_a_priori_effective_hold_computed_at_assembly(tmp_path):
    """v2.2 logging contract (smoke R2 fix): the cap row's
    ``a_priori_effective_hold_s`` is ``min(H, (2644.2 - lead s_road AT
    ASSEMBLY) / v_wall)`` - the ASSEMBLY position recorded on the
    ``assembled`` tick, never the release position."""
    hold_s = 600.0
    frames = _merge_crossing_frames(T0, T0 + 120, lead_start_s=1000.0)
    summary, run_dir, _session = _drive(_world(120.0), wall_arm(hold_s=hold_s),
                                        frames, tmp_path)
    assert summary["wall_outcome"] == "merge_released"
    cap = [e for e in _jsonl(run_dir / "wall_events.jsonl")
           if e["kind"] == "hold_cap"][0]
    # assembly completed at T0+2 with the lead at s = 1000 + 17 = 1017
    # (the helper's first hold tick increments the lead by one wall speed)
    assert cap["lead_s_road_at_assembly_m"] == pytest.approx(1017.0)
    assert cap["a_priori_effective_hold_s"] == pytest.approx(
        min(hold_s, (2644.2 - 1017.0) / 17.0))
    # NOT the release position: the lead crossed 2644.2 before the
    # release tick, so a release-time "a priori" number would be ~0
    assert cap["lead_s_road_at_release_m"] > cap["lead_s_road_at_assembly_m"]
    assert cap["a_priori_effective_hold_s"] > 90.0
    assert (2644.2 - cap["lead_s_road_at_release_m"]) / 17.0 < 1.0


def test_fixed_arm_hold_expiry_before_merge_releases_with_cap_logged(
        tmp_path):
    """The registered draw releases first (dispatcher hold term): outcome
    assembled_released, and the cap row logs capped_at_merge=False."""
    hold_s = 6.0
    frames = _merge_crossing_frames(T0, T0 + 20, lead_start_s=1000.0)
    summary, run_dir, _session = _drive(_world(20.0),
                                        wall_arm(hold_s=hold_s), frames,
                                        tmp_path)
    assert summary["wall_outcome"] == "assembled_released"
    events = _jsonl(run_dir / "wall_events.jsonl")
    assert not [e for e in events
                if e.get("event") == "merge_release"]
    released = [e for e in events
                if e.get("kind") == "wall_event"
                and e.get("event") == "released"][0]
    assert released["timestamp_s"] == pytest.approx(T0 + 2.0 + hold_s)
    caps = [e for e in events if e["kind"] == "hold_cap"]
    assert len(caps) == 1
    assert caps[0]["capped_at_merge"] is False
    assert caps[0]["requested_hold_s"] == pytest.approx(hold_s)
    assert caps[0]["effective_hold_s"] == pytest.approx(hold_s)
    record = json.loads(
        (run_dir / "branch_record.json").read_text(encoding="utf-8"))
    assert record["outcome"] == "assembled_released"
    assert record["merge_release_s"] is None
    assert record["hold_capped_at_merge"] is False


def test_dmin_refusal_at_proposal_logged_with_reason(tmp_path):
    """A pair that cannot assemble + hold before the merge is refused at
    the proposal: the branch runs native and the not_feasible row names
    dmin_unreachable with the numbers."""
    # lead 500 m upstream of the merge; v_wall = 17 -> D_min 1190 m
    frames = {}
    for t in range(int(T0), int(T0) + 130):
        frames[float(t)] = entry_frames(500.0, now=float(t))
    summary, run_dir, _session = _drive(
        _world(130.0), wall_arm(hold_s=6.0), frames, tmp_path)
    assert summary["wall_outcome"] == "not_feasible"
    events = _jsonl(run_dir / "wall_events.jsonl")
    not_feasible = [e for e in events if e["kind"] == "not_feasible"][0]
    assert not_feasible["reason"] == "dmin_unreachable"
    assert not_feasible["dmin"]["dmin_m"] == pytest.approx(1190.0)
    assert not_feasible["dmin"]["lead_distance_to_merge_m"] == \
        pytest.approx(500.0)
    record = json.loads(
        (run_dir / "branch_record.json").read_text(encoding="utf-8"))
    assert record["outcome"] == "not_feasible"
    assert record["members"] is None


def test_request_row_carries_the_v22_dmin_fields(tmp_path):
    frames = _merge_crossing_frames(T0, T0 + 12, lead_start_s=1000.0)
    summary, run_dir, _session = _drive(_world(12.0),
                                        wall_arm(hold_s=4.0), frames,
                                        tmp_path)
    request = [e for e in _jsonl(run_dir / "wall_events.jsonl")
               if e["kind"] == "wall_request"][0]
    assert request["lead_distance_to_merge_m"] == pytest.approx(1644.2)
    assert request["dmin_m"] == pytest.approx(17.0 * 70.0)
    assert request["lead_s_road_m"] == pytest.approx(1000.0)
    assert request["lead_distance_to_merge_m"] >= request["dmin_m"]


def test_outcome_released_road_end_is_never_recorded():
    assert wall_driver.OUTCOME_RELEASED_ROAD_END \
        not in wall_driver.OUTCOMES
    assert wall_driver.OUTCOME_MERGE_RELEASED in wall_driver.OUTCOMES


def test_active_wall_at_horizon_still_fails_loudly(tmp_path):
    frames = _merge_crossing_frames(T0, T0 + 6, lead_start_s=1000.0)
    with pytest.raises(wall_driver.WallDriverError, match="still active"):
        _drive(_world(6.0), wall_arm(hold_s=600.0), frames, tmp_path)
