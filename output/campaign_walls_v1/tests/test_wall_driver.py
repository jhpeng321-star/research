"""Wall-branch DRIVER end-to-end on the REAL frozen dispatcher session.

The dispatcher lifecycle (assembly -> hold -> handback, safety aborts)
is the REAL ``MixedAssignmentSession`` + ``WallGroupSession``; only the
api/sensor/ownership are fakes issuing scripted REPORT frames - the same
two-level pattern as the WM-43 fixture.  Zero SUMO.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from campaign_walls_v1 import wall_driver
from campaign_walls_v1.common import wall_id_for
from wall_fakes import (
    CAVS,
    EDGE,
    MEMBERS,
    T0,
    FakeApi,
    FakeSensor,
    converging_frames,
    feasible_frames,
    make_frame,
    real_session,
    restore_entry,
    wall_arm,
)


def _world(duration_s: float):
    return {"world": "w300p10", "seed": 300, "branch_length_s": duration_s}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _drive(world, arm, frames, tmp_path, *, cav_ids=CAVS, live=MEMBERS,
           owner=None):
    api = FakeApi(t0=T0, live=live)
    session, owner = real_session(cav_ids, owner)
    sensor = FakeSensor(frames)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary = wall_driver.drive_wall_branch(
        api, world, arm, run_dir, restore_entry(cav_ids), session=session,
        ownership=owner, sensor=sensor)
    return summary, run_dir, session


def test_early_request_assembles_releases_with_locked_rows(tmp_path):
    hold_s = 6.0
    frames = converging_frames(T0, T0 + 10, wall_speed=17.0)
    summary, run_dir, session = _drive(
        _world(10.0), wall_arm(hold_s=hold_s), frames, tmp_path)
    # requested at the FIRST tick >= t0 (early), from reports only
    assert summary["request_s"] == pytest.approx(T0)
    assert summary["wall_outcome"] == "assembled_released"
    assert summary["wall_speed_mps"] == pytest.approx(17.0)
    events = _jsonl(run_dir / "wall_events.jsonl")
    request = [e for e in events if e["kind"] == "wall_request"][0]
    assert request["source"] == "cav_report_frames"
    assert request["timing"] == "early"
    assert set(request["members"]) == set(MEMBERS)
    assert request["lanes"] == [0, 1]                # one member per lane
    assert request["span_m"] == pytest.approx(40.0)
    kinds = [(e.get("event"), e.get("timestamp_s")) for e in events
             if e.get("kind") == "wall_event"]
    assert ("assembly_start", T0) in kinds
    assembled_at = dict(kinds)["assembled"]
    assert assembled_at == pytest.approx(T0 + 2.0)   # scripted convergence
    handback = [e for e in events if e["kind"] == "wall_handback"][0]
    assert handback["how"] == "released" and handback["handback"] == "native"
    # the recorded hold: release counted FROM assembly completion
    released_at = dict(kinds)["released"]
    assert released_at == pytest.approx(assembled_at + hold_s)
    # dispatcher rows: members LOCKED (wall-v2, assembly/hold) from the
    # request to the handback; native afterwards.  The substep row stamped
    # exactly at released_at re-stamps the PREVIOUS control tick (the
    # declared 0.5 s contract), so 'after' is strictly past it.
    rows = _jsonl(run_dir / "dispatcher_rows.jsonl")
    member_rows = [r for r in rows if r["vehicle_id"] == MEMBERS[0]]
    active = [r for r in member_rows
              if request["timestamp_s"] - 1e-9 <= r["timestamp_s"]
              <= released_at]
    assert active and all(
        r["action"] == "wall-v2" and r["locked"] is True and r["wall_id"]
        and r["wall_phase"] in ("assembly", "hold") for r in active)
    phases = {r["wall_phase"] for r in active}
    assert phases == {"assembly", "hold"}
    after = [r for r in member_rows if r["timestamp_s"] > released_at]
    assert after
    assert all(r["action"] == "native" and r["locked"] is False
               for r in after)
    # command rows: the wall paced ONLY its members
    commands = _jsonl(run_dir / "suffix_commands.jsonl")
    wall_rows = [c for c in commands if c.get("owner") == "wall"]
    assert wall_rows
    assert {c["vehicle_id"] for c in wall_rows} == set(MEMBERS)
    record = json.loads((run_dir / "branch_record.json").read_text(encoding="utf-8"))
    assert record["outcome"] == "assembled_released"
    assert record["released_at_s"] == pytest.approx(released_at)
    assert set(record["members"]) == set(MEMBERS)
    assert record["hold_s"] == hold_s


def test_non_members_untouched(tmp_path):
    live = ("cav.0",) + MEMBERS
    frames = converging_frames(T0, T0 + 10, wall_speed=17.0)
    frames = {t: dict(fs) for t, fs in frames.items()}
    for t, fs in frames.items():
        fs["cav.0"] = make_frame("cav.0", 22.0, 0, 300.0, f"{EDGE}_0", t)
    summary, run_dir, session = _drive(
        _world(10.0), wall_arm(hold_s=6.0), frames, tmp_path,
        cav_ids=live, live=live)
    rows = _jsonl(run_dir / "dispatcher_rows.jsonl")
    non_member_rows = [r for r in rows if r["vehicle_id"] == "cav.0"]
    assert non_member_rows
    assert all(r["action"] == "native" and r["locked"] is False
               and r["wall_id"] is None for r in non_member_rows)
    commands = _jsonl(run_dir / "suffix_commands.jsonl")
    assert all(c["vehicle_id"] != "cav.0" for c in commands
               if c.get("owner") == "wall")
    # the member choice ignores the far non-member (span 220 m > 75 m to
    # the lane-1 member) and picks the near member pair
    assert set(summary["members"]) == set(MEMBERS)


def test_late_request_at_first_feasible_tick_after_t0_plus_60(tmp_path):
    frames = converging_frames(T0, T0 + 85, feasible_from=T0 + 65,
                               wall_speed=12.0)
    arm = wall_arm(hold_s=6.0, arm="W-lo-late")
    arm["wall"]["timing"] = "late"
    summary, run_dir, session = _drive(_world(85.0), arm, frames, tmp_path)
    assert summary["request_s"] == pytest.approx(T0 + 65.0)  # first FEASIBLE
    request = [e for e in _jsonl(run_dir / "wall_events.jsonl")
               if e["kind"] == "wall_request"][0]
    assert request["timing"] == "late"
    assert request["target_time_s"] == pytest.approx(T0 + 60.0)
    assert request["speed_factor"] == 0.60
    assert summary["wall_speed_mps"] == pytest.approx(12.0)  # 0.60 x 20 m/s
    assert summary["wall_outcome"] == "assembled_released"


def test_not_feasible_after_120s_continues_native(tmp_path):
    frames = converging_frames(T0, T0 + 130, feasible_from=T0 + 10 ** 6)
    summary, run_dir, session = _drive(_world(130.0), wall_arm(hold_s=6.0),
                                       frames, tmp_path)
    assert summary["wall_outcome"] == "not_feasible"
    not_feasible = [e for e in _jsonl(run_dir / "wall_events.jsonl")
                    if e["kind"] == "not_feasible"]
    assert len(not_feasible) == 1
    assert not_feasible[0]["timestamp_s"] == pytest.approx(T0 + 121.0)
    assert not_feasible[0]["deadline_s"] == pytest.approx(T0 + 120.0)
    record = json.loads((run_dir / "branch_record.json").read_text(encoding="utf-8"))
    assert record["outcome"] == "not_feasible"
    assert record["members"] is None and record["request_s"] is None
    # the world ran native: no wall rows anywhere
    rows = _jsonl(run_dir / "dispatcher_rows.jsonl")
    assert all(r["locked"] is False and r["wall_id"] is None for r in rows)


def test_abort_in_assembly_logs_reason_and_hands_back(tmp_path):
    frames = converging_frames(T0, T0 + 6, wall_speed=17.0)
    for t in (T0 + 1.0, T0 + 2.0, T0 + 3.0, T0 + 4.0, T0 + 5.0):
        frames[t] = {}                     # member reports vanish BEFORE
        # assembly can complete (the scripted convergence needed t0 + 2)
    summary, run_dir, session = _drive(_world(6.0), wall_arm(hold_s=600.0),
                                       frames, tmp_path)
    assert summary["wall_outcome"] == "aborted_in_assembly"
    events = _jsonl(run_dir / "wall_events.jsonl")
    abort = [e for e in events if e.get("event") == "safety_abort"][0]
    assert abort["reason"] == "member_frame_missing"
    handback = [e for e in events if e["kind"] == "wall_handback"][0]
    assert handback["how"] == "safety_abort"
    assert handback["phase"] == "assembly"
    record = json.loads((run_dir / "branch_record.json").read_text(encoding="utf-8"))
    assert record["outcome"] == "aborted_in_assembly"
    assert record["abort_reason"] == "member_frame_missing"
    assert record["assembled_at_s"] is None
    # members went NATIVE at the abort: no locked rows afterwards
    rows = _jsonl(run_dir / "dispatcher_rows.jsonl")
    late = [r for r in rows
            if r["vehicle_id"] == MEMBERS[0]
            and r["timestamp_s"] > abort["timestamp_s"]]
    assert all(r["action"] == "native" and r["locked"] is False for r in late)


def test_abort_in_hold(tmp_path):
    frames = converging_frames(T0, T0 + 8, wall_speed=17.0)
    frames[T0 + 6.0] = {}                  # vanish while HOLDING
    frames[T0 + 7.0] = {}
    summary, run_dir, session = _drive(_world(8.0), wall_arm(hold_s=600.0),
                                       frames, tmp_path)
    assert summary["wall_outcome"] == "aborted_in_hold"
    record = json.loads((run_dir / "branch_record.json").read_text(encoding="utf-8"))
    assert record["outcome"] == "aborted_in_hold"
    assert record["assembled_at_s"] is not None
    events = _jsonl(run_dir / "wall_events.jsonl")
    handback = [e for e in events if e["kind"] == "wall_handback"][0]
    assert handback["phase"] == "hold"


def test_active_wall_at_horizon_fails_loudly(tmp_path):
    frames = converging_frames(T0, T0 + 6, wall_speed=17.0)
    with pytest.raises(wall_driver.WallDriverError, match="still active"):
        _drive(_world(6.0), wall_arm(hold_s=600.0), frames, tmp_path)


def test_wall_plan_is_strict_roster_native_elsewhere(tmp_path):
    live = ("cav.0",) + MEMBERS
    frames = converging_frames(T0, T0 + 10, wall_speed=17.0)
    frames = {t: dict(fs) for t, fs in frames.items()}
    for t, fs in frames.items():
        fs["cav.0"] = make_frame("cav.0", 22.0, 0, 300.0, f"{EDGE}_0", t)
    summary, run_dir, session = _drive(
        _world(10.0), wall_arm(hold_s=6.0), frames, tmp_path,
        cav_ids=live, live=live)
    # exactly ONE plan was ever applied: the wall request itself
    actions = {(r["vehicle_id"], r["action"]) for r in
               _jsonl(run_dir / "dispatcher_rows.jsonl")}
    assert ("cav.0", "native") in actions
    assert (MEMBERS[0], "wall-v2") in actions and \
        (MEMBERS[1], "wall-v2") in actions
    assert summary["wall_id"] == wall_id_for(summary["wall_id"].split(":")[0])


def test_second_load_state_refused_by_guard():
    api = FakeApi(t0=T0)
    guard = wall_driver.install_load_state_guard(api)
    api.simulation.loadState("x")          # the ONE restore
    assert guard["calls"] == 1
    with pytest.raises(wall_driver.WallDriverError, match="SECOND loadState"):
        api.simulation.loadState("x")


def test_feasible_request_frame_reference():
    """The feasibility helper used everywhere rejects a same-lane pair."""
    frames = feasible_frames()
    frames[MEMBERS[1]] = make_frame(MEMBERS[1], 20.0, 0, 100.0,
                                    f"{EDGE}_0", T0)
    from campaign_walls_v1 import wall_request

    assert wall_request.choose_pair(frames) is None
