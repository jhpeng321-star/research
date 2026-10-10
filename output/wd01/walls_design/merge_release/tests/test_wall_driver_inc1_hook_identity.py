"""Frozen post driver vs post_inc1 hook driver: byte-for-byte identity.

REGISTRATION_WM_N1_INC1_ADDENDUM_1 A1.2: with both new kwargs ABSENT the
patched driver must behave identically to the frozen v2.2 post driver.  This
test drives both modules through the SAME fake-traci fixtures and compares
EVERY command/log row (all JSONL files + the branch/outcome records, ignoring
only the wall-clock ``created_utc`` field).

It also exercises the two new kwargs (present) to pin their contract:
- ``post_restore_hook`` is called exactly once, before the first control tick,
  and its return value is logged in the branch record;
- ``branch_length_s_override`` sets the branch horizon.

Zero SUMO.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from campaign_walls_v1 import wall_driver as frozen
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

POST_INC1 = (Path(__file__).resolve().parent.parent / "copies" / "post_inc1"
             / "wall_driver.py")


def _load_post_inc1():
    spec = importlib.util.spec_from_file_location(
        "wall_driver_inc1_hook", str(POST_INC1))
    module = importlib.util.module_from_spec(spec)
    sys.modules["wall_driver_inc1_hook"] = module
    spec.loader.exec_module(module)
    return module


post_inc1 = _load_post_inc1()


def _world(duration_s: float):
    return {"world": "w300p10", "seed": 300, "branch_length_s": duration_s}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _drive(mod, world, arm, frames, tmp_path, **kwargs):
    api = FakeApi(t0=T0, live=MEMBERS)
    session, owner = real_session(CAVS)
    sensor = FakeSensor(frames)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    summary = mod.drive_wall_branch(
        api, world, arm, run_dir, restore_entry(CAVS), session=session,
        ownership=owner, sensor=sensor, **kwargs)
    return summary, run_dir


def _frame(vid, s_road, lane_index, speed, now):
    return frame_at_s(vid, s_road, lane_index=lane_index, speed=speed, now=now)


def _merge_crossing_frames(t0, end, *, wall_speed=17.0, lead_start_s=1000.0):
    frames = {}
    lead_s = float(lead_start_s)
    for t in range(int(t0), int(end)):
        now = float(t)
        if t == int(t0):
            pair = entry_frames(ENTRY_LEN_M - lead_s, speeds=(20.0, 20.0),
                                now=now)
        elif t == int(t0) + 1:
            pair = entry_frames(ENTRY_LEN_M - (lead_s + 20.0),
                                speeds=(18.5, 18.5), now=now)
        else:
            lead_s += float(wall_speed)
            pair = {
                MEMBERS[0]: _frame(MEMBERS[0], lead_s, 0, wall_speed, now),
                MEMBERS[1]: _frame(MEMBERS[1], lead_s - 3.0, 1, wall_speed,
                                   now)}
        frames[now] = pair
    return frames


def _dmin_frames(t0, end):
    return {float(t): entry_frames(500.0, now=float(t))
            for t in range(int(t0), int(end))}


def _snapshot(run_dir: Path) -> dict:
    snap = {}
    for name in ("physics_steps.jsonl", "dispatcher_rows.jsonl",
                 "suffix_commands.jsonl", "wall_events.jsonl",
                 "insertions.jsonl"):
        snap[name] = _jsonl(run_dir / name)
    for name in ("branch_record.json", "outcome_record.json"):
        record = json.loads((run_dir / name).read_text(encoding="utf-8"))
        record.pop("created_utc", None)
        snap[name] = record
    return snap


CASES = {
    "merge_released": lambda: (
        _world(120.0), wall_arm(hold_s=600.0),
        _merge_crossing_frames(T0, T0 + 120)),
    "assembled_released": lambda: (
        _world(20.0), wall_arm(hold_s=6.0),
        _merge_crossing_frames(T0, T0 + 20)),
    "not_feasible": lambda: (
        _world(130.0), wall_arm(hold_s=6.0), _dmin_frames(T0, T0 + 130)),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_frozen_and_patched_drivers_are_identical(case, tmp_path):
    world, arm, frames = CASES[case]()
    frozen_summary, frozen_dir = _drive(
        frozen, world, arm, frames, tmp_path / "frozen")
    inc1_summary, inc1_dir = _drive(
        post_inc1, world, arm, frames, tmp_path / "inc1")
    assert frozen_summary == inc1_summary
    frozen_snap = _snapshot(frozen_dir)
    inc1_snap = _snapshot(inc1_dir)
    assert frozen_snap.keys() == inc1_snap.keys()
    for name in frozen_snap:
        assert inc1_snap[name] == frozen_snap[name], f"{name} differs"
    # no hook key leaks into the identity run
    assert "post_restore_hook" not in frozen_snap["branch_record.json"]
    assert "post_restore_hook" not in inc1_snap["branch_record.json"]


def test_active_wall_at_horizon_fails_identically(tmp_path):
    world = _world(6.0)
    arm = wall_arm(hold_s=600.0)
    frames = _merge_crossing_frames(T0, T0 + 6)
    with pytest.raises(frozen.WallDriverError, match="still active"):
        _drive(frozen, world, arm, frames, tmp_path / "frozen")
    with pytest.raises(post_inc1.WallDriverError, match="still active"):
        _drive(post_inc1, world, arm, frames, tmp_path / "inc1")


def test_post_restore_hook_called_once_and_logged(tmp_path):
    world = _world(12.0)
    arm = wall_arm(hold_s=4.0)
    frames = _merge_crossing_frames(T0, T0 + 12)
    calls = []

    def hook(api, t_s):
        calls.append((float(api.simulation.getTime()), float(t_s)))
        return {"obstacle_id": "incident.obstacle", "inserted_at_s": t_s}

    _summary, run_dir = _drive(
        post_inc1, world, arm, frames, tmp_path, post_restore_hook=hook)
    assert calls == [(T0, T0)]                      # exactly once, at t0
    record = json.loads(
        (run_dir / "branch_record.json").read_text(encoding="utf-8"))
    assert record["post_restore_hook"] == {
        "obstacle_id": "incident.obstacle", "inserted_at_s": T0}
    # the first control tick in the command log is still t0 (hook ran before)
    commands = _jsonl(run_dir / "suffix_commands.jsonl")
    first_command = [c for c in commands if c["kind"] == "command"][0]
    assert first_command["timestamp_s"] == T0


def test_branch_length_s_override_sets_horizon(tmp_path):
    world = _world(600.0)          # the campaign default would be 600 s
    arm = wall_arm(hold_s=4.0)
    frames = _merge_crossing_frames(T0, T0 + 12)
    summary, _run_dir = _drive(
        post_inc1, world, arm, frames, tmp_path, branch_length_s_override=300.0)
    assert summary["horizon_s"] == pytest.approx(T0 + 300.0)
    assert summary["t0_s"] == pytest.approx(T0)
