"""WD-01-PB-V04A: schedule-authoritative actions, arm verification and the
planner-known schedule summary (pi shares)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from conftest import T0
from v04a_fixtures import (_command_rows, _cmds, make_world,
                           write_schedule)

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import actions_v04 as A  # noqa: E402


def _tl(arm, **kw):
    return A.ActionTimeline.from_run_dir(arm, schedule_authoritative=True,
                                         **kw)


def test_pi_token_is_pi_when_schedule_authoritative(tmp_path):
    arms = make_world(tmp_path)
    tl = _tl(arms["pi"])
    assert tl.action_at("a", 245.0) == "pi"
    # default (realised commands win) would label the PI CAV damping
    # (the PI writer's owner is 'damping'): the documented defect the
    # schedule-authoritative mode fixes
    legacy = A.ActionTimeline.from_run_dir(arms["pi"])
    assert legacy.action_at("a", 245.0) == "damping"
    assert tl.verify_arm("pi_fleet")["ok"]


def test_switching_action_follows_the_schedule_between_events(tmp_path):
    arms = make_world(tmp_path)
    tl = _tl(arms["switching1"])
    assert tl.action_at("a", 245.0) == "native"
    assert tl.action_at("a", 245.3) == "damping"
    assert tl.action_at("a", 248.9) == "pi"
    assert tl.action_at("c", 100.0) == "unknown"     # before first entry
    assert tl.verify_arm("switching")["n_cavs_multi_event"] == 2


def test_action_shares_time_fractions_and_unknown_excluded(tmp_path):
    arms = make_world(tmp_path)
    tl = _tl(arms["switching1"])
    sh = tl.action_shares(["a"], 240.0, 30.0)
    assert sh.sum() == pytest.approx(1.0)
    assert sh[0] == pytest.approx(5.3 / 30)
    assert sh[1] == pytest.approx(3.6 / 30)
    assert sh[2] == pytest.approx(21.1 / 30)
    # a window starting before the first entry: unknown time is excluded
    # and shares renormalised over known time
    sh2 = tl.action_shares(["a"], 230.0, 20.0)   # 10 s unknown, 10 s known
    assert sh2.sum() == pytest.approx(1.0)
    assert sh2.tolist() == pytest.approx([0.53, 0.36, 0.11])
    assert not tl.action_shares([], 240.0, 30.0).any()
    # fleet run: exact one-hot
    base = A.ActionTimeline.from_run_dir(arms["base"])
    assert base.action_shares(["a"], 240.0, 30.0).tolist() == [0, 1, 0]


@pytest.mark.parametrize("kind,arm,ok", [
    ("native_fleet", "native", True), ("pi_fleet", "pi", True),
    ("switching", "switching1", True), ("fixed", "native", True),
    ("switching", "native", False), ("fixed", "switching1", False),
    ("native_fleet", "pi", False), ("pi_fleet", "native", False),
    ("damping_fleet", "base", True), ("native_fleet", "base", False)])
def test_verify_arm_matrix(tmp_path, kind, arm, ok):
    arms = make_world(tmp_path)
    tl = _tl(arms[arm])
    if ok:
        assert tl.verify_arm(kind)["mismatches"] == 0
    else:
        with pytest.raises(A.ArmPolicyMismatch):
            tl.verify_arm(kind)


def test_native_schedule_with_a_damping_owner_is_a_mismatch(tmp_path):
    arms = make_world(tmp_path, native_owner="damping")
    with pytest.raises(A.ArmPolicyMismatch, match="disagree"):
        _tl(arms["native"]).verify_arm("native_fleet")


def test_pi_schedule_without_the_damping_writer_is_a_mismatch(tmp_path):
    arms = make_world(tmp_path)
    _cmds(arms["pi"] / "run" / "suffix_commands.jsonl",
          _command_rows("a", 241, 250, "native"))
    with pytest.raises(A.ArmPolicyMismatch, match="damping writer"):
        _tl(arms["pi"]).verify_arm("pi_fleet")


def test_missing_commands_or_schedule_refuse(tmp_path):
    arms = make_world(tmp_path)
    _cmds(arms["native"] / "run" / "suffix_commands.jsonl", [])
    with pytest.raises(A.ArmPolicyMismatch, match="no kind"):
        _tl(arms["native"]).verify_arm("native_fleet")
    arms2 = make_world(tmp_path / "b")
    write_schedule(arms2["native"], "native", [])
    with pytest.raises(A.ArmPolicyMismatch, match="no switch_schedule|manifest fleet policy"):
        _tl(arms2["native"]).verify_arm("native_fleet")


def test_non_command_rows_are_not_owners(tmp_path):
    arms = make_world(tmp_path)
    p = arms["pi"] / "run" / "suffix_commands.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    rows.append(dict(kind="pi_owner_receipt", timestamp_s=245.0,
                     vehicle_id="a", owner="pi-external-u"))
    rows.append(dict(kind="pi_source_evidence", timestamp_s=245.0,
                     vehicle_id="a", frame={}))
    _cmds(p, rows)
    tl = _tl(arms["pi"])                    # would raise on a bad owner
    assert tl.verify_arm("pi_fleet")["ok"]
    assert tl.n_command_rows == 9


# ------------------------------------------- WM-N1 label coverage: wall ----

def _wall_arm(tmp_path, owners=("wall", "native"), *, with_schedule=False):
    """A walls-pilot-shaped branch arm: no switch_schedule, command rows
    owned by the wall writer and the native fallback."""
    name = "w-lo-early"
    d = tmp_path / "runs" / "wwall1m2700p10t420__w-lo-early"
    from conftest import synth_physics_rows, write_mini_arm
    write_mini_arm(d, synth_physics_rows(speed_mode=27), T0)
    if with_schedule:
        write_schedule(d, name, [("a", T0, "native")])
    _cmds(d / "run" / "suffix_commands.jsonl",
          _command_rows("a", 241, 243, owners[0])
          + _command_rows("a", 243, 245, owners[1]))
    return d


def test_wall_kind_passes_on_wall_owned_commands(tmp_path):
    d = _wall_arm(tmp_path)
    rec = _tl(d).verify_arm("wall")
    assert rec["ok"] and rec["kind"] == "wall"
    assert rec["owner_counts"] == {"wall": 2, "native": 2}


def test_wall_kind_refuses_native_only_and_foreign_owners(tmp_path):
    d = _wall_arm(tmp_path / "a", owners=("native", "native"))
    with pytest.raises(A.ArmPolicyMismatch, match="never owned"):
        _tl(d).verify_arm("wall")
    d2 = _wall_arm(tmp_path / "b")
    p = d2 / "run" / "suffix_commands.jsonl"
    _cmds(p, _command_rows("a", 241, 245, "pi"))
    with pytest.raises(A.ArmPolicyMismatch, match="owners outside"):
        _tl(d2).verify_arm("wall")


def test_wall_kind_refuses_a_schedule_and_empty_commands(tmp_path):
    d = _wall_arm(tmp_path / "a", with_schedule=True)
    with pytest.raises(A.ArmPolicyMismatch, match="switch schedule exists"):
        _tl(d).verify_arm("wall")
    d2 = _wall_arm(tmp_path / "b")
    _cmds(d2 / "run" / "suffix_commands.jsonl", [])
    with pytest.raises(A.ArmPolicyMismatch, match="no kind"):
        _tl(d2).verify_arm("wall")
    with pytest.raises(ValueError, match="unknown arm kind"):
        _tl(d2).verify_arm("spaceship")
