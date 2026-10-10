"""The WALL-1 child: restore, ONE loadState, receipt (fakes only)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from campaign_walls_v1 import run_wall_branch
from wall_fakes import (
    CAVS,
    T0,
    FakeApi,
    FakeOwnership,
    FakeSensor,
    base_saved_state,
    converging_frames,
    real_session,
    wall_arm,
)


def _plan(bases_root: Path, duration_s: float = 10.0, hold_s: float = 6.0):
    branch = wall_arm(run_name="w300p10t300__w-mid-early", t0=T0,
                      hold_s=hold_s)
    branch["restores_state"]["from_run"] = "w300p10t300__base"
    world = {
        "world": "w300p10t300", "seed": 300, "loading_veh_h": 2700,
        "penetration": 0.1, "split": "TRAIN", "role": "exposed",
        "t0_s": T0, "bases_root": str(bases_root),
        "branch_length_s": duration_s,
        "base_run": "w300p10t300__base",
        "branches": [branch],
    }
    return {"worlds": [world]}


def _run(plan_path_tmp, bases_root, duration_s=10.0, hold_s=6.0):
    runs_root = plan_path_tmp / "runs"
    runs_root.mkdir()
    plan = _plan(bases_root, duration_s, hold_s)
    api = FakeApi(t0=T0)
    session, owner = real_session(CAVS)
    frames = converging_frames(T0, T0 + duration_s, wall_speed=17.0)
    sensor = FakeSensor(frames)
    receipt = run_wall_branch.run_wall_branch_child(
        {"run": "w300p10t300__w-mid-early"}, plan, runs_root,
        plan_path_tmp / "REG.md", api=api, session=session, ownership=owner,
        sensor=sensor)
    return receipt, runs_root, api


def test_child_restores_once_drives_and_writes_receipt(tmp_path):
    bases_root = tmp_path / "chunk2"
    base_saved_state(bases_root / "w300p10t300__base", T0)
    receipt, runs_root, api = _run(tmp_path, bases_root)
    assert receipt["schema"] == "wm59_wall_branch_receipt/1"
    assert receipt["exit_status"] == "complete"
    assert receipt["load_state_calls"] == 1        # exactly ONE loadState
    assert receipt["horizon_mode"] == "registered_full"
    assert receipt["wall_summary"]["wall_outcome"] == "assembled_released"
    assert receipt["saved_state_id"] == api_restored_state_id(tmp_path / "chunk2")
    run_dir = runs_root / "w300p10t300__w-mid-early"
    for name in ("RUN_RECEIPT.json", "run/wall_events.jsonl",
                 "run/physics_steps.jsonl", "run/dispatcher_rows.jsonl",
                 "run/insertions.jsonl", "run/branch_record.json",
                 "run/outcome_record.json"):
        assert (run_dir / name).is_file(), name
    record = json.loads(
        (run_dir / "run" / "branch_record.json").read_text(encoding="utf-8"))
    assert record["outcome"] == "assembled_released"


def api_restored_state_id(bases_root: Path) -> str:

    from campaign_stage1_v3.common import sha256_file, sha256_text

    state_dir = bases_root / "w300p10t300__base" / "saved_state"
    manifest_text = (state_dir / "STATE_MANIFEST.json").read_text(encoding="utf-8")
    return sha256_text(manifest_text + sha256_file(state_dir / "state.xml"))


def test_child_refuses_an_existing_run_dir(tmp_path):
    bases_root = tmp_path / "chunk2"
    base_saved_state(bases_root / "w300p10t300__base", T0)
    runs_root = tmp_path / "runs"
    (runs_root / "w300p10t300__w-mid-early").mkdir(parents=True)
    plan = _plan(bases_root)
    with pytest.raises(run_wall_branch.WallChildError, match="existing"):
        run_wall_branch.run_wall_branch_child(
            {"run": "w300p10t300__w-mid-early"}, plan, runs_root,
            tmp_path / "REG.md", api=FakeApi(t0=T0),
            session=FakeOwnership(CAVS), ownership=FakeOwnership(CAVS),
            sensor=FakeSensor({}))


def test_child_refuses_partial_injection(tmp_path):
    bases_root = tmp_path / "chunk2"
    base_saved_state(bases_root / "w300p10t300__base", T0)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    plan = _plan(bases_root)
    with pytest.raises(run_wall_branch.WallChildError, match="partial"):
        run_wall_branch.run_wall_branch_child(
            {"run": "w300p10t300__w-mid-early"}, plan, runs_root,
            tmp_path / "REG.md", api=FakeApi(t0=T0),
            session=FakeOwnership(CAVS))


def test_child_starts_own_sumo_when_no_api(monkeypatch, tmp_path):
    calls = []

    def fake_session(base_world_dir, seed, world, copy_dir=None):
        calls.append((str(base_world_dir), seed, str(copy_dir)))
        raise _Started()

    import campaign_pilot_v1.run_pilot as rp

    monkeypatch.setattr(rp, "branch_sumo_session", fake_session)
    bases_root = tmp_path / "chunk2"
    base_saved_state(bases_root / "w300p10t300__base", T0)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    with pytest.raises(_Started):
        run_wall_branch.run_wall_branch_child(
            {"run": "w300p10t300__w-mid-early"}, _plan(bases_root), runs_root,
            tmp_path / "REG.md")
    # WM-64: the wall child targets the rewritten-scenario COPY inside its
    # own run dir (SUMO opens the copy, the recorded base stays read-only)
    assert calls == [(
        str(bases_root / "w300p10t300__base"), 300,
        str(runs_root / "w300p10t300__w-mid-early" / "run" / "scenario"))]


class _Started(Exception):
    pass


def test_guard_refuses_a_second_restore_in_the_same_process(tmp_path):
    from campaign_walls_v1.common import WallsPilotError

    bases_root = tmp_path / "chunk2"
    base_saved_state(bases_root / "w300p10t300__base", T0)
    receipt, runs_root, api = _run(tmp_path, bases_root)
    assert api.load_state_calls == 1
    with pytest.raises(WallsPilotError, match="SECOND loadState"):
        api.simulation.loadState("x")   # any SECOND restore refuses


def test_resolve_run_rejects_non_wall_arms():
    plan = {"worlds": [{"world": "w", "branches": [
        {"run": "w__native", "arm": "native", "mode": "restore_suffix"}]}]}
    with pytest.raises(run_wall_branch.WallChildError, match="not a wall"):
        run_wall_branch.resolve_run(plan, "w__native")
    with pytest.raises(run_wall_branch.WallChildError, match="not in the plan"):
        run_wall_branch.resolve_run(plan, "w__missing")


# ---- WALL-2 audit addition: the CLI argument/plan wiring ----

def test_cli_wiring_forwards_run_plan_and_optional_flags(monkeypatch,
                                                         tmp_path, capsys):
    captured = {}

    def fake_child(run, plan, runs_root, registration_path, *,
                   bases_root=None, horizon_ticks=None, **_kwargs):
        captured.update(run=run, plan=plan, runs_root=runs_root,
                        registration_path=registration_path,
                        bases_root=bases_root, horizon_ticks=horizon_ticks)
        return {"run": run["run"], "exit_status": "complete",
                "wall_summary": {"wall_outcome": "assembled_released"}}

    monkeypatch.setattr(run_wall_branch, "run_wall_branch_child", fake_child)
    plan_path = tmp_path / "WORLD_PLAN.json"
    plan_payload = {"schema": "wm59_walls_pilot_world_plan/1", "worlds": []}
    plan_path.write_text(json.dumps(plan_payload), encoding="utf-8")
    runs_root = tmp_path / "runs"
    registration = tmp_path / "REG.md"
    bases_root = tmp_path / "bases"

    code = run_wall_branch.main([
        "_exec_wall_branch", "--run", "w__w-mid-early",
        "--plan", str(plan_path), "--runs-root", str(runs_root),
        "--registration", str(registration)])
    assert code == 0
    assert captured["run"] == {"run": "w__w-mid-early"}
    assert captured["plan"] == plan_payload
    assert captured["runs_root"] == runs_root
    assert captured["registration_path"] == registration
    assert captured["bases_root"] is None          # defaults: plan wins later
    assert captured["horizon_ticks"] is None       # full registered branch
    printed = json.loads(capsys.readouterr().out)
    assert printed == {"run": "w__w-mid-early", "exit_status": "complete",
                       "wall_outcome": "assembled_released"}

    code = run_wall_branch.main([
        "_exec_wall_branch", "--run", "w__w-lo-late",
        "--plan", str(plan_path), "--runs-root", str(runs_root),
        "--registration", str(registration), "--bases-root", str(bases_root),
        "--horizon-ticks", "75"])
    assert code == 0
    assert captured["bases_root"] == bases_root    # the CLI flag wins
    assert captured["horizon_ticks"] == 75.0       # the smoke cap


def test_cli_requires_run_and_plan(tmp_path):
    with pytest.raises(SystemExit):   # argparse: missing required --run
        run_wall_branch.main(["_exec_wall_branch",
                              "--plan", str(tmp_path / "WORLD_PLAN.json")])
    with pytest.raises(SystemExit):   # argparse: no subcommand at all
        run_wall_branch.main([])
