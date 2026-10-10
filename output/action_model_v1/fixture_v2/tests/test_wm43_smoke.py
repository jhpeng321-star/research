"""WM-43B standing pre-launch smoke: gate + orchestrator tests (no SUMO).

The smoke itself (real TraCI, 70-tick A4 (wide-pair closure) run into <runs_root>_smoke) is
exercised here only through the fake-TraCI child path; the parent flow
(preflight, allocation row, write-once SMOKE_VERDICT.json bound to the
registration sha256 + package code digest, gate refusals) is exercised
end-to-end with an in-process launcher stub that runs the REAL loop and
REAL smoke checks - exactly the work the OS child performs.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from conftest import FIXTURE_DIR, REPO_ROOT  # noqa: F401 - pytest path helper
from fake_sumo import ScriptedDampingController, build_fake_session
from wm43_checks import evaluate_smoke
from wm43_plan import (HORIZON_TICKS, SMOKE_ARM, SMOKE_CHECK_IDS,
                       SMOKE_HORIZON_TICKS, SMOKE_MAX_HORIZON_TICKS,
                       SMOKE_RUNS_ROOT_SUFFIX, SMOKE_SEED, check_plan_shape,
                       load_plan)
import run_fixture

SMOKE_RUN = f"{SMOKE_ARM.lower()}_seed{SMOKE_SEED}"


# ---------------------------------------------------------------------------
# registered shape
# ---------------------------------------------------------------------------

def test_smoke_block_registered_in_plan():
    plan = load_plan()
    assert check_plan_shape(plan) == []
    smoke = plan["smoke"]
    assert smoke["arm"] == SMOKE_ARM == "A4" and smoke["seed"] == SMOKE_SEED == 920264
    assert smoke["horizon_ticks"] == SMOKE_HORIZON_TICKS == 70
    assert smoke["horizon_ticks"] <= SMOKE_MAX_HORIZON_TICKS == 80
    assert smoke["runs_root_relative"] == \
        plan["constants"]["runs_root_relative"] + SMOKE_RUNS_ROOT_SUFFIX
    assert smoke["used_as_data"] is False
    assert smoke["simulated_seconds"] == smoke["horizon_ticks"] == 70


def test_smoke_check_ids_registered():
    assert SMOKE_CHECK_IDS == (
        "S01_smoke_horizon_complete", "S02_smoke_zero_collisions",
        "S03_smoke_zero_teleports", "S04_smoke_single_writer_path",
        "S05_smoke_assembly_started", "S06_smoke_assembled_by_cap")


# ---------------------------------------------------------------------------
# digest + RSS helpers
# ---------------------------------------------------------------------------

def test_package_code_digest_binds_module_bytes():
    digest = run_fixture.package_code_digest()
    assert digest == run_fixture.package_code_digest()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for module in FIXTURE_DIR.glob("*.py"):
            shutil.copyfile(module, tmp_dir / module.name)
        (tmp_dir / "wm43_plan.py").write_text(
            (tmp_dir / "wm43_plan.py").read_text(encoding="utf-8") + "\n# x\n",
            encoding="utf-8")
        assert run_fixture.package_code_digest(tmp_dir) != digest


def test_peak_rss_sample_shape():
    sample = run_fixture.peak_rss_sample()
    assert isinstance(sample, dict) and "available" in sample
    if sample["available"]:
        assert isinstance(sample["self_peak_rss"], int)
        assert sample["self_peak_rss"] >= 0


# ---------------------------------------------------------------------------
# capped loop + smoke checks (real loop code, fake TraCI, real shapes)
# ---------------------------------------------------------------------------

def _run_smoke_window(tmp_path, horizon=SMOKE_HORIZON_TICKS):
    from traffic_shepherd.config import load_config
    from wm43_loop import run_mechanics
    world = {"run": SMOKE_RUN, "arm": SMOKE_ARM, "seed": SMOKE_SEED,
             "loading": 2000, "penetration": 1.0, "role": "exposed",
             "duration_s": HORIZON_TICKS, "wall_speed_mps": 8.0}
    run_dir = tmp_path / world["run"]
    run_dir.mkdir(parents=True)
    api, _ = build_fake_session(str(run_dir), world, None, None)
    summary = run_mechanics(world, run_dir, api=api,
                            controller=ScriptedDampingController(load_config()),
                            config=load_config(), horizon_ticks=horizon)
    return world, run_dir, summary


def test_smoke_horizon_cap_real_paths(tmp_path):
    world, run_dir, summary = _run_smoke_window(tmp_path)
    assert summary["ticks_executed"] == SMOKE_HORIZON_TICKS == 70
    events = [e["event"] for e in summary["wall_events"]]
    assert events[0] == "assembly_start"
    assert "assembled" in events  # the A4 closure critical point sits well inside 70 s
    assert summary["totals"]["collision_involvements"] == 0
    checks = evaluate_smoke(world, run_dir,
                            receipt={"exit_status": "complete",
                                     "summary": summary},
                            horizon_ticks=SMOKE_HORIZON_TICKS)
    assert checks["schema"] == "wm43_smoke_checks/1"
    assert checks["ok"], [c for c in checks["checks"] if not c["ok"]]
    assert [c["id"] for c in checks["checks"]] == list(SMOKE_CHECK_IDS)


def test_smoke_checks_fail_on_truncated_window(tmp_path):
    world, run_dir, summary = _run_smoke_window(tmp_path)
    truncated = {"exit_status": "complete",
                 "summary": {**summary, "ticks_executed": 69}}
    checks = evaluate_smoke(world, run_dir, receipt=truncated,
                            horizon_ticks=SMOKE_HORIZON_TICKS)
    assert not checks["ok"]
    failed = [c["id"] for c in checks["checks"] if not c["ok"]]
    assert "S01_smoke_horizon_complete" in failed


# ---------------------------------------------------------------------------
# parent flow: gate refusals, verdict binding, e2e with a real-child stub
# ---------------------------------------------------------------------------

def _verdict_file(tmp_path: Path, registration_path: Path, *, passed=True,
                  registration_sha256=None, package_code_sha256=None,
                  name="verdict.json") -> Path:
    verdict = {
        "schema": "wm43_smoke_verdict/1", "passed": passed,
        "registration_sha256": registration_sha256
        or run_fixture.sha256_file(registration_path),
        "package_code_sha256": package_code_sha256
        or run_fixture.package_code_digest(),
        "smoke_runs_root": str(tmp_path / "smoke"), "run": SMOKE_RUN,
        "horizon_ticks": 70, "created_utc": "2026-09-27T00:00:00+00:00",
    }
    path = tmp_path / name
    path.write_text(json.dumps(verdict), encoding="utf-8")
    return path


def test_launch_gate_refuses_without_verdict(tmp_path):
    plan_path = FIXTURE_DIR / "PLAN.json"
    registration = FIXTURE_DIR / "REGISTRATION_WM43B_DRAFT.md"
    with pytest.raises(run_fixture.RunnerError,
                       match="no pre-launch smoke verdict"):
        run_fixture.launch(plan_path, FIXTURE_DIR / "AUTHORIZATION.json",
                           registration, workers=2, runs_root=tmp_path / "rr",
                           smoke_verdict_path=tmp_path / "absent.json")


def test_launch_gate_refuses_failed_or_stale_verdict(tmp_path):
    plan_path = FIXTURE_DIR / "PLAN.json"
    registration = FIXTURE_DIR / "REGISTRATION_WM43B_DRAFT.md"
    failed = _verdict_file(tmp_path, registration, passed=False,
                           name="v_failed.json")
    with pytest.raises(run_fixture.RunnerError, match="did NOT pass"):
        run_fixture.launch(plan_path, FIXTURE_DIR / "AUTHORIZATION.json",
                           registration, workers=2, runs_root=tmp_path / "rr1",
                           smoke_verdict_path=failed)
    stale_reg = _verdict_file(tmp_path, registration, registration_sha256="0" * 64,
                              name="v_reg.json")
    with pytest.raises(run_fixture.RunnerError, match="registration"):
        run_fixture.launch(plan_path, FIXTURE_DIR / "AUTHORIZATION.json",
                           registration, workers=2, runs_root=tmp_path / "rr2",
                           smoke_verdict_path=stale_reg)
    stale_code = _verdict_file(tmp_path, registration, package_code_sha256="1" * 64,
                               name="v_code.json")
    with pytest.raises(run_fixture.RunnerError, match="package code digest"):
        run_fixture.launch(plan_path, FIXTURE_DIR / "AUTHORIZATION.json",
                           registration, workers=2, runs_root=tmp_path / "rr3",
                           smoke_verdict_path=stale_code)


def test_smoke_parent_flow_end_to_end(tmp_path, monkeypatch):
    """smoke() with an in-process stub that runs the REAL loop + checks."""
    plan_path = FIXTURE_DIR / "PLAN.json"
    registration = FIXTURE_DIR / "REGISTRATION_WM43B_DRAFT.md"
    # the write-once verdict must land in the test sandbox, not the package
    monkeypatch.setattr(run_fixture, "FIXTURE_DIR", tmp_path)
    saved_preflight = run_fixture.preflight
    run_fixture.preflight = lambda *a, **k: {
        "ok": True, "problems": [], "seeds": {"ok": True, "refusals": []}}
    try:
        def real_child_stub(argv):
            argv = [str(a) for a in argv]
            assert "_exec_smoke" in argv
            run_name = argv[argv.index("--run") + 1]
            runs_root = Path(argv[argv.index("--runs-root") + 1])
            horizon = int(argv[argv.index("--horizon-ticks") + 1])
            assert horizon == SMOKE_HORIZON_TICKS == 70
            from traffic_shepherd.config import load_config
            from wm43_loop import run_mechanics
            plan = load_plan(Path(argv[argv.index("--plan") + 1]))
            run = next(r for r in plan["runs"] if r["run"] == run_name)
            run_dir = runs_root / run["run"]
            run_dir.mkdir(parents=True)
            api, _ = build_fake_session(str(run_dir), run, None, None)
            summary = run_mechanics(run, run_dir, api=api,
                                    controller=ScriptedDampingController(
                                        load_config()),
                                    config=load_config(), horizon_ticks=horizon)
            checks = evaluate_smoke(run, run_dir,
                                    receipt={"exit_status": "complete",
                                             "summary": summary},
                                    horizon_ticks=horizon)
            run_fixture.write_json(run_dir / "SMOKE_CHECKS.json", checks)
            run_fixture.write_json(run_dir / run_fixture.RECEIPT_NAME, {
                "schema": run_fixture.RECEIPT_SCHEMA, "run": run["run"],
                "exit_status": "complete", "summary": summary,
                "peak_rss": run_fixture.peak_rss_sample()})
            return 0

        summary = run_fixture.smoke(plan_path, registration,
                                    runs_root=tmp_path / "smoke",
                                    launcher=real_child_stub)
        assert summary["passed"] is True
        assert summary["horizon_ticks"] == 70
        assert summary["simulated_seconds"] == 70
        verdict = json.loads(Path(summary["verdict"]).read_text(encoding="utf-8"))
        assert verdict["passed"] is True
        assert verdict["registration_sha256"] == \
            run_fixture.sha256_file(registration)
        assert verdict["package_code_sha256"] == run_fixture.package_code_digest()
        assert verdict["ticks_executed"] == 70
        allocations = (tmp_path / "smoke" / "allocations.jsonl").read_text(
            encoding="utf-8").strip().splitlines()
        assert len(allocations) == 1
        assert json.loads(allocations[0])["mode"] == "smoke"
        # the launch gate accepts exactly this verdict
        gate = run_fixture.check_smoke_verdict(Path(summary["verdict"]),
                                               registration)
        assert gate["passed"] is True
    finally:
        run_fixture.preflight = saved_preflight
    # write-once discipline: the same smoke root refuses a second smoke
    with pytest.raises(run_fixture.RunnerError, match="already exists"):
        run_fixture.smoke(plan_path, registration,
                          runs_root=tmp_path / "smoke", launcher=lambda a: 0)


def test_no_shipped_smoke_verdict_in_the_draft_package():
    """The draft ships NO SMOKE_VERDICT.json: the gate must refuse a launch
    until the standing smoke has been run against THIS registration."""
    assert not (FIXTURE_DIR / "SMOKE_VERDICT.json").exists()
    assert not (FIXTURE_DIR / "AUTHORIZATION.json").exists()


def test_deferred_plan_survives_delayed_insertion(tmp_path):
    """The A3 fix: with real insertion timing (roster visible from t=2),
    the plan is deferred until the roster is fully live, then the wall
    assembles normally - assembly_start inside the cap, smoke green."""
    from traffic_shepherd.config import load_config
    from wm43_loop import run_mechanics
    world = {"run": SMOKE_RUN, "arm": SMOKE_ARM, "seed": SMOKE_SEED,
             "loading": 2000, "penetration": 1.0, "role": "exposed",
             "duration_s": HORIZON_TICKS, "wall_speed_mps": 8.0}
    run_dir = tmp_path / world["run"]
    run_dir.mkdir(parents=True)
    api, _ = build_fake_session(str(run_dir), world, None, None,
                                insertion_delay_s=2.0)
    summary = run_mechanics(world, run_dir, api=api,
                            controller=ScriptedDampingController(load_config()),
                            config=load_config(), horizon_ticks=70)
    telemetry = [json.loads(line) for line in
                 (run_dir / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    deferred = [r for r in telemetry if r["kind"] == "plan_deferred"]
    submitted = [r for r in telemetry if r["kind"] == "plan_submitted"]
    assert [r["timestamp_s"] for r in deferred] == [0.0, 1.0]
    assert deferred[0]["missing"] == ["wall.0", "wall.1"]
    assert len(submitted) == 1 and submitted[0]["timestamp_s"] == 2.0
    events = [e["event"] for e in summary["wall_events"]]
    # release_at_s=90 lies OUTSIDE the 70 s smoke cap: the smoke window
    # ends in the hold phase (release is a full-run A1 check, not smoke)
    assert events == ["assembly_start", "assembled"]
    assert summary["wall_events"][0]["timestamp_s"] == 2.0
    assert summary["wall_events"][1]["timestamp_s"] <= 70.0
    checks = evaluate_smoke(world, run_dir,
                            receipt={"exit_status": "complete",
                                     "summary": summary},
                            horizon_ticks=70)
    assert checks["ok"], [c for c in checks["checks"] if not c["ok"]]


def test_honest_failure_when_roster_never_becomes_live(tmp_path):
    """If the roster never goes fully live inside the cap, no wall is ever
    staged and the smoke fails honestly at S05/S06 (no silent pass)."""
    from traffic_shepherd.config import load_config
    from wm43_loop import run_mechanics
    world = {"run": SMOKE_RUN, "arm": SMOKE_ARM, "seed": SMOKE_SEED,
             "loading": 2000, "penetration": 1.0, "role": "exposed",
             "duration_s": HORIZON_TICKS, "wall_speed_mps": 8.0}
    run_dir = tmp_path / world["run"]
    run_dir.mkdir(parents=True)
    api, _ = build_fake_session(str(run_dir), world, None, None,
                                insertion_delay_s=500.0)
    summary = run_mechanics(world, run_dir, api=api,
                            controller=ScriptedDampingController(load_config()),
                            config=load_config(), horizon_ticks=70)
    assert summary["wall_events"] == []
    checks = evaluate_smoke(world, run_dir,
                            receipt={"exit_status": "complete",
                                     "summary": summary},
                            horizon_ticks=70)
    assert not checks["ok"]
    assert {c["id"] for c in checks["checks"] if not c["ok"]} ==         {"S05_smoke_assembly_started", "S06_smoke_assembled_by_cap"}
