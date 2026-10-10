"""The registered smoke: checks, verdict write-once, digest binding."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from campaign_walls_v1 import smoke_walls
from campaign_walls_v1.common import package_code_digest
from campaign_walls_v1.wall_driver import WALL_ACTION
from wall_fakes import MEMBERS, T0


def _events_dir(run_dir: Path, *, request=True, assembled=True,
                abort_reason=None, locked=True, member_rows=True,
                wall_commands=True, handback=True):
    run = run_dir / "run"
    run.mkdir(parents=True, exist_ok=True)
    events = []
    if request:
        events.append({
            "kind": "wall_request", "wall_id": "w:wall1",
            "timestamp_s": T0, "timing": "early", "target_time_s": T0,
            "deadline_s": T0 + 120.0, "edge": "1191554636",
            "members": list(MEMBERS), "lanes": [0, 1], "span_m": 40.0,
            "speed_factor": 0.85, "mean_member_speed_mps": 20.0,
            "wall_speed_mps": 17.0, "hold_s": 60.0,
            "source": "cav_report_frames"})
    events.append({"kind": "wall_event", "wall_id": "w:wall1",
                   "event": "assembly_start", "timestamp_s": T0})
    if assembled:
        events.append({"kind": "wall_event", "wall_id": "w:wall1",
                       "event": "assembled", "timestamp_s": T0 + 2,
                       "members": list(MEMBERS), "wall_speed_mps": 17.0})
    if abort_reason is not None:
        events.append({"kind": "wall_event", "wall_id": "w:wall1",
                       "event": "safety_abort", "timestamp_s": T0 + 4,
                       "reason": abort_reason, "handback": "native"})
    if handback and (assembled or abort_reason):
        events.append({"kind": "wall_handback", "wall_id": "w:wall1",
                       "timestamp_s": T0 + 6, "how":
                           "released" if assembled else "safety_abort",
                       "members": list(MEMBERS), "handback": "native"})
    (run / "wall_events.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    rows = []
    for t in (T0 + 0.5, T0 + 1.0, T0 + 2.0):
        for vid in MEMBERS:
            rows.append({
                "timestamp_s": t, "plan_id": "w:wall@300", "vehicle_id": vid,
                "action": WALL_ACTION if locked and member_rows else "native",
                "wall_id": "w:wall1" if locked and member_rows else None,
                "wall_phase": "assembly" if locked and member_rows else None,
                "time_in_phase_s": 0.0 if locked and member_rows else None,
                "wall_speed_mps": 17.0 if locked and member_rows else None,
                "locked": bool(locked and member_rows),
                "controller_state": {"damping_target_speed_mps": None,
                                     "pi_running_target_mps": None}})
        rows.append({
            "timestamp_s": t, "plan_id": "w:wall@300", "vehicle_id": "cav.0",
            "action": "native", "wall_id": None, "wall_phase": None,
            "time_in_phase_s": None, "wall_speed_mps": None, "locked": False,
            "controller_state": {"damping_target_speed_mps": None,
                                 "pi_running_target_mps": None}})
    (run / "dispatcher_rows.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    commands = []
    if wall_commands:
        for vid in MEMBERS:
            commands.append({"kind": "command", "timestamp_s": T0,
                             "vehicle_id": vid, "owner": "wall",
                             "issued_target_speed_mps": 16.8,
                             "fallback_reason": None})
    commands.append({"kind": "command", "timestamp_s": T0,
                     "vehicle_id": "cav.0", "owner": "native",
                     "issued_target_speed_mps": None,
                     "fallback_reason": "selected_native"})
    (run / "suffix_commands.jsonl").write_text(
        "".join(json.dumps(c) + "\n" for c in commands), encoding="utf-8")
    return run_dir


def test_smoke_checks_pass_on_a_green_run(tmp_path):
    run_dir = _events_dir(tmp_path / "run1")
    checks = smoke_walls.evaluate_smoke_checks(
        run_dir, members=list(MEMBERS), speed_factor=0.85,
        wall_speed_mps=17.0, request_s=T0, load_state_calls=1)
    assert checks["ok"] is True, checks
    ids = {c["id"] for c in checks["checks"]}
    assert ids == {"request_from_reports", "assembly_complete_or_logged_abort",
                   "members_locked_during_assembly_and_hold",
                   "non_members_never_commanded", "one_load_state"}


def test_smoke_checks_accept_a_logged_abort_instead_of_assembly(tmp_path):
    run_dir = _events_dir(tmp_path / "run2", assembled=False,
                          abort_reason="member_span_exceeded")
    checks = smoke_walls.evaluate_smoke_checks(
        run_dir, members=list(MEMBERS), speed_factor=0.85,
        wall_speed_mps=17.0, request_s=T0, load_state_calls=1)
    assert checks["ok"] is True


@pytest.mark.parametrize("mutate, failing", [
    ({"request": False}, "request_from_reports"),
    ({"assembled": False, "abort_reason": ""}, "assembly_complete_or_logged_abort"),
    ({"locked": False}, "members_locked_during_assembly_and_hold"),
    ({"member_rows": False}, "members_locked_during_assembly_and_hold"),
])
def test_smoke_check_failure_modes(tmp_path, mutate, failing):
    run_dir = _events_dir(tmp_path / "run3", **mutate)
    checks = smoke_walls.evaluate_smoke_checks(
        run_dir, members=list(MEMBERS), speed_factor=0.85,
        wall_speed_mps=17.0, request_s=T0, load_state_calls=1)
    assert checks["ok"] is False
    assert any(c["id"] == failing and not c["ok"] for c in checks["checks"])


def test_wall_command_to_a_non_member_fails(tmp_path):
    run_dir = _events_dir(tmp_path / "run4")
    commands_path = run_dir / "run" / "suffix_commands.jsonl"
    rows = [json.loads(line) for line in
            commands_path.read_text(encoding="utf-8").splitlines() if line]
    rows.append({"kind": "command", "timestamp_s": T0, "vehicle_id": "cav.0",
                 "owner": "wall", "issued_target_speed_mps": 10.0,
                 "fallback_reason": None})
    commands_path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    checks = smoke_walls.evaluate_smoke_checks(
        run_dir, members=list(MEMBERS), speed_factor=0.85,
        wall_speed_mps=17.0, request_s=T0, load_state_calls=1)
    assert checks["ok"] is False
    assert any(c["id"] == "non_members_never_commanded" and not c["ok"]
               for c in checks["checks"])


def test_two_load_states_fail_the_smoke(tmp_path):
    run_dir = _events_dir(tmp_path / "run5")
    checks = smoke_walls.evaluate_smoke_checks(
        run_dir, members=list(MEMBERS), speed_factor=0.85,
        wall_speed_mps=17.0, request_s=T0, load_state_calls=2)
    assert checks["ok"] is False
    assert any(c["id"] == "one_load_state" and not c["ok"]
               for c in checks["checks"])


def test_run_smoke_writes_write_once_bound_verdict(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    plan_dir = tmp_path / "results_plan"
    plan_dir.mkdir()
    plan = {"schema": "wm59_walls_pilot_world_plan/1", "smoke": {
        "arm": "W-mid-early", "penetration": 0.4, "world": "w40",
        "run": "w40__w-mid-early", "max_simulated_s": 80.0,
        "horizon_ticks": 75, "bases_root": str(tmp_path / "bases"),
        "restores_state": {"from_run": "w40__base", "at_s": T0}},
        "worlds": [{"world": "w40", "penetration": 0.4, "t0_s": T0,
                    "bases_root": str(tmp_path / "bases"),
                    "base_run": "w40__base",
                    "branches": [{
                        "arm": "W-mid-early", "run": "w40__w-mid-early",
                        "wall": {"speed_factor": 0.85, "timing": "early",
                                 "target_time_s": T0, "request_window_s": 120.0,
                                 "deadline_s": T0 + 120.0,
                                 "speed_floor_mps": 2.5, "hold_s": 60.0}}]}],
        "runs": []}
    plan_path = plan_dir / "WORLD_PLAN.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    registration = tmp_path / "REGISTRATION.md"
    registration.write_text("# registration\n", encoding="utf-8")

    def fake_child(argv):
        # parse the dispatch and fabricate a PASSING smoke run
        run_name = argv[argv.index("--run") + 1]
        runs_root = Path(argv[argv.index("--runs-root") + 1])
        run_dir = runs_root / run_name
        run_dir.mkdir(parents=True)
        _events_dir(run_dir)
        (run_dir / "RUN_RECEIPT.json").write_text(json.dumps({
            "schema": "wm59_wall_branch_receipt/1", "exit_status": "complete",
            "load_state_calls": 1,
            "wall_summary": {"wall_outcome": "assembled_released",
                             "members": list(MEMBERS),
                             "wall_speed_mps": 17.0,
                             "request_s": T0}}), encoding="utf-8")
        return 0

    runs_root = tmp_path / "smoke_runs"
    out_dir = tmp_path / "smoke_out"
    summary = smoke_walls.run_smoke(
        plan_path, registration, runs_root=runs_root, out_dir=out_dir,
        launcher=fake_child, package_dir=pkg)
    assert summary["passed"] is True
    verdict = json.loads(
        (out_dir / "SMOKE_VERDICT.json").read_text(encoding="utf-8"))
    assert verdict["schema"] == "wm59_walls_pilot_smoke/1"
    assert verdict["passed"] is True
    import hashlib

    assert verdict["registration_sha256"] == hashlib.sha256(
        registration.read_bytes()).hexdigest()
    assert verdict["package_code_sha256"] == package_code_digest(pkg)
    # write-once: a second smoke refuses
    with pytest.raises(smoke_walls.SmokeError, match="write-once"):
        smoke_walls.run_smoke(plan_path, registration, runs_root=runs_root,
                              out_dir=out_dir, launcher=fake_child,
                              package_dir=pkg)
    # the verdict passes the launch gate
    gate = smoke_walls.check_smoke_verdict(out_dir / "SMOKE_VERDICT.json",
                                           registration, package_dir=pkg)
    assert gate["passed"] is True


def _verdict(registration: Path, pkg: Path, **overrides):
    import hashlib

    payload = {
        "schema": "wm59_walls_pilot_smoke/1", "passed": True,
        "registration_sha256": hashlib.sha256(
            registration.read_bytes()).hexdigest(),
        "package_code_sha256": package_code_digest(pkg),
        "smoke_runs_root": "E:/x", "created_utc": "utc"}
    payload.update(overrides)
    return payload


def test_launch_gate_refuses_missing_failed_and_stale_verdicts(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    registration = tmp_path / "REGISTRATION.md"
    registration.write_text("# registration\n", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    verdict_path = out / "SMOKE_VERDICT.json"
    with pytest.raises(smoke_walls.SmokeError, match="no pre-launch smoke"):
        smoke_walls.check_smoke_verdict(verdict_path, registration,
                                        package_dir=pkg)
    verdict_path.write_text(json.dumps(_verdict(registration, pkg, passed=False)),
                            encoding="utf-8")
    with pytest.raises(smoke_walls.SmokeError, match="did NOT pass"):
        smoke_walls.check_smoke_verdict(verdict_path, registration,
                                        package_dir=pkg)
    verdict_path.write_text(json.dumps(
        _verdict(registration, pkg, registration_sha256="0" * 64)),
        encoding="utf-8")
    with pytest.raises(smoke_walls.SmokeError, match="re-run the smoke"):
        smoke_walls.check_smoke_verdict(verdict_path, registration,
                                        package_dir=pkg)
    verdict_path.write_text(json.dumps(
        _verdict(registration, pkg, package_code_sha256="1" * 64)),
        encoding="utf-8")
    with pytest.raises(smoke_walls.SmokeError, match="re-run the smoke"):
        smoke_walls.check_smoke_verdict(verdict_path, registration,
                                        package_dir=pkg)
    verdict_path.write_text(json.dumps(_verdict(registration, pkg, schema="x/9")),
                            encoding="utf-8")
    with pytest.raises(smoke_walls.SmokeError, match="schema"):
        smoke_walls.check_smoke_verdict(verdict_path, registration,
                                        package_dir=pkg)


# ---- the WALL-2 audit addition: the --dry-run pre-flight ----

def _fresh_tmp():
    import tempfile

    return Path(tempfile.mkdtemp(prefix="walls_smoke_dryrun_test_"))


def _dry_run_plan(tmp_path, *, horizon_ticks=75):
    """A plan whose smoke branch carries the full wall shape, plus the
    fake base saved state and scenario config the child restores from."""
    from wall_fakes import base_saved_state

    bases_root = tmp_path / "bases"
    base_saved_state(bases_root / "w40__base", T0)
    scenario_dir = bases_root / "w40__base" / "run" / "scenario"
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "scenario.sumocfg").write_text(
        '<configuration>\n  <input>\n'
        '    <net-file value="net.net.xml"/>\n'
        '    <route-files value="requests.rou.xml"/>\n'
        '  </input>\n</configuration>\n', encoding="utf-8")
    plan = {
        "schema": "wm59_walls_pilot_world_plan/1",
        "smoke": {
            "arm": "W-mid-early", "penetration": 0.4, "world": "w40",
            "run": "w40__w-mid-early", "max_simulated_s": 80.0,
            "horizon_ticks": horizon_ticks, "bases_root": str(bases_root),
            "restores_state": {"from_run": "w40__base", "at_s": T0}},
        "worlds": [{"world": "w40", "penetration": 0.4, "seed": 300,
                    "t0_s": T0, "bases_root": str(bases_root),
                    "base_run": "w40__base",
                    "branches": [{
                        "arm": "W-mid-early", "run": "w40__w-mid-early",
                        "mode": "restore_wall",
                        "restores_state": {"from_run": "w40__base",
                                           "at_s": T0},
                        "wall": {"speed_factor": 0.85, "timing": "early",
                                 "target_time_s": T0,
                                 "request_window_s": 120.0,
                                 "deadline_s": T0 + 120.0,
                                 "speed_floor_mps": 2.5,
                                 "hold_s": 60.0}}]}],
        "runs": []}
    plan_dir = tmp_path / "results_plan"
    plan_dir.mkdir()
    plan_path = plan_dir / "WORLD_PLAN.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    registration = tmp_path / "WALLS_PILOT_REGISTRATION.md"
    registration.write_text("# registration\n", encoding="utf-8")
    return plan_path, registration, tmp_path / "smoke_runs", tmp_path / "smoke_out"


def test_dry_run_prints_child_and_sumo_commands_creating_nothing(capsys):
    plan_path, registration, runs_root, out_dir = _dry_run_plan(_fresh_tmp())
    code = smoke_walls.main([
        "--dry-run", "--plan", str(plan_path),
        "--registration", str(registration),
        "--runs-root", str(runs_root), "--out", str(out_dir)])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["run"] == "w40__w-mid-early"
    assert payload["seed"] == 300
    assert payload["horizon_ticks"] == 75
    assert payload["simulated_seconds"] == 75 <= payload["max_simulated_s"]
    # the exact child dispatch the smoke would launch
    child = payload["child_command"]
    assert child[1].endswith("run_wall_branch.py")
    assert child[2:4] == ["_exec_wall_branch", "--run"]
    assert child[4] == "w40__w-mid-early"
    assert child[child.index("--horizon-ticks") + 1] == "75"
    assert child[child.index("--bases-root") + 1].endswith("bases")
    # the exact SUMO command the child's branch_sumo_session would start
    sumo = payload["sumo_command"]
    assert sumo[0] == "sumo"
    assert sumo[sumo.index("--seed") + 1] == "300"
    cfg = sumo[sumo.index("-c") + 1]
    assert cfg.endswith("scenario.sumocfg")
    # the child opens the REWRITTEN COPY: <runs_root>/<run>/run/scenario
    assert str(Path("w40__w-mid-early") / "run" / "scenario") in cfg
    assert payload["input_overrides"] == []
    # READ-ONLY: nothing was created anywhere
    assert not runs_root.exists()
    assert not out_dir.exists()
    assert not (runs_root / "w40__w-mid-early").exists()


def test_dry_run_refuses_missing_base_saved_state():
    tmp = _fresh_tmp()
    plan_path, registration, runs_root, out_dir = _dry_run_plan(tmp)
    state_xml = (tmp / "bases" / "w40__base" / "saved_state" / "state.xml")
    state_xml.unlink()
    with pytest.raises(smoke_walls.SmokeError, match="base saved state"):
        smoke_walls.main([
            "--dry-run", "--plan", str(plan_path),
            "--registration", str(registration),
            "--runs-root", str(runs_root), "--out", str(out_dir)])


def test_dry_run_refuses_existing_runs_root_and_verdict():
    tmp = _fresh_tmp()
    plan_path, registration, runs_root, out_dir = _dry_run_plan(tmp)
    runs_root.mkdir(parents=True)
    with pytest.raises(smoke_walls.SmokeError, match="already exists"):
        smoke_walls.main([
            "--dry-run", "--plan", str(plan_path),
            "--registration", str(registration),
            "--runs-root", str(runs_root), "--out", str(out_dir)])
    runs_root.rmdir()
    out_dir.mkdir(parents=True)
    (out_dir / "SMOKE_VERDICT.json").write_text("{}", encoding="utf-8")
    with pytest.raises(smoke_walls.SmokeError, match="write-once"):
        smoke_walls.main([
            "--dry-run", "--plan", str(plan_path),
            "--registration", str(registration),
            "--runs-root", str(runs_root), "--out", str(out_dir)])


def test_dry_run_refuses_missing_registration_and_bad_horizon():
    tmp = _fresh_tmp()
    plan_path, registration, runs_root, out_dir = _dry_run_plan(tmp)
    registration.unlink()
    with pytest.raises(smoke_walls.SmokeError, match="registration"):
        smoke_walls.main([
            "--dry-run", "--plan", str(plan_path),
            "--registration", str(registration),
            "--runs-root", str(runs_root), "--out", str(out_dir)])
    tmp2 = _fresh_tmp()
    plan_path, registration, runs_root, out_dir = _dry_run_plan(
        tmp2, horizon_ticks=100)
    with pytest.raises(smoke_walls.SmokeError, match="budget"):
        smoke_walls.main([
            "--dry-run", "--plan", str(plan_path),
            "--registration", str(registration),
            "--runs-root", str(runs_root), "--out", str(out_dir)])
