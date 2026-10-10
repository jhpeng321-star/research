"""WM-43B fixture-package tests: mocked TraCI / fake controller only, no SUMO.

Covers: plan shape + budget; registration digest consistency; the seeds
diff dry-applies to the CURRENT seeds.py and gates the 15 planned worlds;
the seeds gate's pending-diff reporting; the scenario builder (determinism,
roster bounds, containment); preflight (everything except the seeds gate);
the authorization gate and launch refusals; and the full mocked-TraCI loop
per arm with the machine-evaluated acceptance checks — plus negative tests
proving the checks fail on corrupted evidence.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from conftest import FIXTURE_DIR, REPO_ROOT  # noqa: F401 - pytest path helper
from fake_sumo import ScriptedDampingController, build_fake_session
from wm43_checks import evaluate as evaluate_checks
from wm43_plan import (ARM_SPECS, ARMS, CHECKS_SCHEMA, DIGEST_PATH, HORIZON_TICKS,
                       PLAN_PATH, RUNS_TOTAL, SEEDS_BY_ARM,
                       SIMULATED_SECONDS_TOTAL, WORKERS_MAX, check_plan_shape,
                       load_plan)
import run_fixture
import wm43_checks


# ---------------------------------------------------------------------------
# plan / registration / seeds
# ---------------------------------------------------------------------------

def test_plan_shape_and_budget():
    plan = load_plan(PLAN_PATH)
    assert check_plan_shape(plan) == []
    assert len(plan["runs"]) == RUNS_TOTAL == 25
    assert sum(r["duration_s"] for r in plan["runs"]) == SIMULATED_SECONDS_TOTAL == 15000
    arms = {(r["arm"], r["seed"]) for r in plan["runs"]}
    assert len(arms) == 25 and arms == {(a, s) for a in ARMS for s in SEEDS_BY_ARM[a]}
    # the fresh block is contiguous, right after WM43_WALL_MECHANICS_SEEDS
    all_seeds = sorted(s for a in ARMS for s in SEEDS_BY_ARM[a])
    assert all_seeds == list(range(920249, 920274))
    assert plan["constants"]["runs_root_relative"] == "runs/active/wm43b_wall_closure_v1"
    assert plan["constants"]["law_version"] == "wall-v2.1-closure"
    assert plan["constants"]["workers_max"] == WORKERS_MAX == 6
    assert plan["constants"]["retries"] == 0


def test_plan_shape_rejects_corruption():
    plan = load_plan(PLAN_PATH)
    corrupt = json.loads(json.dumps(plan))
    corrupt["runs"][0]["duration_s"] = 601
    assert check_plan_shape(corrupt)
    corrupt2 = json.loads(json.dumps(plan))
    corrupt2["constants"]["workers_max"] = 12
    assert check_plan_shape(corrupt2)


def test_registration_digest_consistency():
    import hashlib
    digest = json.loads(DIGEST_PATH.read_text(encoding="utf-8"))
    reg_sha = hashlib.sha256((FIXTURE_DIR / "REGISTRATION_WM43B_DRAFT.md").read_bytes()).hexdigest()
    diff_sha = hashlib.sha256((FIXTURE_DIR / "seeds_py_wm43b.diff").read_bytes()).hexdigest()
    audit_sha = hashlib.sha256(
        (FIXTURE_DIR / "EXPOSURE_AUDIT_WM43B.json").read_bytes()).hexdigest()
    assert digest["registration_sha256"] == reg_sha
    assert digest["seeds_diff_sha256"] == diff_sha
    assert digest["exposure_audit_sha256"] == audit_sha


def test_seeds_diff_dry_applies_and_gates():
    """The shipped diff applies to the CURRENT seeds.py and gates the plan.

    State-agnostic: once the integrator has applied the diff, the LIVE
    module is audited directly (dry-applying would double-apply and must
    fail); on an unpatched worktree the diff is dry-applied in memory.
    Both branches assert the same registration invariants."""
    live_text = (REPO_ROOT / "traffic_shepherd" / "seeds.py").read_text(encoding="utf-8")
    if "WM43B_WALL_CLOSURE_SEEDS" in live_text:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "wm43_live_seeds", REPO_ROOT / "traffic_shepherd" / "seeds.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        diff_text = (FIXTURE_DIR / "seeds_py_wm43b.diff").read_text(encoding="utf-8")
        patched = run_fixture.apply_unified_diff(live_text, diff_text)
        module = run_fixture._load_seeds_module_from_text("wm43_patched_seeds", patched)
    # fresh block registered, gate accepts exactly the planned cases
    for seed in range(920249, 920274):
        module.require_development_case(seed, 2000, "exposed")
        assert seed in module.WM43B_WALL_CLOSURE_SEEDS
        assert module.WM43B_SEED_LOADING_MAP[seed] == 2000
    with pytest.raises(ValueError):
        module.require_development_case(920249, 2700, "exposed")
    # the previous WM-43 block is untouched and disjoint from the new one
    assert not (set(module.WM43_WALL_MECHANICS_SEEDS) & set(module.WM43B_WALL_CLOSURE_SEEDS))
    # the WM-28 / WM-34 / WM-41 blocks are untouched by the patch
    for name in ("WM28_SEED_LOADING_MAP", "WM34_SEED_LOADING_MAP",
                 "WM41_SEED_LOADING_MAP"):
        assert getattr(module, name)  # present in the patched module
    module.require_development_case(8401, 2700, "exposed")  # WM-41 case intact
    # every planned world is accepted by the patched module
    plan = load_plan(PLAN_PATH)
    for run in plan["runs"]:
        module.require_development_case(run["seed"], run["loading"], run["role"])
    # never a sealed/prospective seed
    assert not (set(module.WM43B_WALL_CLOSURE_SEEDS) & module.SEALED_SEEDS)
    assert not (set(module.WM43B_WALL_CLOSURE_SEEDS)
                & set(module.PROSPECTIVE_EVALUATION_EXCLUDED_SEEDS))


def test_seeds_gate_reports_pending_diff_not_error():
    """On an unpatched worktree the gate fails as PENDING (dry-apply OK)."""
    plan = load_plan(PLAN_PATH)
    gate = run_fixture.seeds_gate(plan)
    if gate["ok"]:
        return  # integrator already applied the diff: registered state is fine
    assert gate["state"] == "pending_diff"
    assert any("PENDING INTEGRATOR ACTION" in r for r in gate["refusals"])


# ---------------------------------------------------------------------------
# scenario builder
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", ["A1", "A2", "A3", "A4", "A5"])
def test_scenario_builder_deterministic_and_bounded(tmp_path, arm):
    from traffic_shepherd.config import load_config
    from wm43_scenario import build_scenario, derive_roster
    config = load_config()
    world = {"run": f"t_{arm.lower()}", "arm": arm, "seed": SEEDS_BY_ARM[arm][0], "loading": 2000,
             "penetration": 1.0, "role": "exposed", "wall_speed_mps": 8.0}
    scene_a, roster_a, manifest_a = build_scenario(tmp_path / "a", world, config)
    scene_b, roster_b, manifest_b = build_scenario(tmp_path / "b", world, config)
    assert scene_a.read_bytes() == scene_b.read_bytes()
    assert manifest_a["roster"] == manifest_b["roster"]
    expected_len = 3 if arm == "A2" else 2
    assert len(roster_a) == expected_len
    members = {v["vehicle_id"]: v for v in manifest_a["roster"]
               if v["vehicle_id"] in ARM_SPECS[arm]["members"]}
    assert len(members) == 2
    positions = [v["depart_pos_m"] for v in members.values()]
    wide = arm in ("A4", "A5")
    if wide:
        # wall-v2.1 wide pairs: BEYOND the 6 m assembly span, inside the 75 m cap
        assert 40.0 <= max(positions) - min(positions) <= 60.0
    else:
        assert max(positions) - min(positions) <= 4.0  # inside the 6 m assembly span
    assert {v["lane_index"] for v in members.values()} == {0, 1}
    assert all(19.0 <= v["depart_speed_mps"] <= 21.0 for v in members.values())
    if arm == "A2":
        follower = next(v for v in manifest_a["roster"] if v["vehicle_id"] == "follow.0")
        rear_member = min(v["depart_pos_m"] for v in members.values())
        assert 70.0 <= rear_member - follower["depart_pos_m"] <= 100.0
        assert 22.0 <= follower["depart_speed_mps"] <= 26.0
    if arm in ("A3", "A5"):
        assert manifest_a["injection"]["member"] == "wall.1"
        front = max(v["depart_pos_m"] for v in members.values())
        injected = members["wall.1"]["depart_pos_m"]
        assert injected == front  # the injection member is always the front member
    # seed variation changes the scenario, derivation is per (arm, seed)
    roster_c = derive_roster(arm, SEEDS_BY_ARM[arm][1])
    assert [v.depart_pos_m for v in roster_c] != [v.depart_pos_m for v in roster_a]
    # containment: only the scenario directory is written inside the run dir
    run_dir = tmp_path / "c"
    build_scenario(run_dir, world, config)
    assert sorted(p.name for p in run_dir.iterdir()) == ["scenario"]
    with pytest.raises(ValueError):
        build_scenario(run_dir, world, config)  # refuses to overwrite


# ---------------------------------------------------------------------------
# preflight / authorization / launch refusals
# ---------------------------------------------------------------------------

def test_preflight_clean_except_pending_seeds(tmp_path):
    report = run_fixture.preflight(PLAN_PATH, runs_root=tmp_path / "x")
    environment = report["environment"]
    assert environment["traci_import"] == "ok"
    assert environment["sumo_binary"]
    assert environment["mixed_assignment_version"] == "mixed_assignment/3"
    assert report["identity"]["model_matches_adapter"] is True
    assert len(report["scenario_routes_sha256"]) == 25
    assert report["runs_parent_ok"] is True  # tmp_path exists; runs/x does not
    if not report["seeds"]["ok"]:
        # the ONLY allowed failure class is the pending seeds diff
        assert report["seeds"]["state"] == "pending_diff"
        assert all("seeds" in p.lower() or "seeds" in p.lower() or "PENDING" in p
                   for p in report["problems"])


def test_authorization_gate(tmp_path):
    import hashlib
    reg = FIXTURE_DIR / "REGISTRATION_WM43B_DRAFT.md"
    reg_sha = hashlib.sha256(reg.read_bytes()).hexdigest()
    good = tmp_path / "AUTHORIZATION.json"
    good.write_text(json.dumps({
        "schema": "wm43_authorization/1", "registration_sha256": reg_sha,
        "authorized_by": "owner-test", "authorized_utc": "2026-09-27T08:00:00+00:00"}),
        encoding="utf-8")
    gate = run_fixture.check_authorization(good, reg)
    assert gate["registration_sha256"] == reg_sha
    stale = tmp_path / "STALE.json"
    stale.write_text(json.dumps({"schema": "wm43_authorization/1",
                                 "registration_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(run_fixture.RunnerError):
        run_fixture.check_authorization(stale, reg)
    with pytest.raises(run_fixture.RunnerError):
        run_fixture.check_authorization(tmp_path / "MISSING.json", reg)


def test_launch_refusals(tmp_path):
    plan_path = PLAN_PATH
    with pytest.raises(run_fixture.RunnerError):
        run_fixture.launch(plan_path, FIXTURE_DIR / "AUTHORIZATION.json",
                           FIXTURE_DIR / "REGISTRATION_WM43B_DRAFT.md", workers=WORKERS_MAX + 1,
                           runs_root=tmp_path / "rr")
    with pytest.raises(run_fixture.RunnerError):
        run_fixture.launch(plan_path, FIXTURE_DIR / "AUTHORIZATION.json",
                           FIXTURE_DIR / "REGISTRATION_WM43B_DRAFT.md", workers=2,
                           runs_root=tmp_path)  # runs root already exists
    def _fail(argv):  # any dispatch attempt is a test failure
        raise AssertionError("no child may be launched in the refusal tests")
    # the fourth refusal is forced (a failed preflight must stop the launch
    # BEFORE any dispatch, whatever the seeds state is on this worktree)
    monkeypatch_run = {"preflight": lambda *a, **k: {
        "ok": False, "problems": ["forced refusal"], "seeds": {"ok": False}}}
    saved = run_fixture.preflight
    run_fixture.preflight = monkeypatch_run["preflight"]
    try:
        with pytest.raises(run_fixture.RunnerError):
            run_fixture.launch(plan_path, FIXTURE_DIR / "AUTHORIZATION.json",
                               FIXTURE_DIR / "REGISTRATION_WM43B_DRAFT.md", workers=2,
                               runs_root=tmp_path / "rr2", launcher=_fail)
    finally:
        run_fixture.preflight = saved


# ---------------------------------------------------------------------------
# full mocked-TraCI loop per arm + checks
# ---------------------------------------------------------------------------

def _world(arm, seed=None, run=None):
    seed = SEEDS_BY_ARM[arm][0] if seed is None else seed
    return {"run": run or f"{arm.lower()}_seed{seed}", "arm": arm, "seed": seed,
            "loading": 2000, "penetration": 1.0, "role": "exposed",
            "duration_s": HORIZON_TICKS, "wall_speed_mps": 8.0}


def _run_arm(arm, tmp_path, seed=None):
    from traffic_shepherd.config import load_config
    from wm43_loop import run_mechanics
    world = _world(arm, seed)
    run_dir = tmp_path / world["run"]
    run_dir.mkdir(parents=True)
    api, _manifest = build_fake_session(run_dir, world, None, load_config())
    controller = ScriptedDampingController(load_config())
    summary = run_mechanics(world, run_dir, api=api, controller=controller,
                            config=load_config())
    receipt = {"exit_status": "complete", "summary": summary}
    checks = evaluate_checks(world, run_dir, receipt=receipt)
    return world, run_dir, summary, checks


def _assert_checks_ok(checks):
    failed = [c for c in checks["checks"] if not c["ok"]]
    for check in failed:
        print(f"FAILED {check['id']}: {check['detail']}")
    assert checks["ok"], f"checks failed: {[c['id'] for c in failed]}"
    assert checks["schema"] == CHECKS_SCHEMA


@pytest.mark.parametrize("arm", ["A1", "A2", "A3", "A4", "A5"])
def test_mocked_loop_arm_checks_pass(arm, tmp_path):
    world, run_dir, summary, checks = _run_arm(arm, tmp_path)
    assert summary["ticks_executed"] == HORIZON_TICKS
    assert summary["totals"]["collision_involvements"] == 0
    assert summary["totals"]["starting_teleports"] == 0
    events = [e["event"] for e in summary["wall_events"]]
    if arm in ("A1", "A2", "A4"):
        assert events == ["assembly_start", "assembled", "released"]
    elif arm == "A3":
        assert events == ["assembly_start", "assembled", "safety_abort"]
        assert summary["scripted_injections"] >= 1
    else:  # A5: closure impossible -> assembly_timeout, never assembled
        assert events == ["assembly_start", "safety_abort"]
        assert summary["scripted_injections"] >= 1
    _assert_checks_ok(checks)


def test_mocked_loop_a2_surveillance_geometry(tmp_path):
    world, run_dir, summary, checks = _run_arm("A2", tmp_path)
    proximity = [json.loads(line) for line in
                 (run_dir / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
                 if '"wall_proximity"' in line]
    assert proximity, "surveillance evidence rows must exist"
    assert all(not p["would_abort"] for p in proximity)
    gaps = [p["rear_gap_m"] for p in proximity]
    assert min(gaps) > 0.0  # the follower never reached the rearmost member


def _rows(run_dir, kind):
    return [json.loads(line) for line in
            (run_dir / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
            if json.loads(line)["kind"] == kind]


def test_a4_wide_pair_closure_evidence(tmp_path):
    """A4 under fakes: the closure law actually closes a 40-60 m pair."""
    world, run_dir, summary, checks = _run_arm("A4", tmp_path)
    _assert_checks_ok(checks)
    events = {e["event"]: e for e in summary["wall_events"]}
    elapsed = events["assembled"]["timestamp_s"] - events["assembly_start"]["timestamp_s"]
    assert 10.0 < elapsed <= 60.0                      # closure took real time
    assert events["assembly_start"]["law_version"] == "wall-v2.1-closure"
    frames = {r["timestamp_s"]: r["members"] for r in _rows(run_dir, "wall_frames")}
    first = frames[events["assembly_start"]["timestamp_s"]]
    span0 = abs(first["wall.1"]["s_m"] - first["wall.0"]["s_m"])
    assert 30.0 <= span0 <= 60.0                       # started WIDE
    last = frames[events["assembled"]["timestamp_s"]]
    assert abs(last["wall.1"]["s_m"] - last["wall.0"]["s_m"]) <= 6.0
    # first-tick rear command is the closure target (v_wall + 2.0 while gap >= 20 m)
    cmds = [r for r in _rows(run_dir, "command") if r["owner"] == "wall"]
    t0 = events["assembly_start"]["timestamp_s"]
    rear0 = next(r for r in cmds if r["timestamp_s"] == t0 and r["vehicle_id"] == "wall.0")
    front0 = next(r for r in cmds if r["timestamp_s"] == t0 and r["vehicle_id"] == "wall.1")
    v_rear, v_front = first["wall.0"]["speed_mps"], first["wall.1"]["speed_mps"]
    assert rear0["issued_acceleration_mps2"] == pytest.approx(
        min(max(10.0 - v_rear, -3.0), 1.5))
    assert front0["issued_acceleration_mps2"] == pytest.approx(
        min(max(8.0 - v_front, -3.0), 1.5))


def test_a5_assembly_timeout_evidence(tmp_path):
    world, run_dir, summary, checks = _run_arm("A5", tmp_path)
    _assert_checks_ok(checks)
    events = {e["event"]: e for e in summary["wall_events"]}
    abort = events["safety_abort"]
    assert abort["reason"] == "assembly_timeout"
    assert abort["timestamp_s"] - events["assembly_start"]["timestamp_s"] == pytest.approx(60.0)
    assert "assembled" not in events and "released" not in events
    injections = _rows(run_dir, "scripted_actuator")
    assert injections and all(r["value_mps"] == 10.5 for r in injections)


def test_a4_a5_checks_fail_on_tampered_evidence(tmp_path):
    world, run_dir, summary, _c = _run_arm("A4", tmp_path)
    # (a) tamper with a REAR assembly command: A1_7 and A4_4 must fail
    tampered = tmp_path / "tamper_rear"
    shutil.copytree(run_dir, tampered)
    tel = tampered / "telemetry.jsonl"
    out, done = [], False
    for line in tel.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if (not done and row["kind"] == "command" and row["owner"] == "wall"
                and row["vehicle_id"] == "wall.0"):
            row["issued_acceleration_mps2"] += 0.5    # off the closure law
            done = True
        out.append(json.dumps(row))
    tel.write_text("\n".join(out) + "\n", encoding="utf-8")
    checks = wm43_checks.evaluate(world, tampered, receipt={"exit_status": "complete"})
    failed = {c["id"] for c in checks["checks"] if not c["ok"]}
    assert "A1_7_wall_pacing_identity" in failed and "A4_4_rear_closure_identity" in failed
    # (b) old-law law_version: C07 must fail
    old = tmp_path / "old_law"
    shutil.copytree(run_dir, old)
    tel = old / "telemetry.jsonl"
    tel.write_text(tel.read_text(encoding="utf-8").replace("wall-v2.1-closure", "wall-v2"),
                   encoding="utf-8")
    checks = wm43_checks.evaluate(world, old, receipt={"exit_status": "complete"})
    assert "C07_law_version_identity" in {c["id"] for c in checks["checks"] if not c["ok"]}
    # (c) A5: an abort with the wrong reason fails A5_2
    w5, r5, _s5, _c5 = _run_arm("A5", tmp_path)
    bad5 = tmp_path / "bad_a5"
    shutil.copytree(r5, bad5)
    tel5 = bad5 / "telemetry.jsonl"
    tel5.write_text(tel5.read_text(encoding="utf-8").replace(
        '"reason": "assembly_timeout"', '"reason": "member_span_exceeded"'), encoding="utf-8")
    checks5 = wm43_checks.evaluate(w5, bad5, receipt={"exit_status": "complete"})
    assert "A5_2_abort_reason_assembly_timeout" in {
        c["id"] for c in checks5["checks"] if not c["ok"]}


def test_checks_fail_on_corrupted_evidence(tmp_path):
    world, run_dir, summary, _checks = _run_arm("A1", tmp_path)
    # (a) duplicate ownership write: C04 must fail
    corrupted = tmp_path / "corrupted_dup"
    shutil.copytree(run_dir, corrupted)
    telemetry = corrupted / "telemetry.jsonl"
    rows = telemetry.read_text(encoding="utf-8").splitlines()
    first_command = next(json.loads(r) for r in rows if '"command"' in r)
    telemetry.write_text("\n".join(rows + [json.dumps(dict(first_command))]) + "\n",
                         encoding="utf-8")
    checks = wm43_checks.evaluate(world, corrupted, receipt={"exit_status": "complete"})
    assert not checks["ok"]
    # (b) missing released event: A1_4 must fail
    corrupted2 = tmp_path / "corrupted_release"
    shutil.copytree(run_dir, corrupted2)
    telemetry2 = corrupted2 / "telemetry.jsonl"
    kept = [line for line in telemetry2.read_text(encoding="utf-8").splitlines()
            if '"released"' not in line]
    telemetry2.write_text("\n".join(kept) + "\n", encoding="utf-8")
    checks2 = wm43_checks.evaluate(world, corrupted2,
                                   receipt={"exit_status": "complete"})
    assert not checks2["ok"]
    ids = {c["id"] for c in checks2["checks"] if not c["ok"]}
    assert "A1_4_released_event" in ids


def test_launch_end_to_end_with_stub(tmp_path, monkeypatch):
    """Full launch() flow with a stubbed child dispatch (no SUMO, no real runs).

    The stub writes a minimal complete receipt per run, so the parent's
    accounting path (allocations BEFORE dispatch, summary, budget rows) runs
    without simulation.  Only 2 runs are injected into a shrunk plan copy to
    keep this cheap; the real launch always carries the 25-run plan.
    """
    plan = load_plan(PLAN_PATH)
    small = dict(plan)
    small["runs"] = plan["runs"][:2]
    small["constants"] = dict(plan["constants"],
                              runs_root_relative=str(tmp_path / "runs" / "wm40"))
    (tmp_path / "runs").mkdir()  # the runner creates only the leaf directory
    small_path = tmp_path / "PLAN_SMALL.json"
    small_path.write_text(json.dumps(small), encoding="utf-8")

    import hashlib
    reg = FIXTURE_DIR / "REGISTRATION_WM43B_DRAFT.md"
    auth = tmp_path / "AUTHORIZATION.json"
    auth.write_text(json.dumps({
        "schema": "wm43_authorization/1",
        "registration_sha256": hashlib.sha256(reg.read_bytes()).hexdigest(),
        "authorized_by": "owner-test", "authorized_utc": "2026-09-27T08:00:00+00:00"}),
        encoding="utf-8")
    # the launch smoke gate requires a CURRENT passed verdict bound to the
    # registration sha256 and the package code digest (build one for the stub)
    verdict = tmp_path / "SMOKE_VERDICT.json"
    verdict.write_text(json.dumps({
        "schema": "wm43_smoke_verdict/1", "passed": True,
        "registration_sha256": hashlib.sha256(reg.read_bytes()).hexdigest(),
        "package_code_sha256": run_fixture.package_code_digest(),
        "smoke_runs_root": str(tmp_path / "runs" / "smoke"),
        "horizon_ticks": 70, "created_utc": "2026-09-27T00:00:00+00:00"}),
        encoding="utf-8")

    from wm43_plan import RECEIPT_SCHEMA  # the stub writes the receipt the child would

    # the parent's budget guard and seeds gate are patched for the shrunk
    # plan copy (the real launch always carries the 25-run plan and the live
    # seeds gate; both guards are exercised unpatched in the other tests)
    monkeypatch.setattr(run_fixture, "RUNS_TOTAL", 2)
    monkeypatch.setattr(run_fixture, "preflight", lambda *a, **k: {
        "ok": True, "problems": [], "seeds": {"ok": True, "refusals": []}})
    monkeypatch.setattr(run_fixture, "seeds_gate",
                        lambda plan: {"ok": True, "state": "registered_in_module",
                                      "refusals": []})

    def stub_child(argv):
        run = argv[argv.index("--run") + 1]
        runs_root = Path(argv[argv.index("--runs-root") + 1])
        target = runs_root / run
        target.mkdir(parents=True, exist_ok=False)
        (target / "RUN_RECEIPT.json").write_text(json.dumps(
            {"schema": RECEIPT_SCHEMA, "run": run, "exit_status": "complete"}),
            encoding="utf-8")
        return 0

    summary = run_fixture.launch(small_path, auth, reg, workers=2,
                                 runs_root=tmp_path / "runs" / "wm40",
                                 launcher=stub_child,
                                 smoke_verdict_path=verdict)
    assert summary["failed"] == 0 and summary["complete"] == 2
    assert summary["smoke_gate"]["passed"] is True
    allocations = (tmp_path / "runs" / "wm40" / "allocations.jsonl").read_text(
        encoding="utf-8").strip().splitlines()
    assert len(allocations) == 2
    assert (tmp_path / "runs" / "wm40" / "LAUNCH_SUMMARY.json").is_file()
    with pytest.raises(run_fixture.RunnerError):
        run_fixture.launch(small_path, auth, reg, workers=2,
                           runs_root=tmp_path / "runs" / "wm40",
                           launcher=stub_child,
                           smoke_verdict_path=verdict)  # existing root: never overwrite
