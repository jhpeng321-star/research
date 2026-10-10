"""WM-59 WALL-2: the pre-launch smoke (registered code; ZERO SUMO here).

The integrator RUNS this in a no-campaign window: restore ONE p40 pilot
world's base saved state, request a ``W-mid-early`` wall, run at most 75
control ticks (<= the 80 simulated-s budget).  PASS requires:

1. the request was made from reports (a ``wall_request`` row sourced
   "cav_report_frames" whose wall speed equals factor x the members' mean
   reported speed, floored);
2. the event log shows assembly complete OR a logged abort WITH a reason;
3. members were locked in ``dispatcher_rows`` during assembly/hold
   (action wall-v2, locked=true, wall id set, phase assembly/hold);
4. non-members were never commanded by the wall (no owner "wall" command
   outside the member set; no non-member row locked);
5. the one-loadState invariant held (the receipt records exactly one
   ``loadState`` call).

The verdict is WRITE-ONCE (``SMOKE_VERDICT.json``) and bound to the
current registration sha256 AND the package code digest - the launch
refuses without a passed, digest-current verdict (any registration or
code edit forces a re-smoke).  Smoke runs are never data.

``--dry-run`` (WALL-2 audit addition): the read-only pre-flight.  It
validates every smoke input (plan schema, smoke selection, branch wall
fields, registration, the base saved state, the horizon budget, write-once
refusals) and prints the exact child command and the exact SUMO command
the smoke WOULD run - without creating a directory or starting a process.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):  # script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from campaign_stage1_v3.common import (  # noqa: E402 - bootstrap above
    load_json,
    sha256_file,
    utc_now,
    write_json,
)
from campaign_walls_v1.common import (  # noqa: E402
    DISK_PREFLIGHT_MIN_FREE_GB,
    PACKAGE,
    PLAN_PATH,
    REGISTRATION_PATH,
    SMOKE_ARM,
    SMOKE_HORIZON_TICKS,
    SMOKE_MAX_SIMULATED_S,
    SMOKE_OUT_DIR,
    SMOKE_RUNS_ROOT,
    SPEED_FLOOR_MPS,
    WallsPilotError,
    package_code_digest,
)
from campaign_walls_v1.wall_driver import WALL_ACTION  # noqa: E402

SMOKE_VERDICT_SCHEMA = "wm59_walls_pilot_smoke/1"
SMOKE_VERDICT_NAME = "SMOKE_VERDICT.json"
SMOKE_CHECKS_NAME = "SMOKE_CHECKS.json"
SMOKE_SUMMARY_NAME = "SMOKE_SUMMARY.json"


class SmokeError(WallsPilotError):
    """A wall-pilot smoke contract violation."""


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_smoke_checks(run_dir: Path, *, members: list[str],
                          speed_factor: float, wall_speed_mps: float,
                          request_s: float | None,
                          load_state_calls: int) -> dict:
    """The five registered smoke checks, parsed from the run artifacts."""
    run_dir = Path(run_dir)
    member_set = {str(v) for v in members}
    events = _read_jsonl(run_dir / "run" / "wall_events.jsonl")
    dispatcher_rows = _read_jsonl(run_dir / "run" / "dispatcher_rows.jsonl")
    commands = _read_jsonl(run_dir / "run" / "suffix_commands.jsonl")
    checks: list[dict] = []

    def _add(check_id: str, ok: bool, detail: str) -> None:
        checks.append({"id": check_id, "ok": bool(ok), "detail": detail})

    # 1. the request came from reports, with the declared speed rule
    requests = [e for e in events if e.get("kind") == "wall_request"]
    request = requests[0] if len(requests) == 1 else None
    speed_rule_ok = False
    if request is not None:
        mean = float(request.get("mean_member_speed_mps") or 0.0)
        expected = max(SPEED_FLOOR_MPS, float(speed_factor) * mean)
        speed_rule_ok = (
            request.get("source") == "cav_report_frames"
            and set(request.get("members") or []) == member_set
            and abs(float(request.get("wall_speed_mps") or -1.0)
                    - float(wall_speed_mps)) <= 1e-6
            and abs(float(request.get("wall_speed_mps") or -1.0) - expected)
            <= 1e-6
            and request.get("lanes") and len(set(request["lanes"])) >= 2
            and float(request.get("span_m") or 1e9) <= 75.0)
    _add("request_from_reports", request is not None and speed_rule_ok,
         "wall_request row sourced cav_report_frames with members/lanes/"
         "span and wall speed = factor x mean reported speed (floor 2.5)"
         if request is not None else "no unique wall_request row")

    # 2. assembly complete OR a logged abort with a reason
    assembled = [e for e in events if e.get("event") == "assembled"]
    aborts = [e for e in events if e.get("event") == "safety_abort"]
    terminal_ok = bool(assembled) or (
        bool(aborts)
        and all(str(e.get("reason") or "").strip() for e in aborts))
    _add("assembly_complete_or_logged_abort", terminal_ok,
         f"assembled={len(assembled)} safety_abort={len(aborts)} "
         "(every abort carries a reason)")

    # 3. members locked in dispatcher_rows during assembly/hold
    window_rows = [r for r in dispatcher_rows
                   if str(r.get("vehicle_id")) in member_set
                   and (request_s is None
                        or float(r["timestamp_s"]) >= float(request_s) - 1e-9)]
    locked_rows = [r for r in window_rows
                   if r.get("action") == WALL_ACTION and r.get("locked") is True
                   and r.get("wall_id")
                   and r.get("wall_phase") in ("assembly", "hold")]
    unlocked_member_rows = [r for r in window_rows
                            if r.get("locked") is not True]
    _add("members_locked_during_assembly_and_hold",
         bool(locked_rows) and not unlocked_member_rows,
         f"{len(locked_rows)} locked member rows, "
         f"{len(unlocked_member_rows)} unlocked member rows after the "
         "request")

    # 4. non-members were never commanded by the wall, never locked
    wall_commanded = {str(c.get("vehicle_id")) for c in commands
                      if c.get("owner") == "wall"}
    strangers = sorted(wall_commanded - member_set)
    locked_non_members = sorted({
        str(r.get("vehicle_id")) for r in dispatcher_rows
        if str(r.get("vehicle_id")) not in member_set
        and (r.get("locked") is True or r.get("action") == WALL_ACTION)})
    _add("non_members_never_commanded", not strangers and not locked_non_members,
         f"wall-commanded non-members={strangers}, "
         f"locked non-members={locked_non_members}")

    # 5. exactly one loadState in the branch process
    _add("one_load_state", load_state_calls == 1,
         f"load_state_calls={load_state_calls}")

    return {"ok": all(c["ok"] for c in checks), "checks": checks}


def render_sumo_command(*, base_world_dir, seed, copy_dir) -> dict:
    """The exact SUMO command the child's ``branch_sumo_session`` will
    build, rendered WITHOUT starting anything: the same constructor as
    ``traffic_shepherd.simulation.start_sumo``, with the -c config the
    child's REWRITTEN scenario copy will carry (WM-64: the copy rewrites
    retired legacy roots, so the runtime input overrides are computed
    against the rewritten text - never the recorded base cfg).  Read-only
    on the base; the rewritten text is materialized in a throwaway temp
    dir only to reuse the production override logic verbatim."""
    from tempfile import mkdtemp

    from campaign_pilot_v1.common import rewrite_legacy_roots
    from traffic_shepherd.config import load_config
    from traffic_shepherd.research_assets import sumo_input_overrides

    base_cfg = (Path(base_world_dir) / "run" / "scenario"
                / "scenario.sumocfg")
    if not base_cfg.is_file():
        raise SmokeError(
            f"dry-run: the base scenario config is missing: {base_cfg}")
    rewritten, _rewrites = rewrite_legacy_roots(
        base_cfg.read_text(encoding="utf-8"))
    scratch = Path(mkdtemp(prefix="walls_smoke_dryrun_cfg_"))
    try:
        scratch_cfg = scratch / "scenario.sumocfg"
        scratch_cfg.write_text(rewritten, encoding="utf-8")
        overrides = list(sumo_input_overrides(scratch_cfg))
    finally:
        import shutil

        shutil.rmtree(scratch, ignore_errors=True)
    config = load_config()
    command = [
        "sumo", "-c", str((Path(copy_dir) / "scenario.sumocfg").resolve()),
        "--seed", str(int(seed)),
        "--step-length", str(config.physics_dt_s),
        "--end", str(config.episode_seconds + 1),
        "--no-step-log", "true", "--duration-log.disable", "true",
    ]
    command.extend(str(a) for a in overrides)
    return {"sumo_command": command, "base_scenario_cfg": str(base_cfg),
            "input_overrides": overrides}


def dry_run_smoke(plan_path, registration_path, *, runs_root=None,
                  out_dir=None, package_dir=None) -> dict:
    """Read-only pre-flight of the registered smoke: validate every input
    and print the exact commands the smoke WOULD run (the child python
    dispatch and the SUMO command the child will start).  Creates nothing
    in the runs root or out dir; refuses loudly on any input the real run
    would refuse (missing plan/registration/base state, an existing runs
    root, a write-once verdict already present, an over-budget horizon)."""
    from campaign_walls_v1.plan_walls_pilot import PLAN_SCHEMA  # noqa: PLC0415

    plan_path = Path(plan_path)
    registration_path = Path(registration_path)
    runs_root = Path(runs_root) if runs_root is not None else Path(SMOKE_RUNS_ROOT)
    out_dir = Path(out_dir) if out_dir is not None else Path(SMOKE_OUT_DIR)
    package_dir = Path(package_dir) if package_dir is not None else PACKAGE
    if not plan_path.is_file():
        raise SmokeError(f"dry-run: plan not found: {plan_path}")
    plan = load_json(plan_path)
    if plan.get("schema") != PLAN_SCHEMA:
        raise SmokeError(f"dry-run: plan schema {plan.get('schema')!r} != "
                         f"{PLAN_SCHEMA!r}")
    world, branch = select_smoke_run(plan)
    if branch.get("mode") != "restore_wall":
        raise SmokeError(f"dry-run: smoke run {branch['run']!r} mode "
                         f"{branch.get('mode')!r} != 'restore_wall'")
    wall = branch.get("wall") or {}
    missing_wall = [f for f in ("speed_factor", "timing", "hold_s",
                                "speed_floor_mps") if f not in wall]
    if missing_wall or "restores_state" not in branch:
        raise SmokeError(f"dry-run: smoke branch {branch['run']!r} is missing "
                         f"wall fields {missing_wall} or restores_state")
    if not registration_path.is_file():
        raise SmokeError(f"dry-run: registration not found: "
                         f"{registration_path}")
    base_world_dir = Path(world["bases_root"]) / \
        branch["restores_state"]["from_run"]
    state_dir = base_world_dir / "saved_state"
    for name in ("state.xml", "STATE_MANIFEST.json"):
        if not (state_dir / name).is_file():
            raise SmokeError(f"dry-run: base saved state missing: "
                             f"{state_dir / name}")
    horizon_ticks = int(plan.get("smoke", {}).get(
        "horizon_ticks", SMOKE_HORIZON_TICKS))
    if horizon_ticks > SMOKE_HORIZON_TICKS:
        raise SmokeError(f"dry-run: horizon_ticks {horizon_ticks} exceeds the "
                         f"registered {SMOKE_HORIZON_TICKS}-tick (<= "
                         f"{SMOKE_MAX_SIMULATED_S:g} sim-s) smoke budget")
    verdict_path = out_dir / SMOKE_VERDICT_NAME
    if verdict_path.exists():
        raise SmokeError(f"dry-run: the real smoke would refuse: {verdict_path} "
                         "already exists (write-once)")
    if runs_root.exists():
        raise SmokeError(f"dry-run: the real smoke would refuse: runs root "
                         f"already exists: {runs_root}")
    if not runs_root.parent.is_dir():
        raise SmokeError(f"dry-run: smoke runs parent does not exist: "
                         f"{runs_root.parent}")
    run_dir = runs_root / branch["run"]
    # the child copies the scenario to <target>/run/scenario (WM-64) and
    # SUMO opens the copy's sumocfg - render THAT exact path
    copy_dir = run_dir / "run" / "scenario"
    child_argv = [
        sys.executable,
        str((package_dir / "run_wall_branch.py").resolve()),
        "_exec_wall_branch",
        "--run", branch["run"],
        "--plan", str(plan_path.resolve()),
        "--runs-root", str(runs_root),
        "--registration", str(registration_path.resolve()),
        "--bases-root", str(world["bases_root"]),
        "--horizon-ticks", str(horizon_ticks),
    ]
    sumo = render_sumo_command(
        base_world_dir=base_world_dir, seed=world["seed"],
        copy_dir=copy_dir)
    return {
        "dry_run": True,
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "registration": str(registration_path),
        "registration_sha256": sha256_file(registration_path),
        "package_code_sha256": package_code_digest(package_dir),
        "world": world["world"],
        "run": branch["run"],
        "arm": branch["arm"],
        "seed": world["seed"],
        "bases_root": str(world["bases_root"]),
        "restores_state": dict(branch["restores_state"]),
        "horizon_ticks": horizon_ticks,
        "simulated_seconds": horizon_ticks,
        "max_simulated_s": SMOKE_MAX_SIMULATED_S,
        "runs_root": str(runs_root),
        "run_dir_would_be": str(run_dir),
        "out_dir": str(out_dir),
        "verdict_path": str(verdict_path),
        "child_command": child_argv,
        "sumo_command": sumo["sumo_command"],
        "base_scenario_cfg": sumo["base_scenario_cfg"],
        "input_overrides": sumo["input_overrides"],
        "note": ("dry run only: nothing was created or launched; the smoke "
                 "runs are never data"),
    }


def select_smoke_run(plan: dict) -> tuple[dict, dict]:
    """(world, branch) of the registered smoke run from the plan."""
    spec = plan.get("smoke") or {}
    if spec.get("arm") != SMOKE_ARM:
        raise SmokeError(f"smoke arm {spec.get('arm')!r} != {SMOKE_ARM!r}")
    if float(spec.get("max_simulated_s") or 0) > SMOKE_MAX_SIMULATED_S:
        raise SmokeError("smoke budget exceeds the registered 80 sim-s")
    world = next((w for w in plan["worlds"]
                  if w["world"] == spec.get("world")), None)
    if world is None:
        raise SmokeError(f"smoke world {spec.get('world')!r} not in the plan")
    branch = next((b for b in world["branches"]
                   if b["run"] == spec.get("run")), None)
    if branch is None:
        raise SmokeError(f"smoke run {spec.get('run')!r} not in the plan")
    return world, branch


def run_smoke(plan_path: Path, registration_path: Path, *,
              runs_root: Path | str | None = None,
              out_dir: Path | str | None = None,
              launcher=None, package_dir: Path | None = None) -> dict:
    """One capped smoke run through the REAL child path; writes the
    write-once verdict bound to the registration + code digests."""
    plan_path = Path(plan_path)
    registration_path = Path(registration_path)
    runs_root = Path(runs_root) if runs_root is not None else Path(SMOKE_RUNS_ROOT)
    out_dir = Path(out_dir) if out_dir is not None else Path(SMOKE_OUT_DIR)
    package_dir = Path(package_dir) if package_dir is not None else PACKAGE
    plan = load_json(plan_path)
    world, branch = select_smoke_run(plan)
    verdict_path = out_dir / SMOKE_VERDICT_NAME
    if verdict_path.exists():
        raise SmokeError(
            f"smoke refuses: {verdict_path} already exists (write-once; "
            "a stale verdict is superseded only by deleting it consciously, "
            "never by overwriting)")
    if runs_root.exists():
        raise SmokeError(f"smoke runs root already exists: {runs_root}")
    if not runs_root.parent.is_dir():
        raise SmokeError(f"smoke runs parent does not exist: {runs_root.parent}")
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir()
    horizon_ticks = int(plan.get("smoke", {}).get(
        "horizon_ticks", SMOKE_HORIZON_TICKS))
    argv = [
        sys.executable,
        str(package_dir / "run_wall_branch.py"),
        "_exec_wall_branch",
        "--run", branch["run"],
        "--plan", str(plan_path),
        "--runs-root", str(runs_root),
        "--registration", str(registration_path),
        "--bases-root", str(world["bases_root"]),
        "--horizon-ticks", str(horizon_ticks),
    ]
    code = (launcher(argv) if launcher is not None
            else subprocess.run([str(a) for a in argv]).returncode)
    run_dir = runs_root / branch["run"]
    receipt_path = run_dir / "RUN_RECEIPT.json"
    receipt = load_json(receipt_path) if receipt_path.is_file() else None
    summary = (receipt or {}).get("wall_summary") or {}
    wall = branch.get("wall") or {}
    if receipt is not None and receipt.get("exit_status") == "complete":
        checks = evaluate_smoke_checks(
            run_dir,
            members=list(summary.get("members") or []),
            speed_factor=float(wall.get("speed_factor") or 0.0),
            wall_speed_mps=float(summary.get("wall_speed_mps") or 0.0),
            request_s=summary.get("request_s"),
            load_state_calls=int(receipt.get("load_state_calls") or 0),
        )
    else:
        checks = {"ok": False, "checks": [{"id": "child_complete", "ok": False,
                                           "detail": "no complete receipt"}]}
    if run_dir.is_dir():
        write_json(run_dir / SMOKE_CHECKS_NAME, checks)
    passed = bool(code == 0 and receipt is not None
                  and receipt.get("exit_status") == "complete"
                  and checks.get("ok") is True)
    verdict = {
        "schema": SMOKE_VERDICT_SCHEMA,
        "passed": passed,
        "registration_sha256": sha256_file(registration_path),
        "package_code_sha256": package_code_digest(package_dir),
        "smoke_runs_root": str(runs_root),
        "world": world["world"],
        "run": branch["run"],
        "arm": branch["arm"],
        "horizon_ticks": horizon_ticks,
        "simulated_seconds": horizon_ticks,
        "load_state_calls": (receipt or {}).get("load_state_calls"),
        "wall_outcome": summary.get("wall_outcome"),
        "exit_code": code,
        "checks_ok": checks.get("ok"),
        "disk_preflight_min_free_gb_declared": DISK_PREFLIGHT_MIN_FREE_GB,
        "created_utc": utc_now(),
        "note": "standing pre-launch gate; smoke runs are never data",
    }
    write_json(verdict_path, verdict)
    smoke_summary = {
        "schema": "wm59_smoke_summary/1", "created_utc": utc_now(),
        "passed": passed, "exit_code": code,
        "runs_root": str(runs_root), "run": branch["run"],
        "horizon_ticks": horizon_ticks, "verdict": str(verdict_path),
        "checks": checks, "retries": 0,
    }
    write_json(out_dir / SMOKE_SUMMARY_NAME, smoke_summary)
    return smoke_summary


def check_smoke_verdict(verdict_path: Path, registration_path: Path, *,
                        package_dir: Path | None = None) -> dict:
    """The launch gate: a CURRENT, passed, write-once smoke verdict bound
    to (a) the sha256 of the current registration and (b) the package
    code digest.  Refuses loudly otherwise."""
    verdict_path = Path(verdict_path)
    package_dir = Path(package_dir) if package_dir is not None else PACKAGE
    if not verdict_path.is_file():
        raise SmokeError(
            f"launch gate: no pre-launch smoke verdict at {verdict_path}; "
            "run the registered smoke first (smoke_walls.py; one capped "
            "run, never data, zero retries)")
    verdict = load_json(verdict_path)
    if verdict.get("schema") != SMOKE_VERDICT_SCHEMA:
        raise SmokeError(
            f"launch gate: verdict schema {verdict.get('schema')!r} != "
            f"{SMOKE_VERDICT_SCHEMA!r}")
    if verdict.get("passed") is not True:
        raise SmokeError(
            "launch gate: the recorded smoke did NOT pass "
            f"(passed={verdict.get('passed')!r}); fix and re-smoke")
    current_registration = sha256_file(Path(registration_path))
    if verdict.get("registration_sha256") != current_registration:
        raise SmokeError(
            "launch gate: smoke verdict is bound to registration "
            f"{str(verdict.get('registration_sha256'))[:16]}... but the "
            f"current registration is {current_registration[:16]}...; "
            "re-run the smoke after any registration change")
    current_code = package_code_digest(package_dir)
    if verdict.get("package_code_sha256") != current_code:
        raise SmokeError(
            "launch gate: smoke verdict is bound to package code digest "
            f"{str(verdict.get('package_code_sha256'))[:16]}... but the "
            f"current package code digest is {current_code[:16]}...; "
            "re-run the smoke after any package code change")
    return {"verdict": str(verdict_path), "passed": True,
            "smoke_runs_root": verdict.get("smoke_runs_root"),
            "registration_sha256": current_registration,
            "package_code_sha256": current_code,
            "smoked_utc": verdict.get("created_utc")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--registration", type=Path, default=REGISTRATION_PATH)
    parser.add_argument("--runs-root", type=Path, default=Path(SMOKE_RUNS_ROOT))
    parser.add_argument("--out", type=Path, default=Path(SMOKE_OUT_DIR))
    parser.add_argument(
        "--dry-run", action="store_true",
        help="validate the smoke inputs read-only and print the exact child "
             "and SUMO commands the smoke WOULD run; creates and launches "
             "nothing")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(
            dry_run_smoke(args.plan, args.registration,
                          runs_root=args.runs_root, out_dir=args.out),
            indent=1))
        return 0
    summary = run_smoke(args.plan, args.registration,
                        runs_root=args.runs_root, out_dir=args.out)
    print(json.dumps({"passed": summary["passed"], "run": summary["run"],
                      "verdict": summary["verdict"]}))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
