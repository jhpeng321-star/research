"""WM-59 WALL-2: the smoke-gated, authorization-gated launch (WALL-1 runs).

Gate order (every refusal is loud, zero retries):

1. ``--workers`` is EXACTLY the registered 10;
2. the smoke gate: a passed, write-once ``SMOKE_VERDICT.json`` bound to
   the current registration sha256 AND the package code digest (a
   missing, failed, or stale verdict refuses - any registration or code
   edit forces a re-smoke);
3. the owner authorization (AUTHORIZATION.json carrying the exact
   registration sha256);
4. the WM-41/WM-43 stage gates (the paths pinned in the registration);
5. the plan gates (schema, 240 runs, 144,000 simulated s);
6. the 45 GB disk preflight on the drive the runs root RESOLVES to;
7. a NEW runs root (an existing one refuses; ``--resume`` archives the
   write-once stage outputs and dispatches only the missing runs).

Dispatch: the 240 branch children (``run_wall_branch.py
_exec_wall_branch``), each a FRESH process that restores its world's
base saved state with EXACTLY ONE loadState (the child installs a guard
that refuses a second call).  The launcher never kills by image name and
never retries a failed child.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path

if __package__ in (None, ""):  # script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from campaign_stage1_v3.common import (  # noqa: E402 - bootstrap above
    archive_rename,
    append_jsonl,
    load_json,
    sha256_file,
    utc_now,
    write_json,
)
from campaign_walls_v1.common import (  # noqa: E402
    BRANCHES_TOTAL,
    DISK_PREFLIGHT_MIN_FREE_GB,
    LAUNCH_AUTHORIZATION,
    LAUNCH_OUT_DIR,
    LAUNCH_RUNS_ROOT,
    PACKAGE,
    PLAN_PATH,
    REGISTRATION_PATH,
    SIMULATED_SECONDS_TOTAL,
    SMOKE_OUT_DIR,
    WALLS_WORKERS,
    WM41_REGISTRATION,
    WM41_VERDICT,
    WM43_SUMMARY,
    WallsPilotError,
    package_code_digest,
)
from campaign_walls_v1.plan_walls_pilot import PLAN_SCHEMA  # noqa: E402
from campaign_walls_v1.smoke_walls import SMOKE_VERDICT_NAME, check_smoke_verdict  # noqa: E402

LAUNCH_SUMMARY_SCHEMA = "wm59_walls_pilot_launch_summary/1"
AUTHORIZATION_SCHEMA = "wm59_authorization/1"
ALLOCATION_LOG = "allocation_log.jsonl"
LAUNCH_LOG = "launch_log.jsonl"


class LaunchError(WallsPilotError):
    """A wall-pilot launch gate refusal."""


def check_disk_preflight(path, min_free_gb: float = DISK_PREFLIGHT_MIN_FREE_GB,
                         disk_free_fn=None) -> dict:
    """Refuse unless the drive ``path`` RESOLVES to (junctions followed)
    has at least ``min_free_gb`` free.  ``disk_free_fn`` is injectable
    (tests); default ``shutil.disk_usage``."""
    resolved = Path(path).resolve()
    free_fn = disk_free_fn or (lambda p: shutil.disk_usage(str(p)).free)
    free_bytes = float(free_fn(resolved.anchor or str(resolved)))
    free_gb = free_bytes / (1024 ** 3)
    if free_gb < float(min_free_gb):
        raise LaunchError(
            f"disk preflight refuses: {resolved.anchor} (the drive "
            f"{str(resolved)} resolves to) has {free_gb:.1f} GB free < "
            f"the declared {float(min_free_gb):g} GB minimum")
    return {"drive": resolved.anchor, "free_gb": round(free_gb, 3),
            "min_free_gb": float(min_free_gb)}


def check_authorization(authorization_path, registration_path) -> dict:
    """The owner-written AUTHORIZATION.json must carry the exact sha256
    of the CURRENT registration (a changed registration is a different
    experiment - re-authorize)."""
    authorization_path = Path(authorization_path)
    registration_path = Path(registration_path)
    if not authorization_path.is_file():
        raise LaunchError(
            f"owner authorization file not found: {authorization_path}; "
            "the launcher refuses to launch without it (write it only "
            "after owner sign-off, carrying the exact registration_sha256)")
    if not registration_path.is_file():
        raise LaunchError(f"registration file not found: {registration_path}")
    authorization = load_json(authorization_path)
    if authorization.get("schema") != AUTHORIZATION_SCHEMA:
        raise LaunchError(
            f"authorization schema must be {AUTHORIZATION_SCHEMA!r}")
    expected = authorization.get("registration_sha256")
    current = sha256_file(registration_path)
    if expected != current:
        raise LaunchError(
            "authorization digest does not match the current registration "
            f"(authorization={expected!r}, current={current!r})")
    return {"authorization": str(authorization_path),
            "registration_sha256": current,
            "authorized_by": authorization.get("authorized_by"),
            "authorized_utc": authorization.get("authorized_utc")}


def check_stage_gates(gate_paths: dict | None = None) -> dict:
    """The pinned WM-41/WM-43 gates: the pilot exactness verdict
    passed=true, the pilot registration present, the wall-mechanics
    fixture summary complete with zero failures."""
    gates = gate_paths or {
        "wm41_verdict": WM41_VERDICT,
        "wm41_registration": WM41_REGISTRATION,
        "wm43_summary": WM43_SUMMARY,
    }
    verdict_path = Path(gates["wm41_verdict"])
    if not verdict_path.is_file():
        raise LaunchError(f"WM-41 gate refuses: verdict missing: {verdict_path}")
    verdict = load_json(verdict_path)
    if verdict.get("passed") is not True:
        raise LaunchError("WM-41 gate refuses: the exactness verdict is "
                          "not passed=true")
    if not Path(gates["wm41_registration"]).is_file():
        raise LaunchError("WM-41 gate refuses: pilot registration missing: "
                          f"{gates['wm41_registration']}")
    wm43_path = Path(gates["wm43_summary"])
    if not wm43_path.is_file():
        raise LaunchError(f"WM-43 gate refuses: summary missing: {wm43_path}")
    wm43 = load_json(wm43_path)
    if int(wm43.get("failed") or 0) != 0 or int(wm43.get("complete") or 0) < 1:
        raise LaunchError("WM-43 gate refuses: the wall-mechanics fixture "
                          f"summary is not a clean pass ({wm43_path})")
    return {"wm41_verdict": str(verdict_path), "wm43_summary": str(wm43_path)}


def check_plan(plan: dict) -> dict:
    """The registered plan shape (never launched from anything else)."""
    if plan.get("schema") != PLAN_SCHEMA:
        raise LaunchError(f"plan schema {plan.get('schema')!r} != "
                          f"{PLAN_SCHEMA!r}")
    runs = plan.get("runs") or []
    if len(runs) != BRANCHES_TOTAL:
        raise LaunchError(f"plan carries {len(runs)} runs; the launch "
                          f"requires exactly {BRANCHES_TOTAL}")
    budget = plan.get("budget") or {}
    if float(budget.get("simulated_seconds_total") or 0) != \
            SIMULATED_SECONDS_TOTAL:
        raise LaunchError("plan budget is not the registered 144,000 sim-s")
    return {"runs": len(runs),
            "simulated_seconds_total": budget["simulated_seconds_total"]}


def _bases_root_by_run(plan: dict) -> dict:
    return {b["run"]: w["bases_root"]
            for w in plan["worlds"] for b in w["branches"]}


def _exec_child(argv: list) -> int:
    return subprocess.run([str(a) for a in argv]).returncode


def launch(plan_path, authorization_path, registration_path, out_dir,
           runs_root, smoke_verdict_path, workers: int, *, launcher=None,
           disk_free_fn=None, resume: bool = False,
           package_dir: Path | None = None, gate_paths: dict | None = None) -> dict:
    """Smoke-gated + authorization-gated dispatch of the 240 branches."""
    plan_path = Path(plan_path)
    registration_path = Path(registration_path)
    out_dir = Path(out_dir)
    runs_root = Path(runs_root)
    package_dir = Path(package_dir) if package_dir is not None else PACKAGE
    if workers != WALLS_WORKERS:
        raise LaunchError(
            f"--workers must be exactly the registered {WALLS_WORKERS} "
            f"(got {workers})")
    smoke_gate = check_smoke_verdict(smoke_verdict_path, registration_path,
                                     package_dir=package_dir)
    authorization = check_authorization(authorization_path, registration_path)
    stage_gates = check_stage_gates(gate_paths)
    plan = load_json(plan_path)
    plan_gate = check_plan(plan)
    disk = check_disk_preflight(runs_root, disk_free_fn=disk_free_fn)
    if runs_root.exists():
        if not resume:
            raise LaunchError(
                f"runs root already exists (a NEW directory, never "
                f"overwritten; pass --resume after an interrupted attempt): "
                f"{runs_root}")
    else:
        runs_root.mkdir(parents=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "LAUNCH_SUMMARY.json"
    if summary_path.exists():
        if not resume:
            raise LaunchError(
                f"{summary_path} already exists (write-once; pass --resume "
                "after an interrupted attempt)")
        archive_rename(summary_path, tag="superseded")
    bases_root_by_run = _bases_root_by_run(plan)

    lock = threading.Lock()

    def append_locked(path: Path, row: dict) -> None:
        with lock:
            append_jsonl(path, row)

    dispatch = launcher or _exec_child

    def run_one(run: dict) -> dict:
        name = str(run["run"])
        target = runs_root / name
        if target.exists():
            receipt_path = target / "RUN_RECEIPT.json"
            if receipt_path.is_file() and load_json(receipt_path).get(
                    "exit_status") == "complete":
                return {"run": name, "status": "resume_kept_complete"}
            archive_rename(target, tag="interrupted")
        append_locked(out_dir / ALLOCATION_LOG, {
            "schema": "wm59_allocation/1", "run": name,
            "world": run.get("world"), "arm": run.get("arm"),
            "run_dir": str(target), "allocated_utc": utc_now(),
            "status": "allocating"})
        argv = [
            sys.executable,
            str(package_dir / "run_wall_branch.py"),
            "_exec_wall_branch",
            "--run", name,
            "--plan", str(plan_path),
            "--runs-root", str(runs_root),
            "--registration", str(registration_path),
            "--bases-root", bases_root_by_run[name],
        ]
        code = dispatch(argv)
        append_locked(out_dir / LAUNCH_LOG, {
            "run": name, "exit_code": code, "ended_utc": utc_now()})
        return {"run": name, "status": "complete" if code == 0 else "failed",
                "exit_code": code}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_one, plan["runs"]))
    complete = [r for r in results if r["status"] in ("complete",
                                                      "resume_kept_complete")]
    failed = [r for r in results if r["status"] not in
              ("complete", "resume_kept_complete")]
    summary = {
        "schema": LAUNCH_SUMMARY_SCHEMA, "created_utc": utc_now(),
        "workers": workers,
        "authorization": authorization, "smoke_gate": smoke_gate,
        "stage_gates": stage_gates, "disk_preflight": disk,
        "plan": str(plan_path), "plan_sha256": sha256_file(plan_path),
        "registration": str(registration_path),
        "registration_sha256": sha256_file(registration_path),
        "package_code_sha256": package_code_digest(package_dir),
        "runs_root": str(runs_root),
        "plan_gate": plan_gate,
        "budget": {"planned_runs": BRANCHES_TOTAL,
                   "planned_simulated_seconds": SIMULATED_SECONDS_TOTAL,
                   "dispatched_runs": len(results), "retries": 0},
        "runs": results,
        "complete": len(complete), "failed": len(failed),
        "resume": bool(resume),
        "finished_utc": utc_now(),
    }
    write_json(summary_path, summary)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("launch")
    run.add_argument("--plan", type=Path, default=PLAN_PATH)
    run.add_argument("--authorization", type=Path, default=Path(LAUNCH_AUTHORIZATION))
    run.add_argument("--registration", type=Path, default=REGISTRATION_PATH)
    run.add_argument("--out", type=Path, default=Path(LAUNCH_OUT_DIR))
    run.add_argument("--runs-root", type=Path, default=Path(LAUNCH_RUNS_ROOT))
    run.add_argument("--smoke-verdict", type=Path,
                     default=Path(SMOKE_OUT_DIR) / SMOKE_VERDICT_NAME)
    run.add_argument("--workers", type=int, default=WALLS_WORKERS)
    run.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    summary = launch(args.plan, args.authorization, args.registration,
                     args.out, args.runs_root, args.smoke_verdict,
                     args.workers, resume=args.resume)
    print(json.dumps({"complete": summary["complete"],
                      "failed": summary["failed"],
                      "summary": str(Path(args.out) / "LAUNCH_SUMMARY.json")}))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
