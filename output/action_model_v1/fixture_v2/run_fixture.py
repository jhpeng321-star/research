"""WM-43B fixture runner (DRAFT, not authorized): prepare-only launch discipline for the wall-v2.1 closure re-validation (5 arms x 5 seeds).

Reuses the WM-41 pilot runner rules (read-only reference
the WM-41 pilot runner (campaign_pilot_v1/run_pilot.py)): authorization file
carrying the registration SHA-256; never pre-create the output directory;
allocation row BEFORE each child launch; one child OS process per run;
explicit worker count (max 6); ZERO automatic retries; write-once receipts;
never overwrite a prior run or silently retry it.

This preparation task launches nothing: no AUTHORIZATION.json exists, and
none may be written by this package (the planner writes it after owner
sign-off).  ``preflight`` is read-only and must pass, or fail ONLY on the
pending seeds diff (traffic_shepherd/seeds.py is not edited by this task;
the block ships as seeds_py_wm43b.diff for the integrator to apply).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import types
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):  # script execution: make the fixture modules importable
    _FIXTURE = Path(__file__).resolve().parent
    _REPO = _FIXTURE.parents[2]
    for _path in (str(_REPO), str(_FIXTURE)):  # repo first: traffic_shepherd, then wm43_*
        if _path not in sys.path:
            sys.path.insert(0, _path)

from wm43_plan import (ALLOCATION_SCHEMA, AUTHORIZATION_PATH, HORIZON_TICKS,
                       PLAN_PATH, PREFLIGHT_SCHEMA, RECEIPT_SCHEMA,
                       REGISTRATION_PATH, RETRIES, RUNS_ROOT_RELATIVE,
                       RUNS_TOTAL, SEEDS_DIFF_PATH, SEEDS_MODULE_RELATIVE,
                       SMOKE_ARM, SMOKE_HORIZON_TICKS,
                       SMOKE_RUNS_ROOT_SUFFIX, SMOKE_SEED, SMOKE_VERDICT_NAME,
                       SMOKE_VERDICT_SCHEMA, SUMMARY_SCHEMA, WORKERS_MAX,
                       check_plan_shape, load_plan)

RECEIPT_NAME = "RUN_RECEIPT.json"
ALLOCATION_LOG = "allocations.jsonl"
LAUNCH_LOG = "launch_log.jsonl"
SUMMARY_NAME = "LAUNCH_SUMMARY.json"
SEEDS_BLOCK = "WM43B_WALL_CLOSURE_SEEDS"
SEEDS_LOADING_MAP = "WM43B_SEED_LOADING_MAP"
#: this package directory: the write-once SMOKE_VERDICT.json lives here
FIXTURE_DIR = Path(__file__).resolve().parent


class RunnerError(RuntimeError):
    """Refusal that must not be retried or worked around."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path = Path(path)
    if path.exists():
        raise RunnerError(f"refusing to overwrite {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def package_code_digest(directory: Path | None = None) -> str:
    """SHA-256 binding of the launch-relevant package sources.

    Sorted (module name, file bytes) pairs over the fixture-root *.py
    modules (tests excluded - they do not execute in a launch).  Any code
    change invalidates a recorded SMOKE_VERDICT, forcing a re-smoke.
    """
    directory = Path(directory) if directory is not None else Path(__file__).resolve().parent
    hasher = hashlib.sha256()
    for module in sorted(directory.glob("*.py")):
        hasher.update(module.name.encode("utf-8"))
        hasher.update(module.read_bytes())
    return hasher.hexdigest()


def peak_rss_sample() -> dict:
    """Best-effort peak RSS of the runner child (and its SUMO child).

    POSIX: resource.getrusage ru_maxrss for SELF and CHILDREN.  Windows:
    ctypes GetProcessMemoryInfo PeakWorkingSetSize for the child process
    itself (the SUMO grandchild handle is not retained by the traci module,
    so children_rss is unavailable there and recorded as such - the smoke
    gate uses pass/fail checks, RSS is evidence metadata only).
    """
    try:
        import resource  # noqa: PLC0415 - POSIX only

        self_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        children_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        return {"available": True, "platform": sys.platform,
                "self_peak_rss": self_rss, "children_peak_rss": children_rss,
                "units": "kiB (Linux) / bytes (macOS)"}
    except ImportError:
        pass
    try:
        import ctypes  # noqa: PLC0415 - Windows only

        class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        # GetProcessMemoryInfo needs the FULL declared buffer passed as cb
        # (the typed struct alone can trigger ERROR_INSUFFICIENT_BUFFER on
        # some builds): overlay the typed view on a large raw buffer.
        raw = ctypes.create_string_buffer(4096)
        counters = ctypes.cast(raw, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS)).contents
        counters.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo.argtypes = (
            ctypes.c_void_p, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS), ctypes.c_ulong)
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), 4096):
            return {"available": True, "platform": sys.platform,
                    "self_peak_rss": int(counters.PeakWorkingSetSize),
                    "children_peak_rss": None, "units": "bytes (Windows)"}
    except BaseException:  # noqa: BLE001 - best-effort telemetry
        pass
    return {"available": False, "platform": sys.platform,
            "self_peak_rss": None, "children_peak_rss": None,
            "reason": "peak RSS sampling unavailable on this platform"}


def append_jsonl(path: Path, row: dict) -> None:
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


# ---------------------------------------------------------------------------
# Seeds gate: frozen module OR the pending seeds_py_wm43b.diff (dry-applied)
# ---------------------------------------------------------------------------

def _load_seeds_module_from_text(name: str, source: str):
    module = types.ModuleType(name)
    module.__file__ = str(repo_root() / SEEDS_MODULE_RELATIVE)
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def load_frozen_seeds_module():
    path = repo_root() / SEEDS_MODULE_RELATIVE
    if not path.is_file():
        raise RunnerError(f"frozen seeds module missing: {path}")
    spec = importlib.util.spec_from_file_location("wm43_frozen_seeds_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_unified_diff(source: str, diff_text: str) -> str:
    """Minimal unified-diff applier (context-verified, single file)."""
    src_lines = source.splitlines(keepends=True)
    out: list[str] = []
    pos = 0
    hunk_re = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    diff_lines = diff_text.splitlines()
    index = 0
    matched_any = False
    while index < len(diff_lines):
        line = diff_lines[index]
        match = hunk_re.match(line)
        if match is None:
            index += 1
            continue
        matched_any = True
        start = int(match.group(1)) - 1 if int(match.group(1)) > 0 else 0
        if start < pos:
            raise RunnerError("diff hunks are not monotonic")
        out.extend(src_lines[pos:start])
        pos = start
        index += 1
        while index < len(diff_lines):
            body = diff_lines[index]
            if body.startswith("@@") or body.startswith("diff ") \
                    or body.startswith("--- ") or body.startswith("+++ "):
                break
            index += 1
            if body.startswith("+"):
                out.append(body[1:] + "\n" if not body[1:].endswith("\n") else body[1:])
            elif body.startswith("-"):
                if pos >= len(src_lines) or src_lines[pos].rstrip("\n") != body[1:].rstrip("\n"):
                    raise RunnerError(
                        f"diff removal does not match the current seeds.py at line "
                        f"{pos + 1}: expected {body[1:].rstrip()!r}, "
                        f"found {src_lines[pos].rstrip() if pos < len(src_lines) else '<eof>'!r}")
                pos += 1
            elif body.startswith(" ") or body == "":
                found = src_lines[pos] if pos < len(src_lines) else "<eof>"
                if pos >= len(src_lines) or src_lines[pos].rstrip("\n") != body[1:].rstrip("\n"):
                    raise RunnerError(
                        f"diff context does not match at line {pos + 1}: "
                        f"expected {body[1:].rstrip()!r}, "
                        f"found {found!r}")
                out.append(src_lines[pos])
                pos += 1
            else:
                raise RunnerError(f"unsupported diff line: {body!r}")
    if not matched_any:
        raise RunnerError("no diff hunks found")
    out.extend(src_lines[pos:])
    return "".join(out)


def _gate_accepts(module, seed: int, loading: int, role: str) -> str | None:
    """None when accepted, else the refusal reason."""
    block = getattr(module, SEEDS_BLOCK, None)
    if block is None:
        return f"frozen seeds module carries no {SEEDS_BLOCK} block"
    if seed not in block:
        return f"seed {seed} not in {SEEDS_BLOCK}"
    mapping = getattr(module, SEEDS_LOADING_MAP, {})
    if mapping.get(seed) != loading:
        return f"{SEEDS_LOADING_MAP}[{seed}] != registered loading {loading}"
    try:
        module.require_development_case(seed, loading, role)
    except BaseException as exc:  # noqa: BLE001 - any refusal is data
        return f"require_development_case refused ({seed}, {loading}, {role}): {exc!r}"
    return None


def seeds_gate(plan: dict) -> dict:
    """Accept if the applied OR dry-applied seeds module accepts every world."""
    worlds = [(r["seed"], r["loading"], r["role"]) for r in plan["runs"]]
    report = {"seeds_module": str(repo_root() / SEEDS_MODULE_RELATIVE),
              "seeds_diff": str(SEEDS_DIFF_PATH), "worlds_checked": len(worlds),
              "refusals": [], "state": "registered_in_module"}
    try:
        module = load_frozen_seeds_module()
    except BaseException as exc:  # noqa: BLE001
        report["state"] = "error"
        report["refusals"].append(f"frozen module load failed: {exc!r}")
        report["ok"] = False
        return report
    refusals = [r for world in worlds if (r := _gate_accepts(module, *world))]
    if not refusals:
        report["ok"] = True
        return report
    report["refusals"] = refusals
    report["state"] = "pending_diff"
    # The live module does not know the block yet: verify the SHIPPED DIFF
    # applies to the current seeds.py bytes and that the patched module
    # would accept every planned world (the integrator's apply is safe).
    try:
        diff_text = SEEDS_DIFF_PATH.read_text(encoding="utf-8")
        patched = apply_unified_diff(
            (repo_root() / SEEDS_MODULE_RELATIVE).read_text(encoding="utf-8"), diff_text)
        candidate = _load_seeds_module_from_text("wm43_seeds_patched", patched)
    except BaseException as exc:  # noqa: BLE001
        report["refusals"].append(
            f"pending seeds diff {SEEDS_DIFF_PATH.name} does not apply cleanly "
            f"to the current {SEEDS_MODULE_RELATIVE}: {exc!r}")
        report["state"] = "error"
        report["ok"] = False
        return report
    patched_refusals = [r for world in worlds if (r := _gate_accepts(candidate, *world))]
    if patched_refusals:
        report["refusals"].extend(patched_refusals)
        report["state"] = "error"
        report["ok"] = False
        return report
    report["refusals"] = [
        "PENDING INTEGRATOR ACTION: seeds_py_wm43b.diff is not applied to "
        "traffic_shepherd/seeds.py yet (the dry-applied patch accepts every "
        "planned world; apply the diff, then re-run preflight)"]
    report["ok"] = False
    return report


# ---------------------------------------------------------------------------
# Preflight (read-only; no SUMO)
# ---------------------------------------------------------------------------

def preflight(plan_path: Path, runs_root: Path | None = None) -> dict:
    plan = load_plan(plan_path)
    problems = check_plan_shape(plan)
    constants = plan.get("constants", {})
    root = repo_root()
    resolved_root = Path(runs_root) if runs_root is not None \
        else root / constants.get("runs_root_relative", RUNS_ROOT_RELATIVE)
    environment: dict = {"python": sys.version.split()[0]}
    try:
        import traci  # noqa: F401, PLC0415 - availability probe only
        environment["traci_import"] = "ok"
    except BaseException as exc:  # noqa: BLE001
        environment["traci_import"] = f"failed: {exc!r}"
        problems.append("traci is not importable in the launch interpreter")
    sumo_binary = shutil.which("sumo")
    environment["sumo_binary"] = sumo_binary
    if sumo_binary is None:
        problems.append("SUMO is not on PATH (sumo binary required for the runs)")
    try:
        from traffic_shepherd import mixed_assignment, wall_group
        environment["mixed_assignment_version"] = mixed_assignment.VERSION
        environment["wall_group_constants"] = {
            name: getattr(wall_group, name)
            for name in ("SAFE_SPEED_MIN_MPS", "REAR_GAP_M", "REAR_SPEED_EXCESS_MPS",
                         "ABORT_SPAN_M", "ASSEMBLY_SPEED_TOL_MPS", "ASSEMBLY_SPAN_TOL_M")}
        if mixed_assignment.VERSION != "mixed_assignment/3":
            problems.append(f"dispatcher version {mixed_assignment.VERSION} != "
                            "mixed_assignment/3")
    except BaseException as exc:  # noqa: BLE001
        environment["dispatcher"] = f"import failed: {exc!r}"
        problems.append("traffic_shepherd dispatcher modules are not importable")
    identity: dict = {}
    try:
        from traffic_shepherd.config import load_config
        config = load_config()
        identity["network"] = {"path": str(config.network_path),
                               "sha256": sha256_file(config.network_path)}
        identity["model_sha256"] = sha256_file(config.model_path)
        identity["model_matches_adapter"] = identity["model_sha256"] == config.model_sha256
        if not identity["model_matches_adapter"]:
            problems.append("CIRCLES model digest differs from the adapter contract")
        if not config.network_path.is_file():
            problems.append("preserved network file is missing")
    except BaseException as exc:  # noqa: BLE001
        identity["config"] = f"load failed: {exc!r}"
        problems.append("adapter configuration failed to load")
    seeds = seeds_gate(plan)
    if not seeds["ok"]:
        problems.append(seeds["refusals"][0])
    # Scenario dry-run per planned run (temp dir, no SUMO, nothing under runs/).
    scenario_digests: dict[str, str] = {}
    import tempfile
    with tempfile.TemporaryDirectory(prefix="wm43_dryrun_") as tmp:
        for run in plan["runs"]:
            try:
                from traffic_shepherd.config import load_config
                from wm43_scenario import build_scenario
                from wm43_validate import validate_scenario
                scene, roster, manifest = build_scenario(
                    Path(tmp) / run["run"], run, load_config())
                scenario_digests[run["run"]] = manifest["routes_sha256"]
                if not scene.is_file() or len(roster) < 2:
                    problems.append(f"{run['run']}: scenario dry-run produced no roster")
                rule_problems = validate_scenario(Path(scene).parent)
                problems.extend(f"{run['run']}: {p}" for p in rule_problems)
            except BaseException as exc:  # noqa: BLE001
                problems.append(f"{run['run']}: scenario dry-run failed: {exc!r}")
    runs_parent_ok = resolved_root.parent.is_dir()
    if not runs_parent_ok:
        problems.append(f"runs parent does not exist (never created by the "
                        f"runner): {resolved_root.parent}")
    if resolved_root.exists():
        problems.append(f"runs root already exists (must be a NEW directory, "
                        f"never overwritten): {resolved_root}")
    report = {"schema": PREFLIGHT_SCHEMA, "created_utc": utc_now(),
              "plan": str(plan_path), "plan_sha256": sha256_file(plan_path),
              "registration": str(REGISTRATION_PATH),
              "registration_sha256": sha256_file(REGISTRATION_PATH),
              "runs_root": str(resolved_root), "runs_parent_ok": runs_parent_ok,
              "environment": environment, "identity": identity,
              "seeds": seeds, "scenario_routes_sha256": scenario_digests,
              "problems": problems, "ok": not problems}
    return report


# ---------------------------------------------------------------------------
# Authorization + launch (parent)
# ---------------------------------------------------------------------------

def check_authorization(authorization_path: Path, registration_path: Path) -> dict:
    authorization_path, registration_path = Path(authorization_path), Path(registration_path)
    if not authorization_path.is_file():
        raise RunnerError(
            f"owner authorization file not found: {authorization_path}; the runner "
            "refuses to launch without it (write it only after owner sign-off, "
            "carrying the exact registration_sha256)")
    if not registration_path.is_file():
        raise RunnerError(f"registration file not found: {registration_path}")
    authorization = load_json(authorization_path)
    if authorization.get("schema") != "wm43_authorization/1":
        raise RunnerError("authorization schema must be wm43_authorization/1")
    expected = authorization.get("registration_sha256")
    current = sha256_file(registration_path)
    if expected != current:
        raise RunnerError(
            "authorization digest does not match the current registration "
            f"(authorization={expected!r}, current={current!r}); a changed "
            "registration is a different experiment - re-authorize")
    return {"authorization": str(authorization_path),
            "registration_sha256": current,
            "authorized_by": authorization.get("authorized_by"),
            "authorized_utc": authorization.get("authorized_utc")}


def _exec_child(argv: list[str]) -> int:
    return subprocess.run([str(a) for a in argv]).returncode


def check_smoke_verdict(verdict_path: Path | None,
                        registration_path: Path) -> dict:
    """The launch gate: a CURRENT, passed, write-once SMOKE_VERDICT.json.

    Bound to (a) the sha256 of the current REGISTRATION.md and (b) the
    package code digest - any registration or code edit invalidates the
    verdict and forces a re-smoke.  Refuses loudly otherwise.
    """
    verdict_path = Path(verdict_path) if verdict_path is not None else \
        FIXTURE_DIR / SMOKE_VERDICT_NAME
    if not verdict_path.is_file():
        raise RunnerError(
            f"launch gate: no pre-launch smoke verdict at {verdict_path}; run "
            "the registered smoke first: run_fixture.py smoke  (one capped "
            "run, never data, zero retries)")
    verdict = load_json(verdict_path)
    if verdict.get("schema") != SMOKE_VERDICT_SCHEMA:
        raise RunnerError(f"launch gate: verdict schema "
                          f"{verdict.get('schema')!r} != "
                          f"{SMOKE_VERDICT_SCHEMA!r}")
    if verdict.get("passed") is not True:
        raise RunnerError("launch gate: the recorded smoke did NOT pass "
                          f"(passed={verdict.get('passed')!r}); fix and re-smoke")
    current_registration = sha256_file(registration_path)
    if verdict.get("registration_sha256") != current_registration:
        raise RunnerError(
            "launch gate: smoke verdict is bound to registration "
            f"{str(verdict.get('registration_sha256'))[:16]}... but the current "
            f"registration is {current_registration[:16]}...; re-run the smoke "
            "after any registration change")
    current_code = package_code_digest()
    if verdict.get("package_code_sha256") != current_code:
        raise RunnerError(
            "launch gate: smoke verdict is bound to package code digest "
            f"{str(verdict.get('package_code_sha256'))[:16]}... but the current "
            f"package code digest is {current_code[:16]}...; re-run the smoke "
            "after any fixture code change")
    return {"verdict": str(verdict_path), "passed": True,
            "smoke_runs_root": verdict.get("smoke_runs_root"),
            "smoke_horizon_ticks": verdict.get("horizon_ticks"),
            "registration_sha256": current_registration,
            "package_code_sha256": current_code,
            "smoked_utc": verdict.get("created_utc")}


def smoke(plan_path: Path, registration_path: Path, runs_root: Path | None = None,
          launcher=None) -> dict:
    """Standing pre-launch smoke: ONE capped run through the REAL child path.

    Registered shape (PLAN.json 'smoke'): the A4 first-seed run capped at
    t=70 s into `<runs_root>_smoke` - real scenario generation, real TraCI
    via the same run-child code, real dispatcher wall assembly start, zero
    retries, never used as data.  Writes the write-once SMOKE_VERDICT.json
    (passed = child complete AND smoke checks green) bound to the
    registration sha256, the package code digest and a peak-RSS sample.
    The full launch refuses without that verdict.
    """
    plan = load_plan(plan_path)
    smoke_spec = plan.get("smoke") or {}
    horizon = int(smoke_spec.get("horizon_ticks", SMOKE_HORIZON_TICKS))
    root = repo_root()
    resolved_root = Path(runs_root) if runs_root is not None else root / str(
        smoke_spec.get("runs_root_relative",
                       plan["constants"]["runs_root_relative"]
                       + SMOKE_RUNS_ROOT_SUFFIX))
    report = preflight(plan_path, runs_root=resolved_root)
    if not report["ok"]:
        raise RunnerError("smoke refuses: preflight did not pass: "
                          + " | ".join(report["problems"][:5]))
    if resolved_root.exists():
        raise RunnerError(f"smoke runs root already exists: {resolved_root}")
    if not resolved_root.parent.is_dir():
        raise RunnerError(f"smoke runs parent does not exist: "
                          f"{resolved_root.parent}")
    verdict_path = FIXTURE_DIR / SMOKE_VERDICT_NAME
    if verdict_path.exists():
        raise RunnerError(
            f"smoke refuses: {verdict_path} already exists (write-once; a "
            "stale verdict is superseded only by deleting it consciously, "
            "never by overwriting)")
    resolved_root.mkdir()
    run = next(r for r in plan["runs"]
               if r["arm"] == smoke_spec.get("arm", SMOKE_ARM)
               and r["seed"] == int(smoke_spec.get("seed", SMOKE_SEED)))
    append_jsonl(resolved_root / ALLOCATION_LOG, {
        "schema": ALLOCATION_SCHEMA, "mode": "smoke", "run": run["run"],
        "arm": run["arm"], "seed": run["seed"], "duration_s": horizon,
        "run_dir": str(resolved_root / run["run"]),
        "allocated_utc": utc_now(), "status": "allocating"})
    dispatch = launcher or _exec_child
    argv = [sys.executable, str(Path(__file__).resolve()), "_exec_smoke",
            "--run", run["run"], "--plan", str(plan_path),
            "--runs-root", str(resolved_root),
            "--registration", str(registration_path),
            "--horizon-ticks", str(horizon)]
    code = dispatch(argv)
    append_jsonl(resolved_root / LAUNCH_LOG, {
        "mode": "smoke", "run": run["run"], "exit_code": code,
        "ended_utc": utc_now()})
    run_dir = resolved_root / run["run"]
    receipt_path = run_dir / RECEIPT_NAME
    checks_path = run_dir / "SMOKE_CHECKS.json"
    receipt = load_json(receipt_path) if receipt_path.is_file() else None
    checks = load_json(checks_path) if checks_path.is_file() else None
    passed = bool(code == 0 and receipt and receipt.get("exit_status") == "complete"
                  and checks and checks.get("ok") is True)
    peak = (receipt or {}).get("peak_rss") or peak_rss_sample()
    verdict = {
        "schema": SMOKE_VERDICT_SCHEMA, "passed": passed,
        "registration_sha256": sha256_file(registration_path),
        "package_code_sha256": package_code_digest(),
        "smoke_runs_root": str(resolved_root), "run": run["run"],
        "arm": run["arm"], "seed": run["seed"],
        "horizon_ticks": horizon, "simulated_seconds": horizon,
        "ticks_executed": (receipt or {}).get("summary", {}).get("ticks_executed"),
        "exit_code": code, "peak_rss": peak,
        "smoke_checks_ok": bool(checks.get("ok")) if checks else None,
        "created_utc": utc_now(),
        "note": "standing pre-launch gate; smoke runs are never data",
    }
    write_json(verdict_path, verdict)
    summary = {"schema": "wm43_smoke_summary/1", "created_utc": utc_now(),
               "passed": passed, "exit_code": code,
               "runs_root": str(resolved_root), "run": run["run"],
               "horizon_ticks": horizon, "simulated_seconds": horizon,
               "peak_rss": peak, "verdict": str(verdict_path), "retries": 0}
    write_json(resolved_root / "SMOKE_SUMMARY.json", summary)
    return summary


def launch(plan_path: Path, authorization_path: Path, registration_path: Path,
           workers: int, runs_root: Path | None = None, launcher=None,
           smoke_verdict_path: Path | None = None) -> dict:
    """Smoke-gated + authorization-gated dispatch of the 25 planned runs.

    Gate order: budget guards -> owner authorization (registration digest)
    -> SMOKE_VERDICT.json (passed, bound to the current registration sha256
    AND the package code digest) -> preflight -> dispatch.  Zero retries.
    """
    if workers < 1:
        raise RunnerError("--workers must be >= 1 (explicit worker count)")
    if workers > WORKERS_MAX:
        raise RunnerError(f"--workers must be <= {WORKERS_MAX} (registered budget)")
    plan = load_plan(plan_path)
    if len(plan.get("runs", [])) != RUNS_TOTAL:
        raise RunnerError(f"plan carries {len(plan['runs'])} runs; the budget guard "
                          f"requires exactly {RUNS_TOTAL}")
    smoke_gate = check_smoke_verdict(smoke_verdict_path, registration_path)
    gate = check_authorization(authorization_path, registration_path)
    root = repo_root()
    resolved_root = Path(runs_root) if runs_root is not None \
        else root / plan["constants"]["runs_root_relative"]
    report = preflight(plan_path, runs_root=resolved_root)
    if not report["ok"]:
        raise RunnerError("launch refuses: preflight did not pass: "
                          + " | ".join(report["problems"][:5]))
    if resolved_root.exists():
        raise RunnerError(f"runs root already exists: {resolved_root}")
    if not resolved_root.parent.is_dir():
        raise RunnerError(f"runs parent does not exist: {resolved_root.parent}")
    resolved_root.mkdir()
    lock = threading.Lock()

    def append_locked(path: Path, row: dict) -> None:
        with lock:
            append_jsonl(path, row)

    dispatch = launcher or _exec_child

    def run_one(run: dict) -> dict:
        target = resolved_root / run["run"]
        if target.exists():
            return {"run": run["run"], "status": "skipped_exists"}
        append_locked(resolved_root / ALLOCATION_LOG, {
            "schema": ALLOCATION_SCHEMA, "run": run["run"], "arm": run["arm"],
            "seed": run["seed"], "duration_s": run["duration_s"],
            "run_dir": str(target), "allocated_utc": utc_now(),
            "status": "allocating"})
        argv = [sys.executable, str(Path(__file__).resolve()), "_exec_run",
                "--run", run["run"], "--plan", str(plan_path),
                "--runs-root", str(resolved_root),
                "--registration", str(registration_path)]
        code = dispatch(argv)
        append_locked(resolved_root / LAUNCH_LOG, {
            "run": run["run"], "exit_code": code, "ended_utc": utc_now()})
        status = "complete" if code == 0 else "failed"
        return {"run": run["run"], "status": status, "exit_code": code}

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_one, plan["runs"]))
    complete = [r for r in results if r["status"] == "complete"]
    failed = [r for r in results if r["status"] not in ("complete",)]
    summary = {
        "schema": SUMMARY_SCHEMA, "created_utc": utc_now(),
        "workers": workers, "authorization": gate, "smoke_gate": smoke_gate,
        "plan": str(plan_path), "plan_sha256": sha256_file(plan_path),
        "registration": str(registration_path),
        "registration_sha256": sha256_file(registration_path),
        "runs_root": str(resolved_root),
        "budget": {"planned_runs": RUNS_TOTAL,
                   "planned_simulated_seconds": HORIZON_TICKS * RUNS_TOTAL,
                   "dispatched_runs": len(results),
                   "retries": RETRIES},
        "runs": results,
        "complete": len(complete), "failed": len(failed),
        "finished_utc": utc_now(),
    }
    write_json(resolved_root / SUMMARY_NAME, summary)
    return summary


# ---------------------------------------------------------------------------
# Child: one mechanics run
# ---------------------------------------------------------------------------

def exec_run(run_name_arg: str, plan_path: Path, runs_root: Path,
             registration_path: Path, *, horizon_ticks: int | None = None,
             smoke: bool = False) -> int:
    """One run child.  ``smoke=True`` caps the horizon (registered smoke:
    70 ticks) and evaluates the SMOKE check set instead of the full 600-tick
    acceptance lists; the receipt records horizon_mode + a peak-RSS sample.
    """
    from wm43_checks import evaluate as evaluate_full
    from wm43_checks import evaluate_smoke
    from wm43_loop import run_mechanics
    from wm43_plan import CHECK_LISTS, SMOKE_CHECK_IDS
    plan = load_plan(plan_path)
    run = next(r for r in plan["runs"] if r["run"] == run_name_arg)
    target = Path(runs_root) / run["run"]
    if target.exists():
        raise RunnerError(f"refusing to write into existing directory: {target}")
    gate = seeds_gate(plan)
    if not gate["ok"]:
        raise RunnerError("seeds gate refused: " + " | ".join(gate["refusals"][:3]))
    target.mkdir(parents=True)
    capped = horizon_ticks is not None
    receipt = {"schema": RECEIPT_SCHEMA, "run": run["run"], "arm": run["arm"],
               "seed": run["seed"], "loading": run["loading"],
               "penetration": run["penetration"], "role": run["role"],
               "duration_s": int(horizon_ticks or run["duration_s"]),
               "horizon_mode": ("smoke_cap" if smoke else
                                "registered_full" if not capped else "capped"),
               "wall_speed_mps": run["wall_speed_mps"],
               "checks_expected": list(SMOKE_CHECK_IDS if smoke
                                       else CHECK_LISTS[run["arm"]]),
               "runs_root": str(runs_root),
               "registration_sha256": sha256_file(registration_path),
               "plan_sha256": sha256_file(plan_path),
               "interpreter": sys.executable, "started_utc": utc_now(),
               "retries": 0}
    status, error = "complete", None
    try:
        summary = run_mechanics(run, target, horizon_ticks=horizon_ticks)
        receipt["summary"] = summary
        receipt["peak_rss"] = peak_rss_sample()
        if smoke:
            checks = evaluate_smoke(run, target,
                                    receipt={"exit_status": "complete",
                                             "summary": summary},
                                    horizon_ticks=int(horizon_ticks))
            checks_name = "SMOKE_CHECKS.json"
        else:
            checks = evaluate_full(run, target,
                                   receipt={"exit_status": "complete",
                                            "summary": summary})
            checks_name = "CHECKS.json"
        write_json(target / checks_name, checks)
        receipt["checks"] = {"ok": checks["ok"], "results": checks["checks"]}
        if not checks["ok"]:
            status = "checks_failed"
            receipt["error"] = ("smoke checks failed; see SMOKE_CHECKS.json"
                                if smoke else
                                "per-run acceptance checks failed; see CHECKS.json")
    except BaseException as exc:  # noqa: BLE001 - receipt captures any failure
        status, error = "failed", repr(exc)
    receipt["exit_status"] = status
    if error is not None:
        receipt["error"] = error
    receipt["finished_utc"] = utc_now()
    write_json(target / RECEIPT_NAME, receipt)
    print(json.dumps({"run": run["run"], "exit_status": status,
                      "checks_ok": receipt.get("checks", {}).get("ok")}))
    return 0 if status == "complete" else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight_parser = sub.add_parser(
        "preflight", help="read-only checks; no simulation, no authorization needed "
                          "(passes once the seeds diff is applied)")
    preflight_parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    preflight_parser.add_argument("--runs-root", type=Path, default=None)
    launch_parser = sub.add_parser(
        "launch", help="authorization-gated dispatch of the 25 planned runs")
    launch_parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    launch_parser.add_argument("--authorization", type=Path, default=AUTHORIZATION_PATH)
    launch_parser.add_argument("--registration", type=Path, default=REGISTRATION_PATH)
    launch_parser.add_argument("--runs-root", type=Path, default=None)
    launch_parser.add_argument("--workers", type=int, required=True,
                               help="explicit worker count (max 6)")
    smoke_parser = sub.add_parser(
        "smoke", help="standing pre-launch smoke: ONE capped run (A1 seed "
                      "A4 first seed, t<=70 s) through the real child path into "
                      "<runs_root>_smoke; writes the write-once "
                      "SMOKE_VERDICT.json the launch gate requires; no "
                      "authorization needed, never data, zero retries")
    smoke_parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    smoke_parser.add_argument("--registration", type=Path, default=REGISTRATION_PATH)
    smoke_parser.add_argument("--runs-root", type=Path, default=None)
    exec_parser = sub.add_parser("_exec_run", help=argparse.SUPPRESS)
    exec_parser.add_argument("--run", required=True)
    exec_parser.add_argument("--plan", type=Path, required=True)
    exec_parser.add_argument("--runs-root", type=Path, required=True)
    exec_parser.add_argument("--registration", type=Path, required=True)
    exec_smoke_parser = sub.add_parser("_exec_smoke", help=argparse.SUPPRESS)
    exec_smoke_parser.add_argument("--run", required=True)
    exec_smoke_parser.add_argument("--plan", type=Path, required=True)
    exec_smoke_parser.add_argument("--runs-root", type=Path, required=True)
    exec_smoke_parser.add_argument("--registration", type=Path, required=True)
    exec_smoke_parser.add_argument("--horizon-ticks", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        report = preflight(args.plan, runs_root=args.runs_root)
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0 if report["ok"] else 1
    if args.command == "smoke":
        summary = smoke(args.plan, args.registration, runs_root=args.runs_root)
        print(json.dumps({"passed": summary["passed"],
                          "exit_code": summary["exit_code"],
                          "runs_root": summary["runs_root"],
                          "run": summary["run"],
                          "horizon_ticks": summary["horizon_ticks"],
                          "verdict": summary["verdict"]}))
        return 0 if summary["passed"] else 1
    if args.command == "launch":
        summary = launch(args.plan, args.authorization, args.registration,
                         args.workers, runs_root=args.runs_root)
        print(json.dumps({"runs_root": summary["runs_root"],
                          "complete": summary["complete"],
                          "failed": summary["failed"],
                          "retries": summary["budget"]["retries"]}))
        return 0 if summary["failed"] == 0 else 1
    if args.command == "_exec_smoke":
        return exec_run(args.run, args.plan, args.runs_root, args.registration,
                        horizon_ticks=args.horizon_ticks, smoke=True)
    return exec_run(args.run, args.plan, args.runs_root, args.registration)


if __name__ == "__main__":
    raise SystemExit(main())
