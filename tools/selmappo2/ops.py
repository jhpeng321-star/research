"""Explicit, CPU-only immutable repair staging for the registered per-CAV trainer.

Importing this module performs no I/O. Only --action stage/dispatch/status calls
the shared SSH transport. The original failed smoke and its dispatch lock remain
immutable. CPU repair checks do not authorize continuation, replay, or evaluation.
"""
import argparse
import base64
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid
import zlib


REPO = Path(__file__).absolute().parents[2]
SHARED = REPO.parent / "c4-selmappo"
WALL = REPO.parent / "c4-wallsw"
REG = Path("E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md")
DESIGN = "docs/paper_results/SELMAPPO2_DESIGN.md"
DESIGN_SHA = "09499f2af60311b3690a940dca8d94605d710465afb9c9eb05d5b0a79b23900c"
HOST = "cpuQ"
HOME = "/home/ubuntu"
PARENT = HOME + "/selmappo2_20261101"
OWNER = "S-SELMAPPO2/per_cav/20261101"
LOCK = "PER_CAV_DISPATCH_ONCE.json"
CONTINUATION_LOCK = "PER_CAV_CONTINUATION_ONCE.json"
HISTORICAL_BINDING = "b42c257b09813c175fcb7261e341c7599d0bb5c4c971529e2856890d2fc8038c"
HISTORICAL_PID, HISTORICAL_START, HISTORICAL_UID = 428649, 47990900, 1000
HISTORICAL_SEED = 18107998
HISTORICAL_LEDGER_SHA = "eb53747c54b045b8a4c3447969d14ac4fc6f24e3719cdc5b18aff31d358417a9"
HISTORICAL_ERROR = "ModuleNotFoundError: No module named 'wms.control.execution'"
EVIDENCE = REPO / "docs/paper_results/selmappo2"
BUILD = re.compile(r"build_[0-9a-f]{64}\Z")
FORBIDDEN = re.compile(
    r"sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|"
    r"vehicle_params|driver_params|vtype|\.rou\.xml", re.I)
EXPOSURE = re.compile(r"(?<!\d)181\d{5}(?!\d)")
PRUNED = frozenset((".git", "node_modules", ".venv", "venv", "__pycache__",
                    "miniconda3", "miniconda", ".cache", "site-packages"))
SHARED_PY = (
    "__init__", "audit", "continuation", "dispatch", "environment", "gate",
    "guards", "handover", "integration", "launch", "learning", "observations",
    "ops", "postmortem", "progress", "protocol", "registration_history", "remote", "runtime", "selector",
    "storage", "test_cpu", "train", "transport",
)
OWN_PY = ("__init__", "shared", "mixed", "training", "ops", "report")
C1_PY = ("__init__", "core", "design", "storage", "sumo", "library")
TESTS = ("tests/test_selmappo2_training.py", "tests/test_selmappo2_mixed.py",
         "tests/test_selmappo2_runtime_gate.py", "tests/test_selmappo2_report.py")
EXPECTED_TESTS, EXPECTED_RUNTIME_TESTS = 93, 10


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def binding_for(sources):
    return sha(canonical(sources).encode("utf-8"))


def checked(path, *, directory=False):
    """Check lexical names BEFORE metadata; reject symlinks in every component."""
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or FORBIDDEN.search(str(path)):
        raise ValueError("unsafe/nonabsolute path")
    current = Path(path.anchor)
    mode = current.lstat().st_mode
    for part in path.parts[1:]:
        current /= part
        mode = current.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError("symlink path refused")
    if not (stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)):
        raise ValueError("wrong path type")
    return path


def relative_name(name):
    path = PurePosixPath(name)
    if (not isinstance(name, str) or not name or "\\" in name
            or path.is_absolute() or ".." in path.parts or ":" in name
            or path.as_posix() != name or FORBIDDEN.search(name)):
        raise ValueError("unsafe bundle name")
    return name


def exclusive_json(path, payload):
    path = Path(path)
    checked(path.parent, directory=True)
    if FORBIDDEN.search(str(path)):
        raise ValueError("unsafe receipt path")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(payload) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)


def read_json(path):
    return json.loads(checked(path).read_text(encoding="utf-8"))


def mkdir(path):
    """Create exactly one new directory under an already verified parent."""
    path = Path(path)
    checked(path.parent, directory=True)
    if FORBIDDEN.search(str(path)):
        raise ValueError("unsafe directory")
    path.mkdir(exist_ok=False)


def bundle():
    """Explicit reviewed sources only, with exact bytes (including registration)."""
    origins = {}
    for root, package, names in ((SHARED, "tools/selmappo", SHARED_PY),
                                 (REPO, "tools/selmappo2", OWN_PY)):
        folder = checked(root / package, directory=True)
        actual = {p.name for p in folder.glob("*.py")}
        expected = {n + ".py" for n in names}
        if actual != expected:
            raise ValueError(f"Python source inventory changed in {package}; explicit review required: "
                             f"added={sorted(actual - expected)}, missing={sorted(expected - actual)}")
        origins.update({f"{package}/{n}.py": root / package / (n + ".py") for n in names})
    origins.update({f"tools/c1switch/{n}.py": WALL / "tools/c1switch" / (n + ".py")
                    for n in C1_PY})
    for name in ("traffic_shepherd/c4/switchable_wall.py", "src/wms/control/wall.py"):
        origins[name] = WALL / name
    origins.update({name: REPO / name for name in TESTS})
    origins.update({DESIGN: REPO / DESIGN, "REGISTRATION.md": REG,
                    "WALLSW_HANDOFF.md": WALL / "HANDOFF.md"})
    files = {relative_name(name): checked(path).read_bytes()
             for name, path in sorted(origins.items())}
    if sha(files[DESIGN]) != DESIGN_SHA:
        raise ValueError("registered per-CAV design digest mismatch")
    from .shared import load
    from .training import validate_allocation
    package = load(SHARED)
    protocol = importlib.import_module(package.__name__ + ".protocol")
    integration = importlib.import_module(package.__name__ + ".integration")
    validate_allocation(SimpleNamespace(protocol=protocol))
    integration.registered(REG, DESIGN_SHA, variant="per_cav")
    if not protocol.approval_line(REG):
        raise PermissionError("shared MAIN approval missing")
    if not protocol.audit_registration(REG, variant="per_cav", design_sha=DESIGN_SHA)["passed"]:
        raise ValueError("registration allocation conflict")
    # Close the read/validation race: the transported bytes must be the audited text.
    if any(checked(origins[name]).read_bytes() != data for name, data in files.items()):
        raise ValueError("source/registration changed during snapshot")
    sources = {name: sha(data) for name, data in files.items()}
    return files, sources, binding_for(sources)


def historical_failure():
    """Validate only the pinned failed run; never import or modify its old code.

    This is the sole content-reading exception to the metadata census: exact own
    ownership/dispatch receipts and the SHA-pinned, three-event failure ledger.
    Neither an absent original root nor a repaired source tree implies a retry.
    """
    root = Path(PARENT) / ("build_" + HISTORICAL_BINDING)
    try:
        checked(root.parent, directory=True)
        root.lstat()
    except FileNotFoundError:
        return None

    def owned(path, *, directory=False):
        path = checked(path, directory=directory)
        if path.lstat().st_uid != HISTORICAL_UID or os.getuid() != HISTORICAL_UID:
            raise ValueError("historical filesystem uid mismatch")
        return path

    owned(root, directory=True)
    owner = read_json(owned(root / "OWNER.json"))
    dispatch = read_json(owned(root / "DISPATCH.json"))
    identity = dict(owner=OWNER, host=HOST, root=str(root), binding=HISTORICAL_BINDING)
    if (any(owner.get(k) != v or dispatch.get(k) != v for k, v in identity.items())
            or owner.get("design_sha256") != DESIGN_SHA
            or not owner.get("files") or binding_for(owner["files"]) != HISTORICAL_BINDING
            or dispatch.get("pid") != HISTORICAL_PID or dispatch.get("uid") != HISTORICAL_UID
            or dispatch.get("proc_start_ticks") != HISTORICAL_START
            or dispatch.get("command") != training_command(root)
            or dispatch.get("out") != str(root / "training")):
        raise ValueError("historical OWNER/DISPATCH identity mismatch")
    for name in owner["files"]:
        relative_name(name)
    lock = read_json(owned(root.parent / LOCK))
    if lock != dict(identity, command=dispatch["command"]):
        raise ValueError("historical parent dispatch lock mismatch")

    def stopped_pid():
        # No process enumeration or signals; even a reused PID is refused.
        procdir = Path("/proc") / str(HISTORICAL_PID)
        try:
            owned(procdir, directory=True)
            fields = owned(procdir / "stat").read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            return "exited"
        if fields[0] != "Z" or int(fields[19]) != HISTORICAL_START:
            raise ValueError("historical PID still live or reused; exclusion refused")
        return "own_zombie"

    stopped_pid()
    training = owned(root / "training", directory=True)
    episode = owned(training / str(HISTORICAL_SEED), directory=True)
    ledger = owned(training / "coordinator.jsonl")
    raw = ledger.read_bytes()
    expected = [dict(event="START", seed=HISTORICAL_SEED, mode="smoke",
                     world=dict(seed=HISTORICAL_SEED, penetration=.4, demand=2700),
                     reserved_sim_s=900, panel=[]),
                dict(event="ACTUAL", seed=HISTORICAL_SEED, actual_sim_s=300),
                dict(event="FAILED", seed=HISTORICAL_SEED, actual_sim_s=300,
                     error=HISTORICAL_ERROR)]
    if (sha(raw) != HISTORICAL_LEDGER_SHA or not raw.endswith(b"\n")
            or [json.loads(line) for line in raw.splitlines()] != expected):
        raise ValueError("historical failure ledger hash/events mismatch")

    # Metadata only below training: no scenario, checkpoint, or result contents.
    checkpoints, pending = [], [training]
    while pending:
        parent = pending.pop()
        with os.scandir(owned(parent, directory=True)) as entries:
            for entry in entries:
                path = parent / entry.name
                if FORBIDDEN.search(path.as_posix()):
                    continue
                if entry.is_symlink():
                    raise ValueError("historical training symlink refused")
                if any(int(match[0]) != HISTORICAL_SEED for match in EXPOSURE.finditer(entry.name)):
                    raise ValueError("historical tree contains another exposed seed")
                if entry.name in ("result.jsonl", "training.jsonl", "FROZEN_FINAL.json",
                                  "evaluation.jsonl"):
                    raise ValueError("historical successful result/update present")
                if entry.is_dir(follow_symlinks=False):
                    if parent == training and path != episode:
                        raise ValueError("historical extra episode directory")
                    pending.append(path)
                elif entry.is_file(follow_symlinks=False):
                    if entry.name.startswith("checkpoint_"):
                        if path != training / "checkpoint_000.pt":
                            raise ValueError("historical checkpoint beyond initial000")
                        owned(path)
                        checkpoints.append(path)
                else:
                    raise ValueError("historical special file refused")
    if checkpoints != [training / "checkpoint_000.pt"]:
        raise ValueError("historical initial checkpoint missing")
    # Recheck the immutable debit and stopped process after the metadata census.
    if owned(ledger).read_bytes() != raw:
        raise ValueError("historical ledger changed during validation")
    state = stopped_pid()
    return dict(path=str(episode), root=str(root), owner=OWNER, binding=HISTORICAL_BINDING,
                pid=HISTORICAL_PID, uid=HISTORICAL_UID, proc_start_ticks=HISTORICAL_START,
                process_state=state, ledger_sha256=HISTORICAL_LEDGER_SHA,
                seed=HISTORICAL_SEED, mode="smoke", status="FAILED",
                reserved_sim_s=900, actual_sim_s=300, error=HISTORICAL_ERROR,
                charge_retained=True, successful_results=0, checkpoints=["checkpoint_000.pt"],
                retry_authorized=False, continuation_authorized=False, new_sumo_seconds=0)


def inventory(roots):
    """Metadata census with a separately validated, exact historical exclusion."""
    roots = list(roots)
    if not roots:
        raise ValueError("empty exposure roots")
    count, hits, errors, skipped, directories = 0, set(), [], 0, 0
    exclusions, pruned_history = [], []
    historical_root = Path(PARENT) / ("build_" + HISTORICAL_BINDING)
    if any(Path(r) == historical_root or Path(r) in historical_root.parents for r in roots):
        try:
            historical = historical_failure()
            if historical is not None:
                exclusions.append(historical)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            errors.append(f"historical exclusion refused: {type(exc).__name__}: {exc}")
    excluded_path = Path(exclusions[0]["path"]) if exclusions else None
    for root in roots:
        try:
            root = checked(root, directory=True)
        except (OSError, ValueError) as exc:
            errors.append(f"root unavailable: {type(exc).__name__}: {exc}")
            continue
        if EXPOSURE.search(root.as_posix()):
            hits.add(root.as_posix())
        pending = [root]
        while pending:
            parent = pending.pop()
            try:
                checked(parent, directory=True)
                with os.scandir(parent) as entries:
                    directories += 1
                    for entry in entries:
                        path = parent / entry.name
                        # Prune before stat/is_dir: even forbidden metadata is not inspected.
                        if FORBIDDEN.search(path.as_posix()) or entry.name in PRUNED:
                            skipped += 1
                            continue
                        try:
                            if path == excluded_path:
                                checked(path, directory=True)
                                if (entry.is_symlink() or not entry.is_dir(follow_symlinks=False)
                                        or path.lstat().st_uid != HISTORICAL_UID):
                                    raise ValueError("validated historical episode metadata changed")
                                pruned_history.append(str(path))
                                continue
                            if EXPOSURE.search(path.as_posix()):
                                hits.add(path.as_posix())
                            if entry.is_symlink():
                                skipped += 1
                                continue
                            if entry.is_dir(follow_symlinks=False):
                                pending.append(path)
                            elif entry.is_file(follow_symlinks=False):
                                count += 1
                        except (OSError, ValueError) as exc:
                            errors.append(f"{path}: {exc}")
            except (OSError, ValueError) as exc:
                errors.append(f"{parent}: {exc}")
    if excluded_path is not None and str(excluded_path) not in pruned_history:
        errors.append("validated historical episode was not observed/pruned in the census")
    return dict(roots=[str(r) for r in roots], safe_filenames=count,
                directories=directories, pruned=skipped, conflicts=sorted(hits),
                traversal_errors=errors, filename_metadata_only=True,
                historical_exclusions=exclusions, historical_paths_pruned=pruned_history,
                registration_design_contents_scanned=False,
                passed=bool(count and directories) and not hits and not errors)


def resources():
    if sys.platform != "linux" or str(Path.home()) != HOME:
        raise ValueError("cpuQ Linux home ownership mismatch")
    checked(HOME, directory=True)
    disk_root = Path(PARENT) if Path(PARENT).exists() else Path(HOME)
    checked(disk_root, directory=True)
    memory = next(int(line.split()[1]) * 1024 for line in
                  checked("/proc/meminfo").read_text().splitlines()
                  if line.startswith("MemAvailable:"))
    import shutil
    cpus, load = os.cpu_count(), os.getloadavg()[0]
    disk = shutil.disk_usage(disk_root).free
    admitted = (cpus is not None and cpus >= 14 and math.isfinite(load)
                and 0 <= load and load + 1 <= min(.8 * cpus, cpus - 2, 24)
                and memory >= 8 * 1024**3 and disk >= 2 * 1024**3)
    return dict(host=HOST, owner=OWNER, cpus=cpus, load=load, memory_available=memory,
                disk_free=disk, admitted=admitted, timestamp_ns=time.time_ns())


def environment(source):
    runtime = checked(HOME + "/e3final/runtime", directory=True)
    checked(runtime / "src", directory=True)
    checked(runtime / "traffic_shepherd", directory=True)
    sumo = checked(HOME + "/miniconda3/lib/python3.10/site-packages/sumo", directory=True)
    checked(sumo / "bin/sumo")
    return dict(os.environ, PYTHONPATH=os.pathsep.join((str(source), str(runtime / "src"),
                                                      str(runtime))),
                PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", SUMO_HOME=str(sumo),
                PATH=str(sumo / "bin") + ":" + HOME + "/miniconda3/bin:/usr/local/bin:/usr/bin:/bin",
                C1SWITCH_RUNTIME=HOME + "/e3final", C1SWITCH_WRAPPERS=HOME + "/wmdata20261031",
                C1SWITCH_MAPPO=HOME + "/mappospeed/production2/policy_00169.pt",
                C1SWITCH_ENGINE_ROOT=HOME + "/c1switch20261031/fullrerun_v3")


def verify_sources(source, sources, binding):
    checked(source, directory=True)
    if not sources or binding_for(sources) != binding:
        raise ValueError("empty/mismatched source manifest")
    for name, expected in sources.items():
        relative_name(name)
        if sha(checked(source / name).read_bytes()) != expected:
            raise ValueError(f"immutable source mismatch: {name}")
    # No additional importable source, unbound file, or symlink may enter the snapshot.
    found, pending = set(), [source]
    while pending:
        parent = pending.pop()
        with os.scandir(checked(parent, directory=True)) as entries:
            for entry in entries:
                name = relative_name((parent / entry.name).relative_to(source).as_posix())
                if entry.is_symlink():
                    raise ValueError("symlink in immutable snapshot")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(parent / entry.name)
                elif entry.is_file(follow_symlinks=False):
                    found.add(name)
                else:
                    raise ValueError("special file in immutable snapshot")
    if found != sources.keys():
        raise ValueError("unexpected/missing immutable source file")


def _cpu_child(source, root, action):
    """Executed only by the foreground remote Linux integration subprocess."""
    from . import training
    training.configure_cpu()
    shared = training.load_shared(source)
    lib = shared.runtime.library(source)
    # setup supplies BOTH runtime/src and runtime (traffic_shepherd) before imports.
    lib.setup()
    import traci
    from unittest.mock import patch
    with patch.object(traci, "start", side_effect=AssertionError("CPU gate: no SUMO start")) as start, \
            patch.object(traci, "simulationStep",
                         side_effect=AssertionError("CPU gate: no SUMO step")) as step:
        importlib.import_module("tools.c1switch.library").install_wall_extension()
        if action == "tests":
            import pytest

            class StagedSources:
                def __init__(self):
                    self.collected, self.skipped = 0, []
                    self.suites = {name: 0 for name in TESTS}

                def pytest_collection_modifyitems(self, items):
                    self.collected = len(items)
                    for item in items:
                        module = item.module
                        name = Path(module.__file__).relative_to(source).as_posix()
                        self.suites[name] += 1
                        if hasattr(module, "FLEET_ROOT"):
                            module.FLEET_ROOT = source / "tools/selmappo"
                        if hasattr(module, "SHARED"):
                            module.SHARED = source / "tools/selmappo"
                        if hasattr(module, "WALL"):
                            module.WALL = source / "traffic_shepherd/c4"

                def pytest_runtest_logreport(self, report):
                    if report.skipped:
                        self.skipped.append(report.nodeid)

                def pytest_collectreport(self, report):
                    if report.skipped:
                        self.skipped.append(report.nodeid)

            plugin = StagedSources()
            # Synthetic fixtures deliberately create seed-named episode directories.
            # Keep these in exclusive /tmp scratch, outside the real home exposure census.
            checked("/tmp", directory=True)
            temp = Path("/tmp") / ("selmappo2_cpu_" + uuid.uuid4().hex)
            if temp.exists() or temp.is_symlink():
                raise ValueError("existing pytest workspace; no retry")
            result = pytest.main(["-p", "no:cacheprovider", "-q", "--basetemp", str(temp),
                                  *[str(source / name) for name in TESTS]], plugins=[plugin])
            if (result or plugin.collected != EXPECTED_TESTS or plugin.skipped
                    or not all(plugin.suites.values())
                    or plugin.suites["tests/test_selmappo2_runtime_gate.py"] != EXPECTED_RUNTIME_TESTS):
                raise ValueError(f"CPU suites failed/count mismatch/skipped: code={result}, "
                                 f"collected={plugin.collected}/{EXPECTED_TESTS}, "
                                 f"suites={plugin.suites}, skipped={plugin.skipped}")
            report = dict(status="CPU_TEST_PASS", collected=plugin.collected, skipped=0,
                          suites=plugin.suites, scratch=str(temp), staged_fixture_paths=True)
        elif action == "gate":
            gate = importlib.import_module(shared.environment.__package__ + ".gate")
            report = gate.check(source)
            training.validate_allocation(shared)
            frames = training.validate_environment(shared)
            training.validate_learning(shared, frames)
            from traffic_shepherd import seeds, scenarios
            from traffic_shepherd.c4 import seeds as c4seeds, env
            targets = ((env, "check_case"), (c4seeds, "check_case"),
                       (seeds, "require_development_case"),
                       (scenarios, "require_development_case"))
            originals = [getattr(module, key) for module, key in targets]
            for seed, smoke in ((training.FIRST_SEED, False), (training.LAST_SEED, False),
                                *[(seed, True) for seed in training.SMOKE_SEEDS]):
                world = (dict(seed=seed, penetration=.4, demand=2700) if smoke else
                         shared.protocol.training_world(seed, "per_cav"))
                with shared.guards.episode_guards(world, "per_cav", smoke=smoke):
                    for module, key in targets:
                        guard = getattr(module, key)
                        args = (seed, world["demand"])
                        guard(*args, *(["exposed"] if key != "check_case" else []))
                        bad = [(seed + 1, world["demand"]), (seed, world["demand"] + 1)]
                        for case in bad:
                            try:
                                guard(*case, *(["exposed"] if key != "check_case" else []))
                            except ValueError:
                                pass
                            else:
                                raise ValueError("per-CAV captured guard admitted another case")
                        if key != "check_case":
                            try:
                                guard(*args, "unexposed")
                            except ValueError:
                                pass
                            else:
                                raise ValueError("per-CAV guard admitted wrong role")
                if any(getattr(m, k) is not old for (m, k), old in zip(targets, originals)):
                    raise ValueError("captured frozen aliases not restored")
            shared.integration.registered(source / "REGISTRATION.md", DESIGN_SHA, variant="per_cav")
            if (not shared.protocol.approval_line(source / "REGISTRATION.md")
                    or not shared.protocol.audit_registration(source / "REGISTRATION.md",
                                                               variant="per_cav",
                                                               design_sha=DESIGN_SHA)["passed"]):
                raise ValueError("staged registration approval/allocation mismatch")
            report.update(variant="per_cav", per_cav_frozen_aliases=True,
                          allocation_validated=True, environment_validated=True,
                          learning_validated=True, design_sha256=DESIGN_SHA)
        else:
            raise ValueError("unknown CPU check")
        if start.called or step.called:
            raise ValueError("CPU integration attempted simulator access")
    report["new_sumo_seconds"] = 0
    print("CHECK_RECEIPT", canonical(report), flush=True)


def tagged(stdout, tag):
    return [json.loads(line[len(tag) + 1:]) for line in stdout.splitlines()
            if line.startswith(tag + " ")]


def run_check(source, root, env, action):
    code = ("from pathlib import Path; from tools.selmappo2.ops import _cpu_child; "
            f"_cpu_child(Path({str(source)!r}), Path({str(root)!r}), {action!r})")
    try:
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=source, env=env,
                                capture_output=True, text=True, timeout=300)
        rows = tagged(result.stdout, "CHECK_RECEIPT")
        valid = (len(rows) == 1 and rows[0].get("new_sumo_seconds") == 0
                 and rows[0].get("status") == ("CPU_TEST_PASS" if action == "tests"
                                               else "CPU_INTEGRATION_PASS")
                 and (action == "tests" or rows[0].get("variant") == "per_cav"))
        return dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                    passed=result.returncode == 0 and valid, checks=rows)
    except subprocess.TimeoutExpired as exc:
        def text(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")
        return dict(returncode=None, stdout=text(exc.stdout), stderr=text(exc.stderr),
                    passed=False, error="CPU check timeout")


def _remote_stage(files, sources, binding, root):
    cpu = resources()
    if not cpu["admitted"]:
        raise RuntimeError("CPU resource admission denied")
    mkdir(root)
    owner = dict(owner=OWNER, host=HOST, root=str(root), binding=binding,
                 design_sha256=DESIGN_SHA, files=sources)
    exclusive_json(root / "OWNER.json", owner)
    source = root / "pytools"
    record = dict(owner, cpu=cpu, tests=dict(passed=False, error="not run"),
                  gate=dict(passed=False, error="not run"), audit_passed=False,
                  new_sumo_seconds=0, passed=False, cpu_only_repair=True,
                  continuation_authorized=False, historical_exclusions=[])
    try:
        audit = inventory([HOME])
        audit.update(owner=OWNER, root=str(root), binding=binding)
        exclusive_json(root / "AUDIT.json", audit)
        print("AUDIT_RECEIPT", canonical(audit), flush=True)
        record["audit_passed"] = audit["passed"]
        record["historical_exclusions"] = audit["historical_exclusions"]
        if not audit["passed"]:
            raise ValueError("exposure conflict or traversal error")
        mkdir(source)
        for name, encoded in sorted(files.items()):
            relative_name(name)
            data = base64.b64decode(encoded, validate=True)
            if sha(data) != sources[name]:
                raise ValueError("transport source mismatch")
            path = source / name
            for parent in reversed(path.parents):
                if parent == source or source in parent.parents:
                    if not parent.exists():
                        mkdir(parent)
                    checked(parent, directory=True)
            with path.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            path.chmod(0o444)
        verify_sources(source, sources, binding)
        env = environment(source)
        exclusive_json(root / "ENV.json", env)
        os.nice(19)
        record["tests"] = run_check(source, root, env, "tests")
        if not record["tests"]["passed"]:
            raise ValueError("CPU test suites failed")
        record["gate"] = run_check(source, root, env, "gate")
        if not record["gate"]["passed"]:
            raise ValueError("CPU real integration gate failed")
        verify_sources(source, sources, binding)
        # Freeze the snapshot tree; training output lives beside it.
        for parent, dirs, _ in os.walk(source, topdown=False, followlinks=False):
            for name in dirs:
                checked(Path(parent) / name, directory=True).chmod(0o555)
            checked(parent, directory=True).chmod(0o555)
        record["passed"] = True
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    cpu_record = dict(owner, resources=cpu, tests=record["tests"], gate=record["gate"],
                      passed=record["tests"]["passed"] and record["gate"]["passed"],
                      new_sumo_seconds=0, cpu_only_repair=True, continuation_authorized=False,
                      historical_exclusions=record["historical_exclusions"])
    exclusive_json(root / "CPU.json", cpu_record)
    print("CPU_RECEIPT", canonical(cpu_record), flush=True)
    exclusive_json(root / "STAGE.json", record)
    print("STAGE_RECEIPT", canonical(record), flush=True)
    if not record["passed"]:
        raise RuntimeError("CPU stage failed; retained for inspection, no restage")
    receipt = dict(schema="selmappo.integration/2", status="CPU_INTEGRATION_PASS",
                   owner=OWNER, variant="per_cav", protocol="S-C1SWITCH2/v2", cadence_s=20,
                   library=["native", "ctrl0", "mappo", "pi", "circles", "wall"],
                   design_sha256=DESIGN_SHA, wall_module="tools.c1switch.library",
                   binding=binding, files=sources,
                   registration_sha256=sources["REGISTRATION.md"], new_sumo_seconds=0,
                   cpu_only_repair=True, continuation_authorized=False,
                   historical_exclusions=record["historical_exclusions"])
    exclusive_json(root / "INTEGRATION.json", receipt)


def own_stage(root, sources, binding, *, require_pass=True):
    checked(root, directory=True)
    if root.stat().st_uid != os.getuid():
        raise ValueError("build filesystem ownership mismatch")
    owner = read_json(root / "OWNER.json")
    for key, value in dict(owner=OWNER, host=HOST, root=str(root), binding=binding,
                           design_sha256=DESIGN_SHA, files=sources).items():
        if owner.get(key) != value:
            raise ValueError("build ownership/current source mismatch")
    source = root / "pytools"
    stage = read_json(root / "STAGE.json")
    if any(stage.get(key) != owner[key] for key in owner):
        raise ValueError("stage ownership mismatch")
    if not require_pass:
        if source.exists():
            verify_sources(source, sources, binding)
        return source, stage, None
    verify_sources(source, sources, binding)
    receipt = read_json(root / "INTEGRATION.json")
    if (not stage.get("passed") or not stage["tests"]["passed"] or not stage["gate"]["passed"]
            or not stage["audit_passed"] or stage["new_sumo_seconds"] != 0
            or receipt.get("owner") != OWNER or receipt.get("variant") != "per_cav"
            or receipt.get("binding") != binding or receipt.get("files") != sources
            or receipt.get("design_sha256") != DESIGN_SHA
            or receipt.get("registration_sha256") != sources["REGISTRATION.md"]
            or receipt.get("schema") != "selmappo.integration/2"
            or receipt.get("status") != "CPU_INTEGRATION_PASS"
            or receipt.get("new_sumo_seconds") != 0
            or receipt.get("protocol") != "S-C1SWITCH2/v2" or receipt.get("cadence_s") != 20
            or receipt.get("library") != ["native", "ctrl0", "mappo", "pi", "circles", "wall"]
            or receipt.get("wall_module") != "tools.c1switch.library"):
        raise ValueError("integration/stage receipt mismatch")
    audit = read_json(root / "AUDIT.json")
    if not audit.get("passed") or audit.get("binding") != binding or audit.get("owner") != OWNER:
        raise ValueError("exposure audit receipt mismatch")
    cpu = read_json(root / "CPU.json")
    if (not cpu.get("passed") or cpu.get("owner") != OWNER or cpu.get("binding") != binding
            or cpu.get("tests") != stage["tests"] or cpu.get("gate") != stage["gate"]):
        raise ValueError("CPU integration receipt mismatch")
    return source, stage, receipt


def continuation_authority(registration):
    lines = checked(registration).read_text(encoding="utf-8").splitlines()
    matches = [line for line in lines
               if line.startswith("- 01:48 MAIN ACKNOWLEDGEMENTS / DECISIONS:")
               and "S-SELMAPPO2 same-cap continuation ACKNOWLEDGED "
                   "(unused second smoke world only, then production)" in line]
    if len(matches) != 1:
        raise PermissionError("specific MAIN01:48 per-CAV continuation acknowledgement absent")
    return matches[0]


def continuation_record(root, sources, binding):
    authority = continuation_authority(root / "pytools/REGISTRATION.md")
    history = historical_failure()
    if history is None:
        raise ValueError("exact original stopped run absent")
    return dict(owner=OWNER, host=HOST, root=str(root), binding=binding,
                authority=authority, original=history, inherited_reserved_sim_s=900,
                inherited_actual_sim_s=300, skipped_seed=HISTORICAL_SEED,
                only_unused_smoke_seed=18107999, production_episodes=7774,
                cap_sim_s=7012800, evaluation_reserved_unstarted_sim_s=14400,
                registration_sha256=sources["REGISTRATION.md"],
                command=training_command(root, continuation=True))


def validate_continuation(root, registration):
    """Revalidate authority, old PID/ledger and exclusive lock in the trainer."""
    owner = read_json(root / "OWNER.json")
    source, _, _ = own_stage(root, owner["files"], owner["binding"])
    if checked(registration) != source / "REGISTRATION.md":
        raise ValueError("continuation registration outside immutable source")
    expected = continuation_record(root, owner["files"], owner["binding"])
    if (read_json(root / "CONTINUATION.json") != expected
            or read_json(root.parent / CONTINUATION_LOCK) != expected):
        raise ValueError("continuation receipt/one-use lock mismatch")
    return expected


def training_command(root, *, continuation=False):
    source = root / "pytools"
    return [sys.executable, "-B", "-m", "tools.selmappo2.training",
            "--shared-root", str(source), "--c1-root", str(source),
            "--integration", str(root / "INTEGRATION.json"), "--design-sha", DESIGN_SHA,
            "--registration", str(source / "REGISTRATION.md"),
            "--out", str(root / "training"), "--workers", "12"] + (
                ["--continuation"] if continuation else [])


def _remote_dispatch(root, sources, binding, *, continuation=False):
    if not continuation:
        try:
            (root.parent / LOCK).lstat()
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("original parent dispatch lock consumed; use explicit approved continuation")
    else:
        # Never inspect a new job as a historical failure or remove the old lock.
        if (root.parent / CONTINUATION_LOCK).exists():
            raise FileExistsError("one-use continuation lock consumed; no automatic retry")
    source, _, _ = own_stage(root, sources, binding)
    cpu = resources()
    if not cpu["admitted"]:
        raise RuntimeError("CPU resource admission denied")
    audit = inventory([HOME])
    audit.update(owner=OWNER, root=str(root), binding=binding)
    exclusive_json(root / "DISPATCH_AUDIT.json", audit)
    print("AUDIT_RECEIPT", canonical(audit), flush=True)
    if not audit["passed"]:
        raise ValueError("pre-dispatch exposure conflict/traversal error")
    if any((root / name).exists() or (root / name).is_symlink()
           for name in ("training", "training.log", "DISPATCH.json")):
        raise ValueError("existing dispatch artifacts; no retry")
    env = environment(source)
    if read_json(root / "ENV.json") != env:
        # SSH session variables can change; only the execution contract is compared.
        saved = read_json(root / "ENV.json")
        keys = ("PYTHONPATH", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "SUMO_HOME", "PATH",
                "C1SWITCH_RUNTIME", "C1SWITCH_WRAPPERS", "C1SWITCH_MAPPO", "C1SWITCH_ENGINE_ROOT",
                "PYTHONDONTWRITEBYTECODE", "PYTEST_DISABLE_PLUGIN_AUTOLOAD")
        if any(saved.get(key) != env[key] for key in keys):
            raise ValueError("staged CPU environment mismatch")
    command = training_command(root, continuation=continuation)
    # Exclusive parent-wide lock is consumed even if spawning subsequently fails.
    lock = dict(owner=OWNER, host=HOST, root=str(root), binding=binding, command=command)
    if continuation:
        record = continuation_record(root, sources, binding)
        exclusive_json(root.parent / CONTINUATION_LOCK, record)
        exclusive_json(root / "CONTINUATION.json", record)
    else:
        exclusive_json(root.parent / LOCK, lock)
    os.nice(19)
    verify_sources(source, sources, binding)
    with (root / "training.log").open("x", encoding="utf-8") as log:
        proc = subprocess.Popen(command, cwd=source, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(lock, pid=proc.pid, uid=os.getuid(), proc_start_ticks=None,
                  out=str(root / "training"), cpu=cpu)
    procdir = Path("/proc") / str(proc.pid)
    try:
        record["proc_start_ticks"] = proc_start(procdir)
    except FileNotFoundError:
        pass  # An immediate trainer failure is still a consumed one-time dispatch.
    exclusive_json(root / "DISPATCH.json", record)
    print("DISPATCH_RECEIPT", canonical(record), flush=True)


def proc_start(procdir):
    # /proc/<pid>/stat's parenthesized comm may itself contain spaces/parentheses.
    return int((procdir / "stat").read_text().rsplit(")", 1)[1].split()[19])


def tail(path, limit=14000):
    with checked(path).open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - limit))
        return dict(bytes=size, tail=stream.read(limit).decode("utf-8", errors="replace"))


def _remote_status(root, sources, binding):
    _, stage, _ = own_stage(root, sources, binding, require_pass=False)
    result = dict(owner=OWNER, host=HOST, root=str(root), binding=binding,
                  timestamp_ns=time.time_ns(), dispatched=False, stage_passed=stage.get("passed"),
                  stage_error=stage.get("error"), summaries={})
    dispatch_path = root / "DISPATCH.json"
    if dispatch_path.exists():
        own_stage(root, sources, binding)
        dispatch = read_json(dispatch_path)
        continuation = (root / "CONTINUATION.json").exists()
        lock = read_json(root.parent / (CONTINUATION_LOCK if continuation else LOCK))
        expected = dict(owner=OWNER, host=HOST, root=str(root), binding=binding,
                        command=training_command(root, continuation=continuation))
        if (any(lock.get(k) != v or dispatch.get(k) != v for k, v in expected.items())
                or (continuation and lock != validate_continuation(root, root / "pytools/REGISTRATION.md"))):
            raise ValueError("dispatch ownership mismatch")
        pid = dispatch["pid"]
        if type(pid) is not int or pid <= 0 or dispatch.get("uid") != os.getuid():
            raise ValueError("invalid own PID/UID")
        result.update(dispatched=True, pid=pid, alive=False)
        procdir = Path("/proc") / str(pid)
        try:
            if procdir.stat().st_uid != dispatch["uid"]:
                raise ValueError("own PID uid mismatch")
            if dispatch["proc_start_ticks"] is None or proc_start(procdir) != dispatch["proc_start_ticks"]:
                raise ValueError("own PID reused/unbound")
            raw = (procdir / "cmdline").read_bytes().rstrip(b"\0")
            command = [arg.decode() for arg in raw.split(b"\0")] if raw else []
            if command and command != dispatch["command"]:
                raise ValueError("own PID command mismatch")
            if not command:
                state = (procdir / "stat").read_text().rsplit(")", 1)[1].split()[0]
                if state not in ("Z", "X"):
                    raise ValueError("own PID has unexpected empty command")
            result.update(alive=bool(command), command=command)
        except (FileNotFoundError, ProcessLookupError):
            pass
        training_owner = root / "training/OWNER.json"
        if training_owner.exists():
            owner = read_json(training_owner)
            expected_bindings = dict(integration_sha256=sha(checked(root / "INTEGRATION.json").read_bytes()),
                                     design_sha256=DESIGN_SHA)
            expected_sources = {PurePosixPath(name).stem: value for name, value in sources.items()
                                if name.startswith("tools/selmappo/")}
            expected_sources.update({"selmappo2." + name: sources[f"tools/selmappo2/{name}.py"]
                                     for name in ("shared", "mixed", "training")})
            if (owner.get("pid") != pid or owner.get("variant") != "per_cav"
                    or owner.get("registration_sha256") != sources["REGISTRATION.md"]
                    or owner.get("bindings") != expected_bindings or owner.get("workers") != 12
                    or owner.get("sources") != expected_sources or owner.get("optimization_seed") != 1):
                raise ValueError("training ownership mismatch")
            phase = read_json(root / "training/TRAIN_OWNER.json")
            if phase.get("pid") != pid:
                raise ValueError("training phase ownership mismatch")
    elif (root / "training").exists() or (root / "training.log").exists():
        raise ValueError("training artifacts without an own dispatch receipt")
    for name in ("training.log", "training/OWNER.json", "training/TRAIN_OWNER.json",
                 "training/coordinator.jsonl", "training/training.jsonl",
                 "training/FROZEN_FINAL.json"):
        path = root / name
        if path.exists():
            result["summaries"][name] = tail(path)
    out = root / "training"
    checkpoints = []
    if out.exists():
        checked(out, directory=True)
        with os.scandir(out) as entries:
            for entry in entries:
                if re.fullmatch(r"checkpoint_\d{3}\.pt", entry.name):
                    info = checked(out / entry.name).stat()
                    checkpoints.append(dict(name=entry.name, bytes=info.st_size,
                                            mtime_ns=info.st_mtime_ns))
    result["checkpoints"] = sorted(checkpoints, key=lambda row: row["name"])
    print("STATUS_RECEIPT", canonical(result), flush=True)


def _remote_main(action, sources, binding, build):
    if sys.platform != "linux" or str(Path.home()) != HOME:
        raise ValueError("remote host/home mismatch")
    if not BUILD.fullmatch(build) or build != "build_" + binding:
        raise ValueError("invalid/mismatched build")
    if binding_for(sources) != binding:
        raise ValueError("source manifest mismatch")
    checked(HOME, directory=True)
    if action == "stage" and not Path(PARENT).exists():
        mkdir(PARENT)
    parent = Path(PARENT) if Path(PARENT).exists() else Path(HOME)
    checked(parent, directory=True)
    if parent.stat().st_uid != os.getuid():
        raise ValueError("deployment parent filesystem ownership mismatch")
    root = Path(PARENT) / build
    if action == "cpu":
        receipt = resources()
        receipt.update(root=str(root), binding=binding)
        print("CPU_RECEIPT", canonical(receipt), flush=True)
        if not receipt["admitted"]:
            raise RuntimeError("CPU resource admission denied")
    elif action == "stage":
        files = json.loads(zlib.decompress(base64.b64decode(sys.stdin.buffer.read())))
        if set(files) != set(sources):
            raise ValueError("transport manifest mismatch")
        _remote_stage(files, sources, binding, root)
    elif action in ("dispatch", "continue"):
        _remote_dispatch(root, sources, binding, continuation=action == "continue")
    elif action == "status":
        _remote_status(root, sources, binding)
    else:
        raise ValueError("unknown remote action")


def save_receipts(stdout, action, binding):
    records = [(tag, record) for tag in ("CPU", "AUDIT", "STAGE", "DISPATCH", "STATUS")
               for record in tagged(stdout, tag + "_RECEIPT")]
    if not records:
        return
    for _, record in records:
        if (record.get("binding") != binding or record.get("owner") != OWNER
                or record.get("root") != PARENT + "/build_" + binding):
            raise ValueError("remote receipt ownership mismatch")
    checked(EVIDENCE.parent, directory=True)
    if not EVIDENCE.exists():
        mkdir(EVIDENCE)
    checked(EVIDENCE, directory=True)
    for tag, record in records:
        name = f"{tag}_{action}_{binding}_{time.time_ns()}_{uuid.uuid4().hex}.json"
        exclusive_json(EVIDENCE / name, record)


def remote(action, files, sources, binding, build):
    from .shared import load
    package = load(SHARED)
    run = importlib.import_module(package.__name__ + ".remote").run
    text = files["tools/selmappo2/ops.py"].decode("utf-8")
    remote_file = PARENT + "/" + build + "/pytools/tools/selmappo2/ops.py"
    script = (f"__name__ = '_selmappo2_remote'\n__file__ = {remote_file!r}\n" + text +
              f"\n_remote_main({action!r}, {sources!r}, {binding!r}, {build!r})\n")
    # The shared transport base64-encodes its command again. Compress the bootstrap
    # to stay below Windows' CreateProcess command-line limit; files use stdin.
    packed = base64.b64encode(zlib.compress(script.encode("utf-8"))).decode("ascii")
    code = f"import base64,zlib;exec(zlib.decompress(base64.b64decode({packed!r})))"
    stdin = (base64.b64encode(zlib.compress(canonical({
        name: base64.b64encode(data).decode("ascii") for name, data in files.items()
    }).encode("utf-8"))).decode("ascii") if action == "stage" else None)
    try:
        result = run(HOST, code, stdin=stdin, timeout=900 if action == "stage" else 180)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        stdout = exc.stdout or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        save_receipts(stdout, action, binding)
        raise
    save_receipts(result.stdout, action, binding)
    tag = "DISPATCH" if action == "continue" else action.upper()
    if len(tagged(result.stdout, tag + "_RECEIPT")) != 1:
        raise ValueError("missing/ambiguous remote action receipt")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", required=True, choices=("stage", "dispatch", "continue", "status"))
    parser.add_argument("--build", help="exact build_<full source SHA256>; required after staging")
    args = parser.parse_args(argv)
    if args.action != "stage" and args.build is None:
        parser.error("dispatch/status require --build")
    if args.action == "stage":
        files, sources, binding = bundle()
        build = "build_" + binding
        if args.build is not None and args.build != build:
            parser.error("build differs from current registered source snapshot")
        checked(EVIDENCE.parent, directory=True)
        if not EVIDENCE.exists():
            mkdir(EVIDENCE)
        # Later peer registry/source additions cannot mutate this task's stage.
        snapshot = EVIDENCE / f"BUNDLE_{binding}.json"
        exclusive_json(snapshot, dict(binding=binding, sources=sources,
                       files={name: base64.b64encode(data).decode("ascii")
                              for name, data in files.items()}))
        # A separate read-only SSH resource receipt precedes any staging writes.
        remote("cpu", files, sources, binding, build)
    else:
        if not BUILD.fullmatch(args.build):
            parser.error("invalid build identifier")
        build, binding = args.build, args.build.removeprefix("build_")
        snapshot = read_json(EVIDENCE / f"BUNDLE_{binding}.json")
        sources = snapshot["sources"]
        files = {name: base64.b64decode(data, validate=True)
                 for name, data in snapshot["files"].items()}
        if (snapshot["binding"] != binding or binding_for(sources) != binding
                or {name: sha(data) for name, data in files.items()} != sources):
            raise ValueError("local immutable bundle mismatch")
    return remote(args.action, files, sources, binding, build)


if __name__ == "__main__":
    main()
