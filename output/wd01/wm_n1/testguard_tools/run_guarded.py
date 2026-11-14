"""Pinned exact-node synthetic launcher. Draft: independent source acceptance required.

The reviewed external JSON selects exact nodes and project code/config hashes.
Only the fresh E: sandbox is writable. Existing logs and evidence are retained.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys

from wm_testguard import _canonical_sha256, _lexical_deny_reason, _norm, _is_under

TOOL_DIR = _norm(os.path.dirname(__file__))
TESTS_DIR = os.path.join(TOOL_DIR, "tests")
MAIN_REPO = _norm("E:/research-dense")
WORKTREES_PARENT = _norm("E:/rd-wt")
COORDINATOR_REVIEW_DIR = _norm("E:/tmp/review")
GUARD_EXIT_CODE = 87
PREFLIGHT_FAIL_EXIT = 3
_ALLOWED_PYTEST_ARGS = ("-q", "-v", "--disable-warnings")
_GUARD_FILES = ("sitecustomize.py", "wm_testguard.py", "run_guarded.py")

_CHILD_BOOTSTRAP = (
    "import os, sys\n"
    "try:\n"
    "    _g = sys.modules.get('wm_testguard')\n"
    "    if _g is None: raise RuntimeError('sitecustomize guard missing; no repair')\n"
    "    _g.assert_startup(strict_pid=True)\n"
    "    import json\n"
    "    _args = json.loads(sys.argv[1])\n"
    "    _g.assert_pytest_argv(_args)\n"
    "    _g.ensure_required_wrappers()\n"
    "    import pytest\n"
    "except BaseException as _exc:\n"
    "    sys.stderr.write('[wm-testguard] bootstrap refused: %r\\n' % (_exc,))\n"
    "    sys.stderr.flush()\n"
    "    os._exit(87)\n"
    "raise SystemExit(pytest.main(_args))\n"
)
_PREFLIGHT_CODE = (
    "import os, sys\n"
    "try:\n"
    "    g = sys.modules.get('wm_testguard')\n"
    "    if g is None: raise RuntimeError('sitecustomize guard missing; no repair')\n"
    "    g.assert_startup(strict_pid=True)\n"
    "    g.ensure_required_wrappers()\n"
    "    g.selfcheck()\n"
    "    if not g.wrappers_active(require_all=True): raise RuntimeError('incomplete wrappers')\n"
    "except BaseException as exc:\n"
    "    sys.stderr.write('[wm-testguard] preflight refused: %r\\n' % (exc,))\n"
    "    sys.stderr.flush()\n"
    "    os._exit(87)\n"
    "os._exit(0)\n"
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_roots(extra_roots=()):
    roots = {MAIN_REPO}
    current = TOOL_DIR
    while os.path.dirname(current) != current:
        if os.path.exists(os.path.join(current, ".git")):
            roots.add(current)
            break
        current = os.path.dirname(current)
    # Metadata only: never open worktree contents or outcome files.
    if os.path.isdir(WORKTREES_PARENT):
        with os.scandir(WORKTREES_PARENT) as entries:
            for entry in entries:
                if entry.is_dir():
                    roots.add(_norm(entry.path))
    for root in extra_roots:
        roots.add(_norm(root))
    return sorted(roots)


def _safe_source_path(path, roots):
    text = os.fspath(path)
    if _lexical_deny_reason(text):
        raise ValueError("D: path refused before resolution")
    lexical = os.path.normcase(os.path.abspath(text))
    resolved = _norm(text)
    if _lexical_deny_reason(resolved):
        raise ValueError("resolved D: path refused")
    for candidate in (lexical, resolved):
        if any(marker in os.path.basename(candidate).lower() for marker in ("test_cal_split", "cal_split_a4")):
            raise ValueError("prohibited real-split path refused before file access")
        if any(_is_under(candidate, os.path.join(root, leaf)) for root in roots for leaf in ("runs", "archive")):
            raise ValueError("protected root refused before file access")
    return resolved


def validate_node(node, roots=()):
    if not isinstance(node, str) or node != node.strip():
        raise ValueError("invalid exact node")
    if ".." in node.replace("\\", "/").split("/"):
        raise ValueError("node traversal refused")
    file_part, separator, selected = node.partition("::")
    if not separator or not selected or not os.path.isabs(file_part):
        raise ValueError("exact absolute file.py::test_name required; broad selection refused")
    parts = selected.split("::")
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\[[A-Za-z0-9_. -]+\])?", part) for part in parts):
        raise ValueError("unsupported exact node syntax")
    if not parts[-1].startswith("test_"):
        raise ValueError("last node component must be a test function")
    resolved = _safe_source_path(file_part, roots)
    if not os.path.basename(resolved).startswith("test_") or not resolved.endswith(".py"):
        raise ValueError("selected file is not a test source")
    return resolved + "::" + selected


def load_allowlist(path, expected_sha256, roots=(), review_dir=COORDINATOR_REVIEW_DIR):
    n = _safe_source_path(path, roots)
    if not _is_under(n, _norm(review_dir)) or not n.endswith(".json"):
        raise ValueError("allowlist must be JSON in coordinator review directory")
    if not re.fullmatch("[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("reviewed allowlist SHA256 required")
    with open(n, "rb") as stream:
        raw = stream.read()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != expected_sha256.lower():
        raise ValueError("allowlist SHA256 mismatch")
    payload = json.loads(raw)
    if payload.get("version") != 3 or not isinstance(payload.get("files"), dict):
        raise ValueError("version 3 manifest with files mapping required")
    raw_nodes = payload.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError("nonempty exact node list required")
    nodes = [validate_node(node, roots) for node in raw_nodes]
    if len(set(nodes)) != len(nodes):
        raise ValueError("duplicate exact nodes")
    files = {}
    # Validate role and path BEFORE any selected file is hashed.
    for path, record in payload["files"].items():
        target = _safe_source_path(path, roots)
        if not os.path.isabs(path) or target in files or not isinstance(record, dict):
            raise ValueError("manifest files must be unique absolute paths")
        role, sha = record.get("role"), record.get("sha256")
        suffix = os.path.splitext(target)[1].lower()
        if role == "code":
            valid_type = suffix == ".py"
        elif role == "config":
            valid_type = suffix in {".json", ".toml", ".ini", ".cfg", ".yaml", ".yml", ".md"}
        else:
            valid_type = False
        if not valid_type or not isinstance(sha, str) or not re.fullmatch("[0-9a-fA-F]{64}", sha):
            raise ValueError("only explicitly reviewed code/config SHA pins are permitted")
        files[target] = {"role": role, "sha256": sha.lower()}
    for node in nodes:
        if files.get(node.partition("::")[0], {}).get("role") != "code":
            raise ValueError("selected test source needs an explicit code SHA pin")
    pythonpath = [TOOL_DIR]
    for directory in payload.get("pythonpath", []):
        target = _safe_source_path(directory, roots)
        if not any(_is_under(p, target) and rec["role"] == "code" for p, rec in files.items()):
            raise ValueError("import directory needs reviewed source descendants")
        if target not in pythonpath:
            pythonpath.append(target)
    helpers = {}
    for path, sha in payload.get("helpers", {}).items():
        target = _safe_source_path(path, roots)
        if not isinstance(sha, str) or not re.fullmatch("[0-9a-fA-F]{64}", sha):
            raise ValueError("CPU helper needs reviewed SHA pin")
        if os.path.basename(target).lower().startswith(("python", "cmd", "powershell", "pwsh")):
            raise ValueError("interpreters/shells cannot be CPU helpers")
        helpers[target] = sha.lower()
    for target, record in files.items():
        if _sha256(target) != record["sha256"]:
            raise ValueError("reviewed source/config SHA mismatch: " + target)
    for target, sha in helpers.items():
        if _sha256(target) != sha:
            raise ValueError("CPU helper SHA mismatch")
    return {"path": n, "sha256": actual_sha, "nodes": nodes,
            "files": files, "pythonpath": pythonpath, "helpers": helpers}


def validate_pytest_extras(tokens):
    if any(token not in _ALLOWED_PYTEST_ARGS for token in tokens):
        raise ValueError("refused pytest argument")
    return list(tokens)


def create_sandbox(root, label, roots):
    root = _safe_source_path(root, roots)
    if os.path.splitdrive(root)[0].lower() != "e:" or not os.path.isdir(root):
        raise ValueError("sandbox parent must be an existing E: directory")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", label):
        raise ValueError("unsafe sandbox label")
    if any(_is_under(root, p) or _is_under(p, root) for p in (*roots, TOOL_DIR, _norm(sys.prefix))):
        raise ValueError("sandbox parent overlaps source/frozen/runtime roots")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    sandbox = os.path.join(root, "%s_%s_%d" % (label, stamp, os.getpid()))
    os.mkdir(sandbox)  # exclusive: collisions refuse, never reuse or clean
    for leaf in ("tmp", "data", "basetemp"):
        os.mkdir(os.path.join(sandbox, leaf))
    return sandbox


def write_pytest_ini(sandbox):
    path = os.path.join(sandbox, "pytest.ini")
    with open(path, "x", encoding="utf-8") as stream:
        stream.write("[pytest]\naddopts = -p no:cacheprovider\ntmp_path_retention_policy = all\n")
    return path


def basetemp_path(sandbox):
    return os.path.join(sandbox, "basetemp", "pytest_new")


def build_spec(sandbox, roots, manifest, pytest_args, pytest_ini):
    files = dict(manifest["files"])
    for name in _GUARD_FILES:
        path = os.path.join(TOOL_DIR, name)
        actual = _sha256(path)
        if path in files and files[path]["sha256"] != actual:
            raise ValueError("guard source changed since review")
        files[path] = {"role": "code", "sha256": actual}
    files[pytest_ini] = {"role": "config", "sha256": _sha256(pytest_ini)}
    return {"version": 3, "sandbox": sandbox, "sandbox_tmp": os.path.join(sandbox, "tmp"),
            "roots": roots, "files": files, "helpers": manifest["helpers"],
            "allowlist_path": manifest["path"], "allowlist_sha256": manifest["sha256"],
            "nodes": manifest["nodes"], "pythonpath": manifest["pythonpath"],
            "pytest_args": pytest_args,
            "interpreter": {"path": _norm(sys.executable), "sha256": _sha256(sys.executable)}}


def build_child_env(spec):
    env = os.environ.copy()
    for key in ("PYTHONHOME", "PYTHONSTARTUP"):
        env.pop(key, None)
    env.update({"PYTHONPATH": os.pathsep.join(spec["pythonpath"]),
                "WM_TESTGUARD_REQUIRE_GUARD": "1",
                "WM_TESTGUARD_POLICY_SPEC": json.dumps(spec, sort_keys=True),
                "WM_TESTGUARD_POLICY_SHA256": _canonical_sha256(spec),
                "WD_FORBID_D": "1", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTEST_ADDOPTS": "",
                "TMP": spec["sandbox_tmp"], "TEMP": spec["sandbox_tmp"], "TMPDIR": spec["sandbox_tmp"]})
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = "4"
    return env


def _preflight(python, env, sandbox):
    proc = subprocess.run([python, "-c", _PREFLIGHT_CODE], env=env, cwd=sandbox,
                          capture_output=True, text=True, timeout=120)
    return proc.returncode == 0, {"exit": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def _receipt(sandbox, phase, payload):
    path = os.path.join(sandbox, phase + ".receipt.json")
    with open(path, "x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pinned synthetic-test launcher")
    parser.add_argument("--allowlist", required=True)
    parser.add_argument("--allowlist-sha256", required=True)
    parser.add_argument("--sandbox-root", required=True)
    parser.add_argument("--extra-root", action="append", default=[])
    parser.add_argument("--pytest-arg", action="append", default=[])
    parser.add_argument("--label", default="guarded_run")
    ns = parser.parse_args(argv)
    try:
        extras = validate_pytest_extras(ns.pytest_arg)
        roots = compute_roots(ns.extra_root)
        manifest = load_allowlist(ns.allowlist, ns.allowlist_sha256, roots)
        sandbox = create_sandbox(ns.sandbox_root, ns.label, roots)
        ini = write_pytest_ini(sandbox)
        base = basetemp_path(sandbox)
        if os.path.exists(base):
            raise ValueError("pytest basetemp must never have existed")
        # Disable repository conftest discovery; dependencies must be explicit pinned imports.
        pytest_log = os.path.join(sandbox, "pytest.log")
        if os.path.lexists(pytest_log):
            raise ValueError("pytest log must be a new sandbox file")
        pytest_args = ["-c", ini, "--rootdir", sandbox, "--basetemp", base, "--noconftest", "--assert=plain", "--capture=sys", "--log-file", pytest_log] + extras + manifest["nodes"]
        spec = build_spec(sandbox, roots, manifest, pytest_args, ini)
        env = build_child_env(spec)
    except (ValueError, OSError, RuntimeError) as exc:
        print("wm-testguard launcher refused: %s" % exc)
        return 2
    payload = {"task": "WM-N1-RECOVERY-01", "status": "started", "sandbox": sandbox,
               "spec_sha256": _canonical_sha256(spec), "spec": spec,
               "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    _receipt(sandbox, "started", payload)
    log_path = os.path.join(sandbox, "guarded_run.log")
    try:
        with open(log_path, "x", encoding="utf-8") as log:
            ok, check = _preflight(sys.executable, env, sandbox)
            log.write(json.dumps({"preflight": check}) + "\n")
            log.flush()
            payload["preflight"] = check
            if not ok:
                code = PREFLIGHT_FAIL_EXIT
                payload["status"] = "preflight_failed"
            else:
                child = [sys.executable, "-c", _CHILD_BOOTSTRAP, json.dumps(pytest_args)]
                payload["child_argv"] = child
                code = subprocess.run(child, env=env, cwd=sandbox, stdout=log, stderr=subprocess.STDOUT).returncode
                payload["status"] = "completed" if code == 0 else "child_failed"
            payload["exit_code"] = code
    except BaseException as exc:
        payload["status"] = "interrupted_or_failed"
        payload["error"] = repr(exc)
        _receipt(sandbox, "interrupted", payload)
        raise
    payload["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    receipt = _receipt(sandbox, "finished", payload)
    print("wm-testguard: receipt=%s exit=%d" % (receipt, code))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
