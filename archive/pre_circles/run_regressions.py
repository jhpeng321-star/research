"""Run unchanged historical tests in a small disposable source/configuration copy.

No training entry point is invoked. Large raw data stay in their one archived
copy. Relative writable fixtures, test caches, locks and output live in the
disposable workspace. No workspace is deleted automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
import uuid

ARCHIVE = Path(__file__).resolve().parent
ROOT = ARCHIVE.parents[1]
OUTPUT = ROOT / "output/adoption_cleanup/archived_regression_workspaces"
SKIP = {"__pycache__", ".pytest_cache", ".ruff_cache", ".git"}


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def ignore_links(directory, names):
    ignored = []
    for name in names:
        child = Path(directory) / name
        attributes = getattr(child.lstat(), "st_file_attributes", 0)
        if name in SKIP or attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT or child.is_symlink():
            ignored.append(name)
    return ignored


def junction(path, target):
    path = path.resolve()
    target = target.resolve()
    if not path.is_relative_to(OUTPUT.resolve()) or not target.is_relative_to(ARCHIVE):
        raise ValueError("Unexpected compatibility alias boundary")
    if path.exists() or path.is_relative_to(target) or target.is_relative_to(path):
        raise ValueError("Occupied or recursive alias")
    path.parent.mkdir(parents=True, exist_ok=True)
    quote = lambda value: "'" + str(value).replace("'", "''") + "'"
    command = f"New-Item -ItemType Junction -Path {quote(path)} -Target {quote(target)} -ErrorAction Stop | Out-Null"
    subprocess.run(["powershell", "-NoProfile", "-Command", command], check=True, capture_output=True)
    if path.resolve() != target:
        raise RuntimeError("Archive dependency alias did not resolve as declared")


def immutable_state(manifest):
    """Full hashes for small source/config/report inputs, metadata for raw runs."""
    values = {}
    for row in manifest["files"]:
        path = ROOT / row["current_path"]
        status = path.stat()
        if row["original_path"].startswith("runs/"):
            values[row["current_path"]] = {"bytes":status.st_size,"mtime_ns":status.st_mtime_ns}
        else:
            values[row["current_path"]] = {"sha256":sha(path)}
            if values[row["current_path"]]["sha256"] != row["sha256"]:
                raise RuntimeError("Archive input already differs from inventory: " + row["current_path"])
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    pytest_args = args.pytest_args
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]
    if not pytest_args:
        pytest_args = ["-q"]
    manifest = json.loads((ARCHIVE / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "relocated_with_explicit_frozen_reader_aliases":
        raise RuntimeError("Archive migration is not terminal")
    stamp = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    workspace = OUTPUT / stamp
    workspace.mkdir(parents=True, exist_ok=False)
    code = workspace / "code"
    shutil.copytree(ARCHIVE / "code", code, ignore=ignore_links)
    shutil.copytree(ARCHIVE / "configs/scenarios", code / "scenarios", ignore=ignore_links)
    shutil.copytree(ARCHIVE / "reports/output", code / "output", ignore=ignore_links)
    junction(code / "runs", ARCHIVE / "results/runs")
    junction(code / "docs", ARCHIVE / "docs")
    # Historical test_policy_runner computes this sibling from __file__ and
    # writes its logs there. Five exact local input copies avoid shared writes.
    fixture = json.loads((ARCHIVE / "regression_fixture_manifest.json").read_text(encoding="utf-8"))
    shared_copy = workspace / "research/highway_merge_shepherd"
    shared_copy.mkdir(parents=True)
    for row in fixture["files"]:
        target = shared_copy / Path(row["local_path"]).name
        shutil.copyfile(ROOT / row["local_path"], target)
        if sha(target) != row["sha256"]:
            raise RuntimeError("Regression fixture identity mismatch")
    registration = {"schema":"isolated_archive_regression_v1","workspace":str(workspace),"source_copy":str(code),"pytest_args":pytest_args,"tests_unchanged":True,"raw_data_duplicated":False,"scenario_inputs_copied":True,"no_automatic_deletion":True,"simulation_scope":"Existing historical regression tests only; SUMO integration tests may run. No training, acceptance, wall generation, or scientific panel is authorized."}
    save(workspace / "registration.json", registration)
    if args.prepare_only:
        print(json.dumps(registration), flush=True)
        return
    before = immutable_state(manifest)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(code)
    env["ZCODE_SUMO_LOCK"] = str(workspace / "sumo_launch.lock")
    env["SUMO_HOME"] = env.get("SUMO_HOME", "C:/Program Files (x86)/Eclipse/Sumo")
    env["PATH"] = str(Path(env["SUMO_HOME"]) / "bin") + os.pathsep + env["PATH"]
    command = [sys.executable, "-m", "pytest", "-o", f"cache_dir={workspace / 'pytest_cache'}", *pytest_args]
    with (workspace / "pytest.log").open("wb") as stream:
        completed = subprocess.run(command, cwd=code, env=env, stdout=stream, stderr=subprocess.STDOUT)
    after = immutable_state(manifest)
    changed = [path for path in before if before[path] != after[path]]
    result = {"schema":"isolated_archive_regression_result_v1","workspace":str(workspace),"exit_code":completed.returncode,"archive_changed_paths":changed,"archive_source_config_report_hashes_unchanged":not changed,"raw_guard":"Size and mtime for every original raw run file; complete post-cleanup raw SHA256 is separately recorded in archive_verification.json.","all_passed":completed.returncode==0 and not changed}
    save(workspace / "result.json", result)
    save(ROOT / "output/adoption_cleanup/isolated_archived_regression_result.json", result)
    print(json.dumps(result), flush=True)
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
