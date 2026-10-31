"""Supervise existing receipts through noon; never dispatch or signal training."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(REPO))
from scripts import selmappo2_watch as watcher  # noqa: E402
from tools.selmappo2 import ops, report  # noqa: E402

DEADLINE = datetime.fromisoformat("2026-11-01T12:00:00+08:00").timestamp()
INTERVAL = 1200
LOCK = ops.EVIDENCE / "SUPERVISE_NOON_ONCE.json"
EVENTS = ops.EVIDENCE / f"WATCH_EVENTS_{watcher.BUILD}.jsonl"


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, check=True,
                          capture_output=True, text=True, timeout=60).stdout


def identity():
    expected = ops.read_json(ops.EVIDENCE / f"WATCH_PROCESS_{watcher.BUILD}.json")
    pid = expected["pid"]
    if type(pid) is not int or pid != 12432:
        raise ValueError("unexpected watcher PID")
    result = subprocess.run([
        "powershell", "-NoProfile", "-Command",
        f'Get-CimInstance Win32_Process -Filter "ProcessId={pid}" | '
        "Select-Object ProcessId,ExecutablePath,CommandLine,CreationDate | ConvertTo-Json -Compress"
    ], check=True, capture_output=True, text=True, timeout=30)
    observed = json.loads(result.stdout) if result.stdout.strip() else None
    if observed:
        command = " ".join(expected["command"])
        if (observed["ProcessId"] != pid
                or observed["ExecutablePath"].casefold() != expected["command"][0].casefold()
                or observed["CommandLine"].strip() != command
                or observed["CreationDate"] != "/Date(1793469838604)/"
                or ops.sha(ops.checked(Path(watcher.__file__)).read_bytes()) != expected["script_sha256"]):
            raise ValueError("watcher process/source identity mismatch")
    return observed


def summarize(event, parsed, now, observed, previous=None):
    status = parsed["status"]
    age = now - status["timestamp_ns"] / 1e9
    records = parsed["records"]
    failures = sum(row["terminal"] == "FAILED" for row in records.values())
    production = sum(seed not in report.SMOKES and row["terminal"] == "RESULT"
                     for seed, row in records.items())
    alerts = []
    if age < -60 or age > 900:
        alerts.append("STALE_OR_FUTURE_REMOTE_OBSERVATION")
    if not observed and parsed["state"] != "COMPLETE":
        alerts.append("WATCHER_ABSENT")
    if not status.get("alive") and parsed["state"] != "COMPLETE":
        alerts.append("TRAINING_ABSENT")
    if parsed["state"] == "FAILED" or failures != 1:
        alerts.append("NEW_TRAINING_FAILURE")
    log = status.get("summaries", {}).get("training.log", {}).get("tail", "")
    if any(marker in log for marker in ("Traceback (most recent call last)", "BrokenProcessPool", "RuntimeError:")):
        alerts.append("TRAINING_LOG_ERROR")
    result = dict(schema="selmappo2.supervision/1", timestamp_ns=int(now * 1e9),
                  local_time=datetime.fromtimestamp(now, timezone(timedelta(hours=8))).isoformat(),
                  observation_timestamp_ns=status["timestamp_ns"], observation_age_s=age,
                  collection=event["collection"], state=parsed["state"],
                  training_pid=watcher.PID, training_alive=status.get("alive"),
                  watcher=observed, updates=len(parsed["points"]), production=production,
                  open=sum(row["terminal"] is None for row in records.values()),
                  failures=failures, actual_sim_s=parsed["actual_sim_s"],
                  reserved_sim_s=parsed["reserved_sim_s"], safety=parsed["safety"],
                  checkpoint_sha256=(parsed["points"][-1]["checkpoint_sha256"]
                                     if parsed["points"] else None),
                  issues=parsed["issues"], alerts=alerts, new_sumo_seconds=0)
    if previous:
        result["delta_updates"] = result["updates"] - previous["updates"]
        result["delta_production"] = production - previous["production"]
        if result["delta_updates"] < 0 or result["delta_production"] < 0:
            alerts.append("REGRESSED_PROGRESS")
        elif not result["delta_updates"] and not result["delta_production"] and parsed["state"] != "COMPLETE":
            alerts.append("RESOURCE_WAIT_OR_STALL")
    return result


def heartbeat(previous=None):
    # Consume the existing collector only: no extra SSH, simulations or watchers.
    events = [json.loads(line) for line in ops.checked(EVENTS).read_text().splitlines()]
    event = next(row for row in reversed(events) if row.get("event") == "REPORT")
    collection = ops.checked(Path(event["collection"]))
    if collection.parent != ops.EVIDENCE or not collection.name.startswith(f"COLLECTION_{watcher.BUILD[6:]}_"):
        raise ValueError("collection outside the exact owned build")
    parsed = report.analyze(ops.read_json(collection), watcher.BUILD, watcher.PID)
    result = summarize(event, parsed, time.time(), identity(), previous)
    result["collection_sha256"] = ops.sha(collection.read_bytes())
    path = ops.EVIDENCE / f"SUPERVISION_{time.time_ns()}.json"
    ops.exclusive_json(path, result)
    stamp = result["local_time"][:16].replace("T", " ")
    alerts = ",".join(result["alerts"]) or "none"
    line = (f"- {stamp} +08:00 automated supervision: cpuQ{watcher.PID} "
            f"alive={result['training_alive']}; watcher12432 identity/live="
            f"{bool(result['watcher'])}; {result['updates']}/169 updates, "
            f"{result['production']}/7774 production, {result['open']} open; "
            f"{result['actual_sim_s']:g} actual/{result['reserved_sim_s']} reserved sim-s; "
            f"failures={result['failures']} (original smoke retained), "
            f"collisions={result['safety']['collisions']:g}/teleports="
            f"{result['safety']['teleports']:g}; alerts={alerts}. "
            f"Evidence `docs/paper_results/selmappo2/{path.name}`.\n")
    with ops.checked(REPO / "HANDOFF.md").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write("\n" + line)
        stream.flush()
        os.fsync(stream.fileno())
    if git("branch", "--show-current").strip() != "c4/selmappo2":
        raise ValueError("unexpected branch; receipt retained, refusing commit")
    if git("diff", "--cached", "--name-only").strip():
        raise ValueError("existing staged work; receipt retained, refusing commit")
    # Narrow ownership allowlist excludes peer notices and unrelated files.
    prefixes = ("COLLECTION_", "LIVE_", "PAPER_BEFORE_", "selmappo2_curve_", "WATCH_EVENTS_", "SUPERVISION_")
    changed = git("ls-files", "--modified", "--others", "--exclude-standard", "-z", "--",
                  "docs/paper_results/selmappo2").split("\0")
    evidence = [name for name in changed if name and Path(name).name.startswith(prefixes)]
    git("status", "--short")
    git("diff", "--stat")
    git("log", "--oneline", "-10")
    git("add", "--", "HANDOFF.md", "docs/paper_results/SELMAPPO2.md", *evidence)
    git("diff", "--cached", "--stat")
    commit = git("commit", "-m", f"selmappo2: supervise {stamp[11:]} progress at {result['updates']} updates")
    print(json.dumps(dict(receipt=str(path), alerts=result["alerts"], updates=result["updates"],
                          production=result["production"], commit=commit.strip())), flush=True)
    return result


def run():
    owner = ops.read_json(LOCK)
    if owner["command"] != command() or owner["script_sha256"] != ops.sha(Path(__file__).read_bytes()):
        raise ValueError("supervision ownership/source mismatch")
    ops.exclusive_json(ops.EVIDENCE / "SUPERVISE_NOON_PROCESS.json", dict(pid=os.getpid(), **owner))
    previous = None
    while True:
        try:
            previous = heartbeat(previous)
        except Exception as exc:
            ops.exclusive_json(ops.EVIDENCE / f"SUPERVISION_ERROR_{time.time_ns()}.json",
                               dict(timestamp_ns=time.time_ns(), error=repr(exc), new_sumo_seconds=0))
            print(f"SUPERVISION_ERROR {exc!r}", flush=True)
        if time.time() >= DEADLINE:
            break
        time.sleep(min(INTERVAL, max(0, DEADLINE - time.time())))
    print("NOON_SUPERVISION_COMPLETE", flush=True)


def command():
    return [sys.executable, "-B", str(Path(__file__).absolute()), "--watch"]


def launch():
    ops.checked(ops.EVIDENCE, directory=True)
    owner = dict(command=command(), script_sha256=ops.sha(Path(__file__).read_bytes()),
                 deadline=datetime.fromtimestamp(DEADLINE, timezone.utc).isoformat(),
                 interval_s=INTERVAL, timestamp_ns=time.time_ns(), new_sumo_seconds=0)
    if time.time() >= DEADLINE:
        raise ValueError("supervision deadline already passed")
    ops.exclusive_json(LOCK, owner)
    with (ops.EVIDENCE / "SUPERVISE_NOON.log").open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(command(), cwd=REPO, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT,
                                 creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
    receipt = dict(pid=child.pid, **owner)
    ops.exclusive_json(ops.EVIDENCE / "SUPERVISE_NOON_DISPATCH.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--launch", action="store_true")
    group.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    launch() if args.launch else run()
