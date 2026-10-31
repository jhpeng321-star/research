"""Read-only owned-run collector and durable paper-report updater; never dispatches.

Launch once after the training dispatch. Every remote read is at least five
minutes apart. Final169 is handed to the evaluation owner, never evaluated here.
Peer evaluation can later be supplied through an explicit path+SHA256 manifest.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


REPO = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(REPO))
from tools.selmappo2 import ops, report  # noqa: E402


BUILD = "build_c141d5715f26af4e9b50d0866cdf971ef92d55dd1223c4312c4f25bdec5a8b29"
PID = 606968
INTERVAL = 300
DEADLINE_HOURS = 48
MARKER = "<!-- SELMAPPO2_LIVE_BEGIN -->"
PAPER = REPO / "docs/paper_results/SELMAPPO2.md"
LOCK = ops.EVIDENCE / f"WATCH_ONCE_{BUILD}.json"
MANIFEST = ops.EVIDENCE / "PEER_EVALUATION_MANIFEST.json"
PEER = REPO.parent / "c4-c1switch2"


def append_event(path, event):
    ops.checked(path.parent, directory=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(ops.canonical(dict(timestamp_ns=time.time_ns(), **event)) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def publish(collection, parsed, peer=None):
    raw = ops.checked(collection).read_bytes()
    token = f"{time.time_ns()}_{uuid.uuid4().hex}"
    snapshot = ops.EVIDENCE / f"LIVE_{token}.md"
    provenance = f"`selmappo2/{collection.name}`; SHA256 `{ops.sha(raw)}`."
    report.publish(snapshot, parsed, provenance, peer)
    # Preserve the complete previous paper report before replacing only our tail.
    previous = ops.checked(PAPER).read_bytes()
    report.exclusive_text(ops.EVIDENCE / f"PAPER_BEFORE_{token}.md", previous.decode("utf-8"))
    prefix = previous.decode("utf-8").split(MARKER, 1)[0].rstrip()
    live = snapshot.read_text(encoding="utf-8")
    # Assets were published beside the immutable snapshot, one directory deeper.
    live = live.replace("](selmappo2_curve_", "](selmappo2/selmappo2_curve_")
    live = live.replace("# SELMAPPO2 results", "## Latest verified live results", 1)
    if ops.checked(PAPER).read_bytes() != previous:
        raise ValueError("paper report changed concurrently; retained snapshot, refusing overwrite")
    with PAPER.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(prefix + "\n\n" + MARKER + "\n\n" + live)
        stream.flush()
        os.fsync(stream.fileno())
    return snapshot


def final_handoff(collection, parsed):
    """Only verified final169; source files remain in the original immutable build."""
    if parsed["state"] != "COMPLETE" or not parsed["checkpoint_sha256"]:
        raise ValueError("final handoff requires verified complete final169")
    payload = json.loads(ops.checked(collection).read_bytes())
    files = report.unpack(payload)
    integration = json.loads(files["INTEGRATION.json"])
    result = dict(schema="selmappo2.final_handoff/1", status="COMPLETE", host="cpuQ",
                  root=payload["root"], pid=PID, build=BUILD, variant="per_cav", iteration=169,
                  checkpoint=payload["root"] + "/training/checkpoint_169.pt",
                  checkpoint_sha256=parsed["checkpoint_sha256"],
                  frozen=json.loads(files["training/FROZEN_FINAL.json"]),
                  training_owner=json.loads(files["training/OWNER.json"]),
                  integration=integration, sources=integration["files"],
                  source_root=payload["root"] + "/pytools",
                  local_bundle=str(ops.EVIDENCE / f"BUNDLE_{payload['binding']}.json"),
                  collection=str(collection), collection_sha256=ops.sha(collection.read_bytes()),
                  reserved_training_sim_s=parsed["reserved_sim_s"],
                  actual_training_sim_s=parsed["actual_sim_s"],
                  local_evaluation_sim_s=0, evaluation_owner="S-C1SWITCH2",
                  panel_sha256="7c780c25237dde73a657937ab577b8e48f71766817f4363f3298982e90144827")
    for path in (ops.EVIDENCE / f"FINAL_HANDOFF_{BUILD}.json", PEER / "SELMAPPO2_FINAL.json"):
        if path.exists():
            if ops.read_json(path) != result:
                raise ValueError("different existing final handoff; refusing overwrite")
        else:
            ops.exclusive_json(path, result)


def peer_evaluation(parsed):
    if not MANIFEST.exists():
        return None
    data = ops.read_json(MANIFEST)
    if data.get("schema") != "selmappo2.peer_manifest/1":
        raise ValueError("invalid explicit peer evaluation manifest")
    result = report.evaluation(Path(data["path"]), data["sha256"], parsed)
    if result["contract_sha256"] != "7c780c25237dde73a657937ab577b8e48f71766817f4363f3298982e90144827":
        raise ValueError("peer evaluation contract is not the frozen PANEL_v2")
    return result


def watch():
    owner = ops.read_json(LOCK)
    if owner["command"] != command() or owner["script_sha256"] != ops.sha(Path(__file__).read_bytes()):
        raise ValueError("watcher source/command differs from exclusive ownership receipt")
    ops.exclusive_json(ops.EVIDENCE / f"WATCH_PROCESS_{BUILD}.json",
                       dict(pid=os.getpid(), command=command(), script_sha256=owner["script_sha256"]))
    events = ops.EVIDENCE / f"WATCH_EVENTS_{BUILD}.jsonl"
    deadline = time.monotonic() + DEADLINE_HOURS * 3600
    collection, parsed, finished, dead_reads = None, None, False, 0
    # No overlap with the manual dispatch verification reads.
    time.sleep(INTERVAL)
    while time.monotonic() < deadline:
        if not finished:
            try:
                collection = report.collect(BUILD, PID, ops.EVIDENCE)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                append_event(events, dict(event="READ_TRANSPORT_ERROR", error=str(exc)))
                time.sleep(INTERVAL)
                continue  # A read-only transport retry never dispatches simulation.
            parsed = report.analyze(ops.read_json(collection), BUILD, PID)
        peer = peer_evaluation(parsed) if parsed["state"] == "COMPLETE" else None
        snapshot = publish(collection, parsed, peer)
        append_event(events, dict(event="REPORT", collection=str(collection), snapshot=str(snapshot),
                                 state=parsed["state"], updates=len(parsed["points"]),
                                 reserved_sim_s=parsed["reserved_sim_s"],
                                 actual_sim_s=parsed["actual_sim_s"],
                                 alive=parsed["status"].get("alive"), evaluation_complete=peer is not None))
        if parsed["state"] == "COMPLETE" and not finished:
            final_handoff(collection, parsed)
            finished = True
        if peer is not None or parsed["state"] == "FAILED":
            return
        dead_reads = dead_reads + 1 if not finished and not parsed["status"].get("alive") else 0
        if dead_reads >= 2:
            append_event(events, dict(event="STOPPED_INCOMPLETE", updates=len(parsed["points"])))
            return
        time.sleep(INTERVAL)
    append_event(events, dict(event="WATCH_DEADLINE", hours=DEADLINE_HOURS))


def command():
    return [sys.executable, "-B", str(Path(__file__).absolute()), "--watch"]


def launch():
    ops.checked(ops.EVIDENCE, directory=True)
    ops.checked(PAPER)
    ops.checked(PEER, directory=True)
    identity = dict(command=command(), script_sha256=ops.sha(Path(__file__).read_bytes()),
                    build=BUILD, training_pid=PID, interval_s=INTERVAL,
                    timestamp_ns=time.time_ns(), read_only_remote=True, new_sumo_seconds=0)
    ops.exclusive_json(LOCK, identity)
    log = ops.EVIDENCE / f"WATCH_{BUILD}.log"
    with log.open("x", encoding="utf-8") as stream:
        kwargs = (dict(creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
                  if os.name == "nt" else dict(start_new_session=True))
        child = subprocess.Popen(command(), cwd=REPO, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, **kwargs)
    receipt = dict(identity, pid=child.pid, log=str(log))
    ops.exclusive_json(ops.EVIDENCE / f"WATCH_DISPATCH_{BUILD}.json", receipt)
    print(ops.canonical(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--launch", action="store_true")
    group.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    launch() if args.launch else watch()
