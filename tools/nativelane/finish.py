"""Server-side continuation after the initial paired measurement driver; laptop-independent."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import drive as D

ROOT = D.ROOT


def main():
    os.nice(19)
    owner = json.loads((ROOT / "DRIVER.json").read_text())
    pid = owner["pid"]
    initial = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    while True:
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            alive = fields[0] != "Z" and fields[19] == initial[19]
        except FileNotFoundError:
            alive = False
        if not alive:
            break
        time.sleep(15)
    qual_owner = ROOT / "qual_retry.pid.json"
    if qual_owner.exists():
        rec = json.loads(qual_owner.read_text())
        while True:
            try:
                fields = Path(f"/proc/{rec['pid']}/stat").read_text().rsplit(")", 1)[1].split()
                alive = fields[0] != "Z" and fields[19] == rec["start_ticks"]
            except FileNotFoundError:
                alive = False
            if not alive:
                break
            time.sleep(15)
    # Only failed infrastructure tasks lack success markers; completed inference is reused.
    taskfile = ROOT / "QUAL_TASKS.json"
    if time.time() < D.DEADLINE and (not taskfile.exists() or not json.loads(taskfile.read_text())["complete"]):
        D.qual()
    D.run("decision_complete", [sys.executable, "tools/nativelane/analyze.py"], measurement=False)
    decision = json.loads((ROOT / "DECISION.json").read_text())
    # gpuR already has the documented cpuX key; its watcher reads this decision
    # and fetches the immutable package without a laptop or a reverse SSH key.
    if decision.get("adopt") and time.time() < D.DEADLINE:
        D.write(ROOT / "ADOPTION_READY.json", {"time": time.time(), "decision": "DECISION.json",
                "executor": "hash-pinned gpuR in-memory watcher; no laptop or cpuX->gpuR key required"})
    if time.time() < D.DEADLINE and not (ROOT / "STOP").exists():
        D.run("c1_ranking", [sys.executable, "tools/nativelane/c1.py"])
    D.write(ROOT / "FINISHED.json", {"time": time.time(), "adopt": decision.get("adopt", False)})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        D.write(ROOT / "FINISH_FAILED.json", {"error": repr(exc), "time": time.time()})
        raise
