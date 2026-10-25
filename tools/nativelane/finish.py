"""Server-side continuation after the initial paired measurement driver; laptop-independent."""
from __future__ import annotations

import json
import os
import subprocess
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
    # A conditional, hash-pinned adoption package may be installed before the laptop cutoff.
    package = ROOT / "ADOPTION_PACKAGE.json"
    if decision.get("adopt") and time.time() < D.DEADLINE:
        if package.exists():
            with (ROOT / "logs/adopt.log").open("ab") as log:
                proc = subprocess.run([sys.executable, "tools/nativelane/adopt.py", "--execute"],
                                      cwd=D.REPO, env=D.ENV, stdin=subprocess.DEVNULL,
                                      stdout=log, stderr=subprocess.STDOUT, check=False)
            D.write(ROOT / "ADOPTION_EXIT.json", {"rc": proc.returncode, "time": time.time()})
        else:
            D.write(ROOT / "ADOPTION_WAIT_OWNER.json", {"reason": "no registered adoption package installed", "time": time.time()})
    if time.time() < D.DEADLINE and not (ROOT / "STOP").exists():
        D.run("c1_ranking", [sys.executable, "tools/nativelane/c1.py"])
    D.write(ROOT / "FINISHED.json", {"time": time.time(), "adopt": decision.get("adopt", False)})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        D.write(ROOT / "FINISH_FAILED.json", {"error": repr(exc), "time": time.time()})
        raise
