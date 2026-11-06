"""Laptop-side off-box sync + milestone watcher for S-MAPPOWZ (N7).  No SUMO, read-only on the box.

    python tools/nm_mappo/sync_offbox.py --host cpuQ --remote /home/ubuntu/mappo_wz20261106 \
        --local E:/research-dense/runs/active/C4-V7FT-20261017/mappo_wz [--every-s 600]

Every ``--every-s`` (<= 15 min) it lists the box files under the evidence dirs only (run dirs,
ledger, sanity outputs, inputs; never the code trees, scenario dirs or fake-env check dirs),
pulls the new/changed ones in ONE ``tar`` stream over one ssh connection (atomic per-file
replace) and writes ``_sync/SYNC_STATE.json``.  It EXITS (so the operator acts) on:
10 = first re-quote point (production iteration >= 2), 11 = a new exported snapshot
(``policy_00006/24/42/84/168.pt``; ``_sync/NEW_SNAPSHOT.json`` carries path + sha256),
20 = anomaly (guard stop, failed episode, collision/teleport, lane command), 0 = production
exited, 30 = the box unreachable for 3 consecutive rounds.  Reported milestones are remembered
in ``_sync/MILESTONES.json`` so a restart does not fire them again.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path

SNAPSHOTS = (6, 24, 42, 84, 168)
INCLUDE = ("run", "run2", "ledger", "sanity_pre", "sanity_pre2", "sanity_post", "inputs")
EXCLUDE = ("/scenarios/", "/check_", "/c4_tmp/", "__pycache__")
FIND = "find {dirs} -type f -printf '%p\\t%s\\t%T@\\n' 2>/dev/null; true"


def ssh(host: str, command: str, timeout: int = 300) -> bytes:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, command],
                          check=True, capture_output=True, timeout=timeout).stdout


def listing(host: str, remote: str) -> dict:
    out = ssh(host, f"cd {remote} && " + FIND.format(dirs=" ".join(INCLUDE))).decode()
    files = {}
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 3 or any(x in "/" + parts[0] for x in EXCLUDE):
            continue
        files[parts[0]] = [int(parts[1]), float(parts[2])]
    return files


def fetch_many(host: str, remote: str, local: Path, rels: list) -> list:
    if not rels:
        return []
    names = " ".join(shlex.quote(r) for r in rels)
    data = ssh(host, f"cd {remote} && tar cf - --ignore-failed-read {names} 2>/dev/null; true",
               timeout=1200)
    done = []
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        for m in tar.getmembers():
            if not m.isfile() or m.name.startswith("/") or ".." in Path(m.name).parts:
                continue
            dest = local / m.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            tmp.write_bytes(tar.extractfile(m).read())
            os.replace(tmp, dest)
            done.append(m.name)
    return done


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(local: Path, milestones: dict, *, run: str = "run2", lane_on: bool = False) -> int | None:
    prod = local / run / "production"
    status = json.loads((prod / "status.json").read_text()) if (prod / "status.json").exists() else {}
    rows = []
    if (prod / "log.jsonl").exists():
        rows = [json.loads(x) for x in (prod / "log.jsonl").read_text().splitlines() if x.strip()]
    iters = [r for r in rows if r.get("type") == "iter"]
    bad = [r for r in iters if (r.get("collisions") or 0) or (r.get("teleports") or 0)
           or (not lane_on and (r.get("lane_commands_issued") or 0)) or (r.get("failed_episodes") or 0)]
    if any(prod.glob("GUARD_STOP_*.json")) or bad or [r for r in rows if r.get("type") == "failure"]:
        if not milestones.get("anomaly_reported"):
            milestones["anomaly_reported"] = True
            return 20
    for it in SNAPSHOTS:
        path = prod / f"policy_{it:05d}.pt"
        key = f"snapshot_{it}"
        if path.exists() and not milestones.get(key):
            milestones[key] = {"path": str(path), "sha256": sha256(path)}
            (local / "_sync" / "NEW_SNAPSHOT.json").write_text(json.dumps({"iteration": it, **milestones[key]}))
            return 11
    if int(status.get("iteration", 0)) >= 2 and not milestones.get("requote"):
        milestones["requote"] = True
        return 10
    seen = milestones.setdefault("exits_reported", [])
    for path in sorted(prod.glob("EXIT_*.json")):       # every supervisor exit (stop or finish) once
        if path.name not in seen:
            seen.append(path.name)
            return 0
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="cpuQ")
    ap.add_argument("--remote", default="/home/ubuntu/mappo_wz20261106")
    ap.add_argument("--local", required=True)
    ap.add_argument("--every-s", type=int, default=600)
    ap.add_argument("--run", default="run2", help="run dir name under --remote (production inside)")
    ap.add_argument("--lane-on", action="store_true", help="lane-capable MAPPO: lane commands expected")
    ap.add_argument("--watch-only", action="store_true",
                    help="never exit on milestones (operator absent); exit only once production has exited")
    a = ap.parse_args(argv)
    local = Path(a.local)
    (local / "_sync").mkdir(parents=True, exist_ok=True)
    state_path, ms_path = local / "_sync" / "SYNC_STATE.json", local / "_sync" / "MILESTONES.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"files": {}}
    milestones = json.loads(ms_path.read_text()) if ms_path.exists() else {}
    misses = 0
    while True:
        try:
            files = listing(a.host, a.remote)
            changed = [rel for rel, meta in sorted(files.items())
                       if state["files"].get(rel) != meta or not (local / rel).exists()]
            got = set(fetch_many(a.host, a.remote, local, changed))
            for rel in got:
                state["files"][rel] = files[rel]
            misses = 0
            state["last_sync_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            state["fetched_last_round"] = len(got)
            state_path.write_text(json.dumps(state, indent=1))
            code = check(local, milestones, run=a.run, lane_on=a.lane_on)
            ms_path.write_text(json.dumps(milestones, indent=1))
            print(f"{state['last_sync_utc']} synced {len(got)}/{len(changed)} files; milestone {code}",
                  flush=True)
            if code is not None and not a.watch_only:
                return code
            if a.watch_only and any((local / a.run / "production").glob("EXIT_production_*.json")):
                return 0
        except (OSError, subprocess.SubprocessError, ValueError, tarfile.TarError) as exc:
            misses += 1
            print(f"sync failed ({misses}): {exc}", flush=True)
            if misses >= 3:
                return 30
        time.sleep(a.every_s)


if __name__ == "__main__":
    sys.exit(main())
