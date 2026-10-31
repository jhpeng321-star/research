"""Read-only receipt/liveness summary for the existing full-corpus CPU ES run.

No dispatch, rollout, checkpoint load, signal, or automatic retry. Remote reads
are confined to the two owned run directories and receipt-identified ES PIDs.
"""
import argparse
import base64
import json
from pathlib import Path
import subprocess


ROOTS = {
    "gpuA": "/root/autodl-tmp/c4/wmfixmerge20261031/es_cpu_v3/es_run_full",
    "cpuQ": "/home/ubuntu/wmfix_es/es_run_full",
}
PYTHONS = {"gpuA": "/root/miniconda3/bin/python",
           "cpuQ": "/home/ubuntu/miniconda3/bin/python"}

REMOTE = r'''
import collections, json, pathlib, time
root = pathlib.Path(ROOT)
def read(name):
    p = root / name
    return json.loads(p.read_text()) if p.is_file() else None
run = read("RUN.json")
summary = dict(root=str(root), observed_unix=time.time(), generations=[], cells={})
if run:
    summary["driver_pid"] = run["pid"]
    summary["elapsed_s"] = round(time.time() - run["started"], 1)
for p in sorted(root.glob("GEN_[0-9][0-9][0-9].json")):
    g = read(p.name)
    summary["generations"].append(dict(gen=g["gen"], seconds=g["seconds"],
        stale=g["stale_generations"], improved=g["improved"],
        candidates={c: {k: v[k] for k in ("fitness", "paired_J_MAE", "source_contrast_MAE", "cells")}
                    for c, v in g["candidates"].items()},
        best=dict(gen=g["best_overall"]["gen"], cand=g["best_overall"]["cand"])))
counts = collections.Counter()
for p in root.glob("CELL_g*_c*_*.json"):
    parts = p.name.split("_")
    counts["_".join(parts[1:4])] += 1
summary["cells"] = dict(sorted(counts.items()))
summary["failed_sweeps"] = {p.name: read(p.name).get("blocker")
    for p in sorted(root.glob("SWEEP_FAILED_*.json"))}
summary["completed_sweeps"] = len(list(root.glob("SWEEP_DONE_*")))
for name in ("ES_ABORT", "ES_FINAL", "FINAL", "SCORING_READY", "VERDICT_PREFLIGHT_FINAL"):
    doc = read(name + ".json")
    summary[name] = None if doc is None else {k: doc[k] for k in
        ("status", "blocker", "blockers", "elapsed_s", "stopped_after_gen", "checkpoint",
         "sha256", "canonical_scoring_reload") if k in doc}
pids = {int(read(p.name)["pid"]) for p in root.glob("ENV_*.json")
        if read(p.name).get("host") == __import__("socket").gethostname()}
if run:
    pids.add(int(run["pid"]))
active = []
for pid in sorted(pids):
    p = pathlib.Path("/proc") / str(pid)
    try:
        cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        if "tools.wmfixmerge.es " in cmd and ("--out " + str(root) + " ") in cmd + " ":
            fields = (p / "stat").read_text().split()
            if fields[2] != "Z":
                active.append(pid)
    except (FileNotFoundError, ProcessLookupError):
        pass
summary["active_receipted_es_pids"] = active
print(json.dumps(summary, indent=2))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=tuple(ROOTS), required=True)
    parser.add_argument("--out", help="optional new immutable local JSON receipt")
    args = parser.parse_args()
    script = "ROOT = " + repr(ROOTS[args.host]) + "\n" + REMOTE
    encoded = base64.b64encode(script.encode()).decode()
    # Base64 keeps the fixed read-only program intact through Windows/SSH shells.
    remote = PYTHONS[args.host] + " -c \"import base64;exec(base64.b64decode('" + encoded + "'))\""
    result = subprocess.run(["ssh", "-o", "ConnectTimeout=8", args.host, remote],
                            text=True, capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError("read-only SSH status failed: " + result.stderr)
    doc = json.loads(result.stdout)
    if args.out:
        with Path(args.out).open("x", encoding="utf-8") as stream:
            json.dump(doc, stream, indent=2, allow_nan=False)
    print(json.dumps(doc, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
