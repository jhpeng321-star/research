"""Run-2 CPU dry-run harness (one process at a time, threads 4, batch 4).

Shells out to ``wms.cli train`` with the frozen TRAIN packs and reports, per
logged update: every loss term, valid targets, entrant counts (if the merged
trainer logs them), step seconds, and the child process peak RSS.  Data is
opened READ-ONLY; no CAL/HOLDOUT/sealed path is referenced.

Usage::

    python _out/run2_dry.py --tag mergecal_10 --unroll-s 10 --steps 2 \
        [--merge-cal] [--entries-v2] [--traffic-flags "--traffic-loss"] [...]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

PY = r"C:\Users\jpeng\miniconda3\envs\rl_driving\python.exe"
ROOT = Path(__file__).resolve().parents[1]
CWD = ROOT / "output" / "wd01" / "wm_n1"
BASE = "D:/rd-staging/wmn1_feat/pack_v3"
ID_BASE = "E:/tmp/wmn1_train_runs_20261002/sidecars3_20261003T0005CN/identities"
D9 = "E:/rd-wt/dspeed-data/_out"
D10 = "E:/rd-wt/damping-snap/_out"
ID_DS = "E:/rd-wt/a9/_out/identities_dspeed"
ID_DM = "E:/rd-wt/damping-snap/_out/identities_damping"
LPO = "E:/rd-wt/ws-p5/_out/lpo"
DEFAULT_OUTCOMES = "E:/rd-wt/ws-p2/_ext/labels_cov"   # TRAIN-only schema/3 sidecar


def base_cmd(out: Path, args) -> list[str]:
    cmd = [
        PY, "-m", "wms.cli", "train",
        "--pack", BASE, "--identities", ID_BASE,
        "--pack", f"{D9}/pack_dspeed", "--identities", ID_DS,
        "--overlay", f"{D9}/labels_overlay_dspeed",
        "--pack", f"{D10}/pack_damping", "--identities", ID_DM,
        "--overlay", f"{D10}/labels_overlay_damping",
        "--window-list", f"{LPO}/windows_train_noPI.json",
        "--exclude-window-list", f"{LPO}/pi_excluded_runs.json",
        "--out", str(out), "--device", "cpu", "--threads", "4", "--num-workers", "0",
        "--batch-size", "4", "--sumo-fidelity", "--start-shift-prob", "0.75",
        "--truncation-steps", "20",
        "--unroll-s", str(args.unroll_s), "--steps", str(args.steps),
    ]
    if args.merge_cal:
        cmd += ["--merge-cal", "--merge-cal-params",
                str(ROOT / "_out" / "merge_params.json")]
    if args.entries_v2:
        cmd += ["--entries-v2", "--entry-pad", str(args.entry_pad)]
    if args.traffic_loss:
        cmd += ["--traffic-loss"]
    if args.extra:
        cmd += args.extra
    return cmd


def monitor(proc: psutil.Process, peak: dict, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            total = proc.memory_info().rss
            for child in proc.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except psutil.Error:
                    pass
            peak["rss"] = max(peak.get("rss", 0), total)
        except psutil.Error:
            pass
        stop.wait(0.2)


def summarize(log: Path, out_json: Path, peak_mb: float, elapsed: float, rc: int) -> dict:
    rows = []
    if log.is_file():
        rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()
                if x.strip()]
    numeric = {}
    entrant = {}
    for row in rows:
        for k, v in row.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                numeric.setdefault(k, []).append(v)
                if "entrant" in k.lower():
                    entrant.setdefault(k, []).append(v)
    summary = {
        "rc": rc, "updates": len(rows),
        "terms": {k: {"first": v[0], "last": v[-1],
                      "finite": all(math.isfinite(float(x)) for x in v)}
                  for k, v in numeric.items()},
        "valid_targets_last": rows[-1].get("valid_targets") if rows else None,
        "entrant_terms": {k: {"sum": sum(v), "last": v[-1]} for k, v in entrant.items()},
        "step_seconds": [r.get("seconds") for r in rows],
        "mean_step_seconds": (sum(r.get("seconds", 0) for r in rows) / len(rows)
                              if rows else None),
        "peak_rss_mb": round(peak_mb, 1), "over_6gb": peak_mb > 6144.0,
        "wall_seconds": round(elapsed, 2),
    }
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--unroll-s", type=float, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--merge-cal", action="store_true")
    ap.add_argument("--entries-v2", action="store_true")
    ap.add_argument("--entry-pad", type=int, default=8)
    ap.add_argument("--traffic-loss", action="store_true")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=None)
    args = ap.parse_args()

    out = ROOT / "_out" / f"run2_dry_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    cmd = base_cmd(out, args)
    print("CMD " + " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("WM_N1_TRAIN_IDENTITIES", ID_BASE)
    env.setdefault("WM_N1_TRAIN_OUTCOMES", DEFAULT_OUTCOMES)
    peak: dict = {}
    started = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=str(CWD), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    p = psutil.Process(proc.pid)
    stop = threading.Event()
    t = threading.Thread(target=monitor, args=(p, peak, stop), daemon=True)
    t.start()
    tail = []
    for line in proc.stdout:
        tail.append(line.rstrip())
        if len(tail) > 25:
            tail.pop(0)
    rc = proc.wait()
    elapsed = time.perf_counter() - started
    stop.set()
    peak_mb = peak.get("rss", 0) / 2 ** 20
    log = out / "train_log.jsonl"
    summary = summarize(log, ROOT / "_out" / f"run2_dry_{args.tag}.json", peak_mb,
                        elapsed, rc)
    (out / "stdout_tail.txt").write_text("\n".join(tail), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
