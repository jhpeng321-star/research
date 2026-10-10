"""Run-3 CPU dry-run harness (one process at a time, threads 4, small batch).

Runs the SAME ``wms.cli train`` entry point and flag set as ``_out/run3_chain.sh``
(real packs, entries-v2, merge-cal, traffic-loss, stability, conservation metric
AND loss, loss-share logging) over a tiny multi-stage curriculum that passes
through every stage (10/30/60/120/300 s).  It then applies the multi-stage smoke
gate ``wms.gate_stages`` to the produced log.

Memory rule: the packs are memory-mapped by the dataset (``mmap_mode='r'``), one
batch is materialised at a time and one run is opened at a time; the child's peak
RSS is monitored and must stay under 6 GB.  Data is READ-ONLY and TRAIN-only.

Usage::

    python _out/run3_dry.py --tag stages_cpu --stage-steps 1,1,1,1,1
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
if str(CWD) not in sys.path:
    sys.path.insert(0, str(CWD))
BASE = "D:/rd-staging/wmn1_feat/pack_v3"
ID_BASE = "E:/tmp/wmn1_train_runs_20261002/sidecars3_20261003T0005CN/identities"
D9 = "E:/rd-wt/dspeed-data/_out"
D10 = "E:/rd-wt/damping-snap/_out"
ID_DS = "E:/rd-wt/a9/_out/identities_dspeed"
ID_DM = "E:/rd-wt/damping-snap/_out/identities_damping"
LPO = "E:/rd-wt/ws-p5/_out/lpo"
DEFAULT_OUTCOMES = "E:/rd-wt/ws-p2/_ext/labels_cov"   # TRAIN-only schema/3 sidecar
STAGES = "10,30,60,120,300"
TERMS = "position,speed,lane,traffic,stability,conservation"


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
        "--batch-size", str(args.batch_size), "--sumo-fidelity", "--start-shift-prob", "0.75",
        "--truncation-steps", "20",
        "--entries-v2", "--entry-pad", "8",
        "--merge-cal", "--merge-cal-params", str(ROOT / "_out" / "merge_params.json"),
        "--traffic-loss",
        "--projection", "--projection-min-gap-m", "2.0",
        "--residual-cap-mps2", "1.5", "--lane-logit-cap", "3.0",
        "--stability-weight", "0.01", "--stability-margin", "1.0", "--stability-probes", "1",
        "--conservation-metric", "--conservation-weight", "0.01",
        "--loss-share-logging",
    ]
    for s in args.unroll_s.replace(",", " ").split():
        cmd += ["--unroll-s", s]
    if args.stage_steps:
        cmd += ["--stage-steps", args.stage_steps]
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


def summarize(log: Path, out_json: Path, peak_mb: float, elapsed: float, rc: int,
              gate: dict | None) -> dict:
    rows = []
    if log.is_file():
        rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()
                if x.strip()]
    per_stage: dict[str, dict] = {}
    for row in rows:
        key = str(round(float(row.get("stage_s", -1.0)), 3))
        bucket = per_stage.setdefault(key, {})
        for term in TERMS.split(","):
            if term in row and isinstance(row[term], (int, float)):
                bucket.setdefault(term, []).append(row[term])
    terms = {stage: {t: {"mean": sum(v) / len(v), "last": v[-1],
                         "finite": all(math.isfinite(float(x)) for x in v)}
                     for t, v in bucket.items()}
             for stage, bucket in per_stage.items()}
    summary = {
        "rc": rc, "updates": len(rows),
        "stages_present": sorted(float(x) for x in per_stage),
        "per_stage_terms": terms,
        "valid_targets_last": rows[-1].get("valid_targets") if rows else None,
        "mean_step_seconds": (sum(r.get("seconds", 0) for r in rows) / len(rows)
                              if rows else None),
        "peak_rss_mb": round(peak_mb, 1), "over_6gb": peak_mb > 6144.0,
        "wall_seconds": round(elapsed, 2),
        "gate": gate,
    }
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--unroll-s", default=STAGES)
    ap.add_argument("--stage-steps", default="1,1,1,1,1")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=None)
    args = ap.parse_args()

    out = ROOT / "_out" / f"run3_dry_{args.tag}"
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
    tail: list[str] = []
    for line in proc.stdout:
        tail.append(line.rstrip())
        if len(tail) > 30:
            tail.pop(0)
    rc = proc.wait()
    elapsed = time.perf_counter() - started
    stop.set()
    peak_mb = peak.get("rss", 0) / 2 ** 20
    (out / "stdout_tail.txt").write_text("\n".join(tail), encoding="utf-8")

    gate = None
    log = out / "train_log.jsonl"
    if rc == 0 and log.is_file():
        from wms.gate_stages import evaluate
        gate = evaluate(log, stages=tuple(float(x) for x in args.unroll_s.split(",")),
                        terms=tuple(TERMS.split(",")))
        (ROOT / "_out" / f"run3_dry_{args.tag}_gate.json").write_text(
            json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8")
    summary = summarize(log, ROOT / "_out" / f"run3_dry_{args.tag}.json", peak_mb,
                        elapsed, rc, gate)
    print(json.dumps(summary, indent=2, sort_keys=True))
    ok = rc == 0 and gate is not None and gate.get("passed") is True and peak_mb <= 6144.0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
