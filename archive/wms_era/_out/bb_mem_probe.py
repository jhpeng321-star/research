"""Memory-bounded runner for the WM-N1 black-box real-data CPU dry run.

Launches ``python -m wms.cli dry-run ... --blackbox-traj`` in a subprocess and
samples the process tree's resident/unique memory.  Aborts (by PID, never by
image name) if the sampled unique-set size exceeds the 6 GB rule.  Prints a JSON
summary to stdout so the peak can be recorded in the report.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

PY = r"C:\Users\jpeng\Miniconda3\envs\rl_driving\python.exe"
WM = Path(r"E:\rd-wt\ws-bb2\output\wd01\wm_n1")
OUT = WM / "wms" / "runs" / "bb_traj_dry_20261006d"
LIMIT_MB = 6000.0
SAMPLE_S = 0.25


def proc_tree(root: psutil.Process):
    yield root
    try:
        for child in root.children(recursive=True):
            yield child
    except psutil.Error:
        return


def sample(root: psutil.Process) -> tuple[float, float, float]:
    rss = uss = priv = 0.0
    for p in proc_tree(root):
        try:
            mi = p.memory_full_info()
            rss += mi.rss / 2 ** 20
            uss += mi.uss / 2 ** 20
            priv += getattr(mi, "private", mi.uss) / 2 ** 20
        except psutil.Error:
            continue
    return rss, uss, priv


def main() -> int:
    env = dict(os.environ)
    env["WM_N1_TRAIN_OUTCOMES"] = r"D:/rd-staging/wmn1_feat/labels_cov_20261004T072826Z"
    env["WM_N1_TRAIN_IDENTITIES"] = (
        r"E:/tmp/wmn1_train_runs_20261002/sidecars3_20261003T0005CN/identities")
    env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "4"
    cmd = [
        PY, "-m", "wms.cli", "dry-run",
        "--pack", r"D:/rd-staging/wmn1_feat/pack_v3",
        "--identities", env["WM_N1_TRAIN_IDENTITIES"],
        "--out", str(OUT), "--threads", "4", "--batch-size", "4",
        "--unroll-s", "10", "--blackbox-traj",
    ]
    log = OUT.with_suffix(".stdout.log")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, cwd=str(WM), env=env, stdout=fh,
                                stderr=subprocess.STDOUT)
        root = psutil.Process(proc.pid)
        peak = [0.0, 0.0, 0.0]
        aborted = False
        while proc.poll() is None:
            rss, uss, priv = sample(root)
            peak = [max(a, b) for a, b in zip(peak, (rss, uss, priv))]
            if uss > LIMIT_MB:
                aborted = True
                try:
                    for p in proc_tree(root):
                        p.kill()
                except psutil.Error:
                    pass
                break
            time.sleep(SAMPLE_S)
        rc = proc.wait()
    result = {
        "returncode": rc,
        "aborted_over_limit": aborted,
        "limit_mb": LIMIT_MB,
        "peak_tree_rss_mb": round(peak[0], 1),
        "peak_tree_uss_mb": round(peak[1], 1),
        "peak_tree_private_mb": round(peak[2], 1),
        "out_dir": str(OUT),
        "log": str(log),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if (rc == 0 and not aborted) else 1


if __name__ == "__main__":
    sys.exit(main())
