"""Run a CPU verification command with workspace-local artifacts and a 6 GB guard.

Usage: python tools/brakefix_verify.py --tag focused_01 -- -m pytest ...
The child's full stdout/stderr and exit status are preserved; tags cannot be reused.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 6_000_000_000


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("args", nargs=argparse.REMAINDER)
    args = p.parse_args()
    if not args.tag or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in args.tag):
        p.error("tag must contain lowercase letters, digits, underscore or hyphen")
    child_args = args.args[1:] if args.args[:1] == ["--"] else args.args
    command = [sys.executable, *child_args]
    if len(command) == 1:
        p.error("a Python command is required")
    out = ROOT / "_out" / "brakefix_verification" / args.tag
    out.mkdir(parents=True, exist_ok=False)
    temp = out / "tmp"
    temp.mkdir()
    env = dict(os.environ, TMP=str(temp), TEMP=str(temp), TMPDIR=str(temp),
               CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    peak_rss = peak_os = 0
    exceeded = False
    started = time.perf_counter()
    with (out / "command.log").open("x", encoding="utf-8") as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT)
        monitor = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                memory = monitor.memory_info()
                peak_rss = max(peak_rss, memory.rss)
                peak_os = max(peak_os, getattr(memory, "peak_wset", memory.rss))
                if max(memory.rss, peak_os) >= LIMIT:
                    exceeded = True
                    proc.kill()
            except psutil.Error:
                pass
            time.sleep(0.05)
        rc = proc.wait()
    receipt = {"command": command, "returncode": rc,
               "status": "passed" if rc == 0 and not exceeded else "failed",
               "peak_sampled_rss_bytes": peak_rss, "peak_os_working_set_bytes": peak_os,
               "memory_limit_bytes": LIMIT, "memory_limit_exceeded": exceeded,
               "wall_seconds": round(time.perf_counter() - started, 3),
               "log": str(out / "command.log"), "device": "cpu"}
    (out / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
