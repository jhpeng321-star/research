"""Run one CPU module with sampled per-process RSS termination and a receipt."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

from .guard import workspace_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--limit-bytes", type=int, default=5_500_000_000)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if len(command) < 2 or command[0] != "-m":
        parser.error("pass -- -m MODULE [arguments]")
    if not 0 < args.limit_bytes < 6_000_000_000:
        parser.error("limit must be strictly below 6 GB")
    workspace_path(Path.cwd())
    receipt = workspace_path(args.receipt)
    if not receipt.parent.is_dir() or receipt.exists():
        parser.error("receipt needs an existing workspace parent and a new filename")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               PYTHONDONTWRITEBYTECODE="1")
    started = time.monotonic()
    peaks, breached = {}, False
    process = subprocess.Popen([sys.executable, "-B", *command], env=env)
    try:
        parent = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                children = [parent, *parent.children(recursive=True)]
            except psutil.NoSuchProcess:
                break
            for child in children:
                try:
                    rss = child.memory_info().rss
                    peaks[str(child.pid)] = max(rss, peaks.get(str(child.pid), 0))
                    if rss >= args.limit_bytes:
                        breached = True
                except psutil.NoSuchProcess:
                    continue
            if breached:
                for child in reversed(children):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                break
            time.sleep(0.05)
        code = process.wait()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        data = {"schema": "lpwo_cpu_memory/1", "command": [sys.executable, "-B", *command],
                "status": "memory_limit" if breached else "passed" if process.returncode == 0 else "failed",
                "returncode": process.returncode, "limit_bytes": args.limit_bytes,
                "sampling_interval_s": 0.05, "peak_rss_by_pid_bytes": peaks,
                "max_sampled_process_rss_bytes": max(peaks.values(), default=0),
                "elapsed_s": time.monotonic() - started,
                "caveat": "sampled RSS, not an instantaneous operating-system memory quota"}
        with receipt.open("x", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        print(json.dumps(data))
    return 3 if breached else code


if __name__ == "__main__":
    raise SystemExit(main())
