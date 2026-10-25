#!/usr/bin/env python3
"""Shared-GPU guard query for loop14 (``guards.gpu_query``) on a GPU shared with another job.

    python scripts/c4/gpu_headroom.py NEED_MIB

loop14's gpu_guard treats ANY stdout line as "GPU busy" (exit 5, nothing is signalled).  On gpuB the
loop shares the GPU with a world-model training job by design (owner 2026-10-25), so instead of
"no compute process at all" this prints a busy line only when the free GPU memory is below
NEED_MIB (the loop job's C4_GPU_MEM_FRACTION cap plus a margin).  It prints nothing when there is
headroom.  If nvidia-smi fails it exits non-zero, and loop14 then fails closed.  Read-only.
"""
import subprocess
import sys


def main() -> int:
    need = int(sys.argv[1])
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60)
    if out.returncode != 0 or not out.stdout.strip():
        print(f"nvidia-smi failed: {out.returncode} {out.stderr.strip()}", file=sys.stderr)
        return 1
    idx, total, used = (int(x) for x in out.stdout.strip().splitlines()[0].split(","))
    free = total - used
    if free < need:
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=60).stdout
        print(f"GPU {idx}: free {free} MiB < need {need} MiB; apps: {' | '.join(apps.split(chr(10))).strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
