"""S-WMFIX-ES task 1: single-group TRAIN-panel CPU throughput probe.

Measures seconds and peak RSS of one TRAIN-panel group (K8, 120+300 s, entry
path conserve_queue+learned admission, mix ON) with torch threads=1 per
process, for the CPU aggregate-throughput comparison in WMFIX_ES.md. Also
prints per-rollout paired-J contrasts for a GPU cross-check against the
FEASIBILITY gate panel (group medians; single cells may differ by hard-event
flips, per the gpuA/gpuR mixed-provenance study).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import socket
import time

from tools.wmfixmerge.data import install_guard, safe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "config", "checkpoint", "controllers", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--group", type=int, default=0)
    parser.add_argument("--horizons", default="120,300")
    parser.add_argument("--arms", default="0,1")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--future-index", type=int, default=0)
    parser.add_argument("--n-futures", type=int, default=8)
    args = parser.parse_args(argv)
    install_guard()
    import torch
    torch.set_num_threads(int(args.threads))
    from tools.wmfixmerge.runtime import Runtime
    began = time.monotonic()
    rt = Runtime(args.prepared, args.config, args.checkpoint, args.controllers, "cpu")
    load_seconds = time.monotonic() - began
    rows = []
    for arm in (int(a) for a in args.arms.split(",")):
        for horizon in (int(h) for h in args.horizons.split(",")):
            began = time.monotonic()
            with torch.no_grad():
                predicted, recorded, _, receipt = rt.rollout(args.group, horizon, arm_index=arm,
                    future_index=args.future_index, n_futures=args.n_futures)
            row = dict(group=args.group, arm=arm, horizon=horizon,
                       seconds=round(time.monotonic() - began, 2),
                       paired_J_contrast=float(((predicted[1, 0] - predicted[0, 0])
                                                - (recorded[1, 0] - recorded[0, 0]))),
                       rollout=receipt)
            rows.append(row)
            print(json.dumps(row), flush=True)
    result = dict(schema="wmfix_es_probe/1", host=socket.gethostname(), pid=os.getpid(),
                   threads=args.threads, torch=torch.__version__, device="cpu",
                   future_index=args.future_index, n_futures=args.n_futures,
                  runtime_load_seconds=round(load_seconds, 1),
                  group_seconds=round(sum(r["seconds"] for r in rows), 1),
                  max_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024., 1),
                  rows=rows)
    with safe(args.out).open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
