"""PILOT v2 panel shard evaluator (MAIN 13:50 TRACK 1).

Scores an immutable exported pilot checkpoint on a SUBSET of the combined
52-group TRAIN panel (arms 0,1 in the fixed sorted arm order of group_rows;
horizons 120/300 s) with the exact integer evaluation path: the Runtime is
constructed WITHOUT smooth_events (training-only flag never set here). The
per-row contrast is identical to pilot.panel: (WM arm - WM native) - (real arm
- real native) on the J column. Writes SHARD_{name}.json once (open 'x').
"""
from __future__ import annotations

import argparse
import json
import socket
import time

from tools.wmfixmerge.data import WM2_SHA, digest, install_guard, safe

HORIZONS = (120, 300)
ARMS = (0, 1)


def shard_rows(rt, groups, horizons=HORIZONS, arms=ARMS):
    import torch
    rows = []
    for h in horizons:
        for group in groups:
            for arm in arms:
                with torch.no_grad():
                    predicted, recorded, _, receipt = rt.rollout(group, h, arm_index=arm)
                contrast = float(((predicted[1, 0] - predicted[0, 0]) -
                                  (recorded[1, 0] - recorded[0, 0])))
                rows.append(dict(group=int(group), horizon=int(h), arm=int(arm),
                                 pair_key=receipt["pair_key"], arms=receipt["arms"],
                                 contrast=contrast))
    if rt.device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "config", "checkpoint", "controllers", "candidate", "out", "name",
                 "groups"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=None)
    args = parser.parse_args(argv)
    install_guard()
    if digest(args.checkpoint) != WM2_SHA:
        raise ValueError("base is not registered WM2")
    out = safe(args.out)
    target = out / f"SHARD_{args.name}.json"
    if target.exists():
        raise FileExistsError("shard receipt already exists: " + target.name)
    groups = [int(g) for g in args.groups.split(",") if g != ""]
    if not groups or len(set(groups)) != len(groups):
        raise ValueError("empty or duplicate shard groups")
    import torch
    torch.set_num_threads(2)
    if args.device == "cuda" and args.memory_fraction:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    from tools.wmfixmerge.panel_eval import load_candidate
    from tools.wmfixmerge.runtime import Runtime
    began = time.monotonic()
    rt = Runtime(args.prepared, args.config, args.checkpoint, args.controllers, args.device)
    if rt.smooth_events:
        raise AssertionError("evaluation must use the exact integer path")
    payload = load_candidate(rt, args.checkpoint, args.candidate)
    rows = shard_rows(rt, groups)
    with target.open("x") as stream:
        json.dump(dict(step=int(payload["wmfix"]["step"]), name=args.name, groups=groups,
                       rows=rows, seconds=round(time.monotonic() - began, 1),
                       producer=dict(host=socket.gethostname(), device=str(rt.device),
                                     candidate_sha256=digest(args.candidate),
                                     prepared_sha256=digest(args.prepared),
                                     base_sha256=WM2_SHA)),
                  stream, indent=2, allow_nan=False)
    print(json.dumps(dict(name=args.name, rows=len(rows), wrote=str(target))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
