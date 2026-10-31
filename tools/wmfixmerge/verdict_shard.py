"""Panel-A shard launcher (MAIN ADDENDUM 2026-10-31 12:45: verdict panels sharded).

Runs tools.wmfixmerge.verdict_panels.validation for a SUBSET of the 36
corrected-validation groups on one GPU. Per-group A_{name}_{i:03d}.json files
are immutable, binding-checked and skipped when they already exist, so disjoint
shards on gpuA/gpuR merge by construction and the one-command verdict driver
(verdict.py, unchanged) later reuses exactly those files for its assembly:
same score.score_batch code path, same inputs, per-call CRN seeds (determinism
proven r1==r0 and by the gate-(c) cross-device group checks). Panel B stays
single-process by design (frozen producer closure flock).
"""
from __future__ import annotations

import argparse
import json

from tools.wmfixmerge.data import digest, install_guard, safe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--plan", default="tools/wmfixmerge/verdict_plan.json")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--groups", required=True, help="e.g. 0-17 (this shard's subset)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=None)
    args = parser.parse_args(argv)
    install_guard()
    plan = json.loads(safe(args.plan).read_text())
    root = safe(args.train_dir)
    final = json.loads(safe(root / "FINAL.json").read_text())
    if final.get("status") != "COMPLETE" or digest(final["checkpoint"]) != final["sha256"]:
        raise ValueError("no immutable trained final checkpoint")
    candidate = safe(final["checkpoint"])
    config = root / "candidate_config.json"
    out = safe(args.out)
    if not out.is_dir():
        raise ValueError("existing shard output directory required")
    import torch
    torch.set_num_threads(2)
    if args.device == "cuda" and args.memory_fraction:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    from tools.wmfixmerge import verdict_panels as panels
    result = panels.validation(plan, candidate, config, out, args.device, groups=args.groups)
    print(json.dumps(result if result.get("schema") == "wmfix_validation_shard/1"
                     else dict(schema="assembled", rows=len(result["rows"])),
                     allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
