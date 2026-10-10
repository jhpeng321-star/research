"""G2 restored-branch parity reference (WM-72S; rented-box exactness gate).

Runs ONE restored chunk-4 TRAIN branch for a fixed length with the live
campaign driver (campaign_stage1_v3.run_chunk.run_branch_child, the same
path the chunk-4 branch stage uses) into a FRESH output dir, then writes a
digest JSON over its physics rows:

    per_tick_digests: [[time_s, sha256(json(sorted (id, lane, pos, speed)))], ...]
    whole_digest:     sha256 over the concatenated per-tick hex digests

The laptop run produces the reference (rented_bundle parity/
restored_branch_check.py --make-reference); the rented box reruns the SAME
command and must be hash-equal.  Positions/speeds are rounded to 9 decimals,
as in the fresh-base G1 gate.

Usage: python tools/g2_branch_reference.py --out <fresh dir> --digest-out <json>
       [--world w8676m2700p20t360] [--arm switching1] [--length-s 120]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("RESEARCH_DENSE_ROOT", "E:/research-dense"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "output"))

PLAN = ROOT / "output/campaign_stage1_v3/results_plan_chunk4/WORLD_PLAN.json"
REG = ROOT / "output/campaign_stage1_v3/results_plan_chunk4/CHUNK4_REGISTRATION.md"
BASES = ROOT / "runs/campaign_stage1_chunk4"


def digests(physics_path: Path) -> dict:
    ticks, whole = [], hashlib.sha256()
    for line in physics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows = sorted(
            (vid, v.get("lane_id"), round(float(v["lane_position_m"]), 9),
             round(float(v["speed_mps"]), 9))
            for vid, v in (row.get("vehicles") or {}).items())
        h = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
        ticks.append([float(row["time_s"]), h])
        whole.update(h.encode())
    return {"per_tick_digests": ticks, "whole_digest": whole.hexdigest(), "ticks": len(ticks)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--digest-out", required=True)
    ap.add_argument("--world", default="w8676m2700p20t360")
    ap.add_argument("--arm", default="switching1")
    ap.add_argument("--length-s", type=float, default=120.0)
    args = ap.parse_args(argv)

    out = Path(args.out)
    digest_out = Path(args.digest_out)
    if not out.is_absolute():
        out = ROOT / out
    if not digest_out.is_absolute():
        digest_out = ROOT / digest_out
    if out.exists() or digest_out.exists():
        print(f"REFUSING: output exists ({out} / {digest_out})", file=sys.stderr)
        return 1

    from campaign_stage1_v3 import run_chunk as rc

    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    world = copy.deepcopy(next(w for w in plan["worlds"] if w["world"] == args.world))
    if world.get("split") != "TRAIN":
        print(f"REFUSING: {args.world} is not TRAIN", file=sys.stderr)
        return 1
    run_name = f"{args.world}__{args.arm}"
    for arm in world["branches"]:
        arm["duration_s"] = args.length_s
    arm = next(a for a in world["branches"] if a["run"] == run_name)
    plan_one = dict(plan, worlds=[world])
    run = arm | {"world": args.world, "seed": world["seed"], "split": world["split"]}
    out.mkdir(parents=True)
    rc.run_branch_child(run, plan_one, out, REG, bases_root=BASES,
                        machine_id="g2-reference", full_rows_required=True)
    receipt = json.loads((out / run_name / "RUN_RECEIPT.json").read_text(encoding="utf-8"))
    if receipt.get("exit_status") != "complete":
        print(f"branch not complete: {receipt.get('exit_status')}", file=sys.stderr)
        return 2
    result = digests(out / run_name / "run" / "physics_steps.jsonl")
    result.update({"world": args.world, "arm": args.arm, "length_s": args.length_s})
    digest_out.parent.mkdir(parents=True, exist_ok=True)
    digest_out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"ticks {result['ticks']} whole {result['whole_digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
