"""WM-N1 featurizer CLI (resumable, multi-worker, zero SUMO).

    python -m features featurize --chunk chunk4 --worlds W --arms base,native \
        --data-root <dir> --workers 2
    python -m features estimate --chunk chunk4
    python -m features verify --data-root <dir>

The split of every world is resolved from its chunk WORLD_PLAN.json BEFORE
any run file is opened; HOLDOUT and excluded worlds are refused, and chunk-5
needs ``--final-test --registration-sha
<LF-sha256 of REGISTRATION_CHUNK5_TEST.md>``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import constants as C
from . import splits as S
from . import writer as W


def _plan_world(plan: dict, world: str) -> dict:
    for w in plan["worlds"]:
        if w["world"] == world:
            return w
    raise S.SplitRefusal(f"world {world!r} not in plan")


def _arm_suffix(branch: dict) -> str:
    run = branch.get("run")
    if run and "__" in str(run):
        return str(run).split("__", 1)[1]
    return str(branch.get("arm"))


def _arms_of(world_row: dict, chunk: str, requested):
    if requested:
        return list(requested)
    if world_row.get("branches"):
        return (["base", "native_base"]
                + [_arm_suffix(b) for b in world_row["branches"]])
    return ["base"]


def _n_ticks_estimate(world_row: dict, arm: str) -> int:
    if arm in ("base", "native_base"):
        d = float((world_row.get("base_arm") or {}).get("duration_s", 900.0))
        return int(round(d / C.DT_S)) + 1
    for b in world_row.get("branches", ()):
        if b["arm"] == arm:
            return int(round(float(b.get("duration_s", 600.0)) / C.DT_S)) + 1
    return int(round(600.0 / C.DT_S)) + 1


def cmd_estimate(args) -> int:
    fmt, K = C.resolve_k_max(format=args.format)
    plan = S.load_plan(args.chunk, args.plan)
    keep, hold = S.chunk_worlds(args.chunk, args.plan)
    worlds = args.worlds.split(",") if args.worlds else keep
    total = 0
    rows = []
    for world in worlds:
        row = _plan_world(plan, world)
        split = S.decide_split(
            args.chunk, world, plan_path=args.plan,
            final_test=args.final_test,
            registration_sha256=args.registration_sha)
        for arm in _arms_of(row, args.chunk, args.arms.split(",")
                            if args.arms else None):
            n = _n_ticks_estimate(row, arm)
            b = W.expected_bytes(n, K, k_max=K)
            rows.append((split, world, arm, n, b))
            total += b
    print(f"estimate chunk={args.chunk} worlds={len(worlds)} "
          f"held_out={len(hold)} format={fmt} k_max={K}")
    for split, world, arm, n, b in sorted(rows):
        print(f"  {split:11s} {world:22s} {arm:12s} n_ticks={n:5d} "
              f"bytes={b/1e6:9.3f} MB")
    print(f"TOTAL expected bytes: {total} ({total/1e9:.3f} GB) "
          f"runs={len(rows)}")
    return 0


def _run_one(job: dict) -> dict:
    from .featurizer import build_shard
    shard = build_shard(job["chunk"], job["world"], job["arm"],
                        job["run_dir"], split=job["split"],
                        plan_row=job["plan_row"],
                        outcomes_root=job.get("outcomes_root"),
                        base_history_run_dir=job.get("base_history_run_dir"),
                        format=job.get("format"))
    rec = W.write_shard(shard, job["data_root"], force=job["force"])
    return dict(status=rec["status"], record=rec["record"])


def cmd_featurize(args) -> int:
    fmt, K = C.resolve_k_max(format=args.format)
    plan = S.load_plan(args.chunk, args.plan)
    keep, hold = S.chunk_worlds(args.chunk, args.plan)
    worlds = args.worlds.split(",") if args.worlds else keep
    arms_req = args.arms.split(",") if args.arms else None

    jobs = []
    for world in worlds:
        row = _plan_world(plan, world)
        split = S.decide_split(
            args.chunk, world, plan_path=args.plan,
            final_test=args.final_test,
            registration_sha256=args.registration_sha)
        for arm in _arms_of(row, args.chunk, arms_req):
            base_hist = None
            if arm not in ("base", "native_base"):
                base_name = (row.get("base_arm") or {}).get("run") \
                    or f"{world}__base"
                base_root = Path(args.runs_root) if args.runs_root \
                    else S.CHUNKS[args.chunk]["runs"]
                base_hist = str(base_root / base_name)
            jobs.append(dict(
                chunk=args.chunk, world=world, arm=arm,
                run_dir=str(S.run_dir(args.chunk, world, arm,
                                      args.runs_root)),
                split=split, plan_row=row, data_root=str(args.data_root),
                outcomes_root=args.outcomes_root, force=args.force,
                base_history_run_dir=base_hist, format=fmt))

    if args.estimate_only:
        total = 0
        for j in jobs:
            n = _n_ticks_estimate(j["plan_row"], j["arm"])
            total += W.expected_bytes(n, K, k_max=K)
        print(f"estimate-only: {len(jobs)} runs, expected "
              f"{total} bytes ({total/1e9:.3f} GB); no shard written")
        return 0

    t_start = time.time()
    records = []
    if args.workers <= 1:
        for j in jobs:
            records.append(_run_one(j))
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for rec in ex.map(_run_one, jobs):
                records.append(rec)

    data_root = Path(args.data_root)
    index = W.load_index(data_root)
    for rec in records:
        W.update_index(index, rec["record"])
    W.save_index(data_root, index)
    W.write_sums(data_root, index)

    n_written = sum(1 for r in records if r["status"] == "written")
    n_skipped = sum(1 for r in records if r["status"] == "skipped")
    secs = time.time() - t_start
    print(f"featurize: runs={len(records)} written={n_written} "
          f"skipped={n_skipped} wall={secs:.1f}s")
    for r in records:
        rec = r["record"]
        print(f"  [{r['status']:7s}] {rec['chunk']}/{rec['world']}__"
              f"{rec['arm']} n_ticks={rec['n_ticks']} "
              f"bytes={rec['total_bytes']} ({rec['total_bytes']/1e6:.3f} MB) "
              f"pair_key={W.pair_key(rec['world'], rec['t0_s'])}")
    return 0


def cmd_verify(args) -> int:
    index = W.load_index(args.data_root)
    rep = W.verify_index(args.data_root, index)
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="features")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--chunk", required=True)
        sp.add_argument("--worlds", default=None)
        sp.add_argument("--arms", default=None)
        sp.add_argument("--plan", default=None)
        sp.add_argument("--runs-root", default=None)
        sp.add_argument("--final-test", action="store_true")
        sp.add_argument("--registration-sha", dest="registration_sha",
                        default=None)
        sp.add_argument("--format", choices=sorted(C.FORMAT_K_MAX),
                        default=C.DEFAULT_FORMAT,
                        help="slot-width format: v2 keeps K_max=128 "
                             "(frozen, byte-reproducible); v3 carries "
                             "K_max=192 (amendment 5)")

    f = sub.add_parser("featurize")
    common(f)
    f.add_argument("--data-root", required=True)
    f.add_argument("--outcomes-root", dest="outcomes_root", default=None)
    f.add_argument("--workers", type=int, default=1)
    f.add_argument("--force", action="store_true")
    f.add_argument("--estimate-only", action="store_true")
    f.set_defaults(func=cmd_featurize)

    e = sub.add_parser("estimate")
    common(e)
    e.set_defaults(func=cmd_estimate)

    v = sub.add_parser("verify")
    v.add_argument("--data-root", required=True)
    v.set_defaults(func=cmd_verify)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
