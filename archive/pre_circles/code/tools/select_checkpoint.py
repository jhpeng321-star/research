"""Checkpoint selection by GATED eval travel time.

Replaces policy_reward_mean selection: each candidate checkpoint is evaluated
on the reserved selection seeds (default 47,48,49 — do NOT use the reporting
seeds 42-46 here) at one density and CAV penetration. The winner is the lowest
mean travel time that passes all gates.

Gates (per eval, all must pass, else the candidate scores +inf):
  - TTC events     <= ttc_factor x that seed's All-HDV TTC (default 1.0)
  - collisions     sum over seeds <= sum of All-HDV collisions (aggregate)
  - mean_time_loss <= 1.05 x that seed's All-HDV mean_time_loss
  - easy-mainline  on seeds with HDV TT <= median: Delta mainline TT <= +2.0 s

HDV references are read from --hdv-dir (default: runs/eval) as
all_hdv_{density}.seed{N}.json; generate them first with:
  python evaluate_sumo_baseline.py <density> --seeds 47,48,49

Examples
--------
# Dry run (default): list candidates and what would be evaluated, no SUMO:
python tools/select_checkpoint.py --run-dir runs/rllib/shepherd_v20m --density medium

# Actually evaluate and pick a winner:
python tools/select_checkpoint.py --run-dir runs/rllib/shepherd_v20m --density medium --execute

Outputs (under runs/eval/):
  select_{run}_iter{N}_{density}_p{pen}.seed{S}.json   per-seed eval metrics
  select_{run}_{density}_summary.json                  full sweep + winner
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "runs" / "eval"
# Shared SUMO network lives under D:\research (research_v2 is a sibling copy).
SUMO_ROOT = Path(r"D:\research\highway_merge_shepherd")
DEFAULT_HDV_DIR = Path(r"D:\research\traffic_shepherd_rl\runs\eval")

TTC_FACTOR = 1.0
TIME_LOSS_FACTOR = 1.05
EASY_ML_DELTA_MAX = 2.0  # seconds; over-intervention guard on easy HDV seeds
POLICY_NAME = "shepherd"


def pen_tag(pen: float) -> str:
    return f"p{int(round(pen * 100)):03d}"


def eval_json_path(out_dir, run_name, cand_name, density, pen, seed) -> Path:
    name = f"select_{run_name}_{cand_name}_{density}_{pen_tag(pen)}.seed{seed}.json"
    return Path(out_dir) / name


def find_candidates(run_dir, every: int = 10):
    """iter_* checkpoints with iteration % every == 0, sorted, plus `final`."""
    ckpt_root = Path(run_dir) / "checkpoints"
    if not ckpt_root.is_dir():
        raise FileNotFoundError(f"no checkpoints/ directory under {run_dir}")
    found = []
    for d in ckpt_root.iterdir():
        m = re.fullmatch(r"iter_(\d+)", d.name)
        if not (d.is_dir() and m):
            continue
        n = int(m.group(1))
        if every and n % every:
            continue
        found.append((n, d.name, d))
    found.sort(key=lambda t: t[0])
    final = ckpt_root / "final"
    if final.is_dir():
        found.append((None, "final", final))
    return found


def load_hdv_refs(density, seeds, out_dir=DEFAULT_HDV_DIR):
    """Per-seed all-HDV reference metrics; returns ({seed: metrics}, [missing paths])."""
    refs, missing = {}, []
    for seed in seeds:
        path = Path(out_dir) / f"all_hdv_{density}.seed{seed}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                refs[seed] = json.load(f)
        else:
            missing.append(path)
    return refs, missing


def _ttc_events(metrics: dict) -> int:
    t = metrics.get("ttc")
    if isinstance(t, dict):
        return int(t.get("total_events") or 0)
    return int(t or 0)


def score_candidate(
    per_seed_metrics,
    hdv_refs,
    time_loss_factor: float = TIME_LOSS_FACTOR,
    ttc_factor: float = TTC_FACTOR,
    easy_ml_delta_max: float = EASY_ML_DELTA_MAX,
):
    """Score one candidate by gated mean travel time.

    Args:
        per_seed_metrics: one evaluate_policy metrics dict per selection seed
            (needs travel_time.all/mainline.mean, ttc, collisions, mean_time_loss).
        hdv_refs: one all-HDV metrics dict per selection seed (same fields).

    Returns:
        (score, gate_report): score is the mean travel_time.all.mean over the
        seeds, or +inf if any gate fails. gate_report holds per-gate detail.
    """
    if len(per_seed_metrics) != len(hdv_refs):
        raise ValueError("per_seed_metrics and hdv_refs length mismatch")

    tt = [float(m["travel_time"]["all"]["mean"]) for m in per_seed_metrics]
    ttc = [_ttc_events(m) for m in per_seed_metrics]
    h_ttc = [_ttc_events(h) for h in hdv_refs]
    coll = [float(m.get("collisions") or 0) for m in per_seed_metrics]
    h_coll = [float(h.get("collisions") or 0) for h in hdv_refs]
    tl = [float(m["mean_time_loss"]) for m in per_seed_metrics]
    h_tl = [float(h["mean_time_loss"]) for h in hdv_refs]
    htt = [float(h["travel_time"]["all"]["mean"]) for h in hdv_refs]
    dml = [
        float(m["travel_time"]["mainline"]["mean"]) - float(h["travel_time"]["mainline"]["mean"])
        for m, h in zip(per_seed_metrics, hdv_refs)
    ]
    median_htt = sorted(htt)[len(htt) // 2]
    easy_flags = [h <= median_htt for h in htt]
    easy_dml = [d if easy else None for d, easy in zip(dml, easy_flags)]

    ttc_pass = all(p <= ttc_factor * h for p, h in zip(ttc, h_ttc))
    coll_sum_pass = sum(coll) <= sum(h_coll)
    tl_pass = all(p <= time_loss_factor * h for p, h in zip(tl, h_tl))
    easy_ml_pass = all(d is None or d <= easy_ml_delta_max for d in easy_dml)

    gates = {
        "ttc": {
            "pass": ttc_pass,
            "per_seed": ttc,
            "hdv_per_seed": h_ttc,
            "factor": ttc_factor,
        },
        "collisions": {
            "pass": coll_sum_pass,
            "per_seed": coll,
            "hdv_per_seed": h_coll,
            "pol_sum": sum(coll),
            "hdv_sum": sum(h_coll),
        },
        "time_loss": {
            "pass": tl_pass,
            "per_seed": tl,
            "hdv_per_seed": h_tl,
            "factor": time_loss_factor,
        },
        "easy_mainline": {
            "pass": easy_ml_pass,
            "per_seed_dml": dml,
            "easy_seeds": [i for i, e in enumerate(easy_flags) if e],
            "median_hdv_tt": median_htt,
            "max_delta": easy_ml_delta_max,
        },
    }
    all_pass = all(g["pass"] for g in gates.values())
    score = sum(tt) / len(tt) if all_pass else math.inf
    report = dict(gates)
    report["all_pass"] = all_pass
    report["travel_time_per_seed"] = tt
    return score, report


def run_seed_eval(ckpt_dir, density, pen, seed, out_json, max_steps):
    """Evaluate one checkpoint on one seed, or load the cached JSON.

    Returns (metrics_dict, cached). A failed eval is written as an error
    record ({"ok": False, "error": ...}) so resume skips it; the caller
    treats records without "travel_time" as failed.
    """
    out_json = Path(out_json)
    if out_json.exists():
        with open(out_json, encoding="utf-8") as f:
            return json.load(f), True
    t1 = time.time()
    try:
        # Lazy imports: dry runs and unit tests must not need gym/ray.
        from evaluate_policy import make_policy, run_policy_evaluation
        from policies.rllib_policy import checkpoint_custom_model_config

        policy = make_policy(POLICY_NAME, checkpoint_path=str(ckpt_dir))
        custom = checkpoint_custom_model_config(str(ckpt_dir)) or {}
        env_overrides = {
            k: custom[k] for k in ("max_cavs_critic", "max_neighbors") if k in custom
        } or None
        metrics = run_policy_evaluation(
            policy=policy,
            density=density,
            project_root=SUMO_ROOT.resolve(),
            sumo_binary="sumo",
            seed=seed,
            max_episode_steps=max_steps,
            hdv_mixture=None,
            cav_penetration=pen,
            policy_name=POLICY_NAME,
            env_overrides=env_overrides,
        )
        metrics["ok"] = True
    except Exception as exc:  # noqa: BLE001 - a failed eval must not kill the sweep
        traceback.print_exc()
        metrics = {"ok": False, "error": str(exc)}
    metrics.update(
        {
            "checkpoint": str(ckpt_dir),
            "density": density,
            "seed": seed,
            "cav_penetration": pen,
            "elapsed_s": time.time() - t1,
        }
    )
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    return metrics, False


def evaluate_candidate(cand_name, ckpt_dir, density, pen, seeds, run_name, max_steps):
    """Gather per-seed metrics (cached or freshly evaluated) for one candidate.

    Returns (per_seed_metrics or None, note or None). per_seed_metrics is None
    when any seed failed; note describes the failures.
    """
    per_seed, notes = [], []
    for seed in seeds:
        out_json = eval_json_path(OUT_DIR, run_name, cand_name, density, pen, seed)
        metrics, cached = run_seed_eval(ckpt_dir, density, pen, seed, out_json, max_steps)
        src = "cached" if cached else f"eval {metrics.get('elapsed_s', 0):.0f}s"
        if "travel_time" not in metrics:
            err = metrics.get("error", "no metrics")
            notes.append(f"seed {seed}: {err}")
            print(f"  seed {seed}: FAIL [{src}]: {err}", flush=True)
        else:
            print(
                f"  seed {seed}: [{src}] all_mean={metrics['travel_time']['all']['mean']:.1f} "
                f"ttc={metrics['ttc']['total_events']} coll={metrics['collisions']} "
                f"time_loss={metrics['mean_time_loss']:.1f}",
                flush=True,
            )
        per_seed.append(metrics)
    if notes:
        return None, "; ".join(notes)
    return per_seed, None


def _gate_cell(report, gate):
    if report is None:
        return "--"
    return "ok" if report[gate]["pass"] else "FAIL"


def print_table(rows, seeds):
    tt_cols = "".join(f"TT(s{s})".rjust(9) for s in seeds)
    print(f"\n{'candidate':<10}{tt_cols}{'ttc':>6}{'coll':>6}{'tloss':>6}{'ezML':>6}{'score':>9}")
    for r in rows:
        report = r["gates"]
        tts = (report or {}).get("travel_time_per_seed") or [None] * len(seeds)
        cells = "".join(f"{v:9.1f}" if v is not None else "--".rjust(9) for v in tts)
        gates = "".join(
            _gate_cell(report, g).rjust(6)
            for g in ("ttc", "collisions", "time_loss", "easy_mainline")
        )
        score = f"{r['score']:9.2f}" if math.isfinite(r["score"]) else "inf".rjust(9)
        print(f"{r['name']:<10}{cells}{gates}{score}")
        if r["note"]:
            print(f"  note: {r['note']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[1:]),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training run directory, e.g. runs/rllib/shepherd_v20m",
    )
    parser.add_argument("--density", required=True, choices=["light", "medium", "dense"])
    parser.add_argument(
        "--every",
        type=int,
        default=10,
        help="Only consider iter_N with N %% EVERY == 0 (default 10; 0 = all).",
    )
    parser.add_argument(
        "--min-iter", type=int, default=40, help="Skip iter_N with N < min-iter (default 40)."
    )
    parser.add_argument("--penetration", type=float, default=0.10)
    parser.add_argument(
        "--seeds",
        default="47,48,49,64,68",
        help="Comma-separated selection seeds (default 47,48,49,64,68 = reserved + hard/easy).",
    )
    parser.add_argument(
        "--hdv-dir",
        type=Path,
        default=DEFAULT_HDV_DIR,
        help="Directory with all_hdv_{density}.seedN.json refs.",
    )
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument(
        "--execute", action="store_true", help="Actually run the SUMO evals (default: dry run)."
    )
    args = parser.parse_args(argv)

    run_name = args.run_dir.name
    seeds = [int(s) for s in args.seeds.split(",")]

    try:
        candidates = find_candidates(args.run_dir, every=args.every)
    except FileNotFoundError as exc:
        raise SystemExit(f"ERROR: {exc}")
    if args.min_iter:
        candidates = [c for c in candidates if c[0] is None or c[0] >= args.min_iter]
    if not candidates:
        raise SystemExit(
            f"ERROR: no iter_* checkpoints (multiple of {args.every}) or final "
            f"under {args.run_dir}/checkpoints"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    refs, missing = load_hdv_refs(args.density, seeds, out_dir=args.hdv_dir)

    print(f"run:       {args.run_dir} ({len(candidates)} candidates, every={args.every})")
    print(f"density:   {args.density}   penetration: {args.penetration}   seeds: {seeds}")
    print(f"hdv-dir:   {args.hdv_dir}")
    print(f"sumo-root: {SUMO_ROOT}")
    if missing:
        msg = "missing HDV reference(s): " + ", ".join(str(p) for p in missing)
        if args.execute:
            raise SystemExit(
                f"ERROR: {msg}\nGenerate them first: "
                f"python evaluate_sumo_baseline.py {args.density} --seeds {args.seeds}"
            )
        print(f"WARNING: {msg} (gates cannot be computed)")
    else:
        coll_sum = sum(float(r.get("collisions") or 0) for r in refs.values())
        tl_mean = sum(float(r["mean_time_loss"]) for r in refs.values()) / len(refs)
        print(
            f"HDV refs:  collisions sum={coll_sum:.0f}  "
            f"mean_time_loss mean={tl_mean:.2f} "
            f"(TL gate: per-seed <= {TIME_LOSS_FACTOR}x HDV; "
            f"TTC <= {TTC_FACTOR}x HDV; easy-ML Delta <= {EASY_ML_DELTA_MAX}s)"
        )

    if not args.execute:
        print("\nDRY RUN - the following evals would run:")
        for _, name, ckpt in candidates:
            print(f"\n{name}  ({ckpt})")
            for seed in seeds:
                out_json = eval_json_path(
                    OUT_DIR, run_name, name, args.density, args.penetration, seed
                )
                status = "cached (skip)" if out_json.exists() else "would evaluate"
                print(f"  seed {seed}: {out_json.name}  [{status}]")
        summary_path = OUT_DIR / f"select_{run_name}_{args.density}_summary.json"
        print(f"\nsummary would be written to {summary_path}")
        print("re-run with --execute to evaluate.")
        return 0

    rows = []
    t0 = time.time()
    for idx, (iteration, name, ckpt) in enumerate(candidates, 1):
        print(f"\n=== [{idx}/{len(candidates)}] {name} ===", flush=True)
        per_seed, note = evaluate_candidate(
            name,
            ckpt,
            args.density,
            args.penetration,
            seeds,
            run_name,
            args.max_episode_steps,
        )
        if per_seed is None:
            score, report = math.inf, None
        else:
            score, report = score_candidate(per_seed, [refs[s] for s in seeds])
        rows.append(
            {
                "name": name,
                "iteration": iteration,
                "checkpoint": str(ckpt),
                "score": score,
                "gates": report,
                "note": note,
            }
        )

    print_table(rows, seeds)

    finite = [r for r in rows if math.isfinite(r["score"])]
    winner = min(finite, key=lambda r: r["score"]) if finite else None
    if winner:
        print(f"\nWINNER: {winner['name']}  score={winner['score']:.2f}s  ({winner['checkpoint']})")
    else:
        print("\nWINNER: none - every candidate failed gates or evals")

    summary_path = OUT_DIR / f"select_{run_name}_{args.density}_summary.json"
    summary = {
        "run_dir": str(args.run_dir),
        "run_name": run_name,
        "density": args.density,
        "cav_penetration": args.penetration,
        "seeds": seeds,
        "every": args.every,
        "policy": POLICY_NAME,
        "gate_config": {
            "ttc_factor": TTC_FACTOR,
            "time_loss_factor": TIME_LOSS_FACTOR,
            "easy_ml_delta_max": EASY_ML_DELTA_MAX,
            "collisions": "sum(pol) <= sum(hdv)",
        },
        "hdv_refs": {
            str(s): str(Path(args.hdv_dir) / f"all_hdv_{args.density}.seed{s}.json") for s in seeds
        },
        "candidates": rows,
        "winner": (
            {
                "name": winner["name"],
                "checkpoint": winner["checkpoint"],
                "score": winner["score"],
            }
            if winner
            else None
        ),
        "elapsed_s": time.time() - t0,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
