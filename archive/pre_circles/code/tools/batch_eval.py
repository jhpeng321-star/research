"""Batch evaluation over baselines × densities × CAV penetration rates.

Replaces the two near-duplicate scripts ``runs/eval/batch_eval.py`` and
``batch_eval_densities.py`` with one parameterized driver. Paths are resolved
relative to the package root, so it runs from any machine without editing.

Examples
--------
# Full grid (default, single seed 42):
python tools/batch_eval.py

# Just medium density, selfish + shepherd, at 10% and 20%:
python tools/batch_eval.py --baselines selfish shepherd --densities medium \\
    --penetrations 0.10 0.20

# Multi-seed sweep paired with the All-HDV baseline seeds:
python tools/batch_eval.py --seeds 42,43,44

Outputs (under runs/eval/):
  {baseline}_{density}_p{pen}.seed{N}.json   per-run full metrics
  summary_all.json                           rolling summary of all runs so far
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_policy import make_policy, run_policy_evaluation  # noqa: E402

OUT_DIR = ROOT / "runs" / "eval"
SUMO_ROOT = ROOT.parent / "highway_merge_shepherd"

# Each trained baseline lives under runs/rllib/<name>/checkpoints/best.
BASELINE_CHECKPOINTS = {
    "selfish": ROOT / "runs/rllib/selfish/checkpoints/best",
    "flat_marl": ROOT / "runs/rllib/flat_marl/checkpoints/best",
    "shepherd": ROOT / "runs/rllib/shepherd/checkpoints/best",
}

ALL_BASELINES = ["selfish", "flat_marl", "shepherd"]
ALL_DENSITIES = ["light", "medium", "dense"]
ALL_PENETRATIONS = [0.05, 0.10, 0.20]


def _tag(baseline: str, density: str, pen: float) -> str:
    return f"{baseline}_{density}_p{int(pen * 100):02d}"


def _summary_row(baseline, density, pen, metrics) -> dict:
    tt = metrics["travel_time"]
    ttc = metrics["ttc"]
    hard_braking = metrics.get("hard_braking") or {}
    return {
        "baseline": baseline,
        "density": density,
        "cav_penetration": pen,
        "seed": metrics["seed"],
        "steps": metrics["steps"],
        "total_reward": metrics["total_reward"],
        "mainline_mean": tt["mainline"]["mean"],
        "mainline_p90": tt["mainline"]["p90"],
        "ramp_mean": tt["ramp"]["mean"],
        "ramp_p90": tt["ramp"]["p90"],
        "all_mean": tt["all"]["mean"],
        "all_p90": tt["all"]["p90"],
        "ttc_events": ttc["total_events"],
        "arrived": metrics.get("arrived"),
        "throughput_veh_per_hour": metrics.get("throughput_veh_per_hour"),
        "mean_speed": metrics.get("mean_speed"),
        "mean_time_loss": metrics.get("mean_time_loss"),
        "hard_braking_events": hard_braking.get("total_events"),
        "collisions": metrics.get("collisions"),
        "teleports": metrics.get("teleports"),
        "elapsed_s": metrics["elapsed_s"],
        "ok": True,
        "error": None,
    }


def run_one(baseline, density, pen, seed, max_steps, ckpt) -> dict:
    # Per-seed output file so multi-seed sweeps (paired with the All-HDV
    # baseline seeds) do not overwrite each other.
    out_json = OUT_DIR / f"{_tag(baseline, density, pen)}.seed{seed}.json"
    t1 = time.time()
    try:
        policy = make_policy(baseline, checkpoint_path=str(ckpt))
        metrics = run_policy_evaluation(
            policy=policy,
            density=density,
            project_root=SUMO_ROOT.resolve(),
            sumo_binary="sumo",
            seed=seed,
            max_episode_steps=max_steps,
            hdv_mixture=None,
            cav_penetration=pen,
            policy_name=baseline,
        )
        metrics["baseline"] = baseline
        metrics["density"] = density
        metrics["cav_penetration"] = pen
        metrics["checkpoint"] = str(ckpt)
        metrics["elapsed_s"] = time.time() - t1
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)
        row = _summary_row(baseline, density, pen, metrics)
        print(
            f"OK {row['baseline']} {row['density']} p{int(pen * 100):02d}  "
            f"all_mean={row['all_mean']:.1f} main_p90={row['mainline_p90']:.1f} "
            f"ramp_p90={row['ramp_p90']:.1f} ttc={row['ttc_events']} "
            f"hb={row['hard_braking_events']} coll={row['collisions']} "
            f"({row['elapsed_s']:.0f}s)",
            flush=True,
        )
        return row
    except Exception as exc:  # noqa: BLE001 - report and continue to next run
        traceback.print_exc()
        row = {
            "baseline": baseline,
            "density": density,
            "cav_penetration": pen,
            "seed": seed,
            "ok": False,
            "error": str(exc),
            "elapsed_s": time.time() - t1,
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)
        print(f"FAIL {_tag(baseline, density, pen)}: {exc}", flush=True)
        return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baselines", nargs="+", default=ALL_BASELINES, choices=ALL_BASELINES)
    parser.add_argument("--densities", nargs="+", default=ALL_DENSITIES, choices=ALL_DENSITIES)
    parser.add_argument("--penetrations", type=float, nargs="+", default=ALL_PENETRATIONS)
    parser.add_argument(
        "--seeds",
        default="42",
        help="Comma-separated SUMO seeds, e.g. 42,43,44 (default: 42). Each "
        "seed is passed through to SUMO as --seed, pairing the runs with the "
        "All-HDV baseline seeds.",
    )
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Override checkpoint path (default: runs/rllib/<baseline>/checkpoints/best)",
    )
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "summary_all.json"

    # Keep any pre-existing successful results so the summary stays complete
    # across multiple partial runs.
    results: list[dict] = []
    if summary_path.exists():
        try:
            prev = json.loads(summary_path.read_text(encoding="utf-8"))
            for r in prev.get("results", []):
                if r.get("ok"):
                    results.append(r)
        except (OSError, json.JSONDecodeError):
            pass

    t0 = time.time()
    total = len(args.baselines) * len(args.densities) * len(args.penetrations) * len(seeds)
    done = 0
    for density in args.densities:
        for baseline in args.baselines:
            ckpt = Path(args.checkpoint) if args.checkpoint else BASELINE_CHECKPOINTS[baseline]
            for pen in args.penetrations:
                for seed in seeds:
                    done += 1
                    print(
                        f"\n=== [{done}/{total}] {_tag(baseline, density, pen)} seed={seed} ===",
                        flush=True,
                    )
                    results.append(
                        run_one(baseline, density, pen, seed, args.max_episode_steps, ckpt)
                    )
                    with open(summary_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {"results": results, "elapsed_s": time.time() - t0},
                            f,
                            indent=2,
                        )

    print(f"\nDONE in {time.time() - t0:.0f}s -> {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
