"""Pure-SUMO All-HDV baseline evaluation (no RL), multi-seed.

Runs the All-HDV scenario once per seed (``--seeds 42,43,44``, each passed
through to SUMO as ``--seed``) and reports per-seed metrics plus mean±std.
The driver-type mixture comes from the route file's embedded
vTypeDistribution (no assignment code here); the observed vType counts are
read back from the tripinfo output and checked against the expected
10/70/20 mixture.
"""

import argparse
import json
import statistics
from pathlib import Path

from evaluation import evaluate_run, run_sumo_evaluation


# Shared SUMO project (parallel layout: D:\research-dense next to D:\research).
DEFAULT_SUMO_ROOT = Path(r"D:\research\highway_merge_shepherd")

# Driver mixture declared by the route file's vTypeDistribution (osm.demand.*.rou.xml).
EXPECTED_TYPE_MIX = {
    "aggressive_car": 0.10,
    "normal_car": 0.70,
    "cautious_car": 0.20,
}
TYPE_MIX_TOLERANCE = 0.05

# Metrics aggregated across seeds as {mean, std, values}.
_AGGREGATE_KEYS = [
    ("all_tt_mean", lambda m: m["travel_time"]["all"]["mean"]),
    ("all_tt_p90", lambda m: m["travel_time"]["all"]["p90"]),
    ("mainline_tt_mean", lambda m: m["travel_time"]["mainline"]["mean"]),
    ("mainline_tt_p90", lambda m: m["travel_time"]["mainline"]["p90"]),
    ("ramp_tt_mean", lambda m: m["travel_time"]["ramp"]["mean"]),
    ("ramp_tt_p90", lambda m: m["travel_time"]["ramp"]["p90"]),
    ("throughput_veh_per_hour", lambda m: m["throughput_veh_per_hour"]),
    ("mean_speed", lambda m: m["mean_speed"]),
    ("mean_time_loss", lambda m: m["mean_time_loss"]),
    ("ttc_events", lambda m: m["ttc"]["total_events"]),
    ("hard_braking_events", lambda m: m["hard_braking"]["total_events"]),
    ("collisions", lambda m: m["collisions"]),
    ("teleports", lambda m: m["teleports"]),
    ("arrived", lambda m: m["arrived"]),
]


def aggregate_seed_metrics(runs: list) -> dict:
    """Aggregate per-seed metric dicts into {mean, std, values} per key.

    The std is the sample std (ddof=1) across seeds; a single seed yields 0.0.
    """
    aggregate = {}
    for name, extract in _AGGREGATE_KEYS:
        values = [extract(m) for m in runs]
        values = [v for v in values if v is not None and v == v]  # drop None/NaN
        if not values:
            aggregate[name] = {"mean": None, "std": None, "values": []}
            continue
        aggregate[name] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    return aggregate


def check_type_mixture(
    fractions: dict, expected: dict = None, tolerance: float = TYPE_MIX_TOLERANCE
) -> dict:
    """Check observed vType fractions against the expected 10/70/20 mixture."""
    expected = expected or EXPECTED_TYPE_MIX
    deviations = {vtype: fractions.get(vtype, 0.0) - target for vtype, target in expected.items()}
    ok = all(abs(dev) <= tolerance for dev in deviations.values())
    return {
        "ok": ok,
        "expected": expected,
        "tolerance": tolerance,
        "deviations": deviations,
    }


def aggregate_type_counts(runs: list) -> dict:
    """Sum per-seed vType counts and compute overall fractions."""
    totals = {}
    for m in runs:
        for vtype, count in m["type_counts"].items():
            totals[vtype] = totals.get(vtype, 0) + count
    total = sum(totals.values())
    fractions = {vtype: count / total for vtype, count in totals.items()} if total else {}
    return {"counts": totals, "fractions": fractions, "total": total}


def run_seed(
    density: str,
    seed: int,
    sumo_root: Path,
    sumo_binary: str,
    re_run: bool,
) -> dict:
    """Run (or reuse) one seeded pure-SUMO evaluation; return its metrics."""
    sumo_config = sumo_root / f"osm.{density}.sumocfg"
    tripinfo_output = sumo_root / f"tripinfos.{density}.seed{seed}.xml"
    fcd_output = sumo_root / f"fcd.{density}.seed{seed}.xml"
    stats_output = sumo_root / f"stats.{density}.seed{seed}.xml"
    collision_output = sumo_root / f"collisions.{density}.seed{seed}.xml"

    if (
        re_run
        or not tripinfo_output.exists()
        or not fcd_output.exists()
        or not stats_output.exists()
    ):
        print(f"Running SUMO (seed={seed}) with config: {sumo_config}")
        run_sumo_evaluation(
            sumo_binary=sumo_binary,
            sumo_config=sumo_config,
            project_root=sumo_root,
            tripinfo_output=tripinfo_output,
            fcd_output=fcd_output,
            seed=seed,
            stats_output=stats_output,
            collision_output=collision_output,
        )
    else:
        print(f"Using existing outputs for seed {seed}.")

    metrics = evaluate_run(tripinfo_output, fcd_output, stats_path=stats_output)
    metrics["seed"] = seed
    metrics["tripinfo_path"] = str(tripinfo_output)
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a pure-SUMO All-HDV scenario over one or more seeds."
    )
    parser.add_argument(
        "density",
        choices=["light", "medium", "dense", "hyper"],
        help="Traffic density preset to evaluate.",
    )
    parser.add_argument(
        "--seeds",
        default="42",
        help="Comma-separated SUMO seeds, e.g. 42,43,44 (default: 42).",
    )
    parser.add_argument(
        "--sumo-root",
        type=Path,
        default=DEFAULT_SUMO_ROOT,
        help="Path to the SUMO project directory (default: sibling highway_merge_shepherd).",
    )
    parser.add_argument(
        "--sumo-binary",
        default="sumo",
        help="SUMO binary to use (default: sumo).",
    )
    parser.add_argument(
        "--re-run",
        action="store_true",
        help="Re-run SUMO even if output files already exist.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file to save the full multi-seed result.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full metrics as JSON.",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    sumo_root = args.sumo_root.resolve()

    runs = [
        run_seed(args.density, seed, sumo_root, args.sumo_binary, args.re_run) for seed in seeds
    ]
    aggregate = aggregate_seed_metrics(runs)
    type_dist = aggregate_type_counts(runs)
    type_check = check_type_mixture(type_dist["fractions"])

    result = {
        "density": args.density,
        "seeds": seeds,
        "sumo_config": str(sumo_root / f"osm.{args.density}.sumocfg"),
        "runs": runs,
        "aggregate": aggregate,
        "type_distribution": {**type_dist, "check": type_check},
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Saved: {args.output}")

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(
            f"\nAll-HDV results for density '{args.density}' "
            f"({len(seeds)} seed(s): {', '.join(map(str, seeds))}):"
        )
        print("-" * 78)
        for m in runs:
            tt = m["travel_time"]["all"]
            print(
                f"seed {m['seed']:>4}: all_mean={tt['mean']:.2f}s "
                f"p90={tt['p90']:.2f}s arrived={m['arrived']} "
                f"thr={m['throughput_veh_per_hour']:.0f} veh/h "
                f"timeLoss={m['mean_time_loss']:.2f}s "
                f"ttc={m['ttc']['total_events']} "
                f"hb={m['hard_braking']['total_events']} "
                f"coll={m['collisions']} tele={m['teleports']}"
            )
        print("-" * 78)
        for name in [
            "all_tt_mean",
            "mainline_tt_mean",
            "ramp_tt_mean",
            "all_tt_p90",
            "throughput_veh_per_hour",
            "mean_speed",
            "mean_time_loss",
            "ttc_events",
            "hard_braking_events",
            "collisions",
            "teleports",
        ]:
            entry = aggregate[name]
            if entry["mean"] is None:
                print(f"{name:24s}: n/a")
            else:
                print(f"{name:24s}: {entry['mean']:.3f} ± {entry['std']:.3f}")
        print("-" * 78)
        print("Driver-type mixture (from tripinfo vType):")
        for vtype in sorted(type_dist["counts"]):
            count = type_dist["counts"][vtype]
            frac = type_dist["fractions"].get(vtype, 0.0)
            expected = type_check["expected"].get(vtype)
            exp_txt = (
                f"expected {expected:.0%} ±{type_check['tolerance']:.0%}"
                if expected is not None
                else "unexpected type"
            )
            print(f"  {vtype:16s}: {count:5d} ({frac:6.1%}) {exp_txt}")
        print(f"  mixture check: {'OK' if type_check['ok'] else 'MISMATCH'}")


if __name__ == "__main__":
    main()
