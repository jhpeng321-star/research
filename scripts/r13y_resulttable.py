"""Validate a finished R13Y run and print the compact arms/paired-CI CSV table."""
import argparse
import csv
import json
import sys

from diag_r13y_offline import ARMS, guard_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", help="finished diagnostic output directory")
    parser.add_argument("--check-yield-overlaps", action="store_true",
                        help="fail if any paired yield run adds HDV-HDV contact onsets")
    args = parser.parse_args()
    root = guard_path(args.results)
    summary = json.loads((root / "summary.json").read_text())
    rows = json.loads((root / "runs.json").read_text())
    assert len(rows) == 360
    assert len({(r["phase"], r["state"], r["sample"], r["arm"]) for r in rows}) == 360
    assert max(r["cav_max_error_m"] for r in rows) <= 1.e-3
    assert sum(r["phase"] == "short" for r in rows) == 288
    assert sum(r["phase"] == "spot" for r in rows) == 72

    if args.check_yield_overlaps:
        checks, failed = {}, False
        for phase in ("short", "spot"):
            indexed = {(r["state"], r["sample"], r["arm"]): r for r in rows if r["phase"] == phase}
            pairs = sorted({(state, sample) for state, sample, _ in indexed})
            added = [pair for pair in pairs
                     if indexed[(*pair, "yield")]["metrics"]["overlap_hdv_hdv_events"]
                     > indexed[(*pair, "off")]["metrics"]["overlap_hdv_hdv_events"]]
            checks[phase] = {"paired_runs": len(pairs), "runs_with_more_hdv_hdv_contacts": len(added),
                "paired_mean_delta_ci95": summary["phases"][phase]["comparisons"]
                    ["yield minus off"]["overlap_hdv_hdv_events"]["world_cluster"]}
            failed |= bool(added)
        print(json.dumps({"gate": "no added HDV-HDV contact onsets in any paired run",
                          "status": "failed" if failed else "passed", "phases": checks}, indent=2))
        return 1 if failed else 0

    def fmt(value):
        point = value.get("mean", value.get("ratio"))
        if point is None:
            return "missing"
        low, high = value["ci95"]
        return f'{point:.4f} [{low:.4f},{high:.4f}]'

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["phase", "metric", "recorded_SUMO_mean_CI95", "off_mean_CI95",
                     "yield_mean_CI95", "yield_emergency9_mean_CI95",
                     "yield_minus_off_paired_CI95", "yield_emergency9_minus_off_paired_CI95"])
    for phase in ("short", "spot"):
        report = summary["phases"][phase]
        window = "60_120s" if phase == "short" else "120_300s"
        metrics = [f"{name}_{window}" for name in (
            "zone_occupancy_veh", "zone_speed_mps", "standing_column_veh")]
        metrics += ["standing_head_m", "standing_episode_mean_s", "stuck_ramps",
                    "stuck_present_end", "ramp_censored_unmerged", "merges_per_run",
                    "delay_up_near_veh_s", "delay_merge_veh_s", "delay_ramp_veh_s",
                    "delay_downstream_veh_s", "ttc_min_s", "ttc_lt1p5_veh_s",
                    "hard_brake_events", "hard_brake_veh_s", "overlap_events", "overlap_pair_s",
                    "overlap_hdv_hdv_events", "overlap_hdv_hdv_pair_s"]
        for metric in metrics:
            values = [fmt(report["absolute"][arm][metric]["world_cluster"])
                      for arm in ("sumo", *ARMS)]
            for arm in ARMS[1:]:
                values.append(fmt(report["comparisons"][arm + " minus off"][metric]["world_cluster"]))
            writer.writerow([phase, metric, *values])
        for name in ("zone_occupancy_veh", "zone_speed_mps", "standing_column_veh"):
            metric = name + "_" + window
            values = [fmt(report["ratio_of_means_to_sumo"][arm][metric]["world_cluster"])
                      for arm in ARMS]
            writer.writerow([phase, metric + "_ratio_of_means_to_SUMO", "1", *values, "", ""])


if __name__ == "__main__":
    raise SystemExit(main())
