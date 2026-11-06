"""Inspect unpaired event labels using the guarded observation whitelist."""
import json
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import merge_fit_v2 as legacy  # noqa: E402
from tools import r9_exact as exact  # noqa: E402
from tools import r9_runtime as runtime  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", action="store_true")
    parser.add_argument("--table", action="store_true")
    args = parser.parse_args()
    episodes, _ = legacy.discover(["v5_native=data/r9_train/v5_native",
                                    "genrand_plain=data/r9_train/genrand_plain"])
    for ep in episodes:
        if args.shapes:
            loaded = runtime.load_scene(ep["path"])
            print(ep["episode"], loaded["kin"].shape,
                  "max_present", loaded["present"].sum(-1).max())
            continue
        report_path = exact.guard(ROOT / "results/r9_support" / ep["source_name"]
                                  / ep["episode"] / "report.json")
        if not report_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if args.table:
            rows = json.loads((report_path.parent / "events.json").read_text(encoding="utf-8"))
            for row in rows:
                if row["is_cav"] or row["support_verdict"] != "vetoed":
                    continue
                d = json.loads(row["geometry_json"])
                ego = d["ego"]
                def overlaps(recorded):
                    targets = (d["diagnostic_recorded_motion_pre_lanes_intersections"] if recorded
                               else d["all_pre_target_intersections"])
                    text = []
                    for v in targets:
                        overlap = (v["recorded_overlap_m"] if recorded else
                                   min(ego["front"], v["front"]) - max(ego["rear"], v["rear"]))
                        text.append(f"{v['vehicle_id']}{'C' if v['is_cav'] else 'H'}:{overlap:.6f}")
                    return ",".join(text) or "none"
                tag = "N" if ep["source_name"] == "v5_native" else "G"
                cause = {"junction_projection_unresolved": "J", "outgoing_target_substep_unverified": "O",
                         "wm_motion_only_intersection": "M", "wm_motion_outside_decision_zone": "Z",
                         "recorded_projected_intersection_unresolved": "R"}[d["veto_cause"]]
                print(f"| {tag}{ep['seed']}/{row['vehicle_id']}/{row['event_time_s']:.1f} "
                      f"| {ego['front']:.6f}; {overlaps(False)} "
                      f"| {ego['recorded_post_front']:.6f}; {overlaps(True)} | {cause} |")
            continue
        if not report["mismatched_events"]:
            continue
        loaded = runtime.load_scene(ep["path"])
        for event in report["mismatched_events"]:
            vehicle, row = map(int, event.rsplit("/", 2)[-2:])
            slots = ((loaded["identity"][row - 1] == vehicle) & loaded["present"][row - 1]).nonzero()[0]
            slot = int(slots[0])
            print(json.dumps(dict(event=event, slot=slot,
                 raw=loaded["raw"][row-2:row+2, slot].tolist(),
                 cav=loaded["is_cav"][row-2:row+2, slot].tolist(),
                 ids=loaded["identity"][row-2:row+2, slot].tolist(),
                 excluded=[e for e in report["cohort"]["exclusions"] if e["vehicle"] == vehicle])))


if __name__ == "__main__":
    main()
