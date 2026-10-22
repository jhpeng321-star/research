"""Markdown tables from the per-episode outputs of ``r9_support_reconcile`` (no replay).

Reads ``<output>/<source>/<episode>/{report,events}.json``.  Counts are HDV events in the
audit window (warm-up excluded).  ``supported`` = eligible under the shared route-frame
geometry; ``excluded`` = vetoed but explained (``merge_pivot_support``); ``vetoed`` =
unexplained.  ``pack`` = verdict of the old collapsed pack frame on the SAME moved states.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def load(output):
    for report_path in sorted(Path(output).glob("*/*/report.json")):
        yield (json.loads(report_path.read_text(encoding="utf-8")),
               json.loads((report_path.parent / "events.json").read_text(encoding="utf-8")))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    args = parser.parse_args(argv)
    per_source = defaultdict(Counter)
    causes, excluded, laws = Counter(), Counter(), defaultdict(Counter)
    zero_events = []
    episodes = 0
    for report, rows in load(args.output):
        episodes += 1
        src = report["source"]
        hdv = [r for r in rows if not r["is_cav"]]
        c = per_source[src]
        c["episodes"] += 1
        c["hdv_events"] += len(hdv)
        c["cav_merges"] += len(rows) - len(hdv)
        for r in hdv:
            c["supported"] += r["support_verdict"] == "supported"
            c["excluded"] += r["support_verdict"] == "excluded_counted"
            c["vetoed"] += r["support_verdict"] == "vetoed"
            if "pack_frame_eligible" in r:
                c["pack_eligible"] += bool(r["pack_frame_eligible"])
                c["pack_known"] += 1
            for kind in ("F1", "M0", "P1-P3"):
                state = r.get(kind + "_mass")
                if state is not None:
                    laws[src][(kind, state)] += 1
                    if state != "positive" and kind in ("F1", "M0"):
                        zero_events.append((kind, state, r["event_id"], r["support_verdict"],
                                            json.loads(r["geometry_json"]).get("zero_mass_reason"),
                                            r.get("absorbed_before_event")))
        causes.update(report["causes"])
        excluded.update(report.get("excluded_causes", {}))
        for r in hdv:
            if "F1_mass" not in r:  # counted-exclusion sequences are dropped from the fitter
                laws[src][("F1", "dropped_counted_exclusion")] += 1
    print(f"episodes: {episodes}\n")
    print("| source | episodes | HDV events | supported | counted exclusions (events) | unexcluded vetoes | "
          "supported % (of all) | pack-frame supported |")
    print("|---|---|---|---|---|---|---|---|")
    total = Counter()
    for src, c in sorted(per_source.items()):
        total.update(c)
        print(f"| {src} | {c['episodes']} | {c['hdv_events']} | {c['supported']} | {c['excluded']} | "
              f"{c['vetoed']} | {100 * c['supported'] / max(c['hdv_events'], 1):.2f} | "
              f"{c['pack_eligible']}/{c['pack_known']} |")
    c = total
    print(f"| ALL | {c['episodes']} | {c['hdv_events']} | {c['supported']} | {c['excluded']} | "
          f"{c['vetoed']} | {100 * c['supported'] / max(c['hdv_events'], 1):.2f} | "
          f"{c['pack_eligible']}/{c['pack_known']} |\n")
    print("unexplained veto causes:", dict(causes))
    print("explained exclusion causes:", dict(excluded), "\n")
    print("| source | law | positive | zero | other |")
    print("|---|---|---|---|---|")
    for src, counts in sorted(laws.items()):
        for kind in ("F1", "M0", "P1-P3"):
            pos = counts[(kind, "positive")]
            zero = counts[(kind, "zero")]
            other = sum(v for (k, s), v in counts.items() if k == kind and s not in ("positive", "zero"))
            print(f"| {src} | {kind} | {pos} | {zero} | {other} |")
    print("\nF1/M0 non-positive events:")
    for item in zero_events:
        print(" ", item)


if __name__ == "__main__":
    main()
