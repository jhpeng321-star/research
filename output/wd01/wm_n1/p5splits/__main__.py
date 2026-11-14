"""CLI: materialise the WM-S P5 held-out splits from a window index.

    python -m p5splits --index <window_index.json|jsonl|parquet> \\
                       [--cal-split CAL_SPLIT_A4.json] [--out p5_splits.json]

Reads METADATA ONLY.  Never opens CAL/HOLDOUT/chunk-5/J4B/INC1 outcome files,
never starts SUMO.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import splits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", required=True,
                    help="window index (json/jsonl/parquet), metadata only")
    ap.add_argument("--cal-split", default=None,
                    help="CAL_SPLIT_A4.json (default: wm_n1/CAL_SPLIT_A4.json)")
    ap.add_argument("--out", default=None,
                    help="write the split artifact JSON here")
    ap.add_argument("--no-cal-tune", action="store_true",
                    help="do not restrict CAL evaluation to CAL-tune")
    args = ap.parse_args(argv)

    records = splits.load_index(args.index)
    cal_tune = None if args.no_cal_tune else \
        splits.load_cal_tune_worlds(args.cal_split)
    artifact = splits.materialize(records, cal_tune_worlds=cal_tune,
                                  source=str(Path(args.index).resolve()))
    if args.out:
        Path(args.out).write_text(json.dumps(artifact, indent=2),
                                  encoding="utf-8")
    for name, spec in artifact["splits"].items():
        v = spec["verification"]
        print(f"{name}: index={spec['n_index_windows']} "
              f"train={spec['n_train_windows']} "
              f"exclude={spec['n_train_exclude_windows']} "
              f"leak={v['leak_count']} "
              f"dev={spec['eval']['dev']['n_runs']} "
              f"reg={spec['eval']['registered']['n_runs']}")
    print("all_exclusions_absent_from_train="
          f"{artifact['all_exclusions_absent_from_train']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
