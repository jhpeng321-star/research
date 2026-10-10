"""Print the batch-eval summary as a compact table.

Usage:  python tools/print_summary.py [path/to/summary_all.json]
(default reads runs/eval/summary_all.json)
"""

import json
import sys
from pathlib import Path

DEFAULT = Path(__file__).resolve().parent.parent / "runs" / "eval" / "summary_all.json"


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    data = json.load(open(src, encoding="utf-8"))
    results = data.get("results", [])
    print("n", len(results))
    for r in results:
        if not r.get("ok"):
            print("FAIL", r.get("density"), r.get("baseline"), r.get("error"))
            continue
        print(
            f"{r['density']:6} {r['baseline']:10} "
            f"p{int(r['cav_penetration'] * 100):02d}  "
            f"mean={r['all_mean']:7.1f} "
            f"main90={r['mainline_p90']:6.1f} "
            f"ramp90={r['ramp_p90']:6.1f} "
            f"ttc={r['ttc_events']}"
        )


if __name__ == "__main__":
    main()
