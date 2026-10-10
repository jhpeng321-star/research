"""Pretty-print the T3 label-coverage probe JSON (TRAIN only)."""
from __future__ import annotations

import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else (
    "E:/research-dense/tmp/overnight_20261004/T3_out/"
    "label_coverage_probe.json")
d = json.load(open(path, encoding="utf-8-sig"))
print("zero_rows", d["n_zero_rows"], "has_rows", d["n_has_rows"],
      "excluded_b", d["n_excluded_b"])
tot = 0
hdr = (f"{'batch':>20} {'chunk':>13} {'arm':>12} {'n':>4} {'zero':>5} "
       f"{'src':>10} {'v04a':>5} {'shardOut':>8} {'rawP':>5} {'rawU':>5} "
       f"{'GB':>6}")
print(hdr)
for g in sorted(d["groups"], key=lambda x: (x["batch"], x["arm"])):
    if g["manifest_zero_rows"] <= 0:
        continue
    tot += g["raw_physics_bytes"]
    print(f"{g['batch']:>20} {g['chunk']:>13} {g['arm']:>12} "
          f"{g['n_items']:>4} {g['manifest_zero_rows']:>5} "
          f"{','.join(str(s) for s in g['resolved_sources']):>10} "
          f"{g['v04a_label_file_present']:>5} "
          f"{g['shard_outcomes_present']:>8} "
          f"{g['raw_run_present']:>5} {g['raw_run_unknown']:>5} "
          f"{g['raw_physics_bytes'] / 1e9:>6.2f}")
print("TOTAL zero-row raw physics GB:", round(tot / 1e9, 2))
zr = sum(g["manifest_zero_rows"] for g in d["groups"])
print("zero-row items:", zr)
