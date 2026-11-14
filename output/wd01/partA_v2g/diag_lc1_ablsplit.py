"""Per-variant disc left/right split of the LC1 ablation events."""
import json
from collections import defaultdict

ev = defaultdict(lambda: defaultdict(int))
for line in open(r"output/wd01/partA_v2g/results/lc1/ablation_events.jsonl",
                 encoding="utf-8"):
    r = json.loads(line)
    ev[r["variant"]][r["key"]] += 1
for k in sorted(ev):
    d = ev[k]
    print(f"{k:20s} disc_left={d.get('disc_left', 0):6d} "
          f"disc_right={d.get('disc_right', 0):6d} "
          f"merge={d.get('merge', 0):5d}")
print("truth reference (window slices): disc_left=1012 disc_right=510 "
      "merge=2406")
