"""Check the urgency_late ablation actually changed merge behaviour vs
nominal (per-window merge counts identical would mean the override was a
no-op)."""
import json
from collections import defaultdict
from pathlib import Path

LC = Path(r"output/wd01/partA_v2g/results/lc1")
rows = defaultdict(dict)
n_no_h120 = 0
for line in open(LC / "ablation_rows.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if "H120" not in r["lc"]:
        n_no_h120 += 1
        continue
    rows[r["variant"]][(r["world"], r["w"])] = r["lc"]["H120"]
print("rows without H120 (truncated windows):", n_no_h120)

nom = rows["nominal"]
late = rows["urgency_late"]
same = diff = 0
for k in nom:
    if nom[k]["merge"] == late[k]["merge"]:
        same += 1
    else:
        diff += 1
print(f"merge counts nominal vs urgency_late: same={same} diff={diff}")
tot_n = sum(v["merge"] for v in nom.values())
tot_l = sum(v["merge"] for v in late.values())
print("total merges:", tot_n, "vs", tot_l)
# also events files
ev = defaultdict(lambda: [0, 0])
for line in open(LC / "ablation_events.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["key"] == "merge":
        ev[r["variant"]][0] += 1
    else:
        ev[r["variant"]][1] += 1
for k in sorted(ev):
    print(k, "merge events:", ev[k][0], "disc events:", ev[k][1])
