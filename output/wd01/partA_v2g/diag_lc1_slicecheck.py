"""Which (world, w) keys contribute the biggest pred merge counts in the
tables slice?  Compare with the eval-covered set."""
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(r"output/wd01/partA_v2g")
LC = HERE / "results" / "lc1"

ev = set()
for line in open(HERE / "results" / "eval_a2_post_lc1" / "raw_rows.jsonl",
                 encoding="utf-8"):
    r = json.loads(line)
    if r["model"] == "v2g" and r["H"] == 120:
        ev.add((r["world"], r["w"]))

cohorts = {}
for line in open(LC / "pred_states.jsonl", encoding="utf-8"):
    r = json.loads(line)
    cohorts[(r["world"], r["w"])] = set(r["vids"])

pe = defaultdict(list)
for line in open(LC / "pred_events.jsonl", encoding="utf-8"):
    e = json.loads(line)
    pe[e["world"]].append(e)

per_key = []
for (world, w), vids in cohorts.items():
    sel = [e for e in pe.get(world, [])
           if w < e["t"] <= w + 120 + 1e-9 and e["vid"] in vids]
    nm = sum(1 for e in sel if (e["f"], e["t_"]) == (2, 1))
    per_key.append((nm, world, w, (world, w) in ev, len(sel)))
per_key.sort(reverse=True)
print("top 15 keys by pred merge events in slice:")
for nm, world, w, cov, n in per_key[:15]:
    print(f"  merges={nm} all_events={n} {world} w={w} eval_covered={cov}")
tot = sum(x[0] for x in per_key)
tot_cov = sum(x[0] for x in per_key if x[3])
print(f"total={tot} over covered={tot_cov}  (n keys={len(per_key)}, "
      f"covered={sum(1 for x in per_key if x[3])})")
