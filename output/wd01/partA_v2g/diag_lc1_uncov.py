"""Inspect pred_window_rows H120 entries NOT covered by the eval."""
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

rows = 0
for line in open(LC / "pred_window_rows.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if "H120" not in r["lc"]:
        continue
    if (r["world"], r["w"]) in ev:
        continue
    rows += 1
    if rows <= 10:
        print(r["world"], r["w"], "n_tracked", r["n_tracked"],
              "merge", r["lc"]["H120"]["merge"],
              "disc", r["lc"]["H120"]["disc_left"] +
              r["lc"]["H120"]["disc_right"])
print("uncovered H120 rows:", rows)
