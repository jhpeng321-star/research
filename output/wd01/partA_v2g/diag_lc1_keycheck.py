"""Per-key consistency check: lc1 pred_window_rows / truth slices vs the
frozen eval_a2_post_lc1 lc_pred/lc_true rows."""
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(r"output/wd01/partA_v2g")
LC = HERE / "results" / "lc1"

# eval rows (post-LC1-fixes)
ev = {}
for line in open(HERE / "results" / "eval_a2_post_lc1" / "raw_rows.jsonl",
                 encoding="utf-8"):
    r = json.loads(line)
    if r["model"] == "v2g":
        ev[(r["world"], r["w"], r["H"])] = r["lc_pred"], r["lc_true"]

# lc1 pred window rows
mis_p = same_p = 0
ex = []
for line in open(LC / "pred_window_rows.jsonl", encoding="utf-8"):
    r = json.loads(line)
    for H in (10, 30, 60, 120):
        key = (r["world"], r["w"], H)
        if key not in ev:
            continue
        mine = r["lc"][f"H{H}"]
        ref = ev[key][0]
        if mine == ref:
            same_p += 1
        else:
            mis_p += 1
            if len(ex) < 5:
                ex.append((key, mine, ref))
print(f"pred: same={same_p} mismatch={mis_p}")
for e in ex:
    print("  ", e)

# lc1 truth windows (dest-tracked = eval convention; physical = fixed)
mis_t = same_t = 0
ex = []
for line in open(LC / "truth_windows.jsonl", encoding="utf-8"):
    r = json.loads(line)
    key = (r["world"], r["w"], r["H"])
    if key not in ev:
        continue
    ref = ev[key][1]
    mine = {"merge": r["merge"], "disc_left": r["disc_left"],
            "disc_right": r["disc_right"]}
    if mine == ref:
        same_t += 1
    else:
        mis_t += 1
        if len(ex) < 8:
            ex.append((key, mine, ref))
print(f"truth (physical-row tracked): same={same_t} mismatch={mis_t}")
for e in ex:
    print("  ", e)
