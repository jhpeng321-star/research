"""Where do the extra H120 pred merges come from? Sum by window index k
and by H, restricted vs not to the eval's covered windows."""
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(r"output/wd01/partA_v2g")
LC = HERE / "results" / "lc1"

ev_h = defaultdict(set)
for line in open(HERE / "results" / "eval_a2_post_lc1" / "raw_rows.jsonl",
                 encoding="utf-8"):
    r = json.loads(line)
    if r["model"] == "v2g":
        ev_h[r["H"]].add((r["world"], r["w"]))

by_k = defaultdict(lambda: [0, 0, 0])   # [n, sum, hist]
for line in open(LC / "pred_window_rows.jsonl", encoding="utf-8"):
    r = json.loads(line)
    for H in (30, 120):
        covered = (r["world"], r["w"]) in ev_h[H]
        st = by_k[(H, covered)]
        st[0] += 1
        m = r["lc"][f"H{H}"]["merge"]
        st[1] += m
        st[2] = max(st[2], m)
for key in sorted(by_k, key=str):
    n, s, mx = by_k[key]
    print(f"H={key[0]} covered={key[1]}: n={n} sum_merge={s} "
          f"mean={s / max(n, 1):.2f} max={mx}")
