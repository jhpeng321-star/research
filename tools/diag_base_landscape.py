"""S-BASEDIAG step 1: bias landscape of the round-8 base (pre-courtesy) on lane-1 HDV rows.

Reads the R11 offline cache (tmp_r11/cache.shard*.pkl: round-8 stage-300 base, teacher-forced
one-step replay, TRAIN windows only) and tabulates model - recorded acceleration by state.
Diagnosis only; no fitting.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

prefix = sys.argv[1] if len(sys.argv) > 1 else "tmp_r11/cache"
W = []
for i in range(8):
    p = Path(f"{prefix}.shard{i}.pkl")
    if p.is_file():
        W.extend(pickle.load(open(p, "rb"))["windows"])
keys = ("pre", "pre_clamped", "rec", "has", "c3", "da", "dend", "tmerge", "v", "hw", "closing", "s")
D = {k: np.concatenate([w[k] for w in W]) for k in keys}
bias = D["pre_clamped"] - D["rec"]
print("windows", len(W), "rows", len(bias), "mean bias all", bias.mean().round(3))


def tab(name, x, edges, mask=None):
    m = np.ones_like(bias, bool) if mask is None else mask
    print(f"\n{name}")
    print("bin | n | rec | model | bias | share rec<-0.5")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = m & (x >= lo) & (x < hi)
        if sel.sum() < 30:
            continue
        print(f"[{lo},{hi}) | {sel.sum()} | {D['rec'][sel].mean():+.2f} | "
              f"{D['pre_clamped'][sel].mean():+.2f} | {bias[sel].mean():+.2f} | "
              f"{(D['rec'][sel] < -0.5).mean():.2f}")


has = D["has"].astype(bool)
tm = D["tmerge"]
tab("partner rows by t - t_merge (s)", -tm, [-6, -4, -3, -2, -1, 0], has)
tab("all lane-1 rows by s (m)", D["s"], [-400, -200, -100, -50, 0, 25, 50, 75.44, 120, 200, 400, 2000])
tab("non-partner rows by s (m)", D["s"], [-400, -200, -100, -50, 0, 25, 50, 75.44, 120, 200, 400, 2000], ~has)
tab("non-partner rows by v (m/s)", D["v"], [0, 0.5, 1, 2, 3, 5, 8, 12, 20, 40], ~has)
tab("non-partner rows v<5 by hw (s)", D["hw"], [0, 0.5, 1, 1.5, 2, 3, 5, 10, 100], (~has) & (D["v"] < 5))
tab("non-partner rows v<5, s outside [-100,120] by hw", D["hw"], [0, 0.5, 1, 1.5, 2, 3, 5, 10, 100],
    (~has) & (D["v"] < 5) & ((D["s"] < -100) | (D["s"] > 120)))
tab("non-partner rows v<5, s in [0,75] by hw", D["hw"], [0, 0.5, 1, 1.5, 2, 3, 5, 10, 100],
    (~has) & (D["v"] < 5) & (D["s"] >= 0) & (D["s"] < 75.44))
last3 = has & (tm > 0) & (tm <= 3)
print("\npartner last-3s rows:", last3.sum(), "v p25/50/75", np.percentile(D["v"][last3], [25, 50, 75]).round(2),
      "hw p25/50/75", np.percentile(D["hw"][last3], [25, 50, 75]).round(2),
      "closing p50", np.median(D["closing"][last3]).round(2), "s p50", np.median(D["s"][last3]).round(1))
tab("partner last-3s rows by v", D["v"], [0, 0.5, 1, 2, 3, 5, 8, 12, 40], last3)
tab("partner last-3s rows by hw", D["hw"], [0, 0.5, 1, 1.5, 2, 3, 5, 10, 100], last3)
tab("partner last-3s rows by s", D["s"], [-100, -50, 0, 25, 50, 75.44, 200], last3)
tab("partner last-3s rows by pre-clamp model accel", D["pre"], [-10, -2, -1, 0, 0.5, 1, 1.5, 2, 2.5, 3.01, 10], last3)
print("share of last-3s partner rows where model is at the +clamp (pre >= 2.99):", (D["pre"][last3] >= 2.99).mean().round(3))
