"""Analyse C4-LANECAL request data: old vs new LaneRules precision/recall (held-out seeds)."""
import glob, json, sys, itertools
import numpy as np

HELD = {4111, 4112}

def load(d):
    rows = []
    for f in sorted(glob.glob(d + "/*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            r["seed"] = int(f.replace("\\", "/").split("/s")[-1].split("_")[0])
            rows.append(r)
    return rows

def arr(rows):
    big = 1e9
    g = lambda k, dv=big: np.array([dv if r[k] is None else r[k] for r in rows], float)
    return dict(v=g("ego_v"), ff=g("tgt_front_gap"), fv=g("tgt_front_v", 0.0), rg=g("tgt_rear_gap"),
                rv=g("tgt_rear_v", 0.0), acc=np.array([r["accepted"] for r in rows]),
                left=np.array([r["dir"] == "left" for r in rows]),
                ramp=np.array([r["ramp"] for r in rows]),
                seed=np.array([r["seed"] for r in rows]),
                rear_tau=g("rear_tau", np.nan), rear_decel=g("rear_decel", np.nan))

def pred(a, base, gt, ct, base_r=None, gtr=None, ctr=None, rear_lat=0.0):
    base_r = base if base_r is None else base_r
    gtr = gt if gtr is None else gtr
    ctr = ct if ctr is None else ctr
    nf = base + gt * a["v"] + ct * np.maximum(0, a["v"] - a["fv"])
    nr = base_r + gtr * a["rv"] + ctr * np.maximum(0, a["rv"] - a["v"])
    return (a["ff"] >= nf) & (a["rg"] >= nr)

def pr(p, y):
    tp = (p & y).sum(); fp = (p & ~y).sum(); fn = (~p & y).sum()
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    return prec, rec, int(p.sum()), int(y.sum())

if __name__ == "__main__":
    rows = load(sys.argv[1])
    a = arr(rows)
    print(len(rows), "requests, accept rate", a["acc"].mean())
