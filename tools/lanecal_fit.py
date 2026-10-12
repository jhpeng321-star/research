"""Fit LaneRules (front/rear separate) to the C4-LANECAL data: max recall s.t. train precision >= target."""
import sys, json
sys.path.insert(0, "tools")
from lanecal_analyze import *

def prec_rec(p, y):
    tp = (p & y).sum(); fp = (p & ~y).sum(); fn = (~p & y).sum()
    return tp / max(1, tp + fp), tp / max(1, tp + fn)

def predq(a, th, quad):
    bf, gtf, ctf, br, gtr, ctr = th
    if quad:
        nf = bf + gtf * a["v"] + np.maximum(0, a["v"] ** 2 - a["fv"] ** 2) / (2 * max(ctf, 1e-3))
        nr = br + gtr * a["rv"] + np.maximum(0, a["rv"] ** 2 - a["v"] ** 2) / (2 * max(ctr, 1e-3))
    else:
        nf = bf + gtf * a["v"] + ctf * np.maximum(0, a["v"] - a["fv"])
        nr = br + gtr * a["rv"] + ctr * np.maximum(0, a["rv"] - a["v"])
    return (a["ff"] >= nf) & (a["rg"] >= nr)

def fit(A, quad, target, seed=0, iters=6000):
    y = A["acc"]
    rng = np.random.default_rng(seed)
    lo = np.array([0, 0, 0.5 if quad else 0, 0, 0, 0.5 if quad else 0.0])
    hi = np.array([20, 3, 12 if quad else 10, 20, 4, 12 if quad else 12])
    def score(th):
        pre, rec = prec_rec(predq(A, th, quad), y)
        return (rec if pre >= target else rec - 5 * (target - pre)), pre, rec
    best = None
    for _ in range(60):
        th = rng.uniform(lo, hi); sc = score(th)
        if best is None or sc[0] > best[0][0]:
            best = (sc, th)
    cur = best
    step = (hi - lo) * 0.15
    for i in range(iters):
        th = np.clip(cur[1] + rng.normal(0, 1, 6) * step * (1 - i / iters) * (rng.random(6) < 0.4), lo, hi)
        sc = score(th)
        if sc[0] >= cur[0][0]:
            cur = (sc, th)
    return cur[1], cur[0]

if __name__ == "__main__":
    rows = load("runs/active/C4-LANECAL/data"); a = arr(rows)
    lane = np.array([r["lane"] for r in rows]); m0 = lane < 2
    tr = ~np.isin(a["seed"], list(HELD))
    sub = lambda m: {k: v[m] for k, v in a.items()}
    A, B = sub(tr & m0), sub(~tr & m0)
    for quad in (False, True):
        for target in (0.9, 0.92, 0.95):
            th, sc = fit(A, quad, target)
            pb = predq(B, th, quad)
            print("quad", quad, "target", target, "theta", np.round(th, 2), "train p/r", np.round(sc[1:], 3),
                  "held p/r", np.round(prec_rec(pb, B["acc"]), 3), "npred", int(pb.sum()))
