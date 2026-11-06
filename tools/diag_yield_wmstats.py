"""S-YIELD Q4: follower response to a waiting ramp car in WM rollouts vs the recorded SUMO future of the same
states, with the SAME definitions as the SUMO analysis (tools/diag_yield_extract / diag_yield_events).
Expected acceleration = SUMO-train lane-1 control-cell means (no ramp-lane car within [s-30, s+150]); a second
set uses the rollout's own lane-1 controls."""
import pickle, sys, collections
sys.path.insert(0, "tools")
import numpy as np
from diag_yield_common import *
from diag_yield_extract import rows_from_arrays
from diag_yield_events import events_from_arrays

PKL = sys.argv[1] if len(sys.argv) > 1 else OUT + "/wm48.pkl"
d = load_samples()
cid = cells(d); hdv = d["cav"] == 0; l1 = d["lane"] == 1
ctrl1 = l1 & hdv & (d["nRamp"] == 0)
u1, inv1 = np.unique(cid, return_inverse=True)
n1 = np.bincount(inv1[ctrl1], minlength=len(u1)); mu1 = np.bincount(inv1[ctrl1], d["a"][ctrl1], minlength=len(u1)) / np.maximum(n1, 1)
mu1 = np.where(n1 >= 20, mu1, np.nan)
z = pickle.load(open(PKL, "rb"))
res = [r for r in z["results"] if not r.get("error")]
PRE, POST = 40, 24
OF = np.arange(-PRE, POST + 1)


def arrays(r, which):
    if which == "wm":
        pos, lane, mask, v = r["pos"], r["lane"], r["mask"], r["v"]
    else:
        pos, lane, mask, v = r["rec_pos"], r["rec_lane"], r["rec_mask"], r["rec_v"]
    lane = np.where((lane >= 0) & (lane <= 2), np.rint(lane), -1).astype(int)
    a = np.zeros_like(v, dtype=np.float64); a[1:] = (v[1:] - v[:-1]) / 0.5
    T, N = pos.shape
    vid = np.broadcast_to(np.arange(N)[None], (T, N)).copy()
    cav = np.broadcast_to(np.asarray(r["is_cav"])[None], (T, N)).copy()
    tt = r["t0"] + 0.5 * np.arange(T)
    return pos.astype(np.float64), v.astype(np.float64), a, lane, mask.astype(bool), vid, cav, tt


def tables(variant, which):
    rows, evs, ems = [], [], []
    seen = set()
    for r in res:
        if r["var"] != variant:
            continue
        if which == "rec":
            if "rec_pos" not in r:
                continue
        s, v, a, lane, m, vid, cav, tt = arrays(r, which)
        dem = float(r.get("dem") or np.nan); pen = float(r.get("pen") or np.nan)
        tab = rows_from_arrays(s, v, a, lane, m, vid, cav, tt, r["idx"], pen, dem, tt[0] + 0.5)
        if len(tab):
            rows.append(tab)
        ev, me = events_from_arrays(s, v, a, lane, m, vid, cav, tt, r["idx"], pen, dem, tt[0], PRE, POST)
        if len(ev):
            evs.append(ev); ems.append(me)
    T = np.concatenate(rows)
    return {c: T[:, i] for i, c in enumerate(C)}, (np.concatenate(evs) if evs else np.zeros((0, len(OF), 13))), (np.concatenate(ems) if ems else np.zeros((0, 8)))


def excess(t, own=None):
    c = cells(t)
    if own is None:
        exp = lookup(u1, mu1, c)
    else:
        exp = lookup(*own, c)
    return t["a"] - exp


def own_model(t):
    c = cells(t); h = (t["cav"] == 0) & (t["lane"] == 1) & (t["nRamp"] == 0)
    u, inv = np.unique(c, return_inverse=True)
    n = np.bincount(inv[h], minlength=len(u)); mu = np.bincount(inv[h], t["a"][h], minlength=len(u)) / np.maximum(n, 1)
    return u, np.where(n >= 10, mu, np.nan)


def own3_model(t):
    c = cells(t)
    h = (t["cav"] == 0) & (t["lane"] == 1) & (t["nzone"] > 0) & (t["tf"] == 0) & ~(t["dA"] < 30) & ~(t["dB"] < 20)
    u, inv = np.unique(c, return_inverse=True)
    n = np.bincount(inv[h], minlength=len(u)); mu = np.bincount(inv[h], t["a"][h], minlength=len(u)) / np.maximum(n, 1)
    return u, np.where(n >= 10, mu, np.nan)


def boot(vals, groups, n=300, seed=0):
    """cluster bootstrap (by state) of the mean of finite vals"""
    rng = np.random.default_rng(seed); g = np.asarray(groups); ok = np.isfinite(vals); vals = vals[ok]; g = g[ok]
    ug = np.unique(g)
    sums = {k: (vals[g == k].sum(), (g == k).sum()) for k in ug}
    out = []
    for _ in range(n):
        pick = rng.choice(ug, len(ug)); s = sum(sums[k][0] for k in pick); c = sum(sums[k][1] for k in pick)
        out.append(s / max(c, 1))
    return np.percentile(out, [2.5, 97.5])


def st(x):
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return "n=%d" % len(x)
    return "n=%6d mean %+.3f med %+.3f p10 %+.3f" % (len(x), x.mean(), np.median(x), np.percentile(x, 10))


res_c3 = {}
out = []
for variant, which in [("r9", "rec"), ("r9", "wm"), ("r8", "wm")]:
    t, ev, me = tables(variant, which)
    label = "SUMO recorded (t0 roster)" if which == "rec" else "WM " + variant
    ex = excess(t); exo = excess(t, own_model(t))
    hdvm = (t["cav"] == 0) & (t["lane"] == 1)
    print("=====", label, "rows", len(t["a"]), "events", len(ev))
    for lo, hi in [(0, 6), (6, 15), (15, 30), (30, 60), (60, 150)]:
        sel = hdvm & (t["tf"] == 1) & (t["dA"] >= lo) & (t["dA"] < hi)
        print("  tf dA %3d-%3d  SUMO-ctrl %s | own-ctrl %s | raw a mean %+.3f" % (lo, hi, st(ex[sel]), st(exo[sel]), np.nanmean(t["a"][sel]) if sel.sum() else np.nan))
    for lo, hi in [(0, 10), (10, 20), (20, 40), (40, 80)]:
        sel = hdvm & (t["tf"] == 1) & (t["dA"] < 20) & (t["dendR"] >= lo) & (t["dendR"] < hi)
        print("  tf dA<20, dend %2d-%2d  SUMO-ctrl %s | own-ctrl %s" % (lo, hi, st(ex[sel]), st(exo[sel])))
    ex3 = excess(t, own3_model(t))
    sel = hdvm & (t["tf"] == 1) & (t["dA"] >= 6) & (t["dA"] < 15)
    ci = boot(ex3[sel], t["ep"][sel])
    print("  [C3-own: lane-1 rows with a waiting ramp car in zone but not adjacent] tf dA 6-15: %s  mean 95%% CI (state bootstrap) [%+.3f, %+.3f]" % (st(ex3[sel]), *ci))
    for lo, hi in [(0, 6), (15, 30)]:
        s2 = hdvm & (t["tf"] == 1) & (t["dA"] >= lo) & (t["dA"] < hi)
        print("  [C3-own] tf dA %d-%d: %s" % (lo, hi, st(ex3[s2])))
    for lo, hi in [(0, 10), (10, 20), (20, 40), (40, 80)]:
        s2 = hdvm & (t["tf"] == 1) & (t["dA"] < 20) & (t["dendR"] >= lo) & (t["dendR"] < hi)
        print("  [C3-own] tf dA<20, dend %2d-%2d: %s" % (lo, hi, st(ex3[s2])))
    res_c3[(variant, which)] = (t, ex3, own3_model(t))
    sel0 = hdvm & (t["nRamp"] == 0)
    print("  control rows (lane1, no ramp near): a mean %+.3f p10 %+.3f  n=%d" % (np.nanmean(t["a"][sel0]), np.nanpercentile(t["a"][sel0], 10), sel0.sum()))
    if len(ev):
        cide = event_cells(ev, np.nan_to_num(me[:, 7], nan=2400.0)); exe = ev[..., 2] - lookup(u1, mu1, cide)
        h = me[:, 4] == 0
        cide3 = event_cells(ev, np.nan_to_num(me[:, 7], nan=2400.0)); exe3 = ev[..., 2] - lookup(*res_c3[(variant, which)][2], cide3)
        for o in [-6, -3, -2, -1, 1, 2]:
            oi = list(OF).index(o * 2); ok = h & (ev[:, oi, 3] == 1) & np.isfinite(exe3[:, oi])
            ci = boot(exe3[ok, oi], me[ok, 0]) if ok.sum() > 5 else (np.nan, np.nan)
            print("   [C3-own] event t_m%+3ds: %s  mean CI [%+.3f, %+.3f]" % (o, st(exe3[ok, oi]), *ci))
        print("  merge events HDV F*: %d (CAV F* %d)" % (h.sum(), (~h).sum()))
        for o in [-10, -6, -4, -3, -2, -1, 0, 1, 2, 3, 5, 8]:
            oi = list(OF).index(o * 2); ok = h & (ev[:, oi, 3] == 1) & np.isfinite(exe[:, oi])
            print("   t_m%+3ds raw a mean %+.3f | excess (SUMO ctrl) %s" % (o, np.nanmean(ev[h & (ev[:, oi, 3] == 1), oi, 2]) if ok.sum() else np.nan, st(exe[ok, oi])))
        # min accel in [-6,+6] s (as discharge check)
        w = slice(list(OF).index(-12), list(OF).index(12) + 1)
        mn = np.nanmin(np.where(ev[:, w, 3] == 1, ev[:, w, 2], np.nan), axis=1)
        pre = np.nanmin(np.where(ev[:, list(OF).index(-12):list(OF).index(0), 3] == 1, ev[:, list(OF).index(-12):list(OF).index(0), 2], np.nan), axis=1)
        post = np.nanmin(np.where(ev[:, list(OF).index(0):list(OF).index(12) + 1, 3] == 1, ev[:, list(OF).index(0):list(OF).index(12) + 1, 2], np.nan), axis=1)
        print("  F* min accel [-6,+6]s p50 %+.2f, pre[-6,0) p50 %+.2f, post[0,6] p50 %+.2f (n=%d)" % (np.nanmedian(mn[h]), np.nanmedian(pre[h]), np.nanmedian(post[h]), h.sum()))
