"""S-YIELD: C3 control = HDV lane-1 rows in zones where a ramp car IS waiting (nzone>0) but the vehicle is not adjacent to it:
not the target follower and no waiting ramp car within 30 m ahead/20 m behind. Fixes 'a ramp car waits in the zone / jam' and
varies only adjacency (the cleanest test of active yielding vs existing jam)."""
import sys
sys.path.insert(0, "tools")
import numpy as np
from diag_yield_common import *
d = load_samples(); cid = cells(d); hdv = d["cav"] == 0; l1 = d["lane"] == 1
ctrl3 = hdv & l1 & (d["nzone"] > 0) & (d["tf"] == 0) & ~(d["dA"] < 30) & ~(d["dB"] < 20)
u, inv = np.unique(cid, return_inverse=True)
n3 = np.bincount(inv[ctrl3], minlength=len(u)); mu3 = np.bincount(inv[ctrl3], d["a"][ctrl3], minlength=len(u)) / np.maximum(n3, 1)
mu3 = np.where(n3 >= 20, mu3, np.nan); ex3 = d["a"] - mu3[inv]
print("C3 control rows", ctrl3.sum(), "cells>=20:", int(np.isfinite(mu3).sum()), "coverage of tf rows:", np.isfinite(ex3[hdv & l1 & (d['tf'] == 1)]).mean())
print("control self-check mean excess %.4f" % np.nanmean(ex3[ctrl3]))
def st(x, sel):
    x = x[sel & np.isfinite(x)]
    return "%7d | %+.3f | %+.3f | %+.3f | %.3f | raw a med %+.3f" % (len(x), x.mean(), np.median(x), np.percentile(x, 10), (x < -1).mean(), 0) if len(x) > 30 else "n=%d" % len(x)
print("\nT1' (C3) target followers by offset dA")
for lo, hi in [(0, 3), (3, 6), (6, 10), (10, 15), (15, 20), (20, 30), (30, 45), (45, 60), (60, 100)]:
    sel = hdv & l1 & (d["dA"] >= lo) & (d["dA"] < hi) & (d["tf"] == 1)
    print("| %g-%g | %s |" % (lo, hi, st(ex3, sel)))
base = hdv & l1 & (d["tf"] == 1) & (d["dA"] < 20)
print("\nT2' by F speed (C3) for tf rows dA<20:")
for lo, hi in [(0, 1), (1, 3), (3, 6), (6, 12), (12, 40)]:
    print("| vF %g-%g | %s |" % (lo, hi, st(ex3, base & (d["v"] >= lo) & (d["v"] < hi))))
print("by dend (C3):")
for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 35), (35, 50), (50, 76)]:
    print("| dend %g-%g | %s |" % (lo, hi, st(ex3, base & (d["dendR"] >= lo) & (d["dendR"] < hi))))
z = np.load(OUT + "/events.npz"); ev = z["ev"]; me = z["meta"]; offs = list(z["offs"]); i0 = offs.index(0)
exe = ev[..., 2] - lookup(u, mu3, event_cells(ev, me[:, 7])); h = me[:, 4] == 0
print("\nT3' event-aligned F* excess (C3)")
for o in [-30, -20, -16, -12, -10, -8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 10]:
    oi = i0 + 2 * o; ok = h & (ev[:, oi, 3] == 1) & np.isfinite(exe[:, oi]); x = exe[ok, oi]
    print("| %+d s | %d | %+.3f | %+.3f | %+.3f | %.3f |" % (o, ok.sum(), x.mean(), np.median(x), np.percentile(x, 10), (x < -0.5).mean()))
def shed(w0, w1):
    rows = slice(i0 + w0, i0 + w1); ok = (ev[:, rows, 3] == 1) & np.isfinite(exe[:, rows])
    return -(np.where(ok, exe[:, rows], 0).sum(1)) * 0.5, ok.sum(1) >= 0.8 * (w1 - w0)
for W in (5, 10):
    sh, okc = shed(-2 * W, 0); shp, okp = shed(-2 * W - 40, -40)
    x = sh[h & okc]; xp = shp[h & okp]
    print("W=%d: pre-merge n=%d mean %.2f med %.2f share>1 %.3f >2 %.3f | placebo n=%d mean %.2f med %.2f share>1 %.3f >2 %.3f" % (W, len(x), x.mean(), np.median(x), (x > 1).mean(), (x > 2).mean(), len(xp), xp.mean(), np.median(xp), (xp > 1).mean(), (xp > 2).mean()))
# jam-state stratified: F* speed 5 s before merge
vF = ev[..., 1]; lane = ev[..., 3]
ok = h & (lane[:, i0 - 10] == 1) & (lane[:, i0 - 1] == 1); v5 = vF[:, i0 - 10]
sh5, ok5 = shed(-10, 0); shp5, okp5 = shed(-50, -40)
print("\nExcess speed shed (C3) in last 5 s vs placebo, by F* speed 5 s before merge")
for lo, hi in [(0, 1), (1, 2), (2, 4), (4, 8), (8, 40)]:
    s = ok & ok5 & (v5 >= lo) & (v5 < hi); sp = ok & okp5 & (vF[:, i0 - 50] >= lo) & (vF[:, i0 - 50] < hi)
    print("| F* v %g-%g | n=%d shed mean %.2f med %.2f share>1 %.3f | placebo n=%d mean %.2f share>1 %.3f |" % (lo, hi, s.sum(), sh5[s].mean(), np.median(sh5[s]), (sh5[s] > 1).mean(), sp.sum(), shp5[sp].mean() if sp.sum() else np.nan, (shp5[sp] > 1).mean() if sp.sum() else np.nan))
wait = ev[:, i0, 12]; v0 = vF[:, i0 - 1]; gap = ev[:, i0 - 1, 11]
act = ok & ok5 & (sh5 > 1.0)
print("\nC. decomposition of merges (C3 shed>1 m/s in last 5 s = active yield):")
okall = ok & ok5
for nm, s in (("active yield (shed>1)", act), ("no active yield, F* slow (v5<4)", okall & ~act & (v5 < 4)), ("no active yield, F* fast (v5>=4)", okall & ~act & (v5 >= 4))):
    print("%-40s count share %.3f | wait-weighted %.3f | median final gap %.1f | median wait %.0f s" % (nm, s.sum() / okall.sum(), wait[s].sum() / wait[okall].sum(), np.nanmedian(gap[s]), np.nanmedian(wait[s])))
