"""S-YIELD Q1/Q2 tables on the recorded TRAIN episodes (analysis only)."""
import sys
sys.path.insert(0, "tools")
import numpy as np
from diag_yield_common import *

d = load_samples(); cid = cells(d); hdv = d["cav"] == 0; l1 = d["lane"] == 1
ctrl1 = l1 & hdv & (d["nRamp"] == 0)
u1, inv1 = np.unique(cid, return_inverse=True)
n1 = np.bincount(inv1[ctrl1], minlength=len(u1)); mu1 = np.bincount(inv1[ctrl1], d["a"][ctrl1], minlength=len(u1)) / np.maximum(n1, 1)
mu1 = np.where(n1 >= 20, mu1, np.nan); ex1 = d["a"] - mu1[inv1]
u0, mu0 = ctrl_model(d); ex0 = d["a"] - lookup(u0, mu0, cid)
print("rows", len(d["a"]), "HDV lane1", (hdv & l1).sum(), "C1 controls", ctrl1.sum(), "cells with >=20 ctrl", int(np.isfinite(mu1).sum()))


def st(x, sel):
    x = x[sel & np.isfinite(x)]
    if len(x) < 30: return "n=%d" % len(x)
    return "%7d | %+.3f | %+.3f | %+.3f | %+.3f | %.3f" % (len(x), x.mean(), np.median(x), np.percentile(x, 10), np.percentile(x, 25), (x < -1.0).mean())


print("\n## T1 follower excess accel vs ramp-car offset dA (HDV lane-1; n | mean | med | p10 | p25 | share<-1.0)")
for tfv, nm in ((1, "target follower"), (0, "other lane-1")):
    print("**%s**" % nm)
    for lo, hi in [(0, 3), (3, 6), (6, 10), (10, 15), (15, 20), (20, 30), (30, 45), (45, 60), (60, 100), (100, 150)]:
        sel = hdv & l1 & (d["dA"] >= lo) & (d["dA"] < hi) & (d["tf"] == tfv)
        print("| %g-%g | C1 %s | C2 %s |" % (lo, hi, st(ex1, sel), st(ex0, sel)))
print("far/no ramp car ahead within 150 (C1 check):", st(ex1, hdv & l1 & (d["nRamp"] == 0)))
base = hdv & l1 & (d["tf"] == 1) & (d["dA"] < 20) & (d["dA"] >= 0)
gap = d["dA"] - 5
print("\n## T2 dependence (target followers, dA<20, C1)")
for name, var, edges in [("wait so far (s)", d["waitR"], [0, 5, 10, 20, 40, 80, 1e9]), ("ramp-car distance to lane end (m)", d["dendR"], [0, 5, 10, 20, 35, 50, 76]),
                         ("ramp-car speed (m/s)", d["vR"], [0, 0.5, 1, 2, 4, 8, 30]), ("bumper gap F->R (m)", gap, [-5, 0, 1, 2, 3, 5, 10, 15]),
                         ("F speed minus R speed (m/s)", d["v"] - d["vR"], [-30, -1, 0, 1, 2, 4, 8, 30]), ("F speed (m/s)", d["v"], [0, 1, 3, 6, 10, 15, 35]),
                         ("demand (vph)", d["dem"], [1800, 2000, 2200, 2500, 3000]), ("CAV penetration", d["pen"], [0, .1, .2, .3, .45])]:
    print("**%s**" % name)
    for lo, hi in zip(edges[:-1], edges[1:]):
        print("| [%g,%g) | %s |" % (lo, hi, st(ex1, base & (var >= lo) & (var < hi))))
# 2-D: dend x wait
print("**mean excess, dend x wait**")
for dl, dh in [(0, 10), (10, 25), (25, 50), (50, 76)]:
    row = []
    for wl, wh in [(0, 10), (10, 30), (30, 80), (80, 1e9)]:
        sel = base & (d["dendR"] >= dl) & (d["dendR"] < dh) & (d["waitR"] >= wl) & (d["waitR"] < wh)
        x = ex1[sel & np.isfinite(ex1)]; row.append("%+.2f (%d)" % (x.mean(), len(x)) if len(x) > 30 else "-")
    print("| dend %g-%g | %s |" % (dl, dh, " | ".join(row)))

z = np.load(OUT + "/events.npz"); ev = z["ev"]; me = z["meta"]; offs = list(z["offs"])
cide = event_cells(ev, me[:, 7]); exe = ev[..., 2] - lookup(u1, mu1, cide)
h = me[:, 4] == 0; i0 = offs.index(0)
print("\n## T3 event-aligned F* excess (HDV F*, C1; every 0.5 s from -8 s to +4 s)")
for o in np.arange(-16, 9):
    oi = i0 + o; ok = h & (ev[:, oi, 3] == 1) & np.isfinite(exe[:, oi]); x = exe[ok, oi]
    print("| %+.1f | %d | %+.3f | %+.3f | %+.3f | %.3f |" % (o / 2, ok.sum(), x.mean(), np.median(x), np.percentile(x, 10), (x < -0.5).mean()))
print("\n## onset: first offset at which mean excess < -0.1 (from -20 s):", end=" ")
for o in range(-40, 0):
    oi = i0 + o; ok = h & (ev[:, oi, 3] == 1) & np.isfinite(exe[:, oi])
    if exe[ok, oi].mean() < -0.15: print("%.1f s" % (o / 2)); break

print("\n## T4 event-level speed shed in the last W s before the merge vs placebo window")
def shed(w0, w1):   # window rows [w0, w1) offsets
    rows = slice(i0 + w0, i0 + w1)
    ok = (ev[:, rows, 3] == 1) & np.isfinite(exe[:, rows])
    sh = -(np.where(ok, exe[:, rows], 0).sum(1)) * 0.5
    return sh, ok.sum(1) >= 0.8 * (w1 - w0)
for W in (5, 10):
    sh, okc = shed(-2 * W, 0); sel = h & okc
    shp, okp = shed(-2 * W - 40, -40); selp = h & okp
    x = sh[sel]; xp = shp[selp]
    print("W=%ds  pre-merge: n=%d mean %.2f med %.2f p75 %.2f p90 %.2f | share>0.5 %.3f >1 %.3f >2 %.3f ; placebo window (-%d..-%d s): n=%d mean %.2f med %.2f | share>0.5 %.3f >1 %.3f >2 %.3f" % (
        W, sel.sum(), x.mean(), np.median(x), np.percentile(x, 75), np.percentile(x, 90), (x > .5).mean(), (x > 1).mean(), (x > 2).mean(),
        W + 20, 20, selp.sum(), xp.mean(), np.median(xp), (xp > .5).mean(), (xp > 1).mean(), (xp > 2).mean()))

print("\n## T5 merge outcome: gap opened by follower slowdown (HDV F*, W=10 s counterfactual 'hold speed')")
vF = ev[..., 1]; gf = ev[:, i0 - 1, 11]; wait = ev[:, i0, 12]
W = 10; ref = vF[:, i0 - 2 * W]
D = np.nansum(ref[:, None] - vF[:, i0 - 2 * W + 1:i0], axis=1) * 0.5
ok = h & np.isfinite(ref) & np.isfinite(gf) & (ev[:, i0 - 2 * W, 3] == 1) & (ev[:, i0 - 1, 3] == 1)
gnf = gf - D
print("events with F* in lane 1 at -10 s and -0.5 s:", ok.sum(), "of", h.sum())
print("gap_f at last pre-merge row p10/50/90:", np.percentile(gf[ok], [10, 50, 90]).round(2))
print("distance given up D (m) pct 10/25/50/75/90:", np.percentile(D[ok], [10, 25, 50, 75, 90]).round(2), "share D>0: %.3f" % (D[ok] > 0).mean())
for thr in (0, 2, 3.4):
    print("share where hold-speed counterfactual gap < %.1f m: %.3f" % (thr, (gnf[ok] < thr).mean()))
# categories: small final gap & big shed vs big final gap
sh5, ok5 = shed(-10, 0)
yield_ev = ok & ok5 & (sh5 > 1.0); small = gf < 5
print("2x2 (HDV F*, W=5 s shed > 1 m/s = 'slowed'; final gap < 5 m = 'tight'):")
for a_, nm in ((yield_ev, "slowed"), (~yield_ev, "not slowed")):
    print("  %-10s tight %.3f  loose %.3f" % (nm, (ok & ok5 & a_ & small).sum() / (ok & ok5).sum(), (ok & ok5 & a_ & ~small).sum() / (ok & ok5).sum()))
print("wait-time weighted share of merges following a slowdown (shed>1): %.3f ; unweighted %.3f" % (
    (wait[ok & ok5 & yield_ev]).sum() / wait[ok & ok5].sum(), (ok & ok5 & yield_ev).sum() / (ok & ok5).sum()))
# did the follower change during the last 10 s?  (target follower identity) via sample table
print("\nR outcomes: merged events total %d ; wait pctl 10/50/90 (s): %s" % (len(ev), np.percentile(wait, [10, 50, 90]).round(1)))
