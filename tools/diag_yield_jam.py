"""S-YIELD: active follower deceleration vs existing slow jam at merge events (raw + matched), HDV F*."""
import sys
sys.path.insert(0, "tools")
import numpy as np
from diag_yield_common import *
d = load_samples(); cid = cells(d); hdv = d["cav"] == 0; l1 = d["lane"] == 1
ctrl1 = l1 & hdv & (d["nRamp"] == 0)
u1, inv1 = np.unique(cid, return_inverse=True)
n1 = np.bincount(inv1[ctrl1], minlength=len(u1)); mu1 = np.bincount(inv1[ctrl1], d["a"][ctrl1], minlength=len(u1)) / np.maximum(n1, 1)
mu1 = np.where(n1 >= 20, mu1, np.nan)
z = np.load(OUT + "/events.npz"); ev = z["ev"]; me = z["meta"]; offs = list(z["offs"]); i0 = offs.index(0)
exe = ev[..., 2] - lookup(u1, mu1, event_cells(ev, me[:, 7])); h = me[:, 4] == 0
vF, aF, lane = ev[..., 1], ev[..., 2], ev[..., 3]
print("A. F* speed at -5 s and -0.5 s, and speed change in the last 5 s (raw), by F* speed 5 s before the merge")
ok = h & (lane[:, i0 - 10] == 1) & (lane[:, i0 - 1] == 1)
v5 = vF[:, i0 - 10]; v0 = vF[:, i0 - 1]
print("events", ok.sum(), "| F* speed at -0.5 s pctl 10/50/90:", np.percentile(v0[ok], [10, 50, 90]).round(2), "| at -5 s:", np.percentile(v5[ok], [10, 50, 90]).round(2))
print("| F* speed 5 s before merge | n | dv last 5 s med | share dv<-0.5 | share dv<-1 | share dv>+0.5 | min raw a last 3 s med | share min a<-1 | matched excess shed (m/s) mean |")
rows = slice(i0 - 6, i0)
sh = -np.nansum(np.where(np.isfinite(exe[:, i0 - 10:i0]), exe[:, i0 - 10:i0], 0), 1) * 0.5
mina = np.nanmin(np.where(lane[:, rows] == 1, aF[:, rows], np.nan), 1)
for lo, hi in [(0, 1), (1, 2), (2, 4), (4, 8), (8, 14), (14, 40)]:
    s = ok & (v5 >= lo) & (v5 < hi); dv = (v0 - v5)[s]
    print("| %g-%g | %d | %+.2f | %.3f | %.3f | %.3f | %+.2f | %.3f | %+.2f |" % (lo, hi, s.sum(), np.median(dv), (dv < -.5).mean(), (dv < -1).mean(), (dv > .5).mean(), np.nanmedian(mina[s]), (mina[s] < -1).mean(), np.nanmean(sh[s])))
print("\nB. same statistics for PLACEBO windows (offsets -25..-20 s, same events) vs real (-5..0)")
v25 = vF[:, i0 - 50]; v20 = vF[:, i0 - 40]; okp = h & (lane[:, i0 - 50] == 1) & (lane[:, i0 - 40] == 1)
dvp = (v20 - v25)[okp]; dvr = (v0 - v5)[ok]
print("dv over 5 s: placebo share<-0.5 %.3f <-1 %.3f ; real share<-0.5 %.3f <-1 %.3f" % ((dvp < -.5).mean(), (dvp < -1).mean(), (dvr < -.5).mean(), (dvr < -1).mean()))
minp = np.nanmin(np.where(lane[:, i0 - 46:i0 - 40] == 1, aF[:, i0 - 46:i0 - 40], np.nan), 1)
print("min raw a over 3 s: placebo share<-1 %.3f ; real share<-1 %.3f ; placebo med %+.2f real med %+.2f" % ((minp[okp] < -1).mean(), (mina[ok] < -1).mean(), np.nanmedian(minp[okp]), np.nanmedian(mina[ok])))
print("\nC. wait-time-weighted decomposition of the merge instant (HDV F*, F* in lane 1 for the last 5 s)")
gap = ev[:, i0 - 1, 11]; wait = ev[:, i0, 12]
yielded = ok & ((v0 - v5) < -0.5)
slow_jam = ok & (v5 < 4) & ~yielded
fast_open = ok & (v5 >= 4) & ~yielded
for nm, s in (("F* slowed >0.5 m/s in last 5 s (active)", yielded), ("not slowed, F* already slow (<4 m/s) = existing jam", slow_jam), ("not slowed, F* fast (>=4) = existing gap", fast_open)):
    print("%-55s count share %.3f | wait-weighted %.3f | median final gap %.1f m | median wait %.0f s" % (nm, s.sum() / ok.sum(), wait[s].sum() / wait[ok].sum(), np.nanmedian(gap[s]), np.nanmedian(wait[s])))
# among 'yielded': what fraction was it a necessity (hold-speed counterfactual overlaps)
D = np.nansum(v5[:, None] - vF[:, i0 - 9:i0], 1) * 0.5
print("among 'slowed' events: median distance given up %.1f m, median final gap %.1f m; share where counterfactual gap(hold speed)<0: %.3f" % (np.median(D[yielded]), np.nanmedian(gap[yielded]), ((gap - D)[yielded] < 0).mean()))
print("among not-slowed/jam events: share with final gap < 5 m: %.3f" % ((gap[slow_jam] < 5).mean()))
# overall raw acceleration of followers at waiting-ramp-car moments (matches the stuck-diag statement)
sel = hdv & l1 & (d["tf"] == 1) & (d["dA"] < 20) & (d["dA"] >= 0)
print("\nD. raw accel of target followers while a ramp car waits (dA<20): median %+.3f p10 %+.3f p25 %+.3f ; speed median %.2f p25-p75 %s ; n=%d" % (np.median(d["a"][sel]), *np.percentile(d["a"][sel], [10, 25]), np.median(d["v"][sel]), np.percentile(d["v"][sel], [25, 75]).round(2), sel.sum()))
print("   same rows matched controls: median expected a %+.3f" % np.nanmedian(mu1[inv1][sel]))
vs = d["v"][sel]
for lo, hi in [(0, 1), (1, 3), (3, 6), (6, 12), (12, 40)]:
    s = sel & (d["v"] >= lo) & (d["v"] < hi); x = d["a"][s]; e = (d["a"] - mu1[inv1])[s]; e = e[np.isfinite(e)]
    print("   v_F %g-%g: share of rows %.3f, raw a med %+.3f p10 %+.3f, matched excess mean %+.3f med %+.3f (n=%d)" % (lo, hi, s.sum() / sel.sum(), np.median(x), np.percentile(x, 10), e.mean(), np.median(e), len(e)))
