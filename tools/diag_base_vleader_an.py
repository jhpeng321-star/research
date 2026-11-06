"""S-BASEDIAG Q2 analysis of tools/diag_base_vleader.py output (recorded SUMO TRAIN rows; diagnostic only).

Tables: recorded vs IDM(own leader) vs IDM(min(own, ramp partner as virtual leader)) for lane-1 HDV
rows, far from the merge, in the merge zone without a partner, and for target followers by time to
the partner's merge / bumper gap.  The per-bin blend weight w (rec ~ own + w (min - own)) is a
least-squares DESCRIPTION of how much of the virtual-leader deceleration SUMO followers realise; it is
not a model parameter.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "tools")
from diag_base_vleader import PSETS, idm  # noqa: E402

D = dict(np.load(sys.argv[1] if len(sys.argv) > 1 else "tmp_basediag/vleader.npz"))
n = len(D["s"])
print("rows", n, "episodes", len(np.unique(D["ep"])))
v, rec = D["v"], D["rec"]
for name, (a, b, T, s0) in PSETS.items():
    D[f"own_{name}"] = idm(v, D["v_l"], D["gap_l"], a, b, T, s0)
    gr = np.nan_to_num(D["gap_r"], nan=1e3)
    vr = np.where(np.isfinite(D["v_r"]), D["v_r"], v)
    D[f"virt_{name}"] = idm(v, vr, gr, a, b, T, s0)
    D[f"min_{name}"] = np.minimum(D[f"own_{name}"], D[f"virt_{name}"])


def line(lbl, m, extra=""):
    if m.sum() < 30:
        return
    r = rec[m]
    out = f"{lbl} | {m.sum()} | {r.mean():+.2f}"
    for name in PSETS:
        o, mn = D[f"own_{name}"][m], D[f"min_{name}"][m]
        d = mn - o
        w = float(((r - o) * d).sum() / max((d * d).sum(), 1e-9))
        out += f" | {o.mean():+.2f} | {mn.mean():+.2f} | {w:.2f}"
    print(out + extra)


hdr = "bin | n | rec | " + " | ".join(f"own_{k} | min_{k} | w_{k}" for k in PSETS)
s = D["s"]
hp = D["has_p"] > 0
zone_p = hp & (D["dend"] >= 0) & (D["dend"] <= 75.44)
pull = D["v_l"] > v + 0.5           # leader pulling away
print("\n## A. far upstream (s < -400 m), lane-1 HDV rows, v < 2 m/s, leader pulling away, by bumper gap")
print(hdr)
for lo, hi in ((0, 3), (3, 5), (5, 8), (8, 12), (12, 20), (20, 40), (40, 1e4)):
    line(f"gap [{lo},{hi})", (s < -400) & (v < 2) & pull & (D["gap_l"] >= lo) & (D["gap_l"] < hi))
print("\n## B. merge zone (0 <= s < 75.44), no ramp partner, v < 2, leader pulling away, by gap")
print(hdr)
for lo, hi in ((0, 3), (3, 5), (5, 8), (8, 12), (12, 20), (20, 40), (40, 1e4)):
    line(f"gap [{lo},{hi})", (s >= 0) & (s < 75.44) & ~hp & (v < 2) & pull & (D["gap_l"] >= lo) & (D["gap_l"] < hi))
print("\n## B2. same, C3-type controls only (a waiting ramp car in the zone, none within -20..+30 m)")
print(hdr)
for lo, hi in ((0, 3), (3, 5), (5, 8), (8, 12), (12, 20), (20, 40), (40, 1e4)):
    line(f"gap [{lo},{hi})", (D["c3"] > 0) & (v < 2) & pull & (D["gap_l"] >= lo) & (D["gap_l"] < hi))
print("\n## B3. upstream approach (-400 <= s < 0), no partner, v < 2, leader pulling away, by gap")
print(hdr)
for lo, hi in ((0, 3), (3, 5), (5, 8), (8, 12), (12, 20), (20, 40), (40, 1e4)):
    line(f"gap [{lo},{hi})", (s >= -400) & (s < 0) & ~hp & (v < 2) & pull & (D["gap_l"] >= lo) & (D["gap_l"] < hi))
print("\n## C. target followers of a merge-zone ramp car, by t - t_merge (s)")
print(hdr)
tm = D["tmerge"]
for lo, hi in ((6, 10), (4, 6), (3, 4), (2, 3), (1, 2), (0, 1)):
    line(f"[-{hi},-{lo})", zone_p & (tm > lo) & (tm <= hi))
line("never merges (in window)", zone_p & ~np.isfinite(tm))
last3 = zone_p & (tm > 0) & (tm <= 3)
print("\n## D. target followers, last 3 s before the merge, by bumper gap to the ramp car (m)")
print(hdr)
for lo, hi in ((-50, 0), (0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 15), (15, 30), (30, 1e4)):
    line(f"gap_r [{lo},{hi})", last3 & (D["gap_r"] >= lo) & (D["gap_r"] < hi))
print("\n## E. target followers, last 3 s, by own leader gap (m) (where is the over-prediction?)")
print(hdr)
for lo, hi in ((0, 3), (3, 5), (5, 8), (8, 12), (12, 20), (20, 40), (40, 1e4)):
    line(f"gap_l [{lo},{hi})", last3 & (D["gap_l"] >= lo) & (D["gap_l"] < hi))
print("\n## F. target followers, last 3 s: own leader gap vs ramp gap")
print("share gap_r < gap_l (ramp car is the nearer obstacle):", float((D["gap_r"][last3] < D["gap_l"][last3]).mean()))
print("median gap_l, gap_r, v, v_l, v_r:", [float(np.median(D[k][last3])) for k in ("gap_l", "gap_r", "v", "v_l", "v_r")])
print("share invisible (junction) vehicle nearer than own leader:", float((D["inv_d"][last3] < D["gap_l"][last3] + 5).mean()))
print("\n## G. all lane-1 rows: invisible vehicle nearer than own leader, by s")
for lo, hi in ((-400, 0), (0, 25), (25, 50), (50, 75.44), (75.44, 150)):
    m = (s >= lo) & (s < hi)
    print(f"s [{lo},{hi}) n {m.sum()} share {float((D['inv_d'][m] < D['gap_l'][m] + 5).mean()):.3f}")
print("\n## H. far partners (gap_r 30+) and ramp car behind: virtual leader is inert?")
line("zone partner gap_r > 30, all times", zone_p & (D["gap_r"] > 30))
line("zone partner gap_r 6..20, tm > 10 or never", zone_p & (D["gap_r"] > 1) & (D["gap_r"] < 15) & ~(tm <= 10))


# ------------------------------------------------------------------ decomposition by matched SUMO responses
from scipy.spatial import cKDTree  # noqa: E402

print("\n## I. SUMO slow-start response far from the merge (s < -400, v < 2, gap_l > 12, leader pulling away)")
m = (s < -400) & (v < 2) & (D["gap_l"] > 12) & pull
print("n", int(m.sum()), "rec p10/50/90/99/max", np.percentile(rec[m], [10, 50, 90, 99, 100]).round(2))
for lo, hi in ((0, 0.3), (0.3, 1), (1, 2)):
    mm = m & (v >= lo) & (v < hi)
    print(f"  v [{lo},{hi}) n {int(mm.sum())} rec mean {rec[mm].mean():+.2f} p90 {np.percentile(rec[mm], 90):+.2f}")
m2 = (s < -400) & (D["gap_l"] > 60) & (v > 5) & (v < 20)
print("free road far upstream (gap > 60, 5 < v < 20): n", int(m2.sum()), "rec mean %+.2f p90 %+.2f p99 %+.2f"
      % (rec[m2].mean(), np.percentile(rec[m2], 90), np.percentile(rec[m2], 99)))


def knn_expect(ref_mask, query_mask, k=25):
    f = np.stack([v / 2.0, np.arcsinh(np.minimum(D["gap_l"], 200) / 5.0), (D["v_l"] - v) / 2.0], -1)
    tree = cKDTree(f[ref_mask])
    _, idx = tree.query(f[query_mask], k=k)
    return rec[ref_mask][idx].mean(-1)


far_ref = (s < -400) & ~hp
zone_ref = (s >= 0) & (s < 75.44) & ~hp & ~((D["inv_d"] < D["gap_l"] + 5))
print("\n## J. last-3 s target followers: own IDM vs SUMO response of state-matched rows far upstream /"
      " in the zone (unpartnered) vs the follower's recorded acceleration")
print("group | n | IDM_wm | IDM_sumo | SUMO far-upstream match | SUMO zone unpartnered match | recorded")
for lbl, q in (("last 3 s, all", last3), ("last 3 s, v<2", last3 & (v < 2)),
               ("last 3 s, v<2, gap_l 5-20", last3 & (v < 2) & (D["gap_l"] > 5) & (D["gap_l"] < 20)),
               ("[-6,-3) s", zone_p & (tm > 3) & (tm <= 6))):
    ef, ez = knn_expect(far_ref, q), knn_expect(zone_ref, q)
    print(f"{lbl} | {int(q.sum())} | {D['own_wm'][q].mean():+.2f} | {D['own_sumo'][q].mean():+.2f} | "
          f"{ef.mean():+.2f} | {ez.mean():+.2f} | {rec[q].mean():+.2f}")
np.save("tmp_basediag/far_ref_mask.npy", far_ref)
