"""S-DIAG9 analysis of ``diag_r9stuck_trace.py`` pickles (offline, numpy only).  DIAGNOSTIC ONLY.

python tools/diag_r9stuck_analyze.py table  main_k5.pkl           # variant table + speed / lane splits
python tools/diag_r9stuck_analyze.py wait   main_k5.pkl r9        # stuck-car waiting-phase statistics
python tools/diag_r9stuck_analyze.py sumo   all_r9.pkl  r9        # SUMO merge instant boundary c / follower yield
"""
from __future__ import annotations

import pickle
import sys

import numpy as np

sys.path.insert(0, "tools")
import diag_r9stuck_check_base as base  # noqa: E402

END = 75.44
DT = 0.5
P0 = dict(s_f=2.648, T_f=0.270, b_f=0.985, s_l=1.184, T_l=0.323, b_l=1.176, tau=0.281, w=0.255)


def load(path):
    return pickle.load(open(path, "rb"))


def by_var(d):
    out = {}
    for r in d["results"]:
        out.setdefault(r["var"], []).append(r)
    return out


def req(p, side, ego_v, other_v):
    if side == "f":
        return p["s_f"] + p["T_f"] * other_v + np.maximum(other_v ** 2 - ego_v ** 2, 0) / (2 * p["b_f"])
    return p["s_l"] + p["T_l"] * ego_v + np.maximum(ego_v ** 2 - other_v ** 2, 0) / (2 * p["b_l"])


def cside(p, side, gap, ego_v, other_v, has=True):
    d = req(p, side, ego_v, other_v)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where((gap > 0) & np.isfinite(gap) & np.isfinite(other_v) & has,
                     np.log(np.maximum(gap, 1e-9) / d), -np.inf)
    return np.where(has, c, np.inf), d


def table(path):
    d = load(path)
    names = [s[0] for s in d["specs"]]
    rows = by_var(d)
    k = d["k"]
    print(f"k={k}")
    print("| variant | stuck in lane 2 | ramp HDVs | median dep WM-SUMO s | HDV ds60 | HDV ds120 | ramp ds60 | "
          "blow-ups (states p>=0.5) | err | spd WM/SUMO @60 | spd WM/SUMO @120 | slow(<2) WM/SUMO @120 | "
          "ds120 by t0 lane 0/1/2 |")
    print("|" + "---|" * 13)
    for n in names:
        ok = [r for r in rows[n] if not r.get("error")]
        err = len(rows[n]) - len(ok)
        tw = np.concatenate([r["t_wm"] for r in ok])
        ts = np.concatenate([r["t_rec"] for r in ok])
        m = np.isfinite(tw) & np.isfinite(ts)
        dt = tw[m] - ts[m]
        st = {}
        for r in ok:
            st.setdefault(r["idx"], []).append(r)
        probs = [base.state_blow_probability(v) for v in st.values()]
        blow = sum(1 for p in probs if np.isfinite(p) and p >= 0.5)

        def med(key, rr=ok):
            return np.nanmedian([r.get(key, np.nan) for r in rr])

        def mean(key, rr=ok):
            return np.nanmean([r.get(key, np.nan) for r in rr])
        lanes = "/".join(f"{med(f'ds120_l{ln}'):.0f}" for ln in (0, 1, 2))
        print(f"| {n} | {int(np.isinf(tw).sum())} | {len(tw)} | {np.median(dt):+.1f} | {med('ds60'):.1f} | "
              f"{med('ds120'):.1f} | {med('ramp_ds60'):.1f} | {blow}/{len(st)} | {err} | "
              f"{mean('spd_wm60'):.1f}/{mean('spd_rec60'):.1f} | {mean('spd_wm120'):.1f}/{mean('spd_rec120'):.1f} | "
              f"{mean('slow_wm120'):.1f}/{mean('slow_rec120'):.1f} | {lanes} |")
    sumo_stuck = int(np.isinf(np.concatenate([r['t_rec'] for r in rows[names[0]] if not r.get('error')])).sum())
    print(f"SUMO twins never off lane 2 in the window: {sumo_stuck} (pooled over samples of {names[0]})")


def car_rows(t, p):
    """Per-row geometry arrays for a trace (row i of the ``c`` arrays = decision at state row i+1)."""
    T = len(t["c"])
    ego_v = t["v"][1:T + 1]
    pos = t["pos"][1:T + 1]
    lane = t["lane"][1:T + 1]
    mask = t["mask"][1:T + 1]
    cf, df = cside(p, "f", t["gap_f"], ego_v, t["v_f"], t["has_f"] > 0.5)
    cl, dl = cside(p, "l", t["gap_l"], ego_v, t["v_l"], t["has_l"] > 0.5)
    return dict(ego_v=ego_v, pos=pos, lane=lane, mask=mask, cf=cf, df=df, cl=cl, dl=dl, c=np.minimum(cf, cl))


def qs(x, qsl=(10, 50, 90)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return "n/a" if not x.size else "/".join(f"{np.percentile(x, q):.1f}" for q in qsl) + f" (n={x.size})"


def wait_stats(path, var):
    d = load(path)
    rr = [r for r in by_var(d)[var] if not r.get("error")]
    tot = sum(len(r["t_wm"]) for r in rr)
    stuck_cars = []
    for r in rr:
        for vid, t in r["traces"].items():
            if t["stuck"]:
                stuck_cars.append((r["idx"], r["sample"], vid, t))
    print(f"[{var}] ramp-car samples {tot}, stuck (never off lane 2 while SUMO twin present) {len(stuck_cars)}")
    rows = dict(v_f=[], gap_f=[], d_f=[], c=[], cf=[], cl=[], elig=[], phys=[], veto=[], binding_f=[], v_l=[],
                gap_l=[], d_l=[], hasf=[], pinned=[], ego_v=[], dist_end=[], cdf=[])
    reach_end = 0
    never_wait = 0
    sumo_dep = []
    per_car = []
    for idx, sample, vid, t in stuck_cars:
        p = t.get("params", P0)
        a = car_rows(t, p)
        T = len(a["c"])
        wait = a["mask"] & (a["lane"] == 2) & (a["pos"] >= END - 10.0) & (a["ego_v"] < 2.0)
        if (a["mask"] & (a["lane"] == 2) & (a["pos"] >= END - 1e-3)).any():
            reach_end += 1
        if not wait.any():
            never_wait += 1
            per_car.append((idx, sample, vid, False, np.nan, np.nan))
            continue
        w = np.flatnonzero(wait)
        for key, arr in (("v_f", t["v_f"]), ("gap_f", t["gap_f"]), ("v_l", t["v_l"]), ("gap_l", t["gap_l"]),
                         ("elig", t["elig"]), ("phys", t["phys"]), ("veto", t["veto"]), ("hasf", t["has_f"]),
                         ("cdf", t["cdf"])):
            rows[key].append(arr[w])
        rows["d_f"].append(a["df"][w]); rows["d_l"].append(a["dl"][w])
        rows["c"].append(t["c"][w]); rows["cf"].append(a["cf"][w]); rows["cl"].append(a["cl"][w])
        rows["binding_f"].append((a["cf"][w] <= a["cl"][w]).astype(float))
        rows["ego_v"].append(a["ego_v"][w]); rows["dist_end"].append(END - a["pos"][w])
        rows["pinned"].append((a["pos"][w] >= END - 1e-3).astype(float))
        per_car.append((idx, sample, vid, True, len(w) * DT, float(np.nanmedian(t["c"][w][np.isfinite(t["c"][w])]))
                        if np.isfinite(t["c"][w]).any() else np.nan))
        sumo_dep.append(t["t_rec"])
    cat = {k: np.concatenate(v) if v else np.array([]) for k, v in rows.items()}
    print(f"  stuck cars that ever touch the lane end (pos>=75.44): {reach_end}; never in waiting phase "
          f"(within 10 m of end and v<2): {never_wait}")
    print(f"  SUMO departure time of the stuck cars' twins (s, p10/50/90): {qs(sumo_dep)}")
    n = cat["c"].size
    print(f"  waiting rows pooled: {n}  (cars {sum(1 for x in per_car if x[3])})")
    print(f"  ego speed (m/s) {qs(cat['ego_v'])}; distance to lane end (m) {qs(cat['dist_end'])}; pinned share "
          f"{np.mean(cat['pinned']):.2f}")
    print(f"  follower present share {np.mean(cat['hasf']):.2f};  v_f (m/s) {qs(cat['v_f'])}; gap_f (m) {qs(cat['gap_f'])}")
    print(f"  d_f required follower gap (m) {qs(cat['d_f'])};  c_f {qs(cat['cf'])}")
    print(f"  leader v_l {qs(cat['v_l'])}; gap_l {qs(cat['gap_l'])}; d_l {qs(cat['d_l'])}; c_l {qs(cat['cl'])}")
    fin = np.isfinite(cat['c'])
    print(f"  model c (finite rows share {fin.mean():.2f}): {qs(cat['c'])}; c>=0 share {np.mean(cat['c'] >= 0):.3f}; "
          f"c>=-0.5 share {np.mean(cat['c'] >= -0.5):.3f}; c>=-1 share {np.mean(cat['c'] >= -1):.3f}")
    print(f"  eligible share {np.mean(cat['elig']):.2f}; physically eligible share {np.mean(cat['phys']):.2f}; "
          f"reservation-veto share {np.mean(cat['veto']):.2f}")
    fin = np.isfinite(cat['cf']) & np.isfinite(cat['cl'])
    print(f"  binding side = follower share (rows with both finite) {np.mean(cat['binding_f'][fin]):.2f}")
    print(f"  integrated crossing cdf (hybrid) during waiting: {qs(cat['cdf'], (10, 50, 90))}")
    # kinematic term share of d_f
    kin_term = np.maximum(cat['v_f'] ** 2 - cat['ego_v'] ** 2, 0) / (2 * P0['b_f'])
    print(f"  follower closing-speed term (m) {qs(kin_term)};  share of d_f {np.nanmedian(kin_term / cat['d_f']):.2f}")
    # counterfactual: what gap would a car need / what gap is available
    ratio = cat['gap_f'] / cat['d_f']
    print(f"  gap_f / d_f {qs(ratio)}")
    # follower closing accel while waiting (nearest-follower speed change)
    accs = []
    for idx, sample, vid, t in stuck_cars:
        p = t.get("params", P0)
        a = car_rows(t, p)
        wait = a["mask"] & (a["lane"] == 2) & (a["pos"] >= END - 10.0) & (a["ego_v"] < 2.0)
        vf = t["v_f"]; gf = t["gap_f"]
        for i in np.flatnonzero(wait[:-1] & wait[1:]):
            if 0 < gf[i] < 80 and 0 < gf[i + 1] < 80 and abs(vf[i + 1] - vf[i]) < 5:
                accs.append((vf[i + 1] - vf[i]) / DT)
    print(f"  WM follower acceleration while a ramp car waits and follower within 80 m (m/s^2, p10/50/90): {qs(accs)}")
    return stuck_cars


def sumo_stats(path, var, stuck_only=False):
    """SUMO merge instants (recorded geometry) and follower behaviour while SUMO twins wait."""
    d = load(path)
    rr = [r for r in by_var(d)[var] if not r.get("error")]
    cs, vfs, gfs, evs, dfs, cls_, dls, gls, vls = [], [], [], [], [], [], [], [], []
    wm_c_at_cross = []
    wait_acc = []
    wait_vf, wait_gf = [], []
    seen = set()
    for r in rr:
        for vid, t in r["traces"].items():
            if stuck_only and not t["stuck"]:
                continue
            key = (r["idx"], vid)
            tr = t["t_rec"]
            if np.isfinite(tr) and key not in seen:     # SUMO twin identical across samples
                seen.add(key)
                row = int(round(tr / DT))
                if 0 < row < len(t["rec_v"]):
                    ev = t["rec_v"][row]
                    gf, vf, gl, vl = (t["rec_gap_f"][row], t["rec_v_f"][row], t["rec_gap_l"][row], t["rec_v_l"][row])
                    cf, df = cside(P0, "f", np.array([gf]), np.array([ev]), np.array([vf]), np.array([np.isfinite(gf)]))
                    cl, dl = cside(P0, "l", np.array([gl]), np.array([ev]), np.array([vl]), np.array([np.isfinite(gl)]))
                    cs.append(float(min(cf[0], cl[0]))); vfs.append(vf); gfs.append(gf); evs.append(ev)
                    dfs.append(df[0]); cls_.append(cl[0]); dls.append(dl[0]); gls.append(gl); vls.append(vl)
                # SUMO twin waiting rows (ego slow near the end, lane 2)
                rp = t["rec_mask"]
                for i in range(1, len(t["rec_v"]) - 1):
                    if (rp[i] and rp[i + 1] and t["rec_lane"][i] == 2 and t["rec_lane"][i + 1] == 2
                            and t["rec_pos"][i] >= END - 10 and t["rec_v"][i] < 2.0):
                        g0, g1 = t["rec_gap_f"][i], t["rec_gap_f"][i + 1]
                        if 0 < g0 < 80 and 0 < g1 < 80:
                            a = (t["rec_v_f"][i + 1] - t["rec_v_f"][i]) / DT
                            if abs(a) < 10:
                                wait_acc.append(a)
                            wait_vf.append(t["rec_v_f"][i]); wait_gf.append(g0)
    print(f"[{var}] SUMO merge instants of {'stuck-twin ' if stuck_only else ''}ramp cars: n={len(cs)}")
    cs = np.array(cs)
    print(f"  SUMO-implied boundary c at its own merge row (r9 params): {qs(cs, (10,25,50,75,90))}; "
          f"share c>=0 {np.mean(cs >= 0):.2f}; c>=-0.5 {np.mean(cs >= -0.5):.2f}; c>=-1 {np.mean(cs >= -1):.2f}; "
          f"c=-inf {np.mean(~np.isfinite(cs)):.2f}")
    print(f"  at SUMO merge: ego v {qs(evs)}; v_f {qs(vfs)}; gap_f {qs(gfs)}; d_f {qs(dfs)}; "
          f"v_l {qs(vls)}; gap_l {qs(gls)}; d_l {qs(dls)}")
    print(f"  SUMO follower while a SUMO ramp car waits (v<2, <=10 m of end): v_f {qs(wait_vf)}; gap_f {qs(wait_gf)}; "
          f"follower accel (m/s^2) {qs(wait_acc)}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "table":
        table(sys.argv[2])
    elif cmd == "wait":
        wait_stats(sys.argv[2], sys.argv[3])
    elif cmd == "sumo":
        sumo_stats(sys.argv[2], sys.argv[3], stuck_only=len(sys.argv) > 4 and sys.argv[4] == "stuck")
