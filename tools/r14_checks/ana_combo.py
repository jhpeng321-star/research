# ruff: noqa: E741, F841  (verbatim copy of the COMBO300 / LONGDIAG analysis code)
"""S-COMBO300 analysis of compact 1 s rollouts (combo_roll.py output; SUMO stream = pNN_{ref,act}_sumo.npz).
usage: python ana_combo.py DIR [--arms base,E,Egs,Egsy,y]  -> prints markdown tables; writes DIR/ana_tables.json"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mech  # noqa: E402

M1, UPN, RAMP_MIN, ROAD, FREE = 75.44, -800.0, -378.66, 553.33, 31.29
LV = 5.0
WIN = [(0, 120), (120, 240), (240, 300)]
rng = np.random.default_rng(7)


def load(path):
    d = np.load(path, allow_pickle=False)
    S, L, V = d["S"].astype(float), d["L"].astype(int), d["V"].astype(float)
    ids = [str(x) for x in d["ids"]]
    cav = d["cav"].astype(int).copy()
    ro = {str(x): bool(c) for x, c in zip(d["roster"], d["roster_cav"])}
    for j, i in enumerate(ids):
        if i in ro:
            cav[j] = int(ro[i])
    return dict(S=S, L=L, V=V, ids=ids, cav=cav, roster=set(ro), roster_cav=ro,
                extra={k: float(d[k]) for k in ("gate_elig", "gate_veto", "yield_n", "sec") if k in d.files})


def lane_pairs(S, L, V, t):
    """consecutive same-lane (follower, leader) index pairs at row t on the road."""
    out = []
    ok = np.isfinite(S[t]) & (S[t] < ROAD)
    for ln in (0, 1, 2):
        idx = np.flatnonzero(ok & (L[t] == ln))
        if len(idx) > 1:
            o = idx[np.argsort(S[t, idx])]
            out.append(np.stack([o[:-1], o[1:]], 1))
    return np.concatenate(out) if out else np.zeros((0, 2), int)


def run_metrics(R):
    S, L, V, cav = R["S"], R["L"], R["V"], R["cav"]
    T, N = S.shape
    fin = np.isfinite(S)
    m = {}
    reg = mech.regions(S, L)
    dl = np.where(np.isfinite(V), np.maximum(0.0, 1.0 - V / FREE), 0.0)
    Z1 = fin & (L == 1) & (S >= 0) & (S < M1)
    occ, std = Z1.sum(1), (Z1 & (V < 2)).sum(1)
    for k, (a, b) in enumerate(WIN):
        m[f"occ_{k}"] = float(occ[a:b].mean())
        m[f"zv_{k}"] = float(np.nanmean(np.where(Z1[a:b], V[a:b], np.nan))) if Z1[a:b].any() else np.nan
        m[f"std_{k}"] = float(std[a:b].mean())
    m["occ"] = float(occ[60:].mean())
    m["std"] = float(std[60:].mean())
    sv = np.where(Z1 & (V < 2), S, np.nan)
    hd = np.nanmax(sv, 1) if np.isfinite(sv).any() else np.full(T, np.nan)
    m["std_head"] = float(np.nanmedian(hd[60:])) if np.isfinite(hd[60:]).any() else np.nan
    m["zv"] = float(np.nanmean(np.where(Z1[60:], V[60:], np.nan))) if Z1[60:].any() else np.nan
    # delay by region, cumulative at 120/240/300
    for r in ("up_near", "ramp", "merge", "downstream"):
        for tt in (120, 240, 300):
            m[f"dly_{r}_{tt}"] = float((dl[:tt] * reg[r][:tt]).sum())
    # stuck ramp roster cars; ramp queue
    ro = [j for j, i in enumerate(R["ids"]) if i in R["roster"] and fin[0, j] and L[0, j] == 2 and cav[j] != 1]
    stuck = 0
    for j in ro:
        mg = (np.isin(L[:, j], (0, 1)) & fin[:, j]).any()
        stuck += (not mg)
    m["n_ramp0"] = len(ro)
    m["stuck"] = stuck
    rq = fin & (L == 2) & (S >= RAMP_MIN) & (S < M1) & (V < 2)
    m["ramp_q"] = float(rq[240:300].sum(1).mean())
    # merges per window (lane 2 -> 0/1) and by target lane
    a, b = L[:-1], L[1:]
    mg = (a == 2) & np.isin(b, (0, 1)) & fin[:-1] & fin[1:]
    mgc = mg.sum(1)
    for k, (x, y) in enumerate(WIN):
        m[f"merge_{k}"] = float(mgc[x:y].sum())
    m["merge"] = float(mgc.sum())
    # lane changes 1->0 / 0->1 on the approach (-200..M1)
    lc10 = ((a == 1) & (b == 0) & fin[:-1] & fin[1:] & (S[:-1] > -200) & (S[:-1] < M1)).sum(1)
    lc01 = ((a == 0) & (b == 1) & fin[:-1] & fin[1:] & (S[:-1] > -200) & (S[:-1] < M1)).sum(1)
    m["lc10"], m["lc01"] = float(lc10.sum()), float(lc01.sum())
    m["lc10_b"] = float(lc10[:, ].sum())
    # TTC<1.5 (gap>=0) and overlap pair-seconds
    ttc = ov = 0
    for t in range(T):
        P = lane_pairs(S, L, V, t)
        if not len(P):
            continue
        f, l = P[:, 0], P[:, 1]
        gap = S[t, l] - S[t, f] - LV
        clos = V[t, f] - V[t, l]
        ov += int((gap < 0).sum())
        ttc += int(((gap >= 0) & (clos > 0) & (gap < 1.5 * np.maximum(clos, 1e-9))).sum())
    m["ttc"], m["overlap"] = float(ttc), float(ov)
    # post-insertion hard brake (HDV inserter, HDV follower; amin over 6 s after, 1 s grid)
    ti, ki = np.nonzero((a == 2) & (b == 1) & fin[:-1] & fin[1:])
    hb = n = 0
    hbi = ni = 0
    gaps = []
    for t0, r in zip(ti, ki):
        tm = t0 + 1
        if tm + 6 >= T or cav[r] == 1:
            continue
        on1 = fin[tm] & (L[tm] == 1)
        on1[r] = False
        beh = np.flatnonzero(on1 & (S[tm] <= S[tm, r]))
        if not len(beh):
            continue
        f = beh[np.argmax(S[tm, beh])]
        gaps.append(S[tm, r] - LV - S[tm, f])
        if cav[f] == 1:
            continue
        w = np.arange(tm, tm + 7)
        if not fin[w, f].all():
            continue
        amin = np.diff(V[w, f]).min()
        n += 1
        hb += int(amin < -3)
    m["hb_n"], m["hb"] = int(n), int(hb)
    m["ins_gap_med"] = float(np.median(gaps)) if gaps else np.nan
    # lane-1 approach speed, free-road crawlers, ramp creep at lane end
    ap = fin & (L == 1) & (S >= -100) & (S < 0)
    m["appr_v"] = float(np.nanmean(np.where(ap[60:], V[60:], np.nan))) if ap[60:].any() else np.nan
    m["appr_p10"] = float(np.nanpercentile(V[60:][ap[60:]], 10)) if ap[60:].any() else np.nan
    crawl = 0
    for t in range(60, T):
        idx = np.flatnonzero(Z1[t] & (V[t] < 3))
        if not len(idx):
            continue
        l1 = np.flatnonzero(fin[t] & (L[t] == 1))
        for j in idx:
            ahead = S[t, l1][S[t, l1] > S[t, j]]
            gp = (ahead.min() - S[t, j] - LV) if len(ahead) else 1e3
            crawl += gp > 40
    m["crawl_vs"] = float(crawl)
    rl = fin & (L == 2) & (S >= 45) & (S < M1)
    m["rampend_n"] = float(rl[60:].sum() / max(1, T - 60))
    m["rampend_stop"] = float((rl & (V < 0.5))[60:].sum() / max(1, rl[60:].sum()))
    m["rampend_v"] = float(np.nanmean(np.where(rl[60:], V[60:], np.nan))) if rl[60:].any() else np.nan
    # waiting time of ramp roster cars = first merge time
    wt = []
    for j in ro:
        ok = np.flatnonzero(np.isin(L[:, j], (0, 1)) & fin[:, j])
        if len(ok):
            wt.append(ok[0])
    m["merge_t_med"] = float(np.median(wt)) if wt else np.nan
    # spillback etc from mech
    mm = mech.metrics_arrays(R["S"], R["L"], R["V"]) if hasattr(mech, "metrics_arrays") else None
    if mm is not None:
        for i in range(5):
            m[f"spill_{i}"] = mm[i]["spill"]
            m[f"spillmax_{i}"] = mm[i]["spill_max"]
            m[f"slowup_{i}"] = mm[i]["slow_up_near"]
            m[f"qm1queued_{i}"] = mm[i]["q_m1_queued"]
        m["spill_240"] = float(np.mean([mm[3]["spill"], mm[4]["spill"]]))
        m["spillmax_end"] = float(mm[4]["spill_max"])
    return m


def jam_ds(W, Su):
    """roster HDVs standing (v<3) in SUMO on the mainline approach/merge in 60..120 s: signed WM-SUMO position at 120/240 s."""
    out = {}
    common = [(i, Su["ids"].index(x)) for i, x in enumerate(W["ids"]) if x in Su["roster"] and x in Su["ids"]]
    if not common:
        return out
    iw = np.array([c[0] for c in common])
    isu = np.array([c[1] for c in common])
    SS, VS, LS = Su["S"][:, isu], Su["V"][:, isu], Su["L"][:, isu]
    hdv = Su["cav"][isu] != 1
    jam = hdv & ((np.isfinite(SS[60:121]) & (VS[60:121] < 3) & np.isin(LS[60:121], (0, 1)) & (SS[60:121] > -150)
                  & (SS[60:121] < M1)).any(0))
    for tt in (120, 240):
        ok = jam & np.isfinite(SS[tt]) & np.isfinite(W["S"][tt, iw])
        d = W["S"][tt, iw][ok] - SS[tt][ok]
        out[tt] = d
    out["n_jam"] = int(jam.sum())
    return out


def collect(d):
    files = glob.glob(os.path.join(d, "p*_*.npz"))
    runs, sumo = {}, {}
    for f in sorted(files):
        mt = re.match(r"p(\d+)_(ref|act)_(\w+?)(?:_k(\d+))?\.npz$", os.path.basename(f))
        if not mt:
            continue
        idx, tag, var = int(mt.group(1)), mt.group(2), mt.group(3)
        if var == "sumo":
            sumo[(idx, tag)] = f
        else:
            runs[(idx, tag, var)] = f
    return runs, sumo


def boot(vals, n=4000):
    """cluster bootstrap over rows (each row = one cluster mean); returns mean, lo, hi."""
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    bs = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)])
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def fmt(t, d=2):
    m, lo, hi = t
    return "n/a" if not np.isfinite(m) else f"{m:.{d}f} [{lo:.{d}f},{hi:.{d}f}]"


def main():
    d = sys.argv[1]
    arms = ("base", "E", "Egs", "Egsy", "y")
    if "--arms" in sys.argv:
        arms = tuple(sys.argv[sys.argv.index("--arms") + 1].split(","))
    TAGS = ("ref",) if "--ref-only" in sys.argv else ("ref", "act")
    runs, sumo = collect(d)
    cache = os.path.join(d, "ana_cache.json")
    cc = json.load(open(cache)) if os.path.exists(cache) else {}
    M, SM, JD = {}, {}, {}
    SU = {}
    for k, f in sumo.items():
        SU[k] = load(f)
        SM[k] = run_metrics(SU[k])
    for k, f in runs.items():
        key = f"{k}|{os.path.getmtime(f)}"
        R = load(f)
        if key in cc:
            M[k] = cc[key]
        else:
            M[k] = run_metrics(R)
            cc[key] = M[k]
        M[k]["_x"] = R["extra"]
        if (k[0], k[1]) in SU:
            JD[k] = jam_ds(R, SU[(k[0], k[1])])
    json.dump(cc, open(cache, "w"))
    pairs = sorted({k[0] for k in runs})
    complete = [p for p in pairs if all((p, t, a) in M for t in TAGS for a in arms) and all((p, t) in SM for t in TAGS)]
    print(f"pairs with data: {pairs}; complete across all arms/both tags: {complete}")
    # (a) state-level table over runs, cluster = pair
    keys = ["occ", "zv", "std", "std_head", "stuck", "ramp_q", "merge_0", "merge_1", "merge_2", "ins_gap_med", "hb", "hb_n", "ttc",
            "overlap", "appr_v", "crawl_vs", "lc10", "rampend_stop", "rampend_v", "merge_t_med",
            "dly_up_near_120", "dly_up_near_240", "dly_up_near_300", "dly_merge_120", "dly_merge_240", "dly_merge_300",
            "dly_ramp_120", "dly_ramp_240", "dly_ramp_300", "dly_downstream_120", "dly_downstream_240", "dly_downstream_300",
            "spill_240", "spillmax_end", "occ_0", "occ_1", "occ_2", "zv_0", "zv_1", "zv_2", "std_0", "std_1", "std_2"]
    use = complete or pairs
    tab = {}
    for arm in ("sumo",) + arms:
        tab[arm] = {}
        for k in keys:
            per_pair = []
            for p in use:
                v = []
                for tg in TAGS:
                    mm = SM.get((p, tg)) if arm == "sumo" else M.get((p, tg, arm))
                    if mm is not None and k in mm and np.isfinite(mm[k]):
                        v.append(mm[k])
                if v:
                    per_pair.append(np.mean(v))
            tab[arm][k] = per_pair
    # markdown
    print("\n## (a) means over pairs (ref+act pooled per pair; n pairs = %d), absolute, with 95%% cluster CI" % len(use))
    cols = ["sumo"] + list(arms)
    print("| metric | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for k in keys:
        print(f"| {k} | " + " | ".join(fmt(boot(tab[a][k]), 2 if k not in ("ttc", "overlap", "hb", "hb_n") else 1) for a in cols) + " |")
    print("\n## (a) paired differences arm-base and arm-SUMO (same pairs)")
    print("| metric | " + " | ".join(f"{a}-"+arms[0]+"" for a in arms[1:]) + " | " + " | ".join(f"{a}-SUMO" for a in arms) + " |")
    print("|---|" + "---|" * (2 * len(arms) - 1 + 1 - 1))
    for k in keys:
        row = []
        for a in arms[1:]:
            row.append(fmt(boot(np.array(tab[a][k]) - np.array(tab[arms[0]][k])), 2))
        for a in arms:
            row.append(fmt(boot(np.array(tab[a][k]) - np.array(tab["sumo"][k])), 2))
        print(f"| {k} | " + " | ".join(row) + " |")
    # hard-brake share, ratios
    print("\n## hard-brake share (pooled events) and zone-occupancy ratio")
    for a in cols:
        hb = sum(np.nansum(tab[a]["hb"]) for _ in [0])
        nn = sum(np.nansum(tab[a]["hb_n"]) for _ in [0])
        print(f"{a}: hb {hb:.1f} of {nn:.1f} (pair-mean units) -> {100 * hb / max(nn, 1e-9):.1f}%; occ ratio to SUMO {np.mean(tab[a]['occ']) / np.mean(tab['sumo']['occ']):.2f}; "
              f"std ratio {np.mean(tab[a]['std']) / max(np.mean(tab['sumo']['std']), 1e-9):.2f}")
    print("\n## gate / yield activity (sum over runs): gate eligible, vetoed; yield-modified entries")
    for a in arms:
        ge = sum(M[(p, t, a)]["_x"].get("gate_elig", 0) for p in use for t in TAGS if (p, t, a) in M)
        gv = sum(M[(p, t, a)]["_x"].get("gate_veto", 0) for p in use for t in TAGS if (p, t, a) in M)
        yn = sum(M[(p, t, a)]["_x"].get("yield_n", 0) for p in use for t in TAGS if (p, t, a) in M)
        sec = np.mean([M[(p, t, a)]["_x"].get("sec", np.nan) for p in use for t in ("ref", "act") if (p, t, a) in M])
        print(f"{a}: elig {ge:.0f} veto {gv:.0f} ({100 * gv / max(ge, 1):.0f}%) yield {yn:.0f}; mean rollout {sec:.0f} s")
    # jam-held survivors
    print("\n## jam-held survivors: signed WM-SUMO position (m), median over runs' jam-held roster cars; share ahead >100 m")
    for a in arms:
        row = []
        for tt in (120, 240):
            dd = np.concatenate([JD[(p, t, a)][tt] for p in use for t in TAGS if (p, t, a) in JD and tt in JD[(p, t, a)]] or [np.zeros(0)])
            row.append(f"{tt}s: n={len(dd)} med {np.median(dd):+.0f} ahead>100 {100 * np.mean(dd > 100):.0f}%" if len(dd) else f"{tt}s: n=0")
        print(f"{a}: " + "; ".join(row))
    if "--ref-only" in sys.argv:
        return
    # (b) action - native
    print("\n## (b) action-minus-native (random-switching arm minus native), per pair, WM vs SUMO")
    bk = ["spill_240", "spillmax_end", "dly_up_near_300", "dly_merge_300", "dly_ramp_300", "dly_downstream_300", "dly_up_near_240",
          "merge_2", "occ_2", "zv_2", "std_2"]
    print("| arm | metric | WM mean diff [CI] | SUMO mean diff [CI] | same sign (n) | WM-SUMO diff of diffs [CI] | r |")
    print("|---|---|---|---|---|---|---|")
    bres = {}
    for a in arms:
        for k in bk:
            w, s = [], []
            for p in use:
                if (p, "act", a) in M and (p, "ref", a) in M and (p, "act") in SM and (p, "ref") in SM:
                    w.append(M[(p, "act", a)][k] - M[(p, "ref", a)][k])
                    s.append(SM[(p, "act")][k] - SM[(p, "ref")][k])
            w, s = np.array(w), np.array(s)
            if len(w) < 2:
                continue
            ss = int((np.sign(w) == np.sign(s)).sum())
            r = float(np.corrcoef(w, s)[0, 1]) if w.std() > 0 and s.std() > 0 else np.nan
            bres[(a, k)] = (w, s)
            print(f"| {a} | {k} | {fmt(boot(w), 1)} | {fmt(boot(s), 1)} | {ss}/{len(w)} | {fmt(boot(w - s), 1)} | {r:.2f} |")
    json.dump({"pairs": use, "arms": list(arms)}, open(os.path.join(d, "ana_tables.json"), "w"))


if __name__ == "__main__":
    main()
