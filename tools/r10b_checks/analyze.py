"""R10B checks: analysis + VERDICT_R10B_CHECKS.md.  numpy only.

  python analyze.py --shm /dev/shm/chk_r10b --out VERDICT_R10B_CHECKS.md [--m5 SUMMARY_R10B.md] [--meta meta.json]
Reads  <shm>/roll/<arm>/*.npz (run_rollouts), <shm>/remove/remove_<arm>.npz (run_remove), <shm>/os/onestep_<arm>.npz (run_onestep),
and optional <shm>/gates.json.  Thresholds are the registered ones (REGISTRATION.md, ROUND 10b REGISTERED REJECT CRITERIA)."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import ARMS, decode_pair_inputs, groups_of  # noqa: E402

REF_ARMS = ("sumo",) + ARMS
C = "ep t vid k cav s v a gapL dvL vL hasL dA vR aR waitR dendR tf nRamp dB nzone pen dem vidR lane".split()
OF = np.arange(-40, 25)
M1_T = (-3.0, -2.5, -2.0, -1.5, -1.0, -0.5)
TH = {"M1": -0.9, "M2_far10": -0.5, "M3": 0.3, "M4_val": -3.0, "M4_share": 0.05}
RNG = np.random.default_rng(0)


# ------------------------------------------------------------------------------------------------ helpers
def boot_mean(vals, groups, n=300):
    vals = np.asarray(vals, float)
    groups = np.asarray(groups)
    ok = np.isfinite(vals)
    vals, groups = vals[ok], groups[ok]
    if len(vals) < 5:
        return float("nan"), float("nan"), float("nan"), len(vals)
    ug, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, vals)
    cnt = np.bincount(inv).astype(float)
    out = []
    for _ in range(n):
        pick = RNG.integers(0, len(ug), len(ug))
        out.append(sums[pick].sum() / max(cnt[pick].sum(), 1))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(vals.mean()), float(lo), float(hi), len(vals)


def fmt(m, lo=None, hi=None, n=None, p=2):
    if not np.isfinite(m):
        return "n/a" + (f" (n={n})" if n is not None else "")
    s = f"{m:+.{p}f}"
    if lo is not None and np.isfinite(lo):
        s += f" [{lo:+.{p}f}, {hi:+.{p}f}]"
    if n is not None:
        s += f" (n={n})"
    return s


def load_arm_rolls(shm, arm):
    files = sorted(glob.glob(os.path.join(shm, "roll", arm, "*.npz")))
    if not files:
        return None
    rows, evs, mes, heads, jsw, jall, misc = [], [], [], [], [], [], []
    for f in files:
        z = np.load(f)
        rows.append(z["rows"])
        evs.append(z["ev"])
        mes.append(z["me"])
        h = z["head"]
        sid = int(re.search(r"s(\d+)_k", f).group(1))
        heads.append(np.column_stack([h, np.full(len(h), sid)]) if len(h) else np.zeros((0, 17), np.float32))
        jsw.append(z["jerk_switch"])
        jall.append((z["jerk_all"], int(z["n_jerk_all"])))
        misc.append(z["misc"])
    cat = lambda xs, shape: np.concatenate([x for x in xs if len(x)]) if any(len(x) for x in xs) else np.zeros(shape)  # noqa: E731
    t = cat(rows, (0, 25)).astype(np.float64)
    ev = cat(evs, (0, 65, 13))
    me = cat(mes, (0, 8))
    return {"t": {c: t[:, i] for i, c in enumerate(C)}, "ev": ev, "me": me, "head": cat(heads, (0, 17)),
            "jerk_sw": cat(jsw, (0,)), "jerk_all": jall, "misc": np.array(misc), "n_runs": len(files)}


def cells_fn():
    import diag_yield_common as dc
    return dc


def own3_model(dc, t):
    c = dc.cells(t)
    h = (t["cav"] == 0) & (t["lane"] == 1) & (t["nzone"] > 0) & (t["tf"] == 0) & ~(t["dA"] < 30) & ~(t["dB"] < 20)
    u, inv = np.unique(c, return_inverse=True)
    n = np.bincount(inv[h], minlength=len(u))
    mu = np.bincount(inv[h], t["a"][h], minlength=len(u)) / np.maximum(n, 1)
    return u, np.where(n >= 10, mu, np.nan)


# ------------------------------------------------------------------------------------------------ M1
def m1(rolls, dc):
    res = {}
    for arm in REF_ARMS:
        R = rolls.get(arm)
        if R is None or len(R["ev"]) == 0:
            continue
        model = own3_model(dc, R["t"])
        ev, me = R["ev"], R["me"]
        cid = dc.event_cells(ev, np.nan_to_num(me[:, 7], nan=2400.0))
        exe = ev[..., 2] - dc.lookup(*model, cid)
        hdv = me[:, 4] == 0
        prof, cols = {}, []
        for o in (-6, -4, -3, -2, -1.5, -1, -0.5, 0, 1, 2):
            oi = list(OF).index(int(round(o * 2)))
            ok = hdv & (ev[:, oi, 3] == 1) & np.isfinite(exe[:, oi])
            prof[o] = boot_mean(exe[ok, oi], me[ok, 0], 200)
        oi_set = [list(OF).index(int(round(o * 2))) for o in M1_T]
        vals, grp = [], []
        for oi in oi_set:
            ok = hdv & (ev[:, oi, 3] == 1) & np.isfinite(exe[:, oi])
            vals.append(exe[ok, oi])
            grp.append(me[ok, 0])
        mean3 = boot_mean(np.concatenate(vals), np.concatenate(grp), 300)
        res[arm] = {"profile": prof, "mean_m3_0": mean3, "n_events": int(hdv.sum())}
    return res


# ------------------------------------------------------------------------------------------------ M2
def m2(shm):
    res = {}
    for arm in ARMS:
        f = os.path.join(shm, "remove", f"remove_{arm}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        out = {}
        for kind in ("near", "far"):
            sel = z["kind"] == kind
            row = {}
            for tsec in (3, 6, 10, 20):
                i = int(tsec * 2)
                if z["dv"].shape[1] <= i:
                    continue
                x = np.where(z["ok"][sel, i], z["dv"][sel, i], np.nan)
                row[tsec] = boot_mean(x, z["state"][sel], 300)
            i10 = 20
            x10 = np.where(z["ok"][sel, i10], z["dv"][sel, i10], np.nan) if z["dv"].shape[1] > i10 else np.array([])
            row["share<-0.5"] = float(np.nanmean(x10 < -0.5)) if len(x10) else float("nan")
            row["n"] = int(sel.sum())
            out[kind] = row
        res[arm] = out
    return res


# ------------------------------------------------------------------------------------------------ M3 / tails (one-step)
def load_onestep(shm, arm):
    f = os.path.join(shm, "os", f"onestep_{arm}.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f)
    cols = list(z["cols"])
    r = z["rows"]
    d = {c: r[:, i] for i, c in enumerate(cols)}
    ev = json.load(open(os.path.join(shm, "os", f"onestep_{arm}_events.json")))
    return d, ev


def onestep_groups(d):
    has = d["has"] == 1
    return {
        "dA>30 m (far partner)": has & (d["dA"] > 30),
        "d_end>40 m": has & (d["dend"] > 40),
        "alongside (bumper gap<1 m)": has & (d["bgap"] < 1.0),
        "ramp car behind (no partner)": (d["rampbehind"] == 1),
        "lane 0": d["lane"] == 0,
    }


def m3_onestep(os_data):
    out = {}
    for arm, v in os_data.items():
        if v is None:
            continue
        d, _ = v
        g = onestep_groups(d)
        row = {}
        for name, sel in g.items():
            bias = (d["fin"] - d["rec"])[sel]
            b = boot_mean(bias, d["win"][sel], 200)
            row[name] = {"bias": b, "head_mean": float(np.mean(d["off"][sel])) if sel.any() else float("nan"),
                         "max_abs_head": float(np.abs(d["off"][sel]).max()) if sel.any() else float("nan"), "n": int(sel.sum())}
        nonpartner = d["has"] == 0
        row["_max_abs_head_nonpartner"] = float(np.abs(d["off"][nonpartner]).max()) if nonpartner.any() else 0.0
        out[arm] = row
    return out


def tails_onestep(os_data):
    out = {}
    for arm, v in os_data.items():
        if v is None:
            continue
        d, evb = v
        row = {}
        evm = d["ev"] >= 0
        key = d["win"][evm] * 10000 + d["ev"][evm]
        fin, rec = d["fin"][evm], d["rec"][evm]
        ks = np.unique(key)
        mins_m = np.array([fin[key == k].min() for k in ks])
        mins_r = np.array([rec[key == k].min() for k in ks])
        row["post_min_p10_model"] = float(np.percentile(mins_m, 10)) if len(ks) else float("nan")
        row["post_min_p10_rec"] = float(np.percentile(mins_r, 10)) if len(ks) else float("nan")
        row["post_min_median_model"] = float(np.median(mins_m)) if len(ks) else float("nan")
        row["post_min_median_rec"] = float(np.median(mins_r)) if len(ks) else float("nan")
        row["n_events"] = int(len(ks))
        on = d["onset"] == 1
        brake = np.array([evb.get(str(int(w)), {}).get(str(int(e)), 0.0) < -1.0 for w, e in zip(d["win"][on], d["ev"][on])])
        bias = (d["fin"] - d["rec"])[on]
        row["onset_n"] = int(on.sum())
        row["onset_bias_brake"] = boot_mean(bias[brake], d["win"][on][brake], 200) if brake.any() else (float("nan"),) * 4
        row["onset_bias_nobrake"] = boot_mean(bias[~brake], d["win"][on][~brake], 200) if (~brake).any() else (float("nan"),) * 4
        tight = (d["ctrl"] == 1) & (d["lane"] == 1) & (d["gapL"] < 8.6)
        row["tight_n"] = int(tight.sum())
        row["tight_p10_model"] = float(np.percentile(d["fin"][tight], 10)) if tight.any() else float("nan")
        row["tight_p10_rec"] = float(np.percentile(d["rec"][tight], 10)) if tight.any() else float("nan")
        row["tight_bias"] = boot_mean((d["fin"] - d["rec"])[tight], d["win"][tight], 200) if tight.any() else (float("nan"),) * 4
        out[arm] = row
    return out


# ------------------------------------------------------------------------------------------------ M4 + rollout safety/tails
def m4(rolls):
    R = rolls.get("r10b")
    out = {}
    if R is None or len(R["head"]) == 0:
        return out
    h = R["head"]
    off = h[:, 2]
    dec = decode_pair_inputs(h[:, 6:16])
    for name, sel in groups_of(dec).items():
        x = off[sel]
        if len(x) == 0:
            continue
        out[name] = {"n": int(len(x)), "p0.1": float(np.percentile(x, 0.1)), "p1": float(np.percentile(x, 1)),
                     "p5": float(np.percentile(x, 5)), "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)),
                     "p99.9": float(np.percentile(x, 99.9)), "share<-3": float(np.mean(x < -3.0)), "mean": float(x.mean())}
    return out


def m3_rollout(rolls):
    R = rolls.get("r10b")
    out = {}
    if R is None or len(R["head"]) == 0:
        return out
    h = R["head"]
    dec = decode_pair_inputs(h[:, 6:16])
    g = groups_of(dec)
    for name in ("dA>30 (far)", "d_end>40", "alongside (bumper gap<1)"):
        x = h[g[name], 2]
        out[name] = boot_mean(x, h[g[name], 16], 200)
    out["_max_abs_offset_non_partner_rows"] = float(R["misc"][:, 0].max())
    return out


def rollout_safety(rolls):
    out = {}
    for arm in REF_ARMS:
        R = rolls.get(arm)
        if R is None:
            continue
        t = R["t"]
        hdv1 = (t["cav"] == 0) & (t["lane"] == 1)
        n_rows = int(hdv1.sum())
        has = hdv1 & (t["hasL"] == 1)
        clos = -t["dvL"]
        ttc = np.where(has & (clos > 1e-3), t["gapL"] / np.maximum(clos, 1e-3), np.inf)
        row = {"n_rows": n_rows,
               "ttc<1.5": float(np.mean((ttc < 1.5)[hdv1])) if n_rows else float("nan"),
               "overlap(gap<0)": float(np.mean((t["gapL"] < 0)[has])) if has.any() else float("nan"),
               "hard_brake(a<-3)": float(np.mean((t["a"] < -3.0)[hdv1])) if n_rows else float("nan"),
               "a_p1": float(np.percentile(t["a"][hdv1], 1)) if n_rows else float("nan")}
        # per-state counts for paired bootstrap
        row["_state"] = t["ep"][hdv1]
        row["_ttc"] = (ttc < 1.5)[hdv1].astype(float)
        row["_ovl"] = np.where(has, (t["gapL"] < 0), False)[hdv1].astype(float)
        row["_hb"] = (t["a"] < -3.0)[hdv1].astype(float)
        ev, me = R["ev"], R["me"]
        hd = me[:, 4] == 0 if len(me) else np.zeros(0, bool)
        if len(ev):
            i0 = list(OF).index(0)
            gap_ins = ev[hd, i0, 11]
            row["min_gap_ins_p5"] = float(np.nanpercentile(gap_ins, 5)) if np.isfinite(gap_ins).any() else float("nan")
            row["min_gap_ins_median"] = float(np.nanmedian(gap_ins)) if np.isfinite(gap_ins).any() else float("nan")
            post = ev[hd][:, i0:i0 + 13, 2]
            ok = ev[hd][:, i0:i0 + 13, 3] == 1
            mins = np.nanmin(np.where(ok, post, np.nan), axis=1) if len(post) else np.array([])
            row["post_ins_min_accel_p10"] = float(np.nanpercentile(mins, 10)) if np.isfinite(mins).any() else float("nan")
            row["post_ins_min_accel_median"] = float(np.nanmedian(mins)) if np.isfinite(mins).any() else float("nan")
            row["n_merge_events_hdv_F"] = int(hd.sum())
        js = R["jerk_sw"]
        row["jerk_switch_p50"] = float(np.median(js)) if len(js) else float("nan")
        row["jerk_switch_p95"] = float(np.percentile(js, 95)) if len(js) else float("nan")
        row["n_switch"] = int(len(js))
        if R["jerk_all"]:
            pa = np.array([j[0] for j in R["jerk_all"]])
            wn = np.array([j[1] for j in R["jerk_all"]], float)
            row["jerk_all_p95"] = float(np.nansum(pa[:, 1] * wn) / max(wn.sum(), 1))
        row["cav_replay_err_m"] = float(np.nanmean(R["misc"][:, 3])) if R["misc"].size else float("nan")
        out[arm] = row
    return out


def paired_diff(a, b, key, n=300):
    """state-cluster bootstrap CI of mean(a) - mean(b) for per-row indicator arrays keyed by state."""
    sa, sb = a["_state"], b["_state"]
    xa, xb = a[key], b[key]
    ug = np.unique(np.concatenate([sa, sb]))
    ia = {u: i for i, u in enumerate(ug)}
    ca = np.zeros(len(ug)); na = np.zeros(len(ug)); cb = np.zeros(len(ug)); nb = np.zeros(len(ug))
    np.add.at(ca, [ia[u] for u in sa], xa); np.add.at(na, [ia[u] for u in sa], 1)
    np.add.at(cb, [ia[u] for u in sb], xb); np.add.at(nb, [ia[u] for u in sb], 1)
    est = ca.sum() / max(na.sum(), 1) - cb.sum() / max(nb.sum(), 1)
    out = []
    for _ in range(n):
        p = RNG.integers(0, len(ug), len(ug))
        out.append(ca[p].sum() / max(na[p].sum(), 1) - cb[p].sum() / max(nb[p].sum(), 1))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(est), float(lo), float(hi)


# ------------------------------------------------------------------------------------------------ report
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--m5", default=None)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--run", default=None, help="round-10b run dir (M7 reads train_log.jsonl)")
    a = ap.parse_args()
    sys.path.insert(0, a.repo + "/tools")
    dc = cells_fn()
    meta = json.load(open(a.meta)) if a.meta and os.path.exists(a.meta) else {}
    rolls = {arm: load_arm_rolls(a.shm, arm) for arm in REF_ARMS}
    rolls = {k: v for k, v in rolls.items() if v}
    os_data = {arm: load_onestep(a.shm, arm) for arm in ARMS}
    gates = json.load(open(os.path.join(a.shm, "gates.json"))) if os.path.exists(os.path.join(a.shm, "gates.json")) else {}

    M1 = m1(rolls, dc)
    M2 = m2(a.shm)
    M3o = m3_onestep(os_data)
    M3r = m3_rollout(rolls)
    M4 = m4(rolls)
    TL = tails_onestep(os_data)
    SF = rollout_safety(rolls)

    L = []
    verdict = {}
    L.append("# VERDICT_R10B_CHECKS - round 10b reviewer reject checks (M1-M4, fix-B tails)\n")
    L.append(f"Generated by tools/r10b_checks (S-R10BCHK). {json.dumps(meta)}\n")
    L.append("Arms: r9 = round 9 stage300 (current code); r10b = round 10b final; headzero = r10b with follower_courtesy.net.4.* zeroed. "
             "`sumo` = the recorded future of the same TRAIN t0 rosters. TRAIN states only; CPU; no sealed data.\n")
    # M1
    L.append("## M1 rollout event profile of the target follower (C3-own matched excess, m/s^2)\n")
    L.append("Reference SUMO TRAIN: onset ~-0.14 at -6 s, peak ~-0.45 at -1.5 s, ~0 at the merge. Rule: ZERO-HEAD if mean over [-3,0) s < -0.9.\n")
    L.append("| arm | mean [-3,0) s | " + " | ".join(f"{o:+g}s" for o in (-6, -4, -3, -2, -1.5, -1, -0.5, 0, 1, 2)) + " | HDV merge events |")
    L.append("|---|---|" + "---|" * 10 + "---|")
    for arm, r in M1.items():
        m, lo, hi, n = r["mean_m3_0"]
        L.append(f"| {arm} | {fmt(m, lo, hi)} | " + " | ".join(fmt(*r["profile"][o][:3]) for o in (-6, -4, -3, -2, -1.5, -1, -0.5, 0, 1, 2))
                 + f" | {r['n_events']} |")
    if "r10b" in M1:
        v = M1["r10b"]["mean_m3_0"][0]
        verdict["M1"] = ("ZERO-HEAD" if v < TH["M1"] else "PASS", f"{v:+.2f} vs threshold {TH['M1']}")
    else:
        verdict["M1"] = ("MISSING", "no r10b rollouts")
    # M2
    L.append("\n## M2 removal test (same seed with/without the waiting ramp car; follower speed change, m/s)\n")
    L.append("Rule: ZERO-HEAD if far-placebo change at 10 s < -0.5 or its 95% CI excludes 0 toward braking. SUMO matched deficit about -1.4 m/s over 5 s.\n")
    L.append("| arm | pairs | 3 s | 6 s | 10 s | 20 s | share dv(10)<-0.5 |")
    L.append("|---|---|---|---|---|---|---|")
    for arm, r in M2.items():
        for kind in ("near", "far"):
            row = r[kind]
            L.append(f"| {arm} {kind} | {row['n']} | " + " | ".join(fmt(*row[t][:3]) if t in row else "n/a" for t in (3, 6, 10, 20))
                     + f" | {row['share<-0.5']:.2f} |")
    if "r10b" in M2 and 10 in M2["r10b"]["far"]:
        m, lo, hi, n = M2["r10b"]["far"][10]
        bad = (m < TH["M2_far10"]) or (np.isfinite(hi) and hi < 0)
        verdict["M2"] = ("ZERO-HEAD" if bad else "PASS", f"far placebo dv(10 s) {fmt(m, lo, hi, n)}; near {fmt(*M2['r10b']['near'][10][:3]) if 10 in M2['r10b']['near'] else 'n/a'}")
    else:
        verdict["M2"] = ("MISSING", "no removal results")
    # M3
    L.append("\n## M3 negative controls (one-step teacher-forced, TRAIN windows): model minus recorded acceleration, mean [95% CI over windows]\n")
    L.append("Rule: ZERO-HEAD if |r10b - recorded| > 0.3 m/s^2 in a group AND the head is the cause (headzero closer to recorded by > 0.05); "
             "if r9/headzero are equally off the miss is base bias, FLAG only. Ramp car behind and lane 0 head offset must be EXACTLY 0.\n")
    names = list(next(iter(M3o.values())).keys()) if M3o else []
    names = [n for n in names if not n.startswith("_")]
    L.append("| group | " + " | ".join(f"{arm} bias" for arm in M3o) + " | r10b head mean | r10b max abs head | n |")
    L.append("|---|" + "---|" * (len(M3o) + 3))
    m3_fail, m3_flag = [], []
    for g in names:
        cells_ = [fmt(*M3o[arm][g]["bias"][:3]) for arm in M3o]
        r = M3o.get("r10b", {}).get(g)
        L.append(f"| {g} | " + " | ".join(cells_) + f" | {r['head_mean']:+.3f} | {r['max_abs_head']:.2g} | {r['n']} |" if r else f"| {g} | ... |")
        if r and "r10b" in M3o:
            b = r["bias"][0]
            if g in ("ramp car behind (no partner)", "lane 0"):
                if r["max_abs_head"] != 0.0:
                    m3_fail.append(f"{g}: head offset not exactly 0 (max {r['max_abs_head']:.3g})")
            if np.isfinite(b) and abs(b) > TH["M3"]:
                bz = M3o.get("headzero", {}).get(g, {}).get("bias", (np.nan,))[0]
                if np.isfinite(bz) and abs(bz) <= abs(b) - 0.05:
                    m3_fail.append(f"{g}: |bias| {abs(b):.2f} (headzero {abs(bz):.2f})")
                else:
                    m3_flag.append(f"{g}: |bias| {abs(b):.2f} but headzero {abs(bz):.2f} (base, not head)")
    if "r10b" in M3o:
        mx = M3o["r10b"]["_max_abs_head_nonpartner"]
        L.append(f"\nMax |head offset| over all one-step rows WITHOUT a ramp partner (lane 0 and lane 1): {mx:.3g} (must be 0).")
        if mx != 0.0:
            m3_fail.append(f"non-partner rows have a head offset (max {mx:.3g})")
    L.append("\nRollout negative controls (head offset on r10b rollout states, mean [CI over states]): "
             + "; ".join(f"{k}: {fmt(*v[:3])}" for k, v in M3r.items() if not k.startswith("_")))
    if M3r:
        L.append(f"\nMax |head offset| over rollout rows where the head does not apply: {M3r['_max_abs_offset_non_partner_rows']:.3g} (must be 0).")
        if M3r["_max_abs_offset_non_partner_rows"] != 0.0:
            m3_fail.append("rollout: head offset on non-applying rows")
        for k, v in M3r.items():
            if not k.startswith("_") and np.isfinite(v[0]) and abs(v[0]) > TH["M3"]:
                m3_fail.append(f"rollout {k}: mean head offset {v[0]:+.2f}")
    verdict["M3"] = ("ZERO-HEAD" if m3_fail else "PASS", "; ".join(m3_fail) or "no head-attributable miss > 0.3" +
                     (" (flags: " + "; ".join(m3_flag) + ")" if m3_flag else ""))
    # M4
    L.append("\n## M4 head output distribution on r10b rollout states (m/s^2)\n")
    L.append(f"Rule: ZERO-HEAD if the head is 'regularly' below -3: share(offset < -3) >= {TH['M4_share']:.0%} (i.e. p5 < -3) in any group with n >= 100.\n")
    L.append("| group | n | p0.1 | p1 | p5 | p50 | p95 | p99.9 | share < -3 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    m4_bad = []
    for g, r in M4.items():
        L.append(f"| {g} | {r['n']} | {r['p0.1']:+.2f} | {r['p1']:+.2f} | {r['p5']:+.2f} | {r['p50']:+.2f} | {r['p95']:+.2f} | {r['p99.9']:+.2f} | {r['share<-3']:.3f} |")
        if r["n"] >= 100 and r["share<-3"] >= TH["M4_share"]:
            m4_bad.append(f"{g}: share<-3 {r['share<-3']:.3f}")
    verdict["M4"] = (("ZERO-HEAD" if m4_bad else "PASS") if M4 else "MISSING", "; ".join(m4_bad) or "head rarely below -3")
    # M5 + tails
    L.append("\n## Fix-B tail reads and safety proxies\n")
    L.append("One-step (teacher-forced TRAIN windows):\n")
    L.append("| arm | events | post-insertion min-accel p10 model (rec) | median model (rec) | onset-row bias, recorded brake | onset-row bias, no brake | tight-gap (<8.6 m) non-insertion braking p10 model (rec) |")
    L.append("|---|---|---|---|---|---|---|")
    for arm, r in TL.items():
        L.append(f"| {arm} | {r['n_events']} | {r['post_min_p10_model']:+.2f} ({r['post_min_p10_rec']:+.2f}) | "
                 f"{r['post_min_median_model']:+.2f} ({r['post_min_median_rec']:+.2f}) | {fmt(*r['onset_bias_brake'][:3])} | "
                 f"{fmt(*r['onset_bias_nobrake'][:3])} | {r['tight_p10_model']:+.2f} ({r['tight_p10_rec']:+.2f}) n={r['tight_n']} |")
    L.append("\nRollouts (120 s, recorded CAV arm replayed, entries off):\n")
    L.append("| arm | lane-1 HDV rows | TTC<1.5 s share | overlap (gap<0) share | hard brake (a<-3) share | a p1 | min bumper gap at insertion p5 (median) | post-insertion F min-accel p10 (median) | jerk at partner switches p50 / p95 (all rows p95) | merge events |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for arm, r in SF.items():
        L.append(f"| {arm} | {r['n_rows']} | {r['ttc<1.5']:.4f} | {r['overlap(gap<0)']:.5f} | {r['hard_brake(a<-3)']:.4f} | {r['a_p1']:+.2f} | "
                 f"{r.get('min_gap_ins_p5', float('nan')):.1f} ({r.get('min_gap_ins_median', float('nan')):.1f}) | "
                 f"{r.get('post_ins_min_accel_p10', float('nan')):+.2f} ({r.get('post_ins_min_accel_median', float('nan')):+.2f}) | "
                 f"{r.get('jerk_switch_p50', float('nan')):.2f} / {r.get('jerk_switch_p95', float('nan')):.2f} ({r.get('jerk_all_p95', float('nan')):.2f}) | {r.get('n_merge_events_hdv_F', 0)} |")
    reject = []
    if "r9" in SF:
        for arm in ("r10b", "headzero"):
            if arm not in SF:
                continue
            for key, nm in (("_ttc", "TTC<1.5 s"), ("_ovl", "overlap/collision proxy"), ("_hb", "hard-brake share")):
                est, lo, hi = paired_diff(SF[arm], SF["r9"], key)
                L.append(f"- {arm} minus r9, {nm} share difference: {est:+.5f} [{lo:+.5f}, {hi:+.5f}]" + ("  WORSE" if lo > 0 else ""))
                if lo > 0 and key in ("_ttc", "_ovl"):
                    reject.append(f"{arm}: {nm} worse than r9")
        if "r10b" in TL and "r9" in TL:
            d = TL["r10b"]["tight_p10_model"] - TL["r9"]["tight_p10_model"]
            L.append(f"- tight-gap non-insertion p10: r10b minus r9 = {d:+.2f} (headzero minus r9 = {TL.get('headzero', {}).get('tight_p10_model', float('nan')) - TL['r9']['tight_p10_model']:+.2f}); flag WORSE if < -0.25")
            if d < -0.25:
                reject.append("tight-gap non-insertion braking p10 worse than r9")
    L.append("\ngate_g_by_k (saturating gate, k -> g): " + (json.dumps(gates) if gates else "n/a"))
    # M5
    L.append("\n## M5 collisions / TTC vs round 9 (official: cpuN SUMMARY_R10B.md)\n")
    if a.m5 and os.path.exists(a.m5):
        L.append(f"Source: {a.m5} (first 60 lines)\n")
        L.append("```\n" + "".join(open(a.m5, encoding="utf-8", errors="replace").readlines()[:60]) + "```")
        m5s = "see excerpt (read by the coordinator)"
    else:
        L.append("PENDING - cpuN SUMMARY_R10B.md not available on gpuR at generation time. The rollout proxies above (TTC<1.5 s, overlap, hard brakes vs r9, "
                 "state-bootstrap CI) are the interim M5 read.")
        m5s = "pending (cpuN)"
    verdict["M5"] = ("REJECT-ROUND (proxy)" if reject else "no proxy regression", "; ".join(reject) or m5s)
    # overall
    L.insert(2, "")
    summary = ["## Verdict table\n", "| check | value | threshold | result |", "|---|---|---|---|"]
    thr = {"M1": "mean [-3,0) s >= -0.9", "M2": "far dv(10 s) >= -0.5 and CI not < 0", "M3": "|model-recorded| <= 0.3, exact 0 for no-partner rows",
           "M4": "share(head < -3) < 5% in every group", "M5": "collisions/TTC not worse than r9", "M7": "90 steps completed"}
    for k in ("M1", "M2", "M3", "M4", "M5", "M7"):
        res_, val = verdict[k]
        summary.append(f"| {k} | {val} | {thr[k]} | **{res_}** |")
    zero = [k for k in ("M1", "M2", "M3", "M4") if verdict[k][0] == "ZERO-HEAD"]
    missing = [k for k in ("M1", "M2", "M3", "M4") if verdict[k][0] == "MISSING"]
    if reject:
        rec_ = "REJECT the round (provisional: proxy M5 regression vs round 9; confirm with cpuN SUMMARY_R10B.md)"
    elif zero:
        rec_ = f"USE r10b_headzero (head zeroed: {', '.join(zero)} failed)"
    elif missing:
        rec_ = f"INCOMPLETE: missing {', '.join(missing)}"
    else:
        rec_ = "USE r10b (M1-M4 pass; M5 per cpuN)"
    summary.append(f"\n**Overall recommendation: {rec_}.** M5 official verdict (collisions / TTC<1.5 s vs round 9) comes from cpuN SUMMARY_R10B.md: {verdict['M5'][1]}.")
    L[3:3] = summary
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump({"verdict": {k: list(v) for k, v in verdict.items()}, "recommendation": rec_}, open(a.out + ".json", "w"), indent=1)
    print("\n".join(summary))


if __name__ == "__main__":
    main()
