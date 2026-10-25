"""S-R14 offline analysis of roll_r14.py output (horizon-aware; COMBO300 / R13E-sanity definitions).

Per run (one file): metrics over the late window [W0, H] (W0 = 120 s at H = 300 s, 60 s at H = 120 s):
  lc10        lane-1 -> lane-0 changes with s in (-200, 75.44) m over the whole horizon (COMBO300 'lc10')
  lc10_up     the same over (-800, 75.44) m;  lc01: lane-0 -> lane-1 on (-200, 75.44)
  stop1       lane-1 exact-stop share (v < 0.1) of lane-1 rows in [0, 75.44) m, late window
  std1        mean standing (v < 2) lane-1 vehicles in [0, 75.44) m, late window (standing column)
  dis1        lane-1 discharge (crossings of s = 75.44 on lane 1) veh/min, late window
  merges_pm   ramp merges (lane 2 -> 0/1) per minute, late window; merges_last60 in the last 60 s
  dly_up_near cumulative up_near delay (veh-s, relu(1 - v/31.29), s in [-800, 0) mainline) at H
  dly_merge / dly_ramp  the same for the merge zone / ramp
  spill       mean spillback tail (m upstream of the merge, mech.tail_series) over the last 60 s
  overlap / ttc  same-lane overlap / TTC<1.5 s pair-seconds; hb_share  HDV-follower post-insertion
               hard brake share (min 1 s speed drop < -3 m/s over 6 s, COMBO300); stuck ramp roster cars
Paired estimates: per cluster (panel = state; pairs = pair with ref+act pooled) arm differences with a
1,000-draw cluster bootstrap 95 % CI.  Pairs mode also: action-minus-native (act - ref) per pair, WM vs SUMO.

  python ana_r14.py DIR --horizon 300 --arms 13E,13EL [--tags ref,act] [--ref 13E]
"""
from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mech  # noqa: E402

M1, UPN, ROAD, FREE, LV = 75.44, -800.0, 553.33, 31.29, 5.0

def load(path):
    d = np.load(path, allow_pickle=False)
    S, L, V = d["S"].astype(float), d["L"].astype(int), d["V"].astype(float)
    ids = [str(x) for x in d["ids"]]
    cav = d["cav"].astype(int).copy()
    ro = {str(x): bool(c) for x, c in zip(d["roster"], d["roster_cav"])}
    for j, i in enumerate(ids):
        if i in ro:
            cav[j] = int(ro[i])
    return dict(S=S, L=L, V=V, ids=ids, cav=cav, roster=set(ro))


def lane_pairs(S, L, t):
    out = []
    ok = np.isfinite(S[t]) & (S[t] < ROAD)
    for ln in (0, 1, 2):
        idx = np.flatnonzero(ok & (L[t] == ln))
        if len(idx) > 1:
            o = idx[np.argsort(S[t, idx])]
            out.append(np.stack([o[:-1], o[1:]], 1))
    return np.concatenate(out) if out else np.zeros((0, 2), int)


def metrics(R, H):
    S, L, V, cav = R["S"], R["L"], R["V"], R["cav"]
    T = S.shape[0]
    H = min(H, T - 1)
    w0 = 120 if H >= 300 else 60
    fin = np.isfinite(S)
    a, b = L[:-1], L[1:]
    both = fin[:-1] & fin[1:]
    m = {}
    m["lc10"] = float(((a == 1) & (b == 0) & both & (S[:-1] > -200) & (S[:-1] < M1)).sum())
    m["lc10_up"] = float(((a == 1) & (b == 0) & both & (S[:-1] > UPN) & (S[:-1] < M1)).sum())
    m["lc01"] = float(((a == 0) & (b == 1) & both & (S[:-1] > -200) & (S[:-1] < M1)).sum())
    Z1 = fin & (L == 1) & (S >= 0) & (S < M1)
    late = slice(w0, H + 1)
    z = Z1[late]
    m["stop1"] = float((z & (V[late] < 0.1)).sum() / max(1, z.sum()))
    m["std1"] = float((z & (V[late] < 2)).sum(1).mean())
    cross1 = (fin[:-1] & fin[1:] & (L[1:] == 1) & (S[:-1] < M1) & (S[1:] >= M1)).sum(1)   # COMBO300 ana_resid
    m["dis1"] = float(cross1[w0:H].sum() / ((H - w0) / 60.0))
    mg = ((a == 2) & np.isin(b, (0, 1)) & both).sum(1)
    m["merges_pm"] = float(mg[w0:H].sum() / ((H - w0) / 60.0))
    m["merges_last60"] = float(mg[H - 60:H].sum())
    m["merges_total"] = float(mg[:H].sum())
    reg = mech.regions(S, L)
    dl = np.where(np.isfinite(V), np.maximum(0.0, 1.0 - V / FREE), 0.0)
    for r in ("up_near", "merge", "ramp"):
        m[f"dly_{r}"] = float((dl[:H + 1] * reg[r][:H + 1]).sum())
    tl = mech.tail_series(S, L, V)
    seg = tl[H - 60:H + 1]
    m["spill"] = float(np.mean(np.where(np.isfinite(seg), -seg, 0.0)))
    ttc = ov = 0
    for t in range(H + 1):
        P = lane_pairs(S, L, t)
        if not len(P):
            continue
        f, ld = P[:, 0], P[:, 1]
        gap = S[t, ld] - S[t, f] - LV
        clos = V[t, f] - V[t, ld]
        ov += int((gap < 0).sum())
        ttc += int(((gap >= 0) & (clos > 0) & (gap < 1.5 * np.maximum(clos, 1e-9))).sum())
    m["overlap"], m["ttc"] = float(ov), float(ttc)
    ti, ki = np.nonzero((a == 2) & (b == 1) & both)
    hb = n = 0
    for t0, r in zip(ti, ki):
        tm = t0 + 1
        if tm + 6 > H or cav[r] == 1:
            continue
        on1 = fin[tm] & (L[tm] == 1)
        on1[r] = False
        beh = np.flatnonzero(on1 & (S[tm] <= S[tm, r]))
        if not len(beh):
            continue
        f = beh[np.argmax(S[tm, beh])]
        if cav[f] == 1:
            continue
        w = np.arange(tm, tm + 7)
        if not fin[w, f].all():
            continue
        n += 1
        hb += int(np.diff(V[w, f]).min() < -3)
    m["hb_share"] = float(hb / n) if n else np.nan
    ro = [j for j, i in enumerate(R["ids"]) if i in R["roster"] and fin[0, j] and L[0, j] == 2 and cav[j] != 1]
    m["stuck"] = float(sum(not (np.isin(L[:H + 1, j], (0, 1)) & fin[:H + 1, j]).any() for j in ro))
    return m


def boot(diffs_by_cluster, rng, n=1000):
    x = np.array([np.mean(v) for v in diffs_by_cluster if len(v)], float)
    x = x[np.isfinite(x)]
    if not len(x):
        return [None, None, None, 0]
    bs = [np.mean(rng.choice(x, len(x))) for _ in range(n)]
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(len(x))]


def collect(d, arms, tags):
    files = {}
    for f in sorted(glob.glob(os.path.join(d, "p*_*.npz"))):
        mt = re.match(r"p(\d+)_(ref|act)_(\w+?)(?:_k(\d+))?\.npz$", os.path.basename(f))
        if not mt or mt.group(2) not in tags:
            continue
        arm = mt.group(3)
        if arm != "sumo" and arm not in arms:
            continue
        files[(int(mt.group(1)), mt.group(2), arm)] = f
    return files


def analyze(d, H, arms, tags, ref=None):
    """Deterministic (fresh seeded RNG) analysis dict of one roll_r14 output directory."""
    rng = np.random.default_rng(14)
    ref = ref or arms[0]
    M = {k: metrics(load(f), int(H)) for k, f in collect(d, arms, tags).items()}
    clusters = sorted({p for p, _, _ in M})
    complete = [p for p in clusters if all((p, t, a) in M for t in tags for a in ["sumo", *arms])]
    keys = list(next(iter(M.values())))
    out = {"horizon_s": float(H), "arms": list(arms), "tags": list(tags), "clusters": complete,
           "absolute": {}, "paired": {}}
    for a in ["sumo", *arms]:
        out["absolute"][a] = {k: boot([[M[(p, t, a)][k] for t in tags] for p in complete], rng) for k in keys}
    for a in arms:
        if a == ref:
            continue
        out["paired"][f"{a}-{ref}"] = {k: boot([[M[(p, t, a)][k] - M[(p, t, ref)][k] for t in tags]
                                                for p in complete], rng) for k in keys}
    if set(tags) == {"ref", "act"}:
        out["action_minus_native"] = {}
        for a in ["sumo", *arms]:
            out["action_minus_native"][a] = {k: boot([[M[(p, "act", a)][k] - M[(p, "ref", a)][k]]
                                                      for p in complete], rng)
                                             for k in ("spill", "dly_up_near", "dly_merge", "dly_ramp")}
            out["action_minus_native"][a]["per_pair_spill"] = [
                round(M[(p, "act", a)]["spill"] - M[(p, "ref", a)]["spill"], 1) for p in complete]
    out["per_run"] = {f"p{p:02d}_{t}_{a}": M[(p, t, a)] for (p, t, a) in sorted(M)}
    return clean(out)


def clean(value):
    """JSON-safe copy: non-finite floats -> None (the gate's reader refuses NaN/inf)."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def fmt(v, sign=False):
    if v[0] is None:
        return "n/a"
    f = "{:+.3g} [{:+.3g},{:+.3g}]" if sign else "{:.3g} [{:.3g},{:.3g}]"
    return f.format(*v[:3])


def main():
    d = sys.argv[1]
    H = float(sys.argv[sys.argv.index("--horizon") + 1]) if "--horizon" in sys.argv else 300.0
    arms = sys.argv[sys.argv.index("--arms") + 1].split(",")
    tags = sys.argv[sys.argv.index("--tags") + 1].split(",") if "--tags" in sys.argv else ["ref", "act"]
    ref = sys.argv[sys.argv.index("--ref") + 1] if "--ref" in sys.argv else arms[0]
    out = analyze(d, H, arms, tags, ref)
    keys = list(out["absolute"]["sumo"])
    complete = out["clusters"]
    cols = ["sumo", *arms]
    print(f"clusters {len(complete)} ({'+'.join(tags)}), H={H:.0f} s; absolute mean [95% CI]")
    print("| metric | " + " | ".join(cols) + " | " + " | ".join(out["paired"]) + " |")
    print("|---|" + "---|" * (len(cols) + len(out["paired"])))
    for k in keys:
        cells = [fmt(out["absolute"][a][k]) for a in cols] + [fmt(out["paired"][c][k], True) for c in out["paired"]]
        print(f"| {k} | " + " | ".join(cells) + " |")
    if "action_minus_native" in out:
        print()
        print("action - native (per pair, mean [CI]):")
        for a, v in out["action_minus_native"].items():
            print(a, {k: (fmt(val, True) if k != "per_pair_spill" else val) for k, val in v.items()})


if __name__ == "__main__":
    main()
