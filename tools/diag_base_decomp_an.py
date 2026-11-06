"""S-BASEDIAG: analysis of tools/diag_base_decomp.py caches (component attribution; diagnostic only).

    python tools/diag_base_decomp_an.py tmp_basediag/r8 [tmp_basediag/r10 ...]
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

COMP = ("rec", "final", "pre", "phys", "attn", "temp", "corr", "gate")


def load(prefix):
    W = []
    for i in range(16):
        p = Path(f"{prefix}.shard{i}.pkl")
        if p.is_file():
            W.extend(pickle.load(open(p, "rb"))["windows"])
    keys = [k for k in W[0] if k not in ("episode", "window")]
    D = {k: np.concatenate([w[k] for w in W]) for k in keys}
    D["ep"] = np.concatenate([np.full(len(w["rec"]), hash(w["episode"]) % 100003) for w in W])
    D["nwin"] = len(W)
    D["_lens"] = [len(w["rec"]) for w in W]
    return D


def hw(D):
    return np.where(D["gap"] < 900, D["gap"] / np.maximum(D["v"], 0.5), 99.0)


def row(lbl, D, m, cf=None):
    if m.sum() < 20:
        return
    vals = [D[k][m].mean() for k in COMP]
    clamp = float((D["pre"][m] >= 2.99).mean())
    out = f"{lbl} | {m.sum()} | " + " | ".join(f"{x:+.2f}" for x in vals[:-1]) + f" | {vals[-1]:.2f}"
    out += f" | {D['final'][m].mean() - D['rec'][m].mean():+.2f} | {D['phys'][m].mean() - D['rec'][m].mean():+.2f} | {clamp:.2f}"
    if cf is not None:
        out += f" | {cf[m].mean() - D['rec'][m].mean():+.2f}"
    print(out)


HDR = ("bin | n | rec | final | pre | IDM core | gate*attn | gate*temp | bounded corr | gate | "
       "final-rec | IDM-rec | share pre>=3 | CF virt-leader final-rec")


def report(prefix):
    D = load(prefix)
    print(f"\n# {prefix}: windows {D['nwin']}, rows {len(D['rec'])}")
    amin, amax = -9.0, 3.0
    has = D["has"].astype(bool)
    # counterfactual: IDM core replaced by min(core, virtual-leader IDM) (partner rows only), residual unchanged
    virt = np.where(has & np.isfinite(D["a_virt"]), D["a_virt"], np.inf)
    cf = np.clip(np.minimum(D["phys"], virt) + D["corr"], amin, amax)
    cf_pos = np.clip(np.where(D["gap_r"] > 0, np.minimum(D["phys"], virt), D["phys"]) + D["corr"], amin, amax)
    tm = D["tmerge"]
    zone = has & (D["dend"] >= 0) & (D["dend"] <= 75.44)
    print("\n## partner (target-follower) rows, merge-zone ramp car, by t - t_merge")
    print(HDR)
    for lo, hi in ((6, 10), (4, 6), (3, 4), (2, 3), (1, 2), (0, 1)):
        row(f"[-{hi},-{lo})", D, zone & (tm > lo) & (tm <= hi), cf_pos)
    last3 = zone & (tm > 0) & (tm <= 3)
    row("last 3 s", D, last3, cf_pos)
    row("never merges in window", D, zone & ~np.isfinite(tm), cf_pos)
    print("\n## last 3 s partner rows by bumper gap to the ramp car (CF = core min virtual leader if gap_r > 0)")
    print(HDR)
    for lo, hi in ((-50, 0), (0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 15), (15, 30), (30, 1e4)):
        row(f"gap_r [{lo},{hi})", D, last3 & (D["gap_r"] >= lo) & (D["gap_r"] < hi), cf_pos)
    print("\n## last 3 s partner rows by follower speed")
    print(HDR)
    for lo, hi in ((0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 8), (8, 40)):
        row(f"v [{lo},{hi})", D, last3 & (D["v"] >= lo) & (D["v"] < hi), cf_pos)
    print("\n## last 3 s partner rows by own-leader bumper gap (model's lead_neighbour)")
    print(HDR)
    for lo, hi in ((0, 3), (3, 5), (5, 8), (8, 12), (12, 20), (20, 50), (50, 1e4)):
        row(f"gap [{lo},{hi})", D, last3 & (D["gap"] >= lo) & (D["gap"] < hi), cf_pos)
    # state-matched unpartnered controls (5-NN in R11 matching coordinates)
    phi = np.stack([D["v"] / 10.0, np.arcsinh(hw(D)), (D["v"] - D["lead_v"]) / 5.0,
                    np.arcsinh((75.44 - D["s"]) / 20.0)], -1)
    ctrl = ~has
    tree = cKDTree(phi[ctrl])
    _, idx = tree.query(phi[last3], k=5)
    cidx = np.nonzero(ctrl)[0][idx.reshape(-1)]
    cm = np.zeros_like(has)
    print("\n## 5-NN state-matched unpartnered controls of the last-3 s partner rows (R11 coordinates)")
    print(HDR)
    sub = {k: D[k][cidx] for k in COMP}
    sub.update(pre=D["pre"][cidx])
    vals = [sub[k].mean() for k in COMP]
    print(f"matched controls | {len(cidx)} | " + " | ".join(f"{x:+.2f}" for x in vals[:-1])
          + f" | {vals[-1]:.2f} | {vals[1] - vals[0]:+.2f} | {vals[3] - vals[0]:+.2f} | "
          f"{float((sub['pre'] >= 2.99).mean()):.2f}")
    print("matched controls in merge zone share:", float(((D['s'][cidx] >= 0) & (D['s'] [cidx] < 75.44)).mean()),
          "median s", float(np.median(D['s'][cidx])), "median gap", float(np.median(D['gap'][cidx])),
          "median v", float(np.median(D['v'][cidx])))
    # controls matched on (v, gap, closing, position) instead of headway
    phi2 = np.stack([D["v"] / 10.0, np.arcsinh(D["gap"] / 5.0), (D["v"] - D["lead_v"]) / 5.0,
                     np.arcsinh((75.44 - D["s"]) / 20.0)], -1)
    tree2 = cKDTree(phi2[ctrl])
    _, idx2 = tree2.query(phi2[last3], k=5)
    cidx2 = np.nonzero(ctrl)[0][idx2.reshape(-1)]
    vals2 = [D[k][cidx2].mean() for k in COMP]
    print(f"matched on gap (not headway) | {len(cidx2)} | " + " | ".join(f"{x:+.2f}" for x in vals2[:-1])
          + f" | {vals2[-1]:.2f} | {vals2[1] - vals2[0]:+.2f} | {vals2[3] - vals2[0]:+.2f}")
    print("\n## unpartnered rows, v < 2 m/s, by region and own-leader gap (generic slow-start check)")
    print(HDR)
    for reg, rm in (("s<-400", D["s"] < -400), ("-400..0", (D["s"] >= -400) & (D["s"] < 0)),
                    ("zone 0..75", (D["s"] >= 0) & (D["s"] < 75.44)), (">75", D["s"] >= 75.44)):
        for lo, hi in ((0, 3), (3, 6), (6, 12), (12, 30), (30, 1e4)):
            row(f"{reg} gap [{lo},{hi})", D, ctrl & rm & (D["v"] < 2) & (D["gap"] >= lo) & (D["gap"] < hi))
    if "params" in D:
        P = D["params"]
        names = ("v0", "T", "s0", "a", "b")
        print("\n## encoded IDM parameters (mean): last-3 s partners vs 5-NN controls vs all lane-1 rows")
        for lbl, m in (("partners last 3 s", last3), ("matched controls", cidx), ("all", slice(None))):
            print(lbl, dict(zip(names, np.round(P[m].mean(0), 2))))
    print("\n## inv_between (junction vehicle between follower and its leader): partners last3",
          float(D["inv_between"][last3].mean()), "all", float(D["inv_between"].mean()))
    return D


if __name__ == "__main__":
    for pre in sys.argv[1:]:
        report(pre)


def hidden_report(prefix):
    """Rows whose replay state hides a junction-transit lane-1 vehicle between the follower and its leader."""
    D = load(prefix)
    if "hid_between" not in D:
        return
    has = D["has"].astype(bool)
    tm = D["tmerge"]
    zone = has & (D["dend"] >= 0) & (D["dend"] <= 75.44)
    last3 = zone & (tm > 0) & (tm <= 3)
    hid = D["hid_between"].astype(bool)
    print(f"\n## {prefix}: hidden (junction-transit, loss_valid=0) lane-1 vehicle between follower and its replay leader")
    print("group | n | share hidden | final-rec (hidden) | final-rec (not hidden) | IDM-rec (hidden) | median hidden gap / replay gap")
    s = D["s"]
    for g, m in (("partners last 3 s", last3), ("partners [-6,-3)", zone & (tm > 3) & (tm <= 6)),
                 ("all lane-1 rows", np.ones_like(has)), ("rows s in [50,75.44)", (s >= 50) & (s < 75.44)),
                 ("rows s in [-25,0)", (s >= -25) & (s < 0))):
        mh, mn = m & hid, m & ~hid
        if mh.sum() < 5:
            print(f"{g} | {int(m.sum())} | {mh.mean() if m.sum() else 0:.3f} | - | - | - | -")
            continue
        print(f"{g} | {int(m.sum())} | {mh.sum() / m.sum():.3f} | {(D['final'] - D['rec'])[mh].mean():+.2f} | "
              f"{(D['final'] - D['rec'])[mn].mean():+.2f} | {(D['phys'] - D['rec'])[mh].mean():+.2f} | "
              f"{np.nanmedian(D['hid_gap'][mh]):.1f} / {np.median(D['gap'][mh]):.1f}")


if __name__ == "__main__" and len(sys.argv) > 1:
    for pre in sys.argv[1:]:
        hidden_report(pre)
