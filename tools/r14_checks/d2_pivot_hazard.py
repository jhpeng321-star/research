"""S-R14 D2: empirical 1 s merge hazard of in-zone ramp cars binned by the S-R12C pivot margin c
(min side log(gap/req), learned scalars s/T/b per side, evaluated on the 1 s grid; approximate: the WM decides on
the post-motion 0.5 s geometry) x distance to the lane end, plus ramp-minus-lane-1 speed. WM vs SUMO.
Diagnostic only (bins are analysis-only).   python d2_pivot_hazard.py DIR arm1,arm2"""
import glob
import json
import os
import sys
import numpy as np
M1, LV = 75.44, 5.0
P = dict(s_f=2.652292490005493, T_f=0.269473135471344, b_f=0.9713364243507385,
         s_l=1.1882646083831787, T_l=0.32612845301628113, b_l=1.164392113685608)   # S-R12C merge_pivot (frozen in 13E)
CB = [-np.inf, -1.0, -0.5, 0.0, 0.5, np.inf]
DB = [0.0, 25.0, 50.0, 75.44]


def rows(S, L, V):
    T, N = S.shape
    fin = np.isfinite(S)
    out = []
    for t in range(T - 1):
        on1 = fin[t] & (L[t] == 1)
        i1 = np.flatnonzero(on1)
        s1 = S[t, i1]
        for j in np.flatnonzero(fin[t] & (L[t] == 2) & (S[t] >= 0) & (S[t] <= M1)):
            s, v = S[t, j], V[t, j]
            ah, bh = i1[s1 > s], i1[s1 <= s]
            cs = []
            dv = np.nan
            if len(ah):
                k = ah[np.argmin(S[t, ah])]
                g = S[t, k] - LV - s
                req = P["s_l"] + P["T_l"] * v + max(0.0, v * v - V[t, k] ** 2) / (2 * P["b_l"])
                cs.append(np.log(g / req) if g > 0 else -np.inf)
                dv = v - V[t, k]
            if len(bh):
                k = bh[np.argmax(S[t, bh])]
                g = s - LV - S[t, k]
                req = P["s_f"] + P["T_f"] * V[t, k] + max(0.0, V[t, k] ** 2 - v * v) / (2 * P["b_f"])
                cs.append(np.log(g / req) if g > 0 else -np.inf)
            c = min(cs) if cs else np.inf
            merged = fin[t + 1, j] and L[t + 1, j] in (0, 1)
            gone = not fin[t + 1, j]
            if not gone:
                out.append((M1 - s, c, merged, v, dv))
    return out


def main():
    d, arms = sys.argv[1], sys.argv[2].split(",")
    res = {}
    for arm in ["sumo"] + arms:
        R = []
        for f in sorted(glob.glob(os.path.join(d, f"p*_*_{arm}*.npz"))):
            z = np.load(f)
            R += rows(z["S"].astype(float), z["L"].astype(int), z["V"].astype(float))
        R = np.array(R, float)
        tab = {}
        for a, b in zip(DB[:-1], DB[1:]):
            for lo, hi in zip(CB[:-1], CB[1:]):
                sel = (R[:, 0] >= a) & (R[:, 0] < b) & (R[:, 1] >= lo) & (R[:, 1] < hi)
                tab[f"d{a:.0f}-{b:.0f} c[{lo},{hi})"] = [float(R[sel, 2].mean()) if sel.any() else np.nan, int(sel.sum())]
            sel = (R[:, 0] >= a) & (R[:, 0] < b)
            tab[f"d{a:.0f}-{b:.0f} v_med"] = [float(np.median(R[sel, 3])), int(sel.sum())]
            dv = R[sel, 4][np.isfinite(R[sel, 4])]
            tab[f"d{a:.0f}-{b:.0f} v-v_lead1 med"] = [float(np.median(dv)) if len(dv) else np.nan, len(dv)]
            tab[f"d{a:.0f}-{b:.0f} share c>=0"] = [float((R[sel, 1] >= 0).mean()), int(sel.sum())]
        res[arm] = tab
    json.dump(res, open(os.path.join(d, "d2_pivot_hazard.json"), "w"), indent=1)
    keys = list(res["sumo"])
    print("| bin | " + " | ".join(res) + " |\n|---|" + "---|" * len(res))
    for k in keys:
        print(f"| {k} | " + " | ".join("{:.3f} ({})".format(*res[a][k]) for a in res) + " |")


if __name__ == "__main__":
    main()
