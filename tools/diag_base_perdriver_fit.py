"""S-BASEDIAG offline identifiability check (TRAIN rows, numpy, no model training).

For each vehicle (window, slot) fit its own IDM accel a and decel b by grid search on its OWN one-step rows
OUTSIDE the evaluation set (rows with no ramp partner and s < 0 m, i.e. before it reaches the merge zone),
keeping the encoded T, s0, v0 and dropping the learned residual.  Then evaluate the pure IDM core with the
fitted (a, b) on the held-out rows: target followers near the merge, matched slow-start rows, all rows.
This bounds what a reachable per-driver encoder (physical mode) can do for the base bias; it uses only
recorded motion (no labels).  Vehicles with < 20 fit rows get the population median of the fitted values.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "tools")
from diag_base_decomp_an import load  # noqa: E402
from diag_base_params_cf import idm  # noqa: E402

A_GRID = np.round(np.arange(0.4, 3.01, 0.1), 2)
B_GRID = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.2])


def main(prefix):
    D = load(prefix)
    P = D["params"]
    v, vl, gap, rec, s = D["v"], D["lead_v"], D["gap"], D["rec"], D["s"]
    has = D["has"].astype(bool)
    # vehicle id = (window index, slot); windows are concatenated in order
    win = np.concatenate([np.full(n, i) for i, n in enumerate(D["_lens"])])
    vid = win * 1000 + D["slot"]
    fit_rows = (~has) & (s < 0)
    uniq, inv = np.unique(vid, return_inverse=True)
    nveh = len(uniq)
    sse = np.zeros((nveh, len(A_GRID), len(B_GRID)))
    cnt = np.bincount(inv[fit_rows], minlength=nveh)
    for ia, a in enumerate(A_GRID):
        for ib, b in enumerate(B_GRID):
            pred = idm(v[fit_rows], vl[fit_rows], gap[fit_rows], P[fit_rows, 0], P[fit_rows, 1],
                       P[fit_rows, 2], np.full(fit_rows.sum(), a), np.full(fit_rows.sum(), b))
            err = np.minimum((pred - rec[fit_rows]) ** 2, 9.0)          # robust (cap at 3 m/s^2)
            sse[:, ia, ib] = np.bincount(inv[fit_rows], weights=err, minlength=nveh)
    best = sse.reshape(nveh, -1).argmin(-1)
    a_fit = A_GRID[best // len(B_GRID)]
    b_fit = B_GRID[best % len(B_GRID)]
    ok = cnt >= 20
    a_med, b_med = float(np.median(a_fit[ok])), float(np.median(b_fit[ok]))
    a_fit = np.where(ok, a_fit, a_med)
    b_fit = np.where(ok, b_fit, b_med)
    print(f"# {prefix}: vehicles {nveh}, with >= 20 fit rows {int(ok.sum())}")
    print("fitted a p10/25/50/75/90:", np.percentile(a_fit[ok], [10, 25, 50, 75, 90]).round(2),
          "| fitted b p10/50/90:", np.percentile(b_fit[ok], [10, 50, 90]).round(2))
    print("encoded a of the same vehicles (median):", float(np.median(P[:, 3])))
    a_row, b_row = a_fit[inv], b_fit[inv]
    core_fit = idm(v, vl, gap, P[:, 0], P[:, 1], P[:, 2], a_row, b_row)
    core_pop = idm(v, vl, gap, P[:, 0], P[:, 1], P[:, 2], np.full_like(v, a_med), np.full_like(v, b_med))
    tm = D["tmerge"]
    zone = has & (D["dend"] >= 0) & (D["dend"] <= 75.44)
    groups = {
        "partners last 3 s": zone & (tm > 0) & (tm <= 3),
        "partners [-6,-3) s": zone & (tm > 3) & (tm <= 6),
        "unpartnered zone v<2": ~has & (s >= 0) & (s < 75.44) & (v < 2),
        "unpartnered v<2 gap 6-30 (slow start)": ~has & (v < 2) & (gap > 6) & (gap < 30),
        "held-out rows (not fit rows)": ~fit_rows,
        "all rows": np.ones_like(has),
        "after merge zone s > 75": s > 75.44,
        "recorded braking rec < -1": rec < -1,
    }
    print("group | n | rec | model as is bias [MAE] | IDM per-driver fitted (a,b), no residual | IDM fitted-population (a,b)")
    for g, m in groups.items():
        print(f"{g} | {int(m.sum())} | {rec[m].mean():+.2f} | {D['final'][m].mean() - rec[m].mean():+.2f} "
              f"[{np.abs(D['final'][m] - rec[m]).mean():.2f}] | {core_fit[m].mean() - rec[m].mean():+.2f} "
              f"[{np.abs(core_fit[m] - rec[m]).mean():.2f}] | {core_pop[m].mean() - rec[m].mean():+.2f} "
              f"[{np.abs(core_pop[m] - rec[m]).mean():.2f}]")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
