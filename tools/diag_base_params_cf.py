"""S-BASEDIAG: IDM-parameter counterfactuals on the decomposition cache (inference-only diagnostic).

The round-8 encoder runs in ``legacy`` mode: encoded = legacy default + shrink * tanh(.) * bound, so
accel >= 2.49 - 0.8 * shrink (~1.78) and decel >= 4.2 - 1.5 * shrink (~2.87).  Here the IDM core is
recomputed from the cached state with the encoded parameters (sanity: reproduces ``phys``) and with
accel / decel replaced, residual correction unchanged, then compared with the recorded acceleration.
Nothing is fitted into a model; the grid over a constant accel only shows what the data prefer.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "tools")
from diag_base_decomp_an import load  # noqa: E402


def idm(v, vl, gap, v0, T, s0, a, b, amin=-9.0, amax=3.0):
    closing = v - vl
    des = s0 + np.clip(v * T + v * closing / (2 * np.sqrt(a * b)), 0, None)
    acc = a * (1 - (v / np.maximum(v0, 1e-3)) ** 4 - (des / np.maximum(gap, 0.3)) ** 2)
    acc = np.clip(acc, amin, amax)
    return np.minimum(np.maximum(acc, -b), a)


def main(prefix):
    D = load(prefix)
    P = D["params"]
    v, vl, gap = D["v"], D["lead_v"], D["gap"]
    base = idm(v, vl, gap, P[:, 0], P[:, 1], P[:, 2], P[:, 3], P[:, 4])
    ok = np.abs(base - D["phys"]) < 1e-3
    print(f"# {prefix}: rows {len(v)}; recomputed IDM == cached core on {ok.mean():.4f} of rows "
          f"(others: lane-end obstacle / ramp rules)")
    corr = D["corr"]
    rec = D["rec"]
    has = D["has"].astype(bool)
    tm = D["tmerge"]
    zone = has & (D["dend"] >= 0) & (D["dend"] <= 75.44)
    last3 = zone & (tm > 0) & (tm <= 3)
    s = D["s"]
    groups = {
        "partners last 3 s": last3,
        "partners [-6,-3) s": zone & (tm > 3) & (tm <= 6),
        "unpartnered zone v<2": ~has & (s >= 0) & (s < 75.44) & (v < 2),
        "unpartnered approach -400..0 v<2": ~has & (s >= -400) & (s < 0) & (v < 2),
        "unpartnered v<2 gap 6-30 (slow start)": ~has & (v < 2) & (gap > 6) & (gap < 30),
        "all lane-1 rows": np.ones_like(has),
        "v > 15": v > 15,
        "recorded braking rec < -1": rec < -1,
        "after merge zone s > 75": s > 75.44,
        "far upstream s<-400, v<2, gap 6-30": (s < -400) & (v < 2) & (gap > 6) & (gap < 30),
    }
    cfs = {
        "encoded (as model)": (P[:, 3], P[:, 4], P[:, 1], P[:, 2]),
        "a=1.2": (np.full_like(v, 1.2), P[:, 4], P[:, 1], P[:, 2]),
        "a=1.2,b=2.0": (np.full_like(v, 1.2), np.full_like(v, 2.0), P[:, 1], P[:, 2]),
        "textbook a1.2 b2.0 T1.2 s0 2.0": (np.full_like(v, 1.2), np.full_like(v, 2.0),
                                           np.full_like(v, 1.2), np.full_like(v, 2.0)),
        "a=enc-0.6 (unfloored shift)": (np.maximum(P[:, 3] - 0.6, 0.3), P[:, 4], P[:, 1], P[:, 2]),
    }
    print("\nmean (model - recorded), final acceleration incl. the unchanged bounded residual; [MAE]")
    print("group | n | rec | " + " | ".join(cfs) + " | a=1.2,b=2.0 WITHOUT residual")
    for g, m in groups.items():
        line = f"{g} | {int(m.sum())} | {rec[m].mean():+.2f}"
        for name, (a, b, T, s0) in cfs.items():
            core = np.where(ok, idm(v, vl, gap, P[:, 0], T, s0, a, b), D["phys"])
            fin = np.clip(core + corr, -9, 3)
            line += f" | {fin[m].mean() - rec[m].mean():+.2f} [{np.abs(fin[m] - rec[m]).mean():.2f}]"
        core = np.where(ok, idm(v, vl, gap, P[:, 0], P[:, 1], P[:, 2], np.full_like(v, 1.2), np.full_like(v, 2.0)), D["phys"])
        line += f" | {np.clip(core, -9, 3)[m].mean() - rec[m].mean():+.2f} [{np.abs(np.clip(core, -9, 3)[m] - rec[m]).mean():.2f}]"
        print(line)
    print("\nconstant-accel grid (other encoded parameters kept): MAE / mean bias on all lane-1 rows")
    for a0 in (0.8, 1.0, 1.2, 1.4, 1.6, 1.78, 2.0, 2.49):
        core = np.where(ok, idm(v, vl, gap, P[:, 0], P[:, 1], P[:, 2], np.full_like(v, a0), P[:, 4]),
                        D["phys"])
        fin = np.clip(core + corr, -9, 3)
        mm = last3
        print(f"a={a0}: all MAE {np.abs(fin - rec).mean():.3f} bias {np.mean(fin - rec):+.3f} | "
              f"last-3 s partners bias {np.mean(fin[mm] - rec[mm]):+.2f} | slow-start bias "
              f"{np.mean((fin - rec)[groups['unpartnered v<2 gap 6-30 (slow start)']]):+.2f}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
