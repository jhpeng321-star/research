"""Part 3e DAWDLE fit: on TRAIN roots with TRUE vTypes, compare the
one-step residuals of the deterministic SUMO IDM against the reviewer's
dawdle hypotheses:
  H0: residual = 0 (deterministic; refuted by p3d)
  H1: residual ~ -U(0, sigma*accel*dt) per vType
  H2: fitted uniform -U(0, A) per (vType, regime) with A from the data
Also reports the free-regime excess (aggressive free mean -0.75).
Outputs: results/dawdle_fit.json"""
import json
from pathlib import Path
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wdpaths  # noqa: E402

HERE = str(Path(wdpaths.RD_WT_ROOT) / "wd01" / "output" / "wd01" / "part3d")
P2C = str(Path(wdpaths.RD_WT_ROOT) / "wd01" / "output" / "wd01" / "partA_data")
P3D = str(Path(wdpaths.RD_WT_ROOT) / "wd01" / "output" / "wd01" / "part3d")
PART3 = str(Path(wdpaths.RD_WT_ROOT) / "wd01" / "output" / "wd01" / "part3")
for p in (P2C, PART3, P3D):
    if p not in sys.path:
        sys.path.insert(0, p)

import p3lib as P  # noqa: E402
from sumo_emu import VTYPES, sumo_idm_acc  # noqa: E402

DT = 0.5
SIGMA = {0: 0.6, 1: 0.4, 2: 0.2}


def ks_uniform(samples, a):
    """KS statistic of samples against U(0, a) (two-sided)."""
    x = np.sort(samples)
    n = len(x)
    cdf = x / a
    d_plus = np.max(np.arange(1, n + 1) / n - cdf)
    d_minus = np.max(cdf - np.arange(0, n) / n)
    return float(max(d_plus, d_minus))


def main():
    # vTypes (from requests.rou.xml): sigma, accel per type
    vt = {0: dict(sigma=0.6, accel=3.0, tau=0.95, decel=4.5, s0=1.5,
                  sf=1.08),
          1: dict(sigma=0.4, accel=2.6, tau=1.1, decel=4.5, s0=2.5,
                  sf=1.00),
          2: dict(sigma=0.2, accel=2.0, tau=1.5, decel=3.0, s0=3.5,
                  sf=0.92)}
    import p3lib as P
    train = sorted({r for r in
                    json.load(open(P2C + "/MANIFEST.json"))["roots"]
                    and []} | set())  # placeholder
    man = json.load(open(P2C + "/MANIFEST.json"))
    train_roots = sorted(name for name, r in man["roots"].items()
                         if r["split"] == "train")
    res = defaultdict(lambda: defaultdict(list))
    for root in train_roots:
        ra = P.root_arrays(root)
        rows = defaultdict(list)
        for i in range(ra.n):
            rows[int(ra.vid[i])].append(i)
        lc_at = defaultdict(set)
        for v, t in zip(ra.lc_vid, ra.lc_tick):
            lc_at[int(v)].add(int(t))
        for vi, idxs in rows.items():
            idxs.sort(key=lambda i: ra.tick[i])
            typ = int(ra.typ[idxs[0]])
            pr = vt[typ]
            for a, b in zip(idxs, idxs[1:]):
                ta, tb = int(ra.tick[a]), int(ra.tick[b])
                if tb != ta + 1 or ta in lc_at.get(vi, ()):
                    continue
                if ra.inj[a] or ra.inj[b]:
                    continue
                la = int(ra.lane_m[a])
                if la != int(ra.lane_m[b]) or la < 0:
                    continue
                ids = ra.tick_rows(ta)
                ids = ids[ra.lane_m[ids] == la]
                s_all = ra.s[ids]
                o = ids[np.argsort(-s_all, kind="stable")]
                pa = int(np.where(o == a)[0][0])
                if pa == 0:
                    gap, dv = 300.0, 0.0
                else:
                    l_ = o[pa - 1]
                    gap = float(ra.s[l_] - ra.s[a])
                    if gap <= 0 or gap > 250:
                        gap, dv = 300.0, 0.0
                    else:
                        dv = float(ra.v[a] - ra.v[l_])
                v0 = pr["sf"] * (22.22 if (la == 2 and ra.s[a] < 378.66)
                                 else 31.29)
                acc = sumo_idm_acc(np.array([float(ra.v[a])]),
                                   np.array([max(gap, 0.1)]),
                                   np.array([dv]), pr["tau"],
                                   pr["accel"], pr["decel"], pr["s0"],
                                   np.array([v0]))[0]
                v_idm = min(max(float(ra.v[a]) + acc * DT, 0.0),
                            1.2 * (22.22 if la == 2 and ra.s[a] < 378.66
                                   else 31.29))
                r = float(ra.v[b]) - v_idm
                regime = "following" if gap <= 250 else "free"
                res[(typ, regime)][pr["accel"]].append(r)
    # aggregate
    out = dict(per_vtype_regime={}, notes=[])
    print("dawdle fit per (vType, regime, accel):")
    for (typ, regime), by_accel in sorted(res.items()):
        merged = []
        for acc_val, rs in by_accel.items():
            merged.extend((r, acc_val) for r in rs)
        pr = vt[typ]
        A_theory = pr["sigma"] * pr["accel"] * DT
        rs = np.array([m[0] for m in merged])
        neg = -rs[rs < 0]
        neg = neg[(neg > 1e-9) & (neg <= 4.0)]
        if len(neg) < 50:
            continue
        A_fit = 2.0 * float(neg.mean())
        ks = ks_uniform(neg, A_fit)
        ks_theory = ks_uniform(neg, A_theory)
        key = f"{typ}/{regime}"
        out["per_vtype_regime"][key] = dict(
            n=len(merged), n_neg=len(neg),
            mean=float(rs.mean()), sd=float(rs.std()),
            min=float(rs.min()), max=float(rs.max()),
            A_theory=A_theory, A_fit=A_fit,
            ks_A_fit=ks, ks_A_theory=ks_theory,
            mean_pred_theory=-A_theory / 2,
            mean_pred_fit=-A_fit / 2)
        print(f"  {key}: n={len(merged)} mean={rs.mean():+.3f} "
              f"min={rs.min():+.3f} max={rs.max():+.3f} "
              f"A_theory={A_theory:.2f} A_fit={A_fit:.2f} "
              f"KS(A_fit)={ks:.3f} KS(A_theory)={ks_theory:.3f}")
    results = Path(wdpaths.RD_WT_ROOT) / "wd01" / "output" / "wd01" / \
        "part3e" / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "dawdle_fit.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    print("dawdle_fit.json written")


if __name__ == "__main__":
    main()
