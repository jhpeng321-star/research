"""B1 one-step validation: SUMO IDM emulator vs logged next speeds on
TRAIN roots, per vType, free vs following; residual distribution per
vType (the sigma finding)."""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "partA_data"))
sys.path.insert(0, str(HERE.parent / "part3"))

import p3lib as P  # noqa: E402
from sumo_emu import VTYPES, leaders_within_lane, lane_limit, sumo_idm_acc  # noqa: E402

DT = 0.5


def main(roots, max_pairs_per_root=40000):
    res = defaultdict(list)          # (vtype, regime) -> residual list
    mae = defaultdict(lambda: [0.0, 0])
    for root in roots:
        ra = P.root_arrays(root)
        rows = defaultdict(list)
        for i in range(ra.n):
            rows[int(ra.vid[i])].append(i)
        lc_at = defaultdict(set)
        for vi, tk in zip(ra.lc_vid, ra.lc_tick):
            lc_at[int(vi)].add(int(tk))
        for vi, idxs in rows.items():
            ch = lc_at.get(vi, ())
            idxs.sort(key=lambda i: ra.tick[i])
            typ = int(ra.typ[idxs[0]])
            for a, b in zip(idxs, idxs[1:]):
                ta, tb = int(ra.tick[a]), int(ra.tick[b])
                if tb != ta + 1 or ta in ch:
                    continue
                if ra.inj[a] or ra.inj[b]:
                    continue
                la, lb = int(ra.lane_m[a]), int(ra.lane_m[b])
                if la != lb or la < 0:
                    continue
                # leader at tick a (same lane, next larger s)
                ids = ra.tick_rows(ta)
                ids = ids[ra.lane_m[ids] == la]
                s_all = ra.s[ids]
                o = ids[np.argsort(-s_all, kind="stable")]
                pa = int(np.where(o == a)[0][0])
                if pa == 0:
                    gap, dv = 300.0, 0.0        # free (frontmost)
                else:
                    l_ = o[pa - 1]              # leader = next larger s
                    gap = float(ra.s[l_] - ra.s[a])
                    if gap <= 0 or gap > 250:
                        gap, dv = 300.0, 0.0
                    else:
                        dv = float(ra.v[a] - ra.v[l_])
                v0 = VTYPES[typ]["sf"] * lane_limit(la, float(ra.s[a]))
                acc = sumo_idm_acc(np.array([float(ra.v[a])]),
                                   np.array([max(gap, 0.1)]),
                                   np.array([dv]), VTYPES[typ]["tau"],
                                   VTYPES[typ]["a_max"], VTYPES[typ]["b"],
                                   VTYPES[typ]["s0"], np.array([v0]))[0]
                v_pred = min(max(float(ra.v[a]) + acc * DT, 0.0),
                             1.2 * lane_limit(la, float(ra.s[a])))
                r = float(ra.v[b]) - v_pred
                regime = "following" if gap <= 250 else "free"
                res[(typ, regime)].append(r)
                mae[(typ, regime)][0] += abs(r)
                mae[(typ, regime)][1] += 1
                if mae[(typ, regime)][1] >= 4000:
                    break
            done = all(mae[(t, reg)][1] >= 4000 for t in (0, 1, 2)
                       for reg in ("free", "following"))
            if done:
                break
    print("per vType x regime: n, MAE(m/s), residual mean, sd")
    for (typ, regime), r in sorted(res.items()):
        r = np.asarray(r)
        print(f"  vtype {typ} {regime:9s}: n={len(r):6d} "
              f"MAE={np.abs(r).mean():.4f} mean={r.mean():+.4f} "
              f"sd={r.std():.4f}")
    out = {}
    for (t, reg), v in res.items():
        arr = np.asarray(v)
        out[f"{t}/{reg}"] = dict(n=len(arr),
                                 mae=float(np.abs(arr).mean()),
                                 mean=float(arr.mean()),
                                 sd=float(arr.std()))
    import json
    (Path(HERE) / "results").mkdir(exist_ok=True)
    (Path(HERE) / "results" / "b1_sigma.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    import p3lib as P
    main(P.roots_by_split()["train"])
