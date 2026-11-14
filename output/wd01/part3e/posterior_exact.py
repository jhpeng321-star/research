"""Part 3e exact-likelihood vType posterior (replaces the p3d classifier).

Per observed step (v[t] -> v[t+1], known same-lane leader at t):
  residual r = v[t+1] - IDM_vType(v[t], gap, dv)
Under the adopted emulator + dawdle, r ~ Uniform(-A(vType, regime), 0)
with A from the p3e dawdle fit (results/dawdle_fit.json); a small tail
epsilon covers outliers.  Posterior = prior * exp(sum log density).
Validation: log loss / accuracy vs prior, on the 4 validation roots."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wdpaths  # noqa: E402

HERE = Path(__file__).resolve().parent
P2C = Path(wdpaths.RD_WT_ROOT) / "wd01" / "output" / "wd01" / "partA_data"
for p in (str(P2C), str(HERE.parent / "part3"), str(HERE.parent / "part3d")):
    if p not in sys.path:
        sys.path.insert(0, p)

import p3lib as P  # noqa: E402
from sumo_emu import sumo_idm_acc  # noqa: E402

DT = 0.5
VT = {0: dict(sigma=0.6, accel=3.0, tau=0.95, decel=4.5, s0=1.5, sf=1.08),
      1: dict(sigma=0.4, accel=2.6, tau=1.1, decel=4.5, s0=2.5, sf=1.00),
      2: dict(sigma=0.2, accel=2.0, tau=1.5, decel=3.0, s0=3.5, sf=0.92)}
A_ADOPTED = {(0, "following"): 0.86, (0, "free"): 1.60,
             (1, "following"): 0.47, (1, "free"): 0.74,
             (2, "following"): 0.27, (2, "free"): 0.40}
EPS = 1e-3


def idm_vnext(ra, i, k):
    typ = int(ra.typ[i])
    pr = VT[typ]
    lane = int(ra.lane_m[i])
    ids = ra.tick_rows(k)
    ids = ids[ra.lane_m[ids] == lane]
    s_all = ra.s[ids]
    o = ids[np.argsort(-s_all, kind="stable")]
    pa = int(np.where(o == i)[0][0])
    if pa == 0:
        gap, dv = 300.0, 0.0
    else:
        l_ = o[pa - 1]
        gap = float(ra.s[l_] - ra.s[i])
        if gap <= 0 or gap > 250:
            gap, dv = 300.0, 0.0
        else:
            dv = float(ra.v[i] - ra.v[l_])
    v0 = pr["sf"] * (22.22 if (lane == 2 and ra.s[i] < 378.66) else 31.29)
    acc = sumo_idm_acc(np.array([float(ra.v[i])]), np.array([gap]),
                       np.array([dv]), pr["tau"], pr["accel"], pr["decel"],
                       pr["s0"], np.array([v0]))[0]
    regime = "following" if gap <= 250 else "free"
    v_pred = min(max(float(ra.v[i]) + acc * DT, 0.0),
                 1.2 * (22.22 if lane == 2 and ra.s[i] < 378.66 else 31.29))
    return v_pred, regime, gap


def collect(root, max_per_root=2500, seed=0):
    """Grouped per-(vid, t0) residual sequences: r[t] = v[t+1] - IDM_vt
    for each candidate vType, plus the true vType label."""
    rng = np.random.default_rng(seed)
    ra = P.root_arrays(root)
    rows = defaultdict(list)
    for i in range(ra.n):
        rows[int(ra.vid[i])].append(i)
    lc_at = defaultdict(set)
    for v, t in zip(ra.lc_vid, ra.lc_tick):
        lc_at[int(v)].add(int(t))
    seqs = []
    for vi, idxs in rows.items():
        idxs.sort(key=lambda i: ra.tick[i])
        typ = int(ra.typ[idxs[0]])
        cand = []
        for a, b in zip(idxs, idxs[1:]):
            ta, tb = int(ra.tick[a]), int(ra.tick[b])
            if tb != ta + 1 or ta in lc_at.get(vi, ()):
                continue
            if ra.inj[a] or ra.inj[b]:
                continue
            la, lb = int(ra.lane_m[a]), int(ra.lane_m[b])
            if la != lb or la < 0:
                continue
            cand.append((a, b, la))
        if len(cand) < 10:
            continue
        if len(cand) > max_per_root:
            keep = rng.choice(len(cand), max_per_root, replace=False)
            cand = [cand[i] for i in sorted(keep)]
        rs = []
        for a, b, la in cand:
            v_pred, regime, _ = idm_vnext(ra, a, int(ra.tick[a]))
            rs.append((float(ra.v[b]) - v_pred,
                       tuple(A_ADOPTED[(t, regime)] for t in (0, 1, 2)),
                       1 if regime == "free" else 0))
        seqs.append((vi, rs, typ))
    return seqs


MU = {(0, 1): -0.751, (0, 0): -0.404, (1, 1): -0.244, (1, 0): -0.212,
      (2, 1): -0.225, (2, 0): -0.082}
SD = {(0, 1): 0.899, (0, 0): 0.388, (1, 1): 0.574, (1, 0): 0.260,
      (2, 1): 0.464, (2, 0): 0.415}


def loglik_seq(rs, typ):
    """Measured Gaussian likelihood per (vType, regime) from B1."""
    tot = 0.0
    for r, a_v, free in rs:
        mu = MU[(typ, free)]
        sd = SD[(typ, free)]
        tot += -0.5 * ((r - mu) / sd) ** 2 - np.log(sd * 2.5066)
    return tot


def main():
    man = json.loads((P2C / "MANIFEST.json").read_text(encoding="utf-8"))
    splits = {}
    for name, r in man["roots"].items():
        splits.setdefault(r["split"], []).append(name)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    prior = np.array([0.1, 0.7, 0.2])
    model_ll = prior_ll = 0.0
    acc = cnt = 0
    per_class = defaultdict(lambda: [0, 0])
    val_set = {"cv20_7005m2000", "cv20_7006m2000", "cv30_7305m2700",
               "seed_7535m2400"}
    val = [r for r in splits["train"] if r in val_set]
    trn = [r for r in splits["train"] if r not in val_set]
    for split, roots in (("train", trn), ("val", val)):
        model_ll = prior_ll = 0.0
        acc = cnt = 0
        for root in roots:
            seqs = collect(root, max_per_root=2500)
            for vi, rs, truth in seqs:
                lls = np.array([loglik_seq(rs, t) for t in (0, 1, 2)])
                post = prior * np.exp(lls - lls.max())
                post /= post.sum()
                model_ll += -np.log(max(post[truth], 1e-12))
                prior_ll += -np.log(prior[truth])
                acc += int(int(np.argmax(post)) == truth)
                cnt += 1
                per_class[truth][0] += int(np.argmax(post) == truth)
                per_class[truth][1] += 1
        print(f"{split}: n_vehicles={cnt} "
              f"model log loss={model_ll / cnt:.4f} "
              f"prior log loss={prior_ll / cnt:.4f} "
              f"acc={acc / cnt:.4f}")
        if split == "val":
            res = dict(split=split, n_vehicles=cnt,
                       model_log_loss=model_ll / cnt,
                       prior_log_loss=prior_ll / cnt,
                       accuracy=acc / cnt,
                       per_class={str(t): dict(n=v[1],
                                               acc=v[0] / max(v[1], 1))
                                  for t, v in sorted(per_class.items())})
    (out / "posterior_exact.json").write_text(json.dumps(res, indent=1),
                                              encoding="utf-8")
    print("posterior_exact.json written")


if __name__ == "__main__":
    main()
