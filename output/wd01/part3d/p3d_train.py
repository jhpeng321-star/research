"""Part 3d training: (B2) vType posterior from <=10 s history; (B3)
per-direction lane-change hazard (class-balanced).  TRAIN roots only.
Weights -> weights/run3d/{posterior.pt, hazard.pt}."""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "partA_data"))
sys.path.insert(0, str(HERE.parent / "part3"))

import engine as E  # noqa: E402
import p3lib as P  # noqa: E402
import models as M  # noqa: E402
from models import Hazard, Posterior, hist_summary, save_model  # noqa: E402

T0 = time.time()
DEV = torch.device("cpu")


def log(m):
    print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)


def collect_posterior(roots):
    """(hist [20,3], type_label) for every vehicle with a full 20-tick
    contiguous history at t0 in {240, 360, 480}."""
    post = []
    for root in roots:
        ra = P.root_arrays(root)
        for k in (240, 360, 480):
            hist = P.history_at(ra, k)
            idx = ra.tick_rows(k)
            idx = idx[ra.lane_m[idx] >= 0]
            for i in idx:
                vi = int(ra.vid[i])
                if vi in hist:
                    post.append((hist[vi], int(ra.typ[i])))
    return post


def collect_hazard(roots, max_pos=6000, seed=0):
    """Per declared lane change: positive sample at the tick before the
    change (label = direction +1 right / -1 left) and one background tick
    of the same vehicle (label 0).  Features x19 at that tick."""
    X, Y = [], []
    n_pos = 0
    for root in roots:
        if n_pos >= max_pos:
            break
        ra = P.root_arrays(root)
        lc_at = defaultdict(list)
        for v, t_new in zip(ra.lc_vid, ra.lc_tick):
            i_new = ra._rowmap.get((int(v), int(t_new)))
            i_prev = ra._rowmap.get((int(v), int(t_new) - 1))
            if i_new is None or i_prev is None:
                continue
            lc_at[(int(v), int(t_new))].append(
                int(ra.lane_m[i_new]) - int(ra.lane_m[i_prev]))
        rng = np.random.default_rng(seed)

        def feats_at(i, k):
            ii = ra.tick_rows(k)
            if len(ii) < 3:
                return None
            hits = np.where(ii == i)[0]
            if len(hits) == 0:
                return None
            pos = int(hits[0])
            sl = E.neighbour_slots(ra.lane_m[ii], ra.s[ii], ra.v[ii],
                                   ra.cav[ii])[pos]
            x = np.zeros(19)
            x[0] = float(ra.v[i]) / 40.0
            x[1] = float(np.clip((454.10 - ra.s[i]) / 500.0, 0, 1))
            x[2:14] = np.clip(sl[:, 1:3].reshape(-1), -250.0, 250.0) / 250.0
            x[14] = 1.0 if (ra.lane_m[i] == 2
                            and ra.s[i] >= 378.66) else 0.0
            return x

        for (vi, t_new), dirs in lc_at.items():
            if n_pos >= max_pos:
                break
            if len(dirs) != 1:
                continue
            i_prev = ra._rowmap.get((vi, t_new - 1))
            if i_prev is None or ra.lane_m[i_prev] < 0:
                continue
            x = feats_at(i_prev, t_new - 1)
            if x is None:
                continue
            X.append(x)
            Y.append(max(-1, min(1, dirs[0])))
            n_pos += 1
            negs = [t2 for t2 in range(t_new - 40, t_new - 4)
                    if (vi, t2) in ra._rowmap
                    and ra.lane_m[ra._rowmap[(vi, t2)]] >= 0
                    and t2 not in lc_at.get(vi, ())
                    and (t2 - 1) not in lc_at.get(vi, ())]
            if negs:
                t2 = int(rng.choice(negs))
                x2 = feats_at(ra._rowmap[(vi, t2)], t2)
                if x2 is not None:
                    X.append(x2)
                    Y.append(0)
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.int64)


def main():
    splits = P.roots_by_split()
    out = HERE / "weights" / "run3d"
    out.mkdir(parents=True, exist_ok=True)

    log("collecting posterior samples ...")
    post = collect_posterior(splits["train"])
    Xh = torch.as_tensor(np.stack([p[0] for p in post]),
                         dtype=torch.float32)
    yv = torch.as_tensor(np.array([p[1] for p in post]), dtype=torch.int64)
    log(f"posterior samples: {len(post)}")

    m2 = M.Posterior()
    opt = torch.optim.Adam(m2.parameters(), lr=2e-3, weight_decay=1e-4)
    for ep in range(30):
        perm = torch.randperm(len(Xh))
        tot, nb = 0.0, 0
        for b in range(0, len(Xh), 4096):
            ii = perm[b:b + 4096]
            loss = F.cross_entropy(m2(Xh[ii]), yv[ii])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        log(f"  posterior ep{ep}: loss {tot / max(nb, 1):.4f}")
    M.save_model(out / "posterior.pt", m2, dict(
        classes=[0, 1, 2], features="hist_summary (11)",
        train_roots=splits["train"], epochs=30))
    log("posterior saved")

    log("collecting hazard samples ...")
    Xh, yh = collect_hazard(splits["train"])
    X = torch.as_tensor(Xh, dtype=torch.float32)
    y = torch.as_tensor(yh, dtype=torch.int64)
    log(f"hazard samples: {len(X)} (events {int((y != 0).sum())})")

    m3 = M.Hazard()
    opt = torch.optim.Adam(m3.parameters(), lr=2e-3, weight_decay=1e-4)
    for ep in range(40):
        perm = torch.randperm(len(X))
        tot, nb = 0.0, 0
        for b in range(0, len(X), 1024):
            ii = perm[b:b + 1024]
            logits = m3(X[ii])
            d = y[ii]
            loss = X[ii].sum() * 0.0
            neg = d == 0
            if neg.any():
                loss = loss + F.binary_cross_entropy_with_logits(
                    logits[neg], torch.zeros((int(neg.sum()), 2)))
            for k, dir_val in ((0, -1), (1, 1)):
                sel = d == dir_val
                if sel.any():
                    loss = loss + F.binary_cross_entropy_with_logits(
                        logits[sel, k], torch.ones(int(sel.sum())),
                        pos_weight=torch.tensor(20.0))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        log(f"  hazard ep{ep}: loss {tot / max(nb, 1):.4f}")
    M.save_model(out / "hazard.pt", m3, dict(
        outputs=["left", "right"],
        features="v, dist_merge, 12 slot feats, merge onehot, "
                 "posterior probs",
        train_roots=splits["train"], epochs=40))
    log("hazard saved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
