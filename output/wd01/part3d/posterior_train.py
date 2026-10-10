"""Part 3d B2 posterior: small torch classifier on the 10 s history
summary.  TRAIN roots only for fitting; validation for calibration.
Self-contained (no part3 engine dependency)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "partA_data"))
sys.path.insert(0, str(HERE.parent / "part3"))

import p3lib as P  # noqa: E402
import torch.nn as nn  # noqa: E402


class Posterior(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(11, 32), nn.SiLU(),
                                 nn.Linear(32, 3))

    def forward(self, hist):
        v, a = hist[..., 0], hist[..., 1]
        x = torch.stack([v[:, -1], a[:, -1], v.mean(1), v.std(1),
                         a.mean(1), a.std(1), v.min(1).values,
                         v.max(1).values, v[:, -4:].mean(1),
                         a[:, -4:].mean(1), v[:, -1] - v[:, 0]], dim=1)
        return self.net(x)


def collect(root, ticks=(240, 360, 480)):
    ra = P.root_arrays(root)
    X, Y = [], []
    for k in ticks:
        hist = P.history_at(ra, k)
        idx = ra.tick_rows(k)
        idx = idx[ra.lane_m[idx] >= 0]
        for i in idx:
            vi = int(ra.vid[i])
            if vi in hist:
                X.append(hist[vi])
                Y.append(int(ra.typ[i]))
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.int64)


def main():
    splits = P.roots_by_split()
    Xtr, Ytr = [], []
    for root in splits["train"]:
        x, y = collect(root)
        Xtr.append(x)
        Ytr.append(y)
    Xtr = torch.as_tensor(np.concatenate(Xtr), dtype=torch.float32)
    Ytr = torch.as_tensor(np.concatenate(Ytr))
    print("posterior train samples:", len(Ytr))
    m = Posterior()
    opt = torch.optim.Adam(m.parameters(), lr=2e-3, weight_decay=1e-4)
    for ep in range(30):
        perm = torch.randperm(len(Xtr))
        for b in range(0, len(Xtr), 4096):
            ii = perm[b:b + 4096]
            loss = F.cross_entropy(m(Xtr[ii]), Ytr[ii])
            opt.zero_grad()
            loss.backward()
            opt.step()
    out = HERE / "weights"
    out.mkdir(exist_ok=True)
    torch.save({"model_state": m.state_dict(),
                "meta": {"classes": [0, 1, 2],
                         "features": "hist_summary 11",
                         "train_roots": splits["train"]}},
               out / "posterior.pt")
    logloss = acc = acc_prior = 0
    n = 0
    with torch.no_grad():
        for root in splits["val"]:
            x, y = collect(root)
            if len(y) == 0:
                continue
            xt = torch.as_tensor(x, dtype=torch.float32)
            yt = torch.as_tensor(y)
            logp = F.log_softmax(m(xt), dim=1)
            logloss += float(-logp[torch.arange(len(yt)), yt].sum())
            n += len(yt)
            acc += int((logp.argmax(1) == yt).sum())
            acc_prior += int((torch.as_tensor(1) == yt).sum())
    print(f"val log loss {logloss / max(n, 1):.4f}  "
          f"acc {acc / max(n, 1):.4f}  prior acc {acc_prior / max(n, 1):.4f}"
          f"  n {n}")
    torch.save(m.state_dict(), HERE / "weights" / "posterior_final.pt")


if __name__ == "__main__":
    main()
