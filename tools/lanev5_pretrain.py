"""LANEV5: offline teacher-forced pre-fit of the per-transition lane network.

Fits the SAME network the world model uses (``wms.model.lane_v5.LaneV5Net``, built from
``--config``) on the teacher-forced rows of TRAIN windows of the config's C4 data mix,
exactly the rows / features of ``loss.lane_event`` for ``model.lane_model='hazard_v5'``
(``wms.train.losses.lane_event.lane_v5_batch_rows``: every recorded step, feasible rows of
each car's own road transition outside the forced end-of-lane window).  The input scaling
(signed log + standardisation) is set from the TRAIN rows; the per-head teacher-forced AP on
the config's TRAIN-scoped validation windows (``data.windows.val``, held-apart TRAIN worlds;
never wall / held-out / CAL sets) is stored in the network's ``head_ap`` buffer, which
``model.lane_decision='gated'`` reads.  Load the output with ``model.lane_v5_pretrain``.

Stay rows of the overtake / return heads are subsampled at ``--stay-keep`` with inverse
weights (the merge head keeps every row); the loss is the training loss (per-head mean binary
CE, summed over heads), weighted accordingly.  Evaluation rows are never subsampled.

Usage (workspace root as CWD; a git worktree points its data at a checkout via --data-root)::

    python tools/lanev5_pretrain.py OUT.pt [--config configs/c4_wm_v5_lane.toml]
        [--data-root E:/research-dense] [--train-batches 40] [--val-batches 10]
        [--batch-size 4] [--epochs 25] [--lr 2e-3] [--stay-keep 0.25] [--seed 0]

Writes OUT.pt (``lanev5_pretrain/1``) and OUT.json (row counts, per-head TRAIN-fit and
TRAIN-val metrics: AUC, AP, log-loss vs constant, p/obs).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np  # noqa: E402

TRANSITIONS = ("overtake", "return", "merge")
_EPS = 1.0e-7


# ------------------------------------------------------------------ metrics
def weighted_auc(score: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted ROC AUC (ties broken by order; NaN without both classes)."""
    pos, neg = float((w * y).sum()), float((w * (1 - y)).sum())
    if pos <= 0 or neg <= 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    yw, nw = (w * y)[order], (w * (1 - y))[order]
    tpr = np.concatenate(([0.0], np.cumsum(yw) / pos))
    fpr = np.concatenate(([0.0], np.cumsum(nw) / neg))
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(tpr, fpr))


def weighted_ap(score: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted average precision (sum over positives of precision at their rank)."""
    pos = float((w * y).sum())
    if pos <= 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    yw, ww = (w * y)[order], w[order]
    precision = np.cumsum(yw) / np.cumsum(ww).clip(min=1e-12)
    return float((precision * yw).sum() / pos)


def log_loss(p: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    p = np.clip(p, _EPS, 1 - _EPS)
    return float(-(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / max(w.sum(), 1e-12))


def head_metrics(p: np.ndarray, y: np.ndarray, w: np.ndarray, head: np.ndarray) -> dict:
    """Per head: rows, events, rate, p/obs, AUC, AP (and the constant's AP), log-loss vs the
    best constant."""
    out = {}
    for h, name in enumerate(TRANSITIONS):
        m = head == h
        if not m.any():
            out[name] = {"rows": 0}
            continue
        pm, ym, wm = p[m], y[m], w[m]
        rate = float((wm * ym).sum() / max(wm.sum(), 1e-12))
        out[name] = {"rows": int(m.sum()), "events": int(ym.sum()), "rate": rate,
                     "p_over_obs": float((wm * pm).sum() / max((wm * ym).sum(), 1e-12)),
                     "auc": weighted_auc(pm, ym, wm), "ap": weighted_ap(pm, ym, wm),
                     "ap_constant": rate, "logloss": log_loss(pm, ym, wm),
                     "logloss_constant": log_loss(np.full_like(pm, rate), ym, wm)}
    return out


# ------------------------------------------------------------------ rows
def collect(model, dataset, settings, *, batches: int, batch_size: int, seed: int,
            stay_keep: float = 1.0) -> dict:
    """Teacher-forced rows of ``batches`` shuffled batches: features [N,F], head, label, weight."""
    import torch
    from torch.utils.data import DataLoader

    from wms.data.collate import collate
    from wms.train.losses.lane_event import lane_v5_batch_rows

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate,
                        num_workers=0, generator=torch.Generator().manual_seed(seed))
    rng = np.random.default_rng(seed + 17)
    parts = {"features": [], "head": [], "label": [], "weight": []}
    stats = {"batches": 0, "windows": 0, "steps": 0, "rows_all": 0, "events": 0,
             "infeasible_events": 0, "forced_window_events": 0}
    for index, batch in enumerate(loader):
        if index >= batches:
            break
        rows = lane_v5_batch_rows(model, batch, settings)
        feats = rows["features"].numpy().astype(np.float32)
        head = rows["head"].numpy().astype(np.int64)
        label = rows["label"].numpy().astype(np.float32)
        weight = np.ones_like(label)
        if stay_keep < 1.0:
            stay = (label < 0.5) & (head != 2)
            keep = ~stay | (rng.random(len(label)) < stay_keep)
            weight = np.where(stay, 1.0 / stay_keep, 1.0).astype(np.float32)
            feats, head, label, weight = feats[keep], head[keep], label[keep], weight[keep]
        for key, value in (("features", feats), ("head", head), ("label", label),
                           ("weight", weight)):
            parts[key].append(value)
        stats["batches"] += 1
        stats["windows"] += int(batch["states"].shape[0])
        stats["steps"] += int(batch["states"].shape[0]) * (int(batch["states"].shape[1]) - 1)
        for key in ("events", "infeasible_events", "forced_window_events"):
            stats[key] += int(rows["stats"][key])
        stats["rows_all"] += int(rows["stats"]["rows"])
        print(f"  batch {index + 1}/{batches}: rows {rows['stats']['rows']} "
              f"events {rows['stats']['events']}", flush=True)
    out = {key: np.concatenate(value) if value else np.zeros((0,)) for key, value in parts.items()}
    out["stats"] = stats
    return out


# ------------------------------------------------------------------ fit
def set_scaling(net, features: np.ndarray, weight: np.ndarray) -> None:
    """Feature standardisation buffers from the (weighted) signed-log TRAIN rows."""
    import torch

    x = np.sign(features) * np.log1p(np.abs(features))
    w = weight[:, None] / max(weight.sum(), 1e-12)
    mu = (w * x).sum(0)
    sd = np.sqrt((w * (x - mu) ** 2).sum(0))
    sd = np.where(sd < 1e-3, 1.0, sd)
    with torch.no_grad():
        net.feature_mu.copy_(torch.as_tensor(mu, dtype=net.feature_mu.dtype))
        net.feature_sd.copy_(torch.as_tensor(sd, dtype=net.feature_sd.dtype))


def fit(net, data: dict, *, epochs: int, lr: float, seed: int, minibatch: int = 4096,
        log=print) -> list[float]:
    """Adam on the per-head weighted mean binary CE (summed over heads), minibatches."""
    import torch

    torch.manual_seed(seed)
    x = torch.as_tensor(data["features"])
    head = torch.as_tensor(data["head"])
    y = torch.as_tensor(data["label"])
    w = torch.as_tensor(data["weight"])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(y)
    history = []
    net.train()
    for epoch in range(int(epochs)):
        perm = torch.randperm(n)
        total, count = 0.0, 0
        for i in range(0, n, minibatch):
            idx = perm[i:i + minibatch]
            logit = net(x[idx]).gather(-1, head[idx].unsqueeze(-1)).squeeze(-1)
            loss = logit.sum() * 0.0
            for h in range(len(TRANSITIONS)):
                sel = head[idx] == h
                if bool(sel.any()):
                    bce = torch.nn.functional.binary_cross_entropy_with_logits(
                        logit[sel], y[idx][sel], weight=w[idx][sel], reduction="sum")
                    loss = loss + bce / w[idx][sel].sum().clamp_min(1e-8)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            count += 1
        history.append(total / max(count, 1))
        log(f"  epoch {epoch + 1}/{epochs}: loss {history[-1]:.5f}")
    net.eval()
    return history


def predict(net, features: np.ndarray, head: np.ndarray) -> np.ndarray:
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(head), 65536):
            x = torch.as_tensor(features[i:i + 65536])
            h = torch.as_tensor(head[i:i + 65536])
            out.append(torch.sigmoid(net(x).gather(-1, h.unsqueeze(-1)).squeeze(-1)).numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out")
    parser.add_argument("--config", default="configs/c4_wm_v5_lane.toml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--train-batches", type=int, default=40)
    parser.add_argument("--val-batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--stay-keep", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--work-dir", default=None)
    args = parser.parse_args(argv)

    from wms.core.guard import find_restricted_token
    for text in (args.out, args.config, args.work_dir or ""):
        token = find_restricted_token(str(text))
        if token is not None:
            raise SystemExit(f"refused: {text!r} contains restricted token {token!r}")
    if args.data_root:
        from wmlane_smoke import point_data
        point_data(Path(args.data_root))
    import torch

    from wms.cli.assemble import assemble
    from wms.config.load import load_train_config
    from wms.model.lane_v5 import save_pretrain
    from wms.train.settings import TrainSettings

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = load_train_config(args.config, ["runtime.device=cpu"])
    settings = TrainSettings.from_config(cfg)
    if cfg.model.lane_model != "hazard_v5":
        raise SystemExit("the config must set model.lane_model='hazard_v5'")
    work = Path(args.work_dir or out.with_suffix("").as_posix() + "_work")
    work.mkdir(parents=True, exist_ok=True)
    started = time.time()
    inputs = assemble(cfg, work)
    model = inputs.model
    model.eval()
    if inputs.val is None:
        raise SystemExit("the config has no TRAIN-scoped validation windows (data.windows.val)")
    print(f"collect TRAIN rows ({args.train_batches} x {args.batch_size} windows)", flush=True)
    train = collect(model, inputs.train, settings, batches=args.train_batches,
                    batch_size=args.batch_size, seed=args.seed, stay_keep=args.stay_keep)
    print(f"collect TRAIN-val rows ({args.val_batches} x {args.batch_size} windows)", flush=True)
    val = collect(model, inputs.val, settings, batches=args.val_batches,
                  batch_size=args.batch_size, seed=args.seed + 1, stay_keep=1.0)
    collected = time.time()
    net = model.lane_v5
    set_scaling(net, train["features"], train["weight"])
    history = fit(net, train, epochs=args.epochs, lr=args.lr, seed=args.seed)
    metrics = {"train_fit": head_metrics(predict(net, train["features"], train["head"]),
                                         train["label"], train["weight"], train["head"]),
               "train_val": head_metrics(predict(net, val["features"], val["head"]),
                                         val["label"], val["weight"], val["head"])}
    ap = [metrics["train_val"][name].get("ap", float("nan")) for name in TRANSITIONS]
    with torch.no_grad():
        net.head_ap.copy_(torch.tensor([a if math.isfinite(a) else -1.0 for a in ap]))
    meta = {"config": args.config, "seed": args.seed, "epochs": args.epochs, "lr": args.lr,
            "stay_keep": args.stay_keep, "rows": int(len(train["label"])),
            "windows": {"train": train["stats"]["windows"], "val": val["stats"]["windows"]},
            "train_stats": train["stats"], "val_stats": val["stats"], "loss_history": history,
            "metrics": metrics, "head_ap_train_val": ap,
            "seconds": {"collect": round(collected - started, 1),
                        "total": round(time.time() - started, 1)}}
    save_pretrain(net, out, meta)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1, allow_nan=True),
                                        encoding="utf-8")
    print(json.dumps({"head_ap_train_val": ap, "metrics": metrics["train_val"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
