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

Over-fit control (all default OFF = the behaviour above):

* ``--es-frac F`` (> 0): early stopping.  A fraction F of the TRAIN *episodes* (runs /
  GENRAND episodes; every window of an episode goes to the same side, deterministic in
  ``--seed``) is carved out of the TRAIN windows as the early-stopping set (``--es-batches``
  batches, never subsampled); the fit uses the remaining TRAIN windows only.  After every epoch
  the early-stopping loss on that set is computed (``--es-criterion normalised``, CLI default raw since 2026-10-14 integrator:
  sum over heads of log-loss / the head's constant-rate log-loss on the same rows, so every
  head counts equally; ``raw``: the plain summed per-head log-loss); training stops after
  ``--patience`` epochs without improvement (``--min-delta``) and the best epoch's weights are
  restored.  The VAL / CAL / held-out sets are never used; the reported TRAIN-val metrics stay
  on the config's TRAIN-scoped validation windows.
* ``--weight-decay W`` (> 0): AdamW with decoupled weight decay W instead of Adam.
* ``--dropout P`` (> 0): dropout P after each trunk activation during the fit (reset to 0
  before saving; the world model never uses it).
* ``--max-minutes M``: wall-clock budget; TRAIN / early-stopping / TRAIN-val collection stop
  at 55 / 70 / 85 % of it and the fit at 100 % (the JSON records what was cut).

OUT.json records ``stop_epoch`` / ``best_epoch`` and the per-head TRAIN-val AP
(``head_ap_train_val``, also stored in the network).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

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
            stay_keep: float = 1.0, deadline: Optional[float] = None) -> dict:
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
             "infeasible_events": 0, "forced_window_events": 0, "cut_by_budget": False}
    for index, batch in enumerate(loader):
        if index >= batches:
            break
        if deadline is not None and time.time() > deadline and stats["batches"] > 0:
            stats["cut_by_budget"] = True
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


ES_CRITERIA = ("normalised", "raw")


def summed_head_logloss(net, data: dict, criterion: str = "raw") -> float:
    """Sum over heads of the per-head (weighted) mean binary log-loss of ``data``.

    ``criterion='normalised'``: each head's log-loss is divided by the log-loss of its best
    constant (its event rate) on the same rows, so every head counts equally (1 = constant)."""
    if criterion not in ES_CRITERIA:
        raise ValueError(f"criterion must be one of {ES_CRITERIA}")
    p = predict(net, data["features"], data["head"])
    total = 0.0
    for h in range(len(TRANSITIONS)):
        m = data["head"] == h
        if m.any():
            y, w = data["label"][m], data["weight"][m]
            value = log_loss(p[m], y, w)
            if criterion == "normalised":
                rate = float((w * y).sum() / max(w.sum(), 1e-12))
                value /= max(log_loss(np.full(int(m.sum()), rate), y, w), 1e-12)
            total += value
    return float(total)


def fit(net, data: dict, *, epochs: int, lr: float, seed: int, minibatch: int = 4096,
        log=print, weight_decay: float = 0.0, dropout: float = 0.0,
        early: Optional[dict] = None, patience: int = 3, min_delta: float = 0.0,
        es_criterion: str = "normalised",
        deadline: Optional[float] = None, report: Optional[dict] = None) -> list[float]:
    """Adam (AdamW when ``weight_decay`` > 0) on the per-head weighted mean binary CE (summed
    over heads), minibatches.  ``early`` (rows) enables early stopping on its summed per-head
    log-loss with ``patience`` and best-weights restore; ``deadline`` stops after the current
    epoch.  ``report`` receives stop/best epoch and the early-stopping curve."""
    import copy

    import torch

    torch.manual_seed(seed)
    x = torch.as_tensor(data["features"])
    head = torch.as_tensor(data["head"])
    y = torch.as_tensor(data["label"])
    w = torch.as_tensor(data["weight"])
    opt = (torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=float(weight_decay))
           if weight_decay > 0.0 else torch.optim.Adam(net.parameters(), lr=lr))
    net.dropout = float(dropout)
    n = len(y)
    history = []
    report = report if report is not None else {}
    report.update({"es_criterion": es_criterion if early is not None else None,
                   "stop_epoch": 0, "best_epoch": None, "stopped_by": "epochs",
                   "es_logloss": [], "optimizer": "AdamW" if weight_decay > 0.0 else "Adam",
                   "weight_decay": float(weight_decay), "dropout": float(dropout)})
    best_loss, best_state, bad = float("inf"), None, 0
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
        report["stop_epoch"] = epoch + 1
        msg = f"  epoch {epoch + 1}/{epochs}: loss {history[-1]:.5f}"
        if early is not None:
            net.eval()
            es = summed_head_logloss(net, early, es_criterion)
            net.train()
            report["es_logloss"].append(es)
            msg += f" | early-stop log-loss {es:.5f}"
            if es < best_loss - float(min_delta):
                best_loss, bad = es, 0
                best_state = copy.deepcopy(net.state_dict())
                report["best_epoch"] = epoch + 1
            else:
                bad += 1
        log(msg)
        if early is not None and bad >= int(patience):
            report["stopped_by"] = "early_stopping"
            break
        if deadline is not None and time.time() > deadline:
            report["stopped_by"] = "max_minutes"
            break
    if best_state is not None:
        net.load_state_dict(best_state)
        report["best_es_logloss"] = best_loss
    net.dropout = 0.0
    net.eval()
    return history


def episode_keys(dataset) -> list:
    """Episode key of every TRAIN window, without decoding any window.

    Pack windows: the simulated WORLD of the window (``corpus.meta``; the arms / t0 branches of
    one world share their pre-t0 traffic, so they stay together); GENRAND windows: the
    episode's world (else its directory).  A window whose episode cannot be resolved is its own
    episode."""
    inner = getattr(dataset, "inner", dataset)
    sources = getattr(inner, "sources", None)
    if sources is None:
        return [("window", i) for i in range(len(inner))]
    keys = []
    for index in range(len(inner)):
        k, local = inner.locate(index)
        source = sources[k]
        loader = getattr(source.dataset, "_loader", None)
        key = None
        corpus, kept = getattr(loader, "corpus", None), getattr(loader, "kept_orig", None)
        if corpus is not None and kept is not None and hasattr(corpus, "meta"):
            meta = corpus.meta(int(kept[local]))
            world = meta.get("world") or meta.get("pair_key")
            if world:
                key = ("world", str(world))
        entries, idx = getattr(loader, "entries", None), getattr(loader, "index", None)
        if key is None and entries is not None and idx is not None:
            entry = entries[int(idx[local][0])]
            world = entry.get("world") if isinstance(entry, dict) else None
            key = ("world", str(world)) if world else (source.name, str(entry.get("dir")))
        keys.append(key if key is not None else (source.name, "window", local))
    return keys


def split_by_episode(keys: list, frac: float, seed: int) -> tuple[list[int], list[int]]:
    """``(fit_indices, early_stop_indices)``: a deterministic ``frac`` of the episodes (hash of
    the key and ``seed``) goes to the early-stopping side, with all its windows."""
    import hashlib

    fit_idx, es_idx = [], []
    for index, key in enumerate(keys):
        digest = hashlib.sha256(f"{seed}|{key!r}".encode()).digest()
        u = int.from_bytes(digest[:8], "big") / 2.0 ** 64
        (es_idx if u < frac else fit_idx).append(index)
    return fit_idx, es_idx


class IndexedWindows:
    """Plain TRAIN-window subset (``__getitem__`` only, so no paired-batch substitution)."""

    def __init__(self, dataset, indices: list[int]) -> None:
        self.inner = getattr(dataset, "inner", dataset)
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        return self.inner[self.indices[i]]


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
    parser.add_argument("--es-frac", type=float, default=0.0,
                        help="fraction of TRAIN episodes held out for early stopping (0 = off)")
    parser.add_argument("--es-batches", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--es-criterion", choices=ES_CRITERIA, default="raw",
                        help="early-stopping loss: per-head log-loss / its constant's, summed "
                             "(normalised, default) or the plain summed log-loss (raw)")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="> 0: AdamW")
    parser.add_argument("--dropout", type=float, default=0.0, help="> 0: trunk dropout")
    parser.add_argument("--max-minutes", type=float, default=0.0, help="> 0: wall budget")
    args = parser.parse_args(argv)
    if not (0.0 <= args.es_frac < 1.0 and 0.0 <= args.dropout < 1.0 and args.weight_decay >= 0.0
            and args.patience >= 1 and args.max_minutes >= 0.0):
        raise SystemExit("bad --es-frac / --dropout / --weight-decay / --patience / --max-minutes")

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
    budget = args.max_minutes * 60.0 if args.max_minutes > 0 else None
    phase = (lambda f: None) if budget is None else (lambda f: started + f * budget)  # noqa: E731
    inputs = assemble(cfg, work)
    model = inputs.model
    model.eval()
    if inputs.val is None:
        raise SystemExit("the config has no TRAIN-scoped validation windows (data.windows.val)")
    fit_source, early, split = inputs.train, None, None
    if args.es_frac > 0.0:
        keys = episode_keys(inputs.train)
        fit_idx, es_idx = split_by_episode(keys, args.es_frac, args.seed)
        if not fit_idx or not es_idx:
            raise SystemExit("the episode split left one side empty; change --es-frac / --seed")
        split = {"episodes": len(set(keys)), "es_episodes": len({keys[i] for i in es_idx}),
                 "fit_windows": len(fit_idx), "es_windows": len(es_idx), "frac": args.es_frac}
        print(f"early-stopping split by episode: {split}", flush=True)
        fit_source = IndexedWindows(inputs.train, fit_idx)
    print(f"collect TRAIN rows ({args.train_batches} x {args.batch_size} windows)", flush=True)
    train = collect(model, fit_source, settings, batches=args.train_batches,
                    batch_size=args.batch_size, seed=args.seed, stay_keep=args.stay_keep,
                    deadline=phase(0.55))
    if split is not None:
        print(f"collect early-stopping rows ({args.es_batches} x {args.batch_size} TRAIN "
              "windows)", flush=True)
        early = collect(model, IndexedWindows(inputs.train, es_idx), settings,
                        batches=args.es_batches, batch_size=args.batch_size, seed=args.seed + 2,
                        stay_keep=1.0, deadline=phase(0.70))
    print(f"collect TRAIN-val rows ({args.val_batches} x {args.batch_size} windows)", flush=True)
    val = collect(model, inputs.val, settings, batches=args.val_batches,
                  batch_size=args.batch_size, seed=args.seed + 1, stay_keep=1.0,
                  deadline=phase(0.85))
    collected = time.time()
    net = model.lane_v5
    set_scaling(net, train["features"], train["weight"])
    fit_report: dict = {}
    history = fit(net, train, epochs=args.epochs, lr=args.lr, seed=args.seed,
                  weight_decay=args.weight_decay, dropout=args.dropout, early=early,
                  patience=args.patience, min_delta=args.min_delta, deadline=phase(1.0),
                  es_criterion=args.es_criterion,
                  report=fit_report)
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
            "stop_epoch": fit_report["stop_epoch"], "best_epoch": fit_report["best_epoch"],
            "fit": fit_report, "early_stopping_split": split,
            "early_stopping_stats": None if early is None else early["stats"],
            "max_minutes": args.max_minutes or None,
            "seconds": {"collect": round(collected - started, 1),
                        "total": round(time.time() - started, 1)}}
    if early is not None:
        metrics["early_stop_set"] = head_metrics(predict(net, early["features"], early["head"]),
                                                 early["label"], early["weight"], early["head"])
    save_pretrain(net, out, meta)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1, allow_nan=True),
                                        encoding="utf-8")
    print(json.dumps({"head_ap_train_val": ap, "metrics": metrics["train_val"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
