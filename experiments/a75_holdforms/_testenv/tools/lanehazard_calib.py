"""LANEFIX: teacher-forced calibration check of a WM-S hazard lane model on TRAIN windows.

For every recorded HDV-step of the TRAIN windows of ``--config`` (default the C4 v3 data mix;
TRAIN only: the validation / held-out / wall / CAL sets are never read) the hazard model is
evaluated on the RECORDED state (exactly as ``loss.lane_event`` does) and every adjacent
lane-change direction becomes one row ``(p, y, category, masked)``:

* per category (overtake 1->0, return 0->1, merge 2->1): rows, events, sum of p, the
  predicted / observed ratio, the share of events the physics mask forbids, the AUC on
  feasible rows and the feasible-row binary log-loss next to the best CONSTANT rate's;
* pooled feasible rows (all three categories): log-loss vs constant and a reliability table.

A calibrated, well-ranked hazard has p/obs ~ 1 and a log-loss below the constant's (the
2026-10-14 diagnosis found 4-10x over-prediction and a log-loss 0.071 vs 0.031 for V3s60).

Usage (workspace root as CWD, data via ``--data-root`` as in the smokes)::

    python tools/lanehazard_calib.py CHECKPOINT [--config configs/c4_wm_v3.toml]
        [--batches 40] [--batch-size 4] [--seed 7] [--data-root E:/research-dense]
        [--out result.json]

The checkpoint's ``physics.json`` must sit next to it (the loader warns otherwise).

LANEV5 checkpoints (``model.lane_model='hazard_v5'``): the per-head model is scored on the
teacher-forced rows of ``loss.lane_event`` (every recorded step; feasible rows of each car's
own road transition outside the forced window) of the config's TRAIN-scoped validation windows
(``--split val``, default for v5; ``--split train`` = TRAIN windows): per head (overtake /
return / merge) rows, events, p/obs, AUC, AP (vs the constant's), log-loss vs the best
constant and a reliability table, with the model's ``lane_logit_offsets`` applied.
``--write-ap OUT.pt`` writes a copy of the checkpoint whose lane network stores these per-head
APs (``head_ap``, read by ``model.lane_decision='gated'``) and copies ``physics.json``.
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

CATEGORIES = {1: "overtake 1->0", 2: "return 0->1", 3: "merge 2->1"}
_CAT_OF = {(1, 0): 1, (0, 1): 2, (2, 1): 3}
#: ``_CAT_TABLE[from, to]`` = category (4 = other adjacent change).
_CAT_TABLE = np.full((3, 3), 4.0)
for (_a, _b), _c in _CAT_OF.items():
    _CAT_TABLE[_a, _b] = float(_c)
_EPS = 1.0e-7


def transition_rows(lanes: np.ndarray, valid: np.ndarray, probs: np.ndarray,
                    masked: np.ndarray) -> np.ndarray:
    """Rows ``[N, 4] = (p, y, category, masked)`` of one window.

    ``lanes``/``valid`` are ``[T+1, K]`` (valid = supervised HDV), ``probs``/``masked`` are
    ``[T, K, 3]`` at the recorded state of step t.  A transition t -> t+1 is used when both
    ends are valid and it moves at most one lane; each existing adjacent target gives a row.
    """
    rows = []
    prev, nxt = lanes[:-1], lanes[1:]
    ok = valid[:-1] & valid[1:] & (np.abs(nxt - prev) <= 1)
    for offset in (-1, 1):
        target = prev + offset
        use = ok & (target >= 0) & (target <= 2)
        t_idx, k_idx = np.nonzero(use)
        a, b = prev[t_idx, k_idx], target[t_idx, k_idx]
        p = probs[t_idx, k_idx, b]
        y = (nxt[t_idx, k_idx] == b).astype(np.float64)
        cat = _CAT_TABLE[a.astype(np.int64), b.astype(np.int64)]
        m = masked[t_idx, k_idx, b].astype(np.float64)
        rows.append(np.stack((p, y, cat, m), -1) if len(p) else np.zeros((0, 4)))
    return np.concatenate(rows, 0)


def auc(p: np.ndarray, y: np.ndarray) -> float:
    """Rank AUC (ties averaged); NaN without both classes."""
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    sorted_p = p[order]
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, _EPS, 1 - _EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()) if len(p) else float("nan")


def summarize(rows: np.ndarray) -> dict:
    """Calibration / ranking summary of :func:`transition_rows` output."""
    p, y, cat, masked = rows.T
    masked = masked.astype(bool)
    out: dict = {"rows": int(len(p)), "events": int(y.sum()), "categories": {}}
    for code, name in CATEGORIES.items():
        sel = cat == code
        if not sel.any():
            continue
        ev = sel & (y == 1)
        feas = sel & ~masked
        base = float(y[feas].mean()) if feas.any() else float("nan")
        out["categories"][name] = {
            "rows": int(sel.sum()), "events": int(ev.sum()), "sum_p": float(p[sel].sum()),
            "p_over_obs": float(p[sel].sum() / max(1.0, ev.sum())),
            "masked_at_event": float(masked[ev].mean()) if ev.any() else float("nan"),
            "auc_feasible": auc(p[feas], y[feas]),
            "feasible_rows": int(feas.sum()), "feasible_rate": base,
            "logloss_feasible": log_loss(p[feas], y[feas]),
            "logloss_constant": log_loss(np.full(int(feas.sum()), base), y[feas])
            if feas.any() else float("nan"),
        }
    feas = np.isin(cat, list(CATEGORIES)) & ~masked
    base = float(y[feas].mean()) if feas.any() else float("nan")
    out["pooled_feasible"] = {
        "rows": int(feas.sum()), "rate": base, "sum_p": float(p[feas].sum()),
        "events": int(y[feas].sum()),
        "logloss": log_loss(p[feas], y[feas]),
        "logloss_constant": log_loss(np.full(int(feas.sum()), base), y[feas])
        if feas.any() else float("nan")}
    edges = [0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 0.5, 1.0001]
    out["reliability"] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = feas & (p >= lo) & (p < hi)
        if b.any():
            out["reliability"].append({"bin": [lo, hi], "n": int(b.sum()),
                                       "mean_p": float(p[b].mean()), "obs": float(y[b].mean())})
    return out


def collect(model, dataset, *, batches: int, batch_size: int, seed: int, settings,
            max_steps: int = 0) -> np.ndarray:
    """Teacher-forced rows of ``batches`` shuffled TRAIN batches (first ``max_steps`` steps of
    each window when > 0)."""
    import torch
    from torch.utils.data import DataLoader

    from wms.data.collate import collate
    from wms.train.inputs import prepare_inputs
    from wms.train.losses.lane_event import (recorded_event_logits, recorded_lane_age,
                                             recorded_lane_blocked)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate,
                        num_workers=0, generator=torch.Generator().manual_seed(seed))
    out = []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= batches:
                break
            n = int(batch["states"].shape[1]) - 1
            if max_steps > 0:
                n = min(n, int(max_steps))
            inp = prepare_inputs(batch, model, settings, torch.device("cpu"), n)
            _, embedding, params = model.encode_drivers(
                inp.initial, driver_history=inp.observed_hist,
                driver_history_mask=inp.observed_hist_mask)
            states = inp.states[:, :n + 1]
            present = inp.loss_valid[:, :n + 1].bool()
            is_cav = inp.is_cav[:, :n + 1].bool()
            lanes = states[..., 1:4].argmax(-1)
            valid = present & ~is_cav
            age0 = model._initial_lane_age({}, inp.present[:, 0], inp.states[:, 0],
                                           inp.observed_hist, inp.observed_hist_mask)
            age = recorded_lane_age(lanes, valid, age0, n)
            blocked = None
            if getattr(model, "_tracks_lane_blocked", False):
                blocked0 = model.initial_lane_blocked(inp.present[:, 0], inp.states[:, 0],
                                                      inp.observed_hist, inp.observed_hist_mask,
                                                      params)
                blocked = recorded_lane_blocked(model.physics, states[:, :n], present[:, :n],
                                                params, blocked0)
            logits = recorded_event_logits(model, states[:, :n], present[:, :n], is_cav[:, :n],
                                           params, embedding, age, blocked)
            probs = torch.softmax(logits, -1).numpy()
            masked = (logits <= -8.0 + 1.0e-6).numpy()
            for b in range(states.shape[0]):
                out.append(transition_rows(lanes[b].numpy(), valid[b].numpy(), probs[b],
                                           masked[b]))
            print(f"batch {index}: rows {sum(len(r) for r in out)}", flush=True)
    return np.concatenate(out, 0) if out else np.zeros((0, 4))


def summarize_v5(prob: np.ndarray, label: np.ndarray, head: np.ndarray) -> dict:
    """LANEV5 per-head calibration / ranking summary (+ reliability per head)."""
    from lanev5_pretrain import TRANSITIONS, head_metrics

    w = np.ones_like(label, dtype=np.float64)
    out = {"rows": int(len(label)), "events": int(label.sum()),
           "heads": head_metrics(prob, label, w, head), "reliability": {}}
    edges = [0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 0.5, 1.0001]
    for h, name in enumerate(TRANSITIONS):
        table = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            b = (head == h) & (prob >= lo) & (prob < hi)
            if b.any():
                table.append({"bin": [lo, hi], "n": int(b.sum()), "mean_p": float(prob[b].mean()),
                              "obs": float(label[b].mean())})
        out["reliability"][name] = table
    return out


def main_v5(args, model, cfg, settings, work: Path, started: float) -> int:
    """LANEV5 branch of :func:`main`."""
    import shutil

    import torch

    from lanev5_pretrain import TRANSITIONS, collect
    from wms.cli.assemble import build_datasets

    train, val, _ = build_datasets(cfg, work)
    dataset = val if args.split == "val" else train
    if dataset is None:
        raise SystemExit("the config has no TRAIN-scoped validation windows (data.windows.val)")
    rows = collect(model, dataset, settings, batches=args.batches, batch_size=args.batch_size,
                   seed=args.seed, stay_keep=1.0)
    offsets = np.asarray(getattr(model, "lane_logit_offsets", (0.0, 0.0, 0.0)), np.float32)
    with torch.no_grad():
        logits = []
        for i in range(0, len(rows["head"]), 65536):
            x = torch.as_tensor(rows["features"][i:i + 65536])
            h = torch.as_tensor(rows["head"][i:i + 65536])
            logits.append(model.lane_v5(x).gather(-1, h.unsqueeze(-1)).squeeze(-1).numpy())
    logit = (np.concatenate(logits) if logits else np.zeros(0)) + offsets[rows["head"]]
    prob = 1.0 / (1.0 + np.exp(-logit))
    summary = summarize_v5(prob, rows["label"], rows["head"])
    result = {"checkpoint": str(args.checkpoint), "config": args.config,
              "split": "TRAIN-val" if args.split == "val" else "TRAIN",
              "batches": args.batches, "batch_size": args.batch_size, "seed": args.seed,
              "lane_model": "hazard_v5", "lane_logit_offsets": offsets.tolist(),
              "head_ap_stored": model.lane_v5.head_ap.tolist(), "row_stats": rows["stats"],
              "seconds": round(time.time() - started, 1), **summary}
    if args.write_ap:
        ap = [summary["heads"][n].get("ap", float("nan")) for n in TRANSITIONS]
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        payload["model"]["lane_v5.head_ap"] = torch.tensor(
            [a if math.isfinite(a) else -1.0 for a in ap], dtype=torch.float32)
        payload["lanev5_head_ap_source"] = {"split": result["split"], "batches": args.batches,
                                            "seed": args.seed, "ap": ap}
        target = Path(args.write_ap)
        torch.save(payload, target)
        physics = Path(args.checkpoint).parent / "physics.json"
        if physics.is_file() and not (target.parent / "physics.json").exists():
            shutil.copy2(physics, target.parent / "physics.json")
        result["wrote_ap"] = {"path": str(target), "ap": ap}
    text = json.dumps(result, indent=1, allow_nan=True)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint")
    parser.add_argument("--config", default="configs/c4_wm_v3.toml")
    parser.add_argument("--batches", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="score only the first N 0.5 s steps of each window (0 = all)")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--work-dir", default=None,
                        help="scratch directory of the dataset assembly (default: next to --out)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--split", choices=("val", "train"), default=None,
                        help="hazard_v5 only: TRAIN-val (default) or TRAIN windows")
    parser.add_argument("--write-ap", default=None,
                        help="hazard_v5 only: write a checkpoint copy storing the per-head AP")
    args = parser.parse_args()

    from wms.core.guard import find_restricted_token
    for text in (args.checkpoint, args.config, args.out or "", args.write_ap or ""):
        token = find_restricted_token(str(text))
        if token is not None:
            raise SystemExit(f"refused: {text!r} contains restricted token {token!r}")
    if args.data_root:
        from wmlane_smoke import point_data
        point_data(Path(args.data_root))
    import torch

    from wms.cli.assemble import build_datasets
    from wms.config.load import load_train_config
    from wms.model.checkpoint import load_checkpoint
    from wms.train.settings import TrainSettings

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    model = load_checkpoint(args.checkpoint).model
    model.eval()
    if getattr(model, "lane_v5", None) is not None:
        cfg = load_train_config(args.config, ["runtime.device=cpu"])
        work = Path(args.work_dir or (Path(args.out).parent / "calib_work" if args.out
                                      else "runs/active/lanehazard_calib_work"))
        work.mkdir(parents=True, exist_ok=True)
        args.split = args.split or "val"
        return main_v5(args, model, cfg, TrainSettings.from_config(cfg), work, time.time())
    if args.split == "val" or args.write_ap:
        raise SystemExit("--split val / --write-ap need a hazard_v5 checkpoint")
    if getattr(model, "lane_hazard", None) is None:
        raise SystemExit("the checkpoint has no hazard lane model (model.lane_model='hazard')")
    cfg = load_train_config(args.config, ["runtime.device=cpu"])
    settings = TrainSettings.from_config(cfg)
    work = Path(args.work_dir or (Path(args.out).parent / "calib_work" if args.out
                                  else "runs/active/lanehazard_calib_work"))
    work.mkdir(parents=True, exist_ok=True)
    started = time.time()
    train, _, _ = build_datasets(cfg, work)   # TRAIN windows only; the val split is unused
    rows = collect(model, train, batches=args.batches, batch_size=args.batch_size,
                   seed=args.seed, settings=settings, max_steps=args.max_steps)
    result = {"checkpoint": str(args.checkpoint), "config": args.config, "split": "TRAIN",
              "batches": args.batches, "batch_size": args.batch_size, "seed": args.seed,
              "max_steps": args.max_steps,
              "switches": {name: getattr(model, name, None) for name in (
                  "lane_decision", "lane_logit_offsets", "lane_hazard_features",
                  "merge_hazard_zone")},
              "seconds": round(time.time() - started, 1), **summarize(rows)}
    text = json.dumps(result, indent=1, allow_nan=True)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0 if all(math.isfinite(result["pooled_feasible"][k]) for k in ("logloss",)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
