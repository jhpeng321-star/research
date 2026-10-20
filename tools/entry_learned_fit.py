"""Fit the V10INSERT learned entrance insertion speed / acceptance hazard.

Data: the TRAIN-val episodes of ``merge_calib/data/{v5_native,genrand_plain}`` (converted
SUMO recordings; evaluation seeds 964000-964199 are refused).  The targets are the
recorded SUMO insertion events:

* speed: ``v0`` at the entrant's first tick, predicted from the state one tick earlier
  (bumper gap and speed of the nearest downstream vehicle in the entry lane, backlog,
  demand, lane cap, entrance);
* hazard: at every tick with a positive entrance backlog, whether at least one entrant
  was inserted on the next tick.  The fitted per-entrance logit offset calibrates the
  pooled insertion rate to SUMO's.

Usage (laptop CPU, no GPU, no SUMO):
  python tools/entry_learned_fit.py --out src/wms/model/entry_learned.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from wms.model.entry_learned import LearnedEntryModel, _MLP, state_features  # noqa: E402
from wms.model.entries import ENTRY_S_M, SUMO_LANE_SPEED_MPS  # noqa: E402

EVAL_SEEDS = range(964000, 964200)
DT_S = 0.5
NO_GAP_M = 500.0
DEMAND_WINDOW_S = 300.0


def world_case(name: str) -> tuple[int, int, float]:
    text = name.split("__", 1)[0]
    if not text.startswith("w"):
        raise ValueError(f"not a C4 world name: {name}")
    rest = text[1:]
    seed_s, rest = rest.split("m", 1)
    demand_s, rest = rest.split("p", 1)
    pen_s, _ = rest.split("t", 1)
    return int(seed_s), int(demand_s), int(pen_s) / 100.0


def _first_insertions(mask: np.ndarray, vid: np.ndarray, lane: np.ndarray):
    """``(tick, slot)`` of every first appearance after tick 0 (a SUMO insertion)."""
    new = mask[1:] & (~mask[:-1] | (vid[1:] != vid[:-1]))
    ti, ki = np.nonzero(new)
    tick = ti + 1
    return tick, ki, lane[tick, ki]


def _lane_rear(kin: np.ndarray, mask: np.ndarray, lane: np.ndarray, ln: int):
    """Per-tick ``(bumper gap from lane start, leader speed)`` of the rearmost vehicle on ``ln``."""
    present = mask & (lane == ln)
    s = np.where(present, kin[..., 0], np.inf)
    idx = np.argmin(s, axis=1)
    rows = np.arange(s.shape[0])
    any_present = np.isfinite(s[rows, idx])
    s_rear = np.where(any_present, s[rows, idx], np.nan)
    l_rear = np.where(any_present, kin[rows, idx, 6], np.nan)
    v_rear = np.where(any_present, kin[rows, idx, 4], np.nan)
    gap = s_rear - l_rear - float(ENTRY_S_M[ln])
    gap = np.where(any_present, gap, NO_GAP_M)
    v_rear = np.where(any_present, v_rear, 0.0)
    return gap, v_rear


def _demand_rates(due: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    out = np.zeros_like(due, dtype=np.float64)
    for i, t in enumerate(time_s):
        n = max(1, min(30, int(float(t) // 10.0)))
        window = n * 10.0
        lo = np.searchsorted(time_s, float(t) - window, side="left")
        out[i] = max(0.0, float(due[i] - due[lo])) / window
    return out


def episode_rows(path: Path, *, max_ticks: int | None = None):
    meta = json.loads((path / "episode.json").read_text(encoding="utf-8"))
    world = meta["world"]
    seed, demand, pen = world_case(world)
    if seed in EVAL_SEEDS:
        raise ValueError(f"refusing evaluation seed {seed}")
    kin = np.load(path / "veh_kin.npy", mmap_mode="r")
    mask = np.load(path / "veh_mask.npy", mmap_mode="r").astype(bool)
    vid = np.load(path / "vid.npy", mmap_mode="r")
    time_s = np.load(path / "time_s.npy")
    n = min(kin.shape[0], int(max_ticks) if max_ticks else kin.shape[0])
    kin = np.asarray(kin[:n], dtype=np.float64)
    mask = np.asarray(mask[:n])
    vid = np.asarray(vid[:n])
    time_s = np.asarray(time_s[:n], dtype=np.float64)
    lane = np.where(mask, kin[..., 1:4].argmax(-1), -1)
    tick, slot, ent_lane = _first_insertions(mask, vid, lane)
    ent = np.where(ent_lane == 2, 1, 0)
    ins = np.zeros((n, 2), np.int64)
    np.add.at(ins, (tick, ent), 1)
    cum = np.cumsum(ins, axis=0)

    from traffic_shepherd.scenarios import request_schedule

    req = request_schedule(seed, demand, penetration=pen, role="exposed")
    dep = []
    for name in ("mainline", "ramp"):
        dep.append(np.asarray(sorted(r.depart_s for r in req
                                     if r.vehicle_id.startswith(name)), np.float64))
    due = np.stack([np.searchsorted(dep[e], time_s + 1.0e-9, side="right") for e in (0, 1)], 1)
    backlog = np.maximum(due - cum, 0).astype(np.float64)
    demand_rate = np.stack([_demand_rates(due[:, e], time_s) for e in (0, 1)], 1)

    gaps = {}
    leads = {}
    for ln in (0, 1, 2):
        gaps[ln], leads[ln] = _lane_rear(kin, mask, lane, ln)
    gap_e = np.stack([np.maximum(gaps[0], gaps[1]), gaps[2]], 1)
    lead_e = np.stack([np.where(gaps[1] >= gaps[0], leads[1], leads[0]), leads[2]], 1)
    cap_e = np.array([SUMO_LANE_SPEED_MPS[0], SUMO_LANE_SPEED_MPS[2]], np.float64)

    speed_rows = []
    for t, k, ln in zip(tick, slot, ent_lane):
        e = 1 if ln == 2 else 0
        if t < 1:
            continue
        length_new = float(kin[t, k, 6])
        gap = float(gaps[int(ln)][t - 1]) - length_new
        speed_rows.append((gap, float(leads[int(ln)][t - 1]), float(backlog[t - 1, e]),
                           float(demand_rate[t - 1, e]), float(cap_e[e]), e,
                           float(kin[t, k, 4]), seed))
    hazard_rows = []
    label = np.zeros((n, 2), np.int64)
    label[:-1] = (ins[1:] > 0)
    for t in range(n - 1):
        for e in (0, 1):
            if backlog[t, e] <= 0.0:
                continue
            hazard_rows.append((float(gap_e[t, e]), float(lead_e[t, e]), float(backlog[t, e]),
                                float(demand_rate[t, e]), float(cap_e[e]), e,
                                int(label[t + 1, e]), seed))
    return speed_rows, hazard_rows


def load_dataset(data_root: Path, sources, max_episodes: int | None):
    speed, hazard = [], []
    seen = 0
    for source in sources:
        root = data_root / source
        names = [line.strip() for line in (root / "val_list.txt").read_text().splitlines()
                 if line.strip()]
        for name in names:
            if max_episodes is not None and seen >= int(max_episodes):
                break
            path = root / name
            if not path.is_dir():
                continue
            s_rows, h_rows = episode_rows(path)
            speed.extend(s_rows)
            hazard.extend(h_rows)
            seen += 1
            if seen % 10 == 0:
                print(f"loaded {seen} episodes: {len(speed)} insertions, "
                      f"{len(hazard)} hazard rows", flush=True)
        if max_episodes is not None and seen >= int(max_episodes):
            break
    return (np.asarray(speed, np.float64), np.asarray(hazard, np.float64))


def _standardize(x):
    mean = x.mean(0)
    scale = x.std(0)
    scale = np.where(scale > 1.0e-6, scale, 1.0)
    return mean, scale


def _train_mlp(x, y, *, hidden: int, seed: int, classification: bool,
               out_scale: float = 1.0, epochs: int = 300, lr: float = 1.0e-3):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    mean, scale = _standardize(x)
    xt = torch.as_tensor((x - mean) / scale, dtype=torch.float32)
    yt = torch.as_tensor(y / out_scale, dtype=torch.float32).reshape(-1, 1)
    net = nn.Sequential(nn.Linear(x.shape[1], hidden), nn.Tanh(), nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1.0e-5)
    loss_fn = nn.BCEWithLogitsLoss() if classification else nn.SmoothL1Loss(beta=0.5)
    n = xt.shape[0]
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        order = torch.randperm(n, generator=gen)
        for start in range(0, n, 4096):
            idx = order[start:start + 4096]
            opt.zero_grad()
            out = net(xt[idx])
            loss = loss_fn(out, yt[idx])
            loss.backward()
            opt.step()
    with torch.no_grad():
        out = net(xt).numpy().reshape(-1)
    w1 = net[0].weight.detach().numpy().T.copy()
    b1 = net[0].bias.detach().numpy().copy()
    w2 = net[2].weight.detach().numpy().reshape(-1).copy()
    b2 = float(net[2].bias.detach().numpy().reshape(-1)[0])
    return _MLP(mean=mean, scale=scale, w1=w1, b1=b1, w2=w2, b2=b2,
                out_scale=float(out_scale)), out


def _calibrate_offset(logits: np.ndarray, labels: np.ndarray, target: float) -> float:
    lo, hi = -20.0, 20.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        pred = 1.0 / (1.0 + np.exp(-(logits + mid)))
        if pred.mean() < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(
        Path("E:/research-dense/runs/active/C4-V7FT-20261017/merge_calib/data")))
    ap.add_argument("--sources", default="v5_native,genrand_plain")
    ap.add_argument("--out", default=str(ROOT / "src/wms/model/entry_learned.json"))
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--val-frac", type=float, default=0.2)
    a = ap.parse_args(argv)

    speed, hazard = load_dataset(Path(a.data_root), a.sources.split(","), a.max_episodes)
    if speed.size == 0 or hazard.size == 0:
        raise SystemExit("no fitting rows loaded")
    print(f"dataset: {len(speed)} speed rows, {len(hazard)} hazard rows")

    def split(rows):
        val = np.asarray([hash_seed(int(s)) % 100 < int(a.val_frac * 100) for s in rows[:, -1]])
        return rows[~val], rows[val]

    train_s, val_s = split(speed)
    train_h, val_h = split(hazard)

    def features(rows):
        return state_features(rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3],
                              rows[:, 4], rows[:, 5])

    x_s, y_s = features(train_s), train_s[:, 6]
    speed_net, _ = _train_mlp(x_s, y_s, hidden=a.hidden, seed=a.seed,
                                        classification=False, out_scale=10.0,
                                        epochs=a.epochs)
    x_h, y_h = features(train_h), train_h[:, 6]
    hazard_net, _ = _train_mlp(x_h, y_h, hidden=a.hidden, seed=a.seed + 1,
                                          classification=True, epochs=a.epochs)
    raw_logit = hazard_net.forward(x_h)
    offsets = np.zeros(2, np.float64)
    for e in (0, 1):
        sel = train_h[:, 5] == e
        if sel.any():
            offsets[e] = _calibrate_offset(raw_logit[sel], y_h[sel], float(y_h[sel].mean()))
    model = LearnedEntryModel(speed=speed_net, hazard=hazard_net, hazard_offset=offsets,
                              metadata={"fit": "TRAIN-val merge_calib", "sources": a.sources,
                                        "n_speed_rows": int(len(speed)),
                                        "n_hazard_rows": int(len(hazard))})

    def metrics(rows_s, rows_h):
        if len(rows_s):
            pred = model.insertion_speed(rows_s[:, 0], rows_s[:, 1], rows_s[:, 2], rows_s[:, 3],
                                         rows_s[:, 4], rows_s[:, 5])
            err = pred - rows_s[:, 6]
            out_s = {"n": int(len(rows_s)), "mae": float(np.abs(err).mean()),
                     "p50_pred": [float(np.median(pred[rows_s[:, 5] == e]))
                                  for e in (0, 1)],
                     "p50_sumo": [float(np.median(rows_s[rows_s[:, 5] == e, 6]))
                                  for e in (0, 1)]}
        else:
            out_s = {"n": 0}
        if len(rows_h):
            h = model.acceptance_hazard(rows_h[:, 0], rows_h[:, 1], rows_h[:, 2], rows_h[:, 3],
                                        rows_h[:, 4], rows_h[:, 5])
            rate = [float(h[rows_h[:, 5] == e].mean()) for e in (0, 1)]
            obs = [float(rows_h[rows_h[:, 5] == e, 6].mean()) for e in (0, 1)]
            out_h = {"n": int(len(rows_h)), "pred_rate": rate, "sumo_rate": obs,
                     "brier": float(np.mean((h - rows_h[:, 6]) ** 2))}
        else:
            out_h = {"n": 0}
        return {"speed": out_s, "hazard": out_h}

    report = {"dataset": {"speed_rows": int(len(speed)), "hazard_rows": int(len(hazard))},
              "offsets": offsets.tolist(), "train": metrics(train_s, train_h),
              "val": metrics(val_s, val_h)}
    Path(a.out).write_text(json.dumps(model.to_dict(), indent=1), encoding="utf-8")
    metrics_path = Path(a.metrics) if a.metrics else Path(a.out).with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    print(f"wrote {a.out} and {metrics_path}")
    return 0


def hash_seed(seed: int) -> int:
    text = f"entry-learned-split:{seed}".encode()
    return int(hashlib.sha256(text).hexdigest()[:8], 16)


if __name__ == "__main__":
    raise SystemExit(main())

