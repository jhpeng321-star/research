"""WMLANE diagnostic (CPU only): short training runs + one-step lane-event precision/recall.

    python tools/wmlane_event_diag.py train OUT --config CFG [--set a.b=c ...] [--data-root R]
    python tools/wmlane_event_diag.py eval  OUT_JSON --ckpt NAME=PATH [--ckpt ...]
                                         [--states E:/research-dense/data_cache/wallc1_states_h60/wall_same.json]
                                         [--limit N]

``eval`` follows WM-S-LANECHECK-20261012/scripts/lanecheck.py: every 0.5 s recorded state of
every window (junction lane -1 forward filled, CAVs excluded) is fed to the model ONCE (no
rollout drift); a *wanted* change is an argmax lane different from the recorded current lane,
collapsed over adjacent ticks per car/direction into an episode.  Exact TP = same transition
tick and direction (lanecheck definition); the +-2 s tolerant variant and the per-category
ranking AUC of the model's change probability (event ticks vs non-event ticks of the same
origin lane) are reported too.  Driver settings/embedding come from the record's pre-t0
history (cars without history get the empty-history encoding); hazard models use the
recorded lane age, residual models their lane gate at k = step index (as in a rollout from
t0).  True driver types are never read.  Diagnostic only - not efficacy evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

DT = 0.5
MASKED_LOGIT = -8.0
TOL_TICKS = 4
CATS = {(2, 1): "merge", (1, 0): "overtake", (0, 1): "return"}


def category(a: int, b: int) -> str:
    return CATS.get((a, b), "other")


# ----------------------------------------------------------------------- train
def cmd_train(args) -> int:
    from wmlane_smoke import point_data

    if args.data_root:
        point_data(Path(args.data_root))
    from wms.cli.train import execute
    from wms.config.load import load_train_config

    cfg = load_train_config(args.config, [f"runtime.out_dir={Path(args.out).as_posix()}",
                                          "runtime.device=cpu", *args.overrides])
    return execute(cfg, "run")


def cmd_train_event(args) -> int:
    """Teacher-forced hazard training (lane_event loss only) on a frozen warm-start model.

    Builds the hazard model of ``--config``, loads every matching weight of ``--init``
    (non-strict: the hazard head is new), freezes everything but ``lane_hazard`` and fits it
    with the lane-event loss over all recorded steps of TRAIN windows (no rollout).
    """
    import time
    from dataclasses import asdict

    from torch.utils.data import DataLoader
    from wmlane_smoke import point_data

    if args.data_root:
        point_data(Path(args.data_root))
    from wms.cli.assemble import build_datasets, build_model, build_physics
    from wms.config.load import load_train_config
    from wms.data.collate import collate
    from wms.model.checkpoint import save_checkpoint
    from wms.train.inputs import prepare_inputs
    from wms.train.losses.lane_event import (lane_event_loss, lane_event_targets,
                                             recorded_event_logits, recorded_lane_age)
    from wms.model.lane_hazard import relative_logits
    from wms.train.settings import TrainSettings

    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_train_config(args.config, ["runtime.device=cpu", *args.overrides])
    settings = TrainSettings.from_config(cfg)
    physics = build_physics(cfg)
    torch.manual_seed(int(cfg.runtime.seed))
    model = build_model(cfg, physics)
    if model.lane_hazard is None:
        raise SystemExit("train-event needs model.lane_model='hazard'")
    init = torch.load(args.init, map_location="cpu", weights_only=False)["model"]
    result = model.load_state_dict(init, strict=False)
    missing = [k for k in result.missing_keys if not k.startswith(("lane_hazard.",
                                                                     "driver_updater."))]
    if missing:
        raise SystemExit(f"warm start misses weights: {missing[:5]}")
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("lane_hazard."))
    model.train()
    train, _, _ = build_datasets(cfg, out)
    gen = torch.Generator().manual_seed(int(cfg.runtime.seed))
    loader = DataLoader(train, batch_size=int(args.batch), shuffle=True, collate_fn=collate,
                        num_workers=0, generator=gen)
    opt = torch.optim.Adam(model.lane_hazard.parameters(), lr=float(args.lr))
    log, step, started = [], 0, time.time()
    gamma = float(cfg.loss.lane_event.gamma)
    max_w = float(cfg.loss.lane_event.max_weight)
    while step < int(args.steps):
        for batch in loader:
            n = int(batch["states"].shape[1]) - 1
            inp = prepare_inputs(batch, model, settings, torch.device("cpu"), n)
            with torch.no_grad():
                _, emb, params = model.encode_drivers(
                    inp.initial, driver_history=inp.observed_hist,
                    driver_history_mask=inp.observed_hist_mask)
                age0 = model._initial_lane_age({}, inp.present[:, 0], inp.states[:, 0],
                                               inp.observed_hist, inp.observed_hist_mask)
            states = inp.states[:, :n + 1]
            present = inp.loss_valid[:, :n + 1]
            cav = inp.is_cav[:, :n + 1]
            lanes = states[..., 1:4].argmax(-1)
            valid = present & ~cav
            action, cat, mask = lane_event_targets(lanes, valid, n)
            age = recorded_lane_age(lanes, valid, age0, n)
            logits = recorded_event_logits(model, states[:, :n], present[:, :n], cav[:, :n],
                                           params, emb, age)
            rel = relative_logits(logits, lanes[:, :n])
            loss, stats = lane_event_loss(rel[mask], action[mask], cat[mask], gamma=gamma,
                                          max_weight=max_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.lane_hazard.parameters(), 1.0)
            opt.step()
            step += 1
            row = {"step": step, "loss": float(loss.detach()), "elapsed_s": time.time() - started,
                   **stats}
            log.append(row)
            if step % 10 == 0 or step == 1:
                print(json.dumps(row), flush=True)
            if step >= int(args.steps) or (time.time() - started) > 60 * float(args.max_minutes):
                break
        if (time.time() - started) > 60 * float(args.max_minutes):
            break
    (out / "train_event_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log),
                                               encoding="utf-8")
    (out / "physics.json").write_text(json.dumps(asdict(physics), indent=1), encoding="utf-8")
    save_checkpoint(out / "wm_s.pt", model, cfg, extra={"wmlane_train_event": {
        "init": args.init, "steps": step, "batch": int(args.batch), "lr": float(args.lr)}})
    print(json.dumps({"status": "done", "steps": step, "out": str(out)}))
    return 0


# ------------------------------------------------------------------------ eval
def _record_tensors(record):
    from wms.eval.normalize import initial_frame, recorded_for

    arm = record["arm"] if record.get("arm") in record.get("recorded", {}) \
        else next(iter(record["recorded"]))
    t0 = float(record["t0_s"])
    frames = [initial_frame(record), *recorded_for(record, arm)]
    by_tick, ids = {}, set()
    for fr in frames:
        tick = int(round((float(fr["time_s"]) - t0) / DT))
        if 0 <= tick <= 240:
            by_tick[tick] = fr
            ids.update(str(v["id"]) for v in fr["vehicles"])
    ids = sorted(ids)
    ix = {vid: j for j, vid in enumerate(ids)}
    t_n = max(by_tick) + 1
    s = np.zeros((t_n, len(ids)), np.float32)
    v = np.zeros_like(s)
    lane = np.full((t_n, len(ids)), -1, np.int64)
    present = np.zeros((t_n, len(ids)), bool)
    cav = np.zeros_like(present)
    for tick, fr in by_tick.items():
        for row in fr["vehicles"]:
            if not row.get("present", True):
                continue
            j = ix[str(row["id"])]
            s[tick, j], v[tick, j] = row["s"], row["v"]
            lane[tick, j] = int(row.get("lane", -1))
            present[tick, j] = True
            cav[tick, j] = bool(row.get("is_cav", False))
    filled = lane.copy()
    for j in range(len(ids)):
        last = -1
        for tick in range(t_n):
            if filled[tick, j] >= 0:
                last = filled[tick, j]
            elif last >= 0 and present[tick, j]:
                filled[tick, j] = last
    acc = np.zeros_like(v)
    both = present[1:] & present[:-1]
    acc[1:] = np.where(both, (v[1:] - v[:-1]) / DT, 0.0)
    return ids, s, v, acc, filled, present & (filled >= 0), cav


def _kin(s, v, acc, lane, valid):
    t_n, n = s.shape
    kin = torch.zeros(t_n, n, 7)
    kin[..., 0] = torch.as_tensor(s)
    kin[..., 4] = torch.as_tensor(v)
    kin[..., 5] = torch.as_tensor(acc)
    kin[..., 6] = 5.0
    for li in range(3):
        kin[..., 1 + li] = torch.as_tensor((lane == li) & valid, dtype=torch.float32)
    return kin


def _driver_history(model, record, ids, present0):
    """Long-history features ``[1, N, H, 9]`` + mask for the record's t0 vehicles."""
    from wms.data.history import history_spec, long_history_features

    n = len(ids)
    hist_s, stride_s = model.history_s, model.history_stride_s
    _, _, want = history_spec(hist_s, stride_s, DT)
    out = torch.zeros(1, n, want, 9)
    mask = torch.zeros(1, n, want, dtype=torch.bool)
    hist = record.get("history") or {}
    if not hist.get("ids"):
        return out, mask
    kin = torch.as_tensor(np.asarray(hist["kin"], np.float32))
    hm = torch.as_tensor(np.asarray(hist["mask"], bool))
    tm = torch.as_tensor(np.asarray(hist["target_mask"], bool)).reshape(-1, kin.shape[1])
    cols = [str(c) for c in hist["ids"]]
    now = torch.as_tensor([not c.startswith("__history_context") for c in cols])
    feat, fmask = long_history_features(kin, hm, tm, now & hm[-1], history_s=hist_s,
                                        history_stride_s=stride_s)
    where = {c: k for k, c in enumerate(cols)}
    for j, vid in enumerate(ids):
        k = where.get(vid)
        if k is not None and present0[j]:
            out[0, j], mask[0, j] = feat[k], fmask[k]
    return out, mask


def _rolling_history(kin, valid, physics, length=10):
    """Recorded rolling history ``[T, N, H, 6]`` (rollout convention, nothing before t0)."""
    from wms.model.history import history_step_features
    from wms.physics.neighbours import lead_neighbour

    gap, _ = lead_neighbour(kin, valid, physics)
    feats = history_step_features(kin, gap)
    t_n, n = valid.shape
    hist = torch.zeros(t_n, n, length, feats.shape[-1])
    mask = torch.zeros(t_n, n, length, dtype=torch.bool)
    for t in range(t_n):
        lo = max(0, t - length + 1)
        span = t - lo + 1
        hist[t, :, -span:] = feats[lo:t + 1].transpose(0, 1)
        mask[t, :, -span:] = valid[lo:t + 1].transpose(0, 1)
    return hist, mask


@torch.no_grad()
def _one_step(model, record):
    from wms.model.driver_encoder import PARAMETER_NAMES
    from wms.model.history import initial_lane_age
    from wms.train.losses.lane_event import recorded_lane_age

    ids, s, v, acc, lane, valid, cav = _record_tensors(record)
    t_n, n = s.shape
    kin = _kin(s, v, acc, lane, valid)
    valid_t = torch.as_tensor(valid)
    cav_t = torch.as_tensor(cav)
    dh, dhm = _driver_history(model, record, ids, valid[0])
    state0 = {"veh_kin": kin[:1], "present": valid_t[:1], "is_cav": cav_t[:1]}
    encoded, emb, params = model.encode_drivers(state0, driver_history=dh,
                                                driver_history_mask=dhm)
    p_t = {k: val.expand(t_n, n) for k, val in params.items()}
    e_t = emb.expand(t_n, n, -1)
    state = {"veh_kin": kin, "present": valid_t, "is_cav": cav_t}
    if getattr(model, "lane_hazard", None) is not None:
        stride = round(model.history_stride_s / DT)
        age0 = initial_lane_age(valid_t[:1], kin.dtype, dh, dhm, long_history=True,
                                stride_steps=stride)
        age = recorded_lane_age(torch.as_tensor(lane)[None], valid_t[None], age0,
                                t_n)[0]
        logits = model.lane_event_logits(state, driver_params=p_t, driver_embedding=e_t,
                                         lane_age=age)
    else:
        hist, hmask = _rolling_history(kin, valid_t, model.physics)
        k = torch.arange(t_n).view(t_n, 1).expand(t_n, n)
        out = model(state, k_steps=k, history=hist, history_mask=hmask, driver_params=p_t,
                    driver_embedding=e_t)
        logits = out["lane_logits"]
    probs = torch.softmax(logits, -1).numpy()
    want = logits.argmax(-1).numpy()
    masked = (logits <= MASKED_LOGIT + 1.0e-6).numpy()
    del encoded, PARAMETER_NAMES
    return ids, lane, valid, cav, want, probs, masked


def _auc(pos, neg):
    """Mann-Whitney AUC with average ranks for ties (constant scores give 0.5)."""
    if not pos or not neg:
        return None
    values = np.concatenate([np.asarray(pos, float), np.asarray(neg, float)])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(1, len(values) + 1)
    uniq, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    ranks = (sums / counts)[inverse]
    n_pos, n_neg = len(pos), len(neg)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def evaluate(model, path, limit=None):
    from wms.eval.normalize import iter_windows

    agg = Counter()
    cats = defaultdict(Counter)
    scores = defaultdict(lambda: ([], []))
    n_windows = 0
    for record in iter_windows(path, limit=limit):
        n_windows += 1
        ids, lane, valid, cav, want, probs, masked = _one_step(model, record)
        t_n, n = lane.shape
        hdv = valid & ~cav
        events = []
        for t in range(t_n - 1):
            for j in range(n):
                a, b = lane[t, j], lane[t + 1, j]
                if hdv[t, j] and hdv[t + 1, j] and a != b and abs(a - b) == 1:
                    events.append((j, t, int(a), int(b)))
        wanted = []
        for j in range(n):
            ticks = [t for t in range(t_n - 1) if hdv[t, j] and want[t, j] != lane[t, j]]
            prev = None
            for t in ticks:
                if prev is None or t != prev + 1 or want[t, j] != want[prev, j]:
                    wanted.append((j, t, int(lane[t, j]), int(want[t, j])))
                prev = t
            for t in range(t_n - 1):
                if hdv[t, j]:
                    agg["hdv_decisions"] += 1
                    agg["wanted_steps"] += int(want[t, j] != lane[t, j])
        event_set = {(j, t, b) for j, t, a, b in events}
        want_ticks = defaultdict(list)
        for j in range(n):
            for t in range(t_n - 1):
                if hdv[t, j] and want[t, j] != lane[t, j]:
                    want_ticks[(j, int(want[t, j]))].append(t)
        for j, t, a, b in events:
            c = cats[category(a, b)]
            c["recorded"] += 1
            c["recorded_physics_masked"] += int(masked[t, j, b])
            exact = any(w[0] == j and w[1] == t and w[3] == b for w in wanted)
            c["exact_tp_recall"] += int(exact)
            near = any(abs(u - t) <= TOL_TICKS for u in want_ticks[(j, b)])
            c["tol_recall"] += int(near)
        rec_ticks = defaultdict(list)
        for j, t, a, b in events:
            rec_ticks[(j, b)].append(t)
        for j, t, a, b in wanted:
            c = cats[category(a, b)]
            c["wanted_episodes"] += 1
            c["exact_tp_precision"] += int((j, t, b) in event_set)
            c["tol_precision"] += int(any(abs(u - t) <= TOL_TICKS for u in rec_ticks[(j, b)]))
        # Ranking AUC of the probability of moving to the recorded event's target lane.
        for (a, b), name in CATS.items():
            pos, neg = scores[name]
            for t in range(t_n - 1):
                for j in range(n):
                    if not (hdv[t, j] and hdv[t + 1, j] and lane[t, j] == a):
                        continue
                    if lane[t + 1, j] == b:
                        pos.append(float(probs[t, j, b]))
                    elif lane[t + 1, j] == a:
                        neg.append(float(probs[t, j, b]))
    out = {"windows": n_windows, "hdv_decisions": agg["hdv_decisions"],
           "wanted_step_rate": agg["wanted_steps"] / max(agg["hdv_decisions"], 1),
           "categories": {}}
    tot = Counter()
    for name, c in sorted(cats.items()):
        row = dict(c)
        row["exact_recall"] = c["exact_tp_recall"] / c["recorded"] if c["recorded"] else None
        row["exact_precision"] = (c["exact_tp_precision"] / c["wanted_episodes"]
                                  if c["wanted_episodes"] else None)
        row["tol2s_recall"] = c["tol_recall"] / c["recorded"] if c["recorded"] else None
        row["tol2s_precision"] = (c["tol_precision"] / c["wanted_episodes"]
                                  if c["wanted_episodes"] else None)
        pos, neg = scores.get(name, ([], []))
        row["auc"] = _auc(pos, neg)
        out["categories"][name] = row
        tot.update(c)
    out["all"] = {
        "recorded": tot["recorded"], "wanted_episodes": tot["wanted_episodes"],
        "exact_recall": tot["exact_tp_recall"] / max(tot["recorded"], 1),
        "exact_precision": tot["exact_tp_precision"] / max(tot["wanted_episodes"], 1),
        "tol2s_recall": tot["tol_recall"] / max(tot["recorded"], 1),
        "tol2s_precision": tot["tol_precision"] / max(tot["wanted_episodes"], 1),
        "recorded_physics_masked": tot["recorded_physics_masked"]}
    return out


def cmd_eval(args) -> int:
    from wms.model.checkpoint import load_checkpoint

    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    result = {"states": args.states, "limit": args.limit, "models": {}}
    for spec in args.ckpt:
        name, _, path = spec.partition("=")
        loaded = load_checkpoint(path, device="cpu")
        model = loaded.model
        model.eval()
        result["models"][name] = {"checkpoint": path, "sha256": loaded.sha256,
                                  "lane_model": getattr(model, "lane_model", "residual"),
                                  **evaluate(model, args.states, args.limit)}
        print(name, json.dumps(result["models"][name]["all"]))
    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("out")
    tr.add_argument("--config", required=True)
    tr.add_argument("--set", dest="overrides", action="append", default=[])
    tr.add_argument("--data-root", default=None)
    te = sub.add_parser("train-event")
    te.add_argument("out")
    te.add_argument("--config", required=True)
    te.add_argument("--init", required=True)
    te.add_argument("--set", dest="overrides", action="append", default=[])
    te.add_argument("--data-root", default=None)
    te.add_argument("--steps", type=int, default=300)
    te.add_argument("--batch", type=int, default=8)
    te.add_argument("--lr", type=float, default=1e-3)
    te.add_argument("--max-minutes", type=float, default=120.0)
    ev = sub.add_parser("eval")
    ev.add_argument("out")
    ev.add_argument("--ckpt", action="append", required=True)
    ev.add_argument("--states",
                    default="E:/research-dense/data_cache/wallc1_states_h60/wall_same.json")
    ev.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    return {"train": cmd_train, "train-event": cmd_train_event, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
