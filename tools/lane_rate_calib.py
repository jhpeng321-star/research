"""LANERATECAL (2026-10-19): closed-loop per-head lane-change RATE calibration of a WM-S
``hazard_v5`` checkpoint under ``physics.lane_learnable`` (inference only; no SUMO, no training).

Problem (v10unblock/eval_same.json, eval_different.json): with ``physics.lane_learnable`` the WM's
free rollout makes far more mainline lane changes than SUMO on the same TRAIN-val starts (overtake
282-439 vs 187-228, return 214-304 vs 106-137 over 16 starts) and ``ds120`` roughly doubles.  The
three learned lane heads sample their own probability; one additive logit offset per head moves the
population rate without touching the lane network.

Method.  Roll the WM on ``N`` TRAIN-val starts (default 48, 120 s), count the lane-change events of
the t0-roster HDVs (overtake 1->0, return 0->1, merge optional 2->1) and compare with the recorded
SUMO events of the SAME starts.  Iterate the fixed point

    offset_h += damping * clip(log(target_count_h / wm_count_h), +-max_step)
    offset_h  = clip(offset_h, +-max_offset)

(default 6 rounds, damping 0.7, max_step 1.0, max_offset 6.0), stopping when every fitted head is
within ``--tol`` (default 10 %).  The best iterate is written as ``lane_calib.json`` IN the
checkpoint directory, next to ``wm_s_best.pt``; ``wms.model.checkpoint.load_checkpoint`` reads it at
load and folds the offsets additively into ``model.lane_logit_offsets`` (absent file = no-op).  The
calibration carries the checkpoint's sha256 so it can never silently transfer to another checkpoint
(LANE_REDESIGN: offsets do not transfer across checkpoints).

TRAIN-val converted episodes only (``split_role == "val"`` of ``C4_EPISODES_MANIFEST.json``, selected
from ``val_list.txt``); never the evaluation seeds 964000-964199 and never sealed/holdout/
cal_split/chunk5/j4b/inc1 paths.

    python tools/lane_rate_calib.py fit CKPT --native-dir DIR --genrand-dir DIR \
        --n-native 24 --n-genrand 24 --horizon-s 120 --workers 48 --device cpu --lane-learnable \
        --rounds 6 --tol 0.10 --heads overtake,return --out-dir CKPT_DIR

Run from the worktree root with ``PYTHONPATH="src;.;tools"``.  ``--rounds 1`` scores the
uncalibrated checkpoint and writes the first update.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src"), str(ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1")
EVAL_SEEDS = range(964000, 964200)
DT = 0.5
HEADS = ("overtake", "return", "merge")
DEFAULT_HEADS = ("overtake", "return")


def guard(path) -> str:
    s = str(path).replace("\\", "/").lower()
    if any(t in s for t in FORBIDDEN):
        raise SystemExit(f"refusing restricted path {path}")
    return str(path)


# --------------------------------------------------------------------------- data
def select_episodes(root: Path, n: int, offset: int = 0) -> list[dict]:
    """The ``n`` TRAIN-val manifest entries at ``offset`` in the source's ``val_list.txt``."""
    lines = [ln.strip() for ln in (root / "val_list.txt").read_text(encoding="utf-8").split()]
    lines = [ln for ln in lines if ln]
    chosen = lines[offset:offset + n]
    if len(chosen) < n:
        raise SystemExit(f"{root}: val_list has {len(lines)} entries, need {n} at offset {offset}")
    doc = json.loads((root / "C4_EPISODES_MANIFEST.json").read_text(encoding="utf-8"))
    by_dir = {e["dir"]: e for e in doc["episodes"]}
    out = []
    for ln in chosen:
        if ln not in by_dir:
            raise SystemExit(f"{root}: {ln} not in manifest")
        e = by_dir[ln]
        if e.get("split_role") != "val":
            raise SystemExit(f"{ln}: not a TRAIN-val episode")
        if int(e["seed"]) in EVAL_SEEDS:
            raise SystemExit(f"evaluation seed {e['seed']} refused")
        out.append(e)
    return out


def build_records(entries: list[dict], root: Path, t0_s: float) -> list[dict]:
    from c4_episode_states import episode_record
    from merge_calib import episode_dict

    out = []
    for e in entries:
        ep = episode_dict(root / e["dir"])
        rec = episode_record(ep, t0_s=float(t0_s), arm=str(e["arm"]), world=str(e["world"]))
        rec["source"] = str(root.name)
        rec["cav_controllers"] = {}
        out.append(rec)
    return out


# --------------------------------------------------------------------------- counting
def lane_events(lane, msk, sel) -> dict:
    """Count overtake (1->0), return (0->1) and merge (2->1) of ``sel`` present both steps."""
    counts = {name: 0 for name in HEADS}
    for k in np.flatnonzero(sel):
        for t in range(lane.shape[0] - 1):
            if not (msk[t, k] and msk[t + 1, k]):
                continue
            a, b = int(lane[t, k]), int(lane[t + 1, k])
            if a == 1 and b == 0:
                counts["overtake"] += 1
            elif a == 0 and b == 1:
                counts["return"] += 1
            elif a == 2 and b == 1:
                counts["merge"] += 1
    return counts


def rate_update(offsets: dict, wm: dict, target: dict, heads, *, damping: float = 0.7,
                max_step: float = 1.0, max_offset: float = 6.0) -> dict:
    """One damped clipped fixed-point step per fitted head (pure; unit-tested)."""
    new = dict(offsets)
    for h in heads:
        tgt = float(target.get(h, 0.0))
        got = max(float(wm.get(h, 0.0)), 0.5)
        step = math.log(max(tgt, 1e-6) / got)
        step = max(-max_step, min(max_step, step))
        new[h] = max(-max_offset, min(max_offset, new[h] + damping * step))
    return new


def relation(wm: dict, target: dict, heads) -> dict:
    """Relative error per fitted head (``inf`` when SUMO has no event but the WM does)."""
    out = {}
    for h in heads:
        tgt = float(target.get(h, 0.0))
        got = float(wm.get(h, 0.0))
        out[h] = abs(got - tgt) / tgt if tgt > 0 else (0.0 if got == 0 else float("inf"))
    return out


def converged(wm: dict, target: dict, heads, tol: float) -> bool:
    return all(rel <= tol for rel in relation(wm, target, heads).values())


# --------------------------------------------------------------------------- worker
W: dict = {}


def _init_model(ckpt: str, base_offsets, device: str, lane_learnable: bool) -> None:
    import torch

    from wm_eval_common import _physics_holders, load_predictor
    torch.set_num_threads(1)
    p, cap = load_predictor(guard(ckpt), "AS_IS", device=device, threads=1, hdv_exits="sumo")
    if lane_learnable and not getattr(p.model.physics, "lane_learnable", False):
        holders = _physics_holders(p.model)
        new = dataclasses.replace(p.model.physics, lane_learnable=True)
        for owner, name in holders:
            setattr(owner, name, new)
        p.model.physics = new
    W["p"] = p
    W["cap"] = cap
    W["base"] = tuple(float(x) for x in base_offsets)


def _set_offsets(vector) -> None:
    base = W["base"]
    W["p"].model.lane_logit_offsets = tuple(base[i] + float(vector[i]) for i in range(3))


def _work(payload) -> dict:
    idx, rec, vector, horizon_s = payload
    from wms.eval.rollout_inputs import recorded_teacher, t0_vehicles
    p, cap = W["p"], W["cap"]
    _set_offsets(vector)
    p.flags = dataclasses.replace(p.flags, lane_seed=int(idx) * 1000 + 1)
    out = {"idx": idx, "error": None}
    try:
        n_steps = int(round(float(horizon_s) / DT))
        t0, ids, kin, is_cav = t0_vehicles(rec)
        teacher, present_seq, _ = recorded_teacher(rec, t0, ids, kin, n_steps, rec["arm"],
                                                   p.backend.tools)
        if teacher is None:
            out["error"] = "no arm"
            return out
        p.predict_states(rec, {**rec.get("cav_controllers", {}), "arm": rec["arm"]},
                         float(horizon_s), 0)
        r = cap["last"]
        n = len(ids)
        tr = teacher[0].numpy()
        rp = present_seq[0].numpy().astype(bool)
        pos = np.concatenate([tr[:1, :, 0], r["positions"][0, :, :n].numpy()], 0)
        lan = np.concatenate([tr[:1, :, 1], r["lanes"][0, :, :n].numpy()], 0).astype(int)
        msk = np.concatenate([rp[:1], r["masks"][0, :, :n].numpy().astype(bool)], 0)
        hdv = ~np.asarray(is_cav, bool)
        finite = bool(np.isfinite(pos).all() and np.isfinite(lan).all())
        out["finite"] = finite
        out["wm"] = lane_events(lan, msk, hdv)
        out["sumo"] = lane_events(tr[:, :, 1].astype(int), rp, hdv)
        for ts in (10, 30, 60, 120):
            i = int(ts / DT)
            both = msk[i] & rp[i] & hdv
            d = np.abs(pos[i] - tr[i, :, 0])
            out[f"ds{ts}"] = float(np.median(d[both])) if both.any() else float("nan")
        out["blowup"] = bool((not finite)
                             or (np.isfinite(out["ds60"]) and out["ds60"] > 50.0
                                 and np.isfinite(out["ds120"]) and out["ds120"] > 100.0))
        return out
    except Exception as exc:  # noqa: BLE001 - recorded per start
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out


def _pooled(results: list) -> dict:
    ok = [r for r in results if not r.get("error")]
    agg = {"starts": len(ok), "errors": len(results) - len(ok),
           "wm": {h: 0 for h in HEADS}, "sumo": {h: 0 for h in HEADS}}
    for r in ok:
        for side in ("wm", "sumo"):
            for h in HEADS:
                agg[side][h] += int(r[side][h])
    for ts in (10, 30, 60, 120):
        v = [r[f"ds{ts}"] for r in ok if np.isfinite(r[f"ds{ts}"])]
        agg[f"ds{ts}"] = float(np.median(v)) if v else float("nan")
    agg["blowups"] = sum(1 for r in ok if r["blowup"])
    agg["nonfinite"] = sum(1 for r in ok if not r.get("finite", True))
    return agg


def run_round(ckpt: str, records: list, base_offsets, vector, horizon_s: float, workers: int,
              device: str, lane_learnable: bool) -> list:
    tasks = [(i, rec, tuple(vector), float(horizon_s)) for i, rec in enumerate(records)]
    if workers <= 1:
        _init_model(ckpt, base_offsets, device, lane_learnable)
        return [_work(t) for t in tasks]
    from multiprocessing import Pool
    with Pool(workers, initializer=_init_model,
              initargs=(ckpt, tuple(base_offsets), device, bool(lane_learnable))) as pool:
        return list(pool.imap(_work, tasks, chunksize=1))


# --------------------------------------------------------------------------- checkpoint helpers
def checkpoint_base_offsets(ckpt: str) -> tuple:
    import torch
    payload = torch.load(guard(ckpt), map_location="cpu", weights_only=True)
    raw = (payload.get("config") or {}).get("lane_logit_offsets", (0.0, 0.0, 0.0))
    vec = tuple(float(x) for x in raw)
    return vec if len(vec) == 3 else (0.0, 0.0, 0.0)


def write_calibration(out_dir: Path, *, ckpt: str, sha256: str, offset_vector, base_offsets,
                      heads, fitted, target, wm_before, wm_after, log, sources, args) -> dict:
    doc = {
        "format": "lane_rate_calib/1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256,
        "heads": list(HEADS),
        "fitted_heads": list(fitted),
        "offsets": {h: float(offset_vector[i]) for i, h in enumerate(HEADS)},
        "offset_vector": [float(x) for x in offset_vector],
        "base_offsets": [float(x) for x in base_offsets],
        "target_counts": {h: int(target[h]) for h in HEADS},
        "wm_counts_before": {h: int(wm_before[h]) for h in HEADS},
        "wm_counts_after": {h: int(wm_after[h]) for h in HEADS},
        "relative_error_after": relation(wm_after, target, fitted),
        "sources": sources,
        "starts": int(args.n_starts),
        "horizon_s": float(args.horizon_s),
        "t0_s": float(args.t0),
        "ground_truth": "SUMOTrainVal",
        "rounds": log,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lane_calib.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return doc


def fit(args) -> int:
    ckpt = guard(args.checkpoint)
    out_dir = Path(guard(args.out_dir))
    sources = []
    entries = []
    if args.native_dir:
        root = Path(guard(args.native_dir))
        eps = select_episodes(root, int(args.n_native), int(args.offset))
        entries += build_records(eps, root, args.t0)
        sources.append({"kind": "native", "root": str(root), "n": len(eps)})
    if args.genrand_dir:
        root = Path(guard(args.genrand_dir))
        eps = select_episodes(root, int(args.n_genrand), int(args.offset))
        entries += build_records(eps, root, args.t0)
        sources.append({"kind": "genrand", "root": str(root), "n": len(eps)})
    if not entries:
        raise SystemExit("no TRAIN-val starts: give --native-dir and/or --genrand-dir")
    args.n_starts = len(entries)
    fitted = tuple(h for h in args.heads.split(",") if h)
    bad = [h for h in fitted if h not in HEADS]
    if bad:
        raise SystemExit(f"unknown head {bad}; choose from {HEADS}")
    base = checkpoint_base_offsets(ckpt)
    from wms.model.checkpoint import sha256_file
    digest = sha256_file(ckpt)
    target = None
    offsets = {h: 0.0 for h in HEADS}
    logs = []
    best = None
    wm_before = None
    t0 = time.time()
    for rnd in range(int(args.rounds) + 1):
        vector = tuple(offsets[h] for h in HEADS)
        results = run_round(ckpt, entries, base, vector, args.horizon_s, args.workers,
                            args.device, args.lane_learnable)
        agg = _pooled(results)
        if rnd == 0:
            target = dict(agg["sumo"])
            wm_before = dict(agg["wm"])
        rel = relation(agg["wm"], target, fitted)
        row = {"round": rnd, "offset_vector": list(vector),
               "wm_counts": {h: int(agg["wm"][h]) for h in HEADS},
               "sumo_counts": {h: int(target[h]) for h in HEADS},
               "relative_error": rel, "ds10": agg["ds10"], "ds30": agg["ds30"],
               "ds60": agg["ds60"], "ds120": agg["ds120"], "blowups": agg["blowups"],
               "nonfinite": agg["nonfinite"], "errors": agg["errors"],
               "seconds": time.time() - t0}
        logs.append(row)
        print(json.dumps({"round": rnd, "offsets": vector, "wm": row["wm_counts"],
                          "sumo": row["sumo_counts"], "rel": rel,
                          "ds60": round(row["ds60"], 2), "ds120": round(row["ds120"], 2),
                          "blowups": row["blowups"]}, sort_keys=True), flush=True)
        is_best = best is None or max(rel.values()) < best[0]
        if is_best:
            best = (max(rel.values()), dict(offsets), dict(agg["wm"]))
        if rnd >= int(args.rounds) or converged(agg["wm"], target, fitted, args.tol):
            break
        offsets = rate_update(offsets, agg["wm"], target, fitted, damping=args.damping,
                              max_step=args.max_step, max_offset=args.max_offset)
    final_vector = [best[1][h] for h in HEADS] if best else [0.0, 0.0, 0.0]
    doc = write_calibration(out_dir, ckpt=ckpt, sha256=digest, offset_vector=final_vector,
                            base_offsets=base, heads=HEADS, fitted=fitted, target=target,
                            wm_before=wm_before, wm_after=(best[2] if best else (wm_before or {})),
                            log=logs, sources=sources, args=args)
    print(json.dumps({"event": "done", "calibration": str(out_dir / "lane_calib.json"),
                      "offset_vector": final_vector,
                      "target_counts": doc["target_counts"],
                      "wm_counts_before": doc["wm_counts_before"],
                      "wm_counts_after": doc["wm_counts_after"],
                      "relative_error_after": doc["relative_error_after"]}, indent=1), flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("checkpoint")
    f.add_argument("--native-dir", default=None)
    f.add_argument("--genrand-dir", default=None)
    f.add_argument("--n-native", type=int, default=24)
    f.add_argument("--n-genrand", type=int, default=24)
    f.add_argument("--n-starts", type=int, default=48,
                   help="informational total; the actual count is n-native + n-genrand")
    f.add_argument("--offset", type=int, default=0, help="skip the first N val_list entries")
    f.add_argument("--t0", type=float, default=420.0)
    f.add_argument("--horizon-s", type=float, default=120.0)
    f.add_argument("--workers", type=int, default=8)
    f.add_argument("--device", default="cpu")
    f.add_argument("--lane-learnable", action="store_true", default=True,
                   help="force physics.lane_learnable at inference (default on)")
    f.add_argument("--no-lane-learnable", dest="lane_learnable", action="store_false")
    f.add_argument("--rounds", type=int, default=6)
    f.add_argument("--tol", type=float, default=0.10)
    f.add_argument("--damping", type=float, default=0.7)
    f.add_argument("--max-step", type=float, default=1.0)
    f.add_argument("--max-offset", type=float, default=6.0)
    f.add_argument("--heads", default=",".join(DEFAULT_HEADS))
    f.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "fit":
        if a.workers < 1 or a.rounds < 1 or a.horizon_s <= 0:
            raise SystemExit("--workers >= 1, --rounds >= 1, --horizon-s > 0")
        return fit(a)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
