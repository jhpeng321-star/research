"""Gate 1 rollouts: WM outcomes per arm, CAVs teacher-forced from the arm's own
recorded trajectory, from the identical t0 state; plus recorded (SUMO) outcomes.

Models: r5b (R5B best checkpoint), c1 (C1 fast checkpoint), physics (R5b network
with the learned residual and lane correction gated to zero and population-default
driver parameters -> pure IDM/MOBIL physics, default settings).

CPU only, <= 4 worker processes, incremental caches (outcomes_<tag>.json).

Usage:  python run_rollouts.py [--workers 4] [--models r5b,c1,physics]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(RUN))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["TEMP"] = os.environ["TMP"] = "E:/agent-work/tmp"

import torch

DATA = Path("E:/research-dense/data_cache")
CKPTS = {
    "r5b": Path("E:/research-dense/runs/active/WM-S-R5B-20261012/box/gpuA/r5b_main_a1/wm_s_best.pt"),
    "c1": DATA / "ckpt_c1fast" / "wm_s.pt",
    "physics": Path("E:/research-dense/runs/active/WM-S-R5B-20261012/box/gpuA/r5b_main_a1/wm_s_best.pt"),
}
HORIZON_S = 60.0
SEED = 0

_WORKER: dict = {}


def physics_only(ckpt: str):
    from wms.eval.predictor import RolloutFlags, WmsPredictor
    pred = WmsPredictor.from_checkpoint(ckpt, RolloutFlags(entries=False, entries_v2=False))
    model = pred.model
    orig_gate, orig_lane_gate = model.gate, model.lane_gate
    model.gate = lambda k: torch.zeros_like(orig_gate(k))
    model.lane_gate = lambda k, shared=None: torch.zeros_like(orig_lane_gate(k))
    model.lane_change_head = None
    model.diag_encoder_population = True
    return pred


def _init(tag: str) -> None:
    torch.set_num_threads(1)
    from wms.eval.predictor import RolloutFlags, WmsPredictor
    if tag == "physics":
        _WORKER["predictor"] = physics_only(str(CKPTS[tag]))
    else:
        _WORKER["predictor"] = WmsPredictor.from_checkpoint(
            str(CKPTS[tag]), RolloutFlags(entries=False, entries_v2=False))
    _WORKER["tag"] = tag


def _work(payload: tuple) -> dict:
    from outcomes import outcomes, roster
    state, arm = payload
    key = f"{state['world']}|{round(float(state['t0_s']), 3)}|{arm}"
    try:
        frames = _WORKER["predictor"].predict_states(state, {"arm": arm},
                                                     HORIZON_S, SEED)
        roster_map = roster(state["initial_state"])
        result = {"key": key, "arm": arm, "world": str(state["world"]),
                  "t0_s": float(state["t0_s"]), "error": None,
                  "outcomes": outcomes(frames, t0_s=float(state["t0_s"]),
                                       roster_map=roster_map)}
    except Exception as exc:  # keep the batch alive; recorded per cell
        import traceback
        result = {"key": key, "arm": arm, "world": str(state["world"]),
                  "t0_s": float(state["t0_s"]),
                  "error": f"{type(exc).__name__}: {exc}: "
                           f"{traceback.format_exc()[-300:]}"}
    return result


def iter_states():
    from wms.eval.normalize import iter_windows
    return iter_windows(RUN / "states_gate1.json")


def recorded_outcomes() -> dict:
    """SUMO-measured outcomes per (world|t0|arm) from the recorded futures."""
    from outcomes import outcomes, roster
    path = RUN / "outcomes_recorded.json"
    cache = {}
    if path.is_file():
        cache = json.loads(path.read_text(encoding="utf-8"))["cells"]
    n = 0
    for state in iter_states():
        roster_map = roster(state["initial_state"])
        for arm, frames in (state.get("recorded") or {}).items():
            key = f"{state['world']}|{round(float(state['t0_s']), 3)}|{arm}"
            if key in cache:
                continue
            cache[key] = {"arm": arm, "world": str(state["world"]),
                          "t0_s": float(state["t0_s"]), "error": None,
                          "outcomes": outcomes(frames, t0_s=float(state["t0_s"]),
                                               roster_map=roster_map)}
            n += 1
    path.write_text(json.dumps({"schema": "gate1_outcomes/1", "model": "recorded",
                                "cells": cache}, separators=(",", ":")),
                    encoding="utf-8")
    print(f"recorded outcomes: {len(cache)} cells ({n} new)")
    return cache


def run_model(tag: str, workers: int) -> dict:
    from concurrent.futures import ProcessPoolExecutor
    path = RUN / f"outcomes_{tag}.json"
    cache = {}
    if path.is_file():
        cache = json.loads(path.read_text(encoding="utf-8"))["cells"]
    todo = []
    for state in iter_states():
        for arm in sorted(state.get("recorded") or {}):
            key = f"{state['world']}|{round(float(state['t0_s']), 3)}|{arm}"
            if key not in cache:
                todo.append((state, arm))
    print(f"{tag}: {len(todo)} cells to roll out ({len(cache)} cached)", flush=True)
    if todo:
        t0 = time.time()
        done = 0
        with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                                 initargs=(tag,)) as pool:
            for row in pool.map(_work, todo, chunksize=1):
                if row["error"]:
                    print(f"  {tag} error {row['key']}: {row['error'][:160]}",
                          flush=True)
                cache[row["key"]] = row
                done += 1
                if done % 20 == 0:
                    print(f"  {tag}: {done}/{len(todo)} ({time.time()-t0:.0f}s)",
                          flush=True)
                    path.write_text(json.dumps(
                        {"schema": "gate1_outcomes/1", "model": tag,
                         "cells": cache}, separators=(",", ":")), encoding="utf-8")
        path.write_text(json.dumps({"schema": "gate1_outcomes/1", "model": tag,
                                    "cells": cache}, separators=(",", ":")),
                        encoding="utf-8")
        print(f"{tag}: done in {time.time()-t0:.0f}s -> {path.name}", flush=True)
    return cache


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--models", default="r5b,c1,physics")
    args = ap.parse_args()
    if not 1 <= args.workers <= 4:
        raise SystemExit("workers must be in 1..4 (CPU cap)")
    torch.set_num_threads(1)
    recorded_outcomes()
    for tag in [m.strip() for m in args.models.split(",") if m.strip()]:
        run_model(tag, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
