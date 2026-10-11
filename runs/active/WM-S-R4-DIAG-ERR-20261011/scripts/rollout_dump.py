"""DIAG 1 step 1: re-rollout C1 and R4 on wall_same (entries OFF, teacher CAVs).

Writes per-state compact arrays (predicted + recorded) to out/rollouts.pkl for reuse by
aggregate.py. CPU only, no training/simulation. Verified against the R4 score cache.
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out"
ROOT = ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

SET_FILE = ROOT / "data_cache" / "wallc1_states_h60" / "wall_same.json"
MODELS = {
    "C1": ROOT / "data_cache" / "ckpt_c1fast" / "wm_s.pt",
    "R4": ROOT / "runs/active/WM-S-R4-20261011/box/gpuA/r4_main_a1/wm_s.pt",
}
N_STEPS = 240
N = 67

_WORKER: dict = {}


def _init(model_key: str) -> None:
    import torch

    torch.set_num_threads(1)
    from wms.eval.predictor import RolloutFlags, WmsPredictor

    _WORKER["predictor"] = WmsPredictor.from_checkpoint(
        MODELS[model_key], RolloutFlags(entries=False, entries_v2=False))


def _arrays(frames: list[dict], t0: float) -> tuple[list[str], dict]:
    ids = sorted({str(v["id"]) for fr in frames for v in fr.get("vehicles", [])})
    idx = {x: i for i, x in enumerate(ids)}
    s = np.full((N_STEPS + 1, len(ids)), np.nan, dtype=np.float32)
    v = np.full((N_STEPS + 1, len(ids)), np.nan, dtype=np.float32)
    lane = np.full((N_STEPS + 1, len(ids)), -2, dtype=np.int8)
    present = np.zeros((N_STEPS + 1, len(ids)), dtype=bool)
    is_cav = np.zeros((N_STEPS + 1, len(ids)), dtype=bool)
    for fr in frames:
        k = int(round((float(fr["time_s"]) - t0) / 0.5))
        if not (0 <= k <= N_STEPS):
            continue
        for veh in fr.get("vehicles", []):
            if not veh.get("present", True):
                continue
            j = idx[str(veh["id"])]
            s[k, j] = float(veh["s"])
            v[k, j] = float(veh["v"])
            lane[k, j] = int(veh.get("lane", -1))
            present[k, j] = True
            is_cav[k, j] = bool(veh.get("is_cav", False))
    return ids, {"s": s, "v": v, "lane": lane, "present": present, "is_cav": is_cav}


def _work(task):
    model_key, lo, hi = task
    from wms.eval.normalize import iter_windows

    rows = []
    t_start = time.time()
    for i, record in enumerate(iter_windows(SET_FILE)):
        if i < lo:
            continue
        if i >= hi:
            break
        t0 = float(record["t0_s"])
        arm = "wall" if "wall" in (record.get("recorded") or {}) else next(iter(record["recorded"]))
        pred = _WORKER["predictor"].predict_states(
            record, {**record.get("cav_controllers", {}), "arm": arm}, 120.0, 0)
        from wms.eval.normalize import recorded_for
        rec = recorded_for(record, arm)
        pids, pa = _arrays(pred, t0)
        rids, ra = _arrays(rec, t0)
        rows.append({
            "idx": i, "key": record.get("key"), "world": str(record.get("world")), "t0": t0,
            "pids": pids, "pred": pa, "rids": rids, "rec": ra,
        })
        print(f"[{model_key}] state {i} done ({time.time()-t_start:.0f}s)", flush=True)
    return model_key, rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, list] = {m: [None] * N for m in MODELS}
    t0 = time.time()
    # One pool per model (the initializer fixes the checkpoint per pool).
    for m in MODELS:
        with ProcessPoolExecutor(max_workers=6, initializer=_init, initargs=(m,)) as pool:
            for model_key, rows in pool.map(_work, [(m, lo, min(lo + 12, N)) for lo in range(0, N, 12)]):
                for row in rows:
                    payloads[model_key][row["idx"]] = row
        print(f"model {m} finished ({time.time()-t0:.0f}s)", flush=True)
    with open(OUT / "rollouts.pkl", "wb") as fh:
        pickle.dump(payloads, fh, protocol=4)
    print(f"wrote {OUT/'rollouts.pkl'} ({(OUT/'rollouts.pkl').stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
