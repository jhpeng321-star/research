"""CHK-TYPEID step 3: physics-only rollouts with IDENTIFIED types.

Harness identical to DIAG-ORACLE run_variants.py (entries OFF, CAVs teacher-forced,
seed 0, dt 0.5 s, R4 checkpoint core, MOBIL baseline lanes, residual OFF via zero
gate).  Variants:

  phys_prior        default (population-prior) IDM params      [verify 15.68/53.12]
  phys_true         TRUE per-type params (out/truth_full.json) [verify 4.31/48.65]
  phys_ident        identified types (argmax) from observable history
  phys_ident_post   posterior-weighted average of the 3 type param vectors

Sets: wall_same (wallc1_states_h60, 5 s history), real_q20 / real_full
(wallc1_states_h60_real quick-20 / all 120 states, 60 s history; the q20 slice
replicates score_round4.quick20 exactly so the C1/R4 cache applies).
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

from typeid import TYPE_NAMES, posteriors, prep, pilot_scale, IDM5, SPEED_FACTOR  # noqa: E402

R4_CKPT = ROOT / "runs/active/WM-S-R4-20261011/box/gpuA/r4_main_a1/wm_s.pt"
TRUTH_FILE = HERE.parent / "out/truth_full.json"
OUT = HERE.parent / "out"
HORIZONS = (10.0, 30.0, 60.0, 120.0)
N_STEPS = 240
MAIN_LIMIT = 31.29

TYPE_IDM = {
    "aggressive_car": (MAIN_LIMIT * 1.08, 0.95, 1.5, 3.0, 4.5),
    "normal_car": (MAIN_LIMIT * 1.00, 1.10, 2.5, 2.6, 4.5),
    "cautious_car": (MAIN_LIMIT * 0.92, 1.50, 3.5, 2.0, 3.0),
}

SET_FILES = {
    "wall_same": ROOT / "data_cache/wallc1_states_h60/wall_same.json",
    "real": ROOT / "data_cache/wallc1_states_h60_real/wall_same.json",
}
N_STATES = {"wall_same": 67, "real": 120}
#: identifier history length per set (5 s dense for the h60 set, 60 s for real)
IDENT_LENGTH = {"wall_same": 5, "real": 60}
PILOT_SCALE = {"wall_same": 1.4402063242539114, "real": 1.262458799676593}

VARIANTS = {
    "phys_prior": dict(params="none", ident=False, post=False),
    "phys_true": dict(params="true", ident=False, post=False),
    "phys_ident": dict(params="ident", ident=True, post=False),
    "phys_ident_post": dict(params="ident", ident=True, post=True),
}
#: (variant, set, slice) runs; q20 indices replicate score_round4.quick20 ordering
RUNS = [
    ("phys_prior", "wall_same", None),
    ("phys_true", "wall_same", None),
    ("phys_ident", "wall_same", None),
    ("phys_ident_post", "wall_same", None),
    ("phys_true", "real", "q20"),
    ("phys_ident", "real", "q20"),
    ("phys_ident_post", "real", "q20"),
    ("phys_ident", "real", "full"),
]

_WORKER: dict = {}


def quick20_indices(path: Path) -> list[int]:
    rows = []
    for i, rec in enumerate(_stream(path)):
        rows.append((i, str(rec.get("world")), float(rec.get("t0_s", 0)),
                     str(rec.get("key"))))
    chosen = {}
    for row in sorted(rows, key=lambda x: (x[1], x[2], x[3])):
        chosen.setdefault(row[1], row[0])
    return [chosen[k] for k in sorted(chosen)[:20]]


def _stream(path):
    from wms.eval.normalize import iter_windows
    return iter_windows(path)


def _init(variant: str, set_name: str, ident_length: int, pilot_scale: float):
    torch.set_num_threads(1)
    from wms.eval.predictor import RolloutFlags, WmsPredictor

    _WORKER.update(variant=variant, set_name=set_name,
                   truth=json.loads(TRUTH_FILE.read_text(encoding="utf-8")))
    _WORKER["predictor"] = WmsPredictor.from_checkpoint(
        R4_CKPT, RolloutFlags(entries=False, entries_v2=False))
    model = _WORKER["predictor"].model
    model.gate = lambda k: torch.zeros((), dtype=torch.float32)
    _WORKER["ident_length"] = ident_length
    _WORKER["pilot_scale"] = pilot_scale


def _ident_params(ids, is_cav, world, record):
    """Identified (argmax or posterior-weighted) TYPE_IDM rows for the roster."""
    length = _WORKER["ident_length"]
    d = prep(record["history"])
    per_length, _ = posteriors(d, (length,), _WORKER["pilot_scale"])
    out = per_length[length]
    col = {x: i for i, x in enumerate(d["ids"])}
    post_variant = VARIANTS[_WORKER["variant"]]["post"]
    rows = np.tile(np.asarray(TYPE_IDM["normal_car"], dtype=np.float64), (len(ids), 1))
    for k, vid in enumerate(ids):
        if is_cav[k] or vid not in col:
            continue                                    # CAVs: teacher-forced
        p = out[col[vid]]["posterior"]
        if post_variant:
            rows[k] = sum(p[t] * np.asarray(TYPE_IDM[t], dtype=np.float64)
                          for t in TYPE_NAMES)
        else:
            rows[k] = TYPE_IDM[out[col[vid]]["argmax"]]
    return rows


def _driver_params(ids, is_cav, world, record, model):
    spec = VARIANTS[_WORKER["variant"]]
    defaults = model._default_driver_parameters(
        torch.zeros(1, len(ids), 7), 1, len(ids))[0]     # [K,9]
    if spec["params"] == "none":
        return None
    if spec["params"] == "true":
        truth = _WORKER["truth"][world]["vehicles"]
        rows = np.asarray([TYPE_IDM.get(truth.get(i, "normal_car"),
                                        TYPE_IDM["normal_car"]) for i in ids],
                          dtype=np.float64)
    else:
        rows = _ident_params(ids, is_cav, world, record)
    params = defaults.clone()
    for c in range(5):
        params[:, c] = torch.as_tensor(rows[:, c], dtype=defaults.dtype)
    return {name: params[:, j][None].clone() for j, name in enumerate(
        ("v0", "time_headway", "min_gap", "accel", "decel",
         "politeness", "mobil_threshold", "merge_gap", "merge_urgency"))}


def _arrays(frames: list[dict], t0: float):
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
    variant, set_name, indices, lo, hi = task
    from wms.eval.metrics import car_metrics
    from wms.eval.normalize import initial_frame, recorded_for
    from wms.eval.predictor import _frames_from_rollout
    from wms.eval.rollout_inputs import (action_timeline_inputs, recorded_teacher,
                                         t0_vehicles)
    from wms.eval.score import car_compact

    spec = VARIANTS[variant]
    predictor = _WORKER["predictor"]
    model = predictor.model
    rows = []
    t_start = time.time()
    stream = _stream(SET_FILES[set_name])
    index_set = set(indices) if indices is not None else None
    picked = 0
    for i, record in enumerate(stream):
        if index_set is not None and i not in index_set:
            continue
        if picked < lo:
            picked += 1
            continue
        if picked >= hi:
            break
        cur = picked
        picked += 1
        t0 = float(record["t0_s"])
        world = str(record.get("world"))
        arm = "wall" if "wall" in (record.get("recorded") or {}) else next(iter(record["recorded"]))
        controllers = {**record.get("cav_controllers", {}), "arm": arm}
        try:
            tools = predictor.backend.tools
            t0_s, ids, kin, is_cav = t0_vehicles(record)
            driver_history, driver_history_mask = predictor._history.build(record, ids)
            action_timeline = action_timeline_inputs(record, controllers, ids, 0, tools)
            initial = {
                "s_m": torch.as_tensor(kin[None, :, 0]),
                "v_mps": torch.as_tensor(kin[None, :, 4]),
                "a_mps2": torch.zeros((1, len(ids))),
                "lane": torch.as_tensor(kin[None, :, 1:4].argmax(-1)),
                "length_m": torch.as_tensor(kin[None, :, 6]),
                "present": torch.ones((1, len(ids)), dtype=torch.bool),
                "is_cav": torch.as_tensor(is_cav[None]),
            }
            teacher, present_seq, junction_seq = recorded_teacher(
                record, t0_s, ids, kin, N_STEPS, arm, tools)
            params = _driver_params(ids, is_cav, world, record, model)
            with torch.no_grad():
                roll = model.rollout(
                    initial, n_steps=N_STEPS, k0=0, cav_teacher=teacher,
                    present_seq=present_seq, junction_seq=junction_seq,
                    cav_controller=None,
                    driver_history=driver_history,
                    driver_history_mask=driver_history_mask,
                    driver_params=params, controller_id=None,
                    action_timeline=action_timeline,
                    projection=predictor.backend.projection)
            pred = _frames_from_rollout(roll, t0_s, ids, kin, is_cav, 0, N_STEPS)
            rec = recorded_for(record, arm)
            car = car_compact(car_metrics(pred, rec, t0_s=t0, horizons=HORIZONS))
            pids, pa = _arrays(pred, t0)
            rids, ra = _arrays(rec, t0)
            rows.append({
                "idx": cur, "key": record.get("key"), "world": world, "t0": t0,
                "error": None, "car": car, "pids": pids, "pred": pa, "rids": rids, "rec": ra,
            })
        except Exception as exc:  # noqa: BLE001 - per state, never abort the pass
            import traceback
            rows.append({"idx": cur, "key": record.get("key"), "world": world, "t0": t0,
                         "error": f"{type(exc).__name__}: {exc}",
                         "trace": traceback.format_exc(limit=4)})
        print(f"[{variant}/{set_name}] state {cur} done ({time.time()-t_start:.0f}s)",
              flush=True)
    return variant, rows


def main() -> None:
    RUN = RUNS
    if len(sys.argv) > 1 and sys.argv[1] != "all":
        want = set(sys.argv[1].split(","))
        RUN = [r for r in RUNS if r[0] in want]
    OUT.mkdir(parents=True, exist_ok=True)
    q20_indices = quick20_indices(SET_FILES["real"])
    print(f"real q20 indices: {q20_indices}", flush=True)
    for variant, set_name, slc in RUN:
        if set_name == "real":
            suffix = "_real_q20" if slc == "q20" else "_real_full"
            indices = q20_indices if slc == "q20" else None
            n = 20 if slc == "q20" else N_STATES["real"]
        else:
            suffix, indices, n = "", None, N_STATES[set_name]
        out_path = OUT / f"variant_{variant}{suffix}.pkl"
        if out_path.is_file():
            print(f"{variant}/{set_name}/{slc}: already done, skip", flush=True)
            continue
        t0 = time.time()
        payloads = [None] * n
        chunk = 12 if n > 20 else 10
        with ProcessPoolExecutor(
                max_workers=6, initializer=_init,
                initargs=(variant, set_name, IDENT_LENGTH[set_name],
                          PILOT_SCALE[set_name])) as pool:
            for _, rows in pool.map(_work, [(variant, set_name, indices, lo,
                                             min(lo + chunk, n))
                                            for lo in range(0, n, chunk)]):
                for row in rows:
                    payloads[row["idx"]] = row
        with open(out_path, "wb") as fh:
            pickle.dump(payloads, fh, protocol=4)
        n_err = sum(1 for r in payloads if r and r.get("error"))
        print(f"{variant}/{set_name}/{slc}: wrote {out_path} "
              f"({time.time()-t0:.0f}s, {n_err} errors)", flush=True)


if __name__ == "__main__":
    main()
