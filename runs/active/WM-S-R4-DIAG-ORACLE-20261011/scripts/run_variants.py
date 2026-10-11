"""DIAG 4 step 2: physics-core rollouts with the TRUE driver settings on wall_same.

Variants (all entries OFF, CAVs teacher-forced, seed 0, dt 0.5 s — the R4 scoring harness;
R4 checkpoint = runs/active/WM-S-R4-20261011/box/gpuA/r4_main_a1/wm_s.pt):

  r4_vanilla      R4 full model, unmodified (verification against the R4 score cache)
  phys_prior (c)  physics-only (g*da=0, g*dlane=0) with the core's population-prior IDM
                  params (2.49/4.2/1.135/2.6, v0=31.29*0.992) + MOBIL baseline lanes
  phys_true (a)   physics-only with TRUE per-type IDM params (v0=31.29*speedFactor,
                  tau, minGap, accel, decel); MOBIL lane params left at model defaults
  phys_true_tfl(b) (a) with HDV lanes teacher-forced from the recording (longitudinal floor)
  r4_true (d)     R4 full model (residual ON) with the physics core's params replaced by
                  the TRUE per-type IDM params
  phys_true_sl(e) (a) + SUMO per-lane speed limits: v0 = 22.22*speedFactor while on the
                  on-ramp (lane 2, s<0), else 31.29*speedFactor
  phys_true_sl_act (e2) (e) + 1 s action step (acceleration held for 2 physics ticks)

True settings come from out/truth.json (campaign requests.rou.xml vTypes; speedDev=0 so
the per-vehicle speedFactor equals the vType value).  Evaluation-only ground truth.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

SET_FILE = ROOT / "data_cache" / "wallc1_states_h60" / "wall_same.json"
R4_CKPT = ROOT / "runs/active/WM-S-R4-20261011/box/gpuA/r4_main_a1/wm_s.pt"
TRUTH = HERE.parent / "out" / "truth.json"
OUT = HERE.parent / "out"
HORIZONS = (10.0, 30.0, 60.0, 120.0)
N_STEPS = 240
N = 67
MAIN_LIMIT, RAMP_LIMIT = 31.29, 22.22

#: type -> (v0_mainline, tau, minGap, accel, decel); lane params stay at model defaults.
TYPE_IDM = {
    "aggressive_car": (MAIN_LIMIT * 1.08, 0.95, 1.5, 3.0, 4.5),
    "normal_car": (MAIN_LIMIT * 1.00, 1.10, 2.5, 2.6, 4.5),
    "cautious_car": (MAIN_LIMIT * 0.92, 1.50, 3.5, 2.0, 3.0),
}
SPEED_FACTOR = {"aggressive_car": 1.08, "normal_car": 1.0, "cautious_car": 0.92}

VARIANTS = {
    "r4_vanilla": dict(gate="learned", params="none", tfl=False, sl=False, act=False),
    "phys_prior": dict(gate="zero", params="none", tfl=False, sl=False, act=False),
    "phys_true": dict(gate="zero", params="true", tfl=False, sl=False, act=False),
    "phys_true_tfl": dict(gate="zero", params="true", tfl=True, sl=False, act=False),
    "r4_true": dict(gate="learned", params="true", tfl=False, sl=False, act=False),
    "phys_true_sl": dict(gate="zero", params="true", tfl=False, sl=True, act=False),
    "phys_true_sl_act": dict(gate="zero", params="true", tfl=False, sl=True, act=True),
}

_WORKER: dict = {}


def _init(variant: str) -> None:
    import torch

    torch.set_num_threads(1)
    from wms.eval.predictor import RolloutFlags, WmsPredictor

    _WORKER["variant"] = variant
    _WORKER["truth"] = json.loads(TRUTH.read_text(encoding="utf-8"))
    _WORKER["predictor"] = WmsPredictor.from_checkpoint(
        R4_CKPT, RolloutFlags(entries=False, entries_v2=False))
    if VARIANTS[variant]["gate"] == "zero":
        model = _WORKER["predictor"].model
        model.gate = lambda k: torch.zeros((), dtype=torch.float32)


def _zero(x):
    return False


def _true_driver_params(ids, is_cav, world, model):
    """dict of [1,K] tensors in PARAMETER_NAMES order (true IDM 5, model-default rest)."""
    truth = _WORKER["truth"][world]["vehicles"]
    defaults = model._default_driver_parameters(
        torch.zeros(1, len(ids), 7), 1, len(ids))[0]     # [K,9]
    rows = np.asarray([TYPE_IDM.get(truth.get(i, "normal_car"), TYPE_IDM["normal_car"])
                       for i in ids], dtype=np.float64)
    params = defaults.clone()
    params[:, 0] = torch.as_tensor(rows[:, 0], dtype=defaults.dtype)   # v0
    params[:, 1] = torch.as_tensor(rows[:, 1], dtype=defaults.dtype)   # tau
    params[:, 2] = torch.as_tensor(rows[:, 2], dtype=defaults.dtype)   # minGap
    params[:, 3] = torch.as_tensor(rows[:, 3], dtype=defaults.dtype)   # accel
    params[:, 4] = torch.as_tensor(rows[:, 4], dtype=defaults.dtype)   # decel
    sf = np.asarray([SPEED_FACTOR.get(truth.get(i, "normal_car"), 1.0) for i in ids])
    return {name: params[:, j][None].clone() for j, name in enumerate(
        ("v0", "time_headway", "min_gap", "accel", "decel",
         "politeness", "mobil_threshold", "merge_gap", "merge_urgency"))}, sf


def _install_hooks(model, variant_spec, teacher, params, sf):
    """Patch model.forward/step for tf-lanes / per-lane v0 / action freeze; return cleanup."""
    cleanups = []

    if variant_spec["tfl"]:
        teacher_lane = teacher[0, :, :, 1].round().long().numpy()   # [T+1, N]
        orig_step = model.step
        counter = [0]

        def step_tf(self, state, **kwargs):
            out = orig_step(state, **kwargs)
            i = counter[0]
            counter[0] += 1
            lane_t = torch.as_tensor(teacher_lane[i + 1], dtype=torch.long,
                                     device=out["next_lane"].device)
            cav = state.get("is_cav")
            mask = torch.ones_like(lane_t, dtype=torch.bool)
            if cav is not None:
                c = cav.bool()
                mask = mask & ~(c[0] if c.dim() == 2 else c)
            mask = mask & (lane_t >= 0)
            out["next_lane"] = torch.where(mask[None, :], lane_t[None, :], out["next_lane"])
            return out

        model.step = types.MethodType(step_tf, model)
        cleanups.append(lambda: delattr(model, "step"))

    if variant_spec["sl"] or variant_spec["act"]:
        sf_t = torch.as_tensor(sf, dtype=torch.float32)[None]     # [1,K]
        base_main = torch.full_like(sf_t, MAIN_LIMIT) * sf_t
        base_ramp = torch.full_like(sf_t, RAMP_LIMIT) * sf_t
        orig_forward = model.forward
        state_ = {"frozen": None, "call": 0}

        def forward_hook(self, state, **kwargs):
            if variant_spec["sl"] and params is not None:
                lane = state["lane"].long()
                s = state["s_m"]
                ramp = (lane == 2) & (s < 0)
                params["v0"] = torch.where(ramp, base_ramp, base_main).clone()
            out = orig_forward(state, **kwargs)
            if variant_spec["act"]:
                i = state_["call"]
                state_["call"] += 1
                if i % 2 == 0:
                    state_["frozen"] = out["accel_mps2"].detach().clone()
                else:
                    out["accel_mps2"] = state_["frozen"].clone()
            return out

        model.forward = types.MethodType(forward_hook, model)
        cleanups.append(lambda: delattr(model, "forward"))
    return cleanups


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
    variant, lo, hi = task
    import torch

    from wms.eval.metrics import car_metrics
    from wms.eval.normalize import initial_frame, iter_windows, recorded_for
    from wms.eval.predictor import _frames_from_rollout
    from wms.eval.rollout_inputs import (action_timeline_inputs, recorded_teacher,
                                         t0_vehicles)
    from wms.eval.score import car_compact

    spec = VARIANTS[variant]
    predictor = _WORKER["predictor"]
    model = predictor.model
    rows = []
    t_start = time.time()
    for i, record in enumerate(iter_windows(SET_FILE)):
        if i < lo:
            continue
        if i >= hi:
            break
        t0 = float(record["t0_s"])
        world = str(record.get("world"))
        arm = "wall" if "wall" in (record.get("recorded") or {}) else next(iter(record["recorded"]))
        controllers = {**record.get("cav_controllers", {}), "arm": arm}
        try:
            if variant == "r4_vanilla":
                pred = predictor.predict_states(record, controllers, 120.0, 0)
            else:
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
                params, sf = (None, None)
                if spec["params"] == "true":
                    params, sf = _true_driver_params(ids, is_cav, world, model)
                cleanups = _install_hooks(model, spec, teacher, params, sf)
                try:
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
                finally:
                    for cleanup in reversed(cleanups):
                        cleanup()
                pred = _frames_from_rollout(roll, t0_s, ids, kin, is_cav, 0, N_STEPS)
            rec = recorded_for(record, arm)
            car = car_compact(car_metrics(pred, rec, t0_s=t0, horizons=HORIZONS))
            pids, pa = _arrays(pred, t0)
            rids, ra = _arrays(rec, t0)
            rows.append({
                "idx": i, "key": record.get("key"), "world": world, "t0": t0,
                "error": None, "car": car, "pids": pids, "pred": pa, "rids": rids, "rec": ra,
            })
        except Exception as exc:  # noqa: BLE001 - record per state, never abort the pass
            import traceback
            rows.append({"idx": i, "key": record.get("key"), "world": world, "t0": t0,
                         "error": f"{type(exc).__name__}: {exc}",
                         "trace": traceback.format_exc(limit=4)})
        print(f"[{variant}] state {i} done ({time.time()-t_start:.0f}s)", flush=True)
    return variant, rows


def main() -> None:
    only = sys.argv[1:] or list(VARIANTS)
    OUT.mkdir(parents=True, exist_ok=True)
    for variant in only:
        if variant not in VARIANTS:
            raise SystemExit(f"unknown variant {variant}")
        out_path = OUT / f"variant_{variant}.pkl"
        if out_path.is_file():
            print(f"{variant}: already done, skip", flush=True)
            continue
        t0 = time.time()
        payloads = [None] * N
        with ProcessPoolExecutor(max_workers=6, initializer=_init,
                                 initargs=(variant,)) as pool:
            for _, rows in pool.map(_work, [(variant, lo, min(lo + 12, N))
                                            for lo in range(0, N, 12)]):
                for row in rows:
                    payloads[row["idx"]] = row
        with open(out_path, "wb") as fh:
            pickle.dump(payloads, fh, protocol=4)
        n_err = sum(1 for r in payloads if r and r.get("error"))
        print(f"{variant}: wrote {out_path} ({time.time()-t0:.0f}s, {n_err} errors)",
              flush=True)


if __name__ == "__main__":
    main()
