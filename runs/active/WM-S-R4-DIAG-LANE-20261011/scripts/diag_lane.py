"""DIAG-LANE (analysis only): why does R4 predict MORE early (0-60 s) lane changes than C1?

Runs instrumented 120 s rollouts on the wall_same quick-20 states (same selection rule as
WM-S-R4-20261011/score/score_round4.py:quick20) with the clean scorer package
(PYTHONPATH=E:/research-dense/src, entries OFF, CAVs teacher-forced from the recorded arm).

Per (model, variant) it decomposes every lane decision:
    lane_logits = MOBIL/merge baseline + g(k) * learned lane delta      (argmax -> next lane)

Variants:
  base    untouched model
  delta0  lane correction zeroed          (MOBIL-only decisions)
  gcap20  lane correction's gate capped at g(20) (min(g(k), g(20)) * delta; accel untouched)

Metrics per run: TRUE lane-change rate (lane forward-filled over junction -1; change =
lane differs with both >= 0; HDVs only) for 0-60 s / 60-120 s, wrong-lane share at 60/120 s,
per-state hdv |ds| medians at 60/120 s, and a per-decision trace of the early window.

CPU only, 1 torch thread per worker, no training/simulation. Outputs: out/tasks/*.pkl.
"""
from __future__ import annotations

import os
import sys

ROOT = "E:/research-dense"
sys.path.insert(0, ROOT + "/src")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ["TEMP"] = os.environ["TMP"] = "E:/agent-work/tmp"
sys.dont_write_bytecode = True

import json
import pickle
import time
from pathlib import Path

import numpy as np

RUN = Path(ROOT) / "runs/active/WM-S-R4-DIAG-LANE-20261011"
OUT = RUN / "out"
TASKS = OUT / "tasks"

MODELS = {
    "C1": ROOT + "/data_cache/ckpt_c1fast/wm_s.pt",
    "R4": ROOT + "/runs/active/WM-S-R4-20261011/box/gpuA/r4_main_a1/wm_s.pt",
}
WALL_SAME = ROOT + "/data_cache/wallc1_states_h60/wall_same.json"
HORIZON_S = 120.0
N_STEPS = 240
EARLY_STEPS = 120          # 0-60 s trace window
GATE_CAP_K = 20

# --------------------------------------------------------------------------- #
# state selection (identical rule to score_round4.quick20)
# --------------------------------------------------------------------------- #
def quick20(path: str) -> list[dict]:
    from wms.eval.normalize import iter_windows
    rows = list(iter_windows(Path(path)))
    chosen: dict = {}
    for r in sorted(rows, key=lambda x: (str(x.get("world")), float(x.get("t0_s", 0)), str(x.get("key")))):
        chosen.setdefault(str(r.get("world")), r)
    return [chosen[k] for k in sorted(chosen)[:20]]


def scored_arm(record: dict) -> str:
    rec = record.get("recorded") or {}
    if "wall" in rec:
        return "wall"
    if len(rec) == 1:
        return next(iter(rec))
    return "native"


# --------------------------------------------------------------------------- #
# worker
# --------------------------------------------------------------------------- #
_W: dict = {}


def _init_worker():
    os.environ["WMS_EVAL_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["TEMP"] = os.environ["TMP"] = "E:/agent-work/tmp"
    import torch
    torch.set_num_threads(1)


def _predictor(model: str):
    if model in _W:
        return _W[model]
    import torch
    from wms.eval.predictor import RolloutFlags, WmsPredictor
    pred = WmsPredictor.from_checkpoint(MODELS[model], RolloutFlags(entries=False, entries_v2=False))
    with torch.no_grad():
        g_cap = float(pred.model.gate(torch.tensor(GATE_CAP_K)))
    # ---- instrumented forward: decompose + optionally rewire lane logits --------- #
    model = pred.model
    orig_forward = model.forward
    trace: list[dict] = []

    def forward(state, **kw):
        out = orig_forward(state, **kw)
        with torch.no_grad():
            base = model.mobil_merge_logits(state)          # same call residual mode uses
            logits = out["lane_logits"]
            corr = logits - base
            if variant[0] == "delta0":
                logits = base
            elif variant[0] == "gcap20" and variant[1] > 0.0:
                g = out["gate"]
                scale = (variant[1] / g).clamp(max=1.0)
                logits = base + scale.unsqueeze(-1) * corr
            lane = state["lane"].long()
            present = state["present"].bool()
            is_cav = state.get("is_cav")
            is_cav = torch.zeros_like(present) if is_cav is None else is_cav.bool()
            k = int(kw.get("k_steps", 0))
            gate = out["gate"]
            gate = float(gate) if gate.dim() == 0 else float(gate.reshape(-1)[0])
            trace.append({
                "k": k, "gate": gate,
                "present": present[0].numpy().copy(),
                "is_cav": is_cav[0].numpy().copy(),
                "lane": lane[0].numpy().copy(),
                "s_m": state["s_m"][0].numpy().copy(),
                "v_mps": state["v_mps"][0].numpy().copy(),
                "base": base[0].numpy().astype(np.float32).copy(),
                "corr": (logits - base)[0].numpy().astype(np.float32).copy(),
                "decision": logits.argmax(-1)[0].numpy().copy(),
            })
        out = dict(out)
        out["lane_logits"] = logits
        return out

    model.forward = forward
    _W[model] = (pred, trace, variant)
    return _W[model]


variant: list = ["base", 0.0]


def run_task(payload):
    model_name, vname, record = payload
    global variant
    t_start = time.time()
    variant = [vname, 0.0]
    pred, trace, _ = _predictor(model_name)
    if vname == "gcap20":
        import torch
        with torch.no_grad():
            variant[1] = float(pred.model.gate(torch.tensor(GATE_CAP_K)))
    del trace[:]
    t0 = float(record["t0_s"])
    arm = scored_arm(record)
    err = None
    try:
        frames = pred.predict_states(record, {**record.get("cav_controllers", {}), "arm": arm},
                                     HORIZON_S, 0)
        result = analyse(model_name, vname, record, frames, trace, t0, arm)
    except Exception as exc:  # keep the batch alive
        import traceback
        err = traceback.format_exc()[-1500:]
        result = None
    return {"model": model_name, "variant": vname, "world": str(record.get("world")),
            "t0_s": t0, "key": record.get("key"), "result": result, "error": err,
            "seconds": time.time() - t_start}


# --------------------------------------------------------------------------- #
# per-run analysis
# --------------------------------------------------------------------------- #
def _arrays(frames: list[dict], recorded: list[dict], t0: float):
    ids = sorted({str(v["id"]) for fr in frames + recorded for v in fr.get("vehicles", [])})
    idx = {x: i for i, x in enumerate(ids)}
    n = N_STEPS
    out = {}
    for prefix, seq in (("p", frames), ("r", recorded)):
        lane = np.full((n + 1, len(ids)), -1, dtype=np.int16)
        speed = np.full((n + 1, len(ids)), np.nan, dtype=float)
        present = np.zeros((n + 1, len(ids)), dtype=bool)
        cav = np.zeros((n + 1, len(ids)), dtype=bool)
        for fr in seq:
            k = int(round((float(fr["time_s"]) - t0) / 0.5))
            if not (0 <= k <= n):
                continue
            for v in fr.get("vehicles", []):
                if not v.get("present", True):
                    continue
                j = idx[str(v["id"])]
                lane[k, j] = int(v.get("lane", -1))
                speed[k, j] = float(v["v"])
                present[k, j] = True
                cav[k, j] = bool(v.get("is_cav", False))
        out[prefix] = (lane, speed, present, cav)
    return ids, out


def true_lane_rate(p_lane, r_lane, p_pres, r_pres, p_cav, r_cav, lo: int, hi: int) -> dict:
    changes = steps = rec_changes = rec_steps = 0
    pred_filled, rec_filled = p_lane.copy(), r_lane.copy()
    for lanes in (pred_filled, rec_filled):
        for tick in range(1, lanes.shape[0]):
            bad = lanes[tick] < 0
            lanes[tick, bad] = lanes[tick - 1, bad]
    for k in range(lo + 1, hi + 1):
        valid = p_pres[k] & p_pres[k - 1] & r_pres[k] & r_pres[k - 1] & ~r_cav[k] & ~p_cav[k]
        for lanes, is_pred in ((pred_filled, True), (rec_filled, False)):
            cur, prev = lanes[k], lanes[k - 1]
            good = valid & (cur >= 0) & (prev >= 0)
            ch, st = int((good & (cur != prev)).sum()), int(good.sum())
            if is_pred:
                changes, steps = changes + ch, steps + st
            else:
                rec_changes, rec_steps = rec_changes + ch, rec_steps + st
    return {"pred_changes": changes, "rec_changes": rec_changes, "pred_steps": steps,
            "rec_steps": rec_steps,
            "pred_per_hdv_per_10s": changes / (steps / 20.0) if steps else None,
            "rec_per_hdv_per_10s": rec_changes / (rec_steps / 20.0) if rec_steps else None}


def analyse(model_name, vname, record, frames, trace, t0, arm):
    from wms.eval.metrics import car_metrics
    from wms.eval.normalize import recorded_for
    recorded = recorded_for(record, arm)
    ids, arr = _arrays(frames, recorded, t0)
    p_lane, p_v, p_pres, p_cav = arr["p"]
    r_lane, r_v, r_pres, r_cav = arr["r"]

    metrics = {}
    cm = car_metrics(frames, recorded, t0_s=t0, horizons=(60.0, 120.0))
    for h in ("60.0", "120.0"):
        block = cm[h]["hdv"]["abs_delta_s_m"]
        metrics[f"ds_med_{h[:-2]}"] = float(block["median"]) if block.get("median") is not None else None
    wrong = {}
    for h in (60, 120):
        k = h * 2
        mask = p_pres[k] & r_pres[k] & ~r_cav[k] & (r_lane[k] >= 0)
        wrong[str(h)] = [int((mask & (p_lane[k] != r_lane[k])).sum()), int(mask.sum())]
    rates = {"0-60": true_lane_rate(p_lane, r_lane, p_pres, r_pres, p_cav, r_cav, 0, 120),
             "60-120": true_lane_rate(p_lane, r_lane, p_pres, r_pres, p_cav, r_cav, 120, 240)}

    # ---- decision trace aggregation (slot space) ------------------------------ #
    decisions = []          # early-window lane-change decisions, with decomposition
    mobil_flips = {"0-60": 0, "60-120": 0}
    hdv_steps = {"0-60": 0, "60-120": 0}
    time_hist = np.zeros(12, dtype=np.int32)       # per-5 s bins, 0-60 s
    lane_hist = np.zeros(3, dtype=np.int32)
    zone_hist = {"s<-100": 0, "-100..0": 0, "0..75.44 (merge zone)": 0, ">75.44 (downstream)": 0}
    cav_dist = {"<25": 0, "25-50": 0, "50-100": 0, ">100": 0, "none": 0}
    junction_self = 0      # recorded lane of this vehicle is -1 (transit) within +-5 s
    junction_near = 0      # some recorded vehicle within +-60 m in junction (lane -1) now
    near_recorded_change = 0   # a recorded TRUE change for the same vehicle within +-5 s
    exit_soon = 0          # slot present now, gone (model) within 10 s
    for st in trace:
        k = st["k"]
        if k >= N_STEPS:
            continue
        hdv = st["present"] & ~st["is_cav"]
        cur = st["lane"]
        dec = st["decision"]
        base, corr = st["base"], st["corr"]
        bucket = "0-60" if k < 120 else "60-120"
        hdv_steps[bucket] += int((hdv & (cur >= 0)).sum())
        mobil_flip = base.argmax(-1) != cur
        mobil_flips[bucket] += int((hdv & (cur >= 0) & mobil_flip).sum())
        changed = hdv & (cur >= 0) & (dec != cur)
        if not changed.any() or k >= EARLY_STEPS:
            continue
        rows = np.nonzero(changed)[0]
        # context: nearest CAV gap
        cav_pos = st["s_m"][st["present"] & st["is_cav"]]
        for j in rows:
            new = int(dec[j])
            c = int(cur[j])
            others = [b for b in range(3) if b != c]
            best_other = max(others, key=lambda b: base[j, b])
            margin = float(base[j, c] - base[j, best_other])
            decisions.append({
                "k": k, "slot": int(j), "cur": c, "new": new, "jump": abs(new - c) == 2,
                "mobil_change": bool(base[j].argmax() != c),
                "mobil_margin": margin,
                "new_base": float(base[j, new]),
                "corr_adv": float(corr[j, new] - corr[j, c]),
                "corr_abs_max": float(np.abs(corr[j]).max()),
                "gate": st["gate"], "v": float(st["v_mps"][j]), "s": float(st["s_m"][j])})
            time_hist[min(k // 10, 11)] += 1
            lane_hist[c] += 1
            s_j = float(st["s_m"][j])
            zone_hist["s<-100" if s_j < -100 else
                      "-100..0" if s_j < 0 else
                      "0..75.44 (merge zone)" if s_j <= 75.44 else ">75.44 (downstream)"] += 1
            if len(cav_pos):
                d = float(np.min(np.abs(cav_pos - s_j)))
                cav_dist["<25" if d < 25 else "25-50" if d < 50 else "50-100" if d < 100 else ">100"] += 1
            else:
                cav_dist["none"] += 1
    # frame-space context per decision (junction proximity, recorded changes, exits)
    if decisions:
        vid_of_slot = {}
        # map model slots -> ids by matching t0 positions (slot order == ids order of t0_vehicles)
        from wms.eval.rollout_inputs import t0_vehicles
        _, ids0, kin0, _ = t0_vehicles(record)
        vid = {jj: str(ii) for jj, ii in enumerate(ids0)}
        id_idx = {x: i for i, x in enumerate(ids)}
        r_filled = r_lane.copy()
        for tick in range(1, r_filled.shape[0]):
            bad = r_filled[tick] < 0
            r_filled[tick, bad] = r_filled[tick - 1, bad]
        rec_change_step = {}
        for k in range(1, N_STEPS + 1):
            valid = r_pres[k] & r_pres[k - 1] & ~r_cav[k]
            for j in np.nonzero(valid & (r_filled[k] != r_filled[k - 1]) & (r_filled[k] >= 0))[0]:
                rec_change_step.setdefault(j, []).append(k)
        for d in decisions:
            j = id_idx.get(vid.get(d["slot"], ""), None)
            k = d["k"]
            if j is None:
                junction_self += 0
                junction_near += 0
                near_recorded_change += 0
                exit_soon += 0
                continue
            lo, hi = max(0, k - 10), min(N_STEPS, k + 10)
            junction_self += int((r_lane[lo:hi + 1, j] < 0).any())
            col = r_lane[k]
            # junction_near: any recorded vehicle with lane==-1 at k within +-60 m
            pos = np.full(len(ids), np.nan)
            for fr in recorded:
                kk = int(round((float(fr["time_s"]) - t0) / 0.5))
                if kk == k:
                    for v in fr.get("vehicles", []):
                        if v.get("present", True):
                            pos[id_idx[str(v["id"])]] = float(v["s"])
                    break
            if np.isfinite(pos).any():
                near = np.abs(pos - pos[j]) <= 60.0
                junction_near += int((near & (col < 0)).any())
            steps_j = rec_change_step.get(j, [])
            near_recorded_change += int(any(abs(kk - k) <= 10 for kk in steps_j))
            exit_soon += int(not p_pres[min(k + 20, N_STEPS), j])
    trace_summary = {"decisions": decisions, "mobil_flips": mobil_flips, "hdv_steps": hdv_steps,
                     "time_hist_5s": time_hist.tolist(), "lane_hist": lane_hist.tolist(),
                     "zone_hist": zone_hist, "cav_dist": cav_dist,
                     "junction_self": junction_self, "junction_near": junction_near,
                     "near_recorded_change": near_recorded_change, "exit_soon": exit_soon}
    return {"metrics": metrics, "wrong_lane": wrong, "rates": rates, "trace": trace_summary}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main():
    from concurrent.futures import ProcessPoolExecutor
    TASKS.mkdir(parents=True, exist_ok=True)
    records = quick20(WALL_SAME)
    print(f"{len(records)} states", flush=True)
    jobs = [("C1", "base"), ("R4", "base"), ("R4", "delta0"), ("R4", "gcap20")]
    payloads = [(m, v, rec) for m, v in jobs for rec in records]
    t0 = time.time()
    n = 0
    with ProcessPoolExecutor(max_workers=5, initializer=_init_worker) as pool:
        for res in pool.map(run_task, payloads, chunksize=1):
            raw = f"{res['model']}__{res['variant']}__{res['world']}__{res['key']}.pkl"
            name = raw.replace(":", "_").replace("\\", "_").replace("/", "_")
            with (TASKS / name).open("wb") as fh:
                pickle.dump(res, fh, protocol=4)
            n += 1
            print(f"[{n}/{len(payloads)}] {res['model']}/{res['variant']} {res['world']} "
                  f"{res['seconds']:.0f}s err={bool(res['error'])} ({time.time() - t0:.0f}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
