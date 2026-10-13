"""HORIZON-CHECK part A: aggregate (RSU-field / team-reward) accuracy vs horizon on recorded
unseen-wall states, next to the per-vehicle median |ds|.

Rollout path = the clean scorer's (``WmsPredictor.predict_states``, CAVs teacher-forced,
entries OFF, seed 0, 120 s), as in runs/active/WM-S-C4-20261013/score/score_c4.py.
The recorded futures in wallc1_states_h60(_real) contain ONLY the t0 roster (no entrants),
so every comparison here is roster-restricted ("cohort") by construction; the effect of
missing arrivals is measured in SUMO by part B (branch_horizon.py).

    python scripts/horizon_check/agg_horizon.py --out <dir> --workers 8 \
        --model C4=<wm_s_best.pt> --model C1=<wm_s.pt> ... [--sets wall_same wall_pilot wall_same_real]

No training, no simulation, no GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

DATA = Path("E:/research-dense/data_cache")
SETS = {
    "wall_same": DATA / "wallc1_states_h60" / "wall_same.json",
    "wall_pilot": DATA / "wallc1_states_h60" / "wall_pilot.json",
    "wall_same_real": DATA / "wallc1_states_h60_real" / "wall_same.json",
}
HORIZONS = (10, 30, 60, 120)
DT = 0.5
NSTEP = 240
ARM = "wall"
REWARD_KEYS = ("reward", "delay", "throughput", "hard_brake", "smooth", "fairness_delay")


def tested_arm(record: dict) -> str:
    rec = record.get("recorded") or {}
    return ARM if ARM in rec else next(iter(rec))


# --------------------------------------------------------------------------- arrays
def frame_arrays(frames: list, ids: list, t0: float, is_cav0: dict) -> dict:
    idx = {x: i for i, x in enumerate(ids)}
    n = len(ids)
    s = np.zeros((NSTEP + 1, n))
    v = np.zeros((NSTEP + 1, n))
    lane = np.full((NSTEP + 1, n), -1, np.int64)
    pres = np.zeros((NSTEP + 1, n), bool)
    for fr in frames:
        k = int(round((float(fr["time_s"]) - t0) / DT))
        if not 0 <= k <= NSTEP:
            continue
        for row in fr["vehicles"]:
            if not row.get("present", True):
                continue
            j = idx.get(str(row["id"]))
            if j is None:
                continue
            s[k, j], v[k, j] = float(row["s"]), float(row["v"])
            try:
                lane[k, j] = int(row.get("lane", -1))
            except (TypeError, ValueError):
                lane[k, j] = -1
            pres[k, j] = True
    cav = np.array([bool(is_cav0.get(x, False)) for x in ids])
    return {"s": s, "v": v, "lane": lane, "present": pres, "is_cav": cav}


def in_zone(s, lane):
    from wms.control.contract import RSU_MAINLINE_S_M, RSU_RAMP_S_M
    # lane -1 (SUMO junction-internal) counts by s on the mainline span (merge transits)
    main = (((lane == 0) | (lane == 1) | (lane == -1))
            & (s >= RSU_MAINLINE_S_M[0]) & (s < RSU_MAINLINE_S_M[1]))
    ramp = (lane == 2) & (s >= RSU_RAMP_S_M[0]) & (s < RSU_RAMP_S_M[1])
    return main | ramp


def fields(a: dict, ks) -> np.ndarray:
    """[len(ks), N_cells, 6] RSU fields at steps ks."""
    import torch
    from wms.control.rsu_field import build_rsu_field
    ks = list(ks)
    st = {"s_m": torch.as_tensor(a["s"][ks], dtype=torch.float32),
          "v_mps": torch.as_tensor(a["v"][ks], dtype=torch.float32),
          "lane": torch.as_tensor(a["lane"][ks]),
          "present": torch.as_tensor(a["present"][ks]),
          "is_cav": torch.as_tensor(np.broadcast_to(a["is_cav"], a["present"][ks].shape).copy())}
    return build_rsu_field(st).numpy()


def reward_curves(a: dict, zone_only: bool) -> dict:
    """Cumulative team_reward components over [0, h] for every h (all 240 transitions batched)."""
    import torch
    from wms.control.contract import RewardWeights
    from wms.control.reward import team_reward
    s, v, lane, pres = a["s"], a["v"], a["lane"], a["present"]
    p0, p1 = pres[:-1], pres[1:]
    cont = p0 & p1
    # exits: carry the last values into the next state (identity preserved)
    s1 = np.where(p1, s[1:], s[:-1])
    v1 = np.where(p1, v[1:], v[:-1])
    l1 = np.where(p1, lane[1:], lane[:-1])
    acc = np.where(cont, (v[1:] - v[:-1]) / DT, 0.0)
    m0 = p0 & (in_zone(s[:-1], lane[:-1]) if zone_only else True)
    st = {"s_m": torch.as_tensor(np.where(m0, s[:-1], 0.0), dtype=torch.float32),
          "v_mps": torch.as_tensor(np.where(m0, v[:-1], 0.0), dtype=torch.float32),
          "lane": torch.as_tensor(np.where(m0, lane[:-1], -1)),
          "present": torch.as_tensor(m0),
          "length_m": torch.full(s[:-1].shape, 5.0)}
    nx = {"s_m": torch.as_tensor(s1, dtype=torch.float32),
          "v_mps": torch.as_tensor(v1, dtype=torch.float32),
          "lane": torch.as_tensor(l1),
          "present": torch.as_tensor(m0 & p1),
          "length_m": torch.full(s[:-1].shape, 5.0)}
    with torch.no_grad():
        out = team_reward(st, nx, torch.as_tensor(np.where(m0, acc, 0.0), dtype=torch.float32),
                          RewardWeights())
    res = {}
    for key in REWARD_KEYS:
        x = out[key].double().numpy()
        res[key] = {str(h): float(x[:2 * h].mean() if key == "fairness_delay" else x[:2 * h].sum())
                    for h in HORIZONS}
    return res


def cv_baseline(R: dict) -> dict:
    """Constant-speed baseline: HDVs keep their t0 speed and lane, exit past the furthest
    recorded position; CAVs copied from the record (teacher-forced, as in the WM)."""
    s0, v0, l0 = R["s"][0], R["v"][0], R["lane"][0]
    t = np.arange(NSTEP + 1)[:, None] * DT
    exit_s = float(R["s"][R["present"]].max())
    s = s0[None] + v0[None] * t
    pres = (s <= exit_s) & R["present"][0][None]
    cav = R["is_cav"][None]
    return {"s": np.where(cav, R["s"], s), "v": np.where(cav, R["v"], np.broadcast_to(v0, s.shape)),
            "lane": np.where(cav, R["lane"], np.broadcast_to(l0, s.shape)),
            "present": np.where(cav, R["present"], pres), "is_cav": R["is_cav"]}


def hdv_ds_median(P: dict, R: dict, k: int):
    m = P["present"][k] & R["present"][k] & ~R["is_cav"]
    return float(np.median(np.abs(P["s"][k][m] - R["s"][k][m]))) if m.any() else None


# --------------------------------------------------------------------------- worker
_W: dict = {}


def _init(models: dict) -> None:
    import torch
    torch.set_num_threads(1)
    from wms.eval.predictor import RolloutFlags, WmsPredictor
    _W["pred"] = {tag: WmsPredictor.from_checkpoint(p, RolloutFlags(entries=False, entries_v2=False))
                  for tag, p in models.items()}
    _W["first"] = next(iter(models))


def _work(payload) -> dict:
    tag, set_name, i, record = payload
    from wms.eval.metrics import car_metrics
    from wms.eval.normalize import initial_frame, recorded_for
    from wms.eval.score import car_compact
    t0 = float(record["t0_s"])
    arm = tested_arm(record)
    key = {"set": set_name, "i": i, "world": str(record.get("world")), "key": record.get("key"),
           "t0_s": t0, "model": tag}
    try:
        pred = _W["pred"][tag].predict_states(record, {**record.get("cav_controllers", {}),
                                                       "arm": arm}, 120.0, 0)
        rec = recorded_for(record, arm)
        car = car_compact(car_metrics(pred, rec, t0_s=t0, horizons=tuple(float(h) for h in HORIZONS)))
        f0 = initial_frame(record)
        is_cav0 = {str(r["id"]): bool(r["is_cav"]) for r in f0["vehicles"]}
        ids = sorted(is_cav0)
        P = frame_arrays(pred, ids, t0, is_cav0)
        R = frame_arrays(rec, ids, t0, is_cav0)
        ks = [0] + [2 * h for h in HORIZONS]
        fp, fr = fields(P, ks), fields(R, ks)
        out = {**key, "error": None,
               "ds_hdv_med": {str(h): car[str(float(h))]["hdv"]["ds_med"] for h in HORIZONS},
               "ds_all_med": {str(h): car[str(float(h))]["all"]["ds_med"] for h in HORIZONS},
               "field_pred": fp[..., [0, 1, 3, 4]].astype(np.float32),
               "field_rec": fr[..., [0, 1, 3, 4]].astype(np.float32),
               "n_roster": len(ids),
               "n_present_rec": {str(h): int(R["present"][2 * h].sum()) for h in HORIZONS},
               "n_present_pred": {str(h): int(P["present"][2 * h].sum()) for h in HORIZONS},
               "reward_pred": {"roster": reward_curves(P, False), "zone": reward_curves(P, True)},
               "reward_rec": {"roster": reward_curves(R, False), "zone": reward_curves(R, True)}}
        if tag == _W["first"]:
            Q = cv_baseline(R)
            fq = fields(Q, ks)
            cv = dict(out, model="CV-baseline",
                      ds_hdv_med={str(h): hdv_ds_median(Q, R, 2 * h) for h in HORIZONS},
                      ds_all_med=None, field_pred=fq[..., [0, 1, 3, 4]].astype(np.float32),
                      n_present_pred={str(h): int(Q["present"][2 * h].sum()) for h in HORIZONS},
                      reward_pred={"roster": reward_curves(Q, False), "zone": reward_curves(Q, True)})
            return [out, cv]
        return [out]
    except Exception as exc:  # keep the batch alive
        import traceback
        return [{**key, "error": f"{type(exc).__name__}: {exc} {traceback.format_exc()[-300:]}"}]


# --------------------------------------------------------------------------- stats
def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def scalar_stats(pred, rec) -> dict:
    p, r = np.asarray(pred, float), np.asarray(rec, float)
    ok = np.isfinite(p) & np.isfinite(r)
    p, r = p[ok], r[ok]
    if p.size == 0:
        return {"n": 0}
    mae = float(np.abs(p - r).mean())
    denom = float(np.abs(r).mean())
    return {"n": int(p.size), "mae": mae, "bias": float((p - r).mean()),
            "rel_err": (mae / denom) if denom > 0 else None, "pearson": pearson(p, r),
            "rec_mean": float(r.mean()), "pred_mean": float(p.mean())}


def summarize(rows: list) -> dict:
    """Per horizon: per-vehicle |ds|, cell-level field errors, zone aggregates, reward sums."""
    good = [r for r in rows if not r.get("error")]
    out = {"n_states": len(good), "n_errors": len(rows) - len(good), "h": {}}
    for hi, h in enumerate(HORIZONS):
        k = hi + 1          # index into the field stack (0 = t0)
        H = str(h)
        blk = {}
        ds = [r["ds_hdv_med"][H] for r in good if r["ds_hdv_med"][H] is not None]
        blk["hdv_abs_ds_median_of_state_medians_m"] = float(np.median(ds)) if ds else None
        fp = np.stack([r["field_pred"][k] for r in good])   # [S, C, 4] density, speed, queue, occ
        fr = np.stack([r["field_rec"][k] for r in good])
        # cells: density everywhere; speed where both occupied; queue flag where either occupied
        dens = scalar_stats(fp[..., 0].ravel(), fr[..., 0].ravel())
        both = (fp[..., 3] > 0) & (fr[..., 3] > 0)
        spd = scalar_stats(fp[..., 1][both], fr[..., 1][both])
        either = (fp[..., 3] > 0) | (fr[..., 3] > 0)
        qp, qr = fp[..., 2][either] > 0.5, fr[..., 2][either] > 0.5
        tp = int((qp & qr).sum())
        queue = {"n_cells": int(either.sum()), "rec_queue_cells": int(qr.sum()),
                 "pred_queue_cells": int(qp.sum()),
                 "accuracy": float((qp == qr).mean()) if qp.size else None,
                 "recall": tp / int(qr.sum()) if qr.sum() else None,
                 "precision": tp / int(qp.sum()) if qp.sum() else None}
        blk["cell"] = {"density_veh_per_km": dens, "mean_speed_mps_both_occupied": spd,
                       "queue_flag": queue,
                       "occupancy_agreement": float(((fp[..., 3] > 0) == (fr[..., 3] > 0)).mean())}
        # zone aggregates per state
        length_km = 0.05
        cnt_p = (fp[..., 0] * length_km).sum(-1)
        cnt_r = (fr[..., 0] * length_km).sum(-1)
        vs_p = (fp[..., 0] * length_km * fp[..., 1]).sum(-1) / np.maximum(cnt_p, 1e-9)
        vs_r = (fr[..., 0] * length_km * fr[..., 1]).sum(-1) / np.maximum(cnt_r, 1e-9)
        okv = (cnt_p > 0.5) & (cnt_r > 0.5)
        blk["zone"] = {"vehicle_count": scalar_stats(cnt_p, cnt_r),
                       "mean_speed_mps": scalar_stats(vs_p[okv], vs_r[okv]),
                       "queued_cells": scalar_stats(fp[..., 2].sum(-1), fr[..., 2].sum(-1))}
        # cumulative team-reward components over [t0, t0+h]
        rew = {}
        for scope in ("roster", "zone"):
            rew[scope] = {key: scalar_stats([r["reward_pred"][scope][key][H] for r in good],
                                            [r["reward_rec"][scope][key][H] for r in good])
                          for key in REWARD_KEYS}
        blk["team_reward_cumulative"] = rew
        blk["roster_present_fraction_rec"] = float(np.mean(
            [r["n_present_rec"][H] / max(r["n_roster"], 1) for r in good]))
        out["h"][H] = blk
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="TAG=path")
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    models = dict(m.split("=", 1) for m in args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    from wms.eval.normalize import iter_windows
    t_start = time.time()
    results = {"schema": "c4_horizon_check_A/1", "models": models, "sets": {},
               "horizons_s": list(HORIZONS), "notes": [
                   "clean-scorer rollout (WmsPredictor.predict_states), CAVs teacher-forced, "
                   "entries off, seed 0; recorded futures contain only the t0 roster, so every "
                   "field/reward here is roster-restricted on BOTH sides",
                   "RSU field = wms.control.rsu_field.build_rsu_field (52 cells x 50 m); team "
                   "reward = wms.control.reward.team_reward (default weights), accel = finite "
                   "difference of speeds, exits carried; scope roster = all t0 vehicles, zone = "
                   "only vehicles inside the RSU zones at each step",
                   "stats over states (zone/reward) or over (state, cell) pairs (cell); rel_err = "
                   "MAE / mean|recorded|"]}
    per_state = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(models,)) as pool:
        for set_name in args.sets:
            t_set = time.time()
            rows, chunk = [], []

            def flush(chunk):
                jobs = [(tag, set_name, i, r) for i, r in chunk for tag in models]
                rows.extend(x for res in pool.map(_work, jobs, chunksize=1) for x in res)
            for i, r in enumerate(iter_windows(SETS[set_name], limit=args.limit)):
                chunk.append((i, r))
                if len(chunk) >= 2 * args.workers:
                    flush(chunk)
                    chunk = []
            if chunk:
                flush(chunk)
            del chunk
            errs = [r for r in rows if r.get("error")]
            for e in errs[:3]:
                print("ERROR", e["model"], e["key"], e["error"][:300], flush=True)
            results["sets"][set_name] = {tag: summarize([r for r in rows if r["model"] == tag])
                                         for tag in list(models) + ["CV-baseline"]}
            for r in rows:
                if r.get("error"):
                    continue
                per_state.append({k: r[k] for k in ("set", "i", "world", "t0_s", "model",
                                                    "ds_hdv_med", "n_roster", "n_present_rec")}
                                 | {"delay_roster_pred": r["reward_pred"]["roster"]["delay"],
                                    "delay_roster_rec": r["reward_rec"]["roster"]["delay"],
                                    "delay_zone_pred": r["reward_pred"]["zone"]["delay"],
                                    "delay_zone_rec": r["reward_rec"]["zone"]["delay"]})
            print(f"{set_name}: {len(rows)} rollouts, {len(errs)} errors, "
                  f"{time.time() - t_set:.0f} s", flush=True)
            (out / "agg_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    results["runtime_s"] = round(time.time() - t_start, 1)
    (out / "agg_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    (out / "agg_per_state.json").write_text(json.dumps(per_state, separators=(",", ":")),
                                            encoding="utf-8")
    print(json.dumps({s: {t: {h: results["sets"][s][t]["h"][h]["hdv_abs_ds_median_of_state_medians_m"]
                              for h in results["sets"][s][t]["h"]} for t in results["sets"][s]}
                      for s in results["sets"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
