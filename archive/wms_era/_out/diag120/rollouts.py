"""Re-run WM-S rollouts for DIAG120 and store compact per-state diagnostics.

Per state (one at a time, memory-safe):
  * per-step (0.5 s) HDV |ds|/|dv| median and p90 vs the recorded arm;
  * per-step signed section metrics (stopped, queue, mean speed) pred vs rec;
  * per-vehicle onset table (first |ds| > 10 m), t0 leader, merge proximity;
  * lane-change and entrant event counts.

Output: one JSONL line per state.  CPU only, no SUMO, no training.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

WS = Path(r"E:/rd-wt/ws-diag120")
sys.path.insert(0, str(WS / "output/wd01/wm_n1"))

from wms_eval.c1v2 import iter_windows  # noqa: E402
from wms_eval.models import load_model  # noqa: E402
from wms_eval.data import initial_frame, recorded_for  # noqa: E402

DT = 0.5
MERGE_LO, MERGE_HI = 0.0, 75.44
ONSET_M = 10.0


def section_of(s: float) -> str:
    if s < 0.0:
        return "up"
    if s <= MERGE_HI:
        return "merge"
    return "down"


def present(frame: dict) -> dict[str, dict]:
    return {str(r["id"]): r for r in frame.get("vehicles", []) if r.get("present", True)}


def per_vehicle_series(frames: list[dict], t0: float, n: int) -> dict[str, dict]:
    """id -> array of 4 (s, v, lane, present) per step."""
    out: dict[str, dict] = {}
    by_t = {}
    for f in frames:
        by_t[int(round((float(f["time_s"]) - t0) / DT))] = f
    for i in range(n + 1):
        f = by_t.get(i)
        if f is None:
            continue
        for r in f["vehicles"]:
            if not r.get("present", True):
                continue
            vid = str(r["id"])
            arr = out.setdefault(vid, {"s": np.full(n + 1, np.nan),
                                       "v": np.full(n + 1, np.nan),
                                       "lane": np.full(n + 1, -9.0)})
            arr["s"][i] = float(r["s"])
            arr["v"][i] = float(r["v"])
            arr["lane"][i] = float(r.get("lane", -1))
    return out


def step_summary(pred: dict[str, dict], rec: dict[str, dict], n: int,
                 hdv_ids: list[str]) -> dict:
    ds_med = np.full(n + 1, np.nan)
    dv_med = np.full(n + 1, np.nan)
    ds_p90 = np.full(n + 1, np.nan)
    dv_p90 = np.full(n + 1, np.nan)
    matched_n = np.zeros(n + 1, dtype=int)
    for i in range(n + 1):
        ds, dv = [], []
        for vid in hdv_ids:
            pa = pred.get(vid)
            ra = rec.get(vid)
            if pa is None or ra is None:
                continue
            sp, sr = pa["s"][i], ra["s"][i]
            if np.isnan(sp) or np.isnan(sr):
                continue
            ds.append(abs(sp - sr))
            dv.append(abs(pa["v"][i] - ra["v"][i]))
        if not ds:
            continue
        ds = np.array(ds)
        dv = np.array(dv)
        ds_med[i] = float(np.median(ds))
        dv_med[i] = float(np.median(dv))
        ds_p90[i] = float(np.percentile(ds, 90))
        dv_p90[i] = float(np.percentile(dv, 90))
        matched_n[i] = len(ds)
    return {"ds_med": ds_med, "dv_med": dv_med, "ds_p90": ds_p90,
            "dv_p90": dv_p90, "n": matched_n}


def signed_sections(frame: dict) -> dict:
    rows = present(frame)
    out = {}
    for name in ("up", "merge", "down"):
        sec = [r for r in rows.values() if section_of(float(r["s"])) == name]
        speeds = [float(r["v"]) for r in sec]
        stopped = [r for r in sec if float(r["v"]) < 0.5]
        queue = (max(float(r["s"]) for r in stopped) - min(float(r["s"]) for r in stopped)
                 if (name == "merge" and stopped) else 0.0)
        out[name] = {"n": len(sec),
                     "mean_speed": (float(np.mean(speeds)) if speeds else None),
                     "stopped": len(stopped), "queue": float(queue)}
    return out


def _sid_list(frames: list[dict], t0: float) -> dict[int, dict]:
    by_t = {}
    for f in frames:
        by_t[int(round((float(f["time_s"]) - t0) / DT))] = f
    return by_t


def leader_map(init_frame: dict) -> dict[str, dict]:
    """t0 leader id per vehicle: nearest vehicle ahead in the same lane."""
    rows = present(init_frame)
    out = {}
    by_lane: dict[int, list] = {}
    for vid, r in rows.items():
        by_lane.setdefault(int(r.get("lane", -1)), []).append((float(r["s"]), vid))
    for lane, items in by_lane.items():
        items.sort()
        for i, (s, vid) in enumerate(items):
            lead = items[i + 1][1] if i + 1 < len(items) else None
            out[vid] = {"leader": lead, "lane": lane}
    return out


def analyse(record: dict, predictor, arm: str, horizon: float) -> dict:
    t0 = float(record.get("t0_s", record.get("state", {}).get("time_s", 0.0)))
    n = int(round(horizon / DT))
    seed = int(record.get("seed", 0))
    init = initial_frame(record)
    rec_frames = recorded_for(record, arm)
    pred_frames = predictor(record, {"arm": arm}, horizon, seed)

    pred = per_vehicle_series(pred_frames, t0, n)
    rec = per_vehicle_series(rec_frames, t0, n)
    init_rows = present(init)
    hdv_ids = [str(vid) for vid, r0 in init_rows.items() if not bool(r0.get("is_cav"))]
    summary = step_summary(pred, rec, n, hdv_ids)

    # section metrics over time
    pred_t = _sid_list(pred_frames, t0)
    rec_t = _sid_list(rec_frames, t0)
    sec_pred, sec_rec = {}, {}
    for i in range(n + 1):
        if i in pred_t:
            sec_pred[i] = signed_sections(pred_t[i])
        if i in rec_t:
            sec_rec[i] = signed_sections(rec_t[i])

    # onset + attribution
    lmap = leader_map(init)
    init_rows = present(init)
    entrant_ids = [vid for vid in rec if vid not in init_rows]
    entrants = []
    for vid in entrant_ids:
        arr = rec[vid]["s"]
        idx = np.flatnonzero(~np.isnan(arr))
        if idx.size:
            j = int(idx[0])
            entrants.append({"id": vid, "t_s": float(j * DT),
                             "s": float(arr[j]), "lane": int(rec[vid]["lane"][j])})
    veh_out = []
    for vid, r0 in init_rows.items():
        if bool(r0.get("is_cav")):
            continue
        ra = rec.get(vid)
        pa = pred.get(vid)
        if ra is None or pa is None:
            continue
        m = ~np.isnan(ra["s"]) & ~np.isnan(pa["s"])
        if not m.any():
            continue
        ds = np.abs(pa["s"] - ra["s"])
        dv = np.abs(pa["v"] - ra["v"])
        over = np.flatnonzero((ds > ONSET_M) & m)
        onset = float(over[0] * DT) if over.size else None
        peak_i = int(np.nanargmax(ds))
        lead = lmap.get(vid, {}).get("leader")
        is_ent = any(e["id"] == vid for e in entrants)
        veh_out.append({
            "id": vid, "s0": float(r0["s"]), "v0": float(r0["v"]),
            "lane0": int(r0.get("lane", -1)),
            "in_merge0": bool(MERGE_LO <= float(r0["s"]) <= MERGE_HI),
            "ds_peak": float(ds[peak_i]), "t_peak_s": float(peak_i * DT),
            "ds120": float(ds[min(n, 240)]), "dv120": float(dv[min(n, 240)]),
            "onset_s": onset, "leader": lead,
            "leader_onset_s": None,  # filled after
            "entrant": is_ent,
            "rec_lane_changes": int(np.sum(np.abs(np.diff(ra["lane"][m])) > 0.5)),
            "pred_lane_changes": int(np.sum(np.abs(np.diff(pa["lane"][m])) > 0.5)),
        })
    onset_of = {v["id"]: v["onset_s"] for v in veh_out}
    for v in veh_out:
        v["leader_onset_s"] = onset_of.get(v["leader"]) if v["leader"] else None

    # blow flag at horizons
    ds60 = summary["ds_med"][int(round(60.0 / DT))]
    ds120 = summary["ds_med"][min(n, int(round(120.0 / DT)))]
    return {
        "world": record.get("world"), "t0_s": t0, "arm": arm, "seed": seed,
        "n_hdv0": sum(1 for r in init_rows.values() if not bool(r.get("is_cav"))),
        "n_entrants": len(entrants), "entrants": entrants[:40],
        "ds60": (None if np.isnan(ds60) else float(ds60)),
        "ds120": (None if np.isnan(ds120) else float(ds120)),
        "blow": bool((not np.isnan(ds60) and ds60 > 50.0) or
                     (not np.isnan(ds120) and ds120 > 100.0)),
        "t": [round(i * DT, 1) for i in range(n + 1)],
        "ds_med": [None if np.isnan(x) else round(float(x), 3) for x in summary["ds_med"]],
        "dv_med": [None if np.isnan(x) else round(float(x), 3) for x in summary["dv_med"]],
        "ds_p90": [None if np.isnan(x) else round(float(x), 3) for x in summary["ds_p90"]],
        "n_matched": [int(x) for x in summary["n"]],
        "sec_pred": {str(k): v for k, v in sec_pred.items()},
        "sec_rec": {str(k): v for k, v in sec_rec.items()},
        "vehicles": veh_out,
        "n_vehicles": len(veh_out),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("struct", "oldhist", "run3"))
    ap.add_argument("--states", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="pi")
    ap.add_argument("--horizon", type=float, default=120.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--inflight", type=int, default=6)
    args = ap.parse_args()

    ck = WS / f"_out/diag120/ckpt/{args.model}_wm_s.pt"
    physics_path = WS / f"_out/diag120/ckpt/{args.model}_physics.json"
    physics = json.loads(physics_path.read_text()) if physics_path.is_file() else None

    import os
    from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
    os.environ["WMS_EVAL_THREADS"] = str(max(1, 4 // args.workers))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    n_done = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(str(ck), physics, args.arm, args.horizon)) as pool:
        pending = {}
        it = enumerate(iter_windows(args.states, limit=args.limit))
        exhausted = False

        def submit():
            nonlocal exhausted
            if exhausted:
                return
            try:
                idx, rec = next(it)
            except StopIteration:
                exhausted = True
                return
            pending[pool.submit(_work, (idx, rec))] = idx

        for _ in range(args.inflight):
            submit()
        with out_path.open("w", encoding="utf-8") as fh:
            while pending:
                done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
                for fut in done:
                    idx, res = fut.result()
                    fh.write(json.dumps(res) + "\n")
                    fh.flush()
                    n_done += 1
                    del pending[fut]
                    submit()
                    if n_done % 10 == 0:
                        print(f"done {n_done} ({time.time()-t_start:.0f}s)", flush=True)
    print(f"wrote {out_path} n={n_done} in {time.time()-t_start:.0f}s", flush=True)
    return 0


_W: dict = {}


def _init(ck: str, physics, arm: str, horizon: float) -> None:
    predictor, _flags = load_model("wms", checkpoint=ck, physics=physics)
    _W.update({"predictor": predictor, "arm": arm, "horizon": horizon})


def _work(payload):
    idx, rec = payload
    try:
        return idx, analyse(rec, _W["predictor"], _W["arm"], _W["horizon"])
    except Exception as exc:  # one state must not abort the run
        return idx, {"world": rec.get("world"), "error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    raise SystemExit(main())
