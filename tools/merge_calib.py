"""MERGECAL (2026-10-18): position-binned merge-head calibration of a WM-S hazard_v5 checkpoint
against SUMO's ramp departure hazard on TRAIN-val starts (inference only; no SUMO, no training).

Problem (runs/active/C4-V7FT-20261017/stuck_diag/STUCK_DIAG.md): ramp HDVs merge too late in WM-S
rollouts.  The merge head of a ``lane_decision='gated'`` model samples only when its teacher-forced
TRAIN-val AP reaches an absolute gate of 0.5 (else argmax, i.e. never at P < 0.5), and even an open,
teacher-forced-calibrated head fires at the wrong positions in a free rollout because the rollout's
feasibility mask is open less often than SUMO's traffic.

Method (two parts, both fitted on TRAIN-val data only):

1. **Gate.**  The merge head samples when it ranks better than a constant predictor: its gate is the
   constant predictor's AP (= the event rate of the merge rows, ``ap_constant`` of the checkpoint's
   ``lane_calib.json`` / ``tools/lanehazard_calib.py``) instead of the absolute 0.5.
2. **Per-position hazard offsets.**  The merge-head logit gets an additive offset per bin of the
   distance to the acceleration-lane end ``d = 75.44 - s`` (``lv5.merge_position_offset``).  The
   offsets are fitted so that the closed-loop discrete-time departure hazard of the rollout,
   ``h_b = departures in bin b / ramp-lane car-steps at risk in bin b`` (t0-roster ramp HDVs, CAVs
   teacher-forced, recorded SUMO exits end slots: the evaluation semantics of
   ``tools/v8wm3_check.py``), equals SUMO's recorded hazard of the same cars on the same starts.
   Fixed-point iteration ``offset_b += clip(log(h_sumo_b / h_wm_b), +-step)``; the forced 5 m
   window is fitted only when the head acts there (``forced_window_mode='hazard'``).

Starts: TRAIN-val episodes (world-grouped ``split_role == "val"`` of converted C4 sources; never the
wall / held-out / CAL sets, never evaluation seeds 964000-964199), t0 every ``--t0`` second.

    python tools/merge_calib.py records --source DIR [--source DIR] --t0 420,780 --out recs.pkl
    python tools/merge_calib.py fit CKPT --records recs.pkl --out-dir DIR [--iters 4] [--workers 10]
        [--gate-ap-constant 0.0525] [--write wm_s_best_ap_mcal.pt]
    python tools/merge_calib.py score CKPT --records recs.pkl --out x.json   (no fitting)

Run from the worktree root with ``PYTHONPATH="src;.;tools"``.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np  # noqa: E402

FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1")
EVAL_SEEDS = range(964000, 964200)
DT = 0.5
ZONE_END_M = 75.44
FORCE_WINDOW_M = 5.0
#: Distance-to-lane-end bins (m); bin 0 is the forced window.
EDGES = (0.0, 5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 80.0)
HORIZON_STEPS = 240
MAX_STEP = 1.5
MAX_OFFSET = 6.0


def guard(path) -> str:
    s = str(path).replace("\\", "/").lower()
    if any(t in s for t in FORBIDDEN):
        raise SystemExit(f"refusing restricted path {path}")
    return str(path)


# ----------------------------------------------------------------------------- records
def episode_dict(ep_dir: Path) -> dict:
    """A converted C4 episode (``wms.data.c4_episodes`` arrays) in the raw-episode layout of
    ``tools/c4_episode_states.EpisodeTimeline`` (vehicle ids ``v<index>``)."""
    a = {name: np.load(ep_dir / f"{name}.npy", allow_pickle=False)
         for name in ("veh_kin", "veh_mask", "vid", "is_cav", "time_s", "counter")}
    meta = json.loads((ep_dir / "episode.json").read_text(encoding="utf-8"))
    kin, present = a["veh_kin"], a["veh_mask"].astype(bool)
    onehot = kin[..., 1:4] > 0.5
    lane = np.where(onehot.any(-1), onehot.argmax(-1), -1)
    vid = a["vid"].astype(np.int64)
    n_v = int(vid.max()) + 1 if vid.size else 0
    cav = np.zeros(n_v, bool)
    sel = present & a["is_cav"].astype(bool) & (vid >= 0)
    cav[np.unique(vid[sel])] = True
    names = np.asarray([f"v{i}" for i in range(n_v)])
    return {"time_s": a["time_s"], "s_m": kin[..., 0], "lane": lane, "v_mps": kin[..., 4],
            "a_mps2": kin[..., 5], "present": present, "vid_index": vid, "vehicle_ids": names,
            "cav_vehicle_ids": names[cav], "counter": a["counter"], "meta": meta}


def build_records(sources: list[str], t0s: list[float], arm_name: str | None = None) -> list:
    from c4_episode_states import EpisodeTimeline  # noqa: F401  (import check)
    import c4_episode_states as ces

    out = []
    for src in sources:
        root = Path(guard(src))
        man = json.loads((root / "C4_EPISODES_MANIFEST.json").read_text(encoding="utf-8"))
        for e in man["episodes"]:
            if e.get("split_role") != "val":
                continue
            if int(e["seed"]) in EVAL_SEEDS:
                raise SystemExit(f"evaluation seed {e['seed']} refused")
            ep_dir = root / e["dir"]
            if not (ep_dir / "veh_kin.npy").is_file():
                print(f"missing {ep_dir}", flush=True)
                continue
            ep = episode_dict(ep_dir)
            arm = arm_name or str(e["arm"])
            for t0 in t0s:
                j0 = int(round((t0 - float(ep["time_s"][0])) / DT))
                rec = ces.episode_record(ep, t0_s=float(t0), arm=arm, world=str(e["world"]))
                rec["counter"] = np.asarray(ep["counter"][j0], np.float32).tolist()
                rec["source"] = str(root.name)
                rec["key"] = f"mcal/{root.name}/{e['episode']}@{t0:g}"
                out.append(rec)
            print(f"{root.name}/{e['episode']}: {len(out)} records", flush=True)
    return out


# ----------------------------------------------------------------------------- exposure
def ramp_exposure(s: np.ndarray, lane: np.ndarray, present: np.ndarray, cars,
                  edges=EDGES, zone_end: float = ZONE_END_M) -> dict:
    """Discrete-time departure exposure of ``cars`` (columns) in ``[T+1, N]`` arrays.

    A car-step t is at risk when the car is present at t and t + 1, in the ramp lane (2) at t and
    inside the merge zone (``0 <= s_t <= zone_end``); its bin is that of ``d = zone_end - s_t``.
    The event is the first step whose lane at t + 1 is not 2 (the car is no longer at risk after
    it).  Returns ``at_risk [nb]``, ``events [nb]`` and the departure distances / times."""
    nb = len(edges) - 1
    risk, ev = np.zeros(nb), np.zeros(nb)
    dep_d, dep_t = [], []
    e = np.asarray(edges, float)
    for k in cars:
        for t in range(s.shape[0] - 1):
            if not (present[t, k] and present[t + 1, k]) or int(lane[t, k]) != 2:
                if int(lane[t, k]) != 2 and present[t, k]:
                    break                       # left the ramp lane before (or at) t
                continue
            if not (0.0 <= s[t, k] <= zone_end):
                continue
            d = zone_end - float(s[t, k])
            b = int(np.searchsorted(e, d, side="right") - 1)
            if not 0 <= b < nb:
                continue
            risk[b] += 1
            if int(lane[t + 1, k]) != 2:
                ev[b] += 1
                dep_d.append(d)
                dep_t.append((t + 1) * DT)
                break
    return {"at_risk": risk, "events": ev, "dep_d": dep_d, "dep_t": dep_t}


def first_off2(lane, msk, k) -> float:
    idx = np.flatnonzero(msk[:, k] & (lane[:, k] != 2))
    return float(idx[0] * DT) if idx.size else np.inf


# ----------------------------------------------------------------------------- workers
G: dict = {}


def apply_variant(model, var: dict) -> None:
    """Set the calibration of ``var`` (``gate`` merge gate, ``calib`` (edges, offsets)) on a
    loaded model (plain attributes; the checkpoint itself is unchanged)."""
    if var.get("gate") is not None:
        g = tuple(model.lane_v5_gates)
        model.lane_v5_gates = (g[0], g[1], float(var["gate"]))
    model.lane_v5_merge_calib = (None if var.get("calib") is None
                                 else (tuple(var["calib"][0]), tuple(var["calib"][1])))


def init(ckpt: str, var: dict) -> None:
    import torch
    import wm_eval_common as wc
    torch.set_num_threads(1)
    p, cap = wc.load_predictor(guard(ckpt), "AS_IS", hdv_exits="sumo")
    apply_variant(p.model, var)
    G["p"], G["cap"] = p, cap


def work(payload) -> dict:
    idx, rec = payload
    from wms.eval.rollout_inputs import recorded_teacher, t0_vehicles
    p, cap = G["p"], G["cap"]
    arm = rec["arm"]
    try:
        t0, ids, kin, is_cav = t0_vehicles(rec)
        teacher, present_seq, _ = recorded_teacher(rec, t0, ids, kin, HORIZON_STEPS, arm,
                                                   p.backend.tools)
        if teacher is None:
            return {"idx": idx, "error": "no arm"}
        p.predict_states(rec, {**rec.get("cav_controllers", {}), "arm": arm},
                         HORIZON_STEPS * DT, 0)
        r = cap["last"]
        n = len(ids)
        tr = teacher[0].numpy()
        rp = present_seq[0].numpy().astype(bool)
        pos = np.concatenate([tr[:1, :, 0], r["positions"][0, :, :n].numpy()], 0)
        lan = np.concatenate([tr[:1, :, 1], r["lanes"][0, :, :n].numpy()], 0)
        msk = np.concatenate([rp[:1], r["masks"][0, :, :n].numpy().astype(bool)], 0)
        hdv = ~np.asarray(is_cav, bool)
        ramp = [k for k in range(n) if hdv[k] and int(tr[0, k, 1]) == 2]
        out = {"idx": idx, "error": None, "n_ramp": len(ramp),
               "wm": ramp_exposure(pos, lan, msk, ramp),
               "sumo": ramp_exposure(tr[:, :, 0], tr[:, :, 1], rp, ramp),
               "t_wm": [first_off2(lan, msk, k) for k in ramp],
               "t_rec": [first_off2(tr[:, :, 1], rp, k) for k in ramp]}
        for ts in (60, 120):
            i = int(ts / DT)
            both = msk[i] & rp[i] & hdv
            d = np.abs(pos[i] - tr[i, :, 0])
            out[f"ds{ts}"] = float(np.median(d[both])) if both.any() else float("nan")
        return out
    except Exception as exc:  # noqa: BLE001 - recorded per start
        return {"idx": idx, "error": f"{type(exc).__name__}: {exc}"}


def run_all(ckpt: str, var: dict, records: list, workers: int) -> list:
    tasks = list(enumerate(records))
    if workers <= 1:
        init(ckpt, var)
        return [work(t) for t in tasks]
    from multiprocessing import Pool
    res = []
    t0 = time.time()
    with Pool(workers, initializer=init, initargs=(ckpt, var)) as pool:
        for j, r in enumerate(pool.imap_unordered(work, tasks, chunksize=1)):
            res.append(r)
            if (j + 1) % 50 == 0:
                print(f"  {j + 1}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    return sorted(res, key=lambda r: r["idx"])


def pooled(results: list) -> dict:
    ok = [r for r in results if not r.get("error")]
    out = {"starts": len(ok), "errors": len(results) - len(ok),
           "ramp_cars": int(sum(r["n_ramp"] for r in ok))}
    for side in ("wm", "sumo"):
        risk = np.sum([r[side]["at_risk"] for r in ok], 0)
        ev = np.sum([r[side]["events"] for r in ok], 0)
        dd = np.concatenate([np.asarray(r[side]["dep_d"], float) for r in ok]) if ok else np.zeros(0)
        out[side] = {"at_risk": risk.tolist(), "events": ev.tolist(),
                     "hazard": (ev / np.maximum(risk, 1.0)).tolist(),
                     "departures": int(len(dd)),
                     "median_dep_d_m": float(np.median(dd)) if dd.size else float("nan"),
                     "share_last5m": float(np.mean(dd < FORCE_WINDOW_M)) if dd.size else float("nan"),
                     "share_gt25m": float(np.mean(dd > 25.0)) if dd.size else float("nan")}
    tw = np.concatenate([np.asarray(r["t_wm"], float) for r in ok])
    ts = np.concatenate([np.asarray(r["t_rec"], float) for r in ok])
    both = np.isfinite(tw) & np.isfinite(ts)
    out["stuck_wm"], out["stuck_sumo"] = int(np.isinf(tw).sum()), int(np.isinf(ts).sum())
    out["median_dep_err_s"] = float(np.median(tw[both] - ts[both])) if both.any() else float("nan")
    out["median_abs_dep_err_s"] = (float(np.median(np.abs(tw[both] - ts[both])))
                                   if both.any() else float("nan"))
    for h in (60, 120):
        out[f"ds{h}_median"] = float(np.nanmedian([r[f"ds{h}"] for r in ok]))
    return out


MIN_RISK = 50


def log_ratio(stats: dict) -> np.ndarray:
    """``log(h_sumo / h_wm)`` per bin with smoothed hazards ``(events + 0.5) / (at_risk + 1)``."""
    def h(side):
        return ((np.asarray(stats[side]["events"]) + 0.5)
                / (np.asarray(stats[side]["at_risk"]) + 1.0))
    return np.log(h("sumo") / h("wm"))


def fitted_bins(stats: dict, fit_window: bool) -> list[int]:
    """Bins the fit acts on: enough exposure on both sides (>= :data:`MIN_RISK` car-steps); the
    forced window only when the merge head acts there (``fit_window``)."""
    rs, rw = np.asarray(stats["sumo"]["at_risk"]), np.asarray(stats["wm"]["at_risk"])
    return [b for b in range(len(rs)) if (b > 0 or fit_window)
            and rs[b] >= MIN_RISK and rw[b] >= MIN_RISK]


MIN_SLOPE, MAX_SECANT_STEP = 0.15, 2.5


def update(offsets: np.ndarray, stats: dict, fit_window: bool, prev=None) -> np.ndarray:
    """One step on :func:`fitted_bins`.  Without ``prev``: fixed point
    ``offset_b += clip(log(h_sumo / h_wm), +-MAX_STEP)``.  With ``prev = (offsets, log_ratio)`` of
    the previous iterate: damped secant ``offset_b += clip(r_b / slope_b, +-MAX_SECANT_STEP)``,
    ``slope_b = clip((r_prev - r) / (o - o_prev), MIN_SLOPE, 1)`` (the hazard's measured response
    per logit unit; 1 when the bin did not move)."""
    new = offsets.copy()
    ratio = log_ratio(stats)
    for b in fitted_bins(stats, fit_window):
        if prev is None:
            step = float(np.clip(ratio[b], -MAX_STEP, MAX_STEP))
        else:
            d_o = float(offsets[b] - prev[0][b])
            slope = 1.0 if abs(d_o) < 0.1 else float(np.clip((prev[1][b] - ratio[b]) / d_o,
                                                             MIN_SLOPE, 1.0))
            step = float(np.clip(ratio[b] / slope, -MAX_SECANT_STEP, MAX_SECANT_STEP))
        new[b] = float(np.clip(new[b] + step, -MAX_OFFSET, MAX_OFFSET))
    return new


def write_checkpoint(src: str, out: str, gate: float | None, edges, offsets, meta: dict) -> None:
    """Copy of ``src`` carrying the calibration (payload ``lane_v5_merge_calib``) and the merge
    gate (``config.lane_v5_gates[2]``); ``physics.json`` is copied next to it."""
    import shutil

    import torch
    payload = torch.load(src, map_location="cpu", weights_only=False)
    cfg = dict(payload.get("config") or {})
    if gate is not None:
        g = list(cfg.get("lane_v5_gates", (0.25, 0.10, 0.5)))
        g[2] = float(gate)
        cfg["lane_v5_gates"] = g
    payload["config"] = cfg
    payload["lane_v5_merge_calib"] = {"edges_m": [float(x) for x in edges],
                                      "offsets": [float(x) for x in offsets], **meta}
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    phys = Path(src).parent / "physics.json"
    if phys.is_file() and not (target.parent / "physics.json").exists():
        shutil.copy2(phys, target.parent / "physics.json")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("records")
    r.add_argument("--source", action="append", required=True)
    r.add_argument("--t0", default="420,780")
    r.add_argument("--out", required=True)
    for name in ("fit", "score"):
        f = sub.add_parser(name)
        f.add_argument("checkpoint")
        f.add_argument("--records", required=True)
        f.add_argument("--out-dir", required=True)
        f.add_argument("--workers", type=int, default=10)
        f.add_argument("--gate-ap-constant", type=float, default=None,
                       help="merge gate = the constant predictor's AP (None = keep the gate)")
        f.add_argument("--calib", default=None, help="score: calibration JSON to apply")
        f.add_argument("--limit", type=int, default=None)
        if name == "fit":
            f.add_argument("--iters", type=int, default=4)
            f.add_argument("--tol", type=float, default=0.10)
            f.add_argument("--write", default=None, help="calibrated checkpoint copy")
            f.add_argument("--resume", default=None,
                           help="fit_log.json of a stopped fit to continue (secant step)")
    a = ap.parse_args(argv)
    if a.cmd == "records":
        recs = build_records(a.source, [float(x) for x in a.t0.split(",")])
        with open(guard(a.out), "wb") as fh:
            pickle.dump(recs, fh)
        print(f"wrote {len(recs)} records to {a.out}")
        return 0
    import wm_eval_common  # noqa: F401  (tools on the path)
    ckpt = guard(a.checkpoint)
    out_dir = Path(guard(a.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for path in a.records.split(","):           # comma-separated record files
        with open(guard(path), "rb") as fh:
            records.extend(pickle.load(fh))
    if a.limit:
        records = records[:a.limit]
    from wms.model.checkpoint import load_checkpoint
    model = load_checkpoint(ckpt).model
    hazard_window = getattr(model.physics, "forced_window_mode", "deterministic") == "hazard"
    nb = len(EDGES) - 1
    offsets = np.zeros(nb)
    if a.calib:
        c = json.loads(Path(a.calib).read_text(encoding="utf-8"))
        offsets = np.asarray(c["offsets"], float)
    log, prev, start = [], None, 0
    if a.cmd == "fit" and a.resume:
        # continue a stopped fit: its iterates enter the log (best-iterate choice) and the next
        # offsets follow from its last two iterates (secant)
        log = json.loads(Path(guard(a.resume)).read_text(encoding="utf-8"))
        last = log[-1]
        p2 = ((np.asarray(log[-2]["offsets"]), np.asarray(log[-2]["log_ratio"]))
              if len(log) > 1 else None)
        offsets = update(np.asarray(last["offsets"], float), last, hazard_window, p2)
        prev = (np.asarray(last["offsets"], float), np.asarray(last["log_ratio"], float))
        start = int(last["iter"]) + 1
    iters = a.iters if a.cmd == "fit" else 0
    for it in range(start, start + iters + 1):
        var = {"gate": a.gate_ap_constant,
               "calib": None if not offsets.any() else (EDGES, tuple(offsets.tolist()))}
        t_start = time.time()
        res = run_all(ckpt, var, records, a.workers)
        st = pooled(res)
        ratio = log_ratio(st)
        entry = {"iter": it, "offsets": offsets.tolist(), "seconds": round(time.time() - t_start),
                 "log_ratio": ratio.tolist(), **st}
        log.append(entry)
        print(json.dumps({k: entry[k] for k in ("iter", "offsets", "log_ratio", "stuck_wm",
                                                "stuck_sumo", "median_dep_err_s", "seconds")}),
              flush=True)
        with open(out_dir / f"iter{it}.pkl", "wb") as fh:
            pickle.dump(res, fh)
        (out_dir / "fit_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
        if a.cmd != "fit":
            break
        entry["fitted_bins"] = fitted_bins(st, hazard_window)
        if it == start + iters or all(abs(ratio[b]) < a.tol for b in entry["fitted_bins"]):
            break
        new = update(offsets, st, hazard_window, prev)
        prev = (offsets.copy(), np.asarray(ratio, float))
        offsets = new
    # the iterate whose fitted bins match SUMO best (exposure-weighted mean |log ratio|)
    def misfit(e):
        bins = fitted_bins(e, hazard_window)
        w = np.asarray(e["sumo"]["at_risk"], float)[bins]
        return float(np.sum(w * np.abs(np.asarray(e["log_ratio"])[bins])) / max(w.sum(), 1.0))
    for e in log:
        e["misfit"] = misfit(e)
    (out_dir / "fit_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    best = min(log, key=lambda e: e["misfit"])
    calib = {"edges_m": list(EDGES), "offsets": best["offsets"], "iter": best["iter"],
             "gate_ap_constant": a.gate_ap_constant, "checkpoint": ckpt,
             "records": str(a.records), "n_starts": best["starts"],
             "fit_window": hazard_window, "method": "closed-loop per-position hazard match"}
    (out_dir / "calib.json").write_text(json.dumps(calib, indent=1), encoding="utf-8")
    if a.cmd == "fit" and a.write:
        write_checkpoint(ckpt, guard(a.write), a.gate_ap_constant, EDGES, best["offsets"],
                         {k: calib[k] for k in ("iter", "gate_ap_constant", "records", "n_starts",
                                                "method")})
        print(f"wrote {a.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
