"""Diagnose the C4 WM-S 72-state blow-ups (inference only, CPU, no SUMO).

The evaluation semantics are those of ``tools/v8wm3_check.py``: the recorded ``wall``
arm of ``states_h60/wall_same.json`` every second window (72 states), CAVs
teacher-forced, entries off, recorded SUMO exits, 120 s, lane seed 0.  A state is a
"blow-up" when the median HDV ``|ds|`` is > 50 m at 60 s and > 100 m at 120 s.

The script has two stages:

``roll``
    Roll one or more checkpoints over the blow-up states plus well-behaved controls and
    store the per-step trajectories and the per-step acceleration decomposition
    (physics/IDM, raw learned residual, gate, lane logits, driver settings).  Results
    are ``{model}_{state}.npz`` under ``--out``.

``report``
    Read the ``.npz`` files and emit a per-state evidence table (JSON, and a Markdown
    table on stdout).  Detection rules are deliberately mechanical: junction transits
    (recorded lane < 0) are excluded from every positional metric because SUMO carries
    the last on-road position through them.

Usage (from the c4/v10blowup worktree)::

    PYTHONPATH="src;.;tools" python tools/wm_blowup_diag.py roll --workers 4
    PYTHONPATH="src;.;tools" python tools/wm_blowup_diag.py report

No GPU, no SUMO, no training.  Restricted-split paths are refused by
``wm_eval_common.guard``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, "tools")
import wm_eval_common as wc  # noqa: E402

DT = 0.5
SET = "E:/agent-work/c4-v6merge/_ref/states_h60/wall_same.json"
OUT = ("E:/research-dense/runs/active/C4-V7FT-20261017/blowup_diag/rolls")
MODELS = {
    "v5m": "E:/research-dense/runs/active/C4-V7FT-20261017/v5m_ref/wm_s_best_ap.pt",
    "round2": "E:/Temp/dc_laptop/ck_round2/wm_s_stage240.pt",
}
#: States whose 60 s / 120 s medians cross the blow-up definition in the score72 runs.
BLOWUP = (16, 18, 24, 74, 90, 96, 98, 100, 102, 106)
#: Well-behaved controls: 40 and 82 are the lowest-error states in score72; 114/132
#: are in the low-error quartile; 50 is an extra low-error state.
CONTROLS = (40, 82, 114, 132, 50)
STATES = BLOWUP + CONTROLS
HORIZON_S = 120.0
MERGE_END_M = 75.44


# --------------------------------------------------------------------------- rolling
def _recs(states):
    """Stream the state file once and keep only the requested windows."""
    from wms.eval.normalize import iter_windows
    want = set(states)
    out = {}
    for i, rec in enumerate(iter_windows(wc.guard(SET))):
        if i in want:
            out[i] = rec
        if len(out) == len(want):
            break
    return out


def _attach(model):
    """Capture the forward tensors that ``rollout`` does not return."""
    orig = model.forward
    cap = {"steps": []}

    def fwd(state, **kw):
        out = orig(state, **kw)
        step = {
            "physics": out["physics_accel_mps2"].detach().float().cpu().numpy()[0],
            "residual": out.get(
                "raw_residual_mps2",
                out["accel_mps2"] - out["physics_accel_mps2"],
            ).detach().float().cpu().numpy()[0],
            "accel": out["accel_mps2"].detach().float().cpu().numpy()[0],
            "gate": (out.get("gate") if out.get("gate") is not None
                     else out["accel_mps2"] * 0.0).detach().float().cpu().numpy()[0],
            "lane_logits": out["lane_logits"].detach().float().cpu().numpy()[0],
        }
        if "driver_parameters" in out:
            step["driver"] = out["driver_parameters"].detach().float().cpu().numpy()[0]
        cap["steps"].append(step)
        return out

    model.forward = fwd
    return orig, cap


def _stack(steps, key, shape, n_rows, fill=np.nan):
    a = np.full((n_rows,) + shape, fill, dtype=np.float32)
    for i, step in enumerate(steps):
        if key in step:
            a[i] = step[key]
    return a


def roll_one(name, idx, rec, out_dir, predictor=None, capture=None, cap=None):
    from wms.eval.rollout_inputs import recorded_teacher, t0_vehicles
    from wms.model.driver_encoder import PARAMETER_NAMES
    if predictor is None:
        predictor, capture = wc.load_predictor(wc.guard(MODELS[name]), "AS_IS",
                                               hdv_exits="sumo", entries="off")
    if cap is None:
        _, cap = _attach(predictor.model)
    cap["steps"].clear()
    out = wc.roll(predictor, capture, rec, "wall", horizon_s=HORIZON_S)
    n = len(out["ids"])
    lanes = capture["last"]["lanes"][0, :, :n].numpy()
    t0, ids, kin, _ = t0_vehicles(rec)
    teacher, _, _ = recorded_teacher(rec, t0, ids, kin, int(HORIZON_S / DT), "wall",
                                     predictor.backend.tools)
    tr = teacher[0].numpy()
    steps = cap["steps"]
    rows = len(steps)
    phys = _stack(steps, "physics", (n,), rows)
    resid = _stack(steps, "residual", (n,), rows)
    accel = _stack(steps, "accel", (n,), rows)
    gate = _stack(steps, "gate", (n,), rows)
    logits = _stack(steps, "lane_logits", (n, 3), rows)
    driver = _stack(steps, "driver", (n, len(PARAMETER_NAMES)), rows)
    np.savez_compressed(
        os.path.join(out_dir, f"{name}_{idx}.npz"),
        wm_pos=out["wm_pos"], wm_vel=out["wm_vel"], wm_msk=out["wm_msk"],
        rec_pos=out["rec_pos"], rec_vel=out["rec_vel"], rec_msk=out["rec_msk"],
        rec_lane=tr[:, :, 1], rec_a=tr[:, :, 3],
        lanes=lanes, ids=np.array(out["ids"]), is_cav=out["is_cav"],
        physics=phys, residual=resid, accel=accel, gate=gate,
        lane_logits=logits, driver=driver,
        driver_names=np.array(PARAMETER_NAMES),
    )


def _roll_chunk(task):
    name, states = task
    recs = _recs(states)
    predictor, capture = wc.load_predictor(wc.guard(MODELS[name]), "AS_IS",
                                           hdv_exits="sumo", entries="off")
    _, cap = _attach(predictor.model)
    for idx in states:
        roll_one(name, idx, recs[idx], OUT, predictor, capture, cap)
        print(f"rolled {name} {idx}", flush=True)
    return task


def cmd_roll(args):
    os.makedirs(OUT, exist_ok=True)
    states = list(STATES)
    if args.states:
        states = [int(x) for x in args.states.split(",")]
    if args.models:
        names = [x for x in args.models.split(",") if x]
    else:
        names = list(MODELS)
    tasks = []
    for name in names:
        for c in range(min(4, len(states))):
            tasks.append((name, states[c::min(4, len(states))]))
    if args.workers <= 1:
        for task in tasks:
            _roll_chunk(task)
    else:
        from multiprocessing import Pool
        with Pool(min(args.workers, len(tasks))) as pool:
            for task in pool.imap_unordered(_roll_chunk, tasks):
                print(f"chunk done {task}", flush=True)


# --------------------------------------------------------------------------- report
def _load(out_dir, model, idx):
    path = os.path.join(out_dir, f"{model}_{idx}.npz")
    d = dict(np.load(path))
    if "rec_lane" not in d:                       # older roll files
        r = dict(np.load(os.path.join(out_dir, f"rec_{idx}.npz")))
        d.update(r)
    return d


def _lanes_at_rows(d):
    """Predicted lane per row; row 0 is the recorded t0 lane (they agree at t0)."""
    return np.concatenate([d["rec_lane"][:1], d["lanes"]], 0)


def _valid_rows(d):
    both = d["wm_msk"].astype(bool) & d["rec_msk"].astype(bool)
    junc = d["junction"].astype(bool) if "junction" in d else np.zeros_like(both)
    ok = both & ~junc
    ok[1:] &= ~junc[:-1]
    ok[:-1] &= ~junc[1:]
    return ok


def _first(ok, ad, k, thr):
    t = ad.shape[0]
    for i in range(1, t - 1):
        if ok[i, k] and ad[i, k] >= thr and (ad[i + 1, k] >= thr or not ok[i + 1, k]):
            return i
    return None


def _detail(d, lane, ids, cav, k, i):
    j = None
    for q in np.flatnonzero(d["wm_msk"][i] & (lane[i] == lane[i, k])):
        if q != k and d["wm_pos"][i, q] > d["wm_pos"][i, k]:
            if j is None or d["wm_pos"][i, q] < d["wm_pos"][i, j]:
                j = q
    row = {
        "id": str(ids[k]), "ramp": str(ids[k]).startswith("ramp"), "cav": bool(cav[k]),
        "lane": int(lane[i, k]), "rec_lane": int(d["rec_lane"][i, k]),
        "s_wm": float(d["wm_pos"][i, k]), "s_rec": float(d["rec_pos"][i, k]),
        "v_wm": float(d["wm_vel"][i, k]), "v_rec": float(d["rec_vel"][i, k]),
        "accel": float(d["accel"][i - 1, k]),
        "idm": float(d["physics"][i - 1, k]),
        "residual": float(d["residual"][i - 1, k]),
        "gate": float(d["gate"][i - 1, k]),
    }
    row["driver"] = {str(n): float(d["driver"][0, k, q])
                     for q, n in enumerate(d["driver_names"])}
    if j is not None:
        row["leader"] = {
            "id": str(ids[j]), "cav": bool(cav[j]), "s_wm": float(d["wm_pos"][i, j]),
            "v_wm": float(d["wm_vel"][i, j]),
            "gap": float(d["wm_pos"][i, j] - d["wm_pos"][i, k] - 5.0),
            "v_rec": float(d["rec_vel"][i, j]),
        }
    return row


def _merge_event(d, lanes, ids, cav, idx_state):
    """Earliest ramp HDV lane-2 -> lane-1 move that SUMO does not make for >= 3 s."""
    t = d["wm_pos"].shape[0]
    best = None
    for k in range(len(ids)):
        if cav[k] or not str(ids[k]).startswith("ramp") or d["rec_lane"][0, k] != 2:
            continue
        for i in range(1, t - 3):
            if (lanes[i, k] != 2 and d["rec_lane"][i, k] == 2
                    and d["rec_lane"][i + 1, k] == 2 and d["rec_lane"][i + 2, k] == 2):
                lg = d["lane_logits"][max(i - 1, 0), k].astype(float)
                z = np.exp(lg - lg.max())
                cand = {"t": i * DT, "id": str(ids[k]),
                        "s": float(d["wm_pos"][i, k]),
                        "v_wm": float(d["wm_vel"][i, k]),
                        "v_rec": float(d["rec_vel"][i, k]),
                        "p_lane1": float((z / z.sum())[1])}
                if best is None or cand["t"] < best["t"]:
                    best = cand
                break
    return best


def _residual_spike(d, lanes, ids, cav):
    """Earliest just-merged ramp HDV held at low speed on lane 1 by a negative residual."""
    t = d["wm_pos"].shape[0]
    best = None
    for k in range(len(ids)):
        if cav[k] or not str(ids[k]).startswith("ramp") or d["rec_lane"][0, k] != 2:
            continue
        for i in range(2, t - 1):
            if (lanes[i, k] == 1 and 70 <= d["wm_pos"][i, k] <= 220
                    and d["wm_vel"][i, k] < 8 and d["rec_vel"][i, k] > 8
                    and d["residual"][i - 1, k] < -3 and d["residual"][i, k] < -3):
                cand = {"t": i * DT, "id": str(ids[k]),
                        "s": float(d["wm_pos"][i, k]),
                        "v_wm": float(d["wm_vel"][i, k]),
                        "v_rec": float(d["rec_vel"][i, k]),
                        "residual": [float(d["residual"][i - 1, k]),
                                     float(d["residual"][i, k])],
                        "idm": float(d["physics"][i - 1, k])}
                if best is None or cand["t"] < best["t"]:
                    best = cand
                break
    return best


def _cluster(d, ok, lo, hi, wm_cond, rec_cond, lanes):
    t = d["wm_pos"].shape[0]
    for i in range(1, t):
        sel = (ok[i] & (lanes[i] == 1) & (d["wm_pos"][i] >= lo) & (d["wm_pos"][i] <= hi)
               & wm_cond(d["wm_vel"][i]) & rec_cond(d["rec_vel"][i]))
        if sel.sum() >= 2:
            ids = d["_ids"]
            return {"t": i * DT, "n": int(sel.sum()),
                    "ids": [str(ids[k]) for k in np.flatnonzero(sel)][:6]}
    return None


def _stuck_ramp(d, lanes, ids, cav):
    """Earliest ramp HDV held at the lane end (lane 2, s >= 70, v < 0.5 m/s) while SUMO
    has merged it or is moving: the signature of the merge-end wait being too strong."""
    t = d["wm_pos"].shape[0]
    best = None
    for k in range(len(ids)):
        if cav[k] or not str(ids[k]).startswith("ramp"):
            continue
        for i in range(1, t - 4):
            if not (lanes[i, k] == 2 and d["wm_pos"][i, k] >= 70
                    and d["wm_vel"][i, k] < 0.5):
                continue
            for q in range(1, 4):
                if not (lanes[i + q, k] == 2 and d["wm_vel"][i + q, k] < 0.5):
                    break
            else:
                if d["rec_lane"][i, k] != 2 or d["rec_vel"][i, k] > 1.5:
                    cand = {"t": i * DT, "id": str(ids[k]),
                            "s": float(d["wm_pos"][i, k]),
                            "v_wm": float(d["wm_vel"][i, k]),
                            "rec_lane": int(d["rec_lane"][i, k]),
                            "v_rec": float(d["rec_vel"][i, k])}
                    if best is None or cand["t"] < best["t"]:
                        best = cand
                break
    return best


def _first_lane_mismatch(d, lanes, ids, cav, ok):
    """First HDV held in a different lane than SUMO for two consecutive valid rows."""
    best = None
    for k in range(len(ids)):
        if cav[k]:
            continue
        for i in range(1, d["wm_pos"].shape[0] - 1):
            if (ok[i, k] and ok[i + 1, k] and lanes[i, k] != d["rec_lane"][i, k]
                    and lanes[i + 1, k] != d["rec_lane"][i + 1, k]):
                cand = {"t": i * DT, "id": str(ids[k]),
                        "lane": int(lanes[i, k]),
                        "rec_lane": int(d["rec_lane"][i, k]),
                        "s": float(d["wm_pos"][i, k]),
                        "v_wm": float(d["wm_vel"][i, k]),
                        "v_rec": float(d["rec_vel"][i, k]),
                        "ramp": str(ids[k]).startswith("ramp")}
                if best is None or cand["t"] < best["t"]:
                    best = cand
                break
    return best


def _lane_mismatch_at(d, lanes, cav, ok, ts):
    i = int(ts / DT)
    sel = ok[i] & ~cav
    mis = sel & (lanes[i] != d["rec_lane"][i])
    ds = np.abs(d["wm_pos"][i] - d["rec_pos"][i])
    return {"n": int(sel.sum()), "mismatch": int(mis.sum()),
            "ds_matched": float(np.median(ds[sel & ~mis])) if (sel & ~mis).any() else None,
            "ds_mismatch": float(np.median(ds[mis])) if mis.any() else None}


def _merge_stats(d, lanes, ids, cav):
    t = d["wm_pos"].shape[0]
    wm, rec = [], []
    for k in range(len(ids)):
        if cav[k] or not str(ids[k]).startswith("ramp") or d["rec_lane"][0, k] != 2:
            continue
        tw = next((i for i in range(1, t) if lanes[i, k] != 2), None)
        tr = next((i for i in range(1, t) if d["rec_lane"][i, k] != 2), None)
        if tw is not None:
            wm.append((tw * DT, float(d["wm_vel"][tw, k])))
        if tr is not None:
            rec.append((tr * DT, float(d["rec_vel"][tr, k])))
    def one(x):
        if not x:
            return None
        a = np.asarray(x)
        return {"n": len(x), "t_med": float(np.median(a[:, 0])),
                "v_med": float(np.median(a[:, 1]))}
    return {"wm": one(wm), "rec": one(rec)}


def summarize(out_dir, model, idx, window=None):
    d = _load(out_dir, model, idx)
    ids = [str(x) for x in d["ids"]]
    d["_ids"] = ids
    cav = d["is_cav"].astype(bool)
    lanes = _lanes_at_rows(d)
    ok = _valid_rows(d)
    ad = np.abs(d["wm_pos"] - d["rec_pos"])
    both = d["wm_msk"].astype(bool) & d["rec_msk"].astype(bool)
    # The blow-up definition itself: both-present HDVs, no junction filtering, median |ds|.
    ds = {}
    for ts in (30, 60, 120):
        i = int(ts / DT)
        sel = both[i] & ~cav
        ds[str(ts)] = float(np.median(ad[i][sel])) if sel.any() else None
    blow = (ds["60"] is not None and ds["60"] > 50
            and ds["120"] is not None and ds["120"] > 100)
    out = {"model": model, "state": idx, "blowup": bool(blow),
           "t0_s": float(d.get("t0_s", 0.0)) if "t0_s" in d else None,
           "n_hdv": int((~cav).sum()), "n_cav": int(cav.sum()),
           "ds": ds, "n_both": {}, "first50": None,
           "first_lane_mismatch": _first_lane_mismatch(d, lanes, ids, cav, ok),
           "lane_mismatch_at60": _lane_mismatch_at(d, lanes, cav, ok, 60.0),
           "first_merge": _merge_event(d, lanes, ids, cav, idx),
           "residual_spike": _residual_spike(d, lanes, ids, cav),
           "stuck_ramp": _stuck_ramp(d, lanes, ids, cav),
           "phantom_cluster": _cluster(d, ok, -20, 160,
                                       lambda v: v < 5, lambda v: v > 15, lanes),
           "missed_cluster": _cluster(d, ok, -20, 160,
                                      lambda v: v > 10, lambda v: v < 2, lanes),
           "merges": _merge_stats(d, lanes, ids, cav)}
    if window is not None:
        tl = window.get("wall_timeline") or {}
        members = [str(x) for x in (tl.get("members") or [])]
        out["wall"] = {
            "members": members,
            "missing_from_t0": [m for m in members if m not in set(ids)],
            "request_rel_s": tl.get("request_rel_s"),
            "release_rel_s": tl.get("released_rel_s"),
            "speed_mps": tl.get("wall_speed_mps"),
            "outcome": tl.get("outcome"),
        }
    else:
        out["wall"] = None
    for ts in (30, 60, 120):
        i = int(ts / DT)
        sel = both[i] & ~cav
        out["n_both"][str(ts)] = int(sel.sum())
    rows = []
    for k in range(len(ids)):
        if cav[k]:
            continue
        i = _first(ok, ad, k, 50.0)
        if i is not None:
            rows.append((i, k))
    rows.sort()
    if rows:
        i, k = rows[0]
        out["first50"] = _detail(d, lanes, ids, cav, k, i)
        out["first50"]["t"] = i * DT
        out["first50"]["next"] = [(float(j * DT), str(ids[q])) for j, q in rows[1:4]]
    return out


def cmd_report(args):
    out_dir = args.out
    models = [x for x in args.models.split(",") if x] if args.models else list(MODELS)
    states = [int(x) for x in args.states.split(",")] if args.states else list(STATES)
    windows = _recs(states)
    rows = []
    for model in models:
        for idx in states:
            path = os.path.join(out_dir, f"{model}_{idx}.npz")
            if os.path.exists(path):
                rows.append(summarize(out_dir, model, idx, windows.get(idx)))
    payload = {"set": SET, "out": out_dir, "states": rows}
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)

    print("| model | state | blow | ds30 | ds60 | ds120 | n120 | wall | first50 | first lane mismatch | merge (t,id,v) | residual spike | stuck ramp | phantom | missed |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        f = r["first50"] or {}
        lm = r["first_lane_mismatch"] or {}
        m = r["first_merge"] or {}
        s = r["residual_spike"] or {}
        st = r["stuck_ramp"] or {}
        p = r["phantom_cluster"] or {}
        mi = r["missed_cluster"] or {}
        w = r["wall"] or {}
        mv = m.get("v_wm")
        rv = (s.get("residual") or [None])[0]
        spd = w.get("speed_mps")
        wall = "n/a" if not w else (
            f"miss{len(w['missing_from_t0'])}/{len(w['members'])} "
            f"v{'' if spd is None else f'{spd:.1f}'}")
        print(f"| {r['model']} | {r['state']} | {'Y' if r['blowup'] else 'n'} "
              f"| {_f(r['ds']['30'])} | {_f(r['ds']['60'])} | {_f(r['ds']['120'])} "
              f"| {r['n_both']['120']} | {wall} "
              f"| {_f(f.get('t'))} {f.get('id', '')} "
              f"| {_f(lm.get('t'))} {lm.get('id', '')} L{lm.get('lane', '')}/{lm.get('rec_lane', '')} "
              f"| {_f(m.get('t'))} {m.get('id', '')} {'' if mv is None else f'v{mv:.1f}'} "
              f"| {_f(s.get('t'))} {s.get('id', '')} {'' if rv is None else f'r{rv:.1f}'} "
              f"| {_f(st.get('t'))} {st.get('id', '')} "
              f"| {_f(p.get('t'))} n{p.get('n', 0)} "
              f"| {_f(mi.get('t'))} n{mi.get('n', 0)} |")


def _f(x):
    return "" if x is None else f"{x:.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("roll")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--states", default="")
    r.add_argument("--models", default="")
    r.set_defaults(func=cmd_roll)
    p = sub.add_parser("report")
    p.add_argument("--out", default=OUT)
    p.add_argument("--json", default=os.path.join(OUT, "blowup_summary.json"))
    p.add_argument("--states", default="")
    p.add_argument("--models", default="")
    p.set_defaults(func=cmd_report)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
