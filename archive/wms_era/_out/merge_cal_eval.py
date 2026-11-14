"""Merge-calibration evaluation harness (TRAIN chunk-4 only, no SUMO).

Runs the open-road A11 rollout for several physics variants and reports the
metrics the WM-S-PREP T2 brief asks for, per state and aggregated:

* merge stopped vehicle-ticks (merge segment s_down in [0, 75.44], v < 0.5)
* merge flow (ramp lane2 -> main_right lane1 transitions / min) and merge
  position distribution along the accel lane
* merge-section mean speed per 30 s
* matched fraction of the t0 recorded vehicles over time (recorded presence
  and rollout presence)
* position error |s_pred - s_true| (median/p90) at 30/60/120 s over vehicles
  present in both

Variants (all physical_exits + open_entries):
  fid       sumo_fidelity, residual on   (current open+fidelity)
  fid_phys  sumo_fidelity, residual off  (current pure physics)
  cal       sumo_fidelity + merge_cal, residual on
  cal_phys  sumo_fidelity + merge_cal, residual off

Usage:
  python merge_cal_eval.py --states w8676... --variants cal_phys --out _out/x.json
Writes partial JSON after every state (resumable).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "output" / "wd01" / "wm_n1"))
sys.path.insert(0, str(ROOT / "_out"))
import numpy as np  # noqa: E402

torch_mod = None  # lazy
CHUNK = Path(r"E:/research-dense/runs/campaign_stage1_chunk4")
DT_S = 0.5
N_STEPS = 240
HORIZONS = (5, 10, 20, 30, 60, 90, 120)
MERGE_HI = 75.44
STOP_V = 0.5
LEN_M = 5.0

FIDELITY = ["w8676m2700p20t360", "w8626m2700p02t300", "w8629m2000p02t420",
            "w8678m2700p20t300", "w8677m2700p20t360", "w8680m2600p20t300",
            "w8679m2400p20t540", "w8683m2400p20t540"]
EXTRA = ["w8617m2700p02t300", "w8618m2000p02t480", "w8619m2700p02t360",
         "w8621m2700p02t240", "w8623m2600p02t540", "w8634m2400p02t300",
         "w8635m2700p05t300", "w8638m2700p05t360", "w8640m2000p05t540",
         "w8643m2600p05t240", "w8653m2600p10t420", "w8689m2600p40t360"]

VARIANTS = {
    "fid": {"sumo_fidelity": True, "merge_cal": False, "residual": True},
    "fid_phys": {"sumo_fidelity": True, "merge_cal": False, "residual": False},
    "cal": {"sumo_fidelity": True, "merge_cal": True, "residual": True},
    "cal_phys": {"sumo_fidelity": True, "merge_cal": True, "residual": False},
}


def t0_of(world: str) -> float:
    return float(world.split("t")[-1])


def read_rows(path: Path, n: int) -> list:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
            if len(out) >= n:
                break
    return out


def rec_vehicle(row: dict, vid: str):
    from features.geometry import is_junction_lane, s_down_of

    v = (row.get("vehicles") or {}).get(vid)
    if v is None:
        return None
    lid = v.get("lane_id")
    if not lid or is_junction_lane(lid):
        return None
    try:
        lp, sd = s_down_of(lid, float(v["lane_position_m"]))
    except Exception:
        return None
    return float(sd), float(v.get("speed_mps", 0.0)), int(lp)


def _q(xs):
    if not xs:
        return None
    a = np.asarray(xs, float)
    return {"n": int(a.size), "mean": float(a.mean()),
            "p50": float(np.median(a)), "p90": float(np.percentile(a, 90))}


def _merge_events(lns, pos, ids, spd, length, msk):
    """(rate/min, positions, lead gaps, lag gaps) of lane2->1 transitions."""
    n_t = lns.shape[0]
    prev = {}
    pos_l, gl_l, gg_l = [], [], []
    for t in range(n_t):
        cur = {}
        for k in np.flatnonzero(msk[t]):
            vid = int(ids[t, k])
            cur[vid] = k
            if vid in prev:
                kp = prev[vid]
                if int(lns[t - 1, kp]) == 2 and int(lns[t, k]) == 1:
                    # target-lane neighbours one step before, shared d axis
                    s_e = float(pos[t - 1, kp])
                    lead = lag = None
                    for k2 in np.flatnonzero(msk[t - 1]):
                        if k2 == kp or int(lns[t - 1, k2]) != 1:
                            continue
                        s2 = float(pos[t - 1, k2])
                        if s2 > s_e and (lead is None or s2 < lead):
                            lead = s2
                        if s2 < s_e and (lag is None or s2 > lag):
                            lag = s2
                    pos_l.append(s_e)
                    gl_l.append((lead - LEN_M - s_e) if lead is not None else None)
                    gg_l.append((s_e - LEN_M - lag) if lag is not None else None)
        prev = cur
    return pos_l, gl_l, gg_l


def rollout_metrics(ro, rec_rows, vids, s0, lane0):
    """Metrics dict for one rollout (arm 0) and the recorded rows."""
    msk, pos, spd, lns, ids = (ro.msk[:, 0], ro.pos[:, 0], ro.spd[:, 0],
                               ro.lns[:, 0], ro.ids[:, 0])
    merge_stop = 0
    merge_v_bins = []
    for b in range(4):                       # 30 s bins
        lo, hi = b * 60, (b + 1) * 60
        vs = []
        for t in range(lo, min(hi, msk.shape[0])):
            m = msk[t] & (pos[t] >= 0.0) & (pos[t] <= MERGE_HI)
            vv = spd[t][m]
            merge_stop += int((vv < STOP_V).sum())
            vs.extend(vv.tolist())
        merge_v_bins.append(float(np.mean(vs)) if vs else 0.0)
    p_l, gl_l, gg_l = _merge_events(lns, pos, ids, spd, None, msk)
    hz = {}
    for h in HORIZONS:
        j = min(int(round(h / DT_S)), msk.shape[0] - 1)
        rec_present = roll_present = both = 0
        dpos = []
        for k in np.flatnonzero(ro.msk[0, 0]):
            vid = vids.get(int(k))
            if vid is None:
                continue
            rv = rec_vehicle(rec_rows[j - 1] if j - 1 < len(rec_rows) else rec_rows[-1], vid)
            if rv is None:
                continue
            rec_present += 1
            if bool(msk[j, k]) and int(ids[j, k]) == int(ro.ids[0, 0, k]):
                roll_present += 1
                both += 1
                dpos.append(abs(float(pos[j, k]) - rv[0]))
        hz[str(h)] = {"rec": rec_present, "roll": roll_present, "both": both,
                      "dpos": _q(dpos)}
    return {
        "merge_stopped": merge_stop,
        "n_merges": len(p_l),
        "merge_rate_veh_min": len(p_l) / (N_STEPS * DT_S / 60.0),
        "merge_pos_s_down": _q([x for x in p_l]),
        "accepted_gap_lead_m": _q([x for x in gl_l if x is not None]),
        "accepted_gap_lag_m": _q([x for x in gg_l if x is not None]),
        "merge_mean_v_30s": merge_v_bins,
        "matched": hz,
    }


def run_state(bundle, world, variants):
    from closedloop.a11_predictor import (NONE_CODE, PLANNER_ACTION_CODES,
                                          _rollout_mask, batched_rollout)
    from closedloop.live_state import DepartureHistory, live_state_from_row

    t0 = t0_of(world)
    base = CHUNK / f"{world}__base"
    man = json.load(open(base / "saved_state" / "STATE_MANIFEST.json"))
    dep = DepartureHistory(base, t0)
    st = live_state_from_row(man["physics_row_at_t0"], dep)
    rmask = _rollout_mask(st)
    cav = np.asarray(st.is_cav, bool) & rmask
    K = len(st.slot_vid)
    codes = np.full((2, K), NONE_CODE, np.int64)
    codes[0, np.flatnonzero(cav)] = PLANNER_ACTION_CODES["native"]
    codes[1, np.flatnonzero(cav)] = PLANNER_ACTION_CODES["pi-external-u"]
    vids = {int(k): st.slot_vid[int(k)] for k in np.flatnonzero(rmask)}
    kin = np.asarray(st.veh_kin, np.float64)
    s0 = kin[:, 0].copy()
    lane0 = np.where(kin[:, 1:4].sum(1) > 0, kin[:, 1:4].argmax(1), -1)
    rec = {a: read_rows(CHUNK / f"{world}__{a}" / "run" / "physics_steps.jsonl",
                        N_STEPS) for a in ("pi", "native")}
    out = {"t0": t0, "n_t0": int(rmask.sum()), "n_cav": int(cav.sum()),
           "variants": {}}
    for name in variants:
        kw = VARIANTS[name]
        t_ = time.time()
        ro = batched_rollout(bundle, st, codes, n_steps=N_STEPS, t0_s=t0,
                             physical_exits=True, open_entries=True, **kw)
        out["variants"][name] = rollout_metrics(ro, rec["pi"], vids, s0, lane0)
        out["variants"][name]["sec"] = round(time.time() - t_, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default="")
    ap.add_argument("--variants", default="cal_phys")
    ap.add_argument("--out", default=str(ROOT / "_out" / "merge_cal_eval.json"))
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    global torch_mod
    import torch
    torch_mod = torch
    torch.set_num_threads(int(args.threads))
    states = ([s for s in args.states.split(",") if s] if args.states
              else FIDELITY + EXTRA)
    variants = [v for v in args.variants.split(",") if v]
    out = Path(args.out)
    partial = {"states": {}, "meta": {"states": states, "variants": variants}}
    if out.is_file():
        partial = json.loads(out.read_text(encoding="utf-8"))
    todo = [w for w in states if w not in partial["states"]]
    if todo:
        from closedloop.fast_hdv import fast_bundle, install_fast_gap_projection

        install_fast_gap_projection()
        from closedloop.a11_predictor import load_a11_bundle

        ck = load_a11_bundle()
        b = fast_bundle(ck.bundle)
        for world in todo:
            t_ = time.time()
            partial["states"][world] = run_state(b, world, variants)
            partial["states"][world]["sec"] = round(time.time() - t_, 1)
            out.write_text(json.dumps(partial), encoding="utf-8")
            print(f"done {world} ({partial['states'][world]['sec']}s)", flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
