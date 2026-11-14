"""Q3: what is special about R4's blow-up states/vehicles on the real-history quick-20?

Identifies the blow-up states (HDV median |ds| > 50 m @60 s AND > 100 m @120 s) from the
reproduced base rollouts, reruns R4 (and C1 for contrast) on the blow-up states plus the
three easiest states, and dissects per-vehicle errors: lane correctness, initial speed,
ramp/mainline, predicted lane-change counts, embedding z-scores (from q2), error location
on the road. Writes out/q3_blowups.json (+ pickled arrays under out/q3/).
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_lib2 import (DIAG, OUT, R4_SCORE, SETS, dump_json, ensure_env, load_json,
                       make_predictor, q, quick20, rollout_arrays, true_lane_changes)

ensure_env()
import numpy as np  # noqa: E402

BLOW_RULE = (50.0, 100.0)


def per_vehicle(arrays, metrics):
    ids = arrays["ids"]
    n = len(ids)
    out = []
    for j, vid in enumerate(ids):
        row = {"id": vid, "ramp": vid.startswith("ramp")}
        p60 = arrays["p_pres"][120, j] & arrays["r_pres"][120, j] & ~arrays["r_cav"][120, j]
        p120 = arrays["p_pres"][240, j] & arrays["r_pres"][240, j] & ~arrays["r_cav"][240, j]
        row["cav"] = bool(arrays["r_cav"][0, j] or arrays["p_cav"][0, j])
        row["v0"] = float(arrays["r_v"][0, j]) if arrays["r_pres"][0, j] else None
        row["s0"] = float(arrays["r_s"][0, j]) if arrays["r_pres"][0, j] else None
        row["lane0"] = int(arrays["r_lane"][0, j]) if arrays["r_pres"][0, j] else None
        if p60:
            row["ds60"] = float(abs(arrays["p_s"][120, j] - arrays["r_s"][120, j]))
            row["wrong_lane60"] = bool(arrays["r_lane"][120, j] >= 0 and
                                       arrays["p_lane"][120, j] != arrays["r_lane"][120, j])
        if p120:
            row["ds120"] = float(abs(arrays["p_s"][240, j] - arrays["r_s"][240, j]))
            row["wrong_lane120"] = bool(arrays["r_lane"][240, j] >= 0 and
                                        arrays["p_lane"][240, j] != arrays["r_lane"][240, j])
        out.append(row)
    return out


def main() -> None:
    t0 = time.time()
    base = load_json(OUT / "q4_interventions.json")
    c1rows = load_json(R4_SCORE / "cache" / "C1__wall_same_real.json")["states"]

    def cache_val(row, h):
        b = ((row.get("arms") or {}).get("wall") or {})
        if b.get("error"):
            return None
        x = b["car"][h]["hdv"]["ds_med"]
        return float(x) if x is not None else None

    states = []
    for row, c1row in zip(base["per_state"]["base"], c1rows):
        states.append({"world": row["world"], "t0_s": row["t0_s"],
                       "R4_60": row["ds60"], "R4_120": row["ds120"],
                       "C1_60": cache_val(c1row, "60.0"), "C1_120": cache_val(c1row, "120.0")})
    valid = [s for s in states if s["R4_60"] is not None and s["R4_120"] is not None]
    blow = [s for s in valid if s["R4_60"] > BLOW_RULE[0] and s["R4_120"] > BLOW_RULE[1]]
    ok = sorted([s for s in valid if s not in blow], key=lambda s: s["R4_60"])[:3]

    records = {r["world"]: r for r in quick20(SETS["real_q20"])}
    slots = load_json(OUT / "q2_slots.json")["real_q20"]
    slot_by_world = {r["world"]: r for r in slots}
    q2 = load_json(OUT / "q2_encoder.json")
    emb_mu = np.asarray(q2["train"]["embedding"]["mean"])
    emb_sd = np.maximum(np.asarray(q2["train"]["embedding"]["std"]), 1e-6)

    (OUT / "q3").mkdir(parents=True, exist_ok=True)
    r4 = make_predictor("R4")
    c1 = make_predictor("C1")
    detail = {}
    for s in blow + ok:
        world = s["world"]
        record = records[world]
        metrics, arrays = rollout_arrays(r4, record)
        if s in blow:
            c1_metrics, c1_arrays = rollout_arrays(c1, record)
        else:
            c1_metrics, c1_arrays = None, None
        pv = per_vehicle(arrays, metrics)
        sl = slot_by_world[world]
        zmap = {}
        for vid, emb in zip(sl["ids"], sl["embedding"]):
            zmap[vid] = float(np.abs((np.asarray(emb) - emb_mu) / emb_sd).mean())
        for row in pv:
            row["emb_mean_abs_z"] = zmap.get(row["id"])
        hdv = [r for r in pv if not r["cav"] and "ds60" in r]
        worst = sorted(hdv, key=lambda r: -r["ds60"])[:10]
        pred_lc = true_lane_changes(arrays["p_lane"], arrays["p_pres"], arrays["p_cav"],
                                    arrays["ids"])
        rec_lc = true_lane_changes(arrays["r_lane"], arrays["r_pres"], arrays["r_cav"],
                                   arrays["ids"])
        detail[world] = {
            "state": s, "n_hdv_matched_60": len(hdv),
            "ds60_hdvs": q([r["ds60"] for r in hdv], (50, 75, 90, 95, 99)),
            "wrong_lane_share_60": float(np.mean([r.get("wrong_lane60", False) for r in hdv])),
            "wrong_lane_share_120": float(np.mean([r.get("wrong_lane120", False) for r in hdv
                                                   if "ds120" in r])),
            "pred_lane_changes": pred_lc, "rec_lane_changes": rec_lc,
            "worst10": worst,
            "emb_mean_abs_z_worst": q([r["emb_mean_abs_z"] for r in worst
                                       if r["emb_mean_abs_z"] is not None]),
            "emb_mean_abs_z_all": q([r["emb_mean_abs_z"] for r in hdv
                                     if r["emb_mean_abs_z"] is not None]),
            "share_error_from_top10_60": float(
                sum(r["ds60"] for r in worst) / max(1e-9, sum(r["ds60"] for r in hdv))),
            "c1": None if c1_metrics is None else {
                "ds60_med": c1_metrics["60.0"]["hdv"]["ds_med"],
                "ds120_med": c1_metrics["120.0"]["hdv"]["ds_med"],
                "wrong_lane_share_60": None},
        }
        if c1_arrays is not None:
            c1pv = per_vehicle(c1_arrays, c1_metrics)
            c1h = [r for r in c1pv if not r["cav"] and "ds60" in r]
            detail[world]["c1"]["wrong_lane_share_60"] = float(
                np.mean([r.get("wrong_lane60", False) for r in c1h]))
            detail[world]["c1"]["ds60_hdvs"] = q([r["ds60"] for r in c1h], (50, 75, 90, 95))
        with (OUT / "q3" / f"{world}.pkl").open("wb") as fh:
            pickle.dump({"arrays": {k: v for k, v in arrays.items()}, "pv": pv,
                         "c1_arrays": None if c1_arrays is None else c1_arrays}, fh,
                        protocol=4)
        print(f"[q3] {world} R4_60 {s['R4_60']:.1f} done ({time.time() - t0:.0f}s)",
              flush=True)

    out = {"blow_rule": "HDV median |ds|>50 @60s and >100 @120s",
           "blowup_states": [s["world"] for s in blow],
           "easiest_states": [s["world"] for s in ok],
           "all_states": states, "detail": detail}
    dump_json(out, OUT / "q3_blowups.json")
    print(f"q3 done in {time.time() - t0:.0f}s; blow-ups: {[s['world'] for s in blow]}",
          flush=True)


if __name__ == "__main__":
    main()
