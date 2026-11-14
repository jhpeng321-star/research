"""RUN1_DIAG: quantify why WM-S run 1 matches fewer vehicles and misses merge queue.

TRAIN only, 20 states, CPU, one process, memory-mapped reads only via the
existing harness.  No SUMO/GPU.  Writes ``_out/run1_diag.json``.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WM = ROOT / "output" / "wd01" / "wm_n1"
sys.path.insert(0, str(WM))

import numpy as np  # noqa: E402

from wms_eval.data import initial_frame, load_states, recorded_for  # noqa: E402
from wms_eval.metrics import _frame_at, _present  # noqa: E402
from wms_eval.models import load_model  # noqa: E402

CKPT = ROOT / "_out" / "wms_ckpt" / "wm_s.pt"
PHYS = ROOT / "_out" / "wms_ckpt" / "physics.json"
STATES = ROOT / "_out" / "states_dev.json"
HZ = (30.0, 60.0)


def _t0_present(rec: dict) -> dict:
    init = initial_frame(rec)
    return {str(v["id"]): v for v in init["vehicles"] if v.get("present", True)}


def classify_missing(vid, rec, rec_frames, t0, modelled, H):
    """Why a recorded-present vehicle at horizon H is absent from the wms set."""
    if vid not in t0:
        return "entrant_not_in_t0"
    if vid not in modelled:
        return "t0_junction_lane_excluded"
    # in t0 modelled slots but dropped: cumulative teacher presence ended earlier.
    t0_s = float(initial_frame(rec)["time_s"])
    for frame in rec_frames:
        dt = float(frame["time_s"]) - t0_s
        if dt < 0:
            continue
        if dt > H:
            break
        row = None
        for r in frame["vehicles"]:
            if str(r["id"]) == vid:
                row = r
                break
        if row is None or not row.get("present", True):
            return "dropped_record_missing_row"
        if int(row.get("lane", -1)) < 0:
            return "dropped_junction_transit"
    return "dropped_other"


def run_state(rec, preds):
    arm = "native"
    t0 = _t0_present(rec)
    init = initial_frame(rec)
    modelled = {str(v["id"]) for v in init["vehicles"]
                if v.get("present", True) and int(v.get("lane", -1)) >= 0}
    rec_frames = recorded_for(rec, arm)
    frames = {name: fn(rec, {"arm": arm}, max(HZ), int(rec.get("seed", 0)))
              for name, fn in preds.items()}
    out = {}
    for H in HZ:
        t = float(initial_frame(rec)["time_s"]) + H
        r = _present(_frame_at(rec_frames, t))
        res = {}
        for name in preds:
            p = _present(_frame_at(frames[name], t))
            matched = set(p) & set(r)
            missing = set(r) - set(p)
            causes = {}
            for vid in missing:
                c = classify_missing(vid, rec, rec_frames, t0, modelled, H)
                if bool(r[vid].get("is_cav")):
                    c = c + "_cav"
                causes[c] = causes.get(c, 0) + 1
            lane_bad = 0
            merge_end_bad = 0
            for vid in matched:
                pr, rr = p[vid], r[vid]
                if pr.get("lane") != rr.get("lane"):
                    lane_bad += 1
                    ps, rs = float(pr.get("s", 0)), float(rr.get("s", 0))
                    if (0.0 <= rs <= 75.44 or 0.0 <= ps <= 75.44) and \
                            {int(pr.get("lane", -1)), int(rr.get("lane", -1))} <= {1, 2}:
                        merge_end_bad += 1
            res[name] = {"n_rec": len(r), "n_matched": len(matched),
                         "mf": len(matched) / len(r) if r else None,
                         "causes": causes, "lane_bad": lane_bad,
                         "merge_end_bad": merge_end_bad}
        # keep_speed vs wms differential: t0 vehicles keep_speed matches but wms drops
        ks = _present(_frame_at(frames["keep_speed"], t))
        wms = _present(_frame_at(frames["wms"], t))
        res["ks_minus_wms_matched"] = len((set(ks) & set(r)) - (set(wms) & set(r)))
        out[str(int(H))] = res
    return out


def main():
    states = load_states(STATES)
    physics = json.loads(PHYS.read_text(encoding="utf-8"))
    if isinstance(physics, dict) and isinstance(physics.get("physics"), dict):
        physics = physics["physics"]
    print("physics flags:", {k: physics.get(k) for k in
                             ("open_road", "sumo_fidelity", "sumo_idm")},
          "merge_cal" in physics, flush=True)
    preds = {
        "wms": load_model("wms", None, str(CKPT), physics)[0],
        "keep_speed": load_model("keep_speed")[0],
    }
    agg = {}
    per_state = {}
    for i, rec in enumerate(states):
        t = time.time()
        res = run_state(rec, preds)
        per_state[rec.get("world", f"s{i}")] = res
        print(f"[{i+1}/{len(states)}] {rec.get('world')} {time.time()-t:.1f}s", flush=True)
        for H in HZ:
            a = agg.setdefault(str(int(H)), {})
            for name in ("wms", "keep_speed"):
                x = res[str(int(H))][name]
                ax = a.setdefault(name, {"n_rec": 0, "n_matched": 0, "causes": {},
                                         "lane_bad": 0, "merge_end_bad": 0,
                                         "mf": []})
                ax["n_rec"] += x["n_rec"]
                ax["n_matched"] += x["n_matched"]
                ax["lane_bad"] += x["lane_bad"]
                ax["merge_end_bad"] += x["merge_end_bad"]
                if x["mf"] is not None:
                    ax["mf"].append(x["mf"])
                for c, n in x["causes"].items():
                    ax["causes"][c] = ax["causes"].get(c, 0) + n
            a["ks_minus_wms"] = a.get("ks_minus_wms", 0) + \
                res[str(int(H))]["ks_minus_wms_matched"]
    summary = {"physics": physics, "n_states": len(states), "agg": {}}
    for H in HZ:
        a = agg[str(int(H))]
        s = {}
        for name in ("wms", "keep_speed"):
            x = a[name]
            s[name] = {"n_rec": x["n_rec"], "n_matched": x["n_matched"],
                       "pooled_mf": x["n_matched"] / x["n_rec"] if x["n_rec"] else None,
                       "median_mf": float(np.median(x["mf"])) if x["mf"] else None,
                       "causes": x["causes"], "lane_bad": x["lane_bad"],
                       "merge_end_bad": x["merge_end_bad"]}
        s["ks_minus_wms_matched"] = a["ks_minus_wms"]
        summary["agg"][str(int(H))] = s
    (ROOT / "_out" / "run1_diag.json").write_text(
        json.dumps({"summary": summary, "per_state": per_state}, indent=2),
        encoding="utf-8")
    print(json.dumps(summary["agg"], indent=2))


if __name__ == "__main__":
    main()
