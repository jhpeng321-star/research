"""Part 3g aggregation: read raw per-window metrics, compute per-H
summary statistics, gate v2 and old-gate verdicts.  Standalone (no
re-run needed)."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE / "results" / "raw"
OUT = HERE / "results"


def agg_rows(rows):
    n = len(rows)
    out = dict(n_windows=n)
    for k in ("pred_vehicle_steps", "matched_vehicle_steps",
              "pred_collisions", "pred_lc", "true_lc",
              "pred_ttc3_exposure", "true_ttc3_exposure",
              "pred_hard_brake_s", "true_hard_brake_s",
              "delay_pred", "delay_true"):
        vals = [r[k] for r in rows if k in r]
        if vals:
            out[k] = round(float(np.mean(vals)), 3)
    sd = [r["speed_rmse"] for r in rows if "speed_rmse" in r]
    if sd:
        out["speed_rmse"] = round(float(np.sqrt(np.mean(
            np.square(sd)))), 4)
    ps = [r["pos_rmse"] for r in rows if "pos_rmse" in r]
    if ps:
        out["pos_rmse"] = round(float(np.sqrt(np.mean(
            np.square(ps)))), 2)
    lf = [r["lane_frac"] for r in rows if "lane_frac" in r]
    if lf:
        out["lane_frac"] = round(float(np.mean(lf)), 4)
    cp = [r["collisions_per_1000"] for r in rows if "collisions_per_1000" in r]
    if cp:
        out["collisions_per_1000"] = round(float(np.mean(cp)), 3)
    pl = [r["pred_lc"] for r in rows if "pred_lc" in r]
    if pl:
        out["pred_lc"] = round(float(np.mean(pl)), 1)
    tl = [r["true_lc"] for r in rows if "true_lc" in r]
    if tl:
        out["true_lc"] = round(float(np.mean(tl)), 1)
    for f in ("ttc3", "hb", "ttc15"):
        t = [bool(r[f"any_{f}_true"]) for r in rows
             if f"any_{f}_true" in r]
        pr = [bool(r[f"any_{f}_pred"]) for r in rows
              if f"any_{f}_pred" in r]
        if t and pr:
            tp = sum(1 for x, y in zip(t, pr) if x and y)
            out[f"any_{f}_recall"] = round(tp / sum(t), 4) if sum(t) else None
            out[f"any_{f}_precision"] = (round(tp / sum(pr), 4)
                                         if sum(pr) else None)
            out[f"any_{f}_base_rate"] = round(sum(t) / len(t), 4)
    if "delay_pred" in out and "delay_true" in out:
        out["delay_bias"] = round(out["delay_pred"] - out["delay_true"], 2)
    return out


def main():
    # read all raw files
    all_data = {}
    for f in sorted(RAW.glob("*.json")):
        parts = f.stem.rsplit("_H", 1)
        variant = parts[0]
        H = int(parts[1])
        rows = json.loads(f.read_text(encoding="utf-8"))
        all_data[(variant, H)] = rows

    # aggregate per (variant, H)
    per_vh = {}
    for (variant, H), rr in sorted(all_data.items()):
        key = f"{variant}@H{H}"
        a = agg_rows(rr)
        # additional fields
        n_vsteps = a.get("pred_vehicle_steps", 0)
        if n_vsteps > 0:
            a["collisions_per_1000"] = round(
                sum(r.get("pred_collisions", 0) for r in rr)
                / n_vsteps * 1000, 3)
        per_vh[key] = a

    (OUT / "per_variant_H.json").write_text(
        json.dumps(per_vh, indent=1, default=str), encoding="utf-8")

    # gate v2 + old gate at H=60
    gate_v2 = {}
    old_gate = {}
    for rung in ("L0", "L1", "L2", "L3", "B1_persistence",
                 "B3_idm_oracle"):
        key = f"{rung}@H60"
        if key not in per_vh:
            continue
        a = per_vh[key]
        coll = a.get("collisions_per_1000", 99)
        sr = a.get("speed_rmse", 99)
        lc_p = a.get("pred_lc", 0)
        lc_t = a.get("true_lc", 1)
        v2 = dict(
            collisions_per_1000=coll, C2_pass=coll <= 1.0,
            lc_pred=lc_p, lc_true=lc_t,
            lc_within_20pct=abs(lc_p - lc_t) / max(lc_t, 1) <= 0.20,
            speed_rmse=sr,
            speed_le_1p05xB2=sr <= 1.05 * 4.97,
            v2_pass=coll <= 1.0 and abs(lc_p - lc_t) / max(lc_t, 1) <= 0.20
                    and sr <= 1.05 * 4.97)
        v2["v2_PASS"] = v2["C2_pass"] and v2["lc_within_20pct"] and \
            v2["speed_le_1p05xB2"]
        gate_v2[rung] = v2
        old_gate[rung] = dict(
            speed_rmse=sr, speed_lt_B2=sr < 4.97,
            collisions_per_1000=coll, collisions_ok=coll <= 1.0,
            old_pass=sr < 4.97 and coll <= 1.0)

    result = dict(gate_v2=gate_v2, old_gate=old_gate, per_vh=per_vh)
    (OUT / "gate_v2.json").write_text(json.dumps(result, indent=1,
                                                  default=str),
                                      encoding="utf-8")
    print(json.dumps(dict(gate_v2=gate_v2, old_gate=old_gate),
                     indent=1, default=str))


if __name__ == "__main__":
    main()
