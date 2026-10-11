"""Aggregate out/tasks/*.pkl into results.json (DIAG-LANE)."""
from __future__ import annotations

import json
import os
import pickle
import statistics as st
from pathlib import Path

import numpy as np

RUN = Path("E:/research-dense/runs/active/WM-S-R4-DIAG-LANE-20261011")
TASKS = RUN / "out" / "tasks"

rows = []
for p in sorted(TASKS.glob("*.pkl")):
    with p.open("rb") as fh:
        rows.append(pickle.load(fh))
assert not any(r["error"] for r in rows), [r["world"] for r in rows if r["error"]]
by = {}
for r in rows:
    by.setdefault((r["model"], r["variant"]), []).append(r)
for k in by:
    by[k].sort(key=lambda r: (r["world"], r["t0_s"], str(r["key"])))


def rate_summary(items, key):
    ch = sum(r["result"]["rates"][key]["pred_changes"] for r in items)
    n = sum(r["result"]["rates"][key]["pred_steps"] for r in items)
    rch = sum(r["result"]["rates"][key]["rec_changes"] for r in items)
    rn = sum(r["result"]["rates"][key]["rec_steps"] for r in items)
    return {"pred_per_hdv_per_10s": ch / (n / 20.0) if n else None,
            "rec_per_hdv_per_10s": rch / (rn / 20.0) if rn else None,
            "pred_changes": ch, "pred_steps": n}


def variant_summary(model, variant):
    items = by[(model, variant)]
    out = {"n_states": len(items), "rates": {}, "wrong_lane": {}, "ds_med": {}}
    for key in ("0-60", "60-120"):
        out["rates"][key] = rate_summary(items, key)
    for h in ("60", "120"):
        a = sum(r["result"]["wrong_lane"][h][0] for r in items)
        b = sum(r["result"]["wrong_lane"][h][1] for r in items)
        out["wrong_lane"][h] = {"share": a / b if b else None, "wrong": a, "n": b}
    for h in ("60", "120"):
        vals = [r["result"]["metrics"][f"ds_med_{h}"] for r in items
                if r["result"]["metrics"][f"ds_med_{h}"] is not None]
        out["ds_med"][h] = {"median": st.median(vals) if vals else None,
                            "mean": float(np.mean(vals)) if vals else None,
                            "p90": float(np.percentile(vals, 90)) if vals else None, "n": len(vals)}
    blows = [(r["result"]["metrics"]["ds_med_60"], r["result"]["metrics"]["ds_med_120"]) for r in items]
    out["blowups"] = sum(1 for a, b in blows if a is not None and b is not None and a > 50 and b > 100)
    tr = [r["result"]["trace"] for r in items]
    dec = [d for t in tr for d in t["decisions"]]
    out["trace"] = trace_stats(tr, dec)
    return out


def trace_stats(tr, dec):
    mobil_would = [d for d in dec if d["mobil_change"]]
    delta_driven = [d for d in dec if not d["mobil_change"]]
    jumps = [d for d in dec if d.get("jump")]

    def q(vals, f):
        vals = [f(d) for d in vals]
        return {"p50": float(np.median(vals)) if vals else None,
                "mean": float(np.mean(vals)) if vals else None,
                "p90": float(np.percentile(vals, 90)) if vals else None, "n": len(vals)}

    early_steps = sum(t["hdv_steps"]["0-60"] for t in tr)
    late_steps = sum(t["hdv_steps"]["60-120"] for t in tr)
    hist = np.sum([np.asarray(t["time_hist_5s"]) for t in tr], axis=0).tolist() if tr else []
    lane = np.sum([np.asarray(t["lane_hist"]) for t in tr], axis=0).tolist() if tr else []
    zone = {}
    cav = {}
    for t in tr:
        for k, v in t["zone_hist"].items():
            zone[k] = zone.get(k, 0) + v
        for k, v in t["cav_dist"].items():
            cav[k] = cav.get(k, 0) + v
    n = len(dec)
    return {
        "early_decisions": n,
        "mobil_would_also_change": len(mobil_would),
        "delta_driven": len(delta_driven),
        "two_lane_jumps": len(jumps),
        "mobil_margin_at_delta_driven": q(delta_driven, lambda d: d["mobil_margin"]),
        "corr_adv_at_delta_driven": q(delta_driven, lambda d: d["corr_adv"]),
        "corr_adv_at_mobil_steps": q(mobil_would, lambda d: d["corr_adv"]),
        "corr_abs_max_all_decisions": q(dec, lambda d: d["corr_abs_max"]),
        "gate_at_decisions": q(dec, lambda d: d["gate"]),
        "k_at_decisions": q(dec, lambda d: d["k"]),
        "mobil_only_flip_rate_0_60_per_step": (sum(t["mobil_flips"]["0-60"] for t in tr) / early_steps) if early_steps else None,
        "mobil_only_flip_rate_60_120_per_step": (sum(t["mobil_flips"]["60-120"] for t in tr) / late_steps) if late_steps else None,
        "mobil_only_flip_rate_0_60_per_10s": (sum(t["mobil_flips"]["0-60"] for t in tr) / early_steps * 20) if early_steps else None,
        "mobil_only_flip_rate_60_120_per_10s": (sum(t["mobil_flips"]["60-120"] for t in tr) / late_steps * 20) if late_steps else None,
        "time_hist_5s": hist, "lane_hist_cur_lane": lane, "zone_hist": zone, "cav_dist": cav,
        "junction_self_within_5s": sum(t["junction_self"] for t in tr),
        "junction_near_within_60m": sum(t["junction_near"] for t in tr),
        "near_recorded_change_within_5s": sum(t["near_recorded_change"] for t in tr),
        "exit_within_10s": sum(t["exit_soon"] for t in tr),
    }


res = {"schema": "wm_s_r4_diag_lane/1",
       "setup": {"set": "wall_same h60 quick-20 (first state per world, 20 worlds)",
                 "horizon_s": 120.0, "entries": False, "cav": "teacher (recorded wall arm)",
                 "models": {"C1": "data_cache/ckpt_c1fast/wm_s.pt",
                            "R4": "runs/active/WM-S-R4-20261011/box/gpuA/r4_main_a1/wm_s.pt"},
                 "variants": {"base": "unchanged", "delta0": "lane correction zeroed (MOBIL only)",
                              "gcap20": "lane-delta gate capped at g(20)=0.1595 (accel gate untouched)"}},
       "variants": {}}
for model in ("C1", "R4"):
    for variant in ("base", "delta0", "gcap20"):
        if (model, variant) in by:
            res["variants"][f"{model}/{variant}"] = variant_summary(model, variant)

# paired per-state comparisons
def paired(a, b, field):
    ka = by[(a[0], a[1])]
    kb = by[(b[0], b[1])]
    out = {}
    for h in ("60", "120"):
        diffs, base_vals = [], []
        for ra, rb in zip(ka, kb):
            va = ra["result"]["metrics"][f"ds_med_{h}"]
            vb = rb["result"]["metrics"][f"ds_med_{h}"]
            if va is not None and vb is not None:
                diffs.append(va - vb)
                base_vals.append(vb)
        out[h] = {"mean_paired_diff": float(np.mean(diffs)) if diffs else None,
                  "median_paired_diff": float(np.median(diffs)) if diffs else None,
                  "n_better": int(sum(d < 0 for d in diffs)), "n_worse": int(sum(d > 0 for d in diffs)),
                  "n": len(diffs)}
    return out


res["paired_ds_med"] = {
    "R4-base_minus_C1-base": paired(("R4", "base"), ("C1", "base"), "ds"),
    "R4-delta0_minus_R4-base": paired(("R4", "delta0"), ("R4", "base"), "ds"),
    "R4-gcap20_minus_R4-base": paired(("R4", "gcap20"), ("R4", "base"), "ds"),
}

(RUN / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def short(tag):
    v = res["variants"][tag]
    r1, r2 = v["rates"]["0-60"]["pred_per_hdv_per_10s"], v["rates"]["60-120"]["pred_per_hdv_per_10s"]
    return (f"{tag}: lane/HDV/10s 0-60 {r1:.3f} | 60-120 {r2:.3f} | wrong-lane 60/120 "
            f"{v['wrong_lane']['60']['share']:.3f}/{v['wrong_lane']['120']['share']:.3f} | "
            f"ds_med 60/120 {v['ds_med']['60']['median']:.2f}/{v['ds_med']['120']['median']:.2f} "
            f"(p90 {v['ds_med']['120']['p90']:.1f}) | blowups {v['blowups']}")


for tag in ("C1/base", "R4/base", "R4/delta0", "R4/gcap20"):
    print(short(tag))
tr = res["variants"]["R4/base"]["trace"]
tc = res["variants"]["C1/base"]["trace"]
for name, t in (("R4", tr), ("C1", tc)):
    print(f"{name} early decisions {t['early_decisions']} (mobil-would {t['mobil_would_also_change']}, "
          f"delta-driven {t['delta_driven']}, jumps {t['two_lane_jumps']}); "
          f"mobil-only flip rate 0-60 {t['mobil_only_flip_rate_0_60_per_10s']:.3f}/10s")
    print(f"  margin@delta-driven p50 {t['mobil_margin_at_delta_driven']['p50']} "
          f"corr_adv p50 {t['corr_adv_at_delta_driven']['p50']} gate p50 {t['gate_at_decisions']['p50']}")
    print(f"  time_hist(5s) {t['time_hist_5s']} lane {t['lane_hist_cur_lane']} zone {t['zone_hist']}")
    print(f"  cav {t['cav_dist']} junction_self {t['junction_self_within_5s']} near_rec_change "
          f"{t['near_recorded_change_within_5s']} exit10s {t['exit_within_10s']}")
print(json.dumps(res["paired_ds_med"], indent=1))
