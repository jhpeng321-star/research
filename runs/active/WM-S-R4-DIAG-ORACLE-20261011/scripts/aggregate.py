"""DIAG 4 step 3: metrics for every variant + paired comparison vs cached C1/R4.

Reuses the R4-score cache state rows (C1, R4-last on wall_same) as comparators and the
DIAG-ERR per-HDV arrays (out/rollouts.pkl of WM-S-R4-DIAG-ERR-20261011) for pooled C1/R4
per-HDV reference numbers.  Writes results.json next to this script's parent.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from wms.eval.report import paired_bootstrap  # noqa: E402
from wms.eval.stats import summarize  # noqa: E402

OUT = HERE.parent / "out"
DIAG_ERR = ROOT / "runs/active/WM-S-R4-DIAG-ERR-20261011/out"
CACHE = ROOT / "runs/active/WM-S-R4-20261011/score/cache"
TRUTH = json.loads((OUT / "truth.json").read_text(encoding="utf-8"))
HORIZONS = (10, 30, 60, 120)
TICKS = {h: h * 2 for h in HORIZONS}
VARIANTS = ("r4_vanilla", "phys_prior", "phys_true", "phys_true_tfl", "r4_true",
            "phys_true_sl", "phys_true_sl_act")


def matched(pred, rec, tick):
    """Pair-wise aligned per-HDV arrays at one tick: (mask, ds, dv, lane_ok)."""
    rids, pids = rec["ids"], pred["ids"]
    ridx = {x: i for i, x in enumerate(rids)}
    n = len(pids)
    ds = np.full(n, np.nan)
    dv = np.full(n, np.nan)
    lane_ok = np.zeros(n, dtype=bool)
    matched_mask = np.zeros(n, dtype=bool)
    for k, vid in enumerate(pids):
        j = ridx.get(vid)
        if j is None:
            continue
        if not (pred["present"][tick, k] and rec["present"][tick, j]):
            continue
        if rec["is_cav"][tick, j] or pred["is_cav"][tick, k]:
            continue
        matched_mask[k] = True
        ds[k] = pred["s"][tick, k] - rec["s"][tick, j]
        dv[k] = pred["v"][tick, k] - rec["v"][tick, j]
        if rec["lane"][tick, j] >= 0:
            lane_ok[k] = pred["lane"][tick, k] == rec["lane"][tick, j]
    return matched_mask, ds, dv, lane_ok


def state_summary(rows):
    out = {}
    for h in HORIZONS:
        vals = []
        for r in rows:
            if r.get("error"):
                continue
            v = r["car"][f"{h:.1f}"]["hdv"]["ds_med"]
            if v is not None and np.isfinite(v):
                vals.append(float(v))
        out[str(h)] = summarize(vals)
    blows = n_blow = 0
    for r in rows:
        if r.get("error"):
            continue
        a = r["car"]["60.0"]["hdv"]["ds_med"]
        b = r["car"]["120.0"]["hdv"]["ds_med"]
        if a is not None and b is not None:
            n_blow += 1
            blows += int(a > 50 and b > 100)
    out["blowups"] = {"n": blows, "denominator": n_blow,
                      "rule": "HDV median |ds| >50m at 60s AND >100m at 120s"}
    return out


def pairing_row(r):
    return {"world": r["world"], "key": r["key"], "source_arm": "wall", "t0_s": r["t0"],
            "arms": {"wall": {"car": {f"{h:.1f}": {"hdv": {"ds_med":
                        (None if r.get("error") else r["car"][f"{h:.1f}"]["hdv"]["ds_med"])}}
                              for h in HORIZONS}}}}


def cache_rows(name):
    d = json.loads((CACHE / f"{name}__wall_same.json").read_text(encoding="utf-8"))
    rows = []
    for s in d["states"]:
        arms = s.get("arms") or {}
        wall = arms.get("wall") or {}
        if wall.get("error"):
            continue
        car = {}
        for h in HORIZONS:
            block = ((wall.get("car") or {}).get(f"{h:.1f}") or {}).get("hdv") or {}
            car[f"{h:.1f}"] = {"hdv": {"ds_med": block.get("ds_med")}}
        rows.append({"world": s["world"], "key": s.get("key"), "source_arm": "wall",
                     "t0_s": s["t0_s"], "arms": {"wall": {"car": car}}})
    return rows


def per_hdvs(rows):
    """Convert stored rows into pred/rec dicts at every 10 s tick."""
    states = []
    for r in rows:
        states.append({"world": r["world"], "key": r["key"], "t0": r["t0"],
                       "pred": {"ids": r["pids"], **r["pred"]},
                       "rec": {"ids": r["rids"], **r["rec"]}})
    return states


def pooled_at_tick(states, tick):
    ds_all, dv_all, lane_ok, lane_n, types = [], [], 0, 0, []
    for st in states:
        m, ds, dv, lok = matched(st["pred"], st["rec"], tick)
        ds_all.extend(ds[m].tolist())
        dv_all.extend(dv[m].tolist())
        lane_ok += int(lok[m].sum())
        lane_n += int(m.sum())
        veh = TRUTH[st["world"]]["vehicles"]
        types.extend(veh.get(v, "?") for v, k in zip(st["pred"]["ids"], range(len(m))) if m[k])
    ds_all = np.asarray(ds_all)
    dv_all = np.asarray(dv_all)
    return {"n": int(ds_all.size),
            "med_abs_ds": float(np.median(np.abs(ds_all))) if ds_all.size else None,
            "mean_abs_ds": float(np.mean(np.abs(ds_all))) if ds_all.size else None,
            "mean_signed_ds": float(np.mean(ds_all)) if ds_all.size else None,
            "mean_signed_dv": float(np.mean(dv_all)) if dv_all.size else None,
            "wrong_lane_share": (1.0 - lane_ok / lane_n) if lane_n else None}


def per_type_at_tick(states, tick):
    out = {}
    for st in states:
        veh = TRUTH[st["world"]]["vehicles"]
        m, ds, dv, _ = matched(st["pred"], st["rec"], tick)
        for k in np.flatnonzero(m):
            t = veh.get(st["pred"]["ids"][k], "?")
            out.setdefault(t, []).append(abs(float(ds[k])))
    return {t: {"n": len(v), "med_abs_ds": float(np.median(v)), "mean_abs_ds": float(np.mean(v))}
            for t, v in sorted(out.items())}


def main() -> int:
    results = {"schema": "wm_s_diag4_oracle/1", "variants": {}, "paired": {},
               "verification": {}, "notes": {
                   "state_median_metric": "median over states of per-state HDV median |ds| (m)",
                   "pooled": "per-HDV values over all states (matched = present in both, HDV)",
                   "pairing": "state-level ds_med paired diff (variant - C1 / - R4), "
                              "world bootstrap 2000, seed 0 (wms.eval.report.paired_bootstrap)",
               }}
    c1_rows, r4_rows = cache_rows("C1"), cache_rows("R4-last")
    ref = {"C1": c1_rows, "R4": r4_rows}

    # verification of r4_vanilla against the cache, all states
    van = pickle.loads((OUT / "variant_r4_vanilla.pkl").read_bytes())
    diffs = []
    mism = 0
    for r, c in zip(van, r4_rows):
        for h in HORIZONS:
            mine = None if r.get("error") else r["car"][f"{h:.1f}"]["hdv"]["ds_med"]
            theirs = c["arms"]["wall"]["car"][f"{h:.1f}"]["hdv"]["ds_med"]
            if mine is None or theirs is None:
                continue
            diffs.append(abs(float(mine) - float(theirs)))
            mism += int(abs(float(mine) - float(theirs)) > 1e-6)
    results["verification"] = {
        "r4_vanilla_vs_cache": {"checks": len(diffs), "mismatches_gt_1e-6": mism,
                                "max_abs_diff": float(max(diffs)) if diffs else None},
        "c1_reference": "score cache C1__wall_same.json (reproduced bit-exactly by "
                        "WM-S-R4-DIAG-ERR-20261011, 266/266)",
    }

    for name in VARIANTS:
        rows = pickle.loads((OUT / f"variant_{name}.pkl").read_bytes())
        states = per_hdvs(rows)
        block = {"state_median_ds": state_summary(rows)}
        block["pooled"] = {}
        for h in HORIZONS:
            block["pooled"][str(h)] = pooled_at_tick(states, TICKS[h])
        block["pooled"]["types_120"] = per_type_at_tick(states, TICKS[120])
        block["time_profile_10s"] = {str(t): pooled_at_tick(states, t * 2)
                                     for t in range(10, 121, 10)}
        results["variants"][name] = block
        mine = [pairing_row(r) for r in rows]
        results["paired"][name] = {}
        for ref_name, ref_rows in ref.items():
            results["paired"][name][f"minus_{ref_name}"] = {
                str(h): paired_bootstrap(mine, ref_rows, "wall", float(h),
                                         n_boot=2000, seed=0)
                for h in HORIZONS}

    # C1/R4 pooled reference from the DIAG-ERR arrays (same metric definitions)
    with open(DIAG_ERR / "rollouts.pkl", "rb") as fh:
        payloads = pickle.load(fh)
    for model in ("C1", "R4"):
        states = [{"world": r["world"], "key": r["key"], "t0": r["t0"],
                   "pred": {"ids": r["pids"], **r["pred"]},
                   "rec": {"ids": r["rids"], **r["rec"]}} for r in payloads[model]]
        results["variants"][f"ref_{model}"] = {
            "state_median_ds": {
                **{str(h): summarize([float(c["arms"]["wall"]["car"][f"{h:.1f}"]["hdv"]["ds_med"])
                                     for c in (c1_rows if model == "C1" else r4_rows)
                                     if c["arms"]["wall"]["car"][f"{h:.1f}"]["hdv"]["ds_med"]
                                     is not None]) for h in HORIZONS},
                "blowups": {"n": 0, "denominator": 66,
                            "rule": "HDV median |ds| >50m at 60s AND >100m at 120s"}},
            "pooled": {str(h): pooled_at_tick(states, TICKS[h]) for h in HORIZONS},
            "time_profile_10s": {str(t): pooled_at_tick(states, t * 2)
                                 for t in range(10, 121, 10)},
        }

    (HERE.parent / "results.json").write_text(json.dumps(results, indent=1),
                                              encoding="utf-8")
    print(f"wrote {HERE.parent / 'results.json'}")

    # compact console summary
    hdr = f"{'variant':18s} " + " ".join(f"ds{s:>4d}s(state-med/pooled-med)" for s in HORIZONS)
    print(hdr)
    for name in VARIANTS + ("ref_C1", "ref_R4"):
        b = results["variants"][name]
        cells = []
        for h in HORIZONS:
            sm = b["state_median_ds"][str(h)]["median"]
            pm = b["pooled"][str(h)]["med_abs_ds"]
            cells.append(f"{sm:8.2f}/{pm:8.2f}")
        print(f"{name:18s} " + " ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
