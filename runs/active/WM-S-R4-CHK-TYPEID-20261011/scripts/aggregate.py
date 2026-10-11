"""CHK-TYPEID aggregation: verification vs DIAG-ORACLE, tables, paired CIs.

Writes results.json next to this script's parent and prints compact tables.
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
CACHE = ROOT / "runs/active/WM-S-R4-20261011/score/cache"
TRUTH = json.loads((OUT / "truth_full.json").read_text(encoding="utf-8"))
HORIZONS = (10, 30, 60, 120)
TICKS = {h: h * 2 for h in HORIZONS}
DIAG_REF = {
    "phys_prior": {"10": 3.72, "30": 10.17, "60": 15.68, "120": 53.12},
    "phys_true": {"10": 0.46, "30": 1.34, "60": 4.31, "120": 48.65},
}
BLOCKS = [
    ("wall_same", "variant_phys_prior.pkl", "phys_prior"),
    ("wall_same", "variant_phys_true.pkl", "phys_true"),
    ("wall_same", "variant_phys_ident.pkl", "phys_ident5"),
    ("wall_same", "variant_phys_ident_post.pkl", "phys_ident5_post"),
    ("real_q20", "variant_phys_true_real_q20.pkl", "phys_true_real"),
    ("real_q20", "variant_phys_ident_real_q20.pkl", "phys_ident60_real"),
    ("real_q20", "variant_phys_ident_post_real_q20.pkl", "phys_ident60_post_real"),
    ("real_full", "variant_phys_ident_real_full.pkl", "phys_ident60_real_full"),
]
CACHE_SET = {"wall_same": "wall_same", "real_q20": "wall_same_real"}


def matched(pred, rec, tick):
    rids, pids = rec["ids"], pred["ids"]
    ridx = {x: i for i, x in enumerate(rids)}
    n = len(pids)
    ds = np.full(n, np.nan)
    dv = np.full(n, np.nan)
    lane_ok = np.zeros(n, dtype=bool)
    m = np.zeros(n, dtype=bool)
    for k, vid in enumerate(pids):
        j = ridx.get(vid)
        if j is None:
            continue
        if not (pred["present"][tick, k] and rec["present"][tick, j]):
            continue
        if rec["is_cav"][tick, j] or pred["is_cav"][tick, k]:
            continue
        m[k] = True
        ds[k] = pred["s"][tick, k] - rec["s"][tick, j]
        dv[k] = pred["v"][tick, k] - rec["v"][tick, j]
        if rec["lane"][tick, j] >= 0:
            lane_ok[k] = pred["lane"][tick, k] == rec["lane"][tick, j]
    return m, ds, dv, lane_ok


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


def cache_rows(name, set_name):
    d = json.loads((CACHE / f"{name}__{set_name}.json").read_text(encoding="utf-8"))
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
    return [{"world": r["world"], "key": r["key"], "t0": r["t0"],
             "pred": {"ids": r["pids"], **r["pred"]},
             "rec": {"ids": r["rids"], **r["rec"]}} for r in rows if not r.get("error")]


def pooled_at_tick(states, tick):
    ds_all, dv_all, lane_ok, lane_n = [], [], 0, 0
    for st in states:
        m, ds, dv, lok = matched(st["pred"], st["rec"], tick)
        ds_all.extend(ds[m].tolist())
        dv_all.extend(dv[m].tolist())
        lane_ok += int(lok[m].sum())
        lane_n += int(m.sum())
    ds_all = np.asarray(ds_all)
    dv_all = np.asarray(dv_all)
    return {"n": int(ds_all.size),
            "med_abs_ds": float(np.median(np.abs(ds_all))) if ds_all.size else None,
            "mean_signed_ds": float(np.mean(ds_all)) if ds_all.size else None,
            "mean_signed_dv": float(np.mean(dv_all)) if dv_all.size else None,
            "wrong_lane_share": (1.0 - lane_ok / lane_n) if lane_n else None}


def per_type_at_tick(states, tick):
    out = {}
    for st in states:
        veh = TRUTH[st["world"]]["vehicles"]
        m, ds, _, _ = matched(st["pred"], st["rec"], tick)
        for k in np.flatnonzero(m):
            t = veh.get(st["pred"]["ids"][k], "?")
            out.setdefault(t, []).append(abs(float(ds[k])))
    return {t: {"n": len(v), "med_abs_ds": float(np.median(v))}
            for t, v in sorted(out.items())}


def main() -> int:
    results = {"schema": "wm_s_chk_typeid/1", "variants": {}, "paired": {},
               "verification": {}, "identification": json.loads(
                   (OUT / "ident_eval.json").read_text(encoding="utf-8")),
               "notes": {
                   "state_median_metric": "median over states of per-state HDV median |ds| (m)",
                   "pairing": "state-level ds_med paired diff (variant - reference), "
                              "world bootstrap 2000, seed 0",
                   "cache_refs": "C1/R4 from the R4 score cache (same harness)",
               }}
    # verification of the harness against DIAG-ORACLE numbers
    for variant in ("phys_prior", "phys_true"):
        rows = pickle.loads((OUT / f"variant_{variant}.pkl").read_bytes())
        mine = state_summary(rows)
        results["verification"][variant] = {
            str(h): {"mine": mine[str(h)]["median"], "diag_oracle": DIAG_REF[variant][str(h)],
                     "abs_diff": abs(mine[str(h)]["median"] - DIAG_REF[variant][str(h)])}
            for h in HORIZONS}

    refs = {}
    for set_tag in ("wall_same", "real_q20"):
        for model in ("C1", "R4-last"):
            refs[(set_tag, model)] = cache_rows(model, CACHE_SET[set_tag])

    for set_tag, fname, name in BLOCKS:
        rows = pickle.loads((OUT / fname).read_bytes())
        states = per_hdvs(rows)
        block = {"set": set_tag, "n_states": len(rows),
                 "state_median_ds": state_summary(rows)}
        block["pooled"] = {str(h): pooled_at_tick(states, TICKS[h]) for h in HORIZONS}
        block["pooled"]["types_120"] = per_type_at_tick(states, TICKS[120])
        results["variants"][name] = block
        mine = [pairing_row(r) for r in rows]
        results["paired"][name] = {}
        for (set_tag2, model), ref_rows in refs.items():
            if set_tag2 != set_tag:
                continue
            results["paired"][name][f"minus_{model.split('-')[0]}"] = {
                str(h): paired_bootstrap(mine, ref_rows, "wall", float(h),
                                         n_boot=2000, seed=0) for h in HORIZONS}
        # paired vs own phys_true / phys_prior
        for other in ("phys_prior", "phys_true", "phys_ident"):
            key = (name, other)
            oname = other if set_tag == "wall_same" else None
            fname2 = {"phys_prior": "variant_phys_prior.pkl",
                      "phys_true": "variant_phys_true.pkl",
                      "phys_ident": "variant_phys_ident.pkl"}.get(other)
            if set_tag != "wall_same":
                fname2 = {"phys_prior": None,
                          "phys_true": "variant_phys_true_real_q20.pkl",
                          "phys_ident": "variant_phys_ident_real_q20.pkl"}.get(other)
            if fname2 is None or Path(OUT / fname2).stem == Path(OUT / fname).stem:
                continue
            ref_rows = [pairing_row(r) for r in
                        pickle.loads((OUT / fname2).read_bytes()) if not r.get("error")]
            results["paired"][name][f"minus_{other}"] = {
                str(h): paired_bootstrap(mine, ref_rows, "wall", float(h),
                                         n_boot=2000, seed=0) for h in HORIZONS}

    (HERE.parent / "results.json").write_text(json.dumps(results, indent=1),
                                              encoding="utf-8")
    print(f"wrote {HERE.parent / 'results.json'}")
    print("verification:", json.dumps(results["verification"], indent=1))
    hdr = f"{'variant':24s} " + " ".join(f"ds{s:>4d}s" for s in HORIZONS) + "  blowups"
    print(hdr)
    for _, _, name in BLOCKS:
        b = results["variants"][name]["state_median_ds"]
        cells = [f"{b[str(h)]['median']:8.2f}" for h in HORIZONS]
        print(f"{name:24s} " + " ".join(cells)
              + f"  {b['blowups']['n']}/{b['blowups']['denominator']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
