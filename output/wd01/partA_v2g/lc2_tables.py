"""WD-01-PA2G-LC2 aggregation: registered table, world-bootstrap CIs,
pre-registered decision rule (REGISTRATION_PA2G_LC2.md sections 3-4).

Reads the RAW rows saved by run_lc2.py (lc2_rows.jsonl +
lc2_merge_outcomes.jsonl) -- aggregation only, no re-simulation.

Outputs results/lc2/lc2_summary.json (+ printed markdown table):

- per (variant, H): s_RMSE, v_RMSE, delay err signed/MAE, disc-LC and
  merge-LC predicted/true ratios (pooled counts), merge dwell median,
  early-edge share (merge_s < 415.0), dead-end stops, collisions/1000,
  TTC<3 and hard-brake errors (signed mean and MAE);
- world-bootstrap 90 % CI (B = 2000, seed 20260930) of each variant -
  nominal difference (s_RMSE/v_RMSE/delay-MAE as RELATIVE differences,
  the rest as absolute differences);
- the decision rule applied verbatim per variant: BIG / MODEST /
  NO IMPROVEMENT, and the recommendation (smallest BIG variant or
  'keep textbook').

Zero SUMO; no truth file is opened here (raw rows only).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "lc2"
HS = (10, 30, 60, 120)
VARIANTS = ("nominal", "va", "vb", "vc", "vabc")
B_BOOT = 2000
SEED = 20260930
EARLY_EDGE_S = 415.0


def load_rows():
    rows = [json.loads(line) for line in
            (RESULTS / "lc2_rows.jsonl").read_text(encoding="utf-8")
            .splitlines()]
    mrg = [json.loads(line) for line in
           (RESULTS / "lc2_merge_outcomes.jsonl").read_text(
               encoding="utf-8").splitlines()]
    return rows, mrg


def per_world(rows, mrg):
    """world -> variant -> H -> dict of paired per-world quantities."""
    pw = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r["model"] not in VARIANTS:
            continue
        d = pw[r["world"]][r["model"]].setdefault(
            r["H"], dict(n=0, s=0.0, v=0.0, de=0.0, dae=0.0, te=0.0,
                         tae=0.0, he=0.0, hae=0.0, dp=0, dt=0, lp=0,
                         lt=0, mp=0, mt=0, coll=0, vsteps=0))
        d["n"] += 1
        d["s"] += r["s_rmse"]
        d["v"] += r["v_rmse"]
        d["de"] += r["delay_err"]
        d["dae"] += abs(r["delay_err"])
        d["te"] += r["ttc3_err"]
        d["tae"] += abs(r["ttc3_err"])
        d["he"] += r["hard_err"]
        d["hae"] += abs(r["hard_err"])
        d["dp"] += r["lc_pred"]["disc_left"] + r["lc_pred"]["disc_right"]
        d["dt"] += r["lc_true"]["disc_left"] + r["lc_true"]["disc_right"]
        d["mp"] += r["lc_pred"]["merge"]
        d["mt"] += r["lc_true"]["merge"]
        d["coll"] += r["collisions_pred"]
        d["vsteps"] += r["vehicle_steps_pred"]
    # merge outcomes (within-H dwell / early edge / stops)
    macc = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: dict(dwell=[], m_s=[], stops=0))))
    for m in mrg:
        if m["variant"] not in VARIANTS:
            continue
        for H in HS:
            key = f"H{H}"
            a = macc[m["world"]][m["variant"]][H]
            if m["merged_in"][key]:
                if m["dwell_in"][key] is not None:
                    a["dwell"].append(m["dwell_in"][key])
                if m["merge_s"] is not None:
                    a["m_s"].append(m["merge_s"])
            if m["stop_in"][key]:
                a["stops"] += 1
    for world, byv in macc.items():
        for var, byH in byv.items():
            for H, a in byH.items():
                pw[world][var].setdefault(H, {}).update(
                    dwell=a["dwell"], m_s=a["m_s"], stops=a["stops"])
    return pw


def agg_worlds(worlds, pw, var, H):
    """Aggregate one variant/H over a list of worlds (bootstrap unit =
    world; every metric is a per-world mean/count, then averaged)."""
    ds = [pw[w][var].get(H) for w in worlds]
    ds = [d for d in ds if d and d.get("n")]
    if not ds:
        return None
    n = len(ds)
    out = dict(
        n=n,
        s_rmse=float(np.mean([d["s"] / d["n"] for d in ds])),
        v_rmse=float(np.mean([d["v"] / d["n"] for d in ds])),
        delay_err_mean=float(np.mean([d["de"] / d["n"] for d in ds])),
        delay_err_mae=float(np.mean([d["dae"] / d["n"] for d in ds])),
        ttc3_err_mean=float(np.mean([d["te"] / d["n"] for d in ds])),
        ttc3_err_mae=float(np.mean([d["tae"] / d["n"] for d in ds])),
        hard_err_mean=float(np.mean([d["he"] / d["n"] for d in ds])),
        hard_err_mae=float(np.mean([d["hae"] / d["n"] for d in ds])),
        disc_ratio=(sum(d["dp"] for d in ds) /
                    max(sum(d["dt"] for d in ds), 1)),
        merge_ratio=(sum(d["mp"] for d in ds) /
                     max(sum(d["mt"] for d in ds), 1)),
        disc_p=sum(d["dp"] for d in ds), disc_t=sum(d["dt"] for d in ds),
        merge_p=sum(d["mp"] for d in ds), merge_t=sum(d["mt"] for d in ds),
        coll_per_1000=(1000.0 * sum(d["coll"] for d in ds) /
                       max(sum(d["vsteps"] for d in ds), 1)),
        stops=sum(d.get("stops", 0) for d in ds),
    )
    dwell = [x for d in ds for x in d.get("dwell", [])]
    m_s = [x for d in ds for x in d.get("m_s", [])]
    out["dwell_median"] = float(np.median(dwell)) if dwell else float("nan")
    out["n_merge_events"] = len(m_s)
    out["early_edge_share"] = (float(np.mean([x < EARLY_EDGE_S
                                              for x in m_s]))
                               if m_s else float("nan"))
    return out


def boot_ci(worlds, pw, var, H, b=B_BOOT, seed=SEED):
    """World bootstrap 90 % CI of variant - nominal differences."""
    rng = np.random.default_rng(seed)
    n = len(worlds)
    rel_s, rel_v, rel_d, d_disc, d_merge, d_dwell, d_early, d_stops, \
        d_coll = ([] for _ in range(9))
    for _ in range(b):
        idx = rng.integers(0, n, n)
        ws = [worlds[i] for i in idx]
        a = agg_worlds(ws, pw, var, H)
        bn = agg_worlds(ws, pw, "nominal", H)
        if a is None or bn is None:
            continue
        rel_s.append((a["s_rmse"] - bn["s_rmse"]) / bn["s_rmse"])
        rel_v.append((a["v_rmse"] - bn["v_rmse"]) / bn["v_rmse"])
        rel_d.append((a["delay_err_mae"] - bn["delay_err_mae"])
                     / bn["delay_err_mae"])
        d_disc.append(a["disc_ratio"] - bn["disc_ratio"])
        d_merge.append(a["merge_ratio"] - bn["merge_ratio"])
        if np.isfinite(a["dwell_median"]) and \
                np.isfinite(bn["dwell_median"]):
            d_dwell.append(a["dwell_median"] - bn["dwell_median"])
        if np.isfinite(a["early_edge_share"]) and \
                np.isfinite(bn["early_edge_share"]):
            d_early.append(a["early_edge_share"] - bn["early_edge_share"])
        d_stops.append(a["stops"] - bn["stops"])
        d_coll.append(a["coll_per_1000"] - bn["coll_per_1000"])

    def ci(x):
        if not x:
            return None
        lo, hi = np.percentile(x, [5.0, 95.0])
        return [float(lo), float(hi)]

    return dict(rel_s_rmse=ci(rel_s), rel_v_rmse=ci(rel_v),
                rel_delay_mae=ci(rel_d), disc_ratio_diff=ci(d_disc),
                merge_ratio_diff=ci(d_merge), dwell_diff=ci(d_dwell),
                early_share_diff=ci(d_early), stops_diff=ci(d_stops),
                coll1000_diff=ci(d_coll))


def ci_excludes_zero(ci):
    return ci is not None and (ci[0] > 0.0 or ci[1] < 0.0)


def decide(var, table, cis):
    """The registered rule (REGISTRATION_PA2G_LC2.md section 4),
    applied verbatim; operationalization documented in the report."""
    t30, t60, t10 = table["H30"], table["H60"], table["H10"]
    n30, n60, n10 = table["nom_H30"], table["nom_H60"], table["nom_H10"]
    rel30 = (t30["s_rmse"] - n30["s_rmse"]) / n30["s_rmse"]
    rel60 = (t60["s_rmse"] - n60["s_rmse"]) / n60["s_rmse"]
    rel10 = (t10["s_rmse"] - n10["s_rmse"]) / n10["s_rmse"]
    rel_d30 = ((t30["delay_err_mae"] - n30["delay_err_mae"])
               / n30["delay_err_mae"])
    ci30 = cis["H30"]["rel_s_rmse"]
    ci60 = cis["H60"]["rel_s_rmse"]
    excl30 = ci_excludes_zero(ci30) and rel30 < 0
    excl60 = ci_excludes_zero(ci60) and rel60 < 0
    rmse_full = (rel30 <= -0.20 and excl30) and (rel60 <= -0.20 and excl60)
    ratio_ok = (0.67 <= t30["disc_ratio"] <= 1.5) and \
        (0.67 <= t30["merge_ratio"] <= 1.5)
    benefit = rmse_full or ratio_ok
    no_harm = (rel10 <= 0.05 and rel30 <= 0.05 and rel_d30 <= 0.10 and
               t30["coll_per_1000"] <= n30["coll_per_1000"] + 0.1)
    rmse_partial = (not rmse_full) and (
        (rel30 <= -0.20 and excl30) or (rel60 <= -0.20 and excl60) or
        (rel30 <= -0.20 and rel60 <= -0.20))
    ratio_partial = (not ratio_ok) and (
        0.67 <= t30["disc_ratio"] <= 1.5 or
        0.67 <= t30["merge_ratio"] <= 1.5)
    if benefit and no_harm:
        verdict = "BIG IMPROVEMENT"
    elif no_harm and (rmse_partial or ratio_partial):
        verdict = "MODEST"
    else:
        verdict = "NO IMPROVEMENT"
    return dict(rel_s_rmse_H30=rel30, rel_s_rmse_H60=rel60,
                rel_s_rmse_H10=rel10, rel_delay_mae_H30=rel_d30,
                ci_H30_excludes_0=bool(excl30), ci_H60_excludes_0=bool(excl60),
                rmse_criterion=bool(rmse_full), ratio_criterion=bool(ratio_ok),
                benefit=bool(benefit), no_harm=bool(no_harm),
                rmse_partial=bool(rmse_partial),
                ratio_partial=bool(ratio_partial), verdict=verdict)


def main():
    rows, mrg = load_rows()
    pw = per_world(rows, mrg)
    worlds = sorted(pw)
    assert len(worlds) == 32, f"expected 32 worlds, got {len(worlds)}"
    summary = dict(
        analysis_id="WD-01-PA2G-LC2",
        worlds=worlds, b_boot=B_BOOT, seed=SEED,
        early_edge_s=EARLY_EDGE_S,
        table={}, cis={}, decisions={}, recommendation=None)
    for var in VARIANTS:
        for H in HS:
            a = agg_worlds(worlds, pw, var, H)
            summary["table"][f"{var}|H{H}"] = a
    for var in VARIANTS[1:]:
        cis = {f"H{H}": boot_ci(worlds, pw, var, H) for H in HS}
        summary["cis"][var] = cis
        tab = {f"H{H}": summary["table"][f"{var}|H{H}"] for H in HS}
        tab["nom_H10"] = summary["table"]["nominal|H10"]
        tab["nom_H30"] = summary["table"]["nominal|H30"]
        tab["nom_H60"] = summary["table"]["nominal|H60"]
        summary["decisions"][var] = decide(var, tab, cis)
    bigs = [v for v in VARIANTS[1:]
            if summary["decisions"][v]["verdict"] == "BIG IMPROVEMENT"]
    if bigs:
        order = {"va": 1, "vb": 2, "vc": 3, "vabc": 4}
        smallest = sorted(bigs, key=lambda v: order[v])[0]
        summary["recommendation"] = (
            f"option 2 worth registering: smallest BIG variant = {smallest}")
    else:
        summary["recommendation"] = "keep textbook"
    (RESULTS / "lc2_summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8")
    # ---- console table
    print("WD-01-PA2G-LC2 table (means over windows; ratios pooled)")
    hdr = ("variant|H   n    s_RMSE   v_RMSE   dErr    dMAE   "
           "disc p/t        mrg p/t       dwell  early%  stops  "
           "coll/1k  ttc3err  harderr")
    print(hdr)
    for var in VARIANTS:
        for H in HS:
            a = summary["table"][f"{var}|H{H}"]
            print(f"{var}|H{H:<4} {a['n']:<4} {a['s_rmse']:8.3f} "
                  f"{a['v_rmse']:8.3f} {a['delay_err_mean']:7.3f} "
                  f"{a['delay_err_mae']:6.3f}  "
                  f"{a['disc_p']}/{a['disc_t']} ({a['disc_ratio']:.2f}x)  "
                  f"{a['merge_p']}/{a['merge_t']} "
                  f"({a['merge_ratio']:.2f}x)  "
                  f"{a['dwell_median']:6.1f} {100*a['early_edge_share']:6.1f} "
                  f"{a['stops']:5} {a['coll_per_1000']:7.3f} "
                  f"{a['ttc3_err_mean']:7.2f} {a['hard_err_mean']:7.2f}")
    for var in VARIANTS[1:]:
        d = summary["decisions"][var]
        print(f"\n{var}: rel s_RMSE H10/H30/H60 = "
              f"{d['rel_s_rmse_H10']:+.3f}/{d['rel_s_rmse_H30']:+.3f}/"
              f"{d['rel_s_rmse_H60']:+.3f}; rel delay MAE H30 "
              f"{d['rel_delay_mae_H30']:+.3f}; CI H30 {summary['cis'][var]['H30']['rel_s_rmse']}, "
              f"H60 {summary['cis'][var]['H60']['rel_s_rmse']}")
        print(f"  verdict: {d['verdict']}  "
              f"(rmse_full {d['rmse_criterion']}, ratio_ok "
              f"{d['ratio_criterion']}, no_harm {d['no_harm']})")
    print("\nrecommendation:", summary["recommendation"])


if __name__ == "__main__":
    main()
