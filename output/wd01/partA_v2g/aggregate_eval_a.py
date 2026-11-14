"""WD-01-PA2G-A close-out aggregation: registered tables from the SAVED
eval_a raw rows.

ZERO SIMULATION. Reads results/eval_a/raw_rows.jsonl only (never opens
CAL/VALIDATION/HOLDOUT data, never touches SUMO/TraCI) and re-derives the
registered aggregations:

- per model x H: trajectory RMSE (s, v) over tracked vehicles, delay /
  TTC<3 / hard-brake errors (mean SIGNED and mean ABSOLUTE reported
  separately), lane-change counts by type vs truth, predicted collisions
  per 1000 vehicle-steps;
- missing-arrivals curve per H (truth delay tracked-only vs all vehicles,
  untracked share);
- n_free_road_assumed stats per H (predictor-side zone-truncation rule)
  alongside the truth-exact free-road flags;
- descriptive T/a sweep at H=60 (labelled; NO parameter selected);
- breakdowns by penetration and demand;
- H120-vs-H60 sign-cancellation diagnostics (sign counts, per-penetration
  signed means).

Data note (disclosed in the report): the raw file contains 6402 lines.
576 of them are const-speed rows re-emitted by the T/a sweep passes
(identical recomputations of the main-pass const-speed row of the same
window; verified value-identical below before being dropped), and the
v2g_sweep rows share their (world, w, H, model) key across the eight
T/a combos (they are distinguished by the T_s / a_max fields).  The main
tables therefore use 5826 deduplicated rows.

Usage:
    python aggregate_eval_a.py          # writes tables.json + tables.md
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "eval_a"
RAW = RESULTS / "raw_rows.jsonl"
MODELS = ("v2g", "v1_p3o_replica", "const_speed")
HS = (10, 30, 60, 120)
IDENT = ("s_rmse", "v_rmse", "delay_err", "ttc3_err", "hard_err",
         "collisions_pred", "vehicle_steps_pred", "n_matched",
         "delay_pred", "delay_true_tracked")


def mean(xs):
    return st.fmean(xs) if xs else float("nan")


def mae(xs):
    return mean([abs(x) for x in xs])


def load_rows() -> Tuple[List[dict], dict]:
    rows = [json.loads(line) for line in
            RAW.read_text(encoding="utf-8").splitlines() if line.strip()]
    dup_cond = lambda r: r["model"] == "const_speed" and "T_s" in r  # noqa: E731
    dups = [r for r in rows if dup_cond(r)]
    main = [r for r in rows if not dup_cond(r)]
    idx = {(r["world"], r["w"], r["H"], r["model"]): r for r in main}
    mismatches = sum(
        1 for r in dups
        if any(r[f] != idx[(r["world"], r["w"], r["H"],
                            "const_speed")][f] for f in IDENT))
    assert mismatches == 0, \
        f"{mismatches} sweep-pass const rows differ from the main pass"
    meta = dict(raw_lines=len(rows),
                sweep_pass_const_duplicates_dropped=len(dups),
                main_rows_used=len(main),
                v2g_sweep_rows=sum(1 for r in main
                                   if r["model"] == "v2g_sweep"))
    print(f"raw rows {meta['raw_lines']}; sweep-pass const-speed "
          f"duplicates {meta['sweep_pass_const_duplicates_dropped']} "
          f"(value-identical, dropped); main rows {meta['main_rows_used']}")
    return main, meta


def cell(g: List[dict]) -> dict:
    lc_m_p = sum(r["lc_pred"]["merge"] for r in g)
    lc_m_t = sum(r["lc_true"]["merge"] for r in g)
    lc_d_p = sum(r["lc_pred"]["disc_left"] + r["lc_pred"]["disc_right"]
                 for r in g)
    lc_d_t = sum(r["lc_true"]["disc_left"] + r["lc_true"]["disc_right"]
                 for r in g)
    vsteps = sum(r["vehicle_steps_pred"] for r in g)
    colls = sum(r["collisions_pred"] for r in g)
    return dict(
        n=len(g),
        s_rmse_mean=mean([r["s_rmse"] for r in g]),
        v_rmse_mean=mean([r["v_rmse"] for r in g]),
        delay_err_mean=mean([r["delay_err"] for r in g]),
        delay_err_mae=mae([r["delay_err"] for r in g]),
        ttc3_err_mean=mean([r["ttc3_err"] for r in g]),
        ttc3_err_mae=mae([r["ttc3_err"] for r in g]),
        hard_err_mean=mean([r["hard_err"] for r in g]),
        hard_err_mae=mae([r["hard_err"] for r in g]),
        lc_merge_pred=lc_m_p, lc_merge_true=lc_m_t,
        lc_disc_pred=lc_d_p, lc_disc_true=lc_d_t,
        collisions_pred=colls, vehicle_steps_pred=vsteps,
        collisions_per_1000=colls / max(vsteps / 1000.0, 1e-9),
        n_free_road_true_mean=mean([r["n_free_road_true"] for r in g]),
        n_free_road_pred_mean=mean([r["n_free_road_pred"] for r in g]),
        n_tracked_mean=mean([r["n_tracked"] for r in g]),
    )


def rnd(d: dict) -> dict:
    return {k: (round(v, 6) if isinstance(v, float) else v)
            for k, v in d.items()}


def main() -> None:
    rows, counts = load_rows()
    main = [r for r in rows if r["model"] in MODELS]
    sweep = [r for r in rows if r["model"] == "v2g_sweep"]
    by_mh: Dict[tuple, List[dict]] = defaultdict(list)
    for r in main:
        by_mh[(r["model"], int(r["H"]))].append(r)

    per_model_h = {f"{m}|H{h}": cell(g)
                   for (m, h), g in sorted(by_mh.items())}

    curve = {}
    for h in HS:
        g = by_mh[("v2g", h)]
        t_all = sum(r["delay_true_all"] for r in g)
        curve[f"H{h}"] = dict(
            n=len(g),
            delay_true_tracked_mean=mean(
                [r["delay_true_tracked"] for r in g]),
            delay_true_all_mean=mean([r["delay_true_all"] for r in g]),
            arrivals_mean=mean([r["n_arrivals"] for r in g]),
            untracked_delay_share=(sum(r["delay_true_all"]
                                       - r["delay_true_tracked"]
                                       for r in g) / t_all if t_all else 0.0),
        )

    free = {}
    for h in HS:
        g = by_mh[("v2g", h)]
        fp = [r["n_free_road_pred"] for r in g]
        ft = [r["n_free_road_true"] for r in g]
        free[f"H{h}"] = dict(
            n=len(g),
            n_free_road_assumed_mean=mean(fp),
            n_free_road_assumed_median=st.median(fp),
            n_free_road_assumed_max=max(fp),
            share_windows_with_any=sum(1 for x in fp if x > 0) / len(g),
            n_free_road_true_mean=mean(ft),
            n_free_road_true_max=max(ft),
        )

    sw_table = []
    for T, a in sorted({(r["T_s"], r["a_max"]) for r in sweep}):
        g = [r for r in sweep if abs(r["T_s"] - T) < 1e-9
             and abs(r["a_max"] - a) < 1e-9]
        sw_table.append((f"T{T}_a{a}", dict(
            kind="sweep", T_s=T, a_max=a, H=60, n=len(g),
            s_rmse_mean=mean([r["s_rmse"] for r in g]),
            v_rmse_mean=mean([r["v_rmse"] for r in g]),
            delay_err_mean=mean([r["delay_err"] for r in g]))))
    nom = by_mh[("v2g", 60)]
    sw_table.append(("T1.5_a1.0", dict(
        kind="nominal (registered v2g; NOT selected from the sweep)",
        T_s=1.5, a_max=1.0, H=60, n=len(nom),
        s_rmse_mean=mean([r["s_rmse"] for r in nom]),
        v_rmse_mean=mean([r["v_rmse"] for r in nom]),
        delay_err_mean=mean([r["delay_err"] for r in nom]))))

    def brk(key):
        out = {}
        for val, h in sorted({(r[key], int(r["H"])) for r in main
                              if r["model"] == "v2g"}):
            g = [r for r in main if r["model"] == "v2g"
                 and r[key] == val and int(r["H"]) == h]
            out[f"{val}|H{h}"] = dict(
                n=len(g), s_rmse_mean=mean([r["s_rmse"] for r in g]),
                delay_err_mean=mean([r["delay_err"] for r in g]),
                delay_err_mae=mae([r["delay_err"] for r in g]))
        return out

    cancel = {}
    for h in (60, 120):
        g = by_mh[("v2g", h)]
        errs = [r["delay_err"] for r in g]
        by_pen = {}
        for p in sorted({r["penetration"] for r in g}):
            gp = [r for r in g if r["penetration"] == p]
            by_pen[f"p{p}"] = dict(
                n=len(gp),
                delay_err_mean=mean([r["delay_err"] for r in gp]),
                delay_err_mae=mae([r["delay_err"] for r in gp]))
        cancel[f"H{h}"] = dict(
            n=len(g),
            n_nonnegative=sum(1 for e in errs if e >= 0),
            n_negative=sum(1 for e in errs if e < 0),
            delay_err_mean=mean(errs), delay_err_mae=mae(errs),
            by_penetration=by_pen)

    tables = dict(
        analysis_id="WD-01-PA2G-A",
        source="output/wd01/partA_v2g/results/eval_a/raw_rows.jsonl",
        raw_row_counts=dict(
            counts, main_rows_per_model_h={
                k: v["n"] for k, v in per_model_h.items()}),
        per_model_h={k: rnd(v) for k, v in per_model_h.items()},
        missing_arrivals={k: rnd(v) for k, v in curve.items()},
        n_free_road={k: rnd(v) for k, v in free.items()},
        sweep_T_a={k: rnd(v) for k, v in sw_table},
        by_penetration=rnd(brk("penetration")),
        by_demand=rnd(brk("demand")),
        h120_vs_h60_sign_diagnostics={
            k: rnd(v) for k, v in cancel.items()},
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "tables.json").write_text(
        json.dumps(tables, indent=1), encoding="utf-8")
    (RESULTS / "tables.md").write_text(render_md(tables), encoding="utf-8")
    print(render_md(tables))
    print(f"wrote {RESULTS / 'tables.json'} and tables.md")


def render_md(t: dict) -> str:
    L = ["# WD-01-PA2G-A aggregation tables (from saved raw rows)", ""]
    L.append("Raw rows: " + json.dumps(t["raw_row_counts"]))
    L.append("")
    L.append("## Table 1: per model x H (delay/TTC/hard in s; errors are "
             "pred - truth; mean SIGNED and mean ABSOLUTE both shown)")
    L.append("")
    L.append("| model|H | n | s_RMSE | v_RMSE | delay_err(mean) | "
             "delay_err(MAE) | ttc3_err(mean) | hard_err(mean) | "
             "LC merge p/t | LC disc p/t | coll/1000 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, v in t["per_model_h"].items():
        L.append(
            f"| {k} | {v['n']} | {v['s_rmse_mean']:.1f} | "
            f"{v['v_rmse_mean']:.2f} | {v['delay_err_mean']:.1f} | "
            f"{v['delay_err_mae']:.1f} | {v['ttc3_err_mean']:.1f} | "
            f"{v['hard_err_mean']:.1f} | "
            f"{v['lc_merge_pred']}/{v['lc_merge_true']} | "
            f"{v['lc_disc_pred']}/{v['lc_disc_true']} | "
            f"{v['collisions_per_1000']:.2f} |")
    L.append("")
    L.append("## Table 2: missing-arrivals curve (truth delay, v2g windows)")
    L.append("")
    L.append("| H | n | delay_true tracked | delay_true all | "
             "arrivals/window | untracked delay share |")
    L.append("|---|---|---|---|---|---|")
    for k, v in t["missing_arrivals"].items():
        L.append(f"| {k} | {v['n']} | {v['delay_true_tracked_mean']:.0f} | "
                 f"{v['delay_true_all_mean']:.0f} | "
                 f"{v['arrivals_mean']:.1f} | "
                 f"{v['untracked_delay_share']:.3f} |")
    L.append("")
    L.append("## Table 3: n_free_road_assumed stats (v2g windows)")
    L.append("")
    L.append("| H | n | assumed mean | median | max | share windows>0 | "
             "truth-flag mean | truth-flag max |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k, v in t["n_free_road"].items():
        L.append(f"| {k} | {v['n']} | {v['n_free_road_assumed_mean']:.2f} | "
                 f"{v['n_free_road_assumed_median']} | "
                 f"{v['n_free_road_assumed_max']} | "
                 f"{v['share_windows_with_any']:.2f} | "
                 f"{v['n_free_road_true_mean']:.2f} | "
                 f"{v['n_free_road_true_max']} |")
    L.append("")
    L.append("## Table 4: descriptive T/a sweep at H=60 (NO selection)")
    L.append("")
    L.append("| cell | kind | T | a_max | n | s_RMSE | v_RMSE | "
             "delay_err mean |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k, v in t["sweep_T_a"].items():
        L.append(f"| {k} | {v['kind']} | {v['T_s']} | {v['a_max']} | "
                 f"{v['n']} | {v['s_rmse_mean']:.1f} | "
                 f"{v['v_rmse_mean']:.2f} | {v['delay_err_mean']:.1f} |")
    L.append("")
    L.append("## Table 5: breakdowns (v2g): s_RMSE / delay_err mean / MAE")
    L.append("")
    for name in ("by_penetration", "by_demand"):
        L.append(f"### {name}")
        L.append("")
        L.append("| cell | n | s_RMSE | delay_err mean | delay_err MAE |")
        L.append("|---|---|---|---|---|")
        for k, v in t[name].items():
            L.append(f"| {k} | {v['n']} | {v['s_rmse_mean']:.1f} | "
                     f"{v['delay_err_mean']:.1f} | "
                     f"{v['delay_err_mae']:.1f} |")
        L.append("")
    L.append("## Table 6: H120 vs H60 sign diagnostics (v2g delay_err)")
    L.append("")
    L.append("| H | n | >=0 | <0 | mean signed | mean abs |")
    L.append("|---|---|---|---|---|---|")
    for k, v in t["h120_vs_h60_sign_diagnostics"].items():
        L.append(f"| {k} | {v['n']} | {v['n_nonnegative']} | "
                 f"{v['n_negative']} | {v['delay_err_mean']:.1f} | "
                 f"{v['delay_err_mae']:.1f} |")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
