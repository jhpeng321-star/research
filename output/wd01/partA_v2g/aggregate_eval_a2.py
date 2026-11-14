"""WD-01-PA2G-A2 close-out aggregation: registered tables from the SAVED
A2 raw rows (results/eval_a2/raw_rows.jsonl) + the labelled fixcheck
rows (results/eval_fixcheck/raw_rows.jsonl).

ZERO SIMULATION. No sweep-pass duplicates exist in A2 (the sweep and
guards_on passes do not re-emit const-speed rows).

Usage:
    python aggregate_eval_a2.py          # writes tables.json + tables.md
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "eval_a2"
FIXCHECK = HERE / "results" / "eval_fixcheck"
MODELS = ("v2g", "v1_p3o_replica", "const_speed")


def mean(xs):
    return st.fmean(xs) if xs else float("nan")


def mae(xs):
    return mean([abs(x) for x in xs])


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
        n_cav_mean=mean([r.get("n_cav", 0) for r in g]),
    )


def rnd(d: dict) -> dict:
    return {k: (round(v, 6) if isinstance(v, float) else v)
            for k, v in d.items()}


def load(path: Path) -> List[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    rows = load(RESULTS / "raw_rows.jsonl")
    meta = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    main_rows = [r for r in rows if r["model"] in MODELS]
    sweep = [r for r in rows if r["model"] == "v2g_sweep"]
    guards = [r for r in rows if r["model"] == "v2g_guards_on"]
    by_mh: Dict[tuple, List[dict]] = defaultdict(list)
    for r in main_rows:
        by_mh[(r["model"], int(r["H"]))].append(r)

    per_model_h = {f"{m}|H{h}": cell(g)
                   for (m, h), g in sorted(by_mh.items())}

    # labelled guards_on variant
    guards_on = {}
    for h in (10, 30, 60, 120):
        g = [r for r in guards if int(r["H"]) == h]
        if g:
            guards_on[f"H{h}"] = dict(
                n=len(g),
                s_rmse_mean=mean([r["s_rmse"] for r in g]),
                delay_err_mean=mean([r["delay_err"] for r in g]),
                delay_err_mae=mae([r["delay_err"] for r in g]),
                lc_disc_pred=sum(r["lc_pred"]["disc_left"]
                                 + r["lc_pred"]["disc_right"] for r in g),
                lc_disc_true=sum(r["lc_true"]["disc_left"]
                                 + r["lc_true"]["disc_right"] for r in g),
                collisions_per_1000=(
                    sum(r["collisions_pred"] for r in g) /
                    max(sum(r["vehicle_steps_pred"] for r in g) / 1000.0,
                        1e-9)))

    curve = {}
    for h in (10, 30, 60, 120):
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
    for h in (10, 30, 60, 120):
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
        for val, h in sorted({(r[key], int(r["H"])) for r in main_rows
                              if r["model"] == "v2g"}):
            g = [r for r in main_rows if r["model"] == "v2g"
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

    # fixcheck: fixed engine on the PA2G-A truth (fix attribution)
    fix = {}
    if FIXCHECK.exists():
        frows = [r for r in load(FIXCHECK / "raw_rows.jsonl")
                 if r["model"] in ("v2g_fixed", "v2g_fixed_guards_on")]
        for m in ("v2g_fixed", "v2g_fixed_guards_on"):
            fm = {}
            for h in (10, 30, 60, 120):
                g = [r for r in frows if r["model"] == m and int(r["H"]) == h]
                if g:
                    fm[f"H{h}"] = dict(
                        n=len(g),
                        s_rmse_mean=mean([r["s_rmse"] for r in g]),
                        delay_err_mean=mean([r["delay_err"] for r in g]),
                        delay_err_mae=mae([r["delay_err"] for r in g]),
                        lc_disc_pred=sum(
                            r["lc_pred"]["disc_left"]
                            + r["lc_pred"]["disc_right"] for r in g),
                        lc_disc_true=sum(
                            r["lc_true"]["disc_left"]
                            + r["lc_true"]["disc_right"] for r in g),
                        collisions_per_1000=(
                            sum(r["collisions_pred"] for r in g) /
                            max(sum(r["vehicle_steps_pred"] for r in g)
                                / 1000.0, 1e-9)))
            fix[m] = fm
        # paired: the SAME windows in the saved PA2G-A rows (pre-fix)
        pa2ga_path = HERE / "results" / "eval_a" / "raw_rows.jsonl"
        prows = [r for r in load(pa2ga_path) if r["model"] == "v2g"
                 and r["world"] in {r["world"] for r in frows}]
        idx = {(r["world"], r["w"], int(r["H"])): r for r in frows
               if r["model"] == "v2g_fixed"}
        pairs = [(r, idx[(r["world"], r["w"], int(r["H"]))])
                 for r in prows
                 if (r["world"], r["w"], int(r["H"])) in idx]
        paired = {}
        for h in (10, 30, 60, 120):
            gp = [(a, b) for a, b in pairs if int(a["H"]) == h]
            if not gp:
                continue
            paired[f"H{h}"] = dict(
                n=len(gp),
                disc_lc_pre=sum(a["lc_pred"]["disc_left"]
                                + a["lc_pred"]["disc_right"] for a, _ in gp),
                disc_lc_post=sum(b["lc_pred"]["disc_left"]
                                 + b["lc_pred"]["disc_right"] for _, b in gp),
                disc_lc_true=sum(a["lc_true"]["disc_left"]
                                 + a["lc_true"]["disc_right"]
                                 for a, _ in gp),
                coll_pre=sum(a["collisions_pred"] for a, _ in gp),
                coll_post=sum(b["collisions_pred"] for _, b in gp),
                s_rmse_pre=mean([a["s_rmse"] for a, _ in gp]),
                s_rmse_post=mean([b["s_rmse"] for _, b in gp]),
                delay_err_pre=mean([a["delay_err"] for a, _ in gp]),
                delay_err_post=mean([b["delay_err"] for _, b in gp]),
                delay_mae_pre=mae([a["delay_err"] for a, _ in gp]),
                delay_mae_post=mae([b["delay_err"] for _, b in gp]),
            )
        fix["paired_pre_post_same_windows"] = paired

    tables = dict(
        analysis_id="WD-01-PA2G-A2",
        source="output/wd01/partA_v2g/results/eval_a2/raw_rows.jsonl",
        raw_row_counts=dict(
            raw_lines=len(rows), main_rows=len(main_rows),
            sweep_rows=len(sweep), guards_on_rows=len(guards),
            main_rows_per_model_h={k: v["n"] for k, v in per_model_h.items()},
            run_meta={k: meta[k] for k in (
                "n_usable", "skipped_worlds", "stratified", "n_worlds_run",
                "workers", "param_set_id", "wall_clock_s")}),
        per_model_h={k: rnd(v) for k, v in per_model_h.items()},
        guards_on_labelled={k: rnd(v) for k, v in guards_on.items()},
        fixcheck_on_pa2ga_truth={k: {h: rnd(v) for h, v in fm.items()}
                                 for k, fm in fix.items()},
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
    L = ["# WD-01-PA2G-A2 aggregation tables (from saved raw rows)", ""]
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
    L.append("## Table G: labelled v2g_guards_on variant "
             "(design-deviation comparison; NOT headline)")
    L.append("")
    L.append("| H | n | s_RMSE | delay_err mean | delay_err MAE | "
             "LC disc p | LC disc t | coll/1000 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k, v in t["guards_on_labelled"].items():
        L.append(f"| {k} | {v['n']} | {v['s_rmse_mean']:.1f} | "
                 f"{v['delay_err_mean']:.1f} | {v['delay_err_mae']:.1f} | "
                 f"{v['lc_disc_pred']} | {v['lc_disc_true']} | "
                 f"{v['collisions_per_1000']:.2f} |")
    L.append("")
    L.append("## Table F: fixcheck — the FIXED engine on the PA2G-A "
             "chunk-2/3 truth (labelled; same no-CAV-flag treatment as "
             "PA2G-A, so the delta vs the saved PA2G-A v2g rows is the "
             "engine fix)")
    L.append("")
    for m, fm in t["fixcheck_on_pa2ga_truth"].items():
        if m == "paired_pre_post_same_windows":
            continue
        L.append(f"### {m}")
        L.append("")
        L.append("| H | n | s_RMSE | delay_err mean | delay_err MAE | "
                 "LC disc p | LC disc t | coll/1000 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for k, v in fm.items():
            L.append(f"| {k} | {v['n']} | {v['s_rmse_mean']:.1f} | "
                     f"{v['delay_err_mean']:.1f} | "
                     f"{v['delay_err_mae']:.1f} | "
                     f"{v['lc_disc_pred']} | {v['lc_disc_true']} | "
                     f"{v['collisions_per_1000']:.2f} |")
        L.append("")
    pp = t["fixcheck_on_pa2ga_truth"].get("paired_pre_post_same_windows")
    if pp:
        L.append("### paired: PA2G-A v2g (pre-fix) vs v2g_fixed, the SAME "
                 "windows (fix effect isolated; chunk-2/3 truth)")
        L.append("")
        L.append("| H | n windows | disc LC pre | disc LC post | disc LC "
                 "true | coll pre | coll post | s_RMSE pre | s_RMSE post "
                 "| delay_err pre (mean/MAE) | delay_err post (mean/MAE) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for k, v in pp.items():
            L.append(
                f"| {k} | {v['n']} | {v['disc_lc_pre']} | "
                f"{v['disc_lc_post']} | {v['disc_lc_true']} | "
                f"{v['coll_pre']} | {v['coll_post']} | "
                f"{v['s_rmse_pre']:.1f} | {v['s_rmse_post']:.1f} | "
                f"{v['delay_err_pre']:.1f}/{v['delay_mae_pre']:.1f} | "
                f"{v['delay_err_post']:.1f}/{v['delay_mae_post']:.1f} |")
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
