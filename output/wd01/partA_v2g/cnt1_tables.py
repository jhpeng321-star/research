"""WD-01-PA2G-CNT1 tables: registered aggregations, world-bootstrap
90 % CIs and the section-5 decision rule, derived from the SAVED raw
rows (results/eval_cnt1/raw_rows.jsonl; the eval is never re-run).

Outputs:
  results/eval_cnt1/tables.json  (machine-readable, all of the below)
  results/eval_cnt1/tables.md    (report tables)
  stdout summary incl. the WORTH IT / PARTIAL / NOT WORTH IT verdict

Registered conventions:
- primary metric: ALL-vehicle window delay error (pred - truth), signed
  mean and MAE; MAE also reported as % of the mean all-vehicle TRUE
  delay (the pct_partA convention MAE / mean true delay, all-vehicle).
- uncertainty: world bootstrap (B = 2000, seed 20260930, worlds
  resampled with replacement; pooled window statistics per resample),
  90 % percentile CIs on the v2g_cnt - v2g differences.
- decision rule (REGISTRATION_PA2G_CNT1.md section 5) applied verbatim;
  collisions clause interpreted at EVERY H (conservative) and the per-H
  table is printed so the reader can check any other reading.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "results" / "eval_cnt1" / "raw_rows.jsonl"
POST_LC1 = HERE / "results" / "eval_a2_post_lc1" / "summary.json"
OUT_DIR = HERE / "results" / "eval_cnt1"
MODELS = ("v2g_cnt", "v2g", "const_speed", "v1_p3o_replica")
HS = (10, 30, 60, 120)
B = 2000
SEED = 20260930


def per_mh_table(df):
    """Per model x H aggregates from raw rows."""
    out = {}
    for (m, H), g in df.groupby(["model", "H"]):
        out[f"{m}|H{int(H)}"] = dict(
            n=int(len(g)),
            n_worlds=int(g.world.nunique()),
            delay_all_err_mean=float(g.delay_all_err.mean()),
            delay_all_err_mae=float(g.delay_all_err.abs().mean()),
            delay_all_mae_pct=100.0 * float(g.delay_all_err.abs().mean())
            / float(g.delay_true_all.mean()),
            delay_true_all_mean=float(g.delay_true_all.mean()),
            delay_err_mean=float(g.delay_err.mean()),
            delay_err_mae=float(g.delay_err.abs().mean()),
            delay_true_tracked_mean=float(g.delay_true_tracked.mean()),
            s_rmse_mean=float(g.s_rmse.mean()),
            v_rmse_mean=float(g.v_rmse.mean()),
            ttc3_all_err_mean=float(g.ttc3_all_err.mean()),
            ttc3_all_err_mae=float(g.ttc3_all_err.abs().mean()),
            ttc3_err_mean=float(g.ttc3_err.mean()),
            hard_all_err_mean=float(g.hard_all_err.mean()),
            hard_all_err_mae=float(g.hard_all_err.abs().mean()),
            hard_err_mean=float(g.hard_err.mean()),
            collisions_pred=int(g.collisions_pred.sum()),
            vehicle_steps_pred=int(g.vehicle_steps_pred.sum()),
            collisions_per_1000=float(
                g.collisions_pred.sum()
                / max(g.vehicle_steps_pred.sum() / 1000.0, 1e-9)),
            n_arrivals_true_mean=float(g.n_arrivals.mean()),
            n_injected_mean=float(g.n_injected.mean()),
            n_injected_scheduled_mean=float(
                g.n_injected_scheduled.mean()),
        )
    return out


def world_bootstrap(df, model_a="v2g_cnt", model_b="v2g", B=B, seed=SEED):
    """90 % percentile CIs (worlds resampled with replacement) for the
    differences a - b on the pooled windows: all-vehicle delay MAE,
    relative MAE change, mean signed all-vehicle error, mean s_RMSE,
    coll/1000.  Returns {H: {metric: (lo, point, hi)}}."""
    rng = np.random.default_rng(seed)
    worlds = sorted(set(df.world))
    per_h = {}
    for H in HS:
        ga = df[(df.model == model_a) & (df.H == H)].set_index(
            ["world", "w"]).sort_index()
        gb = df[(df.model == model_b) & (df.H == H)].set_index(
            ["world", "w"]).sort_index()
        assert (ga.index == gb.index).all(), "unpaired windows"
        err_a = ga.delay_all_err.abs().to_numpy()
        err_b = gb.delay_all_err.abs().to_numpy()
        sign_a = ga.delay_all_err.to_numpy()
        sign_b = gb.delay_all_err.to_numpy()
        sr_a = ga.s_rmse.to_numpy()
        sr_b = gb.s_rmse.to_numpy()
        coll_a = ga.collisions_pred.to_numpy()
        coll_b = gb.collisions_pred.to_numpy()
        vs_a = ga.vehicle_steps_pred.to_numpy()
        vs_b = gb.vehicle_steps_pred.to_numpy()
        w_of = ga.index.get_level_values(0).to_numpy()
        w_unique = np.array(worlds)
        w_idx = np.searchsorted(w_unique, w_of)
        stats = dict(mae_diff=[], mae_rel=[], sign_diff=[], srmse_diff=[],
                     coll_diff=[])
        for _ in range(B):
            pick = rng.choice(len(w_unique), size=len(w_unique),
                              replace=True)
            reps = np.zeros(len(w_unique), dtype=int)
            for p in pick:
                reps[p] += 1
            mult = reps[w_idx].astype(float)
            ma = float((err_a * mult).sum() / mult.sum())
            mb = float((err_b * mult).sum() / mult.sum())
            stats["mae_diff"].append(ma - mb)
            stats["mae_rel"].append((ma - mb) / mb if mb > 0 else np.nan)
            stats["sign_diff"].append(
                float((sign_a * mult).sum() - (sign_b * mult).sum())
                / mult.sum())
            stats["srmse_diff"].append(
                float((sr_a * mult).sum() - (sr_b * mult).sum())
                / mult.sum())
            stats["coll_diff"].append(
                float((coll_a * mult).sum()
                      / max((vs_a * mult).sum() / 1000.0, 1e-9))
                - float((coll_b * mult).sum()
                        / max((vs_b * mult).sum() / 1000.0, 1e-9)))
        point = dict(
            mae_diff=float((err_a - err_b).mean()),
            mae_rel=float((err_a.sum() - err_b.sum()) / err_b.sum()),
            sign_diff=float((sign_a - sign_b).mean()),
            srmse_diff=float((sr_a - sr_b).mean()),
            coll_diff=float(coll_a.sum() / max(vs_a.sum() / 1000.0, 1e-9)
                            - coll_b.sum()
                            / max(vs_b.sum() / 1000.0, 1e-9)))
        per_h[int(H)] = {}
        for k, vals in stats.items():
            arr = np.asarray(vals, dtype=float)
            lo, hi = np.nanpercentile(arr, [5.0, 95.0])
            per_h[int(H)][k] = dict(lo=float(lo), point=point[k],
                                    hi=float(hi))
    return per_h


def decision_rule(tbl, boot):
    """Section 5 applied verbatim.  Clause (4) is evaluated at every H
    (conservative reading; per-H values reported)."""
    rel = {H: boot[H]["mae_rel"] for H in HS}
    c1 = {}
    for H in (60, 120):
        c1[H] = bool(rel[H]["point"] <= -0.25 and rel[H]["hi"] < 0.0)
    c2 = {H: bool(rel[H]["point"] <= 0.05) for H in (10, 30)}
    c3 = {}
    for H in (10, 30):
        sc = tbl[f"v2g_cnt|H{H}"]["s_rmse_mean"]
        sv = tbl[f"v2g|H{H}"]["s_rmse_mean"]
        c3[H] = bool((sc - sv) / sv <= 0.05)
    c4 = {H: bool(tbl[f"v2g_cnt|H{H}"]["collisions_per_1000"]
                  <= tbl[f"v2g|H{H}"]["collisions_per_1000"] + 0.1)
          for H in HS}
    worth = all(c1.values()) and all(c2.values()) and all(c3.values()) \
        and all(c4.values())
    partial = (sum(c1.values()) == 1) and all(c2.values()) \
        and all(c3.values()) and all(c4.values())
    verdict = "WORTH IT" if worth else ("PARTIAL" if partial
                                        else "NOT WORTH IT")
    return dict(clause1=c1, clause2=c2, clause3=c3, clause4=c4,
                verdict=verdict,
                label="sensitivity; requires counter hardware; does not "
                      "change the CAV-strict headline")


def post_lc1_crosscheck(tbl):
    """Comparators vs the frozen post-LC1 A2 summary (bit-identity at
    the aggregate level; the post-LC1 raw rows are not archived)."""
    try:
        ref = json.loads(POST_LC1.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return dict(available=False)
    ref_mh = ref["summary"]["per_model_h"]
    out = {}
    for key in sorted(ref_mh):
        if key not in tbl:
            continue
        r, m = ref_mh[key], tbl[key]
        diffs = {}
        for field in ("n", "s_rmse_mean", "v_rmse_mean", "delay_err_mean",
                      "delay_err_mae"):
            diffs[field] = m[field] - r[field]
        out[key] = diffs
    maxabs = max(abs(v) for d in out.values()
                 for k, v in d.items() if k != "n")
    return dict(available=True, per_key=out,
                max_abs_float_diff=maxabs,
                all_n_match=all(d["n"] == 0 for d in out.values()))


def main():
    df = pd.read_json(RAW, lines=True)
    assert set(df.model.unique()) <= set(MODELS), df.model.unique()
    tbl = per_mh_table(df)
    boot = world_bootstrap(df)
    rule = decision_rule(tbl, boot)
    cross = post_lc1_crosscheck(tbl)
    # injected vs true arrival counts per window (panel means)
    arr = {}
    for H in HS:
        g = df[(df.model == "v2g_cnt") & (df.H == H)]
        arr[int(H)] = dict(
            n=len(g),
            true_mean=float(g.n_arrivals.mean()),
            injected_mean=float(g.n_injected.mean()),
            scheduled_mean=float(g.n_injected_scheduled.mean()),
            true_sum=int(g.n_arrivals.sum()),
            injected_sum=int(g.n_injected.sum()))
    out = dict(per_model_h=tbl, bootstrap_cnt_minus_v2g=boot,
               decision_rule=rule, injected_vs_true_arrivals=arr,
               post_lc1_crosscheck=cross,
               raw_rows=str(RAW), B=B, seed=SEED)
    (OUT_DIR / "tables.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    lines = ["# WD-01-PA2G-CNT1 tables (derived from saved raw rows)", ""]
    hdr = ("| model|H | n | delay_all_err mean | MAE | MAE % of true | "
           "delay_err tracked mean/MAE | s_RMSE | v_RMSE | ttc3_all_err "
           "mean | hard_all_err mean | coll/1000 | true arr | inj arr |")
    lines += [hdr, "|" + "---|" * 13]
    for m in MODELS:
        for H in HS:
            t = tbl[f"{m}|H{int(H)}"]
            lines.append(
                f"| {m}|H{H} | {t['n']} | {t['delay_all_err_mean']:.1f} | "
                f"{t['delay_all_err_mae']:.1f} | "
                f"{t['delay_all_mae_pct']:.1f} % | "
                f"{t['delay_err_mean']:.1f}/{t['delay_err_mae']:.1f} | "
                f"{t['s_rmse_mean']:.1f} | {t['v_rmse_mean']:.2f} | "
                f"{t['ttc3_all_err_mean']:.1f} | "
                f"{t['hard_all_err_mean']:.1f} | "
                f"{t['collisions_per_1000']:.3f} | "
                f"{t['n_arrivals_true_mean']:.1f} | "
                f"{t['n_injected_mean']:.1f} |")
    lines.append("")
    lines.append("## v2g_cnt - v2g world-bootstrap 90 % CIs "
                 f"(B={B}, seed={SEED})")
    lines.append("")
    lines.append("| H | MAE diff | MAE rel change | signed diff | s_RMSE "
                 "diff | coll/1000 diff |")
    lines.append("|---|---|---|---|---|---|")
    for H in HS:
        b = boot[H]
        def fmt(k):
            d = b[k]
            return f"{d['point']:+.1f} [{d['lo']:+.1f}, {d['hi']:+.1f}]" \
                if k != "mae_rel" else \
                f"{100 * d['point']:+.1f} % " \
                f"[{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]"
        lines.append(f"| H{H} | {fmt('mae_diff')} | {fmt('mae_rel')} | "
                     f"{fmt('sign_diff')} | {fmt('srmse_diff')} | "
                     f"{fmt('coll_diff')} |")
    lines.append("")
    lines.append("## Decision rule (section 5, applied verbatim)")
    lines.append("")
    lines.append(f"- (1) H60: rel {100 * boot[60]['mae_rel']['point']:+.1f} % "
                 f"CI [{100 * boot[60]['mae_rel']['lo']:+.1f}, "
                 f"{100 * boot[60]['mae_rel']['hi']:+.1f}] -> "
                 f"{rule['clause1'][60]}; H120: "
                 f"{100 * boot[120]['mae_rel']['point']:+.1f} % "
                 f"CI [{100 * boot[120]['mae_rel']['lo']:+.1f}, "
                 f"{100 * boot[120]['mae_rel']['hi']:+.1f}] -> "
                 f"{rule['clause1'][120]}")
    lines.append(f"- (2) not-worse-than-5 % at H10/H30: {rule['clause2']}")
    lines.append(f"- (3) tracked s_RMSE not-worse-than-5 % at H10/H30: "
                 f"{rule['clause3']}")
    lines.append(f"- (4) collisions <= v2g + 0.1/1000 at every H: "
                 f"{rule['clause4']}")
    lines.append(f"- **VERDICT: {rule['verdict']}**")
    lines.append(f"- Label: {rule['label']}")
    (OUT_DIR / "tables.md").write_text("\n".join(lines) + "\n",
                                       encoding="utf-8")
    print("\n".join(lines))
    print("\npost-LC1 A2 cross-check:",
          json.dumps({k: v for k, v in cross.items() if k != "per_key"},
                     default=str))
    print("DONE cnt1_tables")


if __name__ == "__main__":
    main()
