"""WD-01-PB-RESCORE1 Q1: label audit on the chunk-2/3 TRUE references.

Per TRAIN/CAL world (label root, TRAIN/CAL by construction) and per
tau in {30, 60, 120, 300, 570}, the realised remaining delay Y under:

  damping      the campaign incumbent damping base arm  at t = t0
  switching    the mislabelled "__native" switching arm at t = t0 + 10 s
  pi_arm       the "__pi" switching arm                at t = t0 + 10 s
  true_native  the TRUE fleet native (J3/WM-73)        at t = t0 + 10 s
  true_pi      the TRUE fleet PI (J3/WM-73)            at t = t0 + 10 s

(the +10 s offset is the v0.3.2-eval convention: branch decision grids
start at t0 + 10 s; same saved state at t0 -- same 10 s window-offset
caveat as the earlier reports).

Contrasts (positive = first arm has MORE remaining delay):
  d_switch_native = Y_switch - Y_true_native
  d_pi_native     = Y_true_pi - Y_true_native
  d_damp_native   = Y_damping - Y_true_native      (TRUE damping contrast)
  d_damp_switch   = Y_damping - Y_switch           (the OLD report contrast)
plus the supplementary d_piswitch_native = Y_pi_arm - Y_true_native.

Outputs (output/wd01/partB_rescore1/):
  q1_raw_rows.csv   one row per (world, tau) BEFORE aggregation
  q1_tables.json    per-tau + per-(demand x penetration) aggregates and
                    old-vs-true damping-contrast sign agreement
  q1_tables.md      report fragment
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

SOURCES = ("base", "native", "pi", "truearms_native", "truearms_pi")


def collect(root: Path, cache: dict, man: dict) -> pd.DataFrame:
    rows = []
    for chunk_id, world, split, dem, pen, t0 in C.world_rows(man):
        frames = {s: C.label_frame(root, chunk_id, world, s, cache)
                  for s in SOURCES}
        for tau in C.TAUS:
            y = {}
            for s in SOURCES:
                t = t0 if s == "base" else t0 + C.T_OFFSET_S
                y[s] = C.y_at(frames[s], t, tau)
            if any(v is None for v in y.values()):
                continue
            d_sw = y["native"] - y["truearms_native"]
            d_pi = y["truearms_pi"] - y["truearms_native"]
            d_dn = y["base"] - y["truearms_native"]
            d_ds = y["base"] - y["native"]
            d_ps = y["pi"] - y["truearms_native"]
            rows.append(dict(
                chunk=chunk_id, world=world, split=split,
                demand_veh_h=dem, penetration=pen, t0_s=t0, tau=tau,
                y_damping=y["base"], y_switch=y["native"], y_pi_arm=y["pi"],
                y_true_native=y["truearms_native"],
                y_true_pi=y["truearms_pi"],
                d_switch_native=d_sw, d_pi_native=d_pi,
                d_damp_native=d_dn, d_damp_switch=d_ds,
                d_piswitch_native=d_ps,
                sign_agree_old_true=bool(
                    np.sign(d_ds) == np.sign(d_dn))))
    return pd.DataFrame(rows)


CONTRASTS = ("d_switch_native", "d_pi_native", "d_damp_native",
             "d_damp_switch", "d_piswitch_native")


def tables(df: pd.DataFrame) -> dict:
    out: dict = {"per_tau": [], "per_cell": [], "sign_agreement_per_tau": [],
                 "old_vs_true_correlation_per_tau": []}
    for tau in C.TAUS:
        g = df[np.isclose(df["tau"], tau)]
        if g.empty:
            continue
        row = dict(tau=tau,
                   n_worlds=int(len(g)),
                   **{f"{c}": C.agg(g[c].to_numpy()) for c in CONTRASTS})
        out["per_tau"].append(row)
        ok = g["sign_agree_old_true"].to_numpy().astype(bool)
        worlds = g["world"].tolist()
        lo, hi = C.world_bootstrap_ci(worlds, ok.astype(float))
        out["sign_agreement_per_tau"].append(dict(
            tau=tau, n=int(len(g)),
            sign_agreement=C.r2(float(ok.mean()), 3),
            ci90_lo=C.r2(lo, 3), ci90_hi=C.r2(hi, 3),
            spearman_old_vs_true=C.r2(
                C._spearman(g["d_damp_switch"].to_numpy(),
                            g["d_damp_native"].to_numpy()), 3)))
        for (dem, pen), gc in g.groupby(["demand_veh_h", "penetration"]):
            out["per_cell"].append(dict(
                tau=tau, demand_veh_h=int(dem), penetration=float(pen),
                n=int(len(gc)),
                **{c: C.agg(gc[c].to_numpy()) for c in CONTRASTS},
                sign_agreement_old_true=C.r2(
                    float(gc["sign_agree_old_true"].astype(bool).mean()),
                    3)))
    return out


def markdown(t: dict) -> str:
    L = ["## Q1 tables", "",
         "Realised remaining-delay contrasts on the chunk-2/3 TRUE "
         "references (veh-s; positive = the first arm of the contrast has "
         "MORE remaining delay).  Branch arms are read at t0 + 10 s "
         "(v0.3.2-eval convention; same saved state at t0, 10 s "
         "window-offset caveat).", "",
         "### Per tau (all 112 TRAIN/CAL worlds)", "",
         "| tau | n | d_switch-y_native mean | median | d_pi(true)-y_native "
         "mean | median | d_damping-y_native mean | median | "
         "d_damping-y_switch (old) mean | median |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for row in t["per_tau"]:
        L.append(
            f"| {row['tau']:.0f} | {row['n_worlds']} "
            f"| {row['d_switch_native']['mean']:.1f} "
            f"| {row['d_switch_native']['median']:.1f} "
            f"| {row['d_pi_native']['mean']:.1f} "
            f"| {row['d_pi_native']['median']:.1f} "
            f"| {row['d_damp_native']['mean']:.1f} "
            f"| {row['d_damp_native']['median']:.1f} "
            f"| {row['d_damp_switch']['mean']:.1f} "
            f"| {row['d_damp_switch']['median']:.1f} |")
    L += ["", "### Old-vs-true damping-contrast sign agreement "
              "(d_damping-y_switch vs d_damping-y_native)", "",
          "| tau | n | sign agreement | world-bootstrap 90% CI | "
          "Spearman(old, true) |", "|---|---|---|---|---|"]
    for row in t["sign_agreement_per_tau"]:
        L.append(
            f"| {row['tau']:.0f} | {row['n']} | {row['sign_agreement']:.3f} "
            f"| [{row['ci90_lo']:.3f}, {row['ci90_hi']:.3f}] "
            f"| {row['spearman_old_vs_true']} |")
    L += ["", "### Per demand x penetration cell (d_damping-y_native "
              "mean, and the old/true sign agreement)", "",
          "| tau | demand | penetration | n | d_switch-native mean "
          "| d_pi-native mean | d_damping-native mean | old contrast mean "
          "| sign agreement |", "|---|---|---|---|---|---|---|---|---|"]
    for row in t["per_cell"]:
        L.append(
            f"| {row['tau']:.0f} | {row['demand_veh_h']} "
            f"| {row['penetration']:.2f} | {row['n']} "
            f"| {row['d_switch_native']['mean']:.1f} "
            f"| {row['d_pi_native']['mean']:.1f} "
            f"| {row['d_damp_native']['mean']:.1f} "
            f"| {row['d_damp_switch']['mean']:.1f} "
            f"| {row['sign_agreement_old_true']:.2f} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(C.LABEL_ROOT))
    args = ap.parse_args(argv)
    root = Path(args.root)
    man = C.load_label_manifest(root)
    cache: dict = {}
    df = collect(root, cache, man)
    assert len(df), "no Q1 pairs collected"
    t = tables(df)
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.RESULTS / "q1_raw_rows.csv", index=False)
    C.write_json(C.RESULTS / "q1_tables.json", t)
    (C.RESULTS / "q1_tables.md").write_text(markdown(t), encoding="utf-8")
    n_w = df["world"].nunique()
    print(f"Q1: {len(df)} (world, tau) rows over {n_w} worlds; "
          f"raw + tables written to {C.RESULTS}")
    for row in t["sign_agreement_per_tau"]:
        print(f"  tau={row['tau']:.0f}: sign agreement old-vs-true "
              f"{row['sign_agreement']:.3f} "
              f"(90% CI [{row['ci90_lo']:.3f}, {row['ci90_hi']:.3f}])")
    return 0


if __name__ == "__main__":
    sys.exit(main())
