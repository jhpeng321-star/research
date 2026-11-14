"""Command-line interface for the WM-N1 combination layer.

Only synthetic data is generated here.  No real dataset root, no SUMO,
no training; HOLDOUT and chunk-5 TEST are never touched.

    python -m combine version
    python -m combine synthetic --out <dir> [--seed 0] [--n-train 8]
                                 [--n-cal 4] [--k 5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _synthetic_pipeline(out_dir, seed: int, n_train: int, n_cal: int,
                        k: int) -> dict:
    from .crps import QuantilePred
    from .disagreement import abstain_curve, disagreement, spearman_abs_error
    from .oof_tree import build_oof_predictions
    from .select_variant import (DEFAULT_CONTRASTS, write_planner_variant)
    from .stacking import (DEFAULT_QUANTILES, fit_stack, interval_coverage)
    from .synthetic import SyntheticQuantileModel, synthetic_rows

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = synthetic_rows(n_train=n_train, n_cal=n_cal, seed=seed)

    oof = build_oof_predictions(rows, lambda: SyntheticQuantileModel(), k=k,
                                targets=["delay"])
    oof_paths = oof.save(out / "oof_tree")
    assert oof.table["model_kind"].isin(["oof", "full"]).all()
    n_oof = int((oof.table["model_kind"] == "oof").sum())
    assert n_oof == int((rows["split"] == "TRAIN").sum())

    cal = rows[rows["split"] == "CALIBRATION"].reset_index(drop=True)
    y = cal["y"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed + 1)
    lv = np.asarray(DEFAULT_QUANTILES, dtype=np.float64)
    from scipy.stats import norm
    z = norm.ppf(lv)
    sigma_n = 0.1 * cal["tau"].to_numpy(dtype=np.float64)
    sigma_t = 0.35 * cal["tau"].to_numpy(dtype=np.float64)
    fn = QuantilePred(lv, y[:, None] + rng.normal(0, 5, (len(y), 1))
                      + z[None, :] * sigma_n[:, None])
    ft = QuantilePred(lv, 0.85 * y[:, None] + rng.normal(0, 5, (len(y), 1))
                      + z[None, :] * sigma_t[:, None])

    stack, xf = fit_stack(fn, ft, y, cal["world"].to_numpy(),
                          cal["split"].to_numpy(), levels=lv, folds=2)
    pooled = stack.pool_quantiles(fn, ft)
    recal = stack.predict(fn, ft)
    cov_before = interval_coverage(pooled, y, levels=lv)
    cov_after = interval_coverage(recal, y, levels=lv)

    scale = float(np.median(np.abs(y))) or 1.0
    d = disagreement(fn, ft, y=y, scale=scale)
    median = pooled[:, int(np.argmin(np.abs(lv - 0.5)))]
    abs_err = np.abs(y - median)
    spear = spearman_abs_error(d, abs_err)
    abst = abstain_curve(d, abs_err)
    abst_path = out / "abstain_curve.csv"
    abst.to_csv(abst_path, index=False)

    crps_scores = {
        "T": {60.0: {c: 1.20 for c in DEFAULT_CONTRASTS},
              120.0: {c: 1.30 for c in DEFAULT_CONTRASTS},
              300.0: {c: 1.45 for c in DEFAULT_CONTRASTS}},
        "A": {60.0: {c: 1.10 for c in DEFAULT_CONTRASTS},
              120.0: {c: 1.18 for c in DEFAULT_CONTRASTS},
              300.0: {c: 1.28 for c in DEFAULT_CONTRASTS}},
        "C": {60.0: {c: 1.05 for c in DEFAULT_CONTRASTS},
              120.0: {c: 1.12 for c in DEFAULT_CONTRASTS},
              300.0: {c: 1.22 for c in DEFAULT_CONTRASTS}},
        "N": {60.0: {c: 1.02 for c in DEFAULT_CONTRASTS},
              120.0: {c: 1.10 for c in DEFAULT_CONTRASTS},
              300.0: {c: 1.30 for c in DEFAULT_CONTRASTS}},
    }
    coverage = {"T": 0.80, "A": 0.80, "C": 0.79, "N": 0.74}
    from .select_variant import select_planner_variant
    chosen = select_planner_variant(crps=crps_scores, coverage=coverage)
    sha = write_planner_variant(out / "PLANNER_VARIANT.json", chosen)

    summary = dict(
        seed=seed, n_train=n_train, n_cal=n_cal, k=oof.k,
        n_oof_rows=n_oof, oof_files=oof_paths,
        pool_weight=float(stack.weight),
        cross_fit_folds=[f["fold"] for f in xf["folds"]],
        mean_oof_crps=xf["mean_oof_crps"],
        coverage_before=float(cov_before), coverage_after=float(cov_after),
        disagreement_mean=float(np.mean(d)),
        spearman_rho=spear["rho"], spearman_p=spear["pvalue"],
        abstain_file=str(abst_path),
        planner_variant=chosen["chosen"], planner_variant_sha=sha)
    (out / "summary.json").write_text(json.dumps(summary, indent=1),
                                      encoding="utf-8")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="combine",
        description="WM-N1 combination layer (synthetic only; no SUMO)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("version", help="print the package version")
    p = sub.add_parser("synthetic", help="run the synthetic smoke pipeline")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-train", type=int, default=8)
    p.add_argument("--n-cal", type=int, default=4)
    p.add_argument("--k", type=int, default=5)
    args = parser.parse_args(argv)

    if args.cmd == "version":
        from . import __version__
        print(json.dumps(dict(package="wm_n1.combine",
                              version=__version__)))
        return 0
    summary = _synthetic_pipeline(args.out, args.seed, args.n_train,
                                  args.n_cal, args.k)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":                       # pragma: no cover
    sys.exit(main())
