"""CLI: run the full STEP 3 offline analysis on an EXAM v1 dataset directory.

Usage (repo root, rl_driving interpreter):
  python -m tools.coopbandit_learn.cli --data <exam-export-dir> --out <results-dir>

Produces: CV comparison (GBT vs linear baseline), policy evaluation (OOF
regret vs oracle, headroom, world-bootstrap CIs), importance/PDP tables and
figures, the stageable policy.json artifact, and SUMMARY.md. Partial
(completion-order) datasets run fine but are labelled exploratory.
"""
import argparse
import json
from pathlib import Path
import sys

from . import artifact as artifact_mod
from . import interpret, loader, policy
from .cv import run as cv_run
from .schema import checked


def analyze(data_dir, out_dir, kinds=("gbt", "linear"), horizon=None, plots=True, seed_data=None):
    if not Path(data_dir).is_dir():
        raise SystemExit("dataset directory not found: " + str(data_dir))
    dataset = loader.load(data_dir)
    out = checked(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    horizon = horizon or 600
    report = dict(data_dir=str(data_dir), states=len(dataset.states), worlds=dataset.states.world_id.nunique(),
                  partial=dataset.partial, horizon_s=horizon, kinds={})
    if dataset.partial:
        print("NOTE: partial (completion-order) dataset - results are exploratory, "
              "not final panel estimates.", file=sys.stderr)
    results, evaluations = {}, {}
    for kind in kinds:
        print("fitting world-grouped CV:", kind, file=sys.stderr)
        result = cv_run(dataset, horizon=horizon, kind=kind)
        evaluation = policy.evaluate(dataset, result, horizon=horizon)
        results[kind], evaluations[kind] = result, evaluation
        report["kinds"][kind] = dict(
            oof_rmse=float((result.oof.residual ** 2).mean() ** 0.5),
            radius_full=result.radius_full,
            fold_radius={str(k): float(v) for k, v in result.fold_radius.items()},
            seconds=result.seconds,
            summary=evaluation.summary, baseline=evaluation.baseline_summary)
        if kind == "gbt":
            imp = interpret.importance_table(result)
            X = dataset.feature_matrix(list(dataset.states.state_id))
            pdp = interpret.partial_dependence_table(result, X)
            interpret.write_tables(out, imp, pdp)
            if plots:
                interpret.plots(result, imp, pdp, out)
            digest = artifact_mod.write(artifact_mod.build(result, source=dataset.provenance.get("design_sha256")),
                                        out / "policy.json")
            report["policy_sha256"] = digest
    primary = evaluations["gbt"]
    report["per_cell"] = policy.per_cell(dataset, primary)
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (out / "SUMMARY.md").write_text(_summary_md(report), encoding="utf-8")
    return report


def _summary_md(report):
    g, l = report["kinds"]["gbt"], report["kinds"]["linear"]
    lines = [
        "# S-COOPBANDIT-LEARN STEP 3 summary", "",
        f"- dataset: {report['states']} states / {report['worlds']} worlds "
        f"({'PARTIAL - exploratory' if report['partial'] else 'complete panel'})",
        "- rule: argmax predicted D among available options, executed only if predicted D - radius > 0, else O0; "
        "radius = finite-sample 95% quantile of per-world max signed overprediction (world-grouped 5-outer/4-inner CV); "
        "no behavioural threshold.", "",
        "| model | OOF RMSE (veh-s) | full radius (veh-s) | learned D mean | oracle headroom | learned regret | cooperate rate |",
        "|---|---|---|---|---|---|---|",
        f"| GBT | {g['oof_rmse']:.2f} | {g['radius_full']:.2f} | {g['summary']['learned']['mean']:.2f} "
        f"| {g['summary']['oracle']['mean']:.2f} | {g['summary']['regret']['mean']:.2f} "
        f"| {g['summary']['cooperate']['mean']:.3f} |",
        f"| linear | {l['oof_rmse']:.2f} | {l['radius_full']:.2f} | {l['summary']['learned']['mean']:.2f} "
        f"| {l['summary']['oracle']['mean']:.2f} | {l['summary']['regret']['mean']:.2f} "
        f"| {l['summary']['cooperate']['mean']:.3f} |",
        "",
        f"95% world-bootstrap CIs (GBT regret): "
        f"[{g['summary']['regret']['ci95'][0]:.2f}, {g['summary']['regret']['ci95'][1]:.2f}]; "
        f"always-O0 baseline regret equals headroom {g['baseline']['regret']['mean']:.2f}.",
        "",
        "Per-cell descriptive regret/headroom and feature diagnostics: report.json, "
        "feature_importance.csv/png, partial_dependence.csv/png. Live-test callable: "
        "policy.json + tools/coopbandit_learn/runtime.py (no sklearn needed remotely).",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="EXAM v1 export directory (exam or exam-partial)")
    parser.add_argument("--out", required=True, help="output directory for tables/figures/policy.json")
    parser.add_argument("--horizon", type=int, default=600, choices=(120, 300, 600))
    parser.add_argument("--kinds", default="gbt,linear")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    analyze(args.data, args.out, kinds=tuple(args.kinds.split(",")), horizon=args.horizon,
            plots=not args.no_plots)


if __name__ == "__main__":
    main()
