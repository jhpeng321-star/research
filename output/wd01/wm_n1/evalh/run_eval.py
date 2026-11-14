"""Generated-fixture-only WM-N1 evaluation harness.

``--synthetic-only`` is mandatory.  The only permitted source is the generated
``SyntheticSource`` and the only predictors are the bundled esynth test
factories.  This legacy CLI cannot read real CAL or final-test outcomes,
load a checkpoint/comparator by import string, or consume use markers.
``evalh.real_cli`` is the separately released metadata preflight and currently
refuses real scoring.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import metrics as M
from . import seals as S
from .protocol import HORIZONS_S
from .sources import SyntheticSource

from . import cal_split


def selection_worlds(supplied=None, *, cal_split_path=None) -> list:
    """CAL-tune worlds allowed on the improvement-loop selection path."""
    return cal_split.resolve_worlds(
        cal_split.CAL_TUNE, supplied, path=cal_split_path,
        context="evalh selection")


def calibration_worlds(supplied=None, *, cal_split_path=None) -> list:
    """CAL-cal worlds allowed on the calibration / variant-choice path."""
    return cal_split.resolve_worlds(
        cal_split.CAL_CAL, supplied, path=cal_split_path,
        context="evalh calibration/variant")


def cal_half_rows(rows, half, *, cal_split_path=None) -> list:
    """Restrict realised CAL rows to one A4.2 half (other half dropped).

    A row whose world is not in the CAL split at all is refused by
    :mod:`evalh.cal_split` (a foreign/sealed world never reaches an eval).
    """
    keep = set(cal_split.half_worlds(half, path=cal_split_path))
    worlds = [str(getattr(r, "world", r)) for r in rows]
    cal_split.filter_worlds(worlds, half, path=cal_split_path)
    return [r for r, w in zip(rows, worlds) if w in keep]


def _fmt_q(v: float) -> str:
    return f"{float(v):.6g}"


def write_rows(path: Path, rows, variant_q, hybrid_q, levels) -> Path:
    """Write one CSV row per (window, policy, tau) BEFORE aggregation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    qnames = [f"q{float(lv):.2f}" for lv in levels]
    header = (["world", "t0_s", "decision_t_s", "tau_s", "policy",
               "y_delay_s", "collision", "veh_steps", "ttc_lt3_s",
               "hard_brake_s"] + [f"variant_{n}" for n in qnames]
              + [f"hybrid_{n}" for n in qnames])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i, r in enumerate(rows):
            w.writerow([r.world, _fmt_q(r.t0_s), _fmt_q(r.decision_t_s),
                        _fmt_q(r.tau_s), r.policy, _fmt_q(r.y_delay_s),
                        _fmt_q(r.collision), _fmt_q(r.veh_steps),
                        _fmt_q(r.ttc_lt3_s), _fmt_q(r.hard_brake_s)]
                       + [_fmt_q(x) for x in variant_q[i]]
                       + [_fmt_q(x) for x in hybrid_q[i]])
    return path


def _table_md(title: str, headers, rows) -> list:
    out = [f"### {title}", "", "| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    out.append("")
    return out


def write_report(out_dir: Path, split: str, receipt: dict, delta: dict,
                 level_crps: dict, coverage_variant: dict, coverage_hybrid: dict,
                 roll_variant: dict, roll_hybrid: dict, decision: dict,
                 n_rows: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = dict(
        schema="wm_n1_eval_report/1",
        split=split, generated_utc=datetime.now(timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        seal_receipt=receipt, n_rows=int(n_rows),
        delta_crps=delta, level_crps=level_crps,
        coverage_10_90=dict(variant=coverage_variant, hybrid=coverage_hybrid),
        short_horizon=dict(variant=roll_variant, hybrid=roll_hybrid),
        decision=decision)
    (out_dir / "REPORT.json").write_text(
        json.dumps(report, indent=1, default=float), encoding="utf-8")

    md = [f"# WM-N1 evaluation report ({split})", "",
          f"Rows: {n_rows}; generated {report['generated_utc']}.", "",
          "Seal receipt: `" + json.dumps(receipt) + "`", "",
          "## Decision rule (frozen, REGISTRATION_WM_N1.md:79-88)", "",
          "> " + decision["quote"].replace("\n", "\n> "), "",
          f"**Verdict: `{decision['verdict']}`**", ""]
    md += _table_md(
        "Decision checks", ["check", "passed", "citation"],
        [[k, v["passed"], v["citation"]] for k, v in decision["checks"].items()])
    d_rows = []
    for contrast in sorted(delta):
        for tau in sorted(delta[contrast]):
            c = delta[contrast][tau]
            d_rows.append([contrast, tau, f"{c['variant_crps']:.3f}",
                           f"{c['hybrid_crps']:.3f}",
                           f"[{c['ci_lo']:.3f}, {c['ci_hi']:.3f}]",
                           c["improves"],
                           f"{c['variant_sign_accuracy']:.3f}"
                           if c["variant_sign_accuracy"] == c["variant_sign_accuracy"]
                           else "nan",
                           c["n_pairs"]])
    md += _table_md("Paired dV CRPS (world-bootstrap 90% CI)",
                    ["contrast", "tau", "variant", "hybrid", "CI", "improves",
                     "sign_acc", "n"], d_rows)
    l_rows = []
    for tau in sorted(level_crps):
        c = level_crps[tau]
        l_rows.append([tau, f"{c['variant_crps']:.3f}",
                       f"{c['hybrid_crps']:.3f}", f"{c['ratio']:.3f}"])
    md += _table_md("Level outcome CRPS", ["tau", "variant", "hybrid", "ratio"],
                    l_rows)
    r_rows = []
    for h in (10.0, 30.0, 60.0, 120.0):
        v = roll_variant["s_rmse_pct"].get(h, {})
        hy = roll_hybrid["s_rmse_pct"].get(h, {})
        r_rows.append([h, f"{v.get('s_rmse_pct', float('nan')):.2f}",
                       f"{hy.get('s_rmse_pct', float('nan')):.2f}",
                       f"{v.get('v_rmse_pct', float('nan')):.2f}"])
    md += _table_md("Short-horizon RMSE (% of true)", ["H_s", "variant s",
                    "hybrid s", "variant v"], r_rows)
    md += _table_md("Validity", ["metric", "variant", "hybrid"],
                    [["collisions/1000", roll_variant["collision_per_1000"],
                      roll_hybrid["collision_per_1000"]],
                     ["TTC<3 s", roll_variant["ttc_lt3_s"],
                      roll_hybrid["ttc_lt3_s"]]])
    (out_dir / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def run(args) -> dict:
    # This historical harness has only generated fixture adapters.  It must
    # refuse before seal markers, data sources, or dynamic predictor imports:
    # a real evaluation belongs to the separately reviewed real_cli path.
    _require_synthetic_fixture(args)
    out_dir = Path(args.out)
    cal_split_path = getattr(args, "cal_split_path", None)

    # Single A4.2 guard: a CAL evaluation must name its half, so the member-1/
    # quick-tune SELECTION path can never silently read the calibration half
    # (and vice versa).  Raised BEFORE the read-once marker is consumed.
    if args.split == "cal" and not getattr(args, "cal_half", None):
        raise S.SealError(
            "--split cal requires --cal-half (cal_tune: the member-1/"
            "quick-tune selection metric; cal_cal: early stopping, "
            "calibration, blend weights and variant choice).  Refusing an "
            "unpinned CAL half (amendment A4.2).")

    # No production read-once marker is created for generated fixtures.
    receipt = {"schema": "wm_n1_generated_eval_fixture/1",
               "split": args.split, "synthetic_only": True,
               "use_marker_consumed": False}
    from .tests import esynth

    source = SyntheticSource(n_worlds=args.synthetic_n, seed=args.seed)
    variant = esynth.variant()
    reference = esynth.hybrid()

    rows = list(source.outcome_rows(args.split))
    rollouts = list(source.rollout_rows(args.split))
    if getattr(args, "cal_half", None):
        if args.split != "cal":
            raise S.SealError(
                f"--cal-half applies only to --split cal, not {args.split!r}")
        rows = cal_half_rows(rows, args.cal_half, cal_split_path=cal_split_path)
        rollouts = cal_half_rows(rollouts, args.cal_half,
                                 cal_split_path=cal_split_path)
        if not rows:
            raise S.SealError(
                f"cal half {args.cal_half!r} selected no rows")

    levels = np.asarray(variant.quantile_levels, float)
    variant_q = np.asarray(variant.predict_quantiles(rows), float)
    hybrid_q = np.asarray(reference.predict_quantiles(rows), float)
    if hybrid_q.shape != variant_q.shape:
        raise ValueError(
            f"predictors disagree: variant {variant_q.shape}, reference "
            f"{hybrid_q.shape}")

    out_dir.mkdir(parents=True, exist_ok=False)
    # Rows FIRST, before any aggregation (registration record discipline).
    rows_path = write_rows(out_dir / "rows.csv", rows, variant_q, hybrid_q,
                           levels)

    delta = M.delta_metrics(M.build_delta_pairs(rows, variant_q, hybrid_q,
                                                levels=levels),
                            threshold=args.sign_threshold
                            if args.sign_threshold is not None
                            else float(getattr(source, "sign_threshold", 0.0)),
                            levels=levels, n_boot=args.n_boot, seed=args.seed)
    level_crps = M.level_crps_pair(rows, variant_q, hybrid_q, levels=levels)
    cov_v = {}
    cov_h = {}
    for tau in sorted(level_crps):
        sel = [i for i, r in enumerate(rows)
               if round(float(r.tau_s), 6) == tau]
        y = [rows[i].y_delay_s for i in sel]
        lo_i = M.level_index(0.10, levels)
        hi_i = M.level_index(0.90, levels)
        cov_v[tau] = M.coverage(variant_q[sel, lo_i], variant_q[sel, hi_i], y)
        cov_h[tau] = M.coverage(hybrid_q[sel, lo_i], hybrid_q[sel, hi_i], y)

    var_roll = variant.sample_rollouts(rollouts, HORIZONS_S)
    hyb_roll = reference.sample_rollouts(rollouts, HORIZONS_S)
    roll_v = M.rollout_metrics(var_roll, rollouts, name=variant.name)
    roll_h = M.rollout_metrics(hyb_roll, rollouts, name=reference.name)

    evidence = M.evidence_from_metrics(delta, level_crps, roll_v, roll_h)
    decision = M.decide(evidence)

    report = write_report(out_dir, args.split, receipt, delta, level_crps,
                          cov_v, cov_h, roll_v, roll_h, decision, len(rows))
    print(f"run_eval split={args.split} rows={len(rows)} "
          f"rows_written={rows_path.name}")
    print(f"  variant={variant.name} reference={reference.name}")
    print(f"  verdict={decision['verdict']} passed={decision['passed']}")
    for k, v in decision["checks"].items():
        print(f"  [{ 'PASS' if v['passed'] else 'FAIL'}] {k} ({v['citation']})")
    print(f"  report={out_dir / 'REPORT.json'} ({report['schema']})")
    return report


def _require_synthetic_fixture(args) -> None:
    """Keep the legacy CLI bound to generated fixtures before any read/import."""
    if not getattr(args, "synthetic_only", False):
        raise S.SealError(
            "legacy evalh.run_eval requires --synthetic-only; real scoring "
            "uses evalh.real_cli and remains disabled")
    if getattr(args, "source", None) is not None:
        raise S.SealError("legacy fixture refuses any custom --source")
    out = Path(args.out)
    if out.exists() or out.is_symlink():
        raise S.SealError(
            "legacy fixture requires a fresh --out; existing evidence is preserved")
    if (args.predictor, args.reference) != ("esynth:variant", "esynth:hybrid"):
        raise S.SealError(
            "legacy fixture accepts only esynth:variant and esynth:hybrid")
    if args.split == "final_test" or getattr(args, "final_test", False):
        raise S.SealError("legacy fixture refuses final-test access")
    if (getattr(args, "marker_dir", None) is not None
            or getattr(args, "registration_sha", None) is not None
            or getattr(args, "chunk5_sha", None) is not None
            or getattr(args, "dry_run", False)):
        raise S.SealError(
            "legacy fixture refuses seal, marker, and dry-run options")
    if args.split == "cal":
        if not getattr(args, "cal_half", None):
            raise S.SealError(
                "--split cal requires --cal-half (cal_tune or cal_cal)")
        path = getattr(args, "cal_split_path", None)
        if not path:
            raise S.SealError(
                "synthetic CAL requires explicit generated --cal-split-path")
        split_path = Path(path)
        out_parent = Path(args.out).absolute().parent
        if (not split_path.is_absolute()
                or split_path.name != "SYNTHETIC_CAL_SPLIT_A4.json"
                or split_path.parent != out_parent
                or split_path.is_symlink()):
            raise S.SealError(
                "synthetic CAL split must be a non-link generated sibling of --out")
        data = cal_split.load_cal_split(split_path)
        if data.get("synthetic_fixture") is not True:
            raise S.SealError(
                "synthetic CAL split requires an explicit generated fixture marker")
        n = int(args.synthetic_n)
        expected = {f"w{i:03d}" for i in range(n)}
        got = set(data[cal_split.CAL_TUNE]) | set(data[cal_split.CAL_CAL])
        if n < 2 or got != expected:
            raise S.SealError(
                "synthetic CAL split must partition exactly generated fixture worlds")
    elif (getattr(args, "cal_half", None) is not None
          or getattr(args, "cal_split_path", None) is not None):
        raise S.SealError("CAL-half options apply only to synthetic --split cal")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evalh.run_eval")
    p.add_argument("--synthetic-only", action="store_true",
                   help="acknowledge generated-fixture evaluation only")
    p.add_argument("--split", required=True, choices=list(S.SPLITS))
    p.add_argument("--out", required=True)
    p.add_argument("--marker-dir", default=None)
    p.add_argument("--predictor", required=True,
                   help="must be esynth:variant (generated fixture)")
    p.add_argument("--reference", required=True,
                   help="must be esynth:hybrid (generated fixture)")
    p.add_argument("--source", default=None,
                   help="retired: any custom source is refused")
    p.add_argument("--synthetic-n", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--sign-threshold", type=float, default=None)
    p.add_argument("--final-test", action="store_true")
    p.add_argument("--registration-sha", dest="registration_sha", default=None)
    p.add_argument("--chunk5-sha", dest="chunk5_sha", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="check the seals without consuming the read-once marker")
    p.add_argument("--cal-half", dest="cal_half", default=None,
                   choices=list(cal_split.HALVES),
                   help="restrict a --split cal run to one A4.2 half "
                        "(cal_cal = calibration/variant, cal_tune = selection)")
    p.add_argument("--cal-split-path", dest="cal_split_path", default=None,
                   help="override CAL_SPLIT_A4.json (synthetic tests only)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
