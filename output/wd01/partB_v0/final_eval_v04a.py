"""WD-01-PB-V04A final evaluation: ONE CAL read (registration s.6, s.7).

Run exactly once, after all fits.  A lock file
(<results-root>/CAL_READ_ONCE_v04a.json) is created BEFORE the first CAL
row is touched; a second invocation refuses.  No model is (re)fitted here
(the fits are loaded from their result dirs; a fit after this read is
forbidden by the stop rule).

Per variant (strict = HEADLINE, detector = SENSITIVITY, strict_noaux =
auxiliary-data ABLATION) it evaluates, on the CALIBRATION split with
AUXILIARY sources excluded from scoring:
  A1 (CRPS vs climatology per tau, overall 10-90 coverage), per-policy
  coverage, demand / penetration strata (report strata only);
  A2 vs ridge per tau (baselines_section, HARD PASS/TIE/FAIL);
  A3 contrast skill (damping-native, pi-native gating; switching-native
  and fixed-native reported) with independent-quantile pairing and the
  comonotone sensitivity; zero-contrast and TRAIN-cell-mean baselines;
  the TRAIN-OOB versions of A1-A3 (labelled: selection-free of CAL,
  in-sample for the conformal shifts).
and writes <results dir>/eval_report.md, eval_v04a.json, refreshes the
OUTPUT_MANIFEST/zip and runs verify; then REPORT_V04A.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import wdpaths  # noqa: E402

from . import acceptance_v04a as AC  # noqa: E402
from .data import V0Dataset, load_manifest  # noqa: E402
from .gbm import (GbmValueModel, _aux_mask, _tau_mask,  # noqa: E402
                  baselines_section, crps_from_quantiles,
                  oob_acceptance_section)
from .outputs import finalize_outputs, verify_outputs  # noqa: E402

TAUS = (30.0, 60.0, 120.0, 300.0, 570.0)
RES_ROOT = Path(wdpaths.RESEARCH_DENSE_ROOT) / "output" / "partB_results"
DATA_ROOT = Path(wdpaths.RESEARCH_DENSE_ROOT) / "output" \
    / "wd01_partB_v0data_v04a"

VARIANTS = {
    "strict": dict(dir="v0.4a_strict", features="deployable_strict",
                   exclude_aux=False,
                   label="deployable_strict (HEADLINE)"),
    "detector": dict(dir="v0.4a_detector", features="deployable_detector",
                     exclude_aux=False,
                     label="deployable_detector (SENSITIVITY)"),
    "strict_noaux": dict(dir="v0.4a_strict_noaux",
                         features="deployable_strict", exclude_aux=True,
                         label="deployable_strict WITHOUT auxiliary "
                               "chunks 1-3 (ABLATION)"),
}


# ------------------------------------------------------------ helpers -----

def oof_quantiles(model: GbmValueModel, ds, row_idx) -> dict:
    """Calibrated, monotone WORLD-GROUPED OOF quantiles for TRAIN label
    rows (the fit's own conformal folds); tau-wise lookup by the row's
    position among the TRAIN rows of its tau."""
    r = ds.rows
    row_idx = np.asarray(row_idx, dtype=np.int64)
    qs = sorted(model.target_quantiles["delay"])
    out = {q: np.full(len(row_idx), np.nan) for q in qs}
    taus = r["tau"].to_numpy()[row_idx]
    for tau in model.taus:
        oof = model.oof_delay.get(float(tau))
        if not oof:
            continue
        idx_tr = np.flatnonzero(_tau_mask(r, tau, "TRAIN"))
        n_fit = len(next(iter(oof.values())))
        if n_fit != len(idx_tr):
            raise RuntimeError(
                f"OOF length {n_fit} != TRAIN rows {len(idx_tr)} at "
                f"tau={tau}: the evaluation dataset differs from the "
                "fit dataset")
        pos = np.full(len(r), -1, dtype=np.int64)
        pos[idx_tr] = np.arange(len(idx_tr))
        m = np.isclose(taus, tau)
        if not m.any():
            continue
        k = pos[row_idx[m]]
        if (k < 0).any():
            raise RuntimeError("OOF query for a non-TRAIN row")
        preds = {q: np.asarray(oof[q], float)[k]
                 + model.shifts.get((float(tau), "delay", q), 0.0)
                 for q in qs}
        preds = model._monotone(preds)
        for q in qs:
            out[q][m] = preds[q]
    return out


def _quantiles_for_design(model, X, taus) -> dict:
    return model.predict_quantiles_matrix(X, taus, "delay",
                                          calibrated=True)


def contrast_block(ds, model, split: str, refs: dict, pairs_train, *,
                   oof: bool = False, n_boot: int = AC.N_BOOT) -> dict:
    """A3 records for the pairs of `split` (CAL: model predictions; TRAIN
    with oof=True: the fit's OOF predictions).  Returns dict(records,
    pairs)."""
    pairs = AC.build_pairs(ds.rows, (split,)).reset_index(drop=True)
    if not len(pairs):
        return dict(records=[], pairs=pairs)
    qs = sorted(model.target_quantiles["delay"])
    u = AC.level_grid()
    if oof:
        qa = oof_quantiles(model, ds, pairs["row_a"].to_numpy())
        qn = oof_quantiles(model, ds, pairs["row_n"].to_numpy())
    else:
        taus = pairs["tau"].to_numpy(float)
        Xa = model.design_rows(ds, pairs["row_a"].to_numpy())
        Xn = model.design_rows(ds, pairs["row_n"].to_numpy())
        qa = _quantiles_for_design(model, Xa, taus)
        qn = _quantiles_for_design(model, Xn, taus)
    sa = AC.quantile_samples(qs, qa, u)
    sn = AC.quantile_samples(qs, qn, u)
    ind = AC.delta_samples(sa, sn, "independent")
    com = AC.delta_samples(sa, sn, "comonotone")
    point = np.asarray(qa[0.5]) - np.asarray(qn[0.5])
    rec = AC.contrast_metrics(pairs, ind, com, point, refs, n_boot=n_boot)
    pairs = pairs.assign(dv_point=point, dv_mean=ind.mean(axis=1))
    return dict(records=rec, pairs=pairs)


def a1_block(ds, model, idx, *, oof: bool, log=print) -> dict:
    """A1 numbers on label rows `idx` (CAL: predictions; TRAIN+oof)."""
    from .evaluate import Climatology
    r = ds.rows
    idx = np.asarray(idx, dtype=np.int64)
    sub = r.iloc[idx]
    y = sub["total_delay_ref_s"].to_numpy(float)
    qs = sorted(model.target_quantiles["delay"])
    if oof:
        pr = oof_quantiles(model, ds, idx)
    else:
        pr = model.predict_quantiles_rows(ds, idx, "delay")
    crps = crps_from_quantiles(qs, pr, y)
    inside = (y >= pr[0.1]) & (y <= pr[0.9])
    clim = Climatology(r[r["split"] == "TRAIN"], "total_delay_ref_s")
    crps_c, _lv = clim.crps(y, sub["tau"].to_numpy(),
                            sub["policy"].tolist(),
                            sub["demand_veh_h"].to_numpy())
    per_tau = []
    for tau in TAUS:
        m = np.isclose(sub["tau"].to_numpy(), tau)
        if not m.any():
            continue
        per_tau.append(dict(tau=tau, n=int(m.sum()),
                            crps_model=float(np.mean(crps[m])),
                            crps_climatology=float(np.nanmean(crps_c[m])),
                            mae_model=float(np.mean(np.abs(pr[0.5][m]
                                                           - y[m]))),
                            coverage=float(inside[m].mean())))
    cov = float(inside.mean())
    by_policy = {p: dict(n=int((sub["policy"] == p).sum()),
                         coverage=float(inside[(sub["policy"] == p)
                                             .to_numpy()].mean()),
                         crps_model=float(np.mean(crps[(sub["policy"] == p)
                                                       .to_numpy()])),
                         crps_climatology=float(np.nanmean(crps_c[
                             (sub["policy"] == p).to_numpy()])))
                 for p in sorted(sub["policy"].unique())}

    def strata(col):
        out = {}
        for v in sorted(sub[col].unique()):
            m = (sub[col] == v).to_numpy()
            out[str(v)] = dict(n=int(m.sum()),
                               coverage=float(inside[m].mean()),
                               crps_model=float(np.mean(crps[m])),
                               crps_climatology=float(np.nanmean(
                                   crps_c[m])))
        return out
    return dict(per_tau=per_tau, coverage_overall=cov,
                crps_overall=float(np.mean(crps)),
                crps_climatology_overall=float(np.nanmean(crps_c)),
                by_policy=by_policy, by_demand=strata("demand_veh_h"),
                by_penetration=strata("penetration"),
                verdict=AC.a1_verdict(per_tau, cov), n=int(len(idx)))


def evaluate_variant(name: str, spec: dict, data_root: Path, res_root: Path,
                     log=print, n_boot: int = AC.N_BOOT, *,
                     eval_split: str = "CALIBRATION",
                     splits=("TRAIN", "CALIBRATION")) -> dict:
    """eval_split/splits differ from the defaults ONLY in the plumbing DRY
    RUN (`--dry-run`): TRAIN rows play the CAL role, no CAL file is
    opened and nothing is written into the model dirs."""
    out = res_root / spec["dir"]
    log(f"[{name}] loading model {out}")
    model = GbmValueModel.load(out, log=log)
    if model.design != "v04a":
        raise RuntimeError(f"{out}: not a v0.4a design model")
    if model.feature_set != spec["features"]:
        raise RuntimeError(f"{out}: feature set {model.feature_set} != "
                           f"{spec['features']}")
    man = load_manifest(data_root, splits=tuple(splits),
                        exclude_aux=spec["exclude_aux"], log=log)
    ds = V0Dataset(man, feature_set=spec["features"], log=log)
    assert man.sha256 == model.data_manifest_sha256, \
        "dataset MANIFEST differs from the one the model was fitted on"
    r = ds.rows
    aux = _aux_mask(r)
    cal = np.flatnonzero((r["split"] == eval_split).to_numpy() & ~aux)
    trn = np.flatnonzero((r["split"] == "TRAIN").to_numpy() & ~aux)
    assert not (r["split"].iloc[cal] != eval_split).any()
    log(f"[{name}] CAL scoring rows {len(cal)} (aux excluded); "
        f"TRAIN scoring rows {len(trn)}")

    # ---- A1 + A2 on CAL --------------------------------------------------
    a1 = a1_block(ds, model, cal, oof=False, log=log)
    pr = model.predict_quantiles_rows(ds, cal, "delay")
    qs = sorted(model.target_quantiles["delay"])
    y_cal = r["total_delay_ref_s"].to_numpy()[cal]
    crps_cal = crps_from_quantiles(qs, pr, y_cal)
    base = baselines_section(ds, cal, pr[0.5], crps_cal, log=log)
    a2 = AC.a2_verdict(base["acceptance"])

    # ---- A3 on CAL -------------------------------------------------------
    train_pairs = AC.build_pairs(ds.rows, ("TRAIN",)).reset_index(drop=True)
    refs = AC.train_references(train_pairs)
    cal_block = contrast_block(ds, model, eval_split, refs, train_pairs,
                               n_boot=n_boot)
    a3_records = cal_block["records"]
    a3 = AC.a3_verdict(a3_records)
    verdict = AC.overall_verdict(a1["verdict"], a2, a3)

    # ---- TRAIN-OOB versions ---------------------------------------------
    a1_oob = a1_block(ds, model, trn, oof=True, log=log)
    oob_acc = oob_acceptance_section(ds, model, log=log)
    a2_oob = AC.a2_verdict(oob_acc["acceptance"])
    oob_block = contrast_block(ds, model, "TRAIN", refs, train_pairs,
                               oof=True, n_boot=n_boot)
    a3_oob = AC.a3_verdict(oob_block["records"])
    verdict_oob = AC.overall_verdict(a1_oob["verdict"], a2_oob, a3_oob)

    res = dict(
        name=name, label=spec["label"], results_dir=str(out),
        features=spec["features"], exclude_aux=spec["exclude_aux"],
        data_manifest_sha256=man.sha256, n_cal_rows=int(len(cal)),
        n_train_scoring_rows=int(len(trn)),
        n_cal_pairs=int(len(cal_block["pairs"])),
        n_train_pairs=int(len(train_pairs)),
        tau_variants=dict(model.tau_variants),
        a1=a1, a2=a2, a2_table=base["acceptance"],
        a2_rows=base["rows"], a3=a3, a3_records=a3_records, verdict=verdict,
        train_oob=dict(a1=a1_block_summary(a1_oob), a2=a2_oob,
                       a2_table=oob_acc["acceptance"], a3=a3_oob,
                       a3_records=oob_block["records"],
                       verdict=verdict_oob,
                       note=("TRAIN-OOB (world-grouped OOF of the fit's "
                             "conformal folds; the shifts were fitted on "
                             "this sample and the climatology / TRAIN "
                             "references are in-sample): selection-free "
                             "of CAL, NOT out-of-sample")),
        train_references={f"{c}|{t:.0f}": dict(
            median_abs=v["median_abs"], pooled_mean=v["pooled_mean"],
            n=v["n"]) for (c, t), v in refs.items()},
        cal_pairs=cal_block["pairs"].to_dict(orient="records"))
    return res


def a1_block_summary(a1: dict) -> dict:
    return dict(per_tau=a1["per_tau"],
                coverage_overall=a1["coverage_overall"],
                verdict=a1["verdict"], by_policy=a1["by_policy"])


# -------------------------------------------------------------- main ------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(DATA_ROOT))
    ap.add_argument("--results-root", default=str(RES_ROOT))
    ap.add_argument("--variants", default="strict,detector,strict_noaux")
    ap.add_argument("--report", default=None,
                    help="REPORT_V04A.md output path")
    ap.add_argument("--n-boot", type=int, default=AC.N_BOOT)
    ap.add_argument("--dry-run", action="store_true",
                    help="plumbing check BEFORE the CAL read: TRAIN rows "
                         "play the CAL role, no CAL file is opened, no "
                         "lock, nothing is written into the model dirs")
    ap.add_argument("--dry-out", default=None,
                    help="dry-run output dir (JSON + report)")
    args = ap.parse_args(argv)
    res_root = Path(args.results_root)
    if args.dry_run:
        from .report_v04a import render_report
        out = Path(args.dry_out or ".")
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        results = {}
        for name in [v for v in args.variants.split(",") if v]:
            results[name] = evaluate_variant(
                name, VARIANTS[name], Path(args.data), res_root,
                log=lambda m: print(f"[dry {time.time() - t0:7.1f}s] {m}",
                                    flush=True),
                n_boot=args.n_boot, eval_split="TRAIN", splits=("TRAIN",))
            (out / f"dry_{name}.json").write_text(
                json.dumps(results[name], indent=1, default=float),
                encoding="utf-8")
        (out / "DRY_RUN_REPORT.md").write_text(
            "# DRY RUN (TRAIN rows in the CAL role; plumbing only, NOT "
            "evidence)\n\n" + render_report(results), encoding="utf-8")
        print(f"dry run done in {time.time() - t0:.0f}s -> {out}")
        return 0
    lock = res_root / "CAL_READ_ONCE_v04a.json"
    if lock.exists():
        print(f"REFUSED: {lock} exists -- the CAL read already happened "
              "(registration s.7: no re-read / no refit after it)")
        return 2
    lock.write_text(json.dumps(dict(
        started_utc=datetime.now(timezone.utc).isoformat(),
        variants=args.variants,
        rule="ONE CAL read for the final acceptance report; no refit "
             "after this point"), indent=1), encoding="utf-8")
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    results = {}
    for name in [v for v in args.variants.split(",") if v]:
        spec = VARIANTS[name]
        results[name] = evaluate_variant(name, spec, Path(args.data),
                                         res_root, log=log,
                                         n_boot=args.n_boot)
        (res_root / spec["dir"] / "eval_v04a.json").write_text(
            json.dumps(results[name], indent=1, default=float),
            encoding="utf-8")
    from .report_v04a import render_variant_report, render_report
    for name, res in results.items():
        d = res_root / VARIANTS[name]["dir"]
        (d / "eval_report.md").write_text(render_variant_report(res),
                                          encoding="utf-8")
        mp = d / "metrics.json"
        met = json.loads(mp.read_text(encoding="utf-8")) if mp.exists()             else {}
        met["eval_v04a"] = {k: res[k] for k in (
            "name", "label", "verdict", "a1", "a2", "a2_table", "a3",
            "a3_records", "n_cal_rows", "n_cal_pairs", "n_train_pairs",
            "data_manifest_sha256")}
        met["eval_v04a"]["train_oob_verdict"] = res["train_oob"]["verdict"]
        mp.write_text(json.dumps(met, indent=1, default=float),
                      encoding="utf-8")
        finalize_outputs(d, log=None)
        ok, problems = verify_outputs(d)
        print(f"verify {d.name}: " + ("OK" if ok else f"FAIL {problems}"))
    if args.report:
        Path(args.report).write_text(render_report(results),
                                     encoding="utf-8")
        log(f"report written: {args.report}")
    lock.write_text(json.dumps(dict(
        started_utc=json.loads(lock.read_text())["started_utc"],
        finished_utc=datetime.now(timezone.utc).isoformat(),
        verdicts={k: v["verdict"]["verdict"] for k, v in results.items()},
        wall_s=round(time.time() - t0, 1)), indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
