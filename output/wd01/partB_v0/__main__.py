"""partB_v0 CLI (PLAN_PARTB_V0_TRAINING.md addenda 2, 3, 3a).

    python -m partB_v0 train  --data <dataset_root> --out <dir> [flags]
    python -m partB_v0 eval   --data <dataset_root> --out <dir> [flags]
    python -m partB_v0 verify --out <dir>

train  = verify data MANIFEST -> timing probe -> M members -> CAL
         evaluation -> weights + metrics + eval_report + env report +
         OUTPUT_MANIFEST + results.zip (+ .sha256).
eval   = re-evaluate existing weights in --out on the CALIBRATION split.
verify = re-hash every OUTPUT_MANIFEST entry, check results.zip contents
         match the manifest exactly and results.zip.sha256 pins the zip.

--out defaults to $OUT_DIR, else /out (addendum 3a: set OUT_DIR on
RunPod/Vast to a persistent volume; pass --out explicitly for local
runs).  Resume is automatic when checkpoints exist in --out
(--no-resume forces a fresh start; a truncated/corrupt latest checkpoint
falls back to the previous one).  Re-running the SAME train command on
a completed OUT_DIR (cloud restart case) does NOT retrain: completed
members are skipped, evaluation is re-run only when its outputs are
missing/stale (a member retrained), and the bundle is rebuilt so
OUTPUT_MANIFEST and results.zip stay consistent.  The bundle holds the
deliverables only; ckpt/ is resume-only state kept in OUT_DIR (excluded
from the manifest and the zip).  The manifest/zip are rebuilt once more
AFTER the last log line, so every manifest hash matches the final
file contents; the final zip sha256 is printed to stdout only (the log
is frozen at that point).  KEEP_ALIVE=1 keeps the process alive after
DONE (cloud containers); unset/0 exits normally.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .data import V0Dataset, default_data_root, load_manifest
from .evaluate import EvalConfig, evaluate_run
from .outputs import (TeeLogger, finalize_outputs, keep_alive_if_requested,
                      verify_outputs, write_env_report)
from .train import TrainConfig, resolve_device, run_training
from .train_net import TrainNetConfig, run_net_training


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", default=None,
                   help="dataset root (default $WD01_V0_DATA, else "
                        "<RESEARCH_DENSE_ROOT>/output/wd01_partB_v0data)")
    p.add_argument("--out", default=None,
                   help="output dir (default $OUT_DIR, else /out)")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "cpu"],
                   help="auto = cuda if available else cpu "
                        "(WD_DEVICE env respected)")
    p.add_argument("--chunks", default=None,
                   help="comma list, e.g. chunk1 (default: all present; "
                        "v0.1 trains on chunk1 only)")
    p.add_argument("--no-verify", action="store_true",
                   help="skip per-file sha256 verification (speed)")
    p.add_argument("--exclude-aux", dest="exclude_aux",
                   action="store_true", default=False,
                   help="v0.4a: drop AUXILIARY sources (chunks 1-3 "
                        "per-CAV switching arms + chunk-1 bases; the "
                        "auxiliary-data ablation)")
    p.add_argument("--train-only", dest="train_only", action="store_true",
                   default=False,
                   help="v0.4a: load the TRAIN split ONLY (CALIBRATION "
                        "files are not opened; CAL is read once, by the "
                        "final evaluation)")
    excl = p.add_mutually_exclusive_group()
    excl.add_argument("--exclude-superseded", dest="exclude_superseded",
                      action="store_true", default=True,
                      help="skip MANIFEST sources marked superseded_by "
                           "(default: on)")
    excl.add_argument("--include-superseded", dest="exclude_superseded",
                      action="store_false",
                      help="train on superseded sources too")


def _eval_outputs_fresh(out: Path, n_members: int,
                        value_model: str = "net") -> bool:
    """True when the evaluation outputs exist and match the trained model
    (so a cloud restart can skip straight to the bundle)."""
    if not (out / "eval_report.md").exists():
        return False
    if value_model == "gbm":
        if not (out / "gbm_value.pkl").exists():
            return False
        try:
            m = json.loads((out / "metrics.json").read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        ev = m.get("eval") or {}
        return ev.get("value_model") == "gbm"
    for i in range(n_members):
        if not (out / f"member_{i}.pt").exists():
            return False
    mp = out / "metrics.json"
    if not mp.exists():
        return False
    try:
        m = json.loads(mp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    ev = m.get("eval") or {}
    return int(ev.get("n_members", -1)) == n_members


def _merge_metrics(old: dict | None, new: dict,
                   eval_skipped: bool) -> dict:
    """Cloud-restart merge: skipped members keep their original training
    records (and probe/data); a skipped evaluation keeps the old eval
    section."""
    if not old:
        return new
    all_skipped = bool(new.get("members")) and \
        all(r.get("already_done") for r in new["members"])
    old_members = old.get("members") or []
    new["members"] = [
        old_members[rep["member"]]
        if rep.get("already_done") and len(old_members) > \
        rep.get("member", -1) else rep
        for rep in new.get("members", [])]
    if all_skipped:
        for k in ("probe", "data", "planned_steps_per_member"):
            if k in old:
                new[k] = old[k]
    if eval_skipped and old.get("eval") is not None:
        new["eval"] = old["eval"]
    return new


def _run_net(args, out: Path, data: Path, chunks, log) -> int:
    """The `net` subcommand (B3b): v0.3 concurrent-member training ->
    D14 calibration fit on OOB -> v0.3 evaluation (incl. the v0.3.1
    acceptance BAR) -> bundle + verify."""
    from .evaluate import evaluate_net_v03, fit_run_calibration, \
        load_net_v03

    cfg = TrainNetConfig(
        data_root=str(data), out_dir=str(out),
        members=args.members, seed=args.seed,
        batch_size=args.batch_size, td_batch=args.td_batch,
        pair_batch=args.pair_batch, lr=args.lr,
        weight_decay=args.weight_decay, dropout=args.dropout,
        d_model=args.d_model, max_steps=args.max_steps,
        time_budget_min=args.time_budget_min,
        ckpt_every_min=args.ckpt_every_min,
        eval_every_steps=args.eval_every, patience=args.patience,
        max_steps_cap=args.max_steps_cap,
        replan_after_steps=args.replan_after, dtype=args.dtype,
        w_cons=args.w_cons, w_delta=args.w_delta,
        mc_weight_min=args.mc_min, ablation=args.ablation,
        ablation_steps=args.ablation_steps, variant=args.variant,
        eval_row_cap=args.eval_cap, smoke=args.smoke,
        device=args.device, exclude_superseded=args.exclude_superseded,
        verify=not args.no_verify, chunks=chunks,
        features=args.features, resume=not args.no_resume)
    log(f"partB_v0 net: data={data} out={out} features={args.features}")
    metrics, vdata = run_net_training(cfg, log=log)

    if not args.no_eval:
        device = resolve_device(args.device)
        bundle = load_net_v03(out, device, log=log)
        metrics["calibration"] = fit_run_calibration(
            bundle, vdata, out, method=args.cal_method, smoke=args.smoke,
            log=log)
        report, em = evaluate_net_v03(out, vdata, device,
                                      smoke=args.smoke, log=log)
        (out / "eval_report.md").write_text(report, encoding="utf-8")
        metrics["eval_v03"] = em
        log("eval: eval_report.md written (v0.3)")

    (out / "metrics.json").write_text(json.dumps(metrics, indent=1),
                                      encoding="utf-8")
    env = write_env_report(out, metrics["data"]["manifest_sha256"],
                           extra=dict(mode="net", smoke=bool(args.smoke),
                                      variant=metrics.get("variant"),
                                      dtype=metrics.get("dtype")))
    log(f"env: python={env['python']} torch={env.get('torch')} "
        f"cuda={env['cuda_available']} gpu={env.get('gpu_name', 'none')} "
        f"git={env['git_commit'][:12]}")

    log("outputs: writing OUTPUT_MANIFEST.json + results.zip (deliverables "
        "only; ckpt/ stays in OUT_DIR for resume)")
    finalize_outputs(out, log=None)
    zpath = out / "results.zip"
    log(f"DONE — download {zpath}")
    log(f"finished at "
        f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    log.close()
    fin = finalize_outputs(out, log=None)      # final silent re-hash
    print(f"bundle: {fin['n_files']} deliverable files, "
          f"results.zip {fin['results_zip_mib']} MiB, sha256 "
          f"{fin['results_zip_sha256']} (verified against "
          f"OUTPUT_MANIFEST.json)")
    ok, problems = verify_outputs(out)
    print("verify: OK -- every OUTPUT_MANIFEST entry re-hashes correctly; "
          "results.zip contents match the manifest; results.zip.sha256 "
          "matches the zip" if ok else f"verify: FAIL {problems}")
    keep_alive_if_requested()
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="partB_v0",
        description="WD-01 Part B V0: train/eval the tau- and "
                    "policy-conditioned value function V(s, tau, policy)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="train M members, then evaluate")
    _add_common(tr)
    tr.add_argument("--members", type=int, default=5)
    tr.add_argument("--seed", type=int, default=20260928)
    tr.add_argument("--batch-size", type=int, default=512)
    tr.add_argument("--time-budget-min", type=float, default=75.0)
    tr.add_argument("--ckpt-every-min", type=float, default=10.0)
    tr.add_argument("--max-steps", type=int, default=None,
                    help="fixed steps per member (overrides budget "
                         "scaling)")
    tr.add_argument("--value-model", choices=["net", "gbm"], default="net",
                    help="net = the M-member value net (default); gbm = "
                         "the calibrated quantile-GBM fallback value "
                         "model, fitted on CPU in minutes (B3 item D17; "
                         "no net training, same eval + bundle)")
    tr.add_argument("--gbm-variant",
                    choices=["hybrid", "pure", "auto"], default="hybrid",
                    help="hybrid (default) = per tau/target a ridge point "
                         "model on the summary features + the quantile-GBM "
                         "fitted on its residuals (trees cannot extrapolate "
                         "the near-linear long-tau growth; fixes the tau >= "
                         "120 HARD-acceptance failures); pure = the "
                         "original quantile-GBM on the target directly; "
                         "auto (v0.3.1) = per tau, hybrid vs ridge-only "
                         "(ridge point + conformal residual quantiles on "
                         "the same world-grouped TRAIN OOF folds) compared "
                         "by TRAIN OOB CRPS of the delay head only -- the "
                         "better one ships for that tau (tie ships ridge); "
                         "the choice + OOB scores land in the sidecar, "
                         "metrics.json and eval_report section 0")
    tr.add_argument("--gbm-max-iter", type=int, default=300,
                    help="max HGB iterations per quantile model for "
                         "--value-model gbm (early stopping at 15 flat "
                         "iterations usually ends sooner; smoke runs use "
                         "a small value)")
    tr.add_argument("--gbm-cal-folds", type=int, default=2,
                    help="TRAIN k-fold hold-out for the conformal "
                         "quantile shifts of --value-model gbm "
                         "(0 = in-fit residual shifts; smoke runs)")
    tr.add_argument("--features", default="oracle",
                    help="feature set: oracle = the historical V0 "
                         "features (exact future-arrival schedule; offline "
                         "upper bound); deployable_strict = the v0.3.2 "
                         "HEADLINE (CAV-STRICT: CAV self-reports only, no "
                         "all-vehicle entrance counts, no "
                         "penetration-revealing ratio); "
                         "deployable_detector = the v0.3.2 labelled "
                         "SENSITIVITY (all-vehicle entrance counts + "
                         "queues); deployable = deprecated v0.3.1 alias "
                         "of the detector blocks.  Default oracle for "
                         "backward compatibility")
    tr.add_argument("--compare-metrics", default=None,
                    help="metrics.json of the oracle run (e.g. the v0.3 "
                         "hybrid); with --features deployable the eval "
                         "report gains the with-vs-without-schedule 4-way "
                         "table (section 8)")
    tr.add_argument("--smoke", action="store_true",
                    help="tiny run (6 steps/member, capped eval) -- CPU "
                         "smoke only, numbers are not meaningful")
    tr.add_argument("--no-resume", action="store_true",
                    help="ignore existing checkpoints and start fresh")
    tr.add_argument("--no-eval", action="store_true",
                    help="skip the evaluation phase (training only)")

    ev = sub.add_parser("eval", help="re-evaluate existing weights")
    _add_common(ev)
    ev.add_argument("--value-model", choices=["net", "gbm"], default="net",
                    help="which value model in --out to evaluate (must "
                         "match how --out was trained)")
    ev.add_argument("--features", default="oracle",
                    help="feature set the model in --out was trained on "
                         "(must match: oracle | deployable_strict | "
                         "deployable_detector | deployable)")
    ev.add_argument("--compare-metrics", default=None,
                    help="metrics.json of the oracle run; with --features "
                         "deployable adds the with-vs-without-schedule "
                         "table (section 8)")
    ev.add_argument("--smoke", action="store_true",
                    help="cap eval rows (numbers are smoke-only)")

    nt = sub.add_parser(
        "net", help="v0.3 net (B3b): M concurrent members, OOB early "
                    "stopping, per-tau targets, delta head, D14 "
                    "calibration")
    _add_common(nt)
    nt.add_argument("--members", type=int, default=5)
    nt.add_argument("--seed", type=int, default=20260929)
    nt.add_argument("--batch-size", type=int, default=512)
    nt.add_argument("--td-batch", type=int, default=256)
    nt.add_argument("--pair-batch", type=int, default=128)
    nt.add_argument("--lr", type=float, default=3e-4)
    nt.add_argument("--weight-decay", type=float, default=0.01)
    nt.add_argument("--dropout", type=float, default=0.0)
    nt.add_argument("--d-model", type=int, default=192)
    nt.add_argument("--max-steps", type=int, default=None,
                    help="fixed lockstep steps (= per member; overrides "
                         "budget scaling)")
    nt.add_argument("--time-budget-min", type=float, default=75.0)
    nt.add_argument("--ckpt-every-min", type=float, default=10.0)
    nt.add_argument("--eval-every", type=int, default=500,
                    help="OOB eval interval (B6: 500)")
    nt.add_argument("--patience", type=int, default=3,
                    help="early-stop patience in evals (B6: 3)")
    nt.add_argument("--max-steps-cap", type=int, default=8000,
                    help="B6 cap on steps per member")
    nt.add_argument("--replan-after", type=int, default=500,
                    help="B11: re-plan the budget from measured ms/step "
                         "(incl. eval/ckpt overhead) after this many "
                         "steps; cosine LR follows")
    nt.add_argument("--dtype", default="auto",
                    choices=["auto", "bf16", "fp16", "fp32"],
                    help="auto: bf16 on Ampere+, fp16 (+ loss scaling) "
                         "on T4, fp32 on CPU")
    nt.add_argument("--w-cons", type=float, default=0.25,
                    help="B8 consistency-penalty weight")
    nt.add_argument("--w-delta", type=float, default=1.0,
                    help="C13 delta-head weight")
    nt.add_argument("--mc-min", type=float, default=0.5,
                    help="B8: MC weight = mc-min + (1-mc-min)*tau/570")
    nt.add_argument("--ablation", action="store_true",
                    help="B7: short base/wd/dropout/d_model128 runs on "
                         "OOB, then the winner is the default")
    nt.add_argument("--ablation-steps", type=int, default=1500)
    nt.add_argument("--variant", default="auto",
                    choices=["auto", "base", "wd", "dropout", "d_model128"],
                    help="auto = chosen by --ablation / ablation.json, "
                         "else base")
    nt.add_argument("--cal-method", default="shift",
                    choices=["shift", "isotonic"],
                    help="D14 post-hoc per-tau calibration map")
    nt.add_argument("--eval-cap", type=int, default=None,
                    help="cap OOB/CAL rows per (tau, policy) group "
                         "(smoke sets its own)")
    nt.add_argument("--features", default="oracle",
                    help="feature set (v0.3.1/v0.3.2): oracle = "
                         "historical V0 features; deployable_strict = "
                         "v0.3.2 HEADLINE (CAV self-reports only); "
                         "deployable_detector = v0.3.2 labelled "
                         "SENSITIVITY; deployable = deprecated v0.3.1 "
                         "alias (the net sizes its input projections "
                         "from the saved stats, so all work)")
    nt.add_argument("--smoke", action="store_true",
                    help="tiny CPU run (6 lockstep steps, capped evals) "
                         "-- numbers are not meaningful")
    nt.add_argument("--no-resume", action="store_true",
                    help="ignore existing checkpoints and start fresh")
    nt.add_argument("--no-eval", action="store_true",
                    help="skip calibration + evaluation (training only)")

    vf = sub.add_parser("verify",
                        help="re-hash OUTPUT_MANIFEST + results.zip")
    vf.add_argument("--out", default=None,
                    help="output dir (default $OUT_DIR, else /out)")

    args = ap.parse_args(argv)

    if args.cmd == "verify":
        out = Path(args.out or os.environ.get("OUT_DIR", "").strip()
                   or "/out")
        ok, problems = verify_outputs(out)
        n = 0
        man = out / "OUTPUT_MANIFEST.json"
        if man.exists():
            n = len(json.loads(man.read_text(encoding="utf-8"))
                    .get("files", {}))
        if ok:
            print(f"verify: OK -- {n}/{n} OUTPUT_MANIFEST entries "
                  f"re-hash correctly; results.zip contents match the "
                  f"manifest; results.zip.sha256 matches the zip")
            return 0
        for p in problems:
            print(f"verify: FAIL {p}")
        print(f"verify: {len(problems)} problem(s) in {out}")
        return 1

    out = Path(args.out or os.environ.get("OUT_DIR", "").strip() or "/out")
    data = Path(args.data or os.environ.get("WD01_V0_DATA", "").strip()
                or str(default_data_root()))
    chunks = (tuple(c.strip() for c in args.chunks.split(","))
              if args.chunks else None)
    if str(args.features).startswith("deployable") \
            and args.cmd == "train" and args.value_model == "net":
        print("REFUSAL: --features deployable* for the legacy v0.2 net "
              "path (train --value-model net) is oracle-shaped; use the "
              "`net` subcommand (v0.3, sizes its projections from the "
              "saved stats) or --value-model gbm", file=sys.stderr)
        return 2

    out.mkdir(parents=True, exist_ok=True)
    log = TeeLogger(out, name=f"{args.cmd}")

    if args.cmd == "net":
        return _run_net(args, out, data, chunks, log)

    if args.cmd == "train":
        cfg = TrainConfig(
            data_root=str(data), out_dir=str(out),
            members=args.members, seed=args.seed,
            batch_size=args.batch_size,
            time_budget_min=args.time_budget_min,
            ckpt_every_min=args.ckpt_every_min,
            max_steps=args.max_steps, smoke=args.smoke,
            device=args.device, exclude_superseded=args.exclude_superseded,
            verify=not args.no_verify, chunks=chunks,
            resume=not args.no_resume, value_model=args.value_model,
            features=args.features,
            extras={"gbm_max_iter": args.gbm_max_iter,
                    "gbm_cal_folds": args.gbm_cal_folds,
                    "gbm_variant": args.gbm_variant,
                    "exclude_aux": bool(args.exclude_aux),
                    "train_only": bool(args.train_only)})
        log(f"partB_v0 train: data={data} out={out} "
            f"value_model={args.value_model} features={args.features}"
            + (f" gbm_variant={args.gbm_variant}"
               if args.value_model == "gbm" else ""))
        if args.value_model == "gbm":
            from .gbm import run_gbm_training
            metrics, ds, all_skipped = run_gbm_training(cfg, log=log)
        else:
            metrics, ds = run_training(cfg, log=log)
            all_skipped = bool(metrics["members"]) and \
                all(r.get("already_done") for r in metrics["members"])
        eval_skipped = False
        if not args.no_eval:
            if all_skipped and _eval_outputs_fresh(out, cfg.members,
                                                   args.value_model):
                eval_skipped = True
                log("eval: outputs present and up to date; skipping "
                    "evaluation (delete eval_report.md to force a "
                    "re-run)")
            else:
                report, eval_metrics = evaluate_run(
                    EvalConfig(out_dir=str(out), device=args.device,
                               smoke=args.smoke,
                               value_model=args.value_model,
                               feature_set=args.features,
                               compare_metrics=args.compare_metrics,
                               members=(None if args.value_model == "gbm"
                                        else cfg.members)),
                    ds, log=log)
                (out / "eval_report.md").write_text(report,
                                                    encoding="utf-8")
                metrics["eval"] = eval_metrics
                log("eval: eval_report.md written")

        old_metrics = None
        mpath = out / "metrics.json"
        if mpath.exists():
            try:
                old_metrics = json.loads(mpath.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                old_metrics = None
        metrics = _merge_metrics(old_metrics, metrics, eval_skipped)
        mpath.write_text(json.dumps(metrics, indent=1), encoding="utf-8")

        env = write_env_report(out, ds.man.sha256, extra=dict(
            mode="train", smoke=bool(args.smoke),
            value_model=args.value_model,
            batch_size_effective=metrics.get("probe", {}).get("batch_size"),
            oom_fallback=bool(metrics.get("probe", {})
                              .get("oom_fallback")),
            oom_events=metrics.get("oom_events", [])))
        log(f"env: python={env['python']} torch={env.get('torch')} "
            f"cuda={env['cuda_available']} "
            f"gpu={env.get('gpu_name', 'none')} "
            f"git={env['git_commit'][:12]}"
            + (f" oom_fallback={env['oom_fallback']}"
               if env.get("oom_fallback") else ""))

        # bundle v1 (logged), then the final log lines, then a SILENT
        # rebuild so OUTPUT_MANIFEST hashes match the finished log
        log("outputs: writing OUTPUT_MANIFEST.json + results.zip "
            "(deliverables only; ckpt/ stays in OUT_DIR for resume)")
        finalize_outputs(out, log=None)
        zpath = out / "results.zip"
        log(f"DONE — download {zpath}")
        log(f"finished at "
            f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
        log.close()
        fin = finalize_outputs(out, log=None)      # final silent re-hash
        print(f"bundle: {fin['n_files']} deliverable files, "
              f"results.zip {fin['results_zip_mib']} MiB, sha256 "
              f"{fin['results_zip_sha256']} (verified against "
              f"OUTPUT_MANIFEST.json)")
        ok, problems = verify_outputs(out)
        print("verify: OK -- every OUTPUT_MANIFEST entry re-hashes "
              "correctly; results.zip contents match the manifest"
              if ok else f"verify: FAIL {problems}")
    else:
        log(f"partB_v0 eval: data={data} out={out} "
            f"value_model={args.value_model} features={args.features}")
        man = load_manifest(data, splits=("TRAIN", "CALIBRATION"),
                            chunks=chunks,
                            exclude_superseded=args.exclude_superseded,
                            exclude_aux=bool(args.exclude_aux),
                            verify_files=not args.no_verify, log=log)
        ds = V0Dataset(man, feature_set=args.features, log=log)
        report, eval_metrics = evaluate_run(
            EvalConfig(out_dir=str(out), device=args.device,
                       smoke=args.smoke, value_model=args.value_model,
                       feature_set=args.features,
                       compare_metrics=args.compare_metrics),
            ds, log=log)
        (out / "eval_report.md").write_text(report, encoding="utf-8")
        mpath = out / "metrics.json"
        metrics = json.loads(mpath.read_text(encoding="utf-8")) \
            if mpath.exists() else {}
        metrics["eval"] = eval_metrics
        mpath.write_text(json.dumps(metrics, indent=1),
                         encoding="utf-8")
        write_env_report(out, ds.man.sha256,
                         extra=dict(mode="eval", smoke=bool(args.smoke),
                                    value_model=args.value_model))
        log("outputs: writing OUTPUT_MANIFEST.json + results.zip")
        finalize_outputs(out, log=None)
        log(f"DONE — download {out / 'results.zip'}")
        log(f"finished at "
            f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
        log.close()
        fin = finalize_outputs(out, log=None)
        print(f"bundle: {fin['n_files']} deliverable files, "
              f"results.zip {fin['results_zip_mib']} MiB, sha256 "
              f"{fin['results_zip_sha256']}")
        ok, problems = verify_outputs(out)
        print("verify: OK -- every OUTPUT_MANIFEST entry re-hashes "
              "correctly; results.zip contents match the manifest"
              if ok else f"verify: FAIL {problems}")

    keep_alive_if_requested()
    return 0


if __name__ == "__main__":
    sys.exit(main())
