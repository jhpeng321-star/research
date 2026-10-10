"""python -m partB_v04b <cmd> -- the v0.4b entry points (run from
output/wd01 so the package resolves).

  selfcheck  read-only binding + split integrity print
  dry-run    TRAIN role-play on <= 4 J4 TRAIN worlds (no CAL row)
  build      the post-ack TRAIN build of the NEW data root (COPY of the
             v0.4a sources + wall t0 rows; MANIFEST + SHA256SUMS)
  fit        full TRAIN fit after the build (still no CAL row)
  cal-read   THE ONE CAL read: OWNER_ACK-gated, read-once-marker-guarded,
             writes REPORT_V04B inputs; refuses without the owner's go
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import _paths  # noqa: F401  (sys.path shims)


def cmd_selfcheck(args) -> int:
    from . import j4plan, reg
    print(f"registration: {reg.REG_PATH}")
    print(f"  LF-sha256    {reg.reg_sha()}")
    print(f"  binding      {reg.REG_SHA_EXPECTED_PREFIX}... OK")
    train = j4plan.split_by_name(args.data_root)
    cal = j4plan.cal_worlds(args.data_root)
    print(f"  J4 split     {len(train)} TRAIN / {len(cal)} CAL by name OK")
    print(f"  taus {reg.TAUS}")
    print(f"  grid {reg.GRID}")
    print(f"  wall arms {reg.WALL_ARMS}")
    print("  CAL lock: dry-run/fit never touch it; cal-read requires "
          f"{reg.RESULTS_DIR / reg.OWNER_ACK_NAME}")
    return 0


def cmd_dry_run(args) -> int:
    from . import dryrun
    return dryrun.main(["--worlds", str(args.worlds)] +
                       (["--out", args.out] if args.out else []) +
                       (["--variants", args.variants]
                        if args.variants else []))


def cmd_build(args) -> int:
    from . import build_v04b as B
    from . import j4plan, reg
    from . import pairs_v04b as PB
    import wdpaths
    import v0lib as V
    root = Path(args.data_root) if args.data_root else \
        Path(wdpaths.RESEARCH_DENSE_ROOT)
    out_root = Path(args.out) if args.out else \
        Path(wdpaths.remap(root / reg.V04B_DATA_ROOT))
    if out_root.exists() and any(out_root.iterdir()) and not args.rebuild:
        raise SystemExit(f"{out_root} exists and is non-empty -- never "
                         "overwrite a prior run (use --out or --rebuild)")
    train = j4plan.split_by_name(root)
    worlds = sorted(train) if not args.worlds else \
        [w for w in args.worlds.split(",") if w]
    assert not (set(worlds) - set(train)), \
        "build worlds must be registered TRAIN worlds"
    out_root.mkdir(parents=True, exist_ok=True)
    refs = []
    B.log(f"v04b TRAIN build -> {out_root}")
    for w in worlds:
        df, ref = B.build_world_rows(w, train[w], B.v04a_root(root),
                                     j4plan.runs_root(root),
                                     j4plan.bases_root(root))
        p = out_root / "rows" / f"{w}__t0_rows.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        PB.to_parquet_ready(df).to_parquet(p, index=False)
        ref["rows_file"] = str(p)
        refs.append(ref)
        B.log(f"built {w}: {len(df)} rows")
    B.log("copying v0.4a sources (COPY only)...")
    files = B.copy_v04a_sources(out_root, worlds, B.v04a_root(root))
    man = B.finalize_root(out_root, refs, sys.argv)
    man["copied_files"] = len(files)
    (out_root / "MANIFEST.json").write_text(
        __import__("json").dumps(man, indent=1), encoding="utf-8")
    n = B.write_sha256sums(out_root)
    B.log(f"MANIFEST + SHA256SUMS written ({n} entries)")
    return 0


def cmd_fit(args) -> int:
    from . import build_v04b as B
    from . import fit_v04b as FI
    from . import j4plan, reg
    from . import pairs_v04b as PB
    from .cal_guard import assert_train_only
    import pandas as pd
    import wdpaths
    root = Path(args.data_root) if args.data_root else \
        Path(wdpaths.RESEARCH_DENSE_ROOT)
    data_root = Path(args.root) if args.root else \
        Path(wdpaths.remap(root / reg.V04B_DATA_ROOT))
    if not data_root.is_dir():
        raise SystemExit(f"{data_root} missing -- run the build first")
    rows_files = sorted((data_root / "rows").glob("*__t0_rows.parquet"))
    if args.worlds:
        want = set(args.worlds.split(","))
        rows_files = [p for p in rows_files if p.name.split("__")[0]
                      in want]
    if not rows_files:
        raise SystemExit("no rows parquet under "
                         f"{data_root / 'rows'}")
    rows = pd.concat([pd.read_parquet(p) for p in rows_files],
                     ignore_index=True)
    assert_train_only(rows["split"], "fit")
    train = j4plan.split_by_name(root)
    assert set(rows["world"]) <= set(train), \
        "fit rows must be registered TRAIN worlds"
    out_dir = Path(args.out) if args.out else reg.RESULTS_DIR / "fit"
    log = FI.log_fn(lambda m: print(f"[fit] {m}", flush=True))
    pairs = PB.build_pairs(rows)
    PB.to_parquet_ready(pairs).to_parquet(out_dir / "pairs.parquet", index=False)
    for variant in (v for v in args.variants.split(",") if v):
        art = FI.fit_variant(pairs, rows, variant, log=log)
        FI.save_artifacts(art, rows, out_dir, log=log)
    print(f"[fit] complete (TRAIN only) -> {out_dir}")
    return 0


def cmd_allocate(args) -> int:
    """Write LEDGER_DRAFT_PB_V04B_ALLOCATION.json (registration s.10:
    the allocation entry is committed BEFORE any build).  Status stays
    'planned' until the owner's ack: the registered run may not start
    before it."""
    import json
    import time
    from . import reg
    p = reg.WD01 / "LEDGER_DRAFT_PB_V04B_ALLOCATION.json"
    entry = dict(
        id="WD-01-PB-V04B",
        date=time.strftime("%Y-%m-%d"),
        family="WD",
        status="planned",
        name="Part B direct-difference tree model v0.4b on J4 wall pairs "
             "(baseline T for Idea A): build, direct-dY fit, wall-pair "
             "acceptance",
        question="Can a tree model predicting the ARM DIFFERENCE dY "
                 "DIRECTLY from deployable state features plus "
                 "request-time planned-schedule/wall descriptors give the "
                 "wall-native contrast genuine VALUE skill (dY CRPS better "
                 "than zero-contrast AND TRAIN-cell-mean) where the v0.4a "
                 "paired-level model had sign skill but no value skill?",
        controller="one quantile-GBM per (tau x sensing variant) on dY "
                   "directly (no level head, no paired predictions); "
                   "PLANNED-SCHEDULE action descriptor (19 x 30 s segments "
                   "+ wall hold/speed/cap); auxiliary chunk-4 contrasts "
                   "training only; wall-only ablation reported",
        baseline_cf="zero-contrast; TRAIN-cell-mean per demand x "
                    "penetration; ridge-on-dY same inputs; v0.4a verdict "
                    "as the level-prediction comparator",
        scenario="J4 wall pilot chunk-4 wall branches + their restored "
                 "TRUE fleet-native siblings (same world + t0 saved "
                 "state); aux chunk-4 contrasts of the SAME 44 worlds",
        seeds_splits="Split counted BY NAME from the J4 WORLD_PLAN (39 "
                     "TRAIN / 5 CAL; one recorded name drift disclosed); "
                     "world-grouped 6-fold CV on TRAIN only; the ONE CAL "
                     "read is OWNER_ACK-gated and read-once-marked; "
                     "HOLDOUT none; chunk-5 never",
        observation_contract="deployable_strict (CAV self-reports only, "
                             "HEADLINE) / deployable_detector "
                             "(SENSITIVITY) at t0 from the shared base "
                             "trace; request-time wall descriptors "
                             "(ALLOWED_RECORD_FIELDS only); no "
                             "penetration/pre_fix/speed_mode/world "
                             "strings; nothing with timestamp > t",
        privileged_info="None as model input; matched counterfactual "
                        "arms are offline labels only",
        changed_vars="direct-dY target (vs v0.4a paired level "
                     "predictions); J4 wall arms; planned-schedule "
                     "action input (owner requirement s.12)",
        fixed_vars="tau set, quantile grid, the fixed 6-candidate tuning "
                   "grid, world-grouped CV, conformal CalMap method, "
                   "CRPS code path (K=40), split rule, CAL guard",
        comparators="zero-contrast; TRAIN-cell-mean; ridge-on-dY; "
                    "wall-only ablation; v0.4a A3 records",
        matching_reuse="Closest prior WD-01-PB-V04A (same pairing "
                       "machinery + A3 rule; verdict FAIL on every "
                       "variant). Reuses partB_v0/partB_v0data read-only "
                       "(CRPS, bootstrap, references, reconstruction); "
                       "new data root by COPY.",
        requested_vs_delivered="Registration "
                               "output/wd01/REGISTRATION_PARTB_V04B.md, "
                               f"LF sha256 {reg.reg_sha()}; owner finalize "
                               "instruction 2026-10-01; executor brief "
                               "authorizes code+tests+dry-run only",
        budget_registered="ZERO SUMO; CPU, 1 worker, BelowNormal; "
                          "declared cap 6 CPU-hours total for the "
                          "registered run (build <= 1 h, tuning + 2 "
                          "variant fits + ablation <= 1.5 h, CAL read + "
                          "report <= 0.5 h)",
        budget_actual="not started (planned; the registered build/fit/"
                      "CAL read begin only after the owner's go; the "
                      "pre-ack TRAIN-only dry-run stays inside "
                      "partB_v04b/results/dryrun)",
        attempts="",
        results="Not run yet (planned).",
        criteria_verdict="Pending (ACCEPT/PARTIAL/FAIL per registration "
                         "section 6)",
        limitations="Small n (39 TRAIN worlds / 156 wall pairs per tau; "
                    "20 CAL pairs); wall law v2.1 data (never claims "
                    "v2.2); ideal-sensing premise; short commitments "
                    "barely represented until J-PULSE; split-name drift "
                    "w8662 disclosed",
        artifacts=[
            "output/wd01/REGISTRATION_PARTB_V04B.md",
            "output/wd01/partB_v04b (code + tests, this commit)",
            "E:/research-dense/output/wd01_partB_v0data_v04b (new data "
            "root, copy only)",
            "E:/research-dense/output/partB_results/v0.4b_strict",
            "E:/research-dense/output/partB_results/v0.4b_detector",
            "E:/research-dense-backups/wd01/ (backup of the new root)",
        ],
        review_status="registered FINAL by the world model session; "
                      "executor task: code + tests ready, fit awaits the "
                      "owner's go")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps([entry], indent=2) + "\n",
                   encoding="utf-8")
    tmp.replace(p)
    print(f"allocation entry -> {p}")
    print("commit it BEFORE the build: git add -f "
          f"{p.name} && git commit -m 'wd01: PB-V04B allocation entry'")
    return 0


def cmd_cal_read(args) -> int:
    from . import cal_read_v04b
    return cal_read_v04b.main(args)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="partB_v04b",
                                 description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("selfcheck", cmd_selfcheck),
                     ("allocate", cmd_allocate),
                     ("dry-run", cmd_dry_run), ("build", cmd_build),
                     ("fit", cmd_fit), ("cal-read", cmd_cal_read)):
        sp = sub.add_parser(name)
        sp.add_argument("--data-root", default=None,
                        help="E:/research-dense root override")
        sp.add_argument("--out", default=None)
        if name == "dry-run":
            sp.add_argument("--worlds", type=int, default=4)
            sp.add_argument("--variants", default=None)
        if name == "build":
            sp.add_argument("--worlds", default="")
            sp.add_argument("--rebuild", action="store_true")
        if name == "fit":
            sp.add_argument("--root", default=None,
                            help="v04b data root override")
            sp.add_argument("--worlds", default="")
            sp.add_argument("--variants",
                            default="deployable_strict,"
                                    "deployable_detector")
        if name == "cal-read":
            sp.add_argument("--root", default=None)
            sp.add_argument("--ack", default=None,
                            help="OWNER_ACK json path override")
            sp.add_argument("--variants",
                            default="deployable_strict,"
                                    "deployable_detector")
        sp.set_defaults(fn=fn)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())

