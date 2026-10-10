"""The v0.4b TRAIN-only dry-run (registration s.2 split FIRST, <= 4 J4
TRAIN worlds, no CAL row): a role-play of the post-ack fit.

Steps:
  1. the J4 plan is loaded and the registered TRAIN/CAL name lists are
     asserted (split first; any drift refuses);
  2. the first n_worlds TRAIN worlds (sorted by name) are built:
     native/aux t0 rows from the v0.4a root (read-only) + the 4 wall
     arm t0 rows from the raw J4 runs (labels from the branch traces;
     request-time wall descriptors; planned-schedule descriptors);
  3. pairs (wall-native headline + aux contrasts) with the s.12
     duration/onset strata and the a-priori representation counts;
  4. the direct-dY quantile-GBM per tau (world-grouped CV over the
     registered grid) + conformal shifts + the wall-only ablation;
  5. TRAIN-OOB A1/A3 + stratified tables (labelled) -- NO CAL number is
     computed or printed anywhere in a dry-run.

Artifacts under partB_v04b/results/dryrun/: dryrun_log.txt, rows
parquet, pairs parquet, per-variant fit sidecars + model pickles, and
dryrun_report.json.  Zero SUMO; CPU 1 worker; thread-limited fits.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from . import _paths  # noqa: F401
from . import build_v04b as B
from . import fit_v04b as FI
from . import j4plan, reg
from . import pairs_v04b as PB
from .cal_guard import assert_train_only

DEFAULT_WORLDS = 4
DEFAULT_VARIANTS = ("deployable_strict", "deployable_detector")


class _TeeLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "w", encoding="utf-8")
        self.t0 = time.time()

    def __call__(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


def run_dry_run(n_worlds: int = DEFAULT_WORLDS,
                variants: tuple = DEFAULT_VARIANTS,
                out_dir: Path | None = None,
                data_root: str | Path | None = None) -> dict:
    out_dir = Path(out_dir) if out_dir else \
        reg.RESULTS_DIR / "dryrun"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = _TeeLog(out_dir / "dryrun_log.txt")
    log(f"WD-01-PB-V04B TRAIN-only dry-run (role-play of the post-ack "
        f"fit); registration LF-sha256 {reg.reg_sha()[:8]}...")
    log(f"worlds: {n_worlds} TRAIN (no CAL row anywhere); variants: "
        f"{', '.join(variants)}; zero SUMO")

    # -- 1. split FIRST, by name ------------------------------------------
    train = j4plan.split_by_name(data_root)
    log(f"J4 plan split asserted: {len(train)} TRAIN / "
        f"{len(j4plan.CAL_WORLDS)} CAL by name; CAL worlds are named "
        "but never opened here")
    if j4plan.SPLIT_NAME_DRIFT:
        log(f"DISCLOSED split-name drift (registration text -> plan): "
            f"{j4plan.SPLIT_NAME_DRIFT} -- the plan is the split "
            "authority; reported in every manifest")
    worlds = sorted(train)[:n_worlds]
    log(f"dry-run worlds: {', '.join(worlds)}")

    # -- 2. build the t0 rows ---------------------------------------------
    runs_root, bases_root = j4plan.runs_root(data_root), \
        j4plan.bases_root(data_root)
    v04a = B.v04a_root(data_root)
    frames, refs = [], []
    for w in worlds:
        df, ref = B.build_world_rows(w, train[w], v04a, runs_root,
                                     bases_root)
        assert_train_only(df["split"], f"dry-run build {w}")
        frames.append(df)
        refs.append(ref)
        log(f"built {w}: {len(df)} t0 rows "
            f"({sum(df['source'].isin(reg.WALL_ARMS))} wall + "
            f"{sum(~df['source'].isin(reg.WALL_ARMS))} native/aux)")
    rows = pd.concat(frames, ignore_index=True)
    PB.to_parquet_ready(rows).to_parquet(out_dir / "rows_t0.parquet", index=False)

    # -- 3. pairs ----------------------------------------------------------
    pairs = PB.build_pairs(rows)
    assert_train_only(pairs["split"], "dry-run pairs")
    PB.to_parquet_ready(pairs).to_parquet(out_dir / "pairs.parquet", index=False)
    rep = PB.representation_counts(pairs[pairs["contrast"] ==
                                         reg.HEADLINE_CONTRAST])
    log(f"pairs: {len(pairs)} rows ({pairs['contrast'].nunique()} "
        f"contrasts x {pairs['tau'].nunique()} taus); representation "
        f"(wall pairs): {json.dumps(rep)}")

    # -- 4./5. fit + TRAIN-OOB acceptance ---------------------------------
    report = dict(
        schema=reg.SCHEMA, kind="v04b_dry_run_report",
        registration_lf_sha256=reg.reg_sha(),
        worlds=worlds, n_pairs=int(len(pairs)),
        representation_counts=rep,
        cal_numbers_computed=False,
        note=("TRAIN-only dry-run: A1/A3 numbers below are the "
              "labelled TRAIN-OOB versions; the registered CAL "
              "acceptance runs only inside the guarded cal-read"),
        variants={})
    for variant in variants:
        art = FI.fit_variant(pairs, rows, variant, log=log)
        paths = FI.save_artifacts(art, rows, out_dir, log=log)
        report["variants"][variant] = dict(
            train_oob=art["train_oob"], artifacts=paths)
        a1 = art["train_oob"]["a1"]
        a3 = art["train_oob"]["a3"]
        log(f"{variant}: TRAIN-OOB A1 passes={a1['passes']} "
            f"(coverage {a1['coverage_overall']:.3f}); A3 wall-native "
            f"passes={a3['passes']} "
            f"({a3['n_taus_pass']}/{a3['n_taus_evaluated']} taus)")
    (out_dir / "dryrun_report.json").write_text(
        json.dumps(report, indent=1, default=FI._json_default),
        encoding="utf-8")
    log("dry-run complete: NO CAL row was read; the registered "
        "acceptance run needs the owner's go (README commands)")
    log.close()
    return report


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worlds", type=int, default=DEFAULT_WORLDS,
                    help="<= 4 J4 TRAIN worlds (owner brief)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    args = ap.parse_args(argv)
    n = int(args.worlds)
    if n < 1 or n > 4:
        raise SystemExit("the dry-run is bounded to <= 4 TRAIN worlds")
    run_dry_run(n_worlds=n, out_dir=args.out,
                variants=tuple(v for v in args.variants.split(",") if v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

