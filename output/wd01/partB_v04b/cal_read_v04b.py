"""THE ONE CAL read of WD-01-PB-V04B (registration s.6/s.7).

Runs ONLY after the owner's go: the OWNER_ACK file (bound to this
registration's full LF-sha256) is checked, then the read-once marker
CAL_READ_ONCE_V04B.json is written BEFORE the first CAL row.  The 5 CAL
worlds are built exactly like the TRAIN worlds (same row builders), the
FROZEN fit models predict, the registered A1/A3 acceptance + verdict +
stratified tables are computed and written -- and nothing is refitted,
reselected or recalibrated afterwards (a crash before the read may be
continued; a crash after it is reported as is).
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import pandas as pd

from . import _paths  # noqa: F401
from . import acceptance_v04b as AC
from . import build_v04b as B
from . import descriptors_v04b as DE
from . import fit_v04b as FI
from . import j4plan, reg
from . import pairs_v04b as PB
from .cal_guard import CalReadOnce, require_owner_ack


def main(args) -> int:
    import wdpaths
    root = Path(args.data_root) if args.data_root else \
        Path(wdpaths.RESEARCH_DENSE_ROOT)
    data_root = Path(args.root) if args.root else \
        Path(wdpaths.remap(root / reg.V04B_DATA_ROOT))
    out_dir = Path(args.out) if args.out else reg.RESULTS_DIR / "cal_read"
    fit_dir = out_dir.parent / "fit"

    # ---- the two locks (both bound to the registration) ----
    ack = require_owner_ack(getattr(args, "ack", None))
    print(f"[cal-read] OWNER_ACK accepted: {ack.get('ack', '')[:80]}")
    cal_read = CalReadOnce(results_dir=out_dir)
    if cal_read.spent():
        raise SystemExit("the ONE CAL read of this registration is "
                         "already spent (marker finished) -- no second "
                         "read, no refit after the CAL read")
    if not data_root.is_dir():
        raise SystemExit(f"{data_root} missing -- run build + fit first")

    rows_files = sorted((data_root / "rows").glob("*__t0_rows.parquet"))
    train_rows = pd.concat([pd.read_parquet(p) for p in rows_files],
                           ignore_index=True)
    train_pairs = PB.build_pairs(train_rows)
    ref = AC.train_references_v04b(
        train_pairs[train_pairs["split"] == "TRAIN"])

    report = dict(
        schema=reg.SCHEMA, kind="v04b_acceptance_report",
        registration_lf_sha256=reg.reg_sha(),
        owner_ack=ack,
        train_oob=dict(label="TRAIN-OOB (in-sample-selection-free)"),
        variants={})
    cal_read.begin()              # marker BEFORE the first CAL row
    print("[cal-read] marker written; reading the 5 CAL worlds")
    cal = j4plan.cal_worlds(root)
    frames = []
    for w, rec in sorted(cal.items()):
        df, _ref = B.build_world_rows(w, rec, B.v04a_root(root),
                                      j4plan.runs_root(root),
                                      j4plan.bases_root(root))
        frames.append(df)
    cal_rows = pd.concat(frames, ignore_index=True)
    cal_pairs = PB.build_pairs(cal_rows)
    report["n_cal_pairs"] = int(len(cal_pairs))
    report["cal_worlds"] = sorted(cal)

    variants = tuple(v for v in args.variants.split(",") if v)
    for variant in variants:
        with open(fit_dir / f"model_{variant}.pkl", "rb") as fh:
            models = pickle.load(fh)
        v_side = json.loads(
            (fit_dir / f"fit_sidecar_{variant}.json").read_text(
                encoding="utf-8"))
        report["train_oob"][variant] = v_side["train_oob"]

        def cal_fn(g, _variant=variant):
            tau = float(g["tau"].iloc[0])
            m = models[str(tau)]
            X = DE.pair_matrix(g, cal_rows, _variant)
            return m.predict_quantiles(X, tau)

        a1 = AC.a1_records(cal_pairs, cal_fn)
        a3_recs: list = []
        strata: list = []
        for tau in reg.TAUS:
            a3_recs.extend(AC.a3_records(cal_pairs, tau, cal_fn, ref))
            strata.extend(AC.stratified_records(cal_pairs, tau, cal_fn,
                                                ref))
        a3 = AC.a3_verdict(a3_recs)
        aux = [r for r in a3_recs if r["contrast"] != reg.HEADLINE_CONTRAST]
        abl_recs: list = []
        for tau in reg.TAUS:
            key = str(("ablation", float(tau)))
            if key not in models:
                continue

            def abl_fn(g, _m=models[key], _tau=float(tau)):
                X = DE.pair_matrix(g, cal_rows, variant)
                return _m.predict_quantiles(X, _tau)

            abl_recs.extend(AC.a3_records(cal_pairs, tau, abl_fn, ref))
        abl_a3 = AC.a3_verdict(abl_recs)
        report["variants"][variant] = dict(
            a1=a1, a3_records=a3_recs, a3=a3, aux_records=aux,
            ablation=dict(a3_records=abl_recs, a3=abl_a3,
                          label="wall-only (no-aux) ablation"),
            strata=strata,
            verdict=AC.verdict(a1, a3))
        print(f"[cal-read] {variant}: A1={a1['passes']} "
              f"A3={a3['passes']} -> VERDICT "
              f"{report['variants'][variant]['verdict']['verdict']}")
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                           time.gmtime())
    cal_read.finish()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "acceptance_report.json").write_text(
        json.dumps(report, indent=1, default=FI._json_default),
        encoding="utf-8")
    print(f"[cal-read] acceptance report -> "
          f"{out_dir / 'acceptance_report.json'} (read-once marker "
          "finished; NO refit after the CAL read)")
    return 0
