"""TD-tuple consistency report over the built V0 dataset.

Checks, per source, the identity required by the TD construction:
    label[t, tau] == D[t, t+30] + label[t+30, tau-30]
on every complete tuple of the released parquets (within one realization
this holds exactly; boundary effects are excluded because both sides are
required complete).  Writes results/td_consistency.json and prints a
per-tau summary.  Exit code 1 on any violation beyond TOL.

Usage:  python td_report.py [--out DIR] [--tol 1e-6]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def check_source(lab_path: Path, td_path: Path, tol: float):
    lab = pd.read_parquet(lab_path)
    td = pd.read_parquet(td_path)
    L = {(float(r.t), float(r.tau)): float(r.total_delay_ref_s)
         for r in lab.itertuples()}
    d30 = {(float(r.t)): float(r.d30_s)
           for r in td[td["tau"] == 30.0].itertuples()}
    per_tau = {}
    n_checked = n_missing_next = 0
    worst = 0.0
    violations = []
    for r in td.itertuples():
        if not r.label_mc_complete:
            continue
        tau = float(r.tau)
        t = float(r.t)
        if tau >= 60.0:
            # direct one-step identity where tau-30 is on the tau grid,
            # else the equivalent chained identity: repeated application of
            # the 30 s recursion down to the tau=60 label
            nxt = L.get((t + 30.0, tau - 30.0))
            if nxt is not None:
                pred = float(r.d30_s) + nxt
                steps = 1
            else:
                k = int(round((tau - 60.0) / 30.0))     # chain down to 60
                anchor = L.get((t + 30.0 * k, 60.0))
                if anchor is None:
                    n_missing_next += 1
                    continue
                chain = []
                ok = True
                for j in range(k):
                    dj = d30.get(t + 30.0 * j)
                    if dj is None:
                        ok = False
                        break
                    chain.append(dj)
                if not ok:
                    n_missing_next += 1
                    continue
                pred = anchor + sum(chain)
                steps = k + 1
        else:
            continue                    # tau=30: pure reward, no recursion
        d = abs(float(r.label_mc_s) - pred)
        per_tau.setdefault(tau, []).append(d)
        if d > tol:
            violations.append(dict(t=t, tau=tau, diff=float(d),
                                   steps=steps))
        worst = max(worst, d)
        n_checked += 1
    summary = {}
    for tau, ds in sorted(per_tau.items()):
        ds = np.asarray(ds)
        summary[tau] = dict(n=int(len(ds)),
                            max_diff=float(ds.max()),
                            mean_diff=float(ds.mean()),
                            violations=int((ds > tol).sum()))
    return dict(n_checked=n_checked, n_missing_next=n_missing_next,
                worst_diff=worst, n_violations=len(violations),
                violations=violations[:20], per_tau=summary)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args(argv)
    out = Path(args.out)
    man = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    report = dict(schema="wd01_partB_v0data_td_consistency/1",
                  generated_utc=datetime.now(timezone.utc).isoformat(),
                  tol=args.tol, sources={})
    tot_checked = tot_viol = 0.0
    worst_all = 0.0
    for key, entry in sorted(man["worlds"].items()):
        files = entry["files"]
        lab_rel = next(f for f in files if f.endswith("_labels.parquet"))
        td_rel = next(f for f in files if f.endswith("_td.parquet"))
        rep = check_source(out / lab_rel, out / td_rel, args.tol)
        report["sources"][key] = rep
        tot_checked += rep["n_checked"]
        tot_viol += rep["n_violations"]
        worst_all = max(worst_all, rep["worst_diff"])
    report["summary"] = dict(n_sources=len(man["worlds"]),
                             n_tuples_checked=int(tot_checked),
                             n_violations=int(tot_viol),
                             worst_diff=worst_all)
    results = out / "results"
    results.mkdir(exist_ok=True)
    (results / "td_consistency.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report["summary"], indent=1))
    for key, rep in report["sources"].items():
        pt = "; ".join(f"tau={t}: n={v['n']} max={v['max_diff']:.2e} "
                       f"viol={v['violations']}"
                       for t, v in rep["per_tau"].items())
        print(f"{key}: checked={rep['n_checked']} worst={rep['worst_diff']:.2e}"
              f" {pt}")
    return 1 if tot_viol else 0


if __name__ == "__main__":
    sys.exit(main())
