"""TRAIN mean target per tau for the v0.4a report (%-of-mean scale).

Reads ONLY TRAIN label parquets of the v0.4a root (CAL files are never
opened again after the single read; HOLDOUT is never selected by
load_manifest).  Column-projected reads; no features, no model, no refit.

Populations (matching the climatology comparator, CHANGES_V04A item 7):
  all        = every TRAIN label row of the root (strict / detector
               datasets include auxiliary sources)
  non_aux    = TRAIN label rows of non-auxiliary sources only
               (strict_noaux ablation dataset)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import wdpaths  # noqa: E402
from partB_v0.data import load_manifest  # noqa: E402

ROOT = Path(wdpaths.RESEARCH_DENSE_ROOT) / "output" / "wd01_partB_v0data_v04a"
TAUS = (30.0, 60.0, 120.0, 300.0, 570.0)


def main() -> int:
    man = load_manifest(ROOT, splits=("TRAIN",), verify_files=False,
                        log=lambda m: print(m, flush=True))
    acc = {t: dict(n_all=0, sum_all=0.0, n_primary=0, sum_primary=0.0)
           for t in TAUS}
    n_src = 0
    for key, entry in sorted(man.entries.items()):
        p = Path(wdpaths.remap(
            ROOT / f"worlds/{entry['chunk_id']}/"
                   f"{entry['world']}__{entry['source']}_labels.parquet"))
        lab = pd.read_parquet(p, columns=["tau", "total_delay_ref_s"])
        y = lab["total_delay_ref_s"].to_numpy(float)
        tau = lab["tau"].to_numpy(float)
        primary = not bool(entry.get("aux"))
        for t in TAUS:
            m = np.isclose(tau, t)
            if not m.any():
                continue
            a = acc[t]
            a["n_all"] += int(m.sum())
            a["sum_all"] += float(y[m].sum())
            if primary:
                a["n_primary"] += int(m.sum())
                a["sum_primary"] += float(y[m].sum())
        n_src += 1
    out = dict(root=str(ROOT), manifest_sha256=man.sha256,
               n_train_sources=n_src, rows_read=int(sum(a["n_all"] for a in acc.values())),
               per_tau={})
    for t in TAUS:
        a = acc[t]
        out["per_tau"][f"{t:.0f}"] = dict(
            n_all=a["n_all"], mean_all=(a["sum_all"] / a["n_all"] if a["n_all"] else None),
            n_primary=a["n_primary"],
            mean_primary=(a["sum_primary"] / a["n_primary"] if a["n_primary"] else None))
    dst = HERE / "train_target_means_v04a.json"
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["per_tau"], indent=1))
    print(f"written: {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
