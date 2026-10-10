"""WD-01-PB-V04A label parity vs the frozen v0.3.2 root (read-only).

Every v0.4a source that also exists in v0.3.2 (chunk-1/2/3/fix27 bases
-> `base`, the campaign switching `__native` arms -> `swnative`) must
reproduce the v0.3.2 label rows EXACTLY at every shared (t, tau); the
only rows allowed to differ are those at t == t0 (v0.4a base rows at t0
use skip_first_tick=True; branch arms gain a t0 row).  Also checks the
v0.3.2 deployable strict/detector demand+context blocks equal the
v0.4a ones at the shared decision times.

Usage: python parity_v032.py --root <v04a root> --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import wdpaths  # noqa: E402

V032 = Path(wdpaths.RESEARCH_DENSE_ROOT) / "output" / "wd01_partB_v0data_v032"
NUM = ["time_on_section_s", "free_flow_ref_s", "traversed_m",
       "wait_mainline_s", "wait_ramp_s", "total_delay_ref_s",
       "ttc_lt3_veh_s", "hard_brake_s", "collision_any", "teleports",
       "n_queued_at_horizon", "n_vehicles_present"]
MAP = {"base": "base", "swnative": "native"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    root = Path(args.root)
    man = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    m32 = json.loads((V032 / "MANIFEST.json").read_text(encoding="utf-8"))
    rows, worst = [], 0.0
    n_cmp = n_missing = n_bad = 0
    dep_bad = 0
    for key, e in sorted(man["worlds"].items()):
        if e["source"] not in MAP or e["chunk_id"] == "chunk4":
            continue
        k32 = f"{e['chunk_id']}/{e['world']}__{MAP[e['source']]}"
        if k32 not in m32["worlds"]:
            n_missing += 1
            rows.append(dict(key=key, status="not_in_v032"))
            continue
        a = pd.read_parquet(root / "worlds" / e["chunk_id"]
                            / f"{e['world']}__{e['source']}_labels.parquet")
        b = pd.read_parquet(V032 / "worlds" / e["chunk_id"]
                            / f"{e['world']}__{MAP[e['source']]}"
                            "_labels.parquet")
        t0 = float(e["t0_s"])
        a = a[~np.isclose(a["t"], t0)].set_index(["t", "tau"]).sort_index()
        b = b[~np.isclose(b["t"], t0)].set_index(["t", "tau"]).sort_index()
        common = a.index.intersection(b.index)
        only_a = len(a.index.difference(b.index))
        only_b = len(b.index.difference(a.index))
        d = 0.0
        for c in NUM:
            d = max(d, float(np.max(np.abs(
                a.loc[common, c].to_numpy(float)
                - b.loc[common, c].to_numpy(float)))) if len(common)
                else 0.0)
        # deployable blocks at shared decision times
        for suf in ("_feat_dep_strict.npz", "_feat_dep_det.npz"):
            za = np.load(root / "worlds" / e["chunk_id"]
                         / f"{e['world']}__{e['source']}{suf}")
            zb = np.load(V032 / "worlds" / e["chunk_id"]
                         / f"{e['world']}__{MAP[e['source']]}{suf}")
            ta = {round(float(t), 6): i for i, t in enumerate(za["t"])}
            for j, t in enumerate(zb["t"]):
                i = ta.get(round(float(t), 6))
                if i is None or np.isclose(t, t0):
                    continue
                if not (np.allclose(za["demand"][i], zb["demand"][j],
                                    atol=1e-6)
                        and np.allclose(za["context"][i], zb["context"][j],
                                        atol=1e-6)
                        and np.array_equal(za["lane_bins"][i],
                                           zb["lane_bins"][j])):
                    dep_bad += 1
        n_cmp += 1
        ok = (d <= 1e-9 and only_a == 0 and only_b == 0)
        if not ok:
            n_bad += 1
        worst = max(worst, d)
        rows.append(dict(key=key, n_common=int(len(common)),
                         only_v04a=only_a, only_v032=only_b,
                         max_abs_diff=d, ok=bool(ok)))
    res = dict(schema="wd01_pb_v04a_parity_v032/1", n_compared=n_cmp,
               n_not_in_v032=n_missing, n_mismatch=n_bad,
               deployable_block_mismatches=dep_bad,
               max_abs_label_diff=worst, rows=rows,
               ok=bool(n_bad == 0 and dep_bad == 0 and n_cmp > 0))
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"parity vs v0.3.2: {n_cmp} sources compared, {n_bad} label "
          f"mismatches, {dep_bad} deployable-block mismatches, "
          f"{n_missing} not in v0.3.2; max |diff| {worst:.3g}; "
          f"{'OK' if res['ok'] else 'FAIL'}")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
