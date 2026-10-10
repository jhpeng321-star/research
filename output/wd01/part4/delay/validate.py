"""WD-01 part 4 validation: truth delay labels for the 26 native roots.

For every root, t0 in {120, 150, ..., 480} s and H in {60, 120, 300, 480}
s with t0+H <= 600, computes the §4.1 total-delay components from the
truth logs, checks the consistency identities, and writes
results/delay_labels.parquet + CSV summaries + a receipt JSON.

Zero simulation: reads the partA_data npz files and the regenerated runs'
requests.rou.xml only.

    python delay/validate.py [--out results]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
if str(HERE.parents[1]) not in sys.path:      # output/wd01 (wdpaths)
    sys.path.insert(0, str(HERE.parents[1]))

import wdpaths  # noqa: E402
import delay  # noqa: E402
from delay.geometry import wm30_geometry  # noqa: E402
from delay.truth import (partA_root, sha256_file,  # noqa: E402
                         vtype_speed_factors, window_truth)

T0S = [120.0 + 30.0 * i for i in range(13)]       # 120..480
HS = [60.0, 120.0, 300.0, 480.0]
MANIFEST_SHA256_EXPECTED = ("2439004d528a860c701bfb147afcda99ab3f864a29e6187b"
                            "e721db81cc82dd27")


def consistency_checks(row: dict, wd) -> list:
    """Return a list of violation strings for one window (empty = ok)."""
    bad = []
    recomposed = (row["time_on_section_s"] - row["free_flow_ref_s"]
                  + row["wait_mainline_s"] + row["wait_ramp_s"])
    if abs(recomposed - row["total_delay_ref_s"]) > 1e-6:
        bad.append(f"component-sum identity off by "
                   f"{recomposed - row['total_delay_ref_s']:.9f}")
    recomposed_o = (row["time_on_section_s"] - row["free_flow_oracle_s"]
                    + row["wait_mainline_s"] + row["wait_ramp_s"])
    if abs(recomposed_o - row["total_delay_oracle_s"]) > 1e-6:
        bad.append("oracle component-sum identity violated")
    for w in wd.waits:
        if w.wait_s < -1e-9:
            bad.append(f"negative wait for {w.vid}: {w.wait_s}")
    # waiting == actual insertion - scheduled departure (matched ids),
    # recomputed independently
    for w in wd.waits:
        if not w.matched or not np.isfinite(w.t_ins):
            continue
        t0, t1 = row["t0"], row["t0"] + row["horizon_s"]
        expect = min(w.t_ins, t1) - max(w.t_dep, t0)
        expect = min(max(expect, 0.0), row["horizon_s"])
        if abs(expect - w.wait_s) > 1e-9:
            bad.append(f"wait mismatch for {w.vid}: {w.wait_s} != {expect}")
    return bad


def window_row(root: str, split: str, loading, wd) -> tuple[dict, list]:
    t = wd.totals()
    t.update(root=root, split=split, loading=loading)
    return t, consistency_checks(t, wd)


def validate_all(out_dir: Path | None = None) -> dict:
    t_start = time.time()
    geom = wm30_geometry()
    man = delay.manifest()
    rows, violations, per_root_extra = [], [], []
    min_wait = float("inf")
    roots = delay.roots_by_split()
    n_windows = 0
    for split in ("train", "holdout"):
        for root in roots[split]:
            t_root = time.time()
            info = man["roots"][root]
            # zero-trust: the npz must match the MANIFEST hash
            got = sha256_file(wdpaths.remap(info["file"]))
            assert got == info["sha256"], f"{root}: npz sha256 mismatch"
            rd = delay.load_root_data(root)
            assert abs(rd.window_t0 - 60.0) < 1e-9 and abs(rd.dt - 0.5) < 1e-9
            run_dir = delay.root_run_dir(root)
            schedule = delay.load_schedule(run_dir)
            speed_factors_by_vtype = vtype_speed_factors(run_dir)
            # vType label ids (pdl.TYPE_IDS order) -> speedFactor
            type_ids = {0: "aggressive_car", 1: "normal_car",
                        2: "cautious_car"}
            sf = {k: speed_factors_by_vtype[v]
                  for k, v in type_ids.items() if v in speed_factors_by_vtype}
            n_unmatched = n_queued = 0
            for t0 in T0S:
                for h in HS:
                    if t0 + h > 600.0:
                        continue
                    wd = window_truth(rd, t0, h, geom, schedule=schedule,
                                      speed_factors=sf)
                    row, bad = window_row(root, split, delay.loading_of(root),
                                          wd)
                    rows.append(row)
                    violations.extend(f"{root} t0={t0} H={h}: {b}"
                                      for b in bad)
                    if wd.waits:
                        min_wait = min(min_wait,
                                       min(w.wait_s for w in wd.waits))
                    n_unmatched += row["n_unmatched_sched_id"]
                    n_queued += row["n_queued_at_horizon"]
                    n_windows += 1
            per_root_extra.append(dict(
                root=root, split=split, seconds=round(time.time() - t_root, 1),
                n_unmatched_total=n_unmatched, n_queued_total=n_queued))
            print(f"[validate] {root:<16} {split:<7} done "
                  f"({time.time() - t_root:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    manifest_sha = sha256_file(partA_root() / "MANIFEST.json")
    receipt = dict(
        n_windows=int(len(df)),
        n_roots=int(df["root"].nunique()),
        t0s=T0S, horizons=HS,
        n_violations=len(violations), violations=violations[:50],
        sums_identity_max_abs=float(
            (df["time_on_section_s"] - df["free_flow_ref_s"]
             + df["wait_mainline_s"] + df["wait_ramp_s"]
             - df["total_delay_ref_s"]).abs().max()),
        min_wait_s=float(min_wait),
        n_unmatched_sched_id_total=int(df["n_unmatched_sched_id"].sum()),
        n_queued_at_horizon_total=int(df["n_queued_at_horizon"].sum()),
        totals_ref_s=dict(
            mean=float(df["total_delay_ref_s"].mean()),
            min=float(df["total_delay_ref_s"].min()),
            max=float(df["total_delay_ref_s"].max())),
        wall_clock_s=round(time.time() - t_start, 1),
        contract_sha256=sha256_file(
            Path(wdpaths.RD_WT_ROOT) / "wd01" / "output" / "wd01" /
            "PREDICT_CONTRACT_v1.md"),
        design_sha256=sha256_file(
            Path(wdpaths.RD_WT_ROOT) / "wd01" / "output" / "wd01" /
            "DESIGN_PARTS_A_B.md"),
        manifest_sha256=manifest_sha,
        manifest_sha256_matches_brief=(manifest_sha
                                       == MANIFEST_SHA256_EXPECTED),
        per_root=per_root_extra,
    )
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_dir / "delay_labels.parquet", index=False)
        summary = df.groupby(["root", "split", "loading"],
                             as_index=False, dropna=False)[
            ["total_delay_ref_s", "total_delay_oracle_s",
             "travel_delay_ref_s", "wait_mainline_s", "wait_ramp_s",
             "n_vehicles", "n_waiting_only"]].mean()
        summary = summary.round(3)
        summary.to_csv(out_dir / "delay_labels_summary.csv", index=False)
        by_loading = df.groupby(["loading"], as_index=False,
                                dropna=False)[
            ["total_delay_ref_s", "total_delay_oracle_s",
             "travel_delay_ref_s", "wait_mainline_s", "wait_ramp_s",
             "time_on_section_s", "free_flow_ref_s"]].mean().round(3)
        by_loading.to_csv(out_dir / "delay_labels_by_loading.csv",
                          index=False)
        (out_dir / "delay_validation_receipt.json").write_text(
            json.dumps(receipt, indent=1), encoding="utf-8")
    return receipt


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()
    rec = validate_all(Path(args.out))
    print(json.dumps({k: v for k, v in rec.items() if k != "violations"},
                     indent=1))
    if rec["n_violations"]:
        print("VIOLATIONS:", rec["violations"])
        raise SystemExit(1)
    print("OK: all consistency checks passed")
