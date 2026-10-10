"""Generate DATA_CARD_V0.md from the built V0 dataset (MANIFEST + parquets).

v0.2: consolidated chunk-1 + chunk-2 dataset.  Adds split x chunk x source
counts, per (chunk, source, split) label counts by tau, the native
tau=570 TRAIN count (flagged if < 150), and the per-chunk speed-mode
resolution census (receipt -> observed -> config-default fallback).

v0.3: chunk3 (incremental) + chunk1_fix27 sources; superseded-source
counts (chunk-1 p40 marked superseded_by='chunk1_fix27', files kept,
excluded from training by default); native tau=570 TRAIN counts both as
built and excluding superseded rows.

Usage:  python make_data_card.py [--out DIR] [--version v0.3]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
TAUS = (30.0, 60.0, 120.0, 300.0, 570.0)
SPLITS = ("TRAIN", "CALIBRATION")
EXPECTED_MODE = {"chunk1": 31, "chunk2": 27, "chunk3": 27,
                 "chunk1_fix27": 27}


def f1(x):
    return f"{x:.1f}"


def label_stats(lab: pd.DataFrame) -> pd.DataFrame:
    g = lab.groupby(["chunk_id", "source", "split", "tau"])[
        "total_delay_ref_s"]
    return g.agg(n="count", mean="mean", std="std",
                 p50=lambda s: s.quantile(0.5),
                 p90=lambda s: s.quantile(0.9),
                 max="max").reset_index()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--version", default="v0.2",
                    help="dataset version for the title (v0.2 | v0.3)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    version = args.version
    man = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    chunks = man.get("chunks", {})
    worlds = man["worlds"]
    sup_keys = sorted(k for k, e in worlds.items() if e.get("superseded_by"))
    sup_worlds = {worlds[k]["world"] for k in sup_keys}

    frames = []
    for key, entry in sorted(worlds.items()):
        rel = next(f for f in entry["files"]
                   if f.endswith("_labels.parquet"))
        frames.append(pd.read_parquet(out / rel))
    lab = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    lab_active = lab[~lab["world"].isin(sup_worlds)] \
        if len(lab) else lab

    # ---------------- split x chunk x source counts ---------------------

    if version == "v0.3":
        head = [
            f"# WD-01 Part B round 0 - V0 (native-policy value) dataset "
            f"{version} - DATA_CARD", "",
            "VERSIONED v0.3: chunk 1 (WM-44, pre-fix fleet) and chunk 2",
            "(WM-47, post-fix fleet) copied from the immutable v0.2 root",
            "with every file sha256-verified before and after the copy;",
            "chunk 3 (WM-54, plan seed 20261010, post-fix fleet) built",
            "incrementally as its campaign runs complete; chunk1_fix27",
            "(speedMode-27 re-runs of the chunk-1 p40 worlds) built when",
            "the fix27 runs land -- the chunk-1 p40 sources are marked",
            "superseded_by='chunk1_fix27' and dropped from training by",
            "the default --exclude-superseded (their files stay on disk).",
            "ZERO SIMULATION: only the campaign run logs, the plans and",
            "the frozen network XML are read.  HOLDOUT worlds are never",
            "opened (listed in MANIFEST skipped_holdout; guarded in",
            "code).",
        ]
    else:
        head = [
            f"# WD-01 Part B round 0 - V0 (native-policy value) dataset "
            f"{version} - DATA_CARD", "",
            "CONSOLIDATED v0.2: chunk 1 (WM-44, pre-fix fleet) copied from the",
            "wd01n worktree with every file sha256-verified against the old",
            "MANIFEST before and after the copy, plus chunk 2 (WM-47, post-fix",
            "fleet) newly built with the chunk-agnostic builder.  ZERO",
            "SIMULATION: only the campaign run logs, the plans and the frozen",
            "network XML are read.  HOLDOUT worlds are never opened (listed in",
            "MANIFEST skipped_holdout; guarded in code).",
        ]
    lines = list(head)
    lines += [
        "",
        "Labels: contract §4.1 total delay over [t, t+tau] (Monte-Carlo on",
        "the logged trajectory, part4 delay library conventions,",
        "equivalence tested), tau in {30, 60, 120, 300, 570} s, complete",
        "horizons only (tau_available recorded; truncated horizons",
        "dropped).  Decision times every 10 s: base source over [120,",
        "t0+570], native source over (t0, t0+570].  TD tuples (s_t,",
        "D[t,t+30], s_{t+30}, tau) at commit 30 s.  Every row carries",
        "chunk_id, cav_speed_mode, pre_fix = (cav_speed_mode == 31), p40 =",
        "(penetration == 0.4), plus TTC<3 exposure (veh·s), hard-brake",
        "seconds (a < -3 m/s²) and collision/teleport flags over the same",
        "window (V_exp / V_brake / P_coll anchors).", "",
        "## Provenance and caveats (read before training)", "",
        "1. **Two sources per world.** `source=base` is the campaign",
        "   incumbent arm: fleet-uniform **damping** (WM-44/WM-47",
        "   DATA_CONTRACT; no controller internals logged).",
        "   `source=native` is the native-IDM branch.  DESIGN PartB v2",
        "   §4 Step 0 includes both; `policy` and `source` let downstream",
        "   training filter (e.g. native-only).",
        "2. **CRN overlap.** The native branch restores the base state at",
        "   t0: post-t0 base and native windows are strongly overlapping",
        "   realizations of the same traffic.  Treat cross-source samples",
        "   as correlated, not independent.",
        "3. **Two fleets.** Chunk 1 is PRE-FIX: cav_speed_mode = 31 on",
        "   every chunk-1 row.  Chunk 2 is POST-FIX: cav_speed_mode = 27",
        "   on every chunk-2 row (WM-47 INDEX; per-run resolution",
        "   receipt -> observed physics values -> per-chunk config",
        "   default).  Do not mix fleets without the pre_fix flag.",
        "4. **Entry waiting dominates at high demand.** Labels include",
        "   entry waiting from the scheduled departures (contract §4.1);",
        "   at m2600-2700 the mainline queue is the main delay term.",
        "5. HOLDOUT worlds are NOT built and their run files are NEVER",
        "   opened (chunk 1: 14 listed, chunk 2: 14 listed).  Worlds with",
        "   incomplete receipts are listed in skipped_incomplete and",
        "   built on their available sources only.",
        "6. vType is supervision-only and is not featurised (tested).",
        "",
        "## Split x chunk x source counts", "",
        "| chunk | split | worlds | sources | decision times |"
        " label rows | TD rows |", "| --- | --- | --- | --- | --- |"
        " --- | --- |"]
    for chunk_id in sorted(chunks):
        for split in SPLITS:
            keys = [k for k, e in worlds.items()
                    if e["chunk_id"] == chunk_id and e["split"] == split]
            if not keys:
                continue
            sub = lab[lab["chunk_id"] == chunk_id]
            sub = sub[sub["split"] == split] if len(sub) else sub
            lines.append(
                f"| {chunk_id} | {split} | "
                f"{len({e['world'] for e in (worlds[k] for k in keys)})} | "
                f"{len(keys)} | "
                f"{sum(worlds[k]['rows']['n_decision_times'] for k in keys)}"
                f" | {len(sub)} | "
                f"{sum(worlds[k]['rows']['n_td_rows'] for k in keys)} |")

    # ---------------- label counts by tau per chunk x source x split ----
    lines += ["", "## Label counts by tau (complete-horizon rows)", "",
              "| chunk | source | split | " +
              " | ".join(f"tau={t:.0f}" for t in TAUS) + " | total |",
              "| " + "--- | " * (3 + len(TAUS) + 1)]
    ct = lab.groupby(["chunk_id", "source", "split", "tau"]).size() \
        if len(lab) else pd.Series(dtype=int)
    for chunk_id in sorted(chunks):
        for source in ("base", "native"):
            for split in SPLITS:
                ns = [int(ct.get((chunk_id, source, split, t), 0))
                      for t in TAUS]
                if sum(ns) == 0:
                    continue
                lines.append(f"| {chunk_id} | {source} | {split} | " +
                             " | ".join(str(n) for n in ns) +
                             f" | {sum(ns)} |")
    # native tau=570 on TRAIN, explicit (flag if < 150); v0.3 also counts
    # the ACTIVE rows (excluding superseded worlds) the trainer sees
    ct_active = lab_active.groupby(["chunk_id", "source", "split", "tau"]) \
        .size() if len(lab_active) else pd.Series(dtype=int)
    lines += ["", "### Native tau=570 label rows on TRAIN (headroom check, "
              "flag if < 150)", ""]
    for chunk_id in sorted(chunks):
        n570 = int(ct.get((chunk_id, "native", "TRAIN", 570.0), 0))
        a570 = int(ct_active.get((chunk_id, "native", "TRAIN", 570.0), 0))
        flag = "" if a570 >= 150 else "  **FLAG: < 150 (active)**"
        suffix = (f" (active after supersession exclusion: **{a570}**)"
                  if n570 != a570 else "")
        lines.append(f"- chunk {chunk_id}: **{n570}** native TRAIN rows at "
                     f"tau=570{suffix}{flag}")

    # ---------------- speed-mode resolution census ----------------------
    lines += ["", "## cav_speed_mode resolution per chunk (receipt -> "
              "observed-cav -> observed -> config fallback)", "",
              "| chunk | receipt | observed-cav | observed | config "
              "(fallback) | n/a (pre-chunk entries) |",
              "| --- | --- | --- | --- | --- | --- |"]
    for chunk_id in sorted(chunks):
        c = Counter(e.get("speed_mode_source", "n/a")
                    for e in worlds.values() if e["chunk_id"] == chunk_id)
        lines.append(f"| {chunk_id} | {c.get('receipt', 0)} | "
                     f"{c.get('observed-cav', 0)} | "
                     f"{c.get('observed', 0)} | "
                     f"{c.get('config', 0) + c.get('config-default', 0)} |"
                     f" {c.get('n/a', 0)} |")
    lines += ["",
              "Resolution tiers: run receipt (config.speed_mode, or the",
              "chunk-3-style top-level cav_speed_mode key) -> speed modes",
              "observed in the CAV physics rows (single distinct value;",
              "chunk 2 receipts record null while CAV rows show 27 and",
              "HDVs 31) -> all physics rows (chunk 1) -> per-chunk config",
              "default (chunk 2: 27).  Chunk-1 entries carry 'observed'",
              "from the flag backfill (single physics-row value 31 = the",
              "pre-fix fleet mode).", ""]

    # ---------------- per demand / penetration --------------------------
    by_dem = lab.groupby(["chunk_id", "split", "source", "demand_veh_h",
                          "penetration"]).agg(
        worlds=("world", "nunique"), label_rows=("world", "size"),
        mean_delay=("total_delay_ref_s", "mean")).reset_index()
    lines += ["## Counts per demand and penetration (label rows)", "",
              "| chunk | split | source | demand (veh/h) | penetration |"
              " worlds | label rows | mean delay (veh·s) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in by_dem.itertuples():
        lines.append(f"| {r.chunk_id} | {r.split} | {r.source} | "
                     f"{r.demand_veh_h} | {r.penetration} | {r.worlds} | "
                     f"{r.label_rows} | {f1(r.mean_delay)} |")

    # ---------------- chunk / flag counts -------------------------------
    lines += ["", "## Chunk and fix-flag counts", "",
              "| chunk | sources | label rows | active label rows "
              "(excl. superseded) | TD rows | pre_fix rows "
              "(cav_speed_mode==31) | p40 rows (penetration==0.4) | "
              "cav_speed_mode values |", "| --- | --- | --- | --- | --- |"
              " --- | --- | --- |"]
    for chunk_id, sec in sorted(chunks.items()):
        keys = set(sec.get("world_keys", []))
        sub = lab[lab["chunk_id"] == chunk_id] if len(lab) else lab
        sub_active = lab_active[lab_active["chunk_id"] == chunk_id] \
            if len(lab_active) else lab_active
        modes = sorted(int(v) for v in sub["cav_speed_mode"].unique()) \
            if len(sub) else []
        assert modes in ([], [EXPECTED_MODE.get(chunk_id)]), \
            f"{chunk_id}: unexpected fleet modes {modes}"
        lines.append(
            f"| {chunk_id} | {len(keys)} | {len(sub)} | {len(sub_active)} | "
            f"{sum(worlds[k]['rows']['n_td_rows'] for k in keys)} | "
            f"{int(sub['pre_fix'].sum()) if len(sub) else 0} | "
            f"{int(sub['p40'].sum()) if len(sub) else 0} | "
            f"{modes or '-'} |")
    lines += ["pre_fix = (cav_speed_mode == 31): chunk-1 rows are all",
              "pre-fix; chunk-2/3 and chunk1_fix27 rows are all post-fix",
              "(27).  p40 flags the 40% penetration block.  Active label",
              "rows exclude superseded worlds (the rows a default trainer",
              "sees).", ""]

    # ---------------- superseded sources (v0.3) --------------------------
    if sup_keys:
        sup_lab = lab[lab["world"].isin(sup_worlds)] if len(lab) else lab
        per_chunk = Counter(worlds[k]["chunk_id"] for k in sup_keys)
        lines += ["## Superseded sources (files kept, excluded from "
                  "training by default)", "",
                  f"{len(sup_keys)} sources ({sum(per_chunk.values())} per "
                  f"chunk: {dict(sorted(per_chunk.items()))}) marked "
                  f"superseded_by='chunk1_fix27': "
                  f"{len(sup_lab)} label rows, "
                  f"{sum(worlds[k]['rows']['n_td_rows'] for k in sup_keys)} "
                  "TD tuples stay on disk but are dropped by the trainer's "
                  "default --exclude-superseded.", ""]
        wkeys = sorted(sup_worlds)
        lines += ["Superseded worlds (chunk-1 p40, TRAIN/CAL): "
                  + ", ".join(wkeys), ""]
        bj = out / "results" / "build_v03_last.json"
        if bj.exists():
            missing = json.loads(bj.read_text(encoding="utf-8")) \
                .get("missing_fix27_worlds", [])
            lines += [f"Worlds still missing their chunk1_fix27 re-run: "
                      f"{len(missing)}"
                      + (": " + ", ".join(missing) if missing else ""),
                      ""]

    # ---------------- label distributions -------------------------------
    lines += ["## Label distributions per tau (total delay, veh·s)", "",
              "| chunk | source | split | tau (s) | n | mean | std | p50 |"
              " p90 | max |", "| --- | --- | --- | --- | --- | --- | --- |"
              " --- | --- | --- |"]
    for r in label_stats(lab).itertuples():
        lines.append(f"| {r.chunk_id} | {r.source} | {r.split} | "
                     f"{r.tau:.0f} | {r.n} | {f1(r.mean)} | {f1(r.std)} | "
                     f"{f1(r.p50)} | {f1(r.p90)} | {f1(r.max)} |")

    # ---------------- TD consistency ------------------------------------
    lines += ["", "## TD-tuple consistency (label[t,tau] = D30 + "
              "label[t+30, tau-30])", ""]
    tdj = out / "results" / "td_consistency.json"
    if tdj.exists():
        rep = json.loads(tdj.read_text(encoding="utf-8"))
        s = rep["summary"]
        lines += [
            f"Checked {s['n_tuples_checked']} complete tuples across "
            f"{s['n_sources']} sources; violations beyond tol "
            f"{rep['tol']:.0e}: **{s['n_violations']}**; worst |diff| "
            f"{s['worst_diff']:.3e} veh·s (identity holds to float noise).",
            "Per-source numbers in results/td_consistency.json.", ""]
    else:
        lines += ["td_report.py not yet run.", ""]

    # exposure summary per tau
    lines += ["## Window exposure per tau (all sources, sums)", "",
              "| tau (s) | rows | TTC<3 exposure (veh·s) | hard brake (s) |"
              " rows w/ collision |", "| --- | --- | --- | --- | --- |"]
    for tau in TAUS:
        sub = lab[lab["tau"] == tau]
        lines.append(f"| {tau:.0f} | {len(sub)} | "
                     f"{f1(sub['ttc_lt3_veh_s'].sum())} | "
                     f"{f1(sub['hard_brake_s'].sum())} | "
                     f"{int(sub['collision_any'].sum())} |")

    # build quality
    n_imp = sum(e["trace_stats"]["n_implausible_boundary"]
                for e in worlds.values())
    n_off = sum(e["trace_stats"]["n_offgrid_rows"] for e in worlds.values())
    n_unmap = sum(e["trace_stats"].get("n_unmapped_s", 0)
                  for e in worlds.values())
    n_exact = sum(e["insertion_crosscheck"]["n_exact"]
                  for e in worlds.values())
    n_mis = sum(e["insertion_crosscheck"].get("n_mismatch", 0)
                for e in worlds.values())
    lines += ["", "## Build quality", "",
              f"- physics rows off the 0.5 s grid: {n_off} (all counted, "
              "none silently dropped)",
              f"- unmapped/unresolvable lane records (s would be NaN): "
              f"{n_unmap}",
              f"- lane-change-at-boundary steps: "
              f"{sum(e['trace_stats']['n_boundary_change_steps'] for e in worlds.values())};"
              f" implausible advances: {n_imp}",
              f"- realised insertions cross-checked vs insertions.jsonl: "
              f"{n_exact} exact, {n_mis} mismatches",
              "", "## Skipped", ""]
    for chunk_id, sec in sorted(chunks.items()):
        lines.append(f"- [{chunk_id}] HOLDOUT worlds (never built or "
                     "opened): "
                     f"{', '.join(s['world'] for s in sec.get('skipped_holdout', []))}")
        if sec.get("skipped_incomplete"):
            lines.append(f"- [{chunk_id}] incomplete receipts (source "
                         "skipped): " + "; ".join(
                f"{s['world']}__{s['source']} ({s['reason']})"
                for s in sec["skipped_incomplete"]))
    lines += ["", "Machine-readable copies: MANIFEST.json, "
              "results/td_consistency.json, results/sanity_v02.json, "
              "results/verify_v0data.json.", ""]
    (out / "DATA_CARD_V0.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"DATA_CARD_V0.md written ({len(lab)} label rows, "
          f"{len(worlds)} sources, chunks {sorted(chunks)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
