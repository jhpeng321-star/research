"""Real-data tests: chunk/fix-flag columns on all rows, TD-tuple
consistency and library equivalence on the built V0 dataset (skipped when
the dataset is not built)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import v0lib as V
from build_v0data import DEFAULT_PLAN, DEFAULT_RUNS
from delay import core

TOL = 1e-6


def _pair(out: Path, entry: dict):
    lab = out / next(f for f in entry["files"]
                     if f.endswith("_labels.parquet"))
    td = out / next(f for f in entry["files"] if f.endswith("_td.parquet"))
    return pd.read_parquet(lab), pd.read_parquet(td)


def test_chunk_and_flag_columns_on_all_rows(built_available):
    here = Path(built_available)
    man = json.loads((here / "MANIFEST.json").read_text(encoding="utf-8"))
    n_rows = 0
    for key, entry in sorted(man["worlds"].items()):
        lab, td = _pair(here, entry)
        for df in (lab, td):
            assert df["chunk_id"].eq(entry["chunk_id"]).all()
            assert df["cav_speed_mode"].eq(entry["cav_speed_mode"]).all()
            assert df["pre_fix"].eq(df["cav_speed_mode"] == 31).all()
            assert df["p40"].eq(
                (df["penetration"] - 0.4).abs() < 1e-9).all()
            n_rows += len(df)
    assert n_rows > 50000
    # per-chunk fleet ground truth: chunk 1 pre-fix speedMode 31, chunk 2
    # post-fix speedMode 27 (WM-47; INDEX cav_speed_mode 27)
    expected_mode = {"chunk1": 31, "chunk2": 27}
    modes_by_chunk = {}
    for entry in man["worlds"].values():
        modes_by_chunk.setdefault(entry["chunk_id"], set()).add(
            entry["cav_speed_mode"])
    for chunk, modes in modes_by_chunk.items():
        assert chunk in expected_mode, f"unknown chunk {chunk}"
        assert modes == {expected_mode[chunk]}, \
            f"{chunk}: speed modes {modes} != " \
            f"{{{expected_mode[chunk]}}}"


def test_td_consistency_on_all_built_sources(built_available):
    """label[t, tau] == D[t,t+30] + label[t+30, tau-30] on complete tuples
    (direct where tau-30 is on the tau grid, chained down to tau=60
    otherwise) -- checked on every built source; violations reported."""
    import td_report
    here = Path(built_available)
    man = json.loads((here / "MANIFEST.json").read_text(encoding="utf-8"))
    checked = 0
    worst = 0.0
    violations = []
    for key, entry in sorted(man["worlds"].items()):
        lab_rel = next(f for f in entry["files"]
                       if f.endswith("_labels.parquet"))
        td_rel = next(f for f in entry["files"]
                      if f.endswith("_td.parquet"))
        rep = td_report.check_source(here / lab_rel, here / td_rel, TOL)
        checked += rep["n_checked"]
        worst = max(worst, rep["worst_diff"])
        if rep["n_violations"]:
            violations.append((key, rep["violations"][:5]))
    assert checked > 5000, f"only {checked} tuples checked"
    assert not violations, violations[:3]
    assert worst <= TOL, f"TD identity violated, worst |diff| = {worst}"
    # persisted report matches what the test just computed
    rep_json = here / "results" / "td_consistency.json"
    if rep_json.exists():
        s = json.loads(rep_json.read_text(encoding="utf-8"))["summary"]
        assert s["n_violations"] == 0


def test_labels_match_library_on_one_built_world(built_available):
    here = Path(built_available)
    man = json.loads((here / "MANIFEST.json").read_text(encoding="utf-8"))
    key, entry = sorted(man["worlds"].items())[0]
    lab, _ = _pair(here, entry)
    sub = lab[lab["tau"] == lab["tau"].min()].head(3)
    w = entry["world"]
    plan, worlds = V.load_plan(DEFAULT_PLAN)
    row = next(x for x in worlds if x["world"] == w)
    t0, runs = V.world_sources(row, Path(DEFAULT_RUNS))
    geom = V.cached_geometry()
    src = entry["source"]
    if src == "native":
        rtb = V.build_run_trace(w, "base", runs["base"], t0, {})
        pre = V.build_t_ins_pre(rtb, t0)
    else:
        pre = {}
    rt = V.build_run_trace(w, src, runs[src], t0, t_ins_pre=pre)
    for r in sub.itertuples():
        mine = V.window_label(rt, float(r.t), float(r.tau))
        assert abs(mine["total_delay_ref_s"]
                   - float(r.total_delay_ref_s)) <= TOL
        assert abs(mine["wait_mainline_s"]
                   - float(r.wait_mainline_s)) <= TOL


def test_manifest_insertion_crosscheck_clean(built_available):
    here = Path(built_available)
    man = json.loads((here / "MANIFEST.json").read_text(encoding="utf-8"))
    n_checked = 0
    for key, entry in man["worlds"].items():
        ic = entry.get("insertion_crosscheck")
        if ic is None:                     # resumed entries: no extraction stats
            continue
        assert ic.get("n_mismatch", 0) == 0, (key, ic)
        assert ic.get("n_exact", 0) > 0
        if entry["source"] == "base":
            n_checked += 1
    assert n_checked > 0
