"""WD-01-PB-RESCORE1 tests.

* HOLDOUT split guard: HOLDOUT worlds are refused BEFORE any run file is
  opened (plan-split selection + holdout_guard exclusions + the in-code
  TRAIN/CAL assertion), and the open-path audit catches any HOLDOUT path;
* copy-not-link: copied label-root files are byte-verified against the
  parent MANIFEST and are NEVER hard links (a hard-linking copy raises);
* label parity: one TRAIN damping base source rebuilt from the raw runs
  with the v0.3.2 label code (build_v0data.build_source) reproduces the
  frozen v0.3.2 root's labels exactly (real-data test, skipped when the
  roots are absent).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]          # output/wd01
RES = HERE / "rescore1"
for _p in (HERE, RES, HERE / "partB_v0data", HERE / "partA_data",
           HERE / "part4"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import wdpaths  # noqa: E402
import common as C  # noqa: E402
import build_truearms_labels as G  # noqa: E402

RD = Path(wdpaths.RESEARCH_DENSE_ROOT)
V032_ROOT = RD / "output" / "wd01_partB_v0data_v032"
RUNS2 = RD / "runs" / "campaign_stage1_chunk2"
PLAN2 = RD / "output" / "campaign_stage1_v2" / "results_plan_chunk2" \
        / "WORLD_PLAN.json"

_real_roots = (C.LABEL_ROOT / "MANIFEST.json").is_file() \
    and V032_ROOT.exists() and RUNS2.exists() and PLAN2.is_file()

skip_real = pytest.mark.skipif(
    not _real_roots, reason="RESCORE1 label root / v0.3.2 root / chunk-2 "
                            "runs not present")


# ----------------------------------------------------- holdout guard ----

def _plan_like():
    return [
        dict(world="w9001m2000p02t240", split="TRAIN", t0_s=240.0,
             loading_veh_h=2000, penetration=0.02),
        dict(world="w9002m2000p02t300", split="CALIBRATION", t0_s=300.0,
             loading_veh_h=2000, penetration=0.02),
        dict(world="w9003m2000p02t360", split="HOLDOUT", t0_s=360.0,
             loading_veh_h=2000, penetration=0.02, role="clean"),
        dict(world="w8407m2400p02t360", split="HOLDOUT", t0_s=360.0,
             loading_veh_h=2400, penetration=0.02, role="exposed"),
    ]


def test_select_worlds_skips_holdout_by_name_before_any_open():
    exclusions = G.load_holdout_exclusions()[0]
    selected, skipped = G.select_worlds(_plan_like(), exclusions)
    assert [w["world"] for w in selected] == \
        ["w9001m2000p02t240", "w9002m2000p02t300"]
    assert {s["world"] for s in skipped} == \
        {"w9003m2000p02t360", "w8407m2400p02t360"}
    # no HOLDOUT name survives anywhere in the selection
    names = json.dumps([w["world"] for w in selected])
    for s in skipped:
        assert s["world"] not in names


def test_select_worlds_refuses_excluded_roots():
    exclusions = G.load_holdout_exclusions()[0]
    guard = G.load_holdout_exclusions()[1]
    bad = dict(world="w8407m2400p02t360", split="TRAIN", t0_s=360.0,
               loading_veh_h=2400, penetration=0.02)
    with pytest.raises(guard.HoldoutExclusionError):
        G.select_worlds([bad], exclusions)


def test_build_world_job_refuses_non_traincal_split():
    job = dict(out_root="x", chunk_id="chunk2", runs_root="r",
               truearms_root="t",
               world=dict(world="wX", split="HOLDOUT", t0_s=240.0,
                          loading_veh_h=2000, penetration=0.02),
               holdout_names=["wX"])
    with pytest.raises(AssertionError, match="refusing non-TRAIN/CAL"):
        G.build_world_job(job)


def test_open_recorder_asserts_holdout_paths():
    rec = G.OpenedPathRecorder()
    rec.paths.add(str(RD / "runs" / "campaign_stage1_chunk2_truearms"
                      / "w8477m2600p02t480__native" / "run"
                      / "physics_steps.jsonl"))
    with pytest.raises(AssertionError, match="HOLDOUT world path"):
        rec.assert_no_holdout({"w8477m2600p02t480"})
    rec2 = G.OpenedPathRecorder()
    rec2.paths.add(str(RUNS2 / "w8480m2600p02t300__base" / "run"
                       / "physics_steps.jsonl"))
    out = rec2.assert_no_holdout({"w8477m2600p02t480"})
    assert out["n_holdout_hits"] == 0 and out["holdout_accessed"] is False


# ------------------------------------------------------ copy-not-link ----

def _fake_parent(tmp_path: Path) -> tuple[Path, dict]:
    parent = tmp_path / "parent"
    rel_dir = parent / "worlds" / "chunkT"
    rel_dir.mkdir(parents=True)
    files = {}
    for kind in ("labels.parquet", "td.parquet"):
        src = rel_dir / f"wX__base_{kind}"
        src.write_bytes(b"parquet-bytes-" + kind.encode())
        files[f"worlds/chunkT/wX__base_{kind}"] = \
            hashlib.sha256(src.read_bytes()).hexdigest()
    entry = dict(chunk_id="chunkT", world="wX", source="base", files=files)
    return parent, entry


def test_copy_source_files_copies_and_never_hardlinks(tmp_path):
    parent, entry = _fake_parent(tmp_path)
    out = tmp_path / "out"
    files = G.copy_source_files(parent, entry, out, "wX", "base")
    for kind in ("labels.parquet", "td.parquet"):
        rel = f"worlds/chunkT/wX__base_{kind}"
        assert files[rel] == entry["files"][rel]
        dst = out / rel
        assert dst.read_bytes() == (parent / rel).read_bytes()
        s1, s2 = os.stat(parent / rel), os.stat(dst)
        assert (s1.st_dev, s1.st_ino) != (s2.st_dev, s2.st_ino), \
            "copied file must not be a hard link"


def test_copy_source_files_refuses_hardlink(tmp_path):
    parent, entry = _fake_parent(tmp_path)
    out = tmp_path / "out"
    (out / "worlds" / "chunkT").mkdir(parents=True)
    dst = out / "worlds" / "chunkT" / "wX__base_labels.parquet"
    os.link(parent / "worlds" / "chunkT" / "wX__base_labels.parquet", dst)
    with pytest.raises(AssertionError, match="COPY-ONLY"):
        G.copy_source_files(parent, entry, out, "wX", "base")

def test_copy_source_files_refuses_hash_mismatch(tmp_path):
    parent, entry = _fake_parent(tmp_path)
    entry["files"]["worlds/chunkT/wX__base_labels.parquet"] = "0" * 64
    with pytest.raises(AssertionError, match="hash mismatch"):
        G.copy_source_files(parent, entry, tmp_path / "out", "wX", "base")


# --------------------------------------------------- real-data parity ----

@skip_real
def test_label_parity_with_v032_code_on_train_damping_source(tmp_path):
    """One TRAIN chunk-2 damping base source rebuilt from the raw runs
    with the v0.3.2 label code reproduces the frozen v0.3.2 root's
    labels (and this task's hash-verified copy of them)."""
    import v0lib as V
    import build_v0data as B
    plan, worlds = V.load_plan(PLAN2)
    wname = next(w["world"] for w in sorted(worlds,
                                            key=lambda x: x["world"])
                 if w["split"] == "TRAIN")
    w = next(w for w in worlds if w["world"] == wname)
    arm = RUNS2 / f"{wname}__base"
    ok, receipt, reason = V.receipt_status(arm)
    assert ok, reason
    entry = B.build_source(
        w, "base", arm, RUNS2, tmp_path, V.cached_geometry(), {}, {},
        chunk_id="chunk2", default_speed_mode=27, receipt=receipt)
    new_lab = pd.read_parquet(tmp_path / f"{wname}__base_labels.parquet")
    v032_lab = pd.read_parquet(
        V032_ROOT / "worlds" / "chunk2" / f"{wname}__base_labels.parquet")
    ours_lab = pd.read_parquet(
        C.LABEL_ROOT / "worlds" / "chunk2"
        / f"{wname}__base_labels.parquet")
    cols = [c for c in new_lab.columns if c != "built_utc"]
    for other in (v032_lab, ours_lab):
        a = new_lab[cols].sort_values(["t", "tau"]).reset_index(drop=True)
        b = other[cols].sort_values(["t", "tau"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(
            a, b, check_dtype=False, check_exact=True)
    # TD tuples agree too (complete-horizon rows)
    new_td = pd.read_parquet(tmp_path / f"{wname}__base_td.parquet")
    v032_td = pd.read_parquet(
        V032_ROOT / "worlds" / "chunk2" / f"{wname}__base_td.parquet")
    tcols = [c for c in new_td.columns]
    pd.testing.assert_frame_equal(
        new_td[tcols].sort_values(["t", "tau"]).reset_index(drop=True),
        v032_td[tcols].sort_values(["t", "tau"]).reset_index(drop=True),
        check_dtype=False, check_exact=True)
