"""v0.3 dataset-root builder tests (WD-01 Part B B1).

Covers, on synthetic roots (and one real chunk-2 stand-in, TRAIN/CAL
only): the verified v0.2 copy, chunk-1 p40 supersession (files kept,
missing fix27 reported), the chunk1_fix27 build with id/split/t0
verification and refusal on mismatch, incremental resume, and the
HOLDOUT-never-opened guarantee (audit-hook recorded).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

import build_v0data as B
import build_v03 as B3
import v0lib as V
from conftest_load import (HoldoutAccessWatch, make_mini_v02_root,
                      rewrite_arm_to_speed_mode)


def _manifest(out):
    return json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))


def _summary(out):
    return json.loads((out / "results" / "build_v03_last.json")
                      .read_text(encoding="utf-8"))


def _manifest_or_none(out):
    p = out / "MANIFEST.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _v03_args(mini, out, extra=()):
    return ["--v02-root", str(mini["root"]), "--out", str(out),
            "--v02-manifest-sha", "", *extra]


# ------------------------------------------------------------ copy stage

def test_copy_verified_supersede_and_idempotent(mini_v02, tmp_path):
    mini = mini_v02
    out = tmp_path / "v03"
    rc = B3.main(_v03_args(mini, out, ["--skip-chunk3"]))
    assert rc == 0

    # copy: every v0.2 data file present and hash-equal (files KEPT on
    # disk even for superseded sources); MANIFEST byte-identical apart
    # from the supersession edit below
    man2 = json.loads((mini["root"] / "MANIFEST.json")
                      .read_text(encoding="utf-8"))
    for key, e in man2["worlds"].items():
        for rel, sha in e["files"].items():
            assert (out / rel).is_file(), rel
            assert V.sha256_file(out / rel) == sha, rel
    marker = json.loads((out / B3.COPY_MARKER).read_text("utf-8"))
    assert marker["mismatches"] == 0 and marker["holdout_accessed"] is False

    man = _manifest(out)
    # supersession: ALL chunk-1 p40 TRAIN/CAL sources marked, files kept
    sup = [k for k, e in man["worlds"].items() if e.get("superseded_by")]
    want = sorted(f"chunk1/{w}__{s}" for w in mini["p40"]
                  for s in ("base", "native"))
    assert sorted(sup) == want
    for k in sup:
        e = man["worlds"][k]
        assert e["superseded_by"] == "chunk1_fix27"
        assert e["p40"] is True and e["split"] in ("TRAIN", "CALIBRATION")
        for rel in e["files"]:
            assert (out / rel).is_file()
    # non-p40 sources untouched
    assert not any(e.get("superseded_by") for k, e in man["worlds"].items()
                   if any(k.endswith(w + "__base") for w in mini["p02"]))
    # summary + SHA256SUMS + README written
    s = _summary(out)
    assert s["missing_fix27_worlds"] == sorted(mini["p40"])
    assert s["n_superseded"] == len(want)
    assert s["holdout_accessed"] is False
    n_sums = len((out / "SHA256SUMS").read_text("utf-8").splitlines())
    assert n_sums == s["n_sha256sums_entries"] == \
        1 + len([p for p in (out / "worlds").rglob("*") if p.is_file()])
    assert "NEVER DELETE" in \
        (out / "README_DATA_NEVER_DELETE.md").read_text("utf-8")

    # idempotent: re-run marks nothing new and keeps every file identical
    blobs = {p.relative_to(out).as_posix(): p.read_bytes()
             for p in sorted((out / "worlds").rglob("*")) if p.is_file()}
    rc = B3.main(_v03_args(mini, out, ["--skip-chunk3"]))
    assert rc == 0
    s2 = _summary(out)
    assert s2["superseded_marked_this_run"] == []
    assert s2["n_superseded"] == s["n_superseded"]
    for rel, blob in blobs.items():
        assert (out / rel).read_bytes() == blob


def test_copy_refuses_wrong_v02_manifest_sha(mini_v02, tmp_path):
    out = tmp_path / "v03bad"
    with pytest.raises(SystemExit, match="expected v0.2"):
        B3.main(["--v02-root", str(mini_v02["root"]), "--out", str(out),
                 "--v02-manifest-sha", "0" * 64, "--skip-chunk3"])
    assert not (out / "MANIFEST.json").exists()


# ----------------------------------- chunk3 stand-in (real chunk-2 logs)

def _chunk2_paths():
    """(plan, runs) of the real chunk-2 campaign, or None if absent."""
    import wdpaths
    rd = __import__("pathlib").Path(wdpaths.RESEARCH_DENSE_ROOT)
    plan = rd / "output" / "campaign_stage1_v3" / "results_plan_chunk2" / \
        "WORLD_PLAN.json"
    runs = rd / "runs" / "campaign_stage1_chunk2"
    if not (plan.is_file() and runs.is_dir()):
        return None
    return plan, runs


def test_chunk3_test_stand_in_with_chunk2_logs(mini_v02, tmp_path):
    """Chunk-2 standing in for chunk 3: one complete TRAIN/CAL world built
    into a tmp root under chunk id chunk3_test; HOLDOUT listed but never
    built or opened; re-run resumes without rebuilding."""
    got = _chunk2_paths()
    if got is None:
        pytest.skip("real chunk-2 plan/runs not available")
    plan, runs = got
    _, worlds = V.load_plan(plan)
    wname = None
    for w in worlds:
        if w["split"] == "HOLDOUT":
            continue
        ok_b, _, _ = V.receipt_status(runs / w["base_arm"]["run"])
        ok_n, _, _ = V.receipt_status(runs / f"{w['world']}__native")
        if ok_b and ok_n:
            wname = w["world"]
            break
    assert wname, "no complete chunk-2 TRAIN/CAL world found"

    mini = mini_v02
    out = tmp_path / "v03c"
    holdouts = [w["world"] for w in worlds if w["split"] == "HOLDOUT"]
    watch = HoldoutAccessWatch(holdouts).install()
    try:
        rc = B3.main(_v03_args(mini, out, [
            "--chunk3-chunk", "chunk3_test", "--plan", str(plan),
            "--runs", str(runs), "--worlds", wname,
            "--default-speed-mode", "27"]))
    finally:
        hits, watch.hits = watch.hits, []
    assert rc == 0
    assert not hits, f"HOLDOUT paths touched: {hits[:3]}"

    man = _manifest(out)
    keys = [k for k in man["worlds"] if k.startswith("chunk3_test/")]
    assert keys == [f"chunk3_test/{wname}__base",
                    f"chunk3_test/{wname}__native"]
    for k in keys:
        e = man["worlds"][k]
        assert e["cav_speed_mode"] == 27 and e["pre_fix"] is False
        assert e["split"] in ("TRAIN", "CALIBRATION")
        lab = pd.read_parquet(out / next(f for f in e["files"]
                                         if f.endswith("_labels.parquet")))
        assert lab["chunk_id"].eq("chunk3_test").all()
        assert set(lab["cav_speed_mode"].unique()) == {27}
        assert not lab["pre_fix"].any()
    sec = man["chunks"]["chunk3_test"]
    listed_holdout = [s["world"] for s in sec["skipped_holdout"]]
    assert sorted(listed_holdout) == sorted(holdouts)
    assert not any(k.split("/")[1].startswith(w)
                   for k in keys for w in holdouts)
    # superseded chunk-1 p40 marks survived the chunk3_test build pass
    assert sum(1 for e in man["worlds"].values() if e.get("superseded_by")) \
        == 4

    s = _summary(out)
    st = s["stages"]["chunk3_test"]
    assert st["n_sources_built_this_run"] == 2
    # chunk-2 is a COMPLETE campaign: every non-HOLDOUT world is eligible,
    # none pending; the --worlds filter limited this run to one world
    assert st["n_worlds_eligible"] == st["n_worlds_plan"] - len(holdouts)
    assert st["n_pending_worlds"] == 0
    assert st["default_speed_mode"] == 27

    # resume: second run rebuilds nothing and keeps bytes identical
    blobs = {rel: (out / rel).read_bytes()
             for k in keys for rel in man["worlds"][k]["files"]}
    rc = B3.main(_v03_args(mini, out, [
        "--chunk3-chunk", "chunk3_test", "--plan", str(plan),
        "--runs", str(runs), "--worlds", wname,
        "--default-speed-mode", "27"]))
    assert rc == 0
    man2 = _manifest(out)
    assert man2["build"]["n_sources_built_this_run"] == 0
    for rel, blob in blobs.items():
        assert (out / rel).read_bytes() == blob


# --------------------------------------------------------- fix27 stage --

def _make_fix27_fixture(mini, root: Path, worlds, *, mode=27,
                        receipt_mode=27, t0_override=None,
                        split_override=None):
    """Tiny fake fix27 runs root + plan for the given chunk-1 p40 worlds:
    copies of the chunk-1 run files with the speed mode rewritten, plus
    optional deliberate plan mismatches."""
    runs = root / "fix27_runs"
    plan_worlds = []
    for wname, split, with_runs in worlds:
        if with_runs:
            for arm in ("base", "native"):
                rewrite_arm_to_speed_mode(
                    mini["runs"] / f"{wname}__{arm}",
                    runs / f"{wname}__{arm}", mode,
                    top_level_receipt_mode=receipt_mode)
        entry = dict(world=wname, seed=1, loading_veh_h=2700,
                     penetration=0.4,
                     split=split_override or split, role="exposed",
                     t0_s=float(mini["t0"] if t0_override is None
                                else t0_override),
                     base_arm=dict(arm="base", run=f"{wname}__base",
                                   controller_name="damping"),
                     branches=[dict(arm="native", run=f"{wname}__native",
                                    suffix_controller="native")])
        plan_worlds.append(entry)
    plan = dict(schema="test", constants={}, worlds=plan_worlds)
    plan_path = root / "FIX27_WORLD_PLAN.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return plan_path, runs


def test_fix27_build_supersession_and_missing_report(mini_v02, tmp_path):
    mini = mini_v02
    out = tmp_path / "v03f"
    fix = tmp_path / "fix"
    # w40a complete in the fix27 runs; w40b in the plan but runs not
    # landed yet -> pending; the p40 HOLDOUT world is listed but must
    # never be opened
    plan, runs = _make_fix27_fixture(
        mini, fix,
        [(mini["p40"][0], "TRAIN", True), (mini["p40"][1], "CALIBRATION",
                                           False)])
    # add the p40 HOLDOUT world to the plan (no run dirs at all)
    p = json.loads(plan.read_text(encoding="utf-8"))
    p["worlds"].append(dict(world=mini["holdout"], seed=1,
                            loading_veh_h=2700, penetration=0.4,
                            split="HOLDOUT", role="exposed",
                            t0_s=float(mini["t0"]),
                            base_arm=dict(arm="base",
                                          run=f"{mini['holdout']}__base"),
                            branches=[]))
    plan.write_text(json.dumps(p), encoding="utf-8")

    watch = HoldoutAccessWatch([mini["holdout"]]).install()
    try:
        rc = B3.main(_v03_args(mini, out, [
            "--skip-chunk3", "--fix27-plan", str(plan),
            "--fix27-runs", str(runs), "--chunk1-plan", str(mini["plan"])]))
    finally:
        hits, watch.hits = watch.hits, []
    assert rc == 0
    assert not hits, f"HOLDOUT paths touched: {hits[:3]}"

    man = _manifest(out)
    assert set(man["chunks"]["chunk1_fix27"]["world_keys"]) == \
        {f"chunk1_fix27/{mini['p40'][0]}__base",
         f"chunk1_fix27/{mini['p40'][0]}__native"}
    for k in man["chunks"]["chunk1_fix27"]["world_keys"]:
        e = man["worlds"][k]
        # id / split / t0 carried from the verified chunk-1 plan
        assert e["world"] == mini["p40"][0] and e["split"] == "TRAIN"
        assert float(e["t0_s"]) == float(mini["t0"])
        assert e["p40"] is True and e["pre_fix"] is False
        assert e["cav_speed_mode"] == 27
        lab = pd.read_parquet(out / next(f for f in e["files"]
                                         if f.endswith("_labels.parquet")))
        assert set(lab["cav_speed_mode"].unique()) == {27}
        assert not lab["pre_fix"].any()
        z = np.load(out / next(f for f in e["files"]
                               if f.endswith("_feat.npz")))
        assert bool(z["pre_fix"]) is False and bool(z["p40"]) is True
    # supersession covers BOTH p40 worlds' chunk-1 sources (owner rule),
    # files kept; the missing fix27 world is reported
    sup = sorted(k for k, e in man["worlds"].items()
                 if e.get("superseded_by"))
    assert sup == sorted(f"chunk1/{w}__{s}" for w in mini["p40"]
                         for s in ("base", "native"))
    s = _summary(out)
    assert s["missing_fix27_worlds"] == [mini["p40"][1]]
    st = s["stages"]["chunk1_fix27"]
    assert st["n_pending_worlds"] == 1          # in plan, runs incomplete
    assert st["pending_worlds"] == [mini["p40"][1]]
    assert [w["world"] for w in
            man["chunks"]["chunk1_fix27"]["skipped_holdout"]] == \
        [mini["holdout"]]
    assert s["holdout_accessed"] is False


def test_fix27_separate_bases_root_and_plan_without_base_arm(mini_v02,
                                                             tmp_path):
    """The real chunk1_fix27 shape: the fix27 plan carries NO base_arm
    entries (null), the mode-27 bases live in a SEPARATE bases root
    (verify27-style <world>__base dirs) and only the native branches are
    in the fix27 runs root.  A world whose native run has not landed stays
    pending; the built base source must point at the bases root."""
    mini = mini_v02
    out = tmp_path / "v03bases"
    fix = tmp_path / "fixbases"
    bases = fix / "bases"              # e.g. campaign_stage1_verify27
    runs = fix / "runs"                # e.g. campaign_stage1_chunk1_fix27
    w1, w2 = mini["p40"]
    rewrite_arm_to_speed_mode(mini["runs"] / f"{w1}__base",
                              bases / f"{w1}__base", 27,
                              top_level_receipt_mode=27)
    rewrite_arm_to_speed_mode(mini["runs"] / f"{w1}__native",
                              runs / f"{w1}__native", 27,
                              top_level_receipt_mode=27)
    rewrite_arm_to_speed_mode(mini["runs"] / f"{w2}__base",
                              bases / f"{w2}__base", 27,
                              top_level_receipt_mode=27)
    plan_worlds = [dict(world=w, seed=1, loading_veh_h=2700,
                        penetration=0.4, split=split, role="exposed",
                        t0_s=float(mini["t0"]), base_arm=None,
                        branches=[])
                   for w, split in ((w1, "TRAIN"), (w2, "CALIBRATION"))]
    plan = dict(schema="test", constants={}, worlds=plan_worlds)
    plan_path = fix / "FIX27_WORLD_PLAN.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    rc = B3.main(_v03_args(mini, out, [
        "--skip-chunk3", "--fix27-plan", str(plan_path),
        "--fix27-runs", str(runs), "--fix27-bases-root", str(bases),
        "--chunk1-plan", str(mini["plan"])]))
    assert rc == 0

    man = _manifest(out)
    sec = man["chunks"]["chunk1_fix27"]
    assert sec["bases_root"] == str(bases)
    assert set(sec["world_keys"]) == {f"chunk1_fix27/{w1}__base",
                                      f"chunk1_fix27/{w1}__native"}
    # base entry reads from the separate bases root, native from runs
    assert man["worlds"][f"chunk1_fix27/{w1}__base"]["run"] == \
        str(bases / f"{w1}__base")
    assert man["worlds"][f"chunk1_fix27/{w1}__native"]["run"] == \
        str(runs / f"{w1}__native")
    for k in sec["world_keys"]:
        e = man["worlds"][k]
        assert e["cav_speed_mode"] == 27 and e["pre_fix"] is False
        lab = pd.read_parquet(out / next(f for f in e["files"]
                                         if f.endswith("_labels.parquet")))
        assert set(lab["cav_speed_mode"].unique()) == {27}
        assert not lab["pre_fix"].any()
    s = _summary(out)
    st = s["stages"]["chunk1_fix27"]
    assert st["bases_root"] == str(bases)
    assert st["pending_worlds"] == [w2]     # base ok in bases root, native
    assert "native: RUN_RECEIPT.json missing" in \
        st["pending_reasons"][w2]
    # supersession of the chunk-1 p40 sources still applies (owner rule)
    assert sorted(k for k, e in man["worlds"].items()
                  if e.get("superseded_by")) == \
        sorted(f"chunk1/{w}__{src}" for w in (w1, w2)
               for src in ("base", "native"))
    assert s["missing_fix27_worlds"] == [w2]


def test_fix27_base_only_worlds_stay_pending_and_resume(mini_v02, tmp_path):
    """Bases land before the branch runs: with NO native runs at all the
    stage builds every available base source, but both worlds stay pending
    and missing (source-aware accounting: a world counts only with BOTH
    sources).  A later run with one native landed builds exactly that
    native and completes the world; the other stays base-only pending."""
    mini = mini_v02
    out = tmp_path / "v03baseonly"
    fix = tmp_path / "fixbaseonly"
    bases, runs = fix / "bases", fix / "runs"
    w1, w2 = mini["p40"]
    for w in (w1, w2):
        rewrite_arm_to_speed_mode(mini["runs"] / f"{w}__base",
                                  bases / f"{w}__base", 27,
                                  top_level_receipt_mode=27)
    plan_worlds = [dict(world=w, seed=1, loading_veh_h=2700,
                        penetration=0.4, split=sp, role="exposed",
                        t0_s=float(mini["t0"]), base_arm=None, branches=[])
                   for w, sp in ((w1, "TRAIN"), (w2, "CALIBRATION"))]
    plan_path = fix / "FIX27_WORLD_PLAN.json"
    plan_path.write_text(json.dumps(dict(schema="test", constants={},
                                         worlds=plan_worlds)),
                         encoding="utf-8")
    args = _v03_args(mini, out, [
        "--skip-chunk3", "--fix27-plan", str(plan_path),
        "--fix27-runs", str(runs), "--fix27-bases-root", str(bases),
        "--chunk1-plan", str(mini["plan"])])

    rc = B3.main(args)
    assert rc == 0
    man = _manifest(out)
    sec = man["chunks"]["chunk1_fix27"]
    assert set(sec["world_keys"]) == {f"chunk1_fix27/{w1}__base",
                                      f"chunk1_fix27/{w2}__base"}
    s = _summary(out)
    st = s["stages"]["chunk1_fix27"]
    assert st["n_sources_complete"] == 2
    assert st["pending_worlds"] == [w1, w2]     # source-aware: no native yet
    assert s["missing_fix27_worlds"] == [w1, w2]
    # bases read from the separate bases root
    assert all(man["worlds"][k]["run"].startswith(str(bases))
               for k in sec["world_keys"])

    # resume: w1's native lands, w2's does not
    rewrite_arm_to_speed_mode(mini["runs"] / f"{w1}__native",
                              runs / f"{w1}__native", 27,
                              top_level_receipt_mode=27)
    rc = B3.main(args)
    assert rc == 0
    man = _manifest(out)
    assert man["build"]["n_sources_built_this_run"] == 1
    assert set(man["chunks"]["chunk1_fix27"]["world_keys"]) == \
        {f"chunk1_fix27/{w1}__base", f"chunk1_fix27/{w1}__native",
         f"chunk1_fix27/{w2}__base"}
    s = _summary(out)
    st = s["stages"]["chunk1_fix27"]
    assert st["pending_worlds"] == [w2]
    assert s["missing_fix27_worlds"] == [w2]
    # the resumed base source was a hash-verified no-op, not rebuilt
    base_sha = man["worlds"][f"chunk1_fix27/{w1}__base"]["files"]
    for rel, sha in base_sha.items():
        assert V.sha256_file(out / rel) == sha


def test_fix27_speed_mode_from_cav_rows_when_receipt_null(mini_v02,
                                                          tmp_path):
    """Receipts record speed_mode null (chunk-2/3 style): the mode comes
    from the CAV physics rows (27), reported as observed-cav."""
    mini = mini_v02
    out = tmp_path / "v03fcav"
    fix = tmp_path / "fixcav"
    plan, runs = _make_fix27_fixture(mini, fix,
                                     [(mini["p40"][0], "TRAIN", True)],
                                     mode=27, receipt_mode=None)
    rc = B3.main(_v03_args(mini, out, [
        "--skip-chunk3", "--fix27-plan", str(plan), "--fix27-runs",
        str(runs), "--chunk1-plan", str(mini["plan"])]))
    assert rc == 0
    man = _manifest(out)
    e = man["worlds"][f"chunk1_fix27/{mini['p40'][0]}__base"]
    assert e["cav_speed_mode"] == 27
    assert e["speed_mode_source"] == "observed-cav"
    assert e["pre_fix"] is False


def test_fix27_refuses_plan_mismatch(mini_v02, tmp_path):
    mini = mini_v02
    for kw, needle in ((dict(t0_override=300.0), "t0"),
                       (dict(split_override="CALIBRATION"), "split")):
        out = tmp_path / f"v03ref_{needle}"
        fix = tmp_path / f"fixref_{needle}"
        plan, runs = _make_fix27_fixture(
            mini, fix, [(mini["p40"][0], "TRAIN", True)], **kw)
        with pytest.raises(SystemExit, match=needle):
            B3.main(_v03_args(mini, out, [
                "--skip-chunk3", "--fix27-plan", str(plan),
                "--fix27-runs", str(runs),
                "--chunk1-plan", str(mini["plan"])]))
        # refusal precedes any build: nothing was created for fix27
        assert not (out / "worlds" / "chunk1_fix27").exists()
        assert "chunk1_fix27" not in \
            _manifest_or_none(out).get("chunks", {})


def test_fix27_refuses_non_p40_world(mini_v02, tmp_path):
    mini = mini_v02
    out = tmp_path / "v03ref_pen"
    fix = tmp_path / "fixref_pen"
    plan, runs = _make_fix27_fixture(mini, fix,
                                     [(mini["p02"][0], "TRAIN", True)])
    with pytest.raises(SystemExit, match="penetration"):
        B3.main(_v03_args(mini, out, [
            "--skip-chunk3", "--fix27-plan", str(plan),
            "--fix27-runs", str(runs),
            "--chunk1-plan", str(mini["plan"])]))
    assert not (out / "worlds" / "chunk1_fix27").exists()


# ------------------------------------------------- HOLDOUT guard (direct)

def test_holdout_never_opened_during_copy(mini_v02, tmp_path):
    mini = mini_v02
    out = tmp_path / "v03h"
    watch = HoldoutAccessWatch([mini["holdout"]]).install()
    try:
        rc = B3.main(_v03_args(mini, out, ["--skip-chunk3"]))
    finally:
        hits, watch.hits = watch.hits, []
    assert rc == 0
    assert not hits, f"HOLDOUT paths touched: {hits[:3]}"
    man = _manifest(out)
    holdout_sources = [k for k in man["worlds"]
                       if man["worlds"][k]["world"] == mini["holdout"]]
    assert holdout_sources == []
    # the p40 HOLDOUT world is NOT superseded and has no fix27 report entry
    assert mini["holdout"] not in _summary(out)["missing_fix27_worlds"]


# ------------------------------------------------------- mode resolution

def test_resolve_speed_mode_priority():
    f = B.resolve_speed_mode
    # chunk-2 style config receipt wins
    assert f({"evaluate_summary": {"config": {"speed_mode": 27}}},
             {"cav_smode_values": [31.0]}, 31) == (27, "receipt")
    # chunk-3 style top-level receipt key wins over physics rows
    assert f({"cav_speed_mode": 27, "evaluate_summary":
              {"config": {"speed_mode": None}}},
             {"cav_smode_values": [31.0]}, 31) == (27, "receipt")
    # CAV-only observed beats the mixed all-row set (chunk-2 receipts null)
    assert f(None, {"cav_smode_values": [27.0],
                    "smode_values": [27.0, 31.0]}, 31) == (27, "observed-cav")
    # all-row single observed (chunk-1 backfill behaviour preserved)
    assert f(None, {"smode_values": [31.0],
                    "cav_smode_values": []}, 27) == (31, "observed")
    assert f(None, {"smode_values": [31.0]}, 27) == (31, "observed")
    # config default on mixed sets
    assert f(None, {"smode_values": [27.0, 31.0],
                    "cav_smode_values": [27.0, 31.0]}, 27) == (27, "config")
    assert f(None, {}, 27) == (27, "config")
