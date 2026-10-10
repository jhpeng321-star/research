"""Builder tests: end-to-end on the synthetic mini-campaign, HOLDOUT and
incomplete-receipt handling, resumability, rebuild-on-corruption, and the
chunk/fix-flag columns."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import build_v0data as B

CHUNK = "chunk1"


def _manifest(out):
    return json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))


def test_builder_end_to_end_and_resumable(synth_world, tmp_path):
    out = tmp_path / "v0"
    args = ["--plan", str(synth_world["plan"]), "--runs",
            str(synth_world["runs"]), "--out", str(out),
            "--worlds", "wsyntm000p02t240,wsynterrp02t240"]
    rc = B.main(args)
    assert rc == 0
    man = _manifest(out)
    # only the complete TRAIN world is built (both sources), under chunk1/
    assert set(man["worlds"]) == {f"{CHUNK}/wsyntm000p02t240__base",
                                  f"{CHUNK}/wsyntm000p02t240__native"}
    # the error world is listed as incomplete, HOLDOUT as skipped
    sec = man["chunks"][CHUNK]
    assert {s["world"] for s in sec["skipped_incomplete"]} == \
        {"wsynterrp02t240"}
    assert {s["world"] for s in sec["skipped_holdout"]} == \
        {"wsynholdp02t240"}
    assert sec["world_keys"] == sorted(man["worlds"])
    assert sec["plan_sha256"] == B.sha_of(synth_world["plan"])
    base = man["worlds"][f"{CHUNK}/wsyntm000p02t240__base"]
    assert base["rows"]["n_decision_times"] > 0
    assert base["rows"]["n_label_rows"] > 0
    assert base["rows"]["n_td_rows"] > 0
    assert base["physics_sha256"]
    assert base["chunk_id"] == CHUNK
    for f, sha in base["files"].items():
        assert f.startswith(f"worlds/{CHUNK}/")
        assert (out / f).exists()
    # every complete label row satisfies tau <= tau_available; flag columns
    lab = pd.read_parquet(
        out / next(f for f in base["files"]
                   if f.endswith("_labels.parquet")))
    assert (lab["tau"] <= lab["tau_available"] + 1e-9).all()
    assert (lab["cav_speed_mode"] == 31).all()
    assert lab["chunk_id"].eq(CHUNK).all()
    assert lab["pre_fix"].all()            # 31 == pre-fix
    assert not lab["p40"].any()            # penetration 0.02

    # ---- resumability: second run rebuilds nothing, files unchanged ----
    before = {f: (out / f).read_bytes() for f in base["files"]}
    rc = B.main(args)
    assert rc == 0
    man2 = _manifest(out)
    assert man2["build"]["n_sources_built_this_run"] == 0
    for f, blob in before.items():
        assert (out / f).read_bytes() == blob

    # ---- corruption: mismatched hash -> exactly that source rebuilds ----
    npz_rel = next(f for f in base["files"] if f.endswith("_feat.npz"))
    npz = out / npz_rel
    npz.write_bytes(b"corrupted")
    rc = B.main(args)
    assert rc == 0
    man3 = _manifest(out)
    assert man3["build"]["n_sources_built_this_run"] == 1
    z = np.load(npz)
    assert "lane_bins" in z.files and "pre_fix" in z.files
    done = json.loads(
        (out / f"worlds/{CHUNK}/wsyntm000p02t240__base_done.json")
        .read_text(encoding="utf-8"))
    assert done["files"][npz_rel] == B.sha_of(npz)


def test_builder_missing_native_skips_source_only(synth_world, tmp_path):
    # delete the native receipt: base must still build, native is skipped
    runs = synth_world["runs"]
    receipt = runs / "wsyntm000p02t240__native" / "RUN_RECEIPT.json"
    saved = receipt.read_text(encoding="utf-8")
    receipt.unlink()
    try:
        out = tmp_path / "v0b"
        rc = B.main(["--plan", str(synth_world["plan"]), "--runs", str(runs),
                     "--out", str(out),
                     "--worlds", "wsyntm000p02t240"])
        assert rc == 0
        man = _manifest(out)
        assert set(man["worlds"]) == {f"{CHUNK}/wsyntm000p02t240__base"}
        reasons = {s["source"]: s["reason"]
                   for s in man["chunks"][CHUNK]["skipped_incomplete"]
                   if s["world"] == "wsyntm000p02t240"}
        assert "native" in reasons and "missing" in reasons["native"]
    finally:
        receipt.write_text(saved, encoding="utf-8")   # session fixture shared


def test_resume_repairs_stale_marker_identity(synth_world, tmp_path):
    """A done marker with WRONG identity fields (the chunk-1 backfill
    carried source='base' onto native entries) must not win on resume:
    the builder stamps world/source/split/demand/t0/policy from the plan
    and the arm key, unconditionally."""
    out = tmp_path / "v0r"
    args = ["--plan", str(synth_world["plan"]), "--runs",
            str(synth_world["runs"]), "--out", str(out),
            "--worlds", "wsyntm000p02t240"]
    assert B.main(args) == 0
    import v0lib as V
    marker = out / "worlds" / CHUNK / "wsyntm000p02t240__native_done.json"
    d = json.loads(marker.read_text(encoding="utf-8"))
    d["source"] = "base"                      # the historical defect
    d["split"] = "HOLDOUT"
    d["demand_veh_h"] = 9999
    d["policy"] = None
    marker.write_text(json.dumps(d), encoding="utf-8")

    assert B.main(args) == 0                  # resume pass, 0 rebuilds
    man = _manifest(out)
    assert man["build"]["n_sources_built_this_run"] == 0
    e = man["worlds"][f"{CHUNK}/wsyntm000p02t240__native"]
    assert e["source"] == "native"
    assert e["split"] == "TRAIN"
    assert e["demand_veh_h"] == 2700
    assert e["policy"] == V.SOURCE_POLICY["native"]
    assert e["world"] == "wsyntm000p02t240" and e["chunk_id"] == CHUNK
    # and the same repair on the base entry (marker left untouched there)
    eb = man["worlds"][f"{CHUNK}/wsyntm000p02t240__base"]
    assert eb["source"] == "base" and eb["split"] == "TRAIN"
