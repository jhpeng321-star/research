"""Upload manifest: split-before-open and sealed/excluded refusal."""
import json

import pytest

import synth
from corpus import _util as U
from corpus import manifest as MF
from corpus import pack


def _runs_with_plans(tmp_path, extra=None):
    runs = synth.default_runs()
    p4 = synth.write_world_plan(
        tmp_path / "plans" / "chunk4.json",
        [{"world": "w1", "split": "TRAIN", "t0_s": 540.0},
         {"world": "w2", "split": "TRAIN", "t0_s": 600.0}])
    p2 = synth.write_world_plan(
        tmp_path / "plans" / "chunk2.json",
        [{"world": "c1", "split": "CALIBRATION", "t0_s": 300.0}])
    for r in runs:
        r["root_rel"] = "runs/x"
        r["plan_path"] = str(p4 if r["chunk"] == "chunk4" else p2)
    if extra:
        runs.extend(extra)
    return runs, p4, p2


def _build_pack(tmp_path, runs):
    data = tmp_path / "shards"
    synth.write_shards(data, runs)
    out = tmp_path / "pack"
    pack.pack_shards(data, out, workers=1)
    return out


def test_manifest_ok(tmp_path):
    runs, _, _ = _runs_with_plans(tmp_path)
    out = _build_pack(tmp_path, runs)
    man = MF.build_upload_manifest(out)
    assert man["schema"] == MF.MANIFEST_SCHEMA
    assert set(man["splits"]) == {"TRAIN", "CALIBRATION"}
    assert "manifest.json" in man["files"] and "INDEX.json" in man["files"]
    assert man["refused"] == []
    sha = MF.write_manifest(man, tmp_path / "upload.json")
    assert len(sha) == 64 and (tmp_path / "upload.json").is_file()


def test_manifest_refuses_holdout_before_payload_open(tmp_path):
    extra = [dict(chunk="chunk4", world="w9", arm="native", policy="native",
                  split="TRAIN", t0_s=540.0, n_ticks=2, n_dec=2)]
    runs, p4, _ = _runs_with_plans(tmp_path, extra)
    synth.write_world_plan(p4, [
        {"world": "w1", "split": "TRAIN", "t0_s": 540.0},
        {"world": "w2", "split": "TRAIN", "t0_s": 600.0},
        {"world": "w9", "split": "HOLDOUT", "t0_s": 540.0}])
    out = _build_pack(tmp_path, runs)
    with pytest.raises(MF.ManifestRefusal):
        MF.build_upload_manifest(out)
    # no packed payload array was opened before the split refusal
    opened = [p for p in U.OpenedAudit.paths if str(p).endswith(".npy")]
    assert opened == [], opened


def test_manifest_refuses_chunk5(tmp_path):
    extra = [dict(chunk="chunk5", world="w5", arm="native", policy="native",
                  split="TRAIN", t0_s=540.0, n_ticks=2, n_dec=2)]
    runs, _, _ = _runs_with_plans(tmp_path, extra)
    out = _build_pack(tmp_path, runs)
    with pytest.raises(MF.ManifestRefusal):
        MF.build_upload_manifest(out)


def test_manifest_refuses_seed_block(tmp_path):
    extra = [dict(chunk="chunk4", world="w8710", arm="native", policy="native",
                  split="TRAIN", t0_s=540.0, n_ticks=2, n_dec=2)]
    runs, p4, _ = _runs_with_plans(tmp_path, extra)
    synth.write_world_plan(p4, [
        {"world": "w1", "split": "TRAIN", "t0_s": 540.0},
        {"world": "w2", "split": "TRAIN", "t0_s": 600.0},
        {"world": "w8710", "split": "TRAIN", "t0_s": 540.0}])
    out = _build_pack(tmp_path, runs)
    with pytest.raises(MF.ManifestRefusal):
        MF.build_upload_manifest(out)


def test_manifest_refuses_excluded(tmp_path):
    runs, _, _ = _runs_with_plans(tmp_path)
    out = _build_pack(tmp_path, runs)
    exc = tmp_path / "excl.json"
    exc.write_text(json.dumps({
        "schema": "holdout_exclusions/1",
        "excluded": [{"world": "w1", "reason": "test"}],
        "kept_disclosed": []}), encoding="utf-8")
    with pytest.raises(MF.ManifestRefusal):
        MF.build_upload_manifest(out, exclusions_path=exc)
