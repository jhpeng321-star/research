"""Synthetic tests for the dspeed physics-sourced identity sidecar.

No real data: a minimal manifest/INDEX-valid pack, a shard receipt triple and a
dummy physics file; ``features.featurizer.parse_run`` is monkeypatched so the
frozen slot assignment is not re-derived here (its exactness is covered by the
real build and the pack_v3 sibling alignment check).
"""
from __future__ import annotations

import gzip
import hashlib
import json

import numpy as np
import pytest

from dspeed import identity as DI

SHA = "a" * 64


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _arrays(shape0=3):
    out = {}
    for name in ("veh_step", "veh_mask", "accel_next", "target_mask",
                 "lane_next"):
        out[name] = dict(file=f"{name}.npy", shape=[shape0, 3])
    out["plan"] = dict(file="plan.npy", shape=[1, 3])
    return out


def _fixture(tmp_path):
    pack = tmp_path / "pack"
    shards = tmp_path / "shards"
    out = tmp_path / "identity"
    pack.mkdir()
    shard = shards / "chunk4" / "wa" / "arm0"
    shard.mkdir(parents=True)
    arrays = _arrays()
    item = dict(batch="chunk4", chunk="chunk4", world="wa", arm="arm0",
                split="TRAIN", t0_s=0.0, tick=1, plan_row=0,
                pair_key="wa@t0=0", outcome_rows=[])
    run = dict(key="chunk4/wa__arm0", batch="chunk4", chunk="chunk4",
               world="wa", arm="arm0", split="TRAIN", t0_s=0.0, n_ticks=3,
               tick_start=0, tick_end=3, dec_start=0, dec_end=1,
               n_outcome_rows=0, input_sha256=SHA,
               shard_dir="Z:/old/chunk4/wa/arm0")
    _write(pack / "manifest.json", dict(schema="wm_n1_pack/1", k_max=3,
                                        arrays=arrays, items=[item]))
    _write(pack / "INDEX.json", dict(schema="wm_n1_corpus_pack/2", k_max=3,
                                     arrays=arrays, runs=[run]))
    vids = ["v0", "v1", "v2"]
    physics = shard / "physics_steps.jsonl.gz"
    with gzip.open(physics, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"time_s": 0.5}) + "\n")
    _write(shard / "slot_vid.json", vids)
    _write(shard / "ALIGN.json", dict(world="wa", arm="arm0", chunk="chunk4",
                                      batch="chunk4", split="TRAIN", t0_s=0.0,
                                      n_ticks=3, physics_gz_sha256=SHA,
                                      input_physics=str(physics)))
    _write(shard / "meta.json", dict(world="wa", arm="arm0", chunk="chunk4",
                                     t0_s=0.0, n_ticks=3, slot_vid=vids))
    return pack, shards, out, shard


def _fake_parse(monkeypatch, *, vids=("v0", "v1", "v2"),
                slots=(( -1, -1, -1), (0, 1, -1), (0, -1, 2)), n_ticks=3):
    from features import featurizer as FZ
    arr = np.asarray(slots, dtype=np.int32)

    def _run(run_dir, t_origin, *, k_max, physics_path=None):
        return dict(n_ticks=n_ticks, slot_vid=list(vids), slot_index=arr)

    monkeypatch.setattr(FZ, "parse_run", _run)


def test_build_and_load_row(tmp_path, monkeypatch):
    pack, shards, out, shard = _fixture(tmp_path)
    _fake_parse(monkeypatch)
    receipt = DI.build_dspeed_identity_sidecar(pack, shards, out)
    assert receipt["n_vehicles"] == 2
    side = DI.load_dspeed_identity_sidecar(pack, out)
    assert side.ids_for_item(0).tolist() == [0, 1, -1]
    assert side.vehicle_map == [("chunk4", "wa", "v0"), ("chunk4", "wa", "v1")]


def test_physics_slot_vid_mismatch_refused(tmp_path, monkeypatch):
    pack, shards, out, shard = _fixture(tmp_path)
    _fake_parse(monkeypatch, vids=("v0", "vX", "v2"))
    with pytest.raises(DI.I.IdentitySidecarError):
        DI.build_dspeed_identity_sidecar(pack, shards, out)
    assert (out / "BUILD_FAILED.json").is_file()


def test_index_physics_sha_mismatch_refused(tmp_path, monkeypatch):
    pack, shards, out, shard = _fixture(tmp_path)
    _fake_parse(monkeypatch)
    align = json.loads((shard / "ALIGN.json").read_text())
    align["physics_gz_sha256"] = "b" * 64
    _write(shard / "ALIGN.json", align)
    with pytest.raises(DI.I.IdentitySidecarError):
        DI.build_dspeed_identity_sidecar(pack, shards, out)


def test_missing_physics_file_refused(tmp_path, monkeypatch):
    pack, shards, out, shard = _fixture(tmp_path)
    _fake_parse(monkeypatch)
    (shard / "physics_steps.jsonl.gz").unlink()
    with pytest.raises(DI.I.IdentitySidecarError):
        DI.build_dspeed_identity_sidecar(pack, shards, out)


def test_receipt_tamper_refused_before_array_read(tmp_path, monkeypatch):
    pack, shards, out, shard = _fixture(tmp_path)
    _fake_parse(monkeypatch)
    DI.build_dspeed_identity_sidecar(pack, shards, out)
    receipt_path = out / DI.I.RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text())
    receipt["source"] = "features"
    _write(receipt_path, receipt)
    with pytest.raises(DI.I.IdentitySidecarError):
        DI.load_dspeed_identity_sidecar(pack, out)


def test_vehicle_map_tamper_refused(tmp_path, monkeypatch):
    pack, shards, out, shard = _fixture(tmp_path)
    _fake_parse(monkeypatch)
    DI.build_dspeed_identity_sidecar(pack, shards, out)
    target = out / DI.I.MAP_NAME
    mapping = json.loads(target.read_text())
    mapping.append(["chunk4", "wa", "vX"])
    _write(target, mapping)
    with pytest.raises(DI.I.IdentitySidecarError):
        DI.load_dspeed_identity_sidecar(pack, out)


def test_receipt_file_hash(tmp_path, monkeypatch):
    pack, shards, out, shard = _fixture(tmp_path)
    _fake_parse(monkeypatch)
    DI.build_dspeed_identity_sidecar(pack, shards, out)
    sha = hashlib.sha256((out / DI.I.RECEIPT_NAME).read_bytes()).hexdigest()
    assert DI.load_dspeed_identity_sidecar(pack, out).receipt_sha256 == sha
