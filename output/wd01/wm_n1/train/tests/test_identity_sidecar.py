"""DRAFT TESTS PENDING GUARD. Only fresh, explicit synthetic fixtures.

Do not execute until the integrating reviewer accepts the test access guard.
These fixtures create no outcomes/physics and import no corpus writer.
"""
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pytest

from train import identity_sidecar as I


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path, *, first=("alice", "bob"), second=("bob", "alice"),
             first_slots=((0, 1, -1), (1, 0, -1)),
             second_slots=((0, 1, -1), (1, -1, 0)), omit=None):
    pack, shards = tmp_path / "pack", tmp_path / "shards"
    pack.mkdir()
    shards.mkdir()
    runs, items = [], []
    # Interleaved excluded item ensures TRAIN-subset ordinal is NOT global ID.
    specs = [("wa", "native", "TRAIN", first, first_slots),
             ("wb", "native", "CALIBRATION", (), ()),
             ("wa", "pi", "TRAIN", second, second_slots)]
    for pos, (world, arm, split, vids, slots) in enumerate(specs):
        rel = f"chunk4/{world}/{arm}"
        rec = dict(key=f"chunk4/{world}__{arm}", batch="chunk4", chunk="chunk4",
                   world=world, arm=arm, split=split, t0_s=0.0,
                   tick_start=pos * 2, tick_end=pos * 2 + 2, n_ticks=2,
                   dec_start=pos, dec_end=pos + 1, n_outcome_rows=0,
                   input_sha256="a" * 64, shard_dir=f"Z:/old/source/{rel}")
        runs.append(rec)
        items.append(dict(batch="chunk4", chunk="chunk4", world=world, arm=arm,
                          split=split, t0_s=0.0, tick=pos * 2, plan_row=pos,
                          pair_key=f"{world}@t0=0", outcome_rows=[]))
        if split != "TRAIN":
            continue  # Any source open for this run must fail; it does not exist.
        shard = shards / rel
        shard.mkdir(parents=True)
        if omit != "slot_index.npy":
            np.save(shard / "slot_index.npy", np.asarray(slots, dtype=np.int32))
        if omit != "slot_vid.json":
            _write(shard / "slot_vid.json", list(vids))
        hashes = {"slot_index": _sha(shard / "slot_index.npy") if omit != "slot_index.npy" else "a" * 64,
                  "slot_vid.json": _sha(shard / "slot_vid.json") if omit != "slot_vid.json" else "a" * 64}
        if omit != "DONE.json":
            occupied_columns = int(np.any(np.asarray(slots) >= 0, axis=0).sum())
            _write(shard / "DONE.json", dict(rec, schema="wm_n1_features_index/1",
                                            n_slots=occupied_columns, outputs=hashes))
    arrays = {n: dict(shape=[3 if n == "plan" else 6, 3], file=f"{n}.npy")
              for n in ("veh_step", "veh_mask", "plan", "accel_next", "target_mask", "lane_next")}
    _write(pack / "manifest.json", dict(schema="wm_n1_pack/1", k_max=3,
                                        arrays=arrays, items=items))
    _write(pack / "INDEX.json", dict(schema="wm_n1_corpus_pack/2", k_max=3,
                                     arrays=arrays, runs=runs))
    return pack, shards, tmp_path / "new_sidecar"


def _edit(path, change):
    value = json.loads(path.read_text(encoding="utf-8"))
    change(value)
    _write(path, value)


def test_swapped_slots_and_same_vid_different_local_index(tmp_path):
    pack, shards, out = _fixture(tmp_path)
    I.build_identity_sidecar(pack, shards, out)
    side = I.load_identity_sidecar(pack, out)
    assert side.ids_for_item(0).tolist() == [0, 1, -1]
    assert side.ids_for_item(2).tolist() == [1, 0, -1]
    with pytest.raises(I.IdentitySidecarError):
        side.ids_for_item(1)


def test_reused_slot_different_vid_is_not_a_match(tmp_path):
    pack, shards, out = _fixture(tmp_path, second=("new", "bob"))
    I.build_identity_sidecar(pack, shards, out)
    side = I.load_identity_sidecar(pack, out)
    assert side.ids_for_item(0)[0] != side.ids_for_item(2)[0]
    assert side.ids_for_item(0)[1] == side.ids_for_item(2)[1]


def test_selected_tick_uses_current_identity(tmp_path):
    pack, shards, out = _fixture(tmp_path, first=("old", "new"),
                                first_slots=((0, -1, -1), (1, -1, -1)))
    _edit(pack / "manifest.json", lambda m: m["items"][0].update(tick=1))
    I.build_identity_sidecar(pack, shards, out)
    mapping = json.loads((out / I.MAP_NAME).read_text())
    side = I.load_identity_sidecar(pack, out)
    assert mapping[side.ids_for_item(0)[0]][2] == "new"


def test_vid_table_larger_than_slot_width_with_reused_columns(tmp_path):
    # Four distinct vehicles sequentially reuse two columns of a K=3 shard.
    pack, shards, out = _fixture(tmp_path, first=("old_a", "old_b", "new_a", "new_b"),
                                first_slots=((0, 1, -1), (2, 3, -1)))
    _edit(pack / "manifest.json", lambda m: m["items"][0].update(tick=1))
    done = json.loads((shards / "chunk4/wa/native/DONE.json").read_text())
    assert done["n_slots"] == 2
    I.build_identity_sidecar(pack, shards, out)
    mapping = json.loads((out / I.MAP_NAME).read_text())
    row = I.load_identity_sidecar(pack, out).ids_for_item(0)
    assert [mapping[int(i)][2] for i in row if i >= 0] == ["new_a", "new_b"]
    assert row[2] == -1


@pytest.mark.parametrize("count", [-1, 4, 0, 1, 3])
def test_invalid_occupied_column_count_refused(tmp_path, count):
    pack, shards, out = _fixture(tmp_path)
    _edit(shards / "chunk4/wa/native/DONE.json", lambda d: d.update(n_slots=count))
    with pytest.raises(I.IdentitySidecarError):
        I.build_identity_sidecar(pack, shards, out)


def test_readonly_pickle_and_portable_load(tmp_path):
    pack, shards, out = _fixture(tmp_path)
    I.build_identity_sidecar(pack, shards, out)
    portable_pack, portable_side = tmp_path / "portable_pack", tmp_path / "portable_side"
    shutil.copytree(pack, portable_pack)
    shutil.copytree(out, portable_side)
    side = pickle.loads(pickle.dumps(I.load_identity_sidecar(portable_pack, portable_side)))
    row = side.ids_for_item(2)
    with pytest.raises(ValueError):
        row[0] = 99
    with pytest.raises(ValueError):
        row.setflags(write=True)
    assert side.receipt_sha256 == _sha(portable_side / I.RECEIPT_NAME)


def test_exact_train_selection_before_source_opens(tmp_path, monkeypatch):
    pack, shards, out = _fixture(tmp_path)
    original = I._path
    touched = []

    def guard(path, **kwargs):
        candidate = Path(path)
        assert not (shards / "chunk4" / "wb") == candidate
        assert (shards / "chunk4" / "wb") not in candidate.parents
        touched.append(candidate)
        return original(path, **kwargs)

    monkeypatch.setattr(I, "_path", guard)
    I.build_identity_sidecar(pack, shards, out)
    assert any(p.name == "slot_index.npy" for p in touched)


@pytest.mark.parametrize("missing", ["DONE.json", "slot_index.npy", "slot_vid.json"])
def test_missing_identity_file_fails_and_preserves_failure(tmp_path, missing):
    pack, shards, out = _fixture(tmp_path, omit=missing)
    with pytest.raises(I.IdentitySidecarError):
        I.build_identity_sidecar(pack, shards, out)
    assert (out / "BUILD_STARTED.json").is_file()
    assert (out / "BUILD_FAILED.json").is_file()


@pytest.mark.parametrize("kind", ["missing_map", "duplicate_map", "duplicate_tick", "width"])
def test_invalid_identity_mapping_fails(tmp_path, kind):
    kw = {"missing_map": dict(first_slots=((9, -1, -1), (0, -1, -1))),
          "duplicate_map": dict(first=("alice", "alice")),
          "duplicate_tick": dict(first_slots=((0, 0, -1), (1, -1, -1))),
          "width": dict(first_slots=((0, 1), (1, 0)))}[kind]
    pack, shards, out = _fixture(tmp_path, **kw)
    with pytest.raises(I.IdentitySidecarError):
        I.build_identity_sidecar(pack, shards, out)


@pytest.mark.parametrize("kind", ["hash", "done_identity", "missing_hash", "tick", "run_key", "split", "mixed_split", "chunk", "missing_runs", "duplicate_run"])
def test_invalid_metadata_fails_before_array_load(tmp_path, monkeypatch, kind):
    pack, shards, out = _fixture(tmp_path)
    done = shards / "chunk4/wa/native/DONE.json"
    changes = {
        "hash": (done, lambda d: d["outputs"].update(slot_index="b" * 64)),
        "done_identity": (done, lambda d: d.update(world="wrong")),
        "missing_hash": (done, lambda d: d["outputs"].pop("slot_index")),
        "tick": (pack / "INDEX.json", lambda d: d["runs"][0].pop("tick_start")),
        "run_key": (pack / "INDEX.json", lambda d: d["runs"][0].pop("key")),
        "split": (pack / "manifest.json", lambda d: d["items"][0].update(split="TRAINING")),
        "mixed_split": (pack / "manifest.json", lambda d: d["items"][0].update(split="CALIBRATION")),
        "chunk": (pack / "INDEX.json", lambda d: d["runs"][0].update(chunk="chunk6")),
        "missing_runs": (pack / "INDEX.json", lambda d: d.pop("runs")),
        "duplicate_run": (pack / "INDEX.json", lambda d: d["runs"].append(d["runs"][0])),
    }
    _edit(*changes[kind])
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("array read before refusal"))
    with pytest.raises(I.IdentitySidecarError):
        I.build_identity_sidecar(pack, shards, out)


def test_missing_index_refused_before_source_access(tmp_path, monkeypatch):
    pack, shards = tmp_path / "pack", tmp_path / "shards"
    pack.mkdir()
    shards.mkdir()
    _write(pack / "manifest.json", {"schema": "wm_n1_pack/1"})
    monkeypatch.setattr(I, "_source", lambda *a: pytest.fail("source opened"))
    with pytest.raises(I.IdentitySidecarError):
        I.build_identity_sidecar(pack, shards, tmp_path / "new_sidecar")


@pytest.mark.parametrize("chunk", ["chunk5", "Chunk-5", "chunk_5", "chunk", "unknown", "chunk6"])
def test_unrecognized_chunk_refused_before_any_source(tmp_path, monkeypatch, chunk):
    pack, shards, out = _fixture(tmp_path)
    _edit(pack / "INDEX.json", lambda d: d["runs"][0].update(chunk=chunk))
    monkeypatch.setattr(I, "_source", lambda *a: pytest.fail("source opened"))
    with pytest.raises(I.IdentitySidecarError):
        I.build_identity_sidecar(pack, shards, out)


@pytest.mark.parametrize("field", ["split", "sources", "files", "pack"])
def test_receipt_tamper_fails_before_array_read(tmp_path, monkeypatch, field):
    pack, shards, out = _fixture(tmp_path)
    I.build_identity_sidecar(pack, shards, out)
    _edit(out / I.RECEIPT_NAME, lambda r: r.update({field: "tampered"}))
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("array read before refusal"))
    with pytest.raises(I.IdentitySidecarError):
        I.load_identity_sidecar(pack, out)


def test_array_tamper_fails_before_array_read(tmp_path, monkeypatch):
    pack, shards, out = _fixture(tmp_path)
    I.build_identity_sidecar(pack, shards, out)
    with (out / I.IDS_NAME).open("ab") as stream:
        stream.write(b"tamper")
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("array read before refusal"))
    with pytest.raises(I.IdentitySidecarError):
        I.load_identity_sidecar(pack, out)


def test_pack_metadata_drift_fails_before_array_read(tmp_path, monkeypatch):
    pack, shards, out = _fixture(tmp_path)
    I.build_identity_sidecar(pack, shards, out)
    _edit(pack / "manifest.json", lambda m: m.update(extra="drift"))
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("array read before refusal"))
    with pytest.raises(I.IdentitySidecarError):
        I.load_identity_sidecar(pack, out)


def _link(link, target, *, directory=False):
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as exc:
        pytest.skip(f"host symlink capability unavailable: {exc}")


def test_identity_file_symlink_escape(tmp_path, monkeypatch):
    pack, shards, out = _fixture(tmp_path, omit="slot_index.npy")
    target = tmp_path / "outside.npy"
    np.save(target, np.zeros((2, 3), dtype=np.int32))
    _link(shards / "chunk4/wa/native/slot_index.npy", target)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("symlink target read"))
    with pytest.raises(I.IdentitySidecarError):
        I.build_identity_sidecar(pack, shards, out)


def test_output_symlink_escape_and_freshness(tmp_path):
    pack, shards, out = _fixture(tmp_path)
    link = tmp_path / "linked"
    _link(link, shards, directory=True)
    for destination in (link / "new", pack / "new", shards / "new", pack):
        with pytest.raises(I.IdentitySidecarError):
            I.build_identity_sidecar(pack, shards, destination)
    I.build_identity_sidecar(pack, shards, out)
    before = _sha(out / I.RECEIPT_NAME)
    with pytest.raises(I.IdentitySidecarError):
        I.build_identity_sidecar(pack, shards, out)
    assert _sha(out / I.RECEIPT_NAME) == before


def test_recomputed_receipt_cannot_supply_paths(tmp_path, monkeypatch):
    pack, shards, out = _fixture(tmp_path)
    I.build_identity_sidecar(pack, shards, out)
    receipt_path = out / I.RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text())
    receipt["files"]["../outside.npy"] = "a" * 64
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = hashlib.sha256(I._canonical(receipt)).hexdigest()
    _write(receipt_path, receipt)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("array read before refusal"))
    with pytest.raises(I.IdentitySidecarError):
        I.load_identity_sidecar(pack, out)


def test_recomputed_receipt_unknown_metadata_refused(tmp_path, monkeypatch):
    pack, shards, out = _fixture(tmp_path)
    I.build_identity_sidecar(pack, shards, out)
    receipt_path = out / I.RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text())
    receipt["pack"]["selection"][0]["chunk"] = "unknown"
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = hashlib.sha256(I._canonical(receipt)).hexdigest()
    _write(receipt_path, receipt)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("array read before refusal"))
    with pytest.raises(I.IdentitySidecarError):
        I.load_identity_sidecar(pack, out)
