"""Create a tiny real-writer pack for a guarded, full sidecar-stage rehearsal.

This test only creates fresh synthetic inputs.  The integrating task pins the
resulting bytes and runs metadata-plan, then train-sidecars, in separate guarded
children.  No sidecar builder is mocked or called from this fixture test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


RUNS = (
    ("w_native_fixture", "base", 5),
    ("w_native_fixture", "native", 0),
    ("w_native_fixture", "pi", 5),
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def test_create_native_sidecar_fixture(tmp_path, monkeypatch):
    """Leave three K192 TRAIN shards and their pack in the fresh E: test root.

    The middle run has no outcomes file from birth and no DONE output key.  The
    two positive runs use the same synthetic parquet schema as corpus tests.
    ``corpus.pack.pack_shards`` writes the actual pack/INDEX/manifest contract;
    only its atomic JSON handoff is adapted to exclusive creation because the
    synthetic test guard forbids rename/replace.
    """
    assert tmp_path.drive.lower() == "e:", "fixture requires a fresh E: tmp_path"

    import numpy as np

    from corpus import _util as util
    from corpus import pack
    from corpus.tests import synth
    from features import writer

    def write_fresh_json(path, obj):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(obj, indent=1, default=util._json_default)
        with target.open("x", encoding="utf-8") as stream:
            stream.write(body)
        return target

    monkeypatch.setattr(util, "write_json_atomic", write_fresh_json)

    root = tmp_path / "native_sidecar_fixture"
    shards = root / "shards"
    packed = root / "pack"
    index = {"schema": synth.INDEX_SCHEMA, "runs": {}, "pairs": {},
             "totals": {}}
    selected = []
    positive = []

    for ordinal, (world, arm, expected_rows) in enumerate(RUNS):
        chunk = "chunk4"
        key = f"{chunk}/{world}__{arm}"
        rel = f"{chunk}/{world}/{arm}"
        shard = shards / rel
        shard.mkdir(parents=True, exist_ok=False)
        arrays = synth._arrays(k=192, n_ticks=3, n_dec=2, seed=ordinal)
        for name, array in arrays.items():
            np.save(shard / f"{name}.npy", array)

        (shard / "slot_vid.json").write_text(
            json.dumps([f"v{slot}" for slot in range(192)]), encoding="utf-8")
        (shard / "meta.json").write_text(
            json.dumps({"batch": chunk, "chunk": chunk, "world": world,
                        "arm": arm, "split": "TRAIN", "t0_s": 540.0,
                        "n_ticks": 3, "policy": arm}), encoding="utf-8")

        outputs = {name: _sha(shard / f"{name}.npy")
                   for name in synth.V1_NPY_FILES}
        outputs["slot_vid.json"] = _sha(shard / "slot_vid.json")
        outputs["meta.json"] = _sha(shard / "meta.json")
        if expected_rows:
            run = {"chunk": chunk, "world": world, "arm": arm,
                   "split": "TRAIN", "t0_s": 540.0}
            outcomes = synth._outcomes(run, ordinal)
            assert len(outcomes) == expected_rows
            outcomes.to_parquet(shard / "outcomes.parquet", index=False)
            outputs["outcomes.parquet"] = _sha(shard / "outcomes.parquet")
            positive.append(rel)
        else:
            assert not (shard / "outcomes.parquet").exists()
            assert "outcomes.parquet" not in outputs

        source_sha = hashlib.sha256(
            f"synthetic-native-sidecar:{key}".encode("ascii")).hexdigest()
        done = {
            "schema": synth.INDEX_SCHEMA, "batch": chunk, "chunk": chunk,
            "world": world, "arm": arm, "split": "TRAIN", "policy": arm,
            "t0_s": 540.0, "n_ticks": 3, "n_slots": 192,
            "n_dropped_slots": 0, "n_dropped_slot_steps": 0,
            "input_sha256": source_sha, "shard_dir": str(shard),
            "outputs": outputs, "n_outcome_rows": expected_rows,
            "total_bytes": sum(p.stat().st_size for p in shard.iterdir()),
        }
        (shard / "DONE.json").write_text(json.dumps(done, indent=1),
                                          encoding="utf-8")
        writer.update_index(index, done)
        selected.append(rel)

    writer.save_index(shards, index)
    packed_index = pack.pack_shards(shards, packed, workers=1)
    assert packed_index["format"] == "v3"
    assert packed_index["k_max"] == 192
    assert [r["n_outcome_rows"] for r in packed_index["runs"]] == [5, 0, 5]
    assert [r["split"] for r in packed_index["runs"]] == ["TRAIN"] * 3

    receipt = {
        "schema": "wm_n1_native_sidecar_fixture/1",
        "synthetic_root": str(tmp_path),
        "pack": str(packed), "shards_root": str(shards),
        "selected_shards": selected, "positive_shards": positive,
        "expected_outcome_rows": [5, 0, 5], "k_max": 192,
        "manifest_sha256": _sha(packed / "manifest.json"),
        "index_sha256": _sha(packed / "INDEX.json"),
    }
    with (tmp_path / "FIXTURE_RECEIPT.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
