"""Pack / verify roundtrip and tamper detection."""
import json

import numpy as np

import synth
from corpus import pack, verify
from corpus import _util as U


def _pack(tmp_path):
    data = tmp_path / "shards"
    synth.write_shards(data)
    out = tmp_path / "pack"
    man = pack.pack_shards(data, out, workers=1)
    return data, out, man


def test_pack_roundtrip_ok(tmp_path):
    _, out, man = _pack(tmp_path)
    assert man["schema"] == pack.PACK_SCHEMA
    assert man["train_schema"] == pack.TRAIN_SCHEMA
    assert man["n_runs"] == 5
    assert man["n_items"] == 5
    assert (out / pack.MANIFEST_NAME).is_file()
    assert (out / pack.SUMS_NAME).is_file()
    train = U.read_json(out / pack.MANIFEST_NAME)
    assert train["schema"] == pack.TRAIN_SCHEMA
    assert set(train["arrays"]) == set(pack.TRAIN_ARRAYS)
    rep = verify.verify_pack(out)
    assert rep["ok"], rep["errors"]
    assert rep["n_runs"] == 5 and rep["n_items"] == 5


def test_verify_detects_tamper(tmp_path):
    _, out, _ = _pack(tmp_path)
    p = out / "veh_step.npy"
    raw = bytearray(p.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    p.write_bytes(bytes(raw))
    rep = verify.verify_pack(out)
    assert not rep["ok"]
    assert any("tampered" in e or "sha256" in e for e in rep["errors"]), rep


def test_verify_detects_missing_sums(tmp_path):
    _, out, _ = _pack(tmp_path)
    (out / pack.SUMS_NAME).unlink()
    rep = verify.verify_pack(out)
    assert not rep["ok"]


def test_verify_detects_duplicate_items(tmp_path):
    _, out, _ = _pack(tmp_path)
    p = out / pack.MANIFEST_NAME
    man = U.read_json(p)
    # simulate the pre-fix collapse: two runs keyed by the same policy arm
    man["items"].append(dict(man["items"][0]))
    p.write_text(json.dumps(man, indent=1), encoding="utf-8")
    rep = verify.verify_pack(out)
    assert not rep["ok"]
    assert any("duplicate item" in e for e in rep["errors"]), rep["errors"]


def test_pack_items_require_policy_and_carry_batch(tmp_path):
    _, out, _ = _pack(tmp_path)
    train = U.read_json(out / pack.MANIFEST_NAME)
    for it in train["items"]:
        assert it["policy"]
        assert it["batch"] == it["chunk"]      # synth fixture: batch == chunk
    native = next(it for it in train["items"] if it["arm"] == "native")
    assert native["policy"] == "native"


def test_convert_roundtrip(tmp_path):
    _, out, _ = _pack(tmp_path)
    out2 = tmp_path / "pack2"
    idx = pack.convert(out, out2, workers=1, recast={"veh_step": "float32"})
    assert idx["arrays"]["veh_step"]["dtype"] == "float32"
    rep = verify.verify_pack(out2)
    assert rep["ok"], rep["errors"]
    train = U.read_json(out2 / pack.MANIFEST_NAME)
    assert train["arrays"]["veh_step"]["dtype"] == "float32"
    a = np.load(out2 / "veh_step.npy", mmap_mode="r")
    assert a.dtype == np.float32


def test_index_json_has_files_and_runs(tmp_path):
    _, out, _ = _pack(tmp_path)
    idx = json.loads((out / pack.INDEX_NAME).read_text(encoding="utf-8"))
    assert idx["files"][pack.MANIFEST_NAME]
    assert all(r["tick_start"] is not None for r in idx["runs"])
    assert idx["totals"]["n_items"] == 5
