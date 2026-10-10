"""P9 corpus passthrough: write / resume / verify / pack roundtrip.

Acceptance checks on the synthetic shard tree (``corpus.tests.synth``, no
data reads): the five Idea A arrays round-trip through
``write_idea_a_channels`` + ``IdeaAPassthrough`` with exact byte
accounting, ``verify_idea_a`` detects tampering, the amendment-2 relabel
flag travels in the corpus metadata while the frozen v1 receipts stay
byte-identical, and the pack/verify roundtrip accepts the new arrays (with
the documented integrator patch applied via monkeypatch).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from corpus import idea_a as CIA
from corpus.tests import synth


def _sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_shard(shard_dir) -> dict:
    """Rebuild a build_shard-shaped dict from a synth shard directory."""
    arrs = {}
    for name in synth.NPY_FILES:
        arrs[name] = np.load(shard_dir / f"{name}.npy")
    meta = json.loads((shard_dir / "meta.json").read_text(encoding="utf-8"))
    meta["n_ticks"] = int(arrs["veh_mask"].shape[0])
    meta.setdefault("split", "TRAIN")
    return dict(arrs, slot_vid=json.loads(
        (shard_dir / "slot_vid.json").read_text(encoding="utf-8")),
        outcomes=None, meta=meta)


@pytest.fixture()
def root(tmp_path):
    synth.write_shards(tmp_path)
    return tmp_path


def _write_all(root):
    from features.idea_a import expected_bytes_idea_a

    records = {}
    index = json.loads((root / "INDEX.json").read_text(encoding="utf-8"))
    for key, rec in index["runs"].items():
        shard_dir = root / rec["shard_dir"]
        rec_out = CIA.write_idea_a_channels(_load_shard(shard_dir), shard_dir)
        assert rec_out["status"] == "written"
        done = json.loads((shard_dir / "IDEA_A_DONE.json")
                          .read_text(encoding="utf-8"))
        n_ticks = int(rec["n_ticks"])
        k_slots = int(np.load(shard_dir / "veh_mask.npy",
                              mmap_mode="r").shape[1])
        # acceptance: expected bytes == written array bytes, exactly
        assert rec_out["arrays_bytes"] == \
            expected_bytes_idea_a(n_ticks, k_slots)
        assert done["arrays_bytes"] == rec_out["arrays_bytes"]
        records[key] = rec_out
    return records


def test_write_roundtrip_and_exact_bytes(root):
    _write_all(root)
    p = CIA.IdeaAPassthrough(root, split="TRAIN")
    assert len(p) == 4                     # 4 TRAIN runs (CAL filtered out)
    assert p.has_idea_a_channels()
    kin = p.get(0, "veh_kin")
    assert kin.shape[-1] == 7 and kin.shape[1] == 16 and \
        kin.dtype == np.float16
    base = np.asarray(p.get(0, "hdv_base", dtype=np.float32))
    assert base.shape[-1] == 6
    assert base[:, :3, :].max() == 0.0     # CAV rows zeroed
    assert base[:, 3:, :].max() > 0.0      # HDV rows populated
    with pytest.raises(KeyError):
        p.get(0, "veh_step")
    p_cal = CIA.IdeaAPassthrough(root, split="CALIBRATION")
    assert len(p_cal) == 1 and p_cal.has_idea_a_channels()


def test_resume_skip_and_force(root):
    records = _write_all(root)
    key, rec = next(iter(records.items()))
    shard_dir = root / json.loads((root / "INDEX.json").read_text(
        encoding="utf-8"))["runs"][key]["shard_dir"]
    again = CIA.write_idea_a_channels(_load_shard(shard_dir), shard_dir)
    assert again["status"] == "skipped"
    forced = CIA.write_idea_a_channels(_load_shard(shard_dir), shard_dir,
                                       force=True)
    assert forced["status"] == "written"
    assert forced["record"]["arrays"]["veh_kin"]["sha256"] == \
        rec["record"]["arrays"]["veh_kin"]["sha256"]


def test_verify_and_tamper_detection(root):
    _write_all(root)
    rep = CIA.verify_idea_a(root)
    assert rep == dict(ok=True, n_runs=5, errors=[])
    index = json.loads((root / "INDEX.json").read_text(encoding="utf-8"))
    key = sorted(index["runs"])[0]
    shard_dir = root / index["runs"][key]["shard_dir"]
    arr = np.lib.format.open_memmap(shard_dir / "hdv_base.npy", mode="r+")
    arr[0, 0, 0] = np.float16(9.0)
    arr.flush()
    rep2 = CIA.verify_idea_a(root)
    assert not rep2["ok"] and any("hdv_base" in e for e in rep2["errors"])


def test_frozen_receipts_untouched_and_relabel_flag(root):
    index = json.loads((root / "INDEX.json").read_text(encoding="utf-8"))
    key = "chunk4/w1__base"
    shard_dir = root / index["runs"][key]["shard_dir"]
    # give the base run a wall token with an aborted terminal tick
    sh = _load_shard(shard_dir)
    wall = np.asarray(sh["wall"]).copy()
    wall[0, 0] = 0.0
    wall[0, 0, 4] = 1.0                     # aborted phase one-hot
    sh["wall"] = wall
    sh["wall_mask"] = np.asarray(sh["wall_mask"]).copy()
    sh["wall_mask"][0, 0] = 1
    before_done = _sha(shard_dir / "DONE.json")
    before_meta = _sha(shard_dir / "meta.json")
    out = CIA.write_idea_a_channels(sh, shard_dir)
    assert out["status"] == "written"
    done = json.loads((shard_dir / "IDEA_A_DONE.json")
                      .read_text(encoding="utf-8"))
    assert done["amendment2_road_end_relabel"] is True
    assert done["road_end_relabelled_n"] == 1
    assert _sha(shard_dir / "DONE.json") == before_done
    assert _sha(shard_dir / "meta.json") == before_meta
    # a non-wall run: flag stays False, count 0
    key2 = "chunk4/w1__pi"
    shard_dir2 = root / index["runs"][key2]["shard_dir"]
    CIA.write_idea_a_channels(_load_shard(shard_dir2), shard_dir2)
    done2 = json.loads((shard_dir2 / "IDEA_A_DONE.json")
                       .read_text(encoding="utf-8"))
    assert done2["amendment2_road_end_relabel"] is False
    assert done2["road_end_relabelled_n"] == 0


def test_idea_a_arrays_constant():
    from features.idea_a import IDEA_A_CHANNELS

    assert CIA.IDEA_A_ARRAYS == tuple(IDEA_A_CHANNELS)
    assert CIA.IDEA_A_DTYPES == {"veh_kin": "float16", "hdv_base": "float16",
                                 "cav_plan_ch": "float16",
                                 "event_tok": "float16",
                                 "event_mask": "uint8"}


def test_pack_verify_roundtrip_with_idea_a_arrays(root, monkeypatch):
    """The documented integrator patch: extend TRAIN_ARRAYS with the P9
    arrays; pack + verify then carry them like every other array."""
    from corpus import pack as PK
    from corpus import verify as VF

    _write_all(root)
    # documented integrator patch; de-duplicated so re-applying it after it
    # has LANDED keeps the array tuple well-formed
    arrays = tuple(dict.fromkeys(PK.TRAIN_ARRAYS + CIA.IDEA_A_ARRAYS))
    assert set(CIA.IDEA_A_ARRAYS) <= set(PK.TRAIN_ARRAYS)
    monkeypatch.setattr(PK, "TRAIN_ARRAYS", arrays)
    monkeypatch.setattr(VF, "TRAIN_ARRAYS", arrays)
    monkeypatch.setattr(VF, "SOURCE_NPY_FILES",
                        arrays + ("slot_index", "decision_t"))
    pack_dir = root / "pack"
    man = PK.pack_shards(root, pack_dir, workers=1)
    for name in CIA.IDEA_A_ARRAYS:
        spec = man["arrays"][name]
        assert list(spec["shape"])[0] == man["totals"]["n_ticks"]
        assert spec["dtype"] == CIA.IDEA_A_DTYPES[name]
    rep = VF.verify_pack(pack_dir)
    assert rep["ok"] is True, rep["errors"]
    assert rep["n_arrays"] == len(arrays)
    # a packed Idea A array is mmap-readable at the concatenated offsets
    arr = np.load(pack_dir / "veh_kin.npy", mmap_mode="r")
    assert arr.shape == tuple(man["arrays"]["veh_kin"]["shape"])
    # tampering a packed Idea A array must fail verify
    arr2 = np.lib.format.open_memmap(pack_dir / "event_mask.npy", mode="r+")
    arr2[0, 0] = 1
    arr2.flush()
    rep2 = VF.verify_pack(pack_dir)
    assert rep2["ok"] is False
    assert any("event_mask" in e for e in rep2["errors"])


def test_verify_cli_reports_ok(root, capsys):
    _write_all(root)
    assert CIA.main(["verify", "--data-root", str(root)]) == 0
    assert '"ok": true' in capsys.readouterr().out
