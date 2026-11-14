"""Packed layout parity with the TRAIN read-side contract (train/dataset.py).

The reader lives in ``train_reader.py`` (a MemmapCorpus-equivalent over the
packed output) and ``packsynth.py`` is the exact fixture the TRAIN tests use
(copied read-only from ``zcode/wmn1train``).  We prove our pack is readable by
that reader and structurally identical to the packsynth fixture.
"""
import json

import packsynth
import synth
from corpus import pack
from train_reader import TrainReader

_REQUIRED_ITEM_KEYS = ("world", "arm", "chunk", "split", "t0_s", "pair_key",
                       "tick", "plan_row", "outcome_rows")


def _our_pack(tmp_path):
    data = tmp_path / "shards"
    synth.write_shards(data)
    out = tmp_path / "pack"
    pack.pack_shards(data, out, workers=1)
    return out


def test_train_reader_over_our_pack(tmp_path):
    out = _our_pack(tmp_path)
    r = TrainReader(out, "TRAIN")
    assert len(r) == 4                      # w1 base/native/pi + w2 native
    assert r.worlds() == ["w1", "w2"]
    item = r[0]
    assert item["veh_step"].shape == (16, 16)
    assert item["plan"].shape == (16, 19, 8)
    assert item["wall"].shape == (4, 11)
    assert item["counter"].shape == (2, 30)
    assert item["outcome_quantiles"].shape == (5, 3)
    assert item["outcome_mask"].shape == (5,)
    cal = TrainReader(out, "CALIBRATION")
    assert len(cal) == 1 and cal.items[0]["world"] == "c1"


def test_structural_parity_with_packsynth(tmp_path):
    ours_pack = _our_pack(tmp_path)
    syn = tmp_path / "synth_pack"
    packsynth.write_default_pack(syn)

    ours = json.loads((ours_pack / "manifest.json").read_text(
        encoding="utf-8"))
    ref = json.loads((syn / "manifest.json").read_text(encoding="utf-8"))

    assert ours["schema"] == ref["schema"] == "wm_n1_pack/1"
    # our pack carries the TRAIN contract arrays PLUS the P9 Idea A channels
    assert set(ref["arrays"]) <= set(ours["arrays"])
    assert set(ours["arrays"]) - set(ref["arrays"]) == \
        {"veh_kin", "hdv_base", "cav_plan_ch", "event_tok", "event_mask"}
    assert ours["outcomes"]["file"] == ref["outcomes"]["file"]
    for name in ref["arrays"]:
        assert ours["arrays"][name]["dtype"] == ref["arrays"][name]["dtype"]
        # per-run axis differs (2-3 ticks/run); the tail (K, F...) must match
        assert ours["arrays"][name]["shape"][1:] == \
            ref["arrays"][name]["shape"][1:]
    for name in ("k_max",):
        assert name in ours and name in ref
    for it in ours["items"]:
        assert all(k in it for k in _REQUIRED_ITEM_KEYS)
    for it in ref["items"]:
        assert all(k in it for k in _REQUIRED_ITEM_KEYS)

    # both packs are consumed by the same reader without error
    assert len(TrainReader(syn, "TRAIN")) == 5
    assert len(TrainReader(ours_pack, "TRAIN")) == 4
