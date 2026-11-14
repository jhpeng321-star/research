"""Amendment 4 CAL-tune / CAL-cal single loader/filter.

All world lists are synthetic and the split file is written to ``tmp_path``;
no real CAL/HOLDOUT/chunk-5 file is ever opened.
"""
from __future__ import annotations

import json

import pytest

from evalh import cal_split as CS


def _write_split(tmp_path, *, tune=("t1", "t2"), cal=("c1", "c2"),
                 schema=CS.CAL_SPLIT_SCHEMA):
    p = tmp_path / CS.CAL_SPLIT_FILENAME
    p.write_text(json.dumps({
        "schema": schema, "seed": 20261001,
        "cal_tune": list(tune), "cal_cal": list(cal)}), encoding="utf-8")
    return p


def test_load_and_disjoint_halves(tmp_path):
    p = _write_split(tmp_path)
    data = CS.load_cal_split(p)
    assert data["cal_tune"] == ["t1", "t2"]
    assert data["cal_cal"] == ["c1", "c2"]
    assert CS.half_worlds(CS.CAL_TUNE, path=p) == ["t1", "t2"]
    assert CS.half_set(CS.CAL_CAL, path=p) == frozenset({"c1", "c2"})


def test_world_half(tmp_path):
    p = _write_split(tmp_path)
    assert CS.world_half("t1", path=p) == CS.CAL_TUNE
    assert CS.world_half("c1", path=p) == CS.CAL_CAL
    assert CS.world_half("zzz", path=p) is None


def test_resolve_none_returns_canonical_half(tmp_path):
    p = _write_split(tmp_path)
    assert CS.resolve_worlds(CS.CAL_TUNE, None, path=p) == ["t1", "t2"]
    assert CS.resolve_worlds(CS.CAL_CAL, None, path=p) == ["c1", "c2"]


def test_resolve_refuses_the_other_half(tmp_path):
    p = _write_split(tmp_path)
    with pytest.raises(CS.CalSplitError):
        CS.resolve_worlds(CS.CAL_CAL, ["t1"], path=p)
    with pytest.raises(CS.CalSplitError):
        CS.resolve_worlds(CS.CAL_TUNE, ["c1"], path=p)


def test_require_refuses_other_and_unknown(tmp_path):
    p = _write_split(tmp_path)
    with pytest.raises(CS.CalSplitError) as ei:
        CS.require_worlds(["t1"], CS.CAL_CAL, path=p)
    assert CS.CAL_CAL in str(ei.value)
    with pytest.raises(CS.CalSplitError):
        CS.require_worlds(["c1", "nope"], CS.CAL_CAL, path=p)
    # the correct half is accepted
    assert CS.require_worlds(["c1", "c2"], CS.CAL_CAL, path=p) == ["c1", "c2"]


def test_filter_keeps_half_and_refuses_foreign(tmp_path):
    p = _write_split(tmp_path)
    assert CS.filter_worlds(["t1", "c1", "t2"], CS.CAL_TUNE, path=p) == \
        ["t1", "t2"]
    assert CS.filter_worlds(["t1", "c1"], CS.CAL_CAL, path=p) == ["c1"]
    with pytest.raises(CS.CalSplitError):
        CS.filter_worlds(["t1", "sealed-world"], CS.CAL_CAL, path=p)


def test_accepts_dicts_with_world(tmp_path):
    p = _write_split(tmp_path)
    items = [{"world": "t1"}, {"world": "c1"}, {"world": "t2"}]
    assert CS.filter_worlds(items, CS.CAL_TUNE, path=p) == ["t1", "t2"]


def test_overlap_refused(tmp_path):
    p = _write_split(tmp_path, tune=("w1",), cal=("w1",))
    with pytest.raises(CS.CalSplitError):
        CS.load_cal_split(p)


def test_empty_half_refused(tmp_path):
    p = _write_split(tmp_path, tune=())
    with pytest.raises(CS.CalSplitError):
        CS.load_cal_split(p)


def test_bad_schema_refused(tmp_path):
    p = _write_split(tmp_path, schema="nope/1")
    with pytest.raises(CS.CalSplitError):
        CS.load_cal_split(p)


def test_missing_file_refused(tmp_path):
    with pytest.raises(CS.CalSplitError):
        CS.load_cal_split(tmp_path / "nope.json")


def test_frozen_real_split_sha_and_halves():
    # The committed CAL_SPLIT_A4.json (names only) matches A4_FREEZE.json and
    # its halves are disjoint.
    assert CS.verify_cal_split_sha() == CS.LF_SHA_CAL_SPLIT_A4
    data = CS.load_cal_split()
    assert set(data["cal_tune"]).isdisjoint(data["cal_cal"])
    assert data["cal_tune"] and data["cal_cal"]
