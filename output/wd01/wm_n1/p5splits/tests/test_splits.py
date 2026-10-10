"""Tests for the WM-S P5 held-out splits (T8).  Synthetic index only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_WM_N1 = Path(__file__).resolve().parents[2]
if str(_WM_N1) not in sys.path:
    sys.path.insert(0, str(_WM_N1))

from p5splits import splits  # noqa: E402


def _rec(run, arm, split, **kw):
    world = run.split("__")[0]
    base = {"run": run, "world": world, "arm": arm, "split": split}
    base.update(kw)
    return base


@pytest.fixture()
def records():
    return [
        _rec("w1__W-mid-early", "W-mid-early", "TRAIN",
             batch="walls_pilot_chunk4", chunk="4", policy="wall"),
        _rec("w2__W-lo-late", "W-lo-late", "TRAIN",
             batch="walls_pilot_chunk4", chunk="4", policy="wall"),
        _rec("w3__pi", "pi", "TRAIN", policy="pi", chunk="2"),
        _rec("w4__native", "native", "TRAIN", policy="native", chunk="2"),
        _rec("w5__base", "base", "TRAIN", policy="damping", chunk="3"),
        _rec("w6__W-mid-early", "W-mid-early", "CALIBRATION",
             batch="walls_pilot_chunk4", chunk="4", policy="wall"),
        _rec("w7__W-lo-late", "W-lo-late", "CALIBRATION",
             batch="walls_pilot_chunk4", chunk="4", policy="wall"),
        _rec("w8__pi", "pi", "CALIBRATION", policy="pi", chunk="2"),
        _rec("w9__native", "native", "CALIBRATION", policy="native", chunk="2"),
    ]


CAL_TUNE = ["w6", "w8"]


def _keys(rows):
    return {(r["run"], r["arm"]) for r in rows}


def test_j4_identification_by_batch_and_by_chunk_arm():
    assert splits.is_j4(_rec("w1__W-mid-early", "W-mid-early", "TRAIN",
                             batch="walls_pilot_chunk4"))
    assert splits.is_j4(_rec("w2__W-lo-late", "W-lo-late", "TRAIN",
                             chunk="4", policy="wall"))
    assert not splits.is_j4(_rec("w4__native", "native", "TRAIN",
                                 chunk="4", policy="native"))


def test_j4_split_requires_wall_policy_not_only_batch_name():
    # A native / PI row of the J4 batch is not a J4 wall arm.
    assert not splits.is_j4(_rec("w4__native", "native", "TRAIN",
                                 batch="walls_pilot_chunk4", policy="native"))
    assert not splits.is_j4(_rec("w4__pi", "pi", "TRAIN",
                                 batch="walls_pilot_chunk4", policy="pi"))
    # An explicit non-wall policy wins over a wall-shaped arm name.
    assert not splits.is_j4(_rec("w1__W-mid-early", "W-mid-early", "TRAIN",
                                 batch="walls_pilot_chunk4", policy="native"))
    # Wall policy plus the frozen v2.1 law identity is J4.
    assert splits.is_j4(_rec("w1__W-mid-early", "W-mid-early", "TRAIN",
                             batch="walls_pilot_chunk4", policy="wall",
                             wall_law="wall-v2.1-closure"))
    # A v2.2 / J4B wall-law identity is never J4.
    assert not splits.is_j4(_rec("w1__W-mid-early", "W-mid-early", "TRAIN",
                                 batch="walls_pilot_chunk4", policy="wall",
                                 wall_law="wall-v2.2"))
    assert not splits.is_j4(_rec("J4B__W-mid-early", "W-mid-early", "TRAIN",
                                 batch="walls_pilot_chunk4", policy="wall"))
    # The split keeps a native row of the batch out of the J4-only training
    # set of test (i).
    records = [
        _rec("w1__W-mid-early", "W-mid-early", "TRAIN",
             batch="walls_pilot_chunk4", policy="wall"),
        _rec("w4__native", "native", "TRAIN",
             batch="walls_pilot_chunk4", policy="native"),
    ]
    spec = splits.build_exclusions(records, "i_unseen_wall_law_v22")
    assert _keys(spec["train_exclude"]) == {("w4__native", "native")}
    assert spec["n_train_windows"] == 1
    assert _keys(spec["eval"]["dev"]["runs"]) == {("w4__native", "native")}
    assert [r["run"] for r in spec["eval"]["registered"]["runs"]] == \
        ["w1__W-mid-early"]


def test_wall_factor_labels():
    assert splits.wall_factor_label(
        _rec("w1__W-mid-early", "W-mid-early", "TRAIN",
             batch="walls_pilot_chunk4")) == "mid"
    assert splits.wall_factor_label(
        _rec("w2__W-lo-late", "W-lo-late", "TRAIN",
             batch="walls_pilot_chunk4")) == "lo"
    assert splits.wall_factor_label(
        _rec("w6__W-mid-early", "W-mid-early", "CALIBRATION",
             wall_speed_factor=0.85, policy="wall")) == "mid"


def test_pi_split_removes_every_pi_arm(records):
    spec = splits.build_exclusions(records, "ii_leave_pi_out",
                                   cal_tune_worlds=CAL_TUNE)
    assert _keys(spec["train_exclude"]) == {("w3__pi", "pi")}
    assert spec["n_train_exclude_windows"] == 1
    assert _keys(spec["eval"]["dev"]["runs"]) == {("w3__pi", "pi")}
    assert _keys(spec["eval"]["registered"]["runs"]) == {("w8__pi", "pi")}
    assert spec["verification"]["leak_count"] == 0


def test_factor_mid_split_removes_only_mid_wall_arms(records):
    spec = splits.build_exclusions(records, "iii_leave_factor_mid_out",
                                   cal_tune_worlds=CAL_TUNE)
    assert _keys(spec["train_exclude"]) == {("w1__W-mid-early", "W-mid-early")}
    assert _keys(spec["eval"]["dev"]["runs"]) == {
        ("w1__W-mid-early", "W-mid-early")}
    assert _keys(spec["eval"]["registered"]["runs"]) == {
        ("w6__W-mid-early", "W-mid-early")}
    train_keys = {splits.record_key(r) for r in records
                  if splits.record_split(r) == "TRAIN"
                  and not splits._excluded("iii_leave_factor_mid_out", r)}
    assert ("w2__W-lo-late", "W-lo-late") in train_keys
    assert spec["verification"]["leak_count"] == 0


def test_unseen_law_split_trains_on_j4_only(records):
    spec = splits.build_exclusions(records, "i_unseen_wall_law_v22",
                                   cal_tune_worlds=CAL_TUNE)
    assert _keys(spec["train_exclude"]) == {
        ("w3__pi", "pi"), ("w4__native", "native"), ("w5__base", "base")}
    reg = spec["eval"]["registered"]
    assert reg["split"] == "J4B (sealed)"
    assert all(r["wall_law"] == "wall-v2.2" and r["sealed"] for r in reg["runs"])
    assert _keys(reg["runs"]) == {
        ("w1__W-mid-early", "W-mid-early"), ("w2__W-lo-late", "W-lo-late")}
    assert spec["verification"]["leak_count"] == 0


def test_materialize_all_zero_and_disjoint(records):
    art = splits.materialize(records, cal_tune_worlds=CAL_TUNE)
    assert art["schema"] == splits.P5_SPLITS_SCHEMA
    assert art["all_exclusions_absent_from_train"] is True
    assert set(art["splits"]) == set(splits.HELD_OUT_TESTS)


def test_leak_count_detects_an_injected_leak(records):
    excluded = [records[2]]                       # the PI TRAIN window
    keys = {splits.record_key(r) for r in excluded}
    assert splits._count_excluded_in_train(records, keys) == [
        {"run": "w3__pi", "arm": "pi", "n_windows": 1}]


def test_load_index_jsonl_and_json(tmp_path):
    rows = [_rec("w1__W-mid-early", "W-mid-early", "TRAIN", policy="wall"),
            _rec("w3__pi", "pi", "TRAIN", policy="pi")]
    jl = tmp_path / "index.jsonl"
    jl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert len(splits.load_index(jl)) == 2
    js = tmp_path / "index.json"
    js.write_text(json.dumps({"windows": rows}), encoding="utf-8")
    assert len(splits.load_index(js)) == 2


def test_unknown_test_refused(records):
    with pytest.raises(ValueError):
        splits.build_exclusions(records, "iv_bogus")
