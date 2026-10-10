"""A3.4 held-out-strategy configs: train-exclusion / eval-inclusion sets."""
from __future__ import annotations

import pytest

from evalh import idea_a as ia
from evalh.idea_a import (HIGHEST_J4_WALL_SPEED_FACTOR, HeldOutViolation,
                          RunTag)


def _tests():
    return ia.held_out_protocol()


def test_protocol_shape_and_j4b_flag():
    tests = _tests()
    assert tuple(t.name for t in tests) == ("i", "ii", "iii")
    assert tests[0].reads_j4b is True
    assert tests[1].reads_j4b is False
    assert tests[2].reads_j4b is False
    assert "v2.1" in tests[0].train_filter


def test_i_trains_on_j4_and_reads_only_j4b():
    t = _tests()[0]
    runs = [RunTag(source="J4", split="train", policy="wall",
                   wall_law=ia.J4_WALL_LAW),
            RunTag(source="J4B", split="test", policy="wall",
                   wall_law=ia.J4B_WALL_LAW),
            RunTag(source="CAL", split="cal", policy="pi")]
    train = ia.held_out_train_runs(t, runs)
    assert [r.source for r in train] == ["J4"]
    ev = ia.held_out_eval_runs(t, runs)
    assert [r.source for r in ev] == ["J4B"]


def test_ii_drops_every_pi_arm_from_train_and_evaluates_cal_pi():
    t = _tests()[1]
    runs = [RunTag(source="TRAIN", split="train", policy="pi"),
            RunTag(source="TRAIN", split="train", policy="damping"),
            RunTag(source="CAL", split="cal", policy="native"),
            RunTag(source="CAL", split="cal", policy="pi")]
    train = ia.held_out_train_runs(t, runs)
    assert all(r.policy != "pi" for r in train)
    assert len(train) == 2
    ev = ia.held_out_eval_runs(t, runs)
    assert ev == [RunTag(source="CAL", split="cal", policy="pi")]


def test_iii_drops_highest_j4_wall_factor():
    t = _tests()[2]
    assert t.factor == HIGHEST_J4_WALL_SPEED_FACTOR == "mid"
    runs = [RunTag(source="CAL", split="cal", policy="wall",
                   wall_speed_factor="mid"),
            RunTag(source="TRAIN", split="train", policy="wall",
                   wall_speed_factor="mid"),
            RunTag(source="TRAIN", split="train", policy="wall",
                   wall_speed_factor="lo"),
            RunTag(source="CAL", split="cal", policy="wall",
                   wall_speed_factor="lo"),
            RunTag(source="TRAIN", split="train", policy="damping")]
    train = ia.held_out_train_runs(t, runs)
    assert all(not (r.policy == "wall" and r.wall_speed_factor == "mid")
               for r in train)
    assert any(r.policy == "wall" and r.wall_speed_factor == "lo"
               for r in train)
    ev = ia.held_out_eval_runs(t, runs)
    assert ev == [RunTag(source="CAL", split="cal", policy="wall",
                         wall_speed_factor="mid")]


def test_j4b_refused_outside_config_i():
    runs = [RunTag(source="J4B", split="test", policy="wall",
                   wall_law=ia.J4B_WALL_LAW)]
    for t in _tests()[1:]:
        with pytest.raises(HeldOutViolation):
            ia.held_out_eval_runs(t, runs)
    with pytest.raises(HeldOutViolation):
        ia.assert_j4b_read_allowed(_tests()[1])
    ia.assert_j4b_read_allowed(_tests()[0])


def test_heldout_config_cli_writes_receipt_without_data(tmp_path):
    import json

    rc = ia.heldout_config_main(["--test", "i", "--out", str(tmp_path)])
    assert rc == 0
    receipt = json.loads(
        (tmp_path / "HELDOUT_CONFIG.json").read_text(encoding="utf-8"))
    assert receipt["test"] == "i"
    assert receipt["reads_j4b"] is True
    assert receipt["kind"] == "headline"
    assert "J4B" in receipt["eval_filter"]
    assert "chunk-5" not in receipt["train_filter"]

