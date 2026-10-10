"""Unit tests for the K_max drop-location census (A5 rule note 搂1-搂3).

Synthetic physics runs with a hand-computed drop/zone ground truth; the
frozen featurizer parser is exercised unmodified through census_lib.
"""
from __future__ import annotations

import json

import pytest

import census_lib as CL
from features import constants as C
from features import featurizer as F
from synth_features import (ACCEL, MAIN_LEFT, MAIN_RIGHT, row, veh,
                            write_run)

JUNCTION_LANE = ":1780739017_1_1"


def _concurrent_run_rows():
    """130 concurrent vehicles at t=0.5 -> exactly 2 drops (1 IN, 1 OUT)."""
    vehs = {}
    for i in range(128):                       # rostered, mainline in-zone
        vehs[f"mainline.{i}"] = veh(MAIN_RIGHT, 2000.0 + i * 0.1, 30.0)
    vehs["mainline.128"] = veh(MAIN_LEFT, 900.0, 30.0)   # s_down 900 -> OUT
    vehs["mainline.129"] = veh(ACCEL, 5.0, 20.0)         # lane 2 -> IN
    return [
        row(0.5, vehs),
        # mainline.0/.1 depart (2 slots freed); mainline.2 enters a junction
        row(1.0, {f"mainline.{i}": veh(MAIN_RIGHT, 2000.0 + i * 0.1, 30.0)
                  for i in range(2, 128)}
            | {"mainline.128": veh(MAIN_LEFT, 900.5, 30.0),
               "mainline.129": veh(ACCEL, 5.5, 20.0),
               "mainline.2": veh(JUNCTION_LANE, 3.0, 25.0)}),
        # mainline.2 departs (1 slot freed); mainline.130 arrives OUT-of-zone
        row(1.5, {f"mainline.{i}": veh(MAIN_RIGHT, 2000.0 + i * 0.1, 30.0)
                  for i in range(3, 128)}
            | {"mainline.128": veh(MAIN_LEFT, 901.0, 30.0),
               "mainline.129": veh(ACCEL, 6.0, 20.0),
               "mainline.130": veh(MAIN_RIGHT, 2900.0, 30.0)}),
    ]


def test_zone_boundaries():
    assert CL.in_zone(2, 100000.0)                # lane 2 anywhere
    assert CL.in_zone(2, -99999.0)
    assert CL.in_zone(0, -800.0)                  # inclusive lower bound
    assert not CL.in_zone(0, -800.5)
    assert CL.in_zone(1, 225.43)
    assert not CL.in_zone(1, 225.44)              # exclusive upper bound
    assert CL.in_zone(1, 0.0)
    assert CL.in_zone(-1, 0.0, junction=True)     # junction-internal IN-zone
    with pytest.raises(Exception):
        CL.in_zone(7, 0.0)


def test_census_drops_and_zones(tmp_path):
    root = write_run(tmp_path / "run", _concurrent_run_rows(), decoys=False)
    out = CL.census_run(root, 0.0)

    # dropped @K=128: mainline.128 (OUT) + mainline.129 (IN) at t=0.5 only
    assert out["dropped_total"] == 2
    assert out["dropped_in"] == 1
    assert out["dropped_out"] == 1
    # rostered: t=0.5 128 IN; t=1.0 125 IN (main_right) + 2 IN (junction,
    # accel) + 1 OUT (main_left 900) ; t=1.5 125 IN + 1 IN (accel)
    # + 1 OUT (main_left) + 1 OUT (main_right 500)
    assert out["rostered"] == 128 + 128 + 128
    assert out["rostered_in"] == 128 + 127 + 126
    assert out["rostered_out"] == 0 + 1 + 2
    # max concurrency 130 < 192: nothing would still be dropped at K=192
    assert out["n_dropped_192"] == 0
    # parser agreement (integrity invariant)
    parsed = F.parse_run(root, 0.0)
    assert out["dropped_total"] == parsed["n_dropped_concurrent"]
    # shard meta counter agreement (build_shard records the same counter)
    shard = F.build_shard("chunk4", "wsynth", "base", root, split="TRAIN",
                          plan_row=dict(world="wsynth", t0_s=0.0,
                                        split="TRAIN"), outcomes_root=None)
    assert out["dropped_total"] == shard["meta"]["n_dropped_slot_steps"]
    # per-tick rostered histogram: ticks 0 (no row <= t_origin), 1, 2, 3
    assert out["tick_hist"][0] == 1
    assert out["tick_hist"][128] == 3
    assert len(out["tick_hist"]) == 129        # rostered counts never hit 130


def test_residual_at_k192(tmp_path):
    vehs = {}
    for i in range(200):                       # 200 concurrent on the mainline
        vehs[f"mainline.{i}"] = veh(MAIN_RIGHT, 2000.0 + i * 0.1, 30.0)
    root = write_run(tmp_path / "run", [row(0.5, vehs),
                                        row(1.0, vehs)], decoys=False)
    out = CL.census_run(root, 0.0)
    assert out["dropped_total"] == 2 * (200 - 128)      # 144 at K=128
    assert out["n_dropped_192"] == 2 * (200 - 192)      # 16 at K=192
    assert out["rostered"] == 2 * 128


def test_frozen_parser_untouched(tmp_path):
    root = write_run(tmp_path / "run", _concurrent_run_rows(), decoys=False)
    json_before = F.json
    k_before = C.K_MAX
    CL.census_run(root, 0.0)
    assert F.json is json_before
    assert C.K_MAX == k_before == 128
    # and the parser still behaves identically after the census
    parsed = F.parse_run(root, 0.0)
    assert parsed["n_dropped_concurrent"] == 2


def test_recorder_mirrors_t_origin_filter(tmp_path):
    rows = _concurrent_run_rows()
    root = write_run(tmp_path / "run", rows, decoys=False)
    # t_origin 0.5 skips the first row entirely (parser 'continue' path)
    out = CL.census_run(root, 0.5)
    assert out["dropped_total"] == 0
    assert out["rostered"] == 256              # only the t=1.0/1.5 rows
    assert json.loads((root / "run" / "physics_steps.jsonl")
                      .read_text(encoding="utf-8").splitlines()[0]
                      )["time_s"] == 0.5
