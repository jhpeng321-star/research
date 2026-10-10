"""Tracking: identities kept, no insertions, exits counted, NaN tails,
free-road counting; adapter sign/lane mapping + the no-vType rule."""
import sys
from pathlib import Path

import numpy as np
import pytest

WD01 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WD01))

from partA_v2g import adapter as AD
from partA_v2g.engine import (PartAV2GEngine, free_road_flags,
                              load_params)

P = load_params()
OFF = P["geometry"]["lane_offsets_m"]
LLEN = P["geometry"]["lane_lengths_m"]


def test_identities_kept_no_insertions_exits_counted():
    """Every input id stays in the snapshots until it exits the road;
    the roster never grows; exits are counted with times."""
    vehicles = {
        "fast": dict(lane=1, s=3150.0, v=30.0),    # exits lane 1 quickly
        "slow": dict(lane=0, s=1000.0, v=5.0),
        "ramp": dict(lane=2, s=100.0, v=15.0),
    }
    e = PartAV2GEngine(P, t0=480.0, vehicles=vehicles)
    e.run(40)
    ids0 = set(k for k in e.snapshots[0]) - {"__t__"}
    assert ids0 == {"fast", "slow", "ramp"}
    for j in range(1, 41):
        idsj = set(k for k in e.snapshots[j]) - {"__t__"}
        assert idsj <= ids0                    # never grows: no insertions
        assert "slow" in idsj and "ramp" in idsj
    assert "fast" in e.exited
    assert e.exited["fast"] > 480.0
    assert e.n_exited_before(500.0) == 1
    assert e.n_exited_before(481.0) == 0
    assert e.n_input == 3


def test_trajectory_nan_suffix_after_exit():
    vehicles = {"fast": dict(lane=1, s=3180.0, v=30.0)}
    e = PartAV2GEngine(P, t0=480.0, vehicles=vehicles)
    e.run(30)
    rows = e.trajectory_table()["fast"]
    n_real = sum(1 for r in rows if r is not None)
    assert 1 < n_real < len(rows)
    assert all(r is None for r in rows[n_real:])
    assert all(r is not None for r in rows[:n_real])


# ------------------------------------------------------------ free road

def test_free_road_flags_predictor_rule():
    """Frontmost vehicle of a truncated zone lane is flagged; a zone
    reaching the lane end is not."""
    vehicles = {
        "a": dict(lane=1, s=2000.0, v=20.0),
        "b": dict(lane=1, s=2100.0, v=20.0),
        "c": dict(lane=0, s=3000.0, v=20.0),
    }
    zone_trunc = {0: (1800.0, 2900.0), 1: (1800.0, 2860.0),
                  2: (0.0, 454.1)}
    zone_full = {0: (0.0, LLEN[0]), 1: (0.0, LLEN[1]), 2: (0.0, LLEN[2])}
    # lane 1 frontmost = b (leader beyond zone -> flagged); lane 0
    # frontmost c (zone truncated at 2900 < 3199 -> flagged)
    assert free_road_flags(vehicles, zone_trunc, LLEN) == ["b", "c"]
    # full-lane zone: a missing leader truly means empty road
    assert free_road_flags(vehicles, zone_full, LLEN) == []
    # no zone info at all: every frontmost vehicle flagged
    assert free_road_flags(vehicles, None, LLEN) == ["b", "c"]


# -------------------------------------------------------------- adapter

def test_r0_group_table_and_sign():
    """g0 -> main_right(1), g1 -> main_left(0), g2/g3 -> accel(2);
    s_down = -d; model s = OFFSETS[lane] + s_down."""
    assert AD.R0_GROUP_TO_LANE_PHYS == {"g0": 1, "g1": 0, "g2": 2, "g3": 2}
    rec = AD.r0_record_to_view({"group": "g0", "d": 100.0})
    assert rec["s_down_m"] == pytest.approx(-100.0)
    assert rec["lane_phys"] == 1
    # a mainline vehicle at model s=2000 on lane 1 has d = 2644.22-2000
    s = AD.view_to_model_s(1, -644.22, OFF)
    assert s == pytest.approx(2000.0)
    # round trip
    assert AD.model_s_to_view_s_down(1, s, OFF) == pytest.approx(-644.22)
    # accel lane: s_down in (-378.66, 75.44]
    s2 = AD.view_to_model_s(2, 20.0, OFF)
    assert s2 == pytest.approx(398.66)          # inside the merge zone
    # ramp/merge split on the accel chain
    rec3 = AD.r0_record_to_view({"group": "g3", "d": 300.0})
    s3 = AD.view_to_model_s(rec3["lane_phys"], rec3["s_down_m"], OFF)
    assert 0.0 < s3 < 378.66                    # ramp segment


def test_adapter_rejects_unmappable_records():
    with pytest.raises(AD.AdapterError):
        AD.r0_record_to_view({"group": "g9", "d": 1.0})
    with pytest.raises(AD.AdapterError):
        AD.r0_record_to_view({"speed_mps": 10.0})     # nothing to map
    with pytest.raises(AD.AdapterError):
        AD.parse_sample({"vehicles": {"x": {"speed_mps": 5.0}}})


def test_adapter_maps_a_full_sample():
    hybrid = {
        "t_s": 480.0,
        "header": {"t_s": 480.0, "k": 3, "K": 8, "sample_weight": 0.125},
        "zone_extent": {"1": [-800.0, 200.0]},
        "vehicles": {
            "m": {"s_down_m": -700.0, "lane_phys": 1, "speed_mps": 28.0,
                  "is_cav": False, "length_m": 5.0,
                  "acceleration_mps2": 0.1},
            "r0raw": {"group": "g3", "d": 300.0, "speed_mps": 12.0},
        },
        "cav_controller_block": {"m": {"action": "native"}},
    }
    t = AD.parse_sample(hybrid, P)
    assert t.t_s == 480.0 and t.k == 3 and t.K == 8
    assert t.sample_weight == pytest.approx(0.125)
    assert t.vehicles["m"]["s"] == pytest.approx(2644.22 - 700.0)
    assert t.vehicles["m"]["a"] == pytest.approx(0.1)
    # the r0 record maps onto the accel ramp segment
    raw = t.vehicles["r0raw"]
    assert raw["lane"] == 2
    assert 0.0 < raw["s"] < 378.66
    assert raw["v"] == pytest.approx(12.0)
    # zone extent maps into model s on lane 1
    assert t.zone_extent_s[1][0] == pytest.approx(2644.22 - 800.0)
    assert t.zone_extent_s[1][1] == pytest.approx(2644.22 + 200.0)
    assert t.cav_block == {"m": {"action": "native"}}


def test_no_vtype_read_anywhere():
    """Identical rollouts with and without vType-ish keys on the input
    records: the engine must never read a driver type."""
    base = {
        "a": dict(lane=1, s=2000.0, v=25.0),
        "b": dict(lane=1, s=2060.0, v=20.0),
        "c": dict(lane=2, s=380.0, v=18.0),
    }
    typed = {k: dict(v, vtype="cautious_car", type_id="aggressive_car",
                     driver_type=2) for k, v in base.items()}
    e1 = PartAV2GEngine(P, t0=480.0, vehicles=base).run(40)
    e2 = PartAV2GEngine(P, t0=480.0, vehicles=typed).run(40)
    assert e1.snapshots == e2.snapshots
    assert e1.lc_counts == e2.lc_counts
