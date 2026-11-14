"""Geometry table + lane mapping tests (frozen SUMO lane ids)."""
from __future__ import annotations

import pytest

from features import geometry as G


def test_lane_table_maps_the_three_physical_lanes():
    assert G.lane_phys_of("1191554636_0") == 1
    assert G.lane_phys_of("1191554636_1") == 0
    assert G.lane_phys_of("1487606179_0") == 2
    assert G.lane_phys_of("1487606179_1") == 1
    assert G.lane_phys_of("1487606179_2") == 0
    assert G.lane_phys_of("1487606178_0") == 1
    assert G.lane_phys_of("1487606178_1") == 0
    assert G.lane_phys_of("11553048#1_0") == 2


def test_s_down_is_merge_relative():
    # merge edge start == lane offset -> s_down 0 at the merge
    assert G.s_down_of("1487606179_1", 0.0) == (1, 0.0)
    assert G.s_down_of("1487606179_2", 5.0) == (0, 5.0)
    lane, sd = G.s_down_of("1487606178_0", 0.0)
    assert (lane, round(sd, 6)) == (1, 75.44)
    lane, sd = G.s_down_of("1191554636_1", 0.0)
    assert (lane, round(sd, 2)) == (0, -2645.67)


def test_junction_lane_and_unknown_lane():
    assert G.is_junction_lane(":1780739017_1_1")
    assert G.lane_phys_of(":1780739017_1_1") is None
    with pytest.raises(G.GeometryError):
        G.lane_phys_of("999999_0")


def test_entrance_classification():
    assert G.entrance_of_lane_phys(2) == G.ENTRANCE_RAMP
    assert G.entrance_of_lane_phys(0) == G.ENTRANCE_MAINLINE
    assert G.entrance_of_lane_phys(1) == G.ENTRANCE_MAINLINE
