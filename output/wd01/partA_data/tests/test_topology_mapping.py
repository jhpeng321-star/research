"""Physical-lane mapping tests against the real (read-only) network."""
import numpy as np
import pytest

import pdl as L


def test_topology_derives_three_lanes_and_offsets():
    topo, net_path, internal = L.topology()
    assert sorted(topo.chains) == ["accel", "main_left", "main_right"]
    # OFFSET[lane] = first segment length; merge zone from the brief.
    # Empirical (network-derived): main_left pre-merge lane (_1) is the
    # LONGER one; the brief's "2644.22 / 2645.67" is in lane-INDEX order.
    assert abs(topo.offsets["accel"] - 378.66) < 0.01
    assert abs(topo.offsets["main_left"] - 2645.67) < 0.05
    assert abs(topo.offsets["main_right"] - 2644.22) < 0.05
    zone = topo.mandatory_merge_zone("accel")
    assert zone is not None and abs(zone[0] - 378.66) < 0.01 \
        and abs(zone[1] - 454.10) < 0.01
    # lane_of maps each documented segment to the physical lane with
    # continuous s across segment boundaries
    for name in ("main_left", "main_right", "accel"):
        segs = topo.segments_of(name)
        for a, b in zip(segs, segs[1:]):
            assert abs(a.s1 - b.s0) < 1e-9      # boundary continuity
            n1, s1 = topo.lane_of(a.lane_id, a.length - 1.0)
            n2, s2 = topo.lane_of(b.lane_id, 1.0)
            assert (n1, n2) == (name, name)
            assert abs(s2 - s1 - 2.0) < 1e-6    # s advances by the 2 m step


def test_junction_internal_lanes_map_to_zero_s_boundary():
    topo, net_path, internal = L.topology()
    assert len(internal) > 0
    # every documented connection's via lane resolves to a physical boundary
    resolved = 0
    for via, (frm, to) in internal.items():
        fs = topo._segments_by_lane.get(frm)
        ts = topo._segments_by_lane.get(to)
        if fs is None or ts is None:
            continue
        assert fs[0] == ts[0]                    # same physical lane
        assert abs(fs[1].s1 - ts[1].s0) < 1e-9   # zero-s junction
        resolved += 1
    assert resolved > 0
    # spot check the two observed junction ids
    for via in (":13638027693_0_0", ":1780739017_1_0"):
        if via in internal:
            b = L.junction_boundary_s(via)
            assert b is not None and b > 0


def test_map_step_junction_and_unmapped():
    topo, net_path, internal = L.topology()
    lane, s, inj = L.map_step("1191554636_0", 10.0)
    assert (lane, inj) == (1, 0) and abs(s - 10.0) < 1e-9
    lane, s, inj = L.map_step(":13638027693_0_0", 0.0)
    assert (lane, inj) == (-1, 1) and s > 0      # boundary s of main_right
    lane, s, inj = L.map_step(":9999999999_0_0", 0.0, prev_s=100.0, prev_v=20.0)
    assert (lane, inj) == (-1, 1) and abs(s - 110.0) < 1e-9   # dead-reckoned
