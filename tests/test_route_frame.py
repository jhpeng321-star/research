"""Lossless route coordinate: junction lengths, causal dead reckoning, shared geometry."""
import gzip
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from wms.data import route_frame as rf
from wms.data.recorded_prefix import causal_scene
from wms.model.merge_pivot import geometry
from wms.physics.config import PhysicsConfig

NET = Path(__file__).resolve().parents[1] / "assets/research/highway_merge_shepherd_long/osm.net.xml.gz"
DT = 0.5


def internal_lane_lengths():
    text = gzip.open(NET, "rt", encoding="utf-8").read()
    return {lane: float(length) for lane, length in
            re.findall(r'<lane id="(:[^"]+)" index="\d+"[^>]*length="([\d.]+)"', text)}


def test_junction_constants_equal_the_unmodified_network_internal_lanes():
    lengths = internal_lane_lengths()
    assert lengths[":1780739017_0_0"] == rf.JUNCTION_BEFORE_RAMP_M
    assert lengths[":1780739017_1_0"] == lengths[":1780739017_1_1"] == rf.JUNCTION_BEFORE_MAIN_M
    assert lengths[":13638027693_0_0"] == lengths[":13638027693_0_1"] == rf.JUNCTION_AFTER_M


def test_static_shift_regions():
    s = torch.tensor([-20., 0., 40., 75.44, 75.440002, 77.])
    main, ramp = torch.full((6,), 1), torch.full((6,), 2)
    assert rf.static_shift(s, main, 0., 75.44).tolist() == [-3., 0., 0., 0., 0., 8.]
    # the ramp lane ends at the merge-edge end and never has a downstream edge
    assert rf.static_shift(s, ramp, 0., 75.44).tolist() == pytest.approx(
        [-2.93, 0., 0., 0., 0., 0.])


def crossing(k_lane=1, v=4.0, ticks=12, start=70.0, junction=rf.JUNCTION_AFTER_M):
    """One vehicle driving through the downstream junction in the pack (collapsed) frame."""
    route = start + v * DT * np.arange(1, ticks + 1)  # Euler with constant v
    kin = np.zeros((ticks, 1, 7), np.float32)
    kin[:, 0, 4], kin[:, 0, 6] = v, 5.
    edge_end, last = 75.44, None
    for i, x in enumerate(route):
        if x <= edge_end:
            s, lane = x, k_lane
        elif x <= edge_end + junction:
            s, lane = last, -1  # frozen pack value, no lane
        else:
            s, lane = x - junction, k_lane
        if lane >= 0:
            last = s
            kin[i, 0, 1 + lane] = 1.
        kin[i, 0, 0] = s
    return kin, route


def test_dead_reckoning_recovers_the_route_position_inside_and_after_a_junction():
    kin, route = crossing()
    assert (kin[:, 0, 1:4].sum(-1) == 0).any(), "fixture must contain junction rows"
    present = np.ones((len(kin), 1), bool)
    cont = np.ones((len(kin) - 1, 1), bool)
    rec = causal_scene(kin, present, cont)
    shift = rf.route_shift(torch.tensor(rec)[None], torch.tensor(present)[None],
                           torch.tensor(cont)[None], dt_s=DT)[0, :, 0].double().numpy()
    np.testing.assert_allclose(rec[:, 0, 0] + shift, route, atol=2e-5)
    assert shift[-1] == pytest.approx(rf.JUNCTION_AFTER_M)


def test_route_shift_is_causal_and_ignores_the_future():
    kin, _ = crossing()
    present = np.ones((len(kin), 1), bool)
    cont = np.ones((len(kin) - 1, 1), bool)
    base = causal_scene(kin, present, cont)
    full = rf.route_shift(torch.tensor(base)[None], torch.tensor(present)[None],
                          torch.tensor(cont)[None], dt_s=DT)
    changed = kin.copy()
    changed[8:, 0, 0] += 50.
    other = causal_scene(changed, present, cont)
    again = rf.route_shift(torch.tensor(other)[None], torch.tensor(present)[None],
                           torch.tensor(cont)[None], dt_s=DT)
    assert torch.equal(full[:, :8], again[:, :8])


def _shift(kin, cont):
    present = np.ones(kin.shape[:2], bool)
    return rf.route_shift(torch.tensor(kin)[None], torch.tensor(present)[None],
                          torch.tensor(cont)[None], dt_s=DT)[0, :, 0].double().numpy()


def test_only_a_vehicle_that_can_be_inside_an_internal_lane_is_dead_reckoned():
    n = 4
    cont = np.ones((n - 1, 1), bool)
    kin = np.zeros((n, 1, 7), np.float32)
    kin[:, 0, 6] = 5.
    # waiting at the dead end of the ramp lane: no junction behind the ramp lane
    kin[:, 0, 0], kin[:, 0, 1 + 2], kin[:, 0, 4] = 75.44, 1., 0.
    assert not _shift(kin, cont).any()
    kin[:, 0, 4] = 3.  # repeated s with speed on the ramp lane (non-physical): still no junction
    assert not _shift(kin, cont).any()
    # mid-edge mainline vehicle with repeated s: it cannot be on an internal lane
    kin[:, 0, 0], kin[:, 0, 1 + 2], kin[:, 0, 1 + 1] = 40., 0., 1.
    assert not _shift(kin, cont).any()
    # mainline vehicle whose Euler step crosses the merge-edge end: junction rows
    kin[:, 0, 0], kin[:, 0, 4] = 74.0, 4.0  # x: 74 -> 76 -> 78 -> 80
    np.testing.assert_allclose(_shift(kin, cont), [0., 2., 4., 6.], atol=1e-5)
    # ...but not once the dead-reckoned position would leave the 8 m internal lane
    kin[:, 0, 4] = 20.0
    assert not _shift(kin, cont).any()


def test_identity_break_restarts_the_chain_from_the_static_shift():
    kin = np.zeros((4, 1, 7), np.float32)
    kin[:, 0, 6], kin[:, 0, 0], kin[:, 0, 1 + 1], kin[:, 0, 4] = 5., 74.0, 1., 4.0
    cont = np.array([[True], [False], [True]])
    np.testing.assert_allclose(_shift(kin, cont), [0., 2., 0., 2.], atol=1e-5)


def test_geometry_uses_the_route_frame_for_the_overlap_veto():
    """Ramp car at the lane end; target-lane leader 5 m long inside the 8 m junction."""
    kin = torch.zeros(1, 2, 7)
    kin[0, 0, :4] = torch.tensor([75.44, 0., 0., 1.])   # ego, ramp lane
    kin[0, 1, :4] = torch.tensor([74.0, 0., 1., 0.])    # stale pack value of the junction vehicle
    kin[..., 6], kin[0, :, 4] = 5., 4.
    present = torch.ones(1, 2, dtype=torch.bool)
    physics = PhysicsConfig()
    pack = geometry(kin, present, physics)
    assert pack["overlap"][0, 0]  # collapsed frame: the leader's body is on top of the ego
    shift = torch.tensor([[0., 7.]])  # leader is 7 m into the junction: x = 81 -> rear 76 > 75.44
    route = geometry(kin, present, physics, route_shift=shift)
    assert not route["overlap"][0, 0] and route["eligible"][0, 0]
    assert route["route_s"][0, 1] == 81. and route["s"][0, 1] == 74.
    assert route["gap_l"][0, 0] == pytest.approx(76. - 75.44, abs=1e-5)
    # touching/overlapping bodies in the route frame are still vetoed
    assert geometry(kin, present, physics, route_shift=torch.tensor([[0., 3.]]))["overlap"][0, 0]
