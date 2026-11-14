"""Tests for the per-step TraCI subscription snapshot (env/snapshot.py).

Runs a real (light) SUMO instance and checks that every value served from
the snapshot equals the direct TraCI getter at the same simulation state,
that newly departed vehicles are filled via direct getters until their
subscription starts, and that arrived vehicles drop out of the snapshot.
"""

import math

import pytest
import traci

from config.defaults import merge_config
from env.snapshot import (
    SNAPSHOT_VARS,
    TrafficSnapshot,
    clear_active_snapshot,
    set_active_snapshot,
)
from env.sumo_utils import close_sumo, start_sumo

LOOK_AHEAD = 200.0


@pytest.fixture(scope="module")
def sumo_run():
    """Start one light SUMO instance and warm it up so vehicles exist."""
    cfg = merge_config({"sumo_config": "osm.light.sumocfg"})
    start_sumo(
        sumo_binary=cfg["sumo_binary"],
        sumo_config=cfg["sumo_config"],
        project_root=cfg["project_root"],
        port=None,
    )
    snap = TrafficSnapshot(look_ahead=LOOK_AHEAD)
    for _ in range(300):
        traci.simulationStep(traci.simulation.getTime() + 1.0)
        snap.update()
        if len(traci.vehicle.getIDList()) >= 5:
            break
    yield snap
    clear_active_snapshot()
    close_sumo()


def _step_and_update(snap: TrafficSnapshot) -> None:
    traci.simulationStep(traci.simulation.getTime() + 1.0)
    snap.update()


def test_subscription_values_match_direct_getters(sumo_run):
    snap = sumo_run
    _step_and_update(snap)
    ids = list(traci.vehicle.getIDList())
    assert ids, "need vehicles for a meaningful comparison"
    assert snap.ordered_ids() == ids
    for vid in ids:
        assert snap.speed(vid) == traci.vehicle.getSpeed(vid)
        assert snap.accel(vid) == traci.vehicle.getAcceleration(vid)
        assert snap.road(vid) == traci.vehicle.getRoadID(vid)
        assert snap.lane_id(vid) == traci.vehicle.getLaneID(vid)
        assert snap.lane_index(vid) == traci.vehicle.getLaneIndex(vid)
        assert snap.lane_pos(vid) == traci.vehicle.getLanePosition(vid)
        assert snap.position(vid) == traci.vehicle.getPosition(vid)
        assert snap.time_loss(vid) == traci.vehicle.getTimeLoss(vid)
        assert snap.vtype(vid) == traci.vehicle.getTypeID(vid)
        assert int(snap.signals(vid)) == int(traci.vehicle.getSignals(vid))


def test_leader_matches_direct_get_leader(sumo_run):
    snap = sumo_run
    _step_and_update(snap)
    ids = list(traci.vehicle.getIDList())
    found = 0
    for vid in ids:
        direct = traci.vehicle.getLeader(vid, LOOK_AHEAD)
        sub = snap.leader(vid)
        sub_norm = None if sub is None or sub[0] == "" else (sub[0], sub[1])
        dir_norm = None if direct is None or direct[0] == "" else (direct[0], direct[1])
        if dir_norm is None:
            assert sub_norm is None, f"{vid}: snapshot leader {sub_norm} vs none"
            continue
        assert sub_norm is not None, f"{vid}: direct leader {dir_norm} vs none"
        assert sub_norm[0] == dir_norm[0]
        assert sub_norm[1] == dir_norm[1]
        found += 1
    assert found > 0, "expected at least one vehicle with a leader"


def test_new_departures_filled_directly(sumo_run):
    """Vehicles departing this step have a complete snapshot record even
    though they were only subscribed after the step (the subscribe command
    returns current values immediately; _complete_record fills any gap)."""
    snap = sumo_run
    # Restart the shared simulation so fresh departures are guaranteed;
    # traci is a singleton connection, reused by the remaining tests.
    cfg = merge_config({"sumo_config": "osm.light.sumocfg"})
    close_sumo()
    start_sumo(
        sumo_binary=cfg["sumo_binary"],
        sumo_config=cfg["sumo_config"],
        project_root=cfg["project_root"],
        port=None,
    )
    snap.reset()
    for _ in range(200):
        traci.simulationStep(traci.simulation.getTime() + 1.0)
        departed = list(traci.simulation.getDepartedIDList())
        # Copy the key set BEFORE update(): getAllSubscriptionResults returns
        # the live results dict, which update() mutates via new subscriptions.
        pre_keys = set(traci.vehicle.getAllSubscriptionResults().keys())
        snap.update()
        fresh = [v for v in departed if v not in pre_keys]
        if not fresh:
            continue
        for vid in fresh:
            # Not covered by subscriptions before this step: record must
            # still be complete and equal to the direct values.
            rec = snap.data.get(vid)
            assert rec is not None
            for var in SNAPSHOT_VARS:
                assert var in rec, f"{vid}: var 0x{var:02x} missing"
            assert snap.speed(vid) == traci.vehicle.getSpeed(vid)
        return
    pytest.fail("no departures observed in the first 200 light steps")


def test_arrived_vehicles_removed(sumo_run):
    snap = sumo_run
    for _ in range(600):
        traci.simulationStep(traci.simulation.getTime() + 1.0)
        snap.update()
        arrived = list(traci.simulation.getArrivedIDList())
        if not arrived:
            continue
        for vid in arrived:
            assert vid not in snap.data
            assert not snap.has(vid)
        return
    pytest.skip("no arrivals within 600 steps on the light scenario")


def test_query_helpers_match_with_and_without_snapshot(sumo_run):
    """traci_query readers must return identical values via the snapshot."""
    from env import traci_query

    snap = sumo_run
    _step_and_update(snap)
    ids = list(traci.vehicle.getIDList())

    # Direct (no snapshot active).
    clear_active_snapshot()
    direct = {
        vid: (
            traci_query.leader_info(vid),
            traci_query.leader_ttc(vid),
            traci_query.vehicles_within_radius(vid, 200.0),
            traci_query.ahead_lane_speeds(vid),
        )
        for vid in ids[:10]
    }

    # Same simulation state, snapshot active.
    set_active_snapshot(snap)
    try:
        for vid, expected in direct.items():
            assert traci_query.leader_info(vid) == expected[0], f"leader_info {vid}"
            got_ttc = traci_query.leader_ttc(vid)
            exp_ttc = expected[1]
            if math.isinf(exp_ttc):
                assert math.isinf(got_ttc), f"leader_ttc {vid}"
            else:
                assert got_ttc == exp_ttc, f"leader_ttc {vid}"
            assert traci_query.vehicles_within_radius(vid, 200.0) == expected[2], (
                f"vehicles_within_radius {vid}"
            )
            assert traci_query.ahead_lane_speeds(vid) == expected[3], f"ahead_lane_speeds {vid}"
    finally:
        clear_active_snapshot()


def test_collect_neighbors_matches_direct(sumo_run):
    from env.features import collect_neighbors

    snap = sumo_run
    _step_and_update(snap)
    ids = list(traci.vehicle.getIDList())[:8]

    clear_active_snapshot()
    direct = {vid: collect_neighbors(vid, 100.0, 20) for vid in ids}

    set_active_snapshot(snap)
    try:
        for vid in ids:
            got = collect_neighbors(vid, 100.0, 20)
            exp = direct[vid]
            assert len(got) == len(exp), f"count {vid}"
            for g, e in zip(got, exp):
                assert g.vehicle_id == e.vehicle_id
                assert g.rel_distance == e.rel_distance
                assert g.rel_speed == e.rel_speed
                assert g.rel_lane == e.rel_lane
                assert g.turn_signal == e.turn_signal
                assert g.abs_distance == e.abs_distance
    finally:
        clear_active_snapshot()


def test_zone_features_match_direct(sumo_run):
    from config.constants import MERGE_EDGE, PRE_MERGE_EDGE, RAMP_EDGE
    from env.features import (
        _edge_mean_speed,
        _edge_occupancy,
        _ramp_slow_count,
        build_zone_features,
    )

    snap = sumo_run
    _step_and_update(snap)
    edges = [PRE_MERGE_EDGE, MERGE_EDGE, RAMP_EDGE]
    ids = list(traci.vehicle.getIDList())[:5]

    clear_active_snapshot()
    occ_d = [_edge_occupancy(e) for e in edges]
    spd_d = [_edge_mean_speed(e, 35.0) for e in edges]
    slow_d = [_ramp_slow_count(e, 5.0) for e in edges]
    zone_d = [build_zone_features(v, 35.0) for v in ids]

    set_active_snapshot(snap)
    try:
        assert [_edge_occupancy(e) for e in edges] == occ_d
        assert [_edge_mean_speed(e, 35.0) for e in edges] == spd_d
        assert [_ramp_slow_count(e, 5.0) for e in edges] == slow_d
        for v, exp in zip(ids, zone_d):
            got = build_zone_features(v, 35.0)
            assert list(got) == list(exp), f"zone {v}"
    finally:
        clear_active_snapshot()


def test_trackers_match_direct(sumo_run):
    """JerkTracker / TimeLossTracker fed from the snapshot match direct reads."""
    from env.rewards import JerkTracker, TimeLossTracker

    snap = sumo_run
    jerk_d, jerk_s = JerkTracker(dt=1.0), JerkTracker(dt=1.0)
    loss_d, loss_s = TimeLossTracker(), TimeLossTracker()
    for _ in range(3):
        _step_and_update(snap)
        ids = list(traci.vehicle.getIDList())
        set_active_snapshot(snap)
        try:
            jerk_s.update(ids)
            loss_s.update(ids)
        finally:
            clear_active_snapshot()
        jerk_d.update(ids)
        loss_d.update(ids)
        assert jerk_s._jerk == jerk_d._jerk
        assert loss_s._rates == loss_d._rates
