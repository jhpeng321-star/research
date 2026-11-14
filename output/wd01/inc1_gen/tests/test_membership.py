"""Membership and feasibility edge cases for R5/R6 (A2.2/A2.3, zero SUMO)."""
from __future__ import annotations

import pytest

from inc1_gen.generators import (R5_MIN_MEMBERS, R5Harmonisation,
                                 R6GapOpening, make_generator)
from inc1_gen.tests._helpers import OBSTACLE, T0_S, as_states, base_world, drop_world_row


def ids_of(gen) -> tuple[str, ...]:
    return gen._members


# ------------------------------------------------------------------- R5 set
def test_r5_membership_is_lanes_01_d_100_incl_to_1500_excl():
    gen = R5Harmonisation(as_states(base_world()), OBSTACLE, t0_s=T0_S)
    # d=100 in, d=600 in, d=1499.999 in; d=1500 out; lane 2 out; HDV out;
    # absent out
    assert ids_of(gen) == ("cav.a", "cav.c", "cav.e")
    assert gen.feasible and gen.infeasible_reason is None


def test_r5_d_boundaries_exact():
    def world_at(d):
        return [dict(vehicle_id="c1", is_cav=True, lane_phys=0,
                     s_down=OBSTACLE.s_down - d, v=20.0, a=0.0,
                     leader_gap=None, leader_v=None, native_accel=1.0,
                     present=True)]

    lo = R5Harmonisation(as_states(world_at(100.0)), OBSTACLE, t0_s=T0_S)
    assert lo._members == ("c1",)
    just_under = R5Harmonisation(as_states(world_at(99.999)), OBSTACLE,
                                 t0_s=T0_S)
    assert just_under._members == ()
    hi = R5Harmonisation(as_states(world_at(1499.9999)), OBSTACLE, t0_s=T0_S)
    assert hi._members == ("c1",)
    at_hi = R5Harmonisation(as_states(world_at(1500.0)), OBSTACLE, t0_s=T0_S)
    assert at_hi._members == ()


@pytest.mark.parametrize("n", [0, 1])
def test_r5_infeasible_below_two_members(n):
    world = [dict(vehicle_id=f"c{i}", is_cav=True, lane_phys=0,
                  s_down=OBSTACLE.s_down - (100.0 + 10 * i), v=20.0, a=0.0,
                  leader_gap=None, leader_v=None, native_accel=1.0,
                  present=True) for i in range(n)]
    gen = R5Harmonisation(as_states(world), OBSTACLE, t0_s=T0_S)
    assert len(gen._members) == n < R5_MIN_MEMBERS
    assert not gen.feasible
    assert gen.infeasible_reason == "infeasible_fewer_than_two_members"
    step = gen.step(T0_S + 1.0, as_states(world))
    assert step.commands == {}
    assert not step.feasible
    assert step.infeasible_reason == "infeasible_fewer_than_two_members"
    # everything live is native, HDVs never appear
    assert step.native == tuple(f"c{i}" for i in range(n))


# ------------------------------------------------------------------- R6 set
def r6_world(openers, approachers):
    rows = []
    for i in range(openers):
        rows.append(dict(vehicle_id=f"op{i}", is_cav=True, lane_phys=0,
                         s_down=OBSTACLE.s_down - (100.0 + 10 * i), v=20.0,
                         a=0.0, leader_gap=50.0, leader_v=18.0,
                         native_accel=1.0, present=True))
    for i in range(approachers):
        rows.append(dict(vehicle_id=f"ap{i}", is_cav=True, lane_phys=1,
                         s_down=OBSTACLE.s_down - (40.0 + 10 * i), v=10.0,
                         a=0.0, leader_gap=30.0, leader_v=8.0,
                         native_accel=0.5, present=True))
    return rows


@pytest.mark.parametrize("n_open,n_app,feasible,reason", [
    (2, 0, True, None),
    (1, 1, True, None),
    (1, 2, True, None),
    (2, 2, True, None),
    (1, 0, False, "infeasible_fewer_than_two_members"),
    (0, 2, False, "infeasible_no_gap_opener"),
    (0, 1, False, "infeasible_fewer_than_two_members"),
    (0, 0, False, "infeasible_fewer_than_two_members"),
])
def test_r6_feasibility_needs_two_members_and_one_opener(
        n_open, n_app, feasible, reason):
    gen = R6GapOpening(as_states(r6_world(n_open, n_app)), OBSTACLE,
                       t0_s=T0_S)
    assert gen.feasible is feasible
    assert gen.infeasible_reason == reason
    step = gen.step(T0_S + 1.0, as_states(r6_world(n_open, n_app)))
    assert step.feasible is feasible
    assert step.infeasible_reason == reason


def test_r6_membership_assigns_openers_and_approachers_by_lane_and_d():
    gen = R6GapOpening(as_states(base_world()), OBSTACLE, t0_s=T0_S)
    # openers: lane 0, d in [0, 600): cav.a (d=100), cav.g (d=0); cav.c/e out
    assert gen.openers == ("cav.a", "cav.g")
    # approachers: lane 1, d in [30, 600): cav.b (d=30); cav.f (29.999) out
    assert gen.approachers == ("cav.b",)
    assert ids_of(gen) == ("cav.a", "cav.b", "cav.g")
    assert gen.feasible


def test_r6_d_boundaries_exact():
    def world_at(lane, d):
        return [dict(vehicle_id="c1", is_cav=True, lane_phys=lane,
                     s_down=OBSTACLE.s_down - d, v=20.0, a=0.0,
                     leader_gap=50.0, leader_v=18.0, native_accel=1.0,
                     present=True)]

    for lane, lo, hi in ((0, 0.0, 600.0), (1, 30.0, 600.0)):
        at_lo = R6GapOpening(as_states(world_at(lane, lo)), OBSTACLE, t0_s=T0_S)
        assert at_lo._members == ("c1",)
        below = R6GapOpening(as_states(world_at(lane, lo - 0.001)), OBSTACLE,
                             t0_s=T0_S)
        assert below._members == ()
        at_hi = R6GapOpening(as_states(world_at(lane, hi)), OBSTACLE,
                             t0_s=T0_S)
        assert at_hi._members == ()
        inside = R6GapOpening(as_states(world_at(lane, hi - 0.001)), OBSTACLE,
                              t0_s=T0_S)
        assert inside._members == ("c1",)


def test_obstacle_and_hdvs_are_never_members():
    rows = base_world() + [dict(vehicle_id="stopped.obstacle", is_cav=False,
                                lane_phys=1, s_down=OBSTACLE.s_down, v=0.0,
                                a=0.0, leader_gap=None, leader_v=None,
                                native_accel=0.0, present=True)]
    r5 = R5Harmonisation(as_states(rows), OBSTACLE, t0_s=T0_S)
    r6 = R6GapOpening(as_states(rows), OBSTACLE, t0_s=T0_S)
    assert "stopped.obstacle" not in r5._members
    assert "stopped.obstacle" not in r6._members
    assert "hdv.1" not in r6._members


# ------------------------------------------------- fixed membership/dropouts
def test_membership_fixed_at_t0_late_arrivals_never_join():
    world = base_world()
    gen = R5Harmonisation(as_states(world), OBSTACLE, t0_s=T0_S)
    assert ids_of(gen) == ("cav.a", "cav.c", "cav.e")
    late = base_world() + [dict(vehicle_id="cav.new", is_cav=True, lane_phys=0,
                                s_down=OBSTACLE.s_down - 300.0, v=20.0,
                                a=0.0, leader_gap=None, leader_v=None,
                                native_accel=1.0, present=True)]
    step = gen.step(T0_S + 1.0, as_states(late))
    assert "cav.new" in step.native
    assert "cav.new" not in step.commands


def test_member_leaving_drops_out_then_native_and_never_returns():
    world = base_world()
    gen = R5Harmonisation(as_states(world), OBSTACLE, t0_s=T0_S)
    gone = drop_world_row(world, "cav.c")
    step = gen.step(T0_S + 2.0, as_states(gone))
    # the dropped member is commanded nowhere and, no longer live, appears in
    # neither mask; the surviving members stay tracked
    assert "cav.c" not in step.commands
    assert "cav.c" not in step.native
    assert set(step.commands) == {"cav.a", "cav.e"}
    assert any(e["kind"] == "member_left" and e["vehicle_id"] == "cav.c"
               for e in gen.events)
    # one drop event only, even on later ticks
    gen.step(T0_S + 3.0, as_states(gone))
    left = [e for e in gen.events if e["kind"] == "member_left"]
    assert len(left) == 1


def test_absent_t0_member_row_is_not_latched():
    rows = [dict(vehicle_id="c1", is_cav=True, lane_phys=0,
                 s_down=OBSTACLE.s_down - 500.0, v=20.0, a=0.0,
                 leader_gap=None, leader_v=None, native_accel=1.0,
                 present=False)]
    gen = R5Harmonisation(as_states(rows), OBSTACLE, t0_s=T0_S)
    assert gen._members == ()
    assert not gen.feasible


def test_make_generator_dispatch_and_reject():
    gen = make_generator("R5", as_states(base_world()), OBSTACLE, t0_s=T0_S)
    assert isinstance(gen, R5Harmonisation)
    gen = make_generator("R6", as_states(base_world()), OBSTACLE, t0_s=T0_S)
    assert isinstance(gen, R6GapOpening)
    with pytest.raises(ValueError):
        make_generator("R4", as_states(base_world()), OBSTACLE, t0_s=T0_S)


def test_infeasible_reason_survives_the_whole_window():
    rows = r6_world(0, 2)
    gen = R6GapOpening(as_states(rows), OBSTACLE, t0_s=T0_S)
    for k in (1.0, 100.0, 500.0):
        step = gen.step(T0_S + k, as_states(rows))
        assert step.infeasible_reason == "infeasible_no_gap_opener"
        assert step.commands == {}


def test_states_after_window_end_all_native():
    world = base_world()
    gen = R5Harmonisation(as_states(world), OBSTACLE, t0_s=T0_S)
    inside = gen.step(T0_S + 119.0, as_states(world))
    assert set(inside.commands) == {"cav.a", "cav.c", "cav.e"}
    at_end = gen.step(T0_S + 120.0, as_states(world))
    assert at_end.commands == {}
    assert at_end.native == tuple(sorted(
        r["vehicle_id"] for r in world
        if r["present"] and r["is_cav"] and r["vehicle_id"] != "cav.gone"))
    after = gen.step(T0_S + 300.0, as_states(world))
    assert after.commands == {}
    assert any(e["kind"] == "window_closed" for e in gen.events)
