"""R0-R4 schedule tests (selection, braking/hold, early-slow, wall pair)."""
from __future__ import annotations

import pytest

from inc1 import constants as C
from inc1 import schedules as S


def cav(vid, s_down, lane=1, speed=20.0, native_accel=None):
    return S.CavState(vehicle_id=vid, speed_mps=speed, s_down_m=s_down,
                      lane_phys=lane, allowed_speed_mps=30.0,
                      native_accel_mps2=native_accel)


def test_r0_is_all_native():
    states = [cav("cav.0", -300), cav("cav.1", -100)]
    plan = S.make_schedule("R0", states, 40.0).step(C.T0_S, states)
    assert plan.wall == {}
    assert set(plan.native) == {"cav.0", "cav.1"}


def test_r1_selects_nearest_upstream_in_either_lane():
    states = [cav("cav.0", 20.0, lane=1), cav("cav.1", -100.0, lane=0),
              cav("cav.2", 900.0, lane=1)]
    schedule = S.make_schedule("R1", states, 40.0)
    assert schedule.selected == "cav.0"          # nearest upstream (s_down 20)
    plan = schedule.step(C.T0_S, states)
    assert plan.wall == {"cav.0": C.R1_BRAKE_MPS2}
    assert set(plan.native) == {"cav.1", "cav.2"}


def test_r1_brakes_then_holds_30s_then_native():
    states = [cav("cav.0", 20.0, speed=10.0)]
    schedule = S.make_schedule("R1", states, 40.0)
    assert schedule.step(C.T0_S, states).wall == {"cav.0": -2.0}
    stopped = [cav("cav.0", 20.0, speed=0.0)]
    holding = schedule.step(C.T0_S + 5, stopped)
    assert holding.wall == {"cav.0": 0.0}
    still_holding = schedule.step(C.T0_S + 34.9, stopped)
    assert still_holding.wall == {"cav.0": 0.0}
    released = schedule.step(C.T0_S + 36, stopped)
    assert released.wall == {}
    assert "cav.0" in released.native


def test_r1_without_qualifier_falls_back_to_r0():
    states = [cav("cav.0", 900.0), cav("cav.1", -800.0)]
    schedule = S.make_schedule("R1", states, 40.0)
    assert schedule.selected is None
    plan = schedule.step(C.T0_S, states)
    assert plan.wall == {}
    assert any(e["kind"] == "r1_no_qualifier" for e in schedule.events)


def test_r1_command_is_bounded_to_registered_interval():
    """R1 must never issue outside [-2.0, 0] m/s^2 (registered limits)."""
    states = [cav("cav.0", 20.0, speed=10.0)]
    schedule = S.make_schedule("R1", states, 40.0)
    for now in (C.T0_S, C.T0_S + 1, C.T0_S + 5):
        plan = schedule.step(now, [cav("cav.0", 20.0, speed=0.0)])
        for accel in plan.wall.values():
            assert -2.0 - 1e-9 <= accel <= 0.0 + 1e-9
    assert S._clip(5.0, C.R1_BRAKE_MPS2, 0.0) == 0.0
    assert S._clip(-99.0, C.R1_BRAKE_MPS2, 0.0) == C.R1_BRAKE_MPS2


def test_r2_window_and_decel_floor():
    states = [cav("cav.0", -500.0), cav("cav.1", -100.0),
              cav("cav.2", -1100.0), cav("cav.3", 0.0)]
    schedule = S.make_schedule("R2", states, 40.0)
    assert schedule.controlled == ["cav.0"]      # only in [-960, -160)
    plan = schedule.step(C.T0_S, states)
    assert plan.wall == {"cav.0": -C.R2_DECEL_MAX_MPS2}
    after = schedule.step(C.T0_S + C.R2_DURATION_S, states)
    assert after.wall == {}


def test_r2_accel_is_capped_at_the_native_idm_accel():
    """R2 may accelerate toward 15 m/s but never above the vehicle's native
    IDM acceleration; the decel floor stays 1.5 m/s^2."""
    # v = 2 -> desired +13 m/s^2, capped at the native accel 2.6
    states = [cav("cav.0", -500.0, speed=2.0, native_accel=2.6)]
    plan = S.make_schedule("R2", states, 40.0).step(C.T0_S, states)
    assert plan.wall == {"cav.0": 2.6}
    # native accel 0.5 -> 0.5
    states = [cav("cav.0", -500.0, speed=2.0, native_accel=0.5)]
    plan = S.make_schedule("R2", states, 40.0).step(C.T0_S, states)
    assert plan.wall == {"cav.0": 0.5}
    # unknown native accel -> never accelerate
    states = [cav("cav.0", -500.0, speed=2.0, native_accel=None)]
    plan = S.make_schedule("R2", states, 40.0).step(C.T0_S, states)
    assert plan.wall == {"cav.0": 0.0}


def test_r2_negative_native_accel_keeps_the_registered_decel_floor():
    """A native decel below -1.5 must not push R2 past the registered floor."""
    states = [cav("cav.0", -500.0, speed=20.0, native_accel=-3.0)]
    plan = S.make_schedule("R2", states, 40.0).step(C.T0_S, states)
    assert plan.wall == {"cav.0": -C.R2_DECEL_MAX_MPS2}


def test_r3_controls_upstream_subset_via_damping():
    states = [cav("cav.0", -500.0), cav("cav.1", 100.0), cav("cav.2", -50.0)]
    schedule = S.make_schedule("R3", states, 40.0)
    assert schedule.selected == ["cav.0", "cav.2"]
    plan = schedule.step(C.T0_S, states)
    assert plan.wall == {}                        # damping = default branch
    assert set(plan.native) == {"cav.1"}


def test_r4_is_not_a_control_plan_and_never_reimplemented():
    """R4 is driven through the patched frozen v2.2 driver
    (``inc1.r4_driver``), not as a ControlPlan schedule; the law is never
    re-implemented here (ADDENDUM_1 A1.2/A1.3)."""
    states = [cav("cav.0", -1450.0, lane=0), cav("cav.1", -1400.0, lane=1)]
    with pytest.raises(S.ScheduleError) as excinfo:
        S.make_schedule("R4", states, 40.0)
    assert "r4_driver" in str(excinfo.value)
    assert not hasattr(S, "R4WallV22")
    assert not hasattr(S, "select_wall_pair")


def test_r5_is_backed_by_the_shared_generator():
    """ADDENDUM_2 A2.2: R5 is a ControlPlan schedule from inc1_gen."""
    from inc1 import scripted
    states = [cav("cav.0", -60.0, lane=1, speed=20.0, native_accel=1.0),
              cav("cav.1", -160.0, lane=0, speed=20.0, native_accel=1.0)]
    schedule = S.make_schedule("R5", states, 40.0)
    assert isinstance(schedule, scripted.ScriptedSchedule)
    assert schedule.feasible is True
    plan = schedule.step(C.T0_S, states)
    assert set(plan.wall) == {"cav.0", "cav.1"}
    for accel in plan.wall.values():
        assert -1.5 - 1e-9 <= accel <= 1.0 + 1e-9


def test_r5_fewer_than_two_members_is_infeasible():
    from inc1 import scripted
    states = [cav("cav.0", -60.0, lane=1, speed=20.0, native_accel=1.0)]
    schedule = S.make_schedule("R5", states, 40.0)
    assert isinstance(schedule, scripted.ScriptedSchedule)
    assert schedule.feasible is False
    assert S.arm_feasibility("R5", schedule) == (
        False, "infeasible_fewer_than_two_members")
    plan = schedule.step(C.T0_S, states)
    assert plan.wall == {}
    assert "cav.0" in plan.native


def test_r6_gap_opener_and_blocked_lane_approacher():
    """ADDENDUM_2 A2.3: lane 0 opener (d<600) + lane 1 approacher (d>=30)."""
    from inc1 import scripted
    opener = S.CavState(vehicle_id="cav.0", speed_mps=20.0, s_down_m=-100.0,
                        lane_phys=0, allowed_speed_mps=30.0,
                        native_accel_mps2=1.0, leader_gap_m=30.0,
                        leader_v_mps=10.0)
    approacher = cav("cav.1", 0.0, lane=1, speed=20.0, native_accel=1.0)
    schedule = S.make_schedule("R6", [opener, approacher], 40.0)
    assert isinstance(schedule, scripted.ScriptedSchedule)
    assert schedule.feasible is True
    plan = schedule.step(C.T0_S, [opener, approacher])
    assert set(plan.wall) == {"cav.0", "cav.1"}
    for accel in plan.wall.values():
        assert -1.5 - 1e-9 <= accel <= 1.0 + 1e-9


def test_arm_feasibility_flags_r1_without_qualifier():
    """A1.3: R1 with no qualifying CAV is infeasible (excluded from the set)."""
    empty = S.make_schedule("R1", [cav("cav.0", 900.0)], 40.0)
    feasible, reason = S.arm_feasibility("R1", empty)
    assert feasible is False and reason == "r1_no_qualifier"
    qualified = S.make_schedule("R1", [cav("cav.0", 20.0)], 40.0)
    assert S.arm_feasibility("R1", qualified) == (True, None)
    assert S.arm_feasibility("R0", S.make_schedule("R0", [], 40.0)) == (
        True, None)
