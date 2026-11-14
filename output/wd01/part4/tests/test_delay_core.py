"""Hand-computed synthetic cases for the §4.1 delay accounting."""
from __future__ import annotations

import numpy as np
import pytest

from delay import core
from conftest import trace

T0 = 0.0


# -------------------------------------------------- per-vehicle accounting ----

def test_present_at_t0_freeflow_exits_at_horizon(synth_geom):
    # 10 m/s on lane 0, section [0,140]: records every 2 s, reaches the
    # downstream end exactly at the horizon t=14.
    tr = trace("a", np.arange(0, 15, 2.0), 0, np.arange(0, 15, 2.0) * 10.0)
    v = core.vehicle_delay(tr, T0, 14.0, synth_geom, dt=2.0)
    assert v.time_on_section_s == pytest.approx(14.0)
    # free-flow time = 100 m @10 + 40 m @20 = 12 s -> delay 2 s
    assert v.free_flow_ref_s == pytest.approx(12.0)
    assert v.traversed_m == pytest.approx(140.0)
    assert v.present_at_t0 and not v.inserted_in_window


def test_present_at_t0_still_on_section_at_horizon(synth_geom):
    # enters the window at s=20, 8 m/s, never leaves: on-section = H = 10,
    # traversed 80 m, free-flow 8 s -> delay 2 s
    t = np.arange(0, 11, 2.0)
    tr = trace("b", t, 0, 20.0 + 8.0 * t)
    v = core.vehicle_delay(tr, T0, 10.0, synth_geom, dt=2.0)
    assert v.time_on_section_s == pytest.approx(10.0)
    assert v.traversed_m == pytest.approx(80.0)
    assert v.free_flow_ref_s == pytest.approx(8.0)


def test_born_mid_window_at_freeflow_zero_delay(synth_geom):
    # inserted at t=4 at s=0, free-flow speed: on-section = 10-4 = 6 s,
    # free-flow = 60 m @10 = 6 s -> delay exactly 0 (no boundary error)
    t = np.arange(4, 11, 2.0)
    tr = trace("c", t, 0, (t - 4.0) * 10.0)
    v = core.vehicle_delay(tr, T0, 10.0, synth_geom, dt=2.0)
    assert v.inserted_in_window
    assert v.time_on_section_s == pytest.approx(6.0)
    assert v.free_flow_ref_s == pytest.approx(6.0)
    assert v.free_flow_ref_s - v.time_on_section_s == pytest.approx(0.0)


def test_exits_mid_window(synth_geom):
    # present at t0 at s=0, free-flow, leaves at t=10; horizon 20: the
    # window sees 10 s on section and 10 s free-flow -> delay 0
    t = np.arange(0, 11, 2.0)
    tr = trace("d", t, 0, t * 10.0)
    v = core.vehicle_delay(tr, T0, 20.0, synth_geom, dt=2.0)
    assert v.time_on_section_s == pytest.approx(10.0)
    assert v.traversed_m == pytest.approx(100.0)
    assert v.free_flow_ref_s == pytest.approx(10.0)


def test_queued_speed_on_mainline(synth_geom):
    # 5 m/s on lane 0 for the whole 10 s window: delay = 10 - 50/10 = 5 s
    t = np.arange(0, 11, 2.0)
    tr = trace("e", t, 0, t * 5.0)
    v = core.vehicle_delay(tr, T0, 10.0, synth_geom, dt=2.0)
    assert v.time_on_section_s - v.free_flow_ref_s == pytest.approx(5.0)


def test_accel_lane_has_its_own_speed_limit(synth_geom):
    # accel lane limit 5 m/s: reaches s=40 at t=8 then holds (blocked at
    # the merge end): free-flow 8 s of a 16 s stay -> delay 8 s
    t = np.arange(0, 17, 2.0)
    tr = trace("f", t, 2, np.minimum(t * 5.0, 40.0))
    v = core.vehicle_delay(tr, T0, 16.0, synth_geom, dt=2.0)
    assert v.free_flow_ref_s == pytest.approx(8.0)
    assert v.traversed_m == pytest.approx(40.0)
    assert v.time_on_section_s == pytest.approx(16.0)


def test_lane_change_within_edge_uses_within_edge_positions(synth_geom):
    # lane0 s=102 (E2 pos 2) -> lane1 s=127 (E2 pos 37): advance is the
    # within-edge difference 35 m, NOT the raw ds of 25 m (the chains'
    # E2 segments are offset by 10 m on this synthetic section)
    tr = trace("g", [2.0, 4.0], [0, 1], [102.0, 127.0])
    v = core.vehicle_delay(tr, T0, 10.0, synth_geom, dt=2.0)
    assert v.traversed_m == pytest.approx(35.0)
    assert v.free_flow_ref_s == pytest.approx(35.0 / 20.0)


def test_junction_steps_bridge_with_zero_length(synth_geom):
    # lane0 s=98 -> junction (s=100, twice) -> lane0 s=110: advances
    # 2 + 0 + 0; free-flow 2/10 + 0 + 10/20 = 0.7 s
    tr = trace("h", [2.0, 4.0, 6.0, 8.0], [0, -1, -1, 0],
               [98.0, 100.0, 100.0, 110.0])
    v = core.vehicle_delay(tr, T0, 10.0, synth_geom, dt=2.0)
    assert v.traversed_m == pytest.approx(12.0)
    assert v.free_flow_ref_s == pytest.approx(0.7)
    assert v.n_junction_records == 2


def test_boundary_change_step_uses_chain_continuity(synth_geom):
    # lane0 s=98 (2 m before the E1 end) -> lane1 s=92 (2 m into its E2):
    # advance = (100-98) + (92-90) = 4, not raw ds = -6
    tr = trace("i", [2.0, 4.0], [0, 1], [98.0, 92.0])
    v = core.vehicle_delay(tr, T0, 10.0, synth_geom, dt=2.0)
    assert v.traversed_m == pytest.approx(4.0)
    assert v.n_boundary_change_steps == 1


def test_records_beyond_horizon_are_truncated(synth_geom):
    t = np.arange(0, 21, 2.0)          # continues to t=20
    tr = trace("j", t, 0, t * 10.0)
    v = core.vehicle_delay(tr, T0, 10.0, synth_geom, dt=2.0)
    assert v.time_on_section_s == pytest.approx(10.0)
    assert v.traversed_m == pytest.approx(100.0)


def test_oracle_speed_factor_variants(synth_geom):
    t = np.arange(0, 11, 2.0)
    tr = trace("k", t, 0, t * 5.0, vtype=0)
    v = core.vehicle_delay(tr, T0, 10.0, synth_geom, dt=2.0,
                           speed_factors={0: 0.5})
    assert v.free_flow_ref_s == pytest.approx(5.0)     # limit-based
    assert v.free_flow_oracle_s == pytest.approx(10.0)  # 0.5 x limit


def test_gapped_records_flagged(synth_geom):
    tr = trace("l", [0.0, 4.0, 6.0], 0, [0.0, 40.0, 60.0])  # missing t=2
    v = core.vehicle_delay(tr, T0, 10.0, synth_geom, dt=2.0)
    assert v.gapped


# ---------------------------------------------------------------- waiting ----

def test_waiting_insertion_after_scheduled():
    sched = {"m1": (2.0, 0)}                      # mainline, t_dep = 2
    ins = {"m1": 8.0}
    waits, diag = core.waiting_component(sched, ins, T0, 10.0)
    assert len(waits) == 1 and waits[0].wait_s == pytest.approx(6.0)
    assert waits[0].entrance == 0 and not waits[0].queued_at_horizon
    assert diag["n_unmatched_sched_id"] == 0


def test_waiting_queued_across_t0():
    sched = {"m2": (-5.0, 0)}
    ins = {"m2": 3.0}
    waits, _ = core.waiting_component(sched, ins, T0, 10.0)
    assert waits[0].wait_s == pytest.approx(3.0)   # truncated at t0


def test_waiting_queued_past_horizon():
    sched = {"m3": (2.0, 1)}                       # ramp
    ins = {}
    waits, _ = core.waiting_component(sched, ins, T0, 10.0)
    assert waits[0].wait_s == pytest.approx(8.0)
    assert waits[0].queued_at_horizon
    assert waits[0].entrance == 1


def test_waiting_already_entered_before_t0():
    sched = {"m4": (-5.0, 0)}
    ins = {"m4": float("-inf")}                    # entered long before
    waits, _ = core.waiting_component(sched, ins, T0, 10.0)
    assert waits[0].wait_s == 0.0 and not waits[0].queued_at_horizon


def test_waiting_unmatched_insertee_counted():
    sched = {}
    ins = {"m5": 4.0}
    waits, diag = core.waiting_component(sched, ins, T0, 10.0)
    assert diag["n_unmatched_sched_id"] == 1
    assert all(w.wait_s == 0.0 for w in waits)


def test_waiting_schedule_beyond_horizon_ignored():
    sched = {"m6": (11.0, 0)}
    waits, _ = core.waiting_component(sched, {}, T0, 10.0)
    assert len(waits) == 0


# ----------------------------------------------------------------- totals ----

def test_window_totals_identity(synth_geom):
    t = np.arange(0, 11, 2.0)
    veh = [
        core.vehicle_delay(trace("v1", t, 0, t * 5.0), T0, 10.0,
                           synth_geom, dt=2.0, speed_factors={0: 0.5}),
        core.vehicle_delay(trace("v2", t, 1, 90 + (t - 4.0) * 5.0), T0, 10.0,
                           synth_geom, dt=2.0),
    ]
    waits, diag = core.waiting_component(
        {"v3": (2.0, 0), "v4": (2.0, 1)}, {"v3": 5.0}, T0, 10.0)
    tot = core.window_totals(veh, waits, T0, 10.0, diag)
    assert tot["n_vehicles"] == 2
    assert tot["travel_delay_ref_s"] == pytest.approx(
        tot["time_on_section_s"] - tot["free_flow_ref_s"])
    assert tot["total_delay_ref_s"] == pytest.approx(
        tot["travel_delay_ref_s"] + tot["wait_mainline_s"]
        + tot["wait_ramp_s"])
    assert tot["total_delay_oracle_s"] == pytest.approx(
        tot["time_on_section_s"] - tot["free_flow_oracle_s"]
        + tot["wait_mainline_s"] + tot["wait_ramp_s"])
    # v3: inserted at 5, scheduled 2 -> 3 s mainline wait; v4: 8 s ramp
    assert tot["wait_mainline_s"] == pytest.approx(3.0)
    assert tot["wait_ramp_s"] == pytest.approx(8.0)
    assert tot["n_waiting_only"] == 2
    assert tot["n_queued_at_horizon"] == 1
