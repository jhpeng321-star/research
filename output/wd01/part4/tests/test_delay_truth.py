"""Real-data tests on the partA_data native roots (read-only, no SIM).

Skipped (with reason) if the authoritative partA_data tree is absent.
"""
from __future__ import annotations

import numpy as np
import pytest

import delay
from delay.contract_io import contract_view_of_truth, \
    delay_from_contract_layout
from delay.geometry import wm30_geometry
from delay.truth import load_root_data, load_schedule, root_run_dir, \
    vtype_speed_factors


@pytest.fixture(scope="module")
def geom(real_data_available):
    if not real_data_available:
        pytest.skip("authoritative partA_data not available")
    return wm30_geometry()


@pytest.fixture(scope="module")
def rd(real_data_available):
    if not real_data_available:
        pytest.skip("authoritative partA_data not available")
    return load_root_data("cv17")


def test_schedule_and_speed_factors(rd):
    sched = load_schedule(root_run_dir(rd.root))
    assert len(sched) >= len(rd.vid_table)      # every logged id scheduled
    assert set(str(v) for v in rd.vid_table) <= set(sched)
    sf = vtype_speed_factors(root_run_dir(rd.root))
    assert sf == {"aggressive_car": 1.08, "normal_car": 1.0,
                  "cautious_car": 0.92}


def test_window_consistency(rd, geom):
    wd = delay.window_truth(rd, 120.0, 60.0, geom)
    t = wd.totals()
    # component-sum identities (contract §8.8 analogue, at fp tolerance)
    assert t["total_delay_ref_s"] == pytest.approx(
        t["time_on_section_s"] - t["free_flow_ref_s"]
        + t["wait_mainline_s"] + t["wait_ramp_s"], abs=1e-9)
    assert t["total_delay_oracle_s"] == pytest.approx(
        t["time_on_section_s"] - t["free_flow_oracle_s"]
        + t["wait_mainline_s"] + t["wait_ramp_s"], abs=1e-9)
    # entry waiting >= 0, waiting == insertion - schedule by id
    assert all(w.wait_s >= 0.0 for w in wd.waits)
    t1 = wd.t0 + wd.horizon_s
    for w in wd.waits:
        if w.matched and np.isfinite(w.t_ins):
            expect = np.clip(min(w.t_ins, t1) - max(w.t_dep, wd.t0),
                             0.0, wd.horizon_s)
            assert w.wait_s == pytest.approx(float(expect), abs=1e-9)
    # no schedule id without a matching logged vehicle (insertees are a
    # subset of the schedule; queued vehicles are schedule rows)
    assert t["n_unmatched_sched_id"] == 0
    # physically bounded on-section time (vehicles may exceed the speed
    # limit, so free-flow time need not be below the actual time)
    assert t["time_on_section_s"] <= (wd.horizon_s + rd.dt) * t["n_vehicles"]


def test_window_grid_all_horizons(rd, geom):
    for t0, h in ((120.0, 60.0), (120.0, 120.0), (300.0, 300.0),
                  (120.0, 480.0)):
        t = delay.window_truth(rd, t0, h, geom).totals()
        assert t["n_vehicles"] > 0
        assert t["total_delay_ref_s"] >= 0.0
        assert t["n_unmatched_sched_id"] == 0


def test_contract_adapter_cross_check(rd, geom):
    """Truth window re-expressed in the contract layout must give the same
    delay up to the coarser 1 s sampling (<= 1 s boundary effect per
    vehicle per component group)."""
    t0, h = 120.0, 60.0
    truth = delay.window_truth(rd, t0, h, geom).totals()
    tracks, arrived, arrivals, schedule, entered = \
        contract_view_of_truth(rd, t0, h)
    out = delay_from_contract_layout(
        t0, h, geom, trajectories=tracks, arrived_trajectories=arrived,
        arrivals=arrivals, schedule=schedule, entered_before_t0=entered,
        dt=1.0)
    tot = out.totals()
    # ids: tracked + arrived = every vehicle seen in the window
    assert tot["n_vehicles"] == truth["n_vehicles"]
    for k in ("time_on_section_s", "travel_delay_ref_s", "total_delay_ref_s",
              "wait_mainline_s", "wait_ramp_s"):
        slack = 1.0 * truth["n_vehicles"] if "wait" not in k else 1.0
        assert tot[k] == pytest.approx(truth[k], abs=slack), \
            f"{k}: contract {tot[k]:.2f} vs truth {truth[k]:.2f}"
