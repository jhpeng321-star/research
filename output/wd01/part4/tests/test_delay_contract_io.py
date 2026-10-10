"""Contract trajectory-layout adapter: synthetic equivalence + errors."""
from __future__ import annotations

import numpy as np
import pytest

from delay import core
from delay.contract_io import (delay_from_contract_layout,
                               traces_from_contract)
from conftest import trace


def _contract_arrays(t0=0.0, dt=1.0, T=11):
    """Two vehicles on the synthetic section in contract layout.

    a: present at t0 at s=0 on lane 0, 10 m/s (free-flow), exits at t=10
       (s=100 = end of the pre-merge segment -> still on section here);
    b: inserted at t=4 on lane 1 at s=0, 7.5 m/s (queued speed),
       still on section at the horizon.
    NaN suffix for a's rows after t=10; NaN prefix for b before t=4.
    """
    t = t0 + dt * np.arange(T)
    a = np.full((T, 4), np.nan)
    a[:, 0] = t
    a[:10, 1] = 10.0 * np.arange(10)      # s at t0..t0+9
    a[:10, 2] = 0
    b = np.full((T, 4), np.nan)
    b[:, 0] = t
    b[4:, 1] = 7.5 * np.arange(0, T - 4)
    b[4:, 2] = 1
    schedule = {"a": (-5.0, "mainline"),   # entered before the window
                "b": (2.0, "ramp"),        # waited 2 s inside the window
                "c": (6.0, "ramp")}        # scheduled, still queued at H
    arrivals = {"t_s": np.array([4.0]), "entrance": np.array(["ramp"])}
    return ({"a": a}, {"b": b}, arrivals, schedule)


def test_contract_layout_matches_direct_traces(synth_geom):
    tr_a, tr_b, arrivals, schedule = _contract_arrays()
    out = delay_from_contract_layout(
        0.0, 10.0, synth_geom, trajectories=tr_a,
        arrived_trajectories=tr_b, arrivals=arrivals, schedule=schedule,
        dt=1.0)
    tot = out.totals()

    # the same scenario as direct core traces at dt = 1
    t = np.arange(11.0)
    direct = core.WindowDelay(
        t0=0.0, horizon_s=10.0,
        vehicles=[
            core.vehicle_delay(trace("a", t[:10], 0, 10.0 * t[:10]),
                               0.0, 10.0, synth_geom, dt=1.0),
            core.vehicle_delay(trace("b", t[4:], 1, 7.5 * (t[4:] - 4.0)),
                               0.0, 10.0, synth_geom, dt=1.0),
        ],
        waits=core.waiting_component(
            {v: (td, 1 if ent == "ramp" else 0) for v, (td, ent) in
             (("a", (-5.0, "mainline")), ("b", (2.0, "ramp")),
              ("c", (6.0, "ramp")))},
            {"a": 0.0, "b": 4.0}, 0.0, 10.0)[0],
        diagnostics={})
    d = direct.totals()
    for k in ("time_on_section_s", "free_flow_ref_s", "traversed_m",
              "wait_mainline_s", "wait_ramp_s", "total_delay_ref_s",
              "n_vehicles"):
        assert tot[k] == pytest.approx(d[k]), k
    # vehicle a: on-section 10 s, free-flow 10 s (100 m @ 10) -> no delay
    # vehicle b: on-section 6 s, free-flow 45/10 = 4.5 -> travel 1.5 s
    # waiting: b 2 s ramp; c 4 s ramp queued at horizon
    assert tot["total_delay_ref_s"] == pytest.approx(1.5 + 2.0 + 4.0)
    assert tot["n_queued_at_horizon"] == 1
    assert tot["n_unmatched_sched_id"] == 0


def test_traces_from_contract_nan_prefix_and_suffix():
    tr_a, tr_b, _, _ = _contract_arrays()
    t0_t, t0_w, ins = traces_from_contract(tr_a, tr_b, 0.0)
    assert [v.vid for v in t0_t] == ["a"]
    assert [v.vid for v in t0_w] == ["b"]
    assert ins == {"b": 4.0}
    assert len(t0_t[0].t) == 10 and len(t0_w[0].t) == 7


def test_non_trailing_nan_rejected(synth_geom):
    a = np.full((6, 4), np.nan)
    a[:, 0] = np.arange(6.0)
    a[0, 1:] = [0.0, 0, 0]
    a[2:4, 1:] = [5.0, 0, 0]               # real rows AFTER a NaN row
    with pytest.raises(ValueError, match="trailing"):
        traces_from_contract({"a": a}, None, 0.0)


def test_tracked_must_start_at_t0(synth_geom):
    a = np.full((4, 4), np.nan)
    a[:, 0] = np.arange(1.0, 5.0)          # starts at t0+1
    a[:, 1:] = [1.0, 0, 0]
    with pytest.raises(ValueError, match="start at t0"):
        traces_from_contract({"a": a}, None, 0.0)


def test_bad_entrance_label_rejected(synth_geom):
    tr_a, tr_b, _, _ = _contract_arrays()
    with pytest.raises(ValueError, match="entrance"):
        delay_from_contract_layout(0.0, 10.0, synth_geom,
                                   trajectories=tr_a,
                                   arrived_trajectories=tr_b,
                                   schedule={"b": (2.0, "left")},
                                   dt=1.0)


def test_speed_limit_attribution_in_contract_layout(synth_geom):
    # a vehicle moving through lane1's E2 segment (limit 20) must get
    # that limit in the free-flow time: 30 m inside E2 -> 1.5 s
    T = 6
    c = np.full((T, 4), np.nan)
    c[:, 0] = np.arange(T, dtype=float)
    c[:, 1] = 100.0 + 10.0 * np.arange(T)      # lane0 E2 [100,140]
    c[:, 2] = 0
    d = c.copy()
    d[:, 1] = 100.0 + 10.0 * np.arange(T)      # lane1 E2 [90,130]
    d[:, 2] = 1
    out = delay_from_contract_layout(0.0, 5.0, synth_geom,
                                     trajectories={"c": c}, dt=1.0)
    tot = out.totals()
    assert tot["traversed_m"] == pytest.approx(50.0)
    assert tot["free_flow_ref_s"] == pytest.approx(2.5)   # all inside E2
