"""The A2.6 two-call-site identity harness test (zero SUMO, synthetic).

The SAME synthetic state sequence is rendered twice -- traci-like dicts
(SUMO-side shim) and tensor-like arrays (WM-side shim) -- and the shared R5/R6
generators must emit bit-identical commands at every tick through either
call site.  A later wiring task reuses ``run_two_site_identity`` on its real
call-site renderings before the gated launch.
"""
from __future__ import annotations

import numpy as np
import pytest

from inc1_gen.adapters import (run_two_site_identity, states_equal,
                               states_from_traci_like, states_from_wm_arrays)

from inc1_gen.tests._helpers import (DT_S, OBSTACLE, T0_S, advance, base_world, to_sumo_rows,
                      to_wm_frame, with_leaders)

HORIZON_TICKS = 200


def render_sequence(world):
    """Evolve the synthetic truth and render every tick for BOTH call sites."""
    frames_sumo, frames_wm = [], []
    for k in range(HORIZON_TICKS):
        world = with_leaders(world)
        now = T0_S + float(k) * DT_S
        frames_sumo.append((now, to_sumo_rows(world)))
        ids, kwargs = to_wm_frame(world)
        frames_wm.append((now, (ids, kwargs)))
        # closed-loop evolution on the shared truth: the WM-side frames are
        # rendered from the SAME rows, so any identity break is an adapter or
        # generator defect, not a scenario drift
        dummy_gen_cmds = {}
        world = advance(world, dummy_gen_cmds, dt_s=DT_S)
    return frames_sumo, frames_wm


@pytest.mark.parametrize("arm", ["R5", "R6"])
def test_sumo_side_and_wm_side_commands_are_identical(arm):
    world = base_world()
    t0_sumo = to_sumo_rows(world)
    t0_wm = to_wm_frame(world)
    frames_sumo, frames_wm = render_sequence(world)
    report = run_two_site_identity(arm, t0_sumo, frames_sumo, t0_wm,
                                   frames_wm, obstacle=OBSTACLE, t0_s=T0_S,
                                   control_dt_s=DT_S)
    assert report["identical"], report["mismatches"]
    assert report["ticks"] == HORIZON_TICKS
    assert report["n_mismatches"] == 0


def test_shims_render_equal_states():
    world = with_leaders(base_world())
    sumo_states = states_from_traci_like(to_sumo_rows(world))
    ids, kwargs = to_wm_frame(world)
    wm_states = states_from_wm_arrays(ids, **kwargs)
    assert states_equal(sumo_states, wm_states)
    assert [s.vehicle_id for s in sumo_states] == sorted(
        r["vehicle_id"] for r in world)


def test_harness_detects_a_perturbed_wm_rendering():
    world = base_world()
    t0_sumo = to_sumo_rows(world)
    frames_sumo, frames_wm = render_sequence(world)
    ids, kwargs = to_wm_frame(world)
    tampered = dict(kwargs)
    tampered["v"] = np.asarray(tampered["v"], dtype=float)
    tampered["v"][0] += 0.5
    report = run_two_site_identity(
        "R5", t0_sumo, frames_sumo, (ids, tampered), frames_wm,
        obstacle=OBSTACLE, t0_s=T0_S, control_dt_s=DT_S)
    assert not report["identical"]
    assert report["n_mismatches"] >= 1


def test_wm_shim_accepts_nan_sentinels_and_torch_like_tensors():
    # "tensor-like": anything numpy.asarray accepts; NaN = none/unknown
    class FakeTensor:
        def __init__(self, values):
            self._values = np.asarray(values, dtype=float)

        def __array__(self, dtype=None):
            return self._values.astype(dtype or self._values.dtype)

    ids = ["cav.a", "hdv.1"]
    states = states_from_wm_arrays(
        ids,
        s_down=FakeTensor([-60.0, -160.0]),
        v=FakeTensor([20.0, 21.0]),
        a=FakeTensor([0.0, -0.5]),
        lane_phys=FakeTensor([0.0, 1.0]),
        is_cav=FakeTensor([1.0, 0.0]),
        present=FakeTensor([1.0, 1.0]),
        native_accel=FakeTensor([float("nan"), 1.2]),
        leader_gap=FakeTensor([25.0, float("nan")]),
        leader_v=FakeTensor([19.0, float("nan")]),
    )
    assert states[0].native_accel is None
    assert states[0].leader_gap == 25.0
    assert states[1].leader_gap is None and states[1].leader_v is None
    assert states[1].is_cav is False


def test_sumo_shim_flags_absent_vehicles_and_cav_types():
    rows = to_sumo_rows(base_world())
    rows["cav.a"]["present"] = False
    states = {s.vehicle_id: s for s in states_from_traci_like(rows)}
    assert states["cav.a"].present is False
    assert states["cav.a"].is_cav is True
    assert states["hdv.1"].is_cav is False
