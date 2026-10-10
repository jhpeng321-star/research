"""Seeded property test: the common limits are never exceeded (A2.1).

Random synthetic states drive both generators closed-loop for many ticks;
every issued command must lie in ``[-1.5, max(native, -1.5)]`` m/s^2, only
latched members may be commanded, and the commands+native sets must partition
the live CAVs exactly.  Zero SUMO; numpy RNG seeded (20261001).
"""
from __future__ import annotations

import numpy as np
import pytest

from inc1_gen.generators import R5Harmonisation, R6GapOpening

from inc1_gen.tests._helpers import (DT_S, OBSTACLE, T0_S, advance, as_states,
                      drop_world_row, random_world, with_leaders)

SEED = 20261001
TICKS = 40
WORLDS = 8
TOL = 1e-12


def _check_limits(gen, world, step, allowed):
    live_cavs = {r["vehicle_id"] for r in world
                 if r["present"] and r["is_cav"]}
    assert set(step.commands) <= allowed
    assert set(step.commands) | set(step.native) == live_cavs
    assert not set(step.commands) & set(step.native)
    by_id = {r["vehicle_id"]: r for r in world}
    for vid, cmd in step.commands.items():
        native = by_id[vid]["native_accel"]
        lo, hi = -1.5, (0.0 if native is None else max(native, -1.5))
        assert lo - TOL <= cmd <= hi + TOL, (
            f"{gen.name} {vid}: cmd {cmd} outside [{lo}, {hi}]")
        assert cmd == cmd  # never NaN


@pytest.mark.parametrize("cls", [R5Harmonisation, R6GapOpening])
def test_limits_hold_on_random_synthetic_states(cls):
    rng = np.random.default_rng(SEED)
    for w in range(WORLDS):
        world = random_world(rng)
        gen = cls(as_states(world), OBSTACLE, t0_s=T0_S, control_dt_s=DT_S)
        allowed = set(gen._members)
        for k in range(TICKS):
            world = with_leaders(world)
            step = gen.step(T0_S + k * DT_S, as_states(world))
            _check_limits(gen, world, step, allowed)
            world = advance(world, step.commands, dt_s=DT_S)
            if k in (10, 25):  # members occasionally leave the network
                for vid in sorted(allowed)[:1]:
                    world = drop_world_row(world, vid)


def test_limits_hold_with_degenerate_native_bounds():
    # native far below the decel floor: command must still respect -1.5
    # and the degenerate clip interval (never below -1.5)
    rows = [dict(vehicle_id="c1", is_cav=True, lane_phys=0,
                 s_down=OBSTACLE.s_down - 200.0, v=25.0, a=0.0,
                 leader_gap=8.0, leader_v=0.0, native_accel=-3.5,
                 present=True),
            dict(vehicle_id="c2", is_cav=True, lane_phys=0,
                 s_down=OBSTACLE.s_down - 150.0, v=25.0, a=0.0,
                 leader_gap=None, leader_v=None, native_accel=None,
                 present=True)]
    r5 = R5Harmonisation(as_states(rows), OBSTACLE, t0_s=T0_S)
    assert r5.feasible
    step = r5.step(T0_S + 1.0, as_states(rows))
    # R5 target ~10.4 for both -> strong decel request, clipped to the floor;
    # unknown native never becomes a push (floor -1.5 applies instead)
    assert step.commands["c1"] == -1.5
    assert step.commands["c2"] == -1.5
    r6 = R6GapOpening(as_states(rows), OBSTACLE, t0_s=T0_S)
    assert r6.feasible
    step = r6.step(T0_S + 1.0, as_states(rows))
    # R6: c1 headway term a_hc = 0.5*(8/3 - 25) ~ -11.9 -> min(-3.5, a_hc)
    # then floor -> -1.5; c2 has no leader and unknown native -> 0.0
    assert step.commands["c1"] == -1.5
    assert step.commands["c2"] == 0.0


def test_hdvs_and_absent_vehicles_never_appear_in_any_plan():
    rng = np.random.default_rng(SEED + 1)
    world = random_world(rng, n=10)
    for cls in (R5Harmonisation, R6GapOpening):
        gen = cls(as_states(world), OBSTACLE, t0_s=T0_S)
        step = gen.step(T0_S + 1.0, as_states(world))
        all_named = set(step.commands) | set(step.native)
        hdvs = {r["vehicle_id"] for r in world
                if r["is_cav"] is False and r["present"]}
        absent = {r["vehicle_id"] for r in world if not r["present"]}
        assert not hdvs & all_named
        assert not absent & all_named
