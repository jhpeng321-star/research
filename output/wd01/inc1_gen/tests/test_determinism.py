"""Determinism tests: identical inputs -> bit-identical plans (zero SUMO)."""
from __future__ import annotations

import json

from inc1_gen.generators import R5Harmonisation, R6GapOpening

from inc1_gen.tests._helpers import (DT_S, OBSTACLE, T0_S, advance, as_states, base_world,
                      with_leaders)


def transcript(gen, world):
    """Run the registered window and capture everything the generator emits."""
    rows = []
    for k in range(0, int(gen.duration_s) + 5):
        step = gen.step(T0_S + float(k) * DT_S, as_states(world))
        rows.append({
            "t_s": step.t_s,
            "commands": sorted(step.commands.items()),
            "native": list(step.native),
            "feasible": step.feasible,
            "infeasible_reason": step.infeasible_reason,
        })
        world = advance(world, step.commands, dt_s=DT_S)
        world = with_leaders(world)
    return rows, json.dumps(gen.events, sort_keys=True)


def run_r5(world):
    gen = R5Harmonisation(as_states(world), OBSTACLE, t0_s=T0_S,
                          control_dt_s=DT_S)
    return transcript(gen, [dict(r) for r in world])


def run_r6(world):
    gen = R6GapOpening(as_states(world), OBSTACLE, t0_s=T0_S,
                       control_dt_s=DT_S)
    return transcript(gen, [dict(r) for r in world])


def test_same_sequence_same_commands_r5():
    world = with_leaders(base_world())
    assert run_r5(world) == run_r5(world)


def test_same_sequence_same_commands_r6():
    world = with_leaders(base_world())
    assert run_r6(world) == run_r6(world)


def test_instances_do_not_interfere():
    """Two live generators stepped in alternation match their solo runs."""
    world = with_leaders(base_world())
    solo_r5, _ = run_r5(world)
    solo_r6, _ = run_r6(world)
    gen_a = R5Harmonisation(as_states(world), OBSTACLE, t0_s=T0_S,
                            control_dt_s=DT_S)
    gen_b = R6GapOpening(as_states(world), OBSTACLE, t0_s=T0_S,
                         control_dt_s=DT_S)
    world_a, world_b = [dict(r) for r in world], [dict(r) for r in world]
    rows_a, rows_b = [], []
    for k in range(max(len(solo_r5), len(solo_r6))):
        step_a = gen_a.step(T0_S + float(k) * DT_S, as_states(world_a))
        step_b = gen_b.step(T0_S + float(k) * DT_S, as_states(world_b))
        rows_a.append({
            "t_s": step_a.t_s,
            "commands": sorted(step_a.commands.items()),
            "native": list(step_a.native),
            "feasible": step_a.feasible,
            "infeasible_reason": step_a.infeasible_reason,
        })
        rows_b.append({
            "t_s": step_b.t_s,
            "commands": sorted(step_b.commands.items()),
            "native": list(step_b.native),
            "feasible": step_b.feasible,
            "infeasible_reason": step_b.infeasible_reason,
        })
        world_a = with_leaders(advance(world_a, step_a.commands, dt_s=DT_S))
        world_b = with_leaders(advance(world_b, step_b.commands, dt_s=DT_S))
    assert rows_a[:len(solo_r5)] == solo_r5
    assert rows_b[:len(solo_r6)] == solo_r6


def test_one_different_input_bit_changes_the_plan():
    world = with_leaders(base_world())
    perturbed = [dict(r) if r["vehicle_id"] != "cav.a"
                 else dict(r, v=r["v"] + 0.125) for r in world]
    solo, _ = run_r5(world)
    other, _ = run_r5(perturbed)
    assert solo != other
