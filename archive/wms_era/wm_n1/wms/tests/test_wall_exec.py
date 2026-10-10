from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from output.wd01.wm_n1.wms.wall_exec import (
    DIRECT, DAMPING, NATIVE, PHASE_ASSEMBLY, PHASE_HANDBACK, PHASE_HOLD,
    NumpyWallExecutor, TorchWallExecutor,
)


ROOT = Path(__file__).resolve().parents[5]
# The wall-session source is a staged (gitignored) copy under _in/; an
# integrated checkout may point WMS_WALL_SOURCE_DIR at an equivalent copy.
SOURCE = Path(os.environ.get("WMS_WALL_SOURCE_DIR")
              or ROOT / "_in" / "traffic_shepherd") / "wall_group.py"
HAS_SOURCE = SOURCE.is_file() and SOURCE.with_name("wall_group_v22.py").is_file()
wall_group = wall_group_v22 = None
if HAS_SOURCE:
    spec = importlib.util.spec_from_file_location("wall_group_fixture", SOURCE)
    wall_group = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(wall_group)
    traffic_shepherd_stub = SimpleNamespace(wall_group=wall_group)
    sys.modules.setdefault("traffic_shepherd", traffic_shepherd_stub)
    sys.modules.setdefault("traffic_shepherd.wall_group", wall_group)
    spec_v22 = importlib.util.spec_from_file_location(
        "wall_group_v22_fixture", SOURCE.with_name("wall_group_v22.py"))
    wall_group_v22 = importlib.util.module_from_spec(spec_v22)
    assert spec_v22.loader is not None
    spec_v22.loader.exec_module(wall_group_v22)


def _frames(position, speed, lanes=(0, 1), extra=None):
    # Source uses s=-distance_to_merge_m; position here is downstream-increasing.
    frames = {
        "c0": SimpleNamespace(vehicle_id="c0", distance_to_merge_m=-float(position[0]),
                               speed_mps=float(speed[0]), lane_index=lanes[0],
                               edge_id=wall_group_v22.ENTRY_EDGE_ID,
                               position_m=float(position[0])),
        "c1": SimpleNamespace(vehicle_id="c1", distance_to_merge_m=-float(position[1]),
                               speed_mps=float(speed[1]), lane_index=lanes[1],
                               edge_id=wall_group_v22.ENTRY_EDGE_ID,
                               position_m=float(position[1])),
    }
    if extra:
        frames.update(extra)
    return frames


def _step(executor, p, v, t, **kwargs):
    return executor.step(np.asarray([p]), np.asarray([v]), now_s=t,
                         lane_index=np.asarray([[0, 1]]), **kwargs)


@pytest.mark.parametrize("scenario", ["assemble_hold_release", "planner_abort", "safety_abort"])
@pytest.mark.skipif(not HAS_SOURCE, reason="staged wall-session reference source is absent")
def test_torch_numpy_and_simulator_parity(scenario):
    members = np.asarray([[True, True]])
    prior = np.asarray([[DAMPING, NATIVE]])
    torch_exec = TorchWallExecutor(members, [15.0], prior_controller=prior)
    numpy_exec = NumpyWallExecutor(members, [15.0], prior_controller=prior)
    sim = wall_group.WallGroupSession("w", members=frozenset(("c0", "c1")),
                                      wall_speed_mps=15.0)

    def tick(t, p, v, abort=False, release=False, extras=None):
        common = {"abort": np.asarray([abort]), "release": np.asarray([release])}
        tn = _step(torch_exec, p, v, t, **common)
        nn = _step(numpy_exec, p, v, t, **common)
        torch.testing.assert_close(tn["acceleration_mps2"], torch.as_tensor(nn["acceleration_mps2"]), atol=1e-12, rtol=0)
        torch.testing.assert_close(tn["target_speed_mps"], torch.as_tensor(nn["target_speed_mps"]), atol=1e-12, rtol=0)
        np.testing.assert_array_equal(tn["phase"].numpy(), nn["phase"])
        if abort and sim.phase != wall_group.PHASE_HANDBACK:
            sim.abort("planner_abort", t)
        if release and sim.phase == wall_group.PHASE_HOLD:
            sim.release(t)
        commands = sim.tick(_frames(p, v, extra=extras), t)
        assert tn["phase_names"][0] == sim.phase
        if sim.phase != wall_group.PHASE_HANDBACK:
            for vehicle, acceleration in commands.items():
                slot = int(vehicle[-1])
                assert abs(float(tn["acceleration_mps2"][0, slot]) - acceleration) <= 1e-6
                assert int(tn["owner"][0, slot]) == DIRECT
        else:
            assert commands == {}
            assert tn["owner"][0].tolist() == prior[0].tolist()
        return tn, commands

    if scenario == "assemble_hold_release":
        first, _ = tick(0, [0.0, -30.0], [15.0, 15.0])
        assert int(first["phase"][0]) == PHASE_ASSEMBLY
        # Move the rear close enough; simulator still emits assembly pacing on
        # the completion tick before entering HOLD.
        tick(1, [0.0, -5.0], [15.0, 15.0])
        assert int(torch_exec.phase[0]) == PHASE_HOLD
        tick(2, [0.0, -5.0], [14.0, 16.0])
        result, _ = tick(3, [0.0, -5.0], [15.0, 15.0], release=True)
        assert result["phase_names"][0] == "handback"
    elif scenario == "planner_abort":
        result, commands = tick(0, [0.0, -30.0], [15.0, 15.0], abort=True)
        assert commands == {} and result["phase_names"][0] == "handback"
        assert result["abort_reason"] == ("planner_abort",)
    else:
        result, commands = tick(0, [50.0, 201.0], [15.0, 15.0])
        assert commands == {} and result["phase_names"][0] == "handback"


def test_member_commands_are_direct_timeline_targets_and_handback_restores_prior_owner():
    executor = TorchWallExecutor([[True, True, False]], [10.0],
                                 prior_controller=[[DAMPING, NATIVE, NATIVE]])
    result = executor.step([[20.0, 0.0, -10.0]], [[10.0, 12.0, 4.0]], now_s=0.0,
                           lane_index=[[0, 1, 2]])
    assert result["controller"].tolist() == [[DIRECT, DIRECT, NATIVE]]
    assert result["target_speed_mps"][0, :2].tolist() == pytest.approx([10.0, 12.0])
    aborted = executor.step([[20.0, 0.0, -10.0]], [[10.0, 12.0, 4.0]], now_s=1.0,
                            abort=[True], base_target_speed_mps=[[22.0, 23.0, 0.0]])
    assert aborted["controller"].tolist() == [[DAMPING, NATIVE, NATIVE]]
    assert aborted["target_speed_mps"].tolist() == [[22.0, 23.0, 0.0]]
    assert int(aborted["phase"][0]) == PHASE_HANDBACK


def test_safety_abort_and_lane_coverage_are_fail_closed():
    ex = TorchWallExecutor([[True, True]], [12.0], prior_controller=[[NATIVE, DAMPING]])
    out = ex.step([[0.0, 160.1]], [[12.0, 12.0]], now_s=0, lane_index=[[0, 1]])
    assert out["abort_reason"] == ("member_span_exceeded",)
    assert out["controller"].tolist() == [[NATIVE, DAMPING]]
    ex2 = TorchWallExecutor([[True, True]], [12.0], prior_controller=[[NATIVE, DAMPING]])
    out2 = ex2.step([[0.0, 1.0]], [[12.0, 12.0]], now_s=0, lane_index=[[0, 0]])
    assert out2["abort_reason"] == ("lane_coverage_violation",)
    assert torch.all(out2["controller"] != DIRECT)

    # Non-member rear-end surveillance uses the same comparable coordinate.
    ex3 = TorchWallExecutor([[True, True, False]], [12.0],
                            prior_controller=[[NATIVE, DAMPING, NATIVE]])
    out3 = ex3.step([[50.0, 55.0, 45.0]], [[12.0, 12.0, 15.0]], now_s=0,
                    lane_index=[[0, 1, 2]])
    assert out3["abort_reason"] == ("non_member_rear_end_risk:2",)
    assert out3["controller"].tolist() == [[NATIVE, DAMPING, NATIVE]]


def test_planner_release_is_rejected_in_assembly_but_abort_is_available():
    ex = TorchWallExecutor([[True, True]], [12.0], prior_controller=[[NATIVE, NATIVE]])
    out = ex.step([[0.0, 30.0]], [[12.0, 12.0]], now_s=0, release=[True])
    assert int(out["phase"][0]) == PHASE_ASSEMBLY
    aborted = ex.step([[0.0, 30.0]], [[12.0, 12.0]], now_s=1, abort=[True])
    assert int(aborted["phase"][0]) == PHASE_HANDBACK
    assert aborted["controller"].tolist() == [[NATIVE, NATIVE]]


@pytest.mark.skipif(not HAS_SOURCE, reason="staged wall-session reference source is absent")
def test_v22_merge_release_matches_copied_simulator_session():
    executor = TorchWallExecutor([[True, True]], [15.0],
                                 prior_controller=[[NATIVE, DAMPING]],
                                 law="wall-v2.2-merge-release")
    sim = wall_group_v22.WallGroupSessionV22(
        "v22", members=frozenset(("c0", "c1")), wall_speed_mps=15.0)
    # First step assembles immediately at a compact span, but still returns
    # the simulator's assembly pacing commands on that transition tick.
    p, v = [2600.0, 2598.0], [15.0, 15.0]
    result = executor.step([p], [v], now_s=0.0, lane_index=[[0, 1]])
    commands = sim.tick(_frames(p, v), 0.0)
    assert sim.phase == "hold" and result["phase_names"][0] == sim.phase
    for vehicle, accel in commands.items():
        assert abs(float(result["acceleration_mps2"][0, int(vehicle[-1])]) - accel) <= 1e-6

    p = [2644.2, 2640.0]
    result = executor.step([p], [v], now_s=1.0, lane_index=[[0, 1]])
    commands = sim.tick(_frames(p, v), 1.0)
    assert commands == {}
    assert sim.phase == "handback" and result["phase_names"][0] == "handback"
    assert result["controller"].tolist() == [[NATIVE, DAMPING]]
