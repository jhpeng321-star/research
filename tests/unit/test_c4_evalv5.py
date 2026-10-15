"""Offline deployment contract checks; no SUMO or traffic simulation is started."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from traffic_shepherd.c4.baselines import make_policy, run_policy_episode
from traffic_shepherd.c4.env import LC_MODE_POLICY, CavSumoEnv
from wms.control.lanes import LaneRules
from wms.control.mappo.agent import export_payload
from wms.control.policy import CavPolicy, PolicyExecutor, RsuGroupEncoder


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def observation():
    # Isolated CAV, HDV, and absent CAV: no simulated dynamics.
    return {
        "s_m": torch.tensor([[-600.0, -300.0, -900.0]]),
        "v_mps": torch.full((1, 3), 10.0),
        "a_mps2": torch.zeros(1, 3),
        "lane": torch.tensor([[0, 1, 0]]),
        "present": torch.tensor([[True, True, False]]),
        "is_cav": torch.tensor([[True, False, True]]),
        "length_m": torch.full((1, 3), 5.0),
    }


@pytest.mark.parametrize("trainer,cap", [("dreamer", 1.2), ("mappo", 2.6)])
def test_collector_default_loader_decodes_export_bounds_and_lane_head(tmp_path, trainer, cap):
    actor, encoder = CavPolicy(accel_max=cap), RsuGroupEncoder()
    with torch.no_grad():
        actor.head[-1].weight.zero_()
        actor.head[-1].bias.copy_(torch.tensor([10.0, -5.0]))
        actor.lane_head[-1].weight.zero_()
        actor.lane_head[-1].bias.copy_(torch.tensor([-10.0, -10.0, 10.0]))
    payload = export_payload(actor, encoder, LaneRules(dwell_s=0), trainer=trainer)
    checkpoint = tmp_path / f"{trainer}.pt"
    torch.save(payload, checkpoint)
    executor = make_policy(str(checkpoint))
    assert isinstance(executor, PolicyExecutor)
    assert executor.has_lane_head
    assert (executor.accel_min, executor.accel_max) == (-3.0, cap)
    decoded = []
    handle = executor.policy.register_forward_hook(lambda module, args, out: decoded.append(out))
    try:
        accel, lanes = executor(observation(), with_lanes=True)
    finally:
        handle.remove()
    assert decoded[0].accel[0, 0].item() == pytest.approx(cap, abs=1e-6)
    # The smooth safety minimum slightly lowers even a free-road saturated command.
    assert cap - 0.1 < accel[0, 0] <= cap
    assert np.isnan(accel[0, 1:]).all()
    assert lanes.tolist() == [[1, -1, -1]]


def test_episode_runner_passes_lane_head_and_resets_executor():
    calls = []
    accel = np.array([[1.2, np.nan, np.nan]])
    lanes = np.array([[1, -1, -1]])
    expected = {"test_result": "done"}

    class Policy:
        has_lane_head = True

        def reset(self):
            calls.append("policy_reset")

        def __call__(self, state, *, with_lanes):
            assert with_lanes and state is initial
            return accel, lanes

    class Stub:
        def reset(self, **kwargs):
            calls.append("env_reset")
            assert kwargs["controller"] == "direct"
            return initial

        def step(self, action, *, lanes):
            assert action is accel
            assert lanes.tolist() == [[1, -1, -1]]
            return initial, {"done": True, "summary": expected}

    initial = observation()
    result = run_policy_episode(Stub(), Policy(), seed=964199, demand_vph=2400,
                                penetration=0.2, warmup_s=60, horizon_s=120)
    assert result is expected
    assert calls == ["policy_reset", "env_reset"]


def test_env_command_mapping_is_cav_only_and_has_no_control_zone():
    env = object.__new__(CavSumoEnv)
    env.k = 3
    env.frame = SimpleNamespace(present=np.array([True, True, False]),
                                is_cav=np.array([True, False, True]),
                                vid_index=np.arange(3))
    env.builder = SimpleNamespace(vehicle_ids=["cav", "hdv", "absent"])
    assert env._accel_by_vehicle([[1.2, 2.0, 2.0]]) == {"cav": 1.2}
    assert env._lanes_by_vehicle([[1, 1, 1]]) == {"cav": 1}
    calls = []
    env._lc_original = {}
    env.api = SimpleNamespace(vehicle=SimpleNamespace(
        getLaneChangeMode=lambda vid: 1621,
        setLaneChangeMode=lambda vid, mode: calls.append((vid, mode))))
    env._lane_control({"cav"}, {"cav", "hdv"})
    assert calls == [("cav", 513)]
    assert LC_MODE_POLICY == 513
    env._lane_control(set(), {"cav", "hdv"})
    assert calls == [("cav", 513), ("cav", 1621)]


def test_native_dispatch_remains_frozen_baseline():
    assert make_policy("native") == "native"
