"""CPU smoke and NumPy-parity checks for the local Torch physics contract."""

import importlib.util

import numpy as np
import pytest


if importlib.util.find_spec("torch") is None:
    pytest.skip("PyTorch is not installed in this workspace interpreter", allow_module_level=True)

import torch

from output.wd01.wm_n1.wms.physics_torch import IDMParameters, PhysicsConfig, PhysicsState, rollout, step


def _reference_idm(s, v, active, *, dt=0.5):
    """Independent small reference for eight free-road/following worlds."""
    result_s = s.copy()
    result_v = v.copy()
    for batch in range(s.shape[0]):
        order = np.argsort(s[batch])[::-1]
        next_v = result_v[batch].copy()
        next_s = result_s[batch].copy()
        for rank, vehicle in enumerate(order):
            if not active[batch, vehicle]:
                continue
            if rank == 0:
                gap, leader_v = 250.0, result_v[batch, vehicle]
            else:
                leader = order[rank - 1]
                gap, leader_v = result_s[batch, leader] - result_s[batch, vehicle], result_v[batch, leader]
            closing = result_v[batch, vehicle] - leader_v
            desired = 2.5 + result_v[batch, vehicle] * 1.1 + result_v[batch, vehicle] * closing / (2 * np.sqrt(2.6 * 4.5))
            desired = max(desired, 2.5)
            acceleration = 2.6 * (1 - (result_v[batch, vehicle] / 31.29) ** 4 - (desired / max(gap, 1e-3)) ** 2)
            acceleration = np.clip(acceleration, -4.5, 2.6)
            new_v = np.clip(result_v[batch, vehicle] + acceleration * dt, 0, 31.29 * 1.2)
            next_v[vehicle] = new_v
            next_s[vehicle] += 0.5 * (result_v[batch, vehicle] + new_v) * dt
        result_v[batch] = next_v
        result_s[batch] = next_s
    return result_s, result_v


def test_eight_world_numpy_parity_without_residual():
    rng = np.random.default_rng(19)
    s = 20.0 + 40.0 * np.arange(12)[None, :] + rng.uniform(0.0, 2.0, size=(8, 12))
    v = 8.0 + 0.8 * np.arange(12)[None, :] + rng.uniform(0.0, 0.2, size=(8, 12))
    active = np.ones((8, 12), dtype=bool)
    state = PhysicsState(
        torch.tensor(s, dtype=torch.float32),
        torch.tensor(v, dtype=torch.float32),
        torch.ones((8, 12), dtype=torch.long),
        torch.ones((8, 12), dtype=torch.bool),
        torch.zeros((8, 12), dtype=torch.bool),
    )
    params = IDMParameters.constant(8, 12)
    torch_states, _ = rollout(
        state,
        params,
        120,
        config=PhysicsConfig(lane_count=1),
        learned_lane_logits=torch.zeros(8, 12, 2),
    )
    ref_s, ref_v = s.copy(), v.copy()
    for horizon in range(1, 121):
        ref_s, ref_v = _reference_idm(ref_s, ref_v, active)
        if horizon in (10, 60, 120):
            np.testing.assert_allclose(torch_states[horizon].s.numpy(), ref_s, rtol=0, atol=2e-4)
            np.testing.assert_allclose(torch_states[horizon].v.numpy(), ref_v, rtol=0, atol=2e-4)


def test_projection_exit_entries_and_gradient():
    state = PhysicsState(
        torch.tensor([[590.0, 10.0]], requires_grad=True),
        torch.tensor([[20.0, 12.0]], requires_grad=True),
        torch.tensor([[1, 1]]),
        torch.tensor([[True, False]]),
        torch.tensor([[False, False]]),
    )
    params = IDMParameters.constant(1, 2)
    result = step(state, params, observed_entry_rate=torch.tensor([1.0]), entry_seed=7)
    assert bool(result.exited[0, 0])
    assert bool(result.entered[0, 1])
    loss = result.state.s.sum() + result.state.v.sum()
    loss.backward()
    assert state.s.grad is not None
    assert torch.isfinite(state.s.grad).all()


def test_mandatory_merge_and_deterministic_entry_draw():
    config = PhysicsConfig(merge_start_m=100.0, merge_end_m=200.0)
    state = PhysicsState(
        torch.tensor([[150.0, 185.0, 145.0]]),
        torch.tensor([[18.0, 18.0, 18.0]]),
        torch.tensor([[2, 1, 1]]),
        torch.ones((1, 3), dtype=torch.bool),
        torch.zeros((1, 3), dtype=torch.bool),
    )
    params = IDMParameters.constant(1, 3)
    first = step(state, params, config=config, observed_entry_rate=torch.tensor([0.0]), entry_seed=11)
    second = step(state, params, config=config, observed_entry_rate=torch.tensor([0.0]), entry_seed=11)
    assert first.state.lane[0, 0].item() == 1
    assert torch.equal(first.state.active, second.state.active)
    assert torch.allclose(first.lane_logits, second.lane_logits)


def test_eight_world_lane_decision_agreement():
    state = PhysicsState(
        torch.tensor([[150.0, 185.0, 130.0]]).expand(8, -1).clone(),
        torch.full((8, 3), 18.0),
        torch.tensor([[2, 1, 1]]).expand(8, -1).clone(),
        torch.ones((8, 3), dtype=torch.bool),
        torch.zeros((8, 3), dtype=torch.bool),
    )
    result = step(state, IDMParameters.constant(8, 3), config=PhysicsConfig())
    assert torch.equal(result.state.lane[:, 0], torch.ones(8, dtype=torch.long))
    assert torch.equal(result.lane_change[:, 0] > 0.5, torch.ones(8, dtype=torch.bool))


def test_torchphys_defaults_are_in_the_s_down_frame():
    """Defaults must be model-frame (s_down) lane geometry, not physical coords.

    Regression for the review blocker: ``downstream_exit_m``/``merge_*`` used
    the physical lane coordinates 600 / 378.66 / 454.10 while the model consumes
    ``s_down``.  The defaults are now the frozen ``model_frame_params`` contract:
    merge zone 0.0..75.44 and per-lane model-frame ends 553.33/553.33/75.44.
    """
    config = PhysicsConfig()
    assert config.merge_start_m == pytest.approx(0.0)
    assert config.merge_end_m == pytest.approx(75.44)
    assert config.downstream_exit_m == float("inf")
    assert config.lane_lengths is not None
    assert config.lane_lengths.tolist() == pytest.approx([553.33, 553.33, 75.44])
    # the old physical-frame values must not survive as s_down defaults
    assert config.merge_start_m != pytest.approx(378.66)
    assert config.merge_end_m != pytest.approx(454.10)

    # a lane-specific model-frame exit actually binds on the mainline
    state = PhysicsState(
        torch.tensor([[600.0]]),
        torch.tensor([[10.0]]),
        torch.tensor([[0]]),
        torch.ones((1, 1), dtype=torch.bool),
        torch.zeros((1, 1), dtype=torch.bool),
    )
    out = step(state, IDMParameters.constant(1, 1), config=config)
    assert bool(out.exited[0, 0])


def test_torchphys_mandatory_merge_checks_the_valid_target_side():
    """Lane 2 -> lane 1 (side 0) must use lane 1, ignoring lane 3 (side 1).

    Regression for the review blocker: the mandatory-merge decision inspected
    side 1 (lane 3) instead of side 0 (lane 1), so a feasible lane-1 merge was
    reported blocked.  Discretionary MOBIL is disabled to isolate the rule.
    """
    config = PhysicsConfig(mobil_threshold_mps2=1e9)
    # lane 1 is clear (a lead far ahead); lane 3 is blocked right behind
    state = PhysicsState(
        torch.tensor([[40.0, 60.0, 39.0]]),
        torch.tensor([[18.0, 18.0, 18.0]]),
        torch.tensor([[2, 1, 3]]),
        torch.ones((1, 3), dtype=torch.bool),
        torch.zeros((1, 3), dtype=torch.bool),
    )
    out = step(state, IDMParameters.constant(1, 3), config=config)
    assert out.state.lane[0, 0].item() == 1
    assert not bool(out.merge_blocked[0, 0])

    # lane 1 occupied too closely must be reported as blocked, not ignored
    blocked = PhysicsState(
        torch.tensor([[40.0, 39.0, 60.0]]),
        torch.tensor([[18.0, 18.0, 18.0]]),
        torch.tensor([[2, 1, 3]]),
        torch.ones((1, 3), dtype=torch.bool),
        torch.zeros((1, 3), dtype=torch.bool),
    )
    out2 = step(blocked, IDMParameters.constant(1, 3), config=config)
    assert out2.state.lane[0, 0].item() == 2
    assert bool(out2.merge_blocked[0, 0])
