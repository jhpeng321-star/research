"""CPU-only contract tests and an executable, file-free Dreamer smoke test.

The fake world model follows WM-S's controller callback and batch-first rollout
contract. In particular it calls the controller on *every* physics step; it does
not quietly implement the trainer's two-step action hold on the trainer's behalf.
"""
from __future__ import annotations

import copy
import json
import math
import random
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from wms.control.contract import PHYSICS_DT_S, STEPS_PER_CONTROL, RewardWeights
from wms.control.dreamer import (
    DreamerConfig,
    DreamerTrainer,
    ensemble_penalty,
    lambda_returns,
)
from wms.control.start_states import (
    StartBatch, StartStateSampler, checked_path, split_start_states,
)


def _seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _state(batch_size, *, speed=8.0, device="cpu"):
    """One CAV, one HDV and one absent slot, with no external data dependency."""
    def expand(row, dtype=torch.float32):
        return torch.tensor(row, dtype=dtype, device=device).expand(batch_size, -1).clone()

    return {
        "s_m": expand([-20.0, -35.0, 0.0]),
        "v_mps": expand([speed, 6.0, 0.0]),
        "a_mps2": expand([0.0, 0.0, 0.0]),
        "lane": expand([1, 1, -1], torch.long),
        "present": expand([True, True, False], torch.bool),
        "is_cav": expand([True, False, False], torch.bool),
        "length_m": expand([5.0, 5.0, 0.0]),
    }


class FakeSampler:
    """Stateful sampler exercising both checkpoint state and all three CPU RNGs."""

    def __init__(self, *, speed=8.0, randomize=False):
        self.speed = speed
        self.randomize = randomize
        self.draws = 0

    def sample(self, batch_size, device="cpu"):
        assert torch.device(device).type == "cpu"
        self.draws += 1
        offset = 0.0
        if self.randomize:
            offset = random.random() + float(np.random.random()) + torch.rand(()).item()
        return StartBatch(
            state=_state(batch_size, speed=self.speed + offset, device=device),
            rollout_kwargs={"driver_params": {"test_gain": torch.tensor(0.25)}},
        )

    def state_dict(self):
        return {"draws": self.draws, "speed": self.speed, "randomize": self.randomize}

    def load_state_dict(self, state):
        self.draws = state["draws"]
        self.speed = state["speed"]
        self.randomize = state["randomize"]


class FakeWorldModel(nn.Module):
    """Differentiable CAV command integration plus a simple HDV response."""

    def __init__(self, *, hdv_bias=0.0):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.25))
        self.hdv_bias = hdv_bias
        self.physics = SimpleNamespace(dt_s=PHYSICS_DT_S)
        self.commands = []
        self.transitions = []
        self.kwargs_seen = []

    def rollout(self, state, *, n_steps, cav_controller, **kwargs):
        assert cav_controller is not None, "analytic control needs the WM-S callback"
        self.kwargs_seen.append(kwargs)
        current = dict(state)
        traces = {key: [] for key in ("positions", "speeds", "lanes", "masks", "accel")}
        for _ in range(n_steps):
            command = cav_controller(current)
            assert command.shape == (*current["v_mps"].shape, 4)
            assert command.device.type == "cpu"
            self.commands.append(command)
            cav = current["present"] & current["is_cav"]
            hdv_accel = self.gain * (current["v_mps"][:, :1] - current["v_mps"])
            hdv_accel = hdv_accel + self.hdv_bias
            speed = current["v_mps"] + PHYSICS_DT_S * hdv_accel
            position = current["s_m"] + PHYSICS_DT_S * speed
            nxt = {
                **current,
                "s_m": torch.where(cav, command[..., 0], position),
                "lane": torch.where(cav, command[..., 1].long(), current["lane"]),
                "v_mps": torch.where(cav, command[..., 2], speed),
                "a_mps2": torch.where(cav, command[..., 3], hdv_accel),
            }
            for name in ("s_m", "v_mps", "a_mps2"):
                nxt[name] = torch.where(current["present"], nxt[name], current[name])
            self.transitions.append((current, nxt))
            for out, name in (
                ("positions", "s_m"), ("speeds", "v_mps"), ("lanes", "lane"),
                ("masks", "present"), ("accel", "a_mps2"),
            ):
                traces[out].append(nxt[name])
            current = nxt
        return {**{key: torch.stack(seq, dim=1) for key, seq in traces.items()},
                "state": current}


def observations(state, report_age_s=None):
    del report_age_s
    return SimpleNamespace(
        own=torch.stack((state["v_mps"], state["a_mps2"]), dim=-1),
        reports=state["v_mps"].unsqueeze(-1).unsqueeze(-1),
        report_mask=state["present"].unsqueeze(-1),
        cav_mask=state["present"] & state["is_cav"],
    )


class FakeGroupEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.signal = nn.Parameter(torch.tensor(0.1))

    def forward(self, obs, state):
        del state
        return self.signal.expand(obs.cav_mask.shape[0], 1)


class FakePolicy(nn.Module):
    def __init__(self, *, feedback=0.0):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(0.2))
        self.log_std = nn.Parameter(torch.tensor(-2.0))
        self.feedback = feedback
        self.calls = []

    def forward(self, obs, group_signal, *, deterministic=False):
        loc = self.mean + group_signal[..., :1] + self.feedback * obs.own[..., 0]
        loc = loc.expand_as(obs.cav_mask)
        distribution = torch.distributions.Normal(loc, self.log_std.exp().expand_as(loc))
        action = loc if deterministic else distribution.rsample()
        action = torch.where(obs.cav_mask, action, torch.zeros_like(action))
        log_prob = torch.where(obs.cav_mask, distribution.log_prob(action),
                               torch.zeros_like(action))
        self.calls.append({"deterministic": deterministic, "action": action})
        return action, log_prob


class FakeCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.slope = nn.Parameter(torch.tensor(0.05))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.inputs = []

    def forward(self, state):
        self.inputs.append((torch.is_grad_enabled(), state["v_mps"].requires_grad))
        return self.slope * state["v_mps"][:, 0] + self.bias


def speed_reward(state, next_state, accel, weights):
    del state, accel
    assert isinstance(weights, RewardWeights)
    reward = next_state["v_mps"][:, 0]
    zero = torch.zeros_like(reward)
    return {"reward": reward, "cost_ttc": torch.full_like(reward, 0.05), "delay": reward,
            "throughput": zero, "smooth": zero, "hard_brake": zero,
            "fairness_delay": zero}


def hdv_reward(state, next_state, accel, weights):
    result = speed_reward(state, next_state, accel, weights)
    result["reward"] = next_state["v_mps"][:, 1]
    return result


def safety(state, accel):
    return torch.where(state["present"] & state["is_cav"], accel,
                       torch.zeros_like(accel))


def _trainer(*, sampler=None, ensemble=1, feedback=0.0, reward_fn=speed_reward, **overrides):
    settings = dict(
        batch_size=2, horizon=4, gamma=0.99, lambda_=0.95,
        actor_lr=1e-3, critic_lr=1e-3, entropy_weight=0.0, uncertainty_weight=0.0,
        ttc_budget=0.1, dual_lr=0.2, lagrange_init=0.0, grad_clip=100.0,
        target_tau=0.25, return_ema=0.9, checkpoint_every=0, validation_every=0,
    )
    settings.update(overrides)
    return DreamerTrainer(
        [FakeWorldModel(hdv_bias=i * 0.4) for i in range(ensemble)],
        FakePolicy(feedback=feedback), FakeGroupEncoder(), FakeCritic(),
        FakeSampler() if sampler is None else sampler,
        observation_builder=observations, reward_fn=reward_fn, safety=safety,
        config=DreamerConfig(**settings), reward_weights=RewardWeights(),
        device="cpu", output_dir=None,
    )


def _assert_finite_metrics(metrics):
    for name in ("actor_grad_norm", "critic_grad_norm", "returns",
                 "lagrange_multiplier", "ensemble_disagreement"):
        assert name in metrics
        assert math.isfinite(float(metrics[name])), (name, metrics[name])
    assert float(metrics["actor_grad_norm"]) > 0.0
    assert float(metrics["critic_grad_norm"]) > 0.0


def _assert_equal(left, right):
    """Recursive exact comparison, including optimizer tensors and RNG arrays."""
    if torch.is_tensor(left):
        assert torch.is_tensor(right)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_equal(a, b)
    else:
        assert left == right


@pytest.fixture(autouse=True)
def cpu_rng():
    """Do not leak these unit tests' CPU randomness into other unit files."""
    previous = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    _seed()
    yield
    random.setstate(previous[0])
    np.random.set_state(previous[1])
    torch.set_rng_state(previous[2])


@pytest.mark.parametrize("gamma,lambda_,expected", [
    (0.0, 0.95, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]),
    (0.9, 0.0, [[19.0, 22.9], [29.0, 32.9], [39.0, 42.9]]),
    (0.9, 1.0, [[34.39, 43.249], [37.1, 43.61], [39.0, 42.9]]),
    (0.9, 0.5, [[24.8725, 30.66475], [33.05, 38.255], [39.0, 42.9]]),
])
def test_lambda_returns_hand_computed(gamma, lambda_, expected):
    rewards = torch.tensor([[1., 4.], [2., 5.], [3., 6.]], dtype=torch.float64)
    values = torch.tensor([[10., 11.], [20., 21.], [30., 31.], [40., 41.]],
                          dtype=torch.float64)
    actual = lambda_returns(rewards, values, gamma=gamma, lambda_=lambda_)
    assert actual.shape == (3, 2)
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64))


def test_lambda_returns_default_arguments_and_gradient():
    rewards = torch.tensor([[1.], [2.]], requires_grad=True)
    values = torch.tensor([[100.], [3.], [4.]])
    actual = lambda_returns(rewards, values)
    last = 2.0 + 0.99 * 4.0
    first = 1.0 + 0.99 * (0.05 * 3.0 + 0.95 * last)
    torch.testing.assert_close(actual, torch.tensor([[first], [last]]))
    actual[0].sum().backward()
    torch.testing.assert_close(rewards.grad, torch.tensor([[1.], [0.99 * 0.95]]))


def test_lambda_returns_one_step_bootstraps_terminal_value():
    rewards = torch.tensor([[2., 3.]])
    values = torch.tensor([[999., -999.], [4., 5.]])
    for mixing in (0.0, 0.5, 1.0):
        torch.testing.assert_close(
            lambda_returns(rewards, values, gamma=0.8, lambda_=mixing),
            torch.tensor([[5.2, 7.]]),
        )


def test_ensemble_penalty_population_std_and_singleton():
    rewards = torch.tensor([[[1., 2.], [4., 8.]], [[3., 6.], [4., 12.]]])
    actual = ensemble_penalty(rewards, weight=0.25)
    assert actual.shape == (2, 2)
    torch.testing.assert_close(actual, torch.tensor([[0.25, 0.5], [0., 0.5]]))
    torch.testing.assert_close(ensemble_penalty(rewards, weight=0.0), torch.zeros(2, 2))
    singleton = ensemble_penalty(rewards[:1], weight=10.0)
    assert torch.isfinite(singleton).all()
    torch.testing.assert_close(singleton, torch.zeros(2, 2))


def test_lagrange_ascent_sign_budget_and_nonnegative_projection():
    trainer = _trainer(lagrange_init=0.3, dual_lr=0.2, ttc_budget=0.1)
    assert trainer.update_lagrange(0.6) == pytest.approx(0.4)
    assert trainer.update_lagrange(0.1) == pytest.approx(0.4)
    assert trainer.update_lagrange(0.0) == pytest.approx(0.38)
    for _ in range(30):
        result = trainer.update_lagrange(0.0)
        assert result >= 0.0
    assert result == 0.0
    assert trainer.update_lagrange(0.6) == pytest.approx(0.1)


def test_imagination_shapes_action_hold_and_physics_reward_sum():
    assert STEPS_PER_CONTROL == 2
    trainer = _trainer(ensemble=2, horizon=3, feedback=0.02)
    batch = FakeSampler().sample(2)
    original = copy.deepcopy(batch.state)
    imagined = trainer.imagine(batch, deterministic=True)
    for key in ("rewards", "costs", "returns", "entropy"):
        assert imagined[key].shape == (2, 3, 2), key
        assert torch.isfinite(imagined[key]).all()
    assert imagined["disagreement"].shape == (3, 2)
    assert imagined["actions"].shape == (2, 3, 2, 3)
    assert isinstance(imagined["components"], dict)
    assert len(imagined["states"]) == 2
    for e, model in enumerate(trainer.world_models):
        assert len(imagined["states"][e]) == 4
        assert len(model.commands) == 6, "callback must run at physics frequency"
        assert all("driver_params" in kwargs for kwargs in model.kwargs_seen)
        for h in range(3):
            first, second = model.commands[2 * h:2 * h + 2]
            torch.testing.assert_close(first[..., 3], second[..., 3])
            torch.testing.assert_close(imagined["actions"][e, h, :, 0], first[:, 0, 3])
            assert torch.count_nonzero(first[:, 1:, 3]) == 0
            # The command is held, but next-state motion must be recomputed each step.
            torch.testing.assert_close(second[:, 0, 2] - first[:, 0, 2],
                                       PHYSICS_DT_S * first[:, 0, 3])
            torch.testing.assert_close(imagined["rewards"][e, h],
                                       first[:, 0, 2] + second[:, 0, 2])
            torch.testing.assert_close(imagined["costs"][e, h], torch.full((2,), 0.1))
            torch.testing.assert_close(imagined["states"][e][h + 1]["v_mps"][:, 0],
                                       second[:, 0, 2])
        assert not torch.equal(model.commands[0][:, 0, 3], model.commands[2][:, 0, 3]), \
            "feedback policy must be called again at the next control boundary"
    _assert_equal(batch.state, original)


def test_analytic_actor_and_group_gradients_through_frozen_world_model():
    trainer = _trainer(horizon=1, gamma=0.0, lambda_=1.0)
    initial_models = [copy.deepcopy(model.state_dict()) for model in trainer.world_models]
    result = trainer.imagine(FakeSampler().sample(2), deterministic=True)
    # v1 = v0 + .5*a; v2 = v0 + a; r = v1+v2 => dr/da = 1.5.
    actor_grad, group_grad = torch.autograd.grad(
        result["returns"].mean(), (trainer.policy.mean, trainer.group_encoder.signal))
    torch.testing.assert_close(actor_grad, torch.tensor(1.5))
    torch.testing.assert_close(group_grad, torch.tensor(1.5))
    actor_before = trainer.policy.mean.detach().clone()
    group_before = trainer.group_encoder.signal.detach().clone()
    metrics = trainer.update()
    _assert_finite_metrics(metrics)
    assert trainer.update_count == 1
    assert trainer.policy.mean > actor_before
    assert trainer.group_encoder.signal > group_before
    for model, before in zip(trainer.world_models, initial_models):
        _assert_equal(model.state_dict(), before)
        assert all(not parameter.requires_grad for parameter in model.parameters())
        assert all(parameter.grad is None for parameter in model.parameters())


def test_actor_credit_reaches_hdv_response():
    trainer = _trainer(horizon=2, reward_fn=hdv_reward, gamma=0.0)
    result = trainer.imagine(FakeSampler().sample(2), deterministic=True)
    grad, = torch.autograd.grad(result["returns"].mean(), (trainer.policy.mean,))
    assert grad > 0, "CAV actions must affect team reward through HDV dynamics"


def test_actor_gradient_includes_frozen_terminal_critic_bootstrap():
    trainer = _trainer(horizon=1, gamma=0.9, lambda_=1.0)
    result = trainer.imagine(FakeSampler().sample(2), deterministic=True)
    grad, = torch.autograd.grad(result["returns"].mean(), (trainer.policy.mean,))
    # Reward derivative 1.5 plus gamma * value slope .05 * final-speed derivative 1.
    torch.testing.assert_close(grad, torch.tensor(1.545))
    assert all(p.grad is None and not p.requires_grad for p in trainer.target_critic.parameters())


def test_injected_bridge_callback_preserves_policy_cadence_and_gradients():
    trainer = _trainer(horizon=2, gamma=0.0, feedback=0.02)
    factory_calls = []

    def factory(policy, group_encoder, safety_layer, record):
        factory_calls.append(record)

        def callback(state):
            obs = observations(state)
            action, _ = policy(obs, group_encoder(obs, state))
            action = safety_layer(state, action)
            v = state["v_mps"] + PHYSICS_DT_S * action
            s = state["s_m"] + PHYSICS_DT_S * state["v_mps"] + 0.5 * PHYSICS_DT_S**2 * action
            return torch.stack((s, state["lane"].float(), v, action), -1)

        return callback

    trainer.controller_factory = factory
    result = trainer.imagine(FakeSampler().sample(2), deterministic=True)
    assert factory_calls == [None]
    assert len(trainer.policy.calls) == 2
    assert len(trainer.world_models[0].commands) == 4
    torch.testing.assert_close(trainer.world_models[0].commands[0][..., 3],
                               trainer.world_models[0].commands[1][..., 3])
    grad, = torch.autograd.grad(result["returns"].mean(), (trainer.policy.mean,))
    assert grad > 0


def test_imagination_ensemble_penalty_reduces_returns():
    trainer = _trainer(ensemble=2, reward_fn=hdv_reward, horizon=2,
                       gamma=0.0, uncertainty_weight=0.0)
    batch = FakeSampler().sample(2)
    baseline = trainer.imagine(batch, deterministic=True)
    trainer.config.uncertainty_weight = 0.75
    penalized = trainer.imagine(batch, deterministic=True)
    expected = ensemble_penalty(baseline["rewards"], weight=0.75)
    assert torch.count_nonzero(expected) > 0
    torch.testing.assert_close(baseline["returns"] - penalized["returns"],
                               expected.unsqueeze(0).expand_as(baseline["returns"]))


def test_critic_fit_uses_detached_states_and_ema_target():
    trainer = _trainer(target_tau=0.25)
    target_before = copy.deepcopy(trainer.target_critic.state_dict())
    metrics = trainer.update()
    _assert_finite_metrics(metrics)
    # Critic regression must not retain actor/WM graphs in its state inputs.
    fitting = [state_grad for grad_enabled, state_grad in trainer.critic.inputs if grad_enabled]
    assert fitting, "online critic must participate in gradient-enabled regression"
    assert not any(fitting), "critic state features must be detached from imagination"
    assert all(not p.requires_grad and p.grad is None
               for p in trainer.target_critic.parameters())
    assert any(grad_enabled and state_grad for grad_enabled, state_grad in trainer.target_critic.inputs), \
        "frozen bootstrap weights must still pass analytic state gradients to the actor"
    for name, value in trainer.target_critic.state_dict().items():
        expected = 0.75 * target_before[name] + 0.25 * trainer.critic.state_dict()[name]
        torch.testing.assert_close(value, expected)


def test_checkpoint_exact_next_update_and_cpu_rng_resume():
    sampler = FakeSampler(randomize=True)
    trainer = _trainer(sampler=sampler, ensemble=2)
    _assert_finite_metrics(trainer.update())
    # Keep checkpoint I/O inside the workspace, even if TEMP points elsewhere.
    with tempfile.TemporaryDirectory(prefix="c4-checkpoint-", dir=Path(__file__).parent) as tmp:
        checkpoint = Path(tmp) / "trainer.pt"
        trainer.save_checkpoint(checkpoint)
        expected_rng = (random.random(), np.random.random(), torch.rand(5))
        expected_metrics = trainer.update()
        expected_modules = [copy.deepcopy(module.state_dict()) for module in
                            (trainer.policy, trainer.group_encoder, trainer.critic,
                             trainer.target_critic)]
        expected_sampler = copy.deepcopy(sampler.state_dict())
        expected_count = trainer.update_count
        expected_checkpoint = Path(tmp) / "expected-next.pt"
        trainer.save_checkpoint(expected_checkpoint)
        # A different seed and fresh optimizer state expose incomplete restore logic.
        _seed(999)
        resumed_sampler = FakeSampler(speed=17.0, randomize=True)
        resumed = _trainer(sampler=resumed_sampler, ensemble=2)
        resumed.load_checkpoint(checkpoint)
        assert resumed.update_count == expected_count - 1
        actual_rng = (random.random(), np.random.random(), torch.rand(5))
        _assert_equal(expected_rng, actual_rng)
        actual_metrics = resumed.update()
        _assert_equal({k: v for k, v in expected_metrics.items() if k != "wall_time_s"},
                      {k: v for k, v in actual_metrics.items() if k != "wall_time_s"})
        for expected, actual in zip(expected_modules,
                                    (resumed.policy, resumed.group_encoder, resumed.critic,
                                     resumed.target_critic)):
            _assert_equal(expected, actual.state_dict())
        _assert_equal(expected_sampler, resumed_sampler.state_dict())
        assert resumed.update_count == expected_count
        actual_checkpoint = Path(tmp) / "actual-next.pt"
        resumed.save_checkpoint(actual_checkpoint)
        # This also compares optimizer moments, return-normalization state, dual
        # state and saved RNGs, without depending on private attribute names.
        _assert_equal(torch.load(expected_checkpoint, map_location="cpu", weights_only=False),
                      torch.load(actual_checkpoint, map_location="cpu", weights_only=False))


def test_validation_uses_supplied_sampler_without_training_side_effects():
    training_sampler = FakeSampler(speed=8.0)
    validation_sampler = FakeSampler(speed=20.0)
    trainer = _trainer(sampler=training_sampler, horizon=2, gamma=0.0)
    _assert_finite_metrics(trainer.update())
    before_sampler = copy.deepcopy(training_sampler.state_dict())
    before_count = trainer.update_count
    before_dual = trainer.update_lagrange(trainer.config.ttc_budget)
    before_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    before_modules = [copy.deepcopy(module.state_dict()) for module in
                      (trainer.policy, trainer.group_encoder, trainer.critic,
                       trainer.target_critic)]
    calls_before = len(trainer.policy.calls)
    value = trainer.validate(validation_sampler, batches=2)
    assert isinstance(value, float) and math.isfinite(value)
    assert value > 35.0, "validation must imagine starts from the supplied sampler"
    assert validation_sampler.draws == 2
    _assert_equal(training_sampler.state_dict(), before_sampler)
    assert trainer.update_count == before_count
    assert trainer.update_lagrange(trainer.config.ttc_budget) == before_dual
    _assert_equal(before_rng, (random.getstate(), np.random.get_state(), torch.get_rng_state()))
    assert all(call["deterministic"] for call in trainer.policy.calls[calls_before:])
    for before, module in zip(before_modules,
                              (trainer.policy, trainer.group_encoder, trainer.critic,
                               trainer.target_critic)):
        _assert_equal(before, module.state_dict())


def _dataset():
    from wms.data.dataset import TrajectoryDataset

    records = []
    for p in (2, 5, 10, 20, 40):
        for window in range(4):
            kin = np.zeros((2, 20, 7), dtype=np.float32)
            kin[..., 0] = np.arange(20) * 10
            kin[..., 2] = 1
            kin[..., 4] = 8
            kin[..., 5] = 0.3
            kin[..., 6] = 5
            cav = np.zeros((2, 20), dtype=bool)
            cav[:, :max(1, round(p / 100 * 20))] = True
            records.append({"states": kin, "present": np.ones((2, 20), dtype=bool),
                            "is_cav": cav, "hist": np.ones((20, 3, 6), dtype=np.float32),
                            "hist_mask": np.ones((20, 3), dtype=bool),
                            "cav_teacher": np.ones((2, 20, 4), dtype=np.float32),
                            "action_timeline": {"unused": True},
                            "meta": {"world": f"w123m2700p{p:02d}", "t0_s": window * 10,
                                     "split": "TRAIN"}})
    return TrajectoryDataset(records, receipt={"split": "TRAIN"})


def test_start_states_mix_exclusions_history_and_no_teacher():
    sampler = StartStateSampler(_dataset(), excluded_penetrations=(0.02, 0.4), seed=7)
    assert sampler.penetrations == [0.05, 0.1, 0.2]
    batch = sampler.sample(120)
    assert {m["world"][-3:] for m in batch.metadata} == {"p05", "p10", "p20"}
    assert batch.state["s_m"].shape == (120, 20)
    assert batch.state["a_mps2"][0, 0] == pytest.approx(0.3)
    assert set(batch.rollout_kwargs) == {"history", "history_mask", "driver_history", "driver_history_mask"}
    assert batch.rollout_kwargs["driver_history"].shape == (120, 20, 3, 6)
    assert "cav_teacher" not in batch.rollout_kwargs
    assert "action_timeline" not in batch.rollout_kwargs
    torch.testing.assert_close(batch.rollout_kwargs["history"], batch.rollout_kwargs["driver_history"])


def test_start_state_optional_relabelling_and_sampler_resume():
    dataset = _dataset()
    original = copy.deepcopy(dataset.records)
    sampler = StartStateSampler(dataset, relabel=True, target_penetration=0.4, seed=12)
    saved = copy.deepcopy(sampler.state_dict())
    expected = sampler.sample(5)
    assert torch.equal(expected.state["is_cav"].sum(-1), torch.full((5,), 8))
    sampler.load_state_dict(saved)
    actual = sampler.sample(5)
    _assert_equal(actual.state, expected.state)
    _assert_equal(actual.metadata, expected.metadata)
    _assert_equal(dataset.records, original)
    with pytest.raises(ValueError, match="target"):
        StartStateSampler(dataset, relabel=True, target_penetration=0.3)


def test_development_validation_windows_are_disjoint_and_stratified():
    dataset = _dataset()
    train, val = split_start_states(dataset, val_fraction=0.25, seed=9)
    train_ids = {i for group in train.groups.values() for i in group}
    val_ids = {i for group in val.groups.values() for i in group}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == set(range(len(dataset)))
    assert train.penetrations == val.penetrations == [0.02, 0.05, 0.1, 0.2, 0.4]
    assert all(len(group) == 1 for group in val.groups.values())


def test_restricted_paths_and_nontraining_metadata_refused_before_loading():
    # Guard tests never create or open any of the forbidden paths.
    for token in ("sealed", "HOLDOUT", "heldout", "cal_split", "chunk5", "j4b", "inc1"):
        with pytest.raises(ValueError, match="restricted"):
            checked_path(f"{token}/never-open.pt")
    with pytest.raises(ValueError, match="outside workspace"):
        checked_path(Path(__file__).resolve().parents[3] / "never-open.pt")  # parent of the repo root
    dataset = _dataset()
    dataset.receipt["split"] = "VAL"
    with pytest.raises(ValueError, match="TRAIN"):
        StartStateSampler(dataset)
    dataset.receipt["split"] = "TRAIN"
    dataset.records[0]["meta"]["split"] = "CAL"
    dataset.records[1]["meta"]["world"] = "excluded_chunk5_p02"
    sampler = StartStateSampler(dataset)
    assert 0 not in sampler.metadata and 1 not in sampler.metadata


def test_cli_cpu_smoke_logging_best_checkpoints_summary_and_resume(monkeypatch):
    from wms.control import train_cli

    monkeypatch.setattr(train_cli, "_load_components", lambda: {
        "policy": FakePolicy(), "group_encoder": FakeGroupEncoder(), "critic": FakeCritic(),
        "observation_builder": observations, "reward_fn": speed_reward, "safety": safety})
    monkeypatch.setattr(train_cli, "_load_world_models", lambda paths, device: [FakeWorldModel()])
    monkeypatch.setattr(train_cli, "_load_starts", lambda data, models, seed, entries=False:
                        split_start_states(_dataset(), seed=seed))
    with tempfile.TemporaryDirectory(prefix="c4-cli-", dir=Path(__file__).parent) as tmp:
        output = Path(tmp)
        arguments = ["--config", "configs/c4_policy.toml", "--wm-ckpt", str(output / "fake.pt"),
                     "--device", "cuda", "--smoke", "--output-dir", str(output)]
        assert train_cli.main(arguments) == 0
        summary = json.loads((output / "RUN_SUMMARY.json").read_text())
        assert summary["status"] == "completed"
        assert summary["updates"] == 2 and summary["device"] == "cpu"
        assert summary["config"]["batch_size"] == 2 and summary["config"]["horizon"] == 4
        assert summary["imagined_horizon_s"] == 4
        assert (output / "policy_best.pt").is_file()
        assert (output / "policy_000002.pt").is_file()
        logs = [json.loads(line) for line in (output / "updates.jsonl").read_text().splitlines()]
        assert len(logs) == 2
        for entry in logs:
            assert {"delay", "throughput", "smooth", "hard_brake", "fairness", "cost_ttc",
                    "entropy", "action_stats", "ensemble_disagreement", "wall_time_s"} <= entry.keys()
        assert train_cli.main(arguments + ["--resume", str(output / "policy_latest.pt")]) == 0
        summary = json.loads((output / "RUN_SUMMARY.json").read_text())
        assert summary["updates"] == 4
        assert len((output / "updates.jsonl").read_text().splitlines()) == 4


def test_resume_refuses_changed_frozen_world_model():
    trainer = _trainer()
    with tempfile.TemporaryDirectory(prefix="c4-identity-", dir=Path(__file__).parent) as tmp:
        checkpoint = Path(tmp) / "policy.pt"
        trainer.save_checkpoint(checkpoint)
        changed = _trainer()
        changed.world_model_ids = ["different-model"]
        with pytest.raises(ValueError, match="same frozen"):
            changed.load_checkpoint(checkpoint)


def _smoke():
    """Two tiny analytic CPU updates; output_dir=None disables all artifact writes."""
    _seed()
    torch.set_num_threads(1)
    started = time.perf_counter()
    trainer = _trainer(batch_size=2, horizon=4)
    updates = []
    for _ in range(2):
        metrics = trainer.update()
        _assert_finite_metrics(metrics)
        updates.append({name: round(float(metrics[name]), 6) for name in
                        ("returns", "actor_grad_norm", "critic_grad_norm")})
    assert trainer.update_count == 2
    assert all(p.device.type == "cpu" for p in trainer.policy.parameters())
    print(json.dumps({"walltime": round(time.perf_counter() - started, 6),
                      "updates": updates, "pass": True}, separators=(",", ":")))


if __name__ == "__main__":
    _smoke()
