"""V6 CPU-only damping, credit diagnostics, and opt-in compatibility contracts."""
from __future__ import annotations

import copy
import json
import math
from dataclasses import replace

import pytest
import torch

from wms.control.dreamer import DreamerConfig, _credit_profile, _model_identity, _rng_state
from wms.control.train_cli import _read_config
from wms.model.wms import _scale_grad

from .test_c4_dreamer import FakeSampler, FakeWorldModel, _assert_equal, _seed, _trainer


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class DampedToy(FakeWorldModel):
    """Toy CAV dynamics with WM-style carried-state gradient scaling (values unchanged)."""

    def __init__(self):
        super().__init__()
        self.grad_decay_per_step = 0.9
        self.checkpoint_forward = False
        self.cav_exec_cap = "off"
        self.decays_seen = []

    def rollout(self, state, *, cav_controller, **kwargs):
        decay = self.grad_decay_per_step
        self.decays_seen.append(decay)
        first = True

        def controller(current):
            nonlocal first
            if not first and decay < 1:
                current = {k: _scale_grad(v, decay) if torch.is_tensor(v) and v.requires_grad else v
                           for k, v in current.items()}
            first = False
            return cav_controller(current)

        return super().rollout(state, cav_controller=controller, **kwargs)


def toy(**kwargs):
    trainer = _trainer(**kwargs)
    trainer.world_models = [DampedToy().requires_grad_(False)]
    trainer.world_model_ids = [_model_identity(trainer.world_models[0],
                                             trainer.config.imagination_grad_decay)]
    return trainer


@pytest.mark.parametrize("value", [0, -0.1, 1.01, float("nan"), float("inf"), True, "0.98"])
def test_decay_validation(value):
    with pytest.raises(ValueError, match="imagination_grad_decay"):
        DreamerConfig(imagination_grad_decay=value)


@pytest.mark.parametrize("value", [-1, 0.5, True, "25", float("inf")])
def test_profile_interval_validation(value):
    with pytest.raises(ValueError, match="credit_profile_every"):
        DreamerConfig(credit_profile_every=value)


def test_v6_config_and_digest():
    raw = _read_config("configs/c4_policy_v6.toml")
    cfg = DreamerConfig(**raw["trainer"])
    assert cfg.imagination_grad_decay == 0.98 and cfg.credit_profile_every == 25
    assert cfg.horizon == 30 and cfg.source_wait_grad and cfg.critic_entry_backlog
    assert raw["run"]["output_dir"] == "runs/active/C4-POLICY-V6"
    assert DreamerConfig().imagination_grad_decay is None
    assert DreamerConfig().credit_profile_every == 0
    model = DampedToy()
    assert _model_identity(model) == _model_identity(model, None)
    assert len({_model_identity(model, d) for d in (None, 0.9, 0.98, 1.0)}) == 4


@pytest.mark.parametrize("checkpoint", [False, True])
def test_default_and_matching_wm_decay_bit_identical(checkpoint):
    snapshots = []
    for decay in (None, 0.9):
        _seed()
        trainer = toy(imagination_grad_decay=decay, checkpoint_imagination=checkpoint)
        metrics = trainer.update()
        snapshots.append(({k: v for k, v in metrics.items() if k != "wall_time_s"},
                          copy.deepcopy(trainer.policy.state_dict()),
                          copy.deepcopy(trainer.actor_optimizer.state_dict()),
                          copy.deepcopy(trainer.critic_optimizer.state_dict()),
                          _rng_state(trainer.device)))
        assert trainer.world_models[0].grad_decay_per_step == 0.9
    _assert_equal(*snapshots)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_early_action_credit_increases_without_damping(checkpoint):
    profiles = []
    for decay in (0.9, 1.0):
        trainer = toy(horizon=30, imagination_grad_decay=decay,
                      checkpoint_imagination=checkpoint)
        imagined = trainer.imagine(FakeSampler().sample(2), deterministic=True, credit_profile=True)
        profile = imagined["credit_profile"]
        assert profile["active_actions"] == 1 and profile["sample_size"] == 1
        profiles.append(profile["norms"])
        # Diagnostic traversals must leave the real training graph usable after restoration.
        imagined["returns"].mean().backward()
        assert trainer.policy.mean.grad.isfinite() and trainer.policy.mean.grad > 0
        assert trainer.world_models[0].grad_decay_per_step == 0.9
        assert trainer.world_models[0].decays_seen == [decay]
    assert profiles[1]["1"] == pytest.approx(1.5)
    for step in (5, 10, 20, 30):
        assert profiles[1][str(step)] == pytest.approx(2.0)
        assert 0 < profiles[0][str(step)] < profiles[1][str(step)]
    assert profiles[0]["30"] < profiles[0]["10"]


def test_rollout_switches_restored_on_exception(monkeypatch):
    trainer = toy(imagination_grad_decay=1.0, cav_exec_cap="idm")
    model = trainer.world_models[0]

    def fail(*args, **kwargs):
        assert model.grad_decay_per_step == 1.0
        assert model.checkpoint_forward is True and model.cav_exec_cap == "idm"
        raise RuntimeError("rollout failed")

    monkeypatch.setattr(model, "rollout", fail)
    with pytest.raises(RuntimeError, match="rollout failed"):
        trainer.imagine(FakeSampler().sample(2))
    assert model.grad_decay_per_step == 0.9
    assert model.checkpoint_forward is False and model.cav_exec_cap == "off"


def test_unsupported_override_refused_without_mutation():
    trainer = _trainer(imagination_grad_decay=0.98)
    model = trainer.world_models[0]
    model.checkpoint_forward = False
    with pytest.raises(ValueError, match="grad_decay_per_step"):
        trainer.imagine(FakeSampler().sample(2))
    assert model.checkpoint_forward is False and not hasattr(model, "grad_decay_per_step")


def test_profile_windows_subsample_and_disconnected_actions():
    action = torch.ones(2, 3, requires_grad=True)
    mask = torch.tensor([[True, False, True], [True, True, True]])
    rewards = torch.stack([action.sum(-1) * (k + 1) for k in range(5)])
    profile = _credit_profile(rewards, action, mask)
    assert profile["norms"] == pytest.approx({"1": math.sqrt(2), "5": 5 * math.sqrt(2)})
    assert action.grad is None
    disconnected = torch.ones(5, 2, requires_grad=True)
    assert _credit_profile(disconnected, action, mask)["norms"] == {"1": 0.0, "5": 0.0}
    assert _credit_profile(rewards, action, torch.zeros_like(mask))["norms"] == {"1": 0.0, "5": 0.0}


def test_profile_cadence_jsonl_and_unchanged_update(tmp_path):
    snapshots = []
    for every in (0, 2):
        _seed()
        trainer = toy(credit_profile_every=every, imagination_grad_decay=0.98)
        trainer.output_dir = tmp_path / str(every)
        logs = [trainer.update(), trainer.update()]
        disk = [json.loads(line) for line in
                (trainer.output_dir / "updates.jsonl").read_text().splitlines()]
        assert disk == logs
        assert "credit_profile" not in logs[0]
        assert ("credit_profile" in logs[1]) == bool(every)
        snapshots.append(([{k: v for k, v in log.items() if k not in
                            ("wall_time_s", "credit_profile")} for log in logs],
                          trainer.actor_optimizer.state_dict(), trainer.critic_optimizer.state_dict(),
                          trainer.policy.state_dict(), _rng_state(trainer.device)))
    _assert_equal(*snapshots)


def test_existing_clip_logs_preclip_norm():
    trainer = toy(imagination_grad_decay=1.0, grad_clip=1e-4)
    metrics = trainer.update()
    clipped = torch.stack([p.grad.norm() for p in trainer.actor_parameters if p.grad is not None]).norm()
    assert metrics["actor_grad_norm"] > trainer.config.grad_clip
    assert clipped <= trainer.config.grad_clip * 1.00001


def test_default_never_computes_credit(monkeypatch):
    from wms.control import dreamer

    def fail(*args):
        raise AssertionError("disabled credit profile traversed autograd")

    monkeypatch.setattr(dreamer, "_credit_profile", fail)
    assert "credit_profile" not in toy().update()


def test_override_checkpoint_exact_resume(tmp_path):
    _seed()
    trainer = toy(imagination_grad_decay=0.98, credit_profile_every=1)
    trainer.update()
    checkpoint = tmp_path / "policy.pt"
    trainer.save_checkpoint(checkpoint)
    expected = trainer.update()
    resumed = toy(imagination_grad_decay=0.98, credit_profile_every=1)
    resumed.load_checkpoint(checkpoint)
    actual = resumed.update()
    _assert_equal({k: v for k, v in expected.items() if k != "wall_time_s"},
                  {k: v for k, v in actual.items() if k != "wall_time_s"})
    _assert_equal(trainer.actor_optimizer.state_dict(), resumed.actor_optimizer.state_dict())
    assert resumed.config.imagination_grad_decay == 0.98
    assert resumed.world_models[0].grad_decay_per_step == 0.9


def test_cli_v6_smoke_settings_and_resume(monkeypatch, tmp_path):
    from wms.control import train_cli
    from wms.control.start_states import split_start_states
    from .test_c4_dreamer import (FakeCritic, FakeGroupEncoder, FakePolicy, _dataset,
                                  observations, safety, speed_reward)

    monkeypatch.setattr(train_cli, "_load_components", lambda: {
        "policy": FakePolicy(), "group_encoder": FakeGroupEncoder(), "critic": FakeCritic(),
        "observation_builder": observations, "reward_fn": speed_reward, "safety": safety})
    monkeypatch.setattr(train_cli, "_load_world_models", lambda paths, device: [DampedToy()])
    monkeypatch.setattr(train_cli, "_load_starts", lambda data, models, seed, entries=False:
                        split_start_states(_dataset(), seed=seed))
    seen = []

    def fit(trainer, max_updates, **kwargs):
        seen.append((torch.get_num_threads(), trainer.config.credit_profile_every,
                     trainer.config.imagination_grad_decay, trainer.config.batch_size,
                     trainer.config.horizon))
        trainer.update_count = max_updates
        trainer.save_checkpoint(trainer.output_dir / "policy_latest.pt")
        return {}

    monkeypatch.setattr(train_cli.DreamerTrainer, "fit", fit)
    args = ["--config", "configs/c4_policy_v6.toml", "--smoke", "--wm-ckpt",
            str(tmp_path / "fake.pt"), "--output-dir", str(tmp_path)]
    assert train_cli.main(args) == 0
    assert train_cli.main(args + ["--resume", str(tmp_path / "policy_latest.pt")]) == 0
    assert seen == [(2, 1, 0.98, 2, 4)] * 2


def test_real_wm_v6_two_tiny_updates(capsys):
    """Synthetic integration only: real WM/bridge/entries/CF, no external checkpoint/data."""
    from .test_c4_srcgrad import _trainer as real_trainer
    from .test_c4_srcwait_imagination import Starts, _world_model

    model = _world_model()
    model.grad_decay_per_step = 0.9
    trainer = real_trainer(model, Starts(True), grad=True, critic_backlog=True,
                           counterfactual=True, horizon=4)
    raw = _read_config("configs/c4_policy_v6.toml")
    trainer.config = replace(DreamerConfig(**raw["trainer"]), batch_size=2, horizon=4,
                             credit_profile_every=1, checkpoint_every=0, validation_every=0,
                             entry_pad_max=8)
    trainer.policy.set_accel_bounds(trainer.policy.accel_min, raw["policy"]["accel_max_mps2"])
    for _ in range(2):
        metrics = trainer.update()
        assert all(math.isfinite(v) for v in metrics.values() if isinstance(v, float))
        assert all(math.isfinite(v) for v in metrics["credit_profile"]["norms"].values())
        assert trainer.world_models[0].grad_decay_per_step == 0.9
        with capsys.disabled():
            print(json.dumps({k: metrics[k] for k in
                              ("update", "returns", "actor_grad_norm", "credit_profile")}))
