"""Sampled futures (FUTURES_NOTES.md): N=1 bit-identity, CRN pairing, shapes, members x samples."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wms.control import futures
from wms.control.contract import PHYSICS_DT_S, STEPS_PER_CONTROL
from wms.control.counterfactual import CounterfactualConfig, build_variants
from wms.control.dreamer import DreamerConfig, DreamerTrainer, ensemble_penalty
from wms.control.futures import FuturesConfig, NormalDraws, SampledFutures, sample_driver_encodings
from wms.control.start_states import StartBatch


def _helpers(name):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_futures_helpers_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D = _helpers("test_c4_dreamer")
P2 = _helpers("test_c4_policy2")


# ------------------------------------------------------------------ fake WM with the sampling API
class SamplingWM(D.FakeWorldModel):
    """FakeWorldModel whose HDV response depends on encoded driver params and accel noise."""

    def __init__(self, hdv_bias=0.0, base_gain=0.25, sigma=0.5):
        super().__init__(hdv_bias=hdv_bias)
        self.base_gain, self.sigma = base_gain, sigma
        self.rollouts = []

    def encode_drivers(self, state, **kwargs):
        values = torch.full(state["s_m"].shape, self.base_gain)
        return {"uncertainty": torch.full_like(values[..., None], 0.1)}, None, {"gain": values}

    def rollout(self, state, *, n_steps, cav_controller, driver_params=None, accel_noise=None,
                accel_noise_sigma_max=None, driver_embedding=None, **kwargs):
        gain = self.base_gain if driver_params is None else driver_params["gain"]
        self.rollouts.append({"driver_params": driver_params, "accel_noise": accel_noise,
                              "sigma_max": accel_noise_sigma_max, "kwargs": kwargs})
        current = dict(state)
        traces = {key: [] for key in ("positions", "speeds", "lanes", "masks", "accel")}
        for t in range(n_steps):
            command = cav_controller(current)
            cav = current["present"] & current["is_cav"]
            hdv_accel = gain * (current["v_mps"][:, :1] - current["v_mps"]) + self.hdv_bias
            if accel_noise is not None:
                hdv_accel = hdv_accel + accel_noise[:, t] * min(self.sigma, accel_noise_sigma_max)
            speed = current["v_mps"] + PHYSICS_DT_S * hdv_accel
            position = current["s_m"] + PHYSICS_DT_S * speed
            nxt = {**current,
                   "s_m": torch.where(cav, command[..., 0], position),
                   "lane": torch.where(cav, command[..., 1].long(), current["lane"]),
                   "v_mps": torch.where(cav, command[..., 2], speed),
                   "a_mps2": torch.where(cav, command[..., 3], hdv_accel)}
            for name in ("s_m", "v_mps", "a_mps2"):
                nxt[name] = torch.where(current["present"], nxt[name], current[name])
            for out, name in (("positions", "s_m"), ("speeds", "v_mps"), ("lanes", "lane"),
                              ("masks", "present"), ("accel", "a_mps2")):
                traces[out].append(nxt[name])
            current = nxt
        return {**{key: torch.stack(seq, dim=1) for key, seq in traces.items()}, "state": current}


class PlainSampler:
    def sample(self, batch_size, device="cpu"):
        return StartBatch(state=D._state(batch_size), rollout_kwargs={})

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass


def _trainer(ensemble=1, hdv_reward=True, **overrides):
    settings = dict(batch_size=2, horizon=3, entropy_weight=0.0, uncertainty_weight=0.5,
                    checkpoint_every=0, validation_every=0, actor_lr=1e-3, critic_lr=1e-3,
                    ttc_budget=0.1, dual_lr=0.2, target_tau=0.25, return_ema=0.9)
    settings.update(overrides)
    return DreamerTrainer(
        [SamplingWM(hdv_bias=0.1 * i) for i in range(ensemble)], D.FakePolicy(),
        D.FakeGroupEncoder(), D.FakeCritic(), PlainSampler(), observation_builder=D.observations,
        reward_fn=D.hdv_reward if hdv_reward else D.speed_reward, safety=D.safety,
        config=DreamerConfig(**settings), device="cpu")


# ------------------------------------------------------------------ config
def test_config_defaults_off_and_validation():
    cfg = DreamerConfig()
    assert cfg.future_samples == 1 and not cfg.future_accel_noise
    assert not FuturesConfig.from_trainer(cfg).active
    assert FuturesConfig(n_samples=2).active and FuturesConfig(accel_noise=True).active
    for bad in (dict(future_samples=0), dict(future_samples=True), dict(future_samples=2.0),
                dict(future_samples=2, future_driver_scale=0.0),
                dict(future_driver_scale=-1.0), dict(future_accel_sigma_max=math.inf)):
        with pytest.raises(ValueError):
            DreamerConfig(**bad)
    DreamerConfig(future_samples=3, future_driver_scale=0.0, future_accel_noise=True)


# ------------------------------------------------------------------ N = 1 is bit-identical
def test_n1_is_the_unchanged_single_deterministic_rollout():
    torch.manual_seed(3)
    base = _trainer(ensemble=2).imagine(PlainSampler().sample(2))
    torch.manual_seed(3)
    trainer = _trainer(ensemble=2, future_samples=1, future_seed=99)
    imagined = trainer.imagine(PlainSampler().sample(2))
    for key in ("rewards", "returns", "values", "disagreement", "actions", "entropy"):
        assert torch.equal(base[key], imagined[key]), key
    assert imagined["rewards"].shape[0] == 2 and imagined["n_futures"] == 1
    for model in trainer.world_models:       # no sampling arguments reach the WM
        assert all(r["driver_params"] is None and r["accel_noise"] is None for r in model.rollouts)
    # whole updates agree as well (metrics are JSON-identical apart from the wall time)
    torch.manual_seed(4)
    a = _trainer(ensemble=2).update()
    torch.manual_seed(4)
    b = _trainer(ensemble=2, future_samples=1).update()
    a.pop("wall_time_s"), b.pop("wall_time_s")
    assert a == b and "future_samples" not in a


# ------------------------------------------------------------------ sample draws / CRN
def test_driver_samples_are_shared_across_members_and_start_level():
    models = [SamplingWM(base_gain=0.25), SamplingWM(base_gain=0.5)]
    state = D._state(2)
    draws = NormalDraws(torch.Generator().manual_seed(0))
    kw = dict(driver_noise_scale=0.1)            # small enough that the +/-15 % cap never binds
    first = sample_driver_encodings(models[0], state, {}, 4, draws, **kw)["driver_params"]["gain"]
    second = sample_driver_encodings(models[1], state, {}, 4, draws, **kw)["driver_params"]["gain"]
    assert first.shape == second.shape == (4, 2, 3)
    # same standard-normal draws, member-specific values (uncertainty 0.1 x scale 0.1)
    z1 = (first - 0.25) / 0.01
    z2 = (second - 0.5) / 0.01
    torch.testing.assert_close(z1, z2, atol=1e-5, rtol=0)
    assert not torch.equal(first[0], first[1])
    wide = sample_driver_encodings(models[0], state, {}, 64, NormalDraws(torch.Generator().manual_seed(1)),
                                   driver_noise_scale=10.0)["driver_params"]["gain"]
    assert (wide >= 0.25 * 0.85 - 1e-7).all() and (wide <= 0.25 * 1.15 + 1e-7).all()  # +/-15 % cap
    assert torch.isclose(wide, torch.tensor(0.25 * 1.15)).any()         and torch.isclose(wide, torch.tensor(0.25 * 0.85)).any()
    with pytest.raises(ValueError):
        sample_driver_encodings(models[0], state, {}, 2, draws, max_rel_dev=1.0)
    off = sample_driver_encodings(models[0], state, {}, 3, NormalDraws(torch.Generator().manual_seed(0)),
                                  driver_noise_scale=0.0)["driver_params"]["gain"]
    assert torch.equal(off, torch.full((3, 2, 3), 0.25))


def test_plan_crn_across_members_reproducibility_and_no_global_rng():
    models = [SamplingWM(base_gain=0.25), SamplingWM(base_gain=0.4)]
    cfg = FuturesConfig(n_samples=3, accel_noise=True, accel_noise_scale=2.0, accel_sigma_max=0.7)
    state = D._state(2)
    torch.manual_seed(1)
    before = torch.get_rng_state()
    plan = SampledFutures(cfg, seed=5).plan(models, state, {}, 6)
    assert torch.equal(torch.get_rng_state(), before)            # global stream untouched
    assert len(plan) == 2 and all(len(per) == 3 for per in plan)
    for s in range(3):
        assert torch.equal(plan[0][s]["accel_noise"], plan[1][s]["accel_noise"])   # CRN across members
        assert plan[0][s]["accel_noise"].shape == (2, 6, 3)
        assert plan[0][s]["accel_noise_sigma_max"] == 0.7
    assert not torch.equal(plan[0][0]["accel_noise"], plan[0][1]["accel_noise"])
    again = SampledFutures(cfg, seed=5).plan(models, state, {}, 6)
    other = SampledFutures(cfg, seed=6).plan(models, state, {}, 6)
    assert torch.equal(plan[1][2]["accel_noise"], again[1][2]["accel_noise"])
    assert not torch.equal(plan[1][2]["accel_noise"], other[1][2]["accel_noise"])


def test_crn_across_counterfactual_variants_and_members():
    trainer = _trainer(ensemble=2, future_samples=3, future_accel_noise=True)
    batch = StartBatch(P2.state([-300.0, -250.0, -200.0, -150.0], [0, 1, 0, 1],
                                cav=[True, True, False, True]), {})
    batch = StartBatch({k: v.expand(2, -1).clone() for k, v in batch.state.items()}, {})
    variants = build_variants(batch.state, CounterfactualConfig(enabled=True, n_individual=2,
                                                               n_group=1, n_flip=1))
    assert variants.n_rows > 2
    imagined = trainer.imagine(batch, variants=variants, future_seed=11)
    index = variants.row_start.tolist()
    sample = 0
    for model in trainer.world_models:
        assert len(model.rollouts) == 3
        for roll in model.rollouts:
            noise, gain = roll["accel_noise"], roll["driver_params"]["gain"]
            assert noise.shape[0] == gain.shape[0] == variants.n_rows
            for row, start in enumerate(index):                  # every row of a start shares it
                assert torch.equal(noise[row], noise[start])
                assert torch.equal(gain[row], gain[start])
        sample += 1
    # the same sample index sees identical draws in both members; different samples differ
    for s in range(3):
        assert torch.equal(trainer.world_models[0].rollouts[s]["accel_noise"],
                           trainer.world_models[1].rollouts[s]["accel_noise"])
    assert not torch.equal(trainer.world_models[0].rollouts[0]["accel_noise"],
                           trainer.world_models[0].rollouts[1]["accel_noise"])
    assert imagined["returns"].shape == (2 * 3, 3, variants.n_rows)


# ------------------------------------------------------------------ shapes, averaging, penalty
def test_returns_are_averaged_over_members_and_samples_and_penalty_uses_both():
    trainer = _trainer(ensemble=2, future_samples=3, future_accel_noise=True)
    imagined = trainer.imagine(PlainSampler().sample(2), future_seed=1)
    assert imagined["n_members"] == 2 and imagined["n_futures"] == 3
    for key in ("rewards", "costs", "returns", "entropy"):
        assert imagined[key].shape == (6, 3, 2), key
    assert imagined["actions"].shape[:3] == (6, 3, 2)
    assert len(imagined["states"]) == 6
    # member-major ordering: pseudo-member e*S+s uses member e, sample s
    for e, model in enumerate(trainer.world_models):
        assert len(model.rollouts) == 3
    rewards = imagined["rewards"]
    torch.testing.assert_close(imagined["disagreement"], ensemble_penalty(rewards, 1.0))
    assert imagined["disagreement"].shape == (3, 2) and (imagined["disagreement"] > 0).all()
    # one member, several samples: the penalty comes from the samples alone
    single = _trainer(ensemble=1, future_samples=4, future_accel_noise=True)
    out = single.imagine(PlainSampler().sample(2), future_seed=1)
    assert out["rewards"].shape == (4, 3, 2) and (out["disagreement"] > 0).all()
    # ...and vanishes when the samples are identical copies of the world
    same = _trainer(ensemble=1, future_samples=1, future_accel_noise=True, future_driver_scale=0.0)
    out = same.imagine(PlainSampler().sample(2), future_seed=1)
    assert out["rewards"].shape == (1, 3, 2) and (out["disagreement"] == 0).all()
    # the actor objective averages over the whole pseudo-member axis
    scale = trainer._return_scale(imagined["returns"])
    loss, _ = trainer.actor_objective(imagined, scale, {"all": 1.0, "individual": 0.0, "group": 0.0,
                                                        "coop": 0.0, "team": 0.0})
    torch.testing.assert_close(loss, -imagined["returns"].mean() / scale)


def test_update_trains_with_samples_and_logs_them():
    trainer = _trainer(ensemble=2, future_samples=2, future_accel_noise=True)
    metrics = trainer.update()
    D._assert_finite_metrics(metrics)
    assert metrics["future_samples"] == 2 and metrics["ensemble_disagreement"] > 0
    assert trainer.validate(PlainSampler()) == trainer.validate(PlainSampler())   # fixed validation worlds


# ------------------------------------------------------------------ real WM-S: accel noise, checkpointing
def _real_wm(mode="hybrid", seed=1):
    from wms.model.spec import ModelSpec
    from wms.model.wms import WmSModel

    torch.manual_seed(seed)
    return WmSModel(ModelSpec(rollout_mode=mode)).eval().requires_grad_(False)


def test_real_wm_accel_noise_is_off_by_default_and_scaled_by_sigma_cap():
    model = _real_wm()
    x = P2.state([-300.0, -260.0, -220.0, -180.0], [0, 0, 1, 1], cav=[True, False, False, False])
    zero = model.rollout(dict(x), n_steps=4, cav_teacher=None,
                         cav_controller=lambda s: torch.stack(
                             (s["s_m"], s["lane"].float(), s["v_mps"], torch.zeros_like(s["s_m"])), -1))
    controller = lambda s: torch.stack(  # noqa: E731
        (s["s_m"], s["lane"].float(), s["v_mps"], torch.zeros_like(s["s_m"])), -1)
    again = model.rollout(dict(x), n_steps=4, cav_teacher=None, cav_controller=controller)
    assert torch.equal(zero["accel"], again["accel"])
    z = torch.randn(1, 4, 4)
    noisy = model.rollout(dict(x), n_steps=4, cav_teacher=None, cav_controller=controller,
                          accel_noise=z, accel_noise_sigma_max=0.0)       # sigma capped to 0
    assert torch.equal(noisy["accel"], again["accel"])
    noisy = model.rollout(dict(x), n_steps=4, cav_teacher=None, cav_controller=controller,
                          accel_noise=z, accel_noise_sigma_max=1.0)
    assert not torch.equal(noisy["accel"], again["accel"])
    assert torch.equal(noisy["accel"][..., 0], again["accel"][..., 0])    # CAV slot untouched


class Starts2:
    def __init__(self, x):
        self.x = x

    def sample(self, batch_size, device="cpu"):
        return StartBatch({k: v.expand(batch_size, -1).clone() for k, v in self.x.items()}, {})


def _real_trainer(models, hidden=16, **cfg):
    from wms.control.bridge import make_cav_controller
    from wms.control.lanes import LaneRules
    from wms.control.obs import build_observations
    from wms.control.policy import CavPolicy, RsuGroupEncoder, TeamCritic
    from wms.control.reward import team_reward
    from wms.control.safety import safe_accel
    from wms.control.train_cli import GroupSignal

    x = P2.state([-900.0, -860.0, -820.0, -700.0, -880.0, -790.0], [0, 1, 0, 1, 1, 1],
                 v=[25.0, 24.0, 26.0, 22.0, 20.0, 23.0], cav=[True, True, True, False, False, True])
    settings = dict(batch_size=2, horizon=3, checkpoint_every=0, validation_every=0,
                    actor_lr=1e-3, critic_lr=1e-3)
    settings.update(cfg)
    return DreamerTrainer(
        models, CavPolicy(hidden_dim=hidden), GroupSignal(RsuGroupEncoder(hidden_dim=hidden)),
        TeamCritic(hidden_dim=hidden), Starts2(x), observation_builder=build_observations,
        reward_fn=team_reward, safety=safe_accel, controller_factory=make_cav_controller,
        config=DreamerConfig(**settings), lane_rules=LaneRules(),
        counterfactual=CounterfactualConfig(enabled=True, n_individual=2, n_group=1, n_flip=1,
                                            group_max_size=4))


@pytest.mark.parametrize("checkpointing", [True, False])
def test_real_wm_trainer_with_samples_checkpointing_equivalence(checkpointing):
    torch.manual_seed(7)
    trainer = _real_trainer([_real_wm()], future_samples=2, future_accel_noise=True,
                            checkpoint_imagination=checkpointing)
    metrics = trainer.update()
    assert metrics["future_samples"] == 2 and math.isfinite(metrics["actor_grad_norm"])
    assert metrics["actor_grad_norm"] > 0
    # gradients of the same sampled objective are identical with and without recomputation
    grads = []
    for flag in (True, False):
        torch.manual_seed(8)
        t = _real_trainer([_real_wm()], future_samples=2, future_accel_noise=True,
                          checkpoint_imagination=flag, entropy_weight=0.0)
        batch = t.start_states.sample(2)
        variants = build_variants(batch.state, t.counterfactual, torch.Generator().manual_seed(0))
        imagined = t.imagine(batch, variants=variants, future_seed=3)
        scale = t._return_scale(imagined["returns"])
        loss, _ = t.actor_objective(imagined, scale, t.counterfactual.weights(1500))
        t.actor_optimizer.zero_grad()
        loss.backward()
        grads.append(torch.cat([p.grad.flatten() for p in t.actor_parameters if p.grad is not None]))
    torch.testing.assert_close(grads[0], grads[1], rtol=1e-4, atol=1e-6)


def test_rsu_driver_samples_unchanged_by_factoring():
    """The RSU check's draws equal the pre-refactor inline implementation (same generator stream)."""
    from wms.control.contract import RewardWeights
    from wms.control.rsu_check import RsuChecker

    model = _real_wm()
    x = P2.state([-300.0, -260.0, -220.0, -180.0, -140.0], [0, 0, 1, 1, 0])
    history = {"history": torch.ones(1, 5, model.history_len, model.history_features)}
    history["history_mask"] = torch.ones_like(history["history"][..., 0], dtype=torch.bool)
    checker = RsuChecker([model], {"policy": lambda: None}, lambda *a: None, RewardWeights(),
                         n_driver_samples=4, driver_noise_scale=0.7, embedding_noise_std=0.05, seed=3,
                         driver_max_rel_dev=0.5)   # the pre-2026-10-13 cap, for the golden
    got = checker._sample_drivers(model, x, history, 2, {})

    generator = torch.Generator().manual_seed(3)             # the original inline code
    encoded, embedding, params = model.encode_drivers(dict(x), **history)
    names = tuple(params)
    values = torch.stack([params[n] for n in names], -1)
    draws = torch.randn((4, *values.shape[1:]), generator=generator)
    scale = encoded["uncertainty"]
    samples = values + draws * scale * 0.7
    samples = torch.maximum(torch.minimum(samples, values * 1.5), values * 0.5).repeat(2, 1, 1)
    want = {"driver_params": {n: samples[..., k] for k, n in enumerate(names)}}
    draws = torch.randn((4, *embedding.shape[1:]), generator=generator)
    want["driver_embedding"] = (embedding + draws * 0.05 * 0.7).repeat(2, 1, 1)
    for name in names:
        assert torch.equal(got["driver_params"][name], want["driver_params"][name])
    assert torch.equal(got["driver_embedding"], want["driver_embedding"])
    assert got["driver_embedding"].shape[0] == 8


def test_physics_constants_assumed_by_the_tests():
    assert STEPS_PER_CONTROL == 2 and SimpleNamespace  # horizon 3 -> 6 physics steps
    assert futures.seed_of("a", 1) == futures.seed_of("a", 1) != futures.seed_of("a", 2)


def test_rsu_checker_defaults_to_point_estimate_drivers():
    """HORIZON_CHECK 2026-10-13: the encoder uncertainty is untrained, so the deployed check imagines
    with the point-estimate drivers (one sample); duplicate noiseless samples are refused."""
    import inspect
    from wms.control.contract import RewardWeights
    from wms.control.rsu_check import RsuChecker
    sig = inspect.signature(RsuChecker.__init__).parameters
    assert sig["n_driver_samples"].default == 1
    assert sig["driver_noise_scale"].default == 0.0 and sig["embedding_noise_std"].default == 0.0
    with pytest.raises(ValueError, match="identical"):
        RsuChecker([SamplingWM(base_gain=0.25)], {"policy": lambda: None}, lambda *a: None,
                   RewardWeights(), n_driver_samples=4)
    model = SamplingWM(base_gain=0.25)
    checker = RsuChecker([model], {"policy": lambda: None}, lambda *a: None, RewardWeights())
    drivers = checker._sample_drivers(model, D._state(1), {}, 3, {})
    assert torch.equal(drivers["driver_params"]["gain"], torch.full((3, 3), 0.25))
