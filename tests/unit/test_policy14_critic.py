"""POLICY14: synthetic CPU-only real-return regression and actor-credit audits."""
from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from wms.control.contract import PHYSICS_DT_S
from wms.control.dreamer import DreamerConfig, DreamerTrainer
from wms.control.real_returns14 import (
    ACCOUNTING, RETURN_UNITS, TERMINAL_SEMANTICS, NPZRealReturnProvider,
    RealReturnBatch, ScaledTeamCritic, warm_start_real_critic, write_real_returns_npz,
)


def _helpers(name):
    spec = importlib.util.spec_from_file_location(
        f"_policy14_{name}", Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D = _helpers("test_c4_dreamer")


@pytest.fixture(autouse=True)
def cpu_state():
    rng, threads = torch.get_rng_state(), torch.get_num_threads()
    torch.manual_seed(14)
    torch.set_num_threads(1)
    yield
    torch.set_rng_state(rng)
    torch.set_num_threads(threads)


def _negative_j_reward(state, next_state, accel, weights):
    # Synthetic authoritative cost: sum over ALL vehicles, not mean over CAVs.
    # A differentiable occupancy proxy makes actor credit observable in this test.
    del state, accel, weights
    cost = PHYSICS_DT_S * ((1 + 0.01 * next_state["v_mps"]) * next_state["present"]).sum(-1)
    return {"reward": -cost, "cost_ttc": cost * 0.1}


def _trainer(*, provider=None, critic=None, **overrides):
    settings = dict(batch_size=2, horizon=2, gamma=1.0, lambda_=1.0,
                    checkpoint_every=0, validation_every=0, actor_lr=1e-3, critic_lr=1e-3,
                    uncertainty_weight=0.0, entropy_weight=0.0)
    settings.update(overrides)
    return DreamerTrainer(
        [D.FakeWorldModel()], D.FakePolicy(), D.FakeGroupEncoder(),
        D.FakeCritic() if critic is None else critic,
        D.FakeSampler(), observation_builder=D.observations, reward_fn=_negative_j_reward,
        safety=D.safety, config=DreamerConfig(**settings), real_return_provider=provider)


class Provider:
    def __init__(self, batch=None):
        self.batch = batch or RealReturnBatch(D._state(3), torch.tensor([-20., -40., -60.]), 1.0)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append((kwargs, torch.is_grad_enabled()))
        return self.batch


@pytest.mark.parametrize("value", [-0.1, 1.1, math.nan, math.inf, -math.inf, True, False,
                                  "0.2", None, [], torch.tensor(0.2)])
def test_mix_validation(value):
    with pytest.raises(ValueError, match="real_return_mix"):
        DreamerConfig(real_return_mix=value)


@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf, -math.inf, True, False,
                                  "10000", None, [], torch.tensor(1.0)])
def test_real_return_scale_validation(value):
    with pytest.raises(ValueError, match="real_return_scale"):
        DreamerConfig(real_return_scale=value)


@pytest.mark.parametrize("value", [0, 1, 0.5, None, "false", [], math.nan])
def test_audit_boolean_validation(value):
    with pytest.raises(ValueError, match="actor_gradient_audit"):
        DreamerConfig(actor_gradient_audit=value)


def test_switch_defaults_and_valid_endpoints():
    assert DreamerConfig().real_return_mix == 0.0
    assert DreamerConfig().real_return_scale == 1.0
    assert DreamerConfig().actor_gradient_audit is False
    for mix in (0, 0.4, 1):
        assert DreamerConfig(real_return_mix=mix, actor_gradient_audit=True).real_return_mix == mix


@pytest.mark.parametrize("cavs,present,expected", [(1, 10, 60), (1, 9, 90), (0, 0, 60),
                                                 (0, 10, 60), (2, 20, 60), (3, 20, 90)])
def test_observed_start_penetration_cap_ignores_metadata_and_absent_cavs(cavs, present, expected):
    from wms.control.start_states import StartBatch
    trainer = _trainer(horizon=90, low_penetration_horizon_cap=True)
    state = D._state(1)
    state["present"] = torch.arange(21)[None] < present
    state["is_cav"] = (torch.arange(21)[None] < cavs) | (torch.arange(21)[None] >= present)
    batch = StartBatch(state, metadata=[{"world": "synthetic_p40"}])
    assert trainer._imagination_config(batch).horizon == expected
    assert trainer.config.horizon == 90
    trainer.config.low_penetration_horizon_cap = False
    assert trainer._imagination_config(batch).horizon == 90


def test_low_penetration_cap_applies_to_rollout_and_preserves_pairing():
    from wms.control.start_states import StartBatch
    trainer = _trainer(horizon=90, low_penetration_horizon_cap=True)
    # Zero-CAV start: observed zero share, despite a misleading config-world label.
    state = D._state(1)
    state["is_cav"].zero_()
    batch = StartBatch(state, metadata=[{"world": "synthetic_p40"}])
    with torch.no_grad():
        imagined = trainer.imagine(batch, deterministic=True)
    assert imagined["rewards"].shape[1] == 60
    assert imagined["imagination_horizon_s"] == 60.
    assert len(trainer.world_models[0].commands) == 120
    assert trainer.config.horizon == 90


@pytest.mark.parametrize("mode", ["absolute", "native_relative"])
def test_wm_safety_report_only_has_no_actor_gradient_or_dual_update(mode):
    trainer = _trainer(wm_safety_report_only=True, ttc_budget_mode=mode,
                       lagrange_init=.5, lagrange_max=2., lagrange_decay=.1)
    costs = torch.tensor([[[20.]]], requires_grad=True)
    imagined = {"costs": costs, "n_base": 1}
    penalty, dual, budget, metrics = trainer._safety_terms(imagined, costs.mean())
    assert penalty == 0 and not penalty.requires_grad
    assert metrics["wm_safety_report_only"] and metrics["ttc_penalty_report"] > 0
    assert trainer.update_lagrange(dual, budget) == .5
    assert not trainer._lagrange_capped
    trainer.config.wm_safety_report_only = False
    penalty, _, _, _ = trainer._safety_terms(imagined, costs.mean())
    assert penalty > 0 and torch.autograd.grad(penalty, costs)[0].abs().sum() > 0


def test_provider_is_explicit_and_disabled_provider_never_runs():
    with pytest.raises(ValueError, match="requires a real_return_provider"):
        _trainer(real_return_mix=0.1)
    with pytest.raises(ValueError, match="callable"):
        _trainer(provider=object())
    def unused(**kwargs):
        pytest.fail("default-off provider was called")
    torch.manual_seed(23)
    legacy = _trainer()
    expected = legacy.update()
    torch.manual_seed(23)
    explicit = _trainer(provider=unused, real_return_mix=0.0, real_return_scale=1.0,
                        actor_gradient_audit=False)
    actual = explicit.update()
    assert "actor_grad_norms" not in actual and "real_return_mix" not in actual
    expected.pop("wall_time_s")
    actual.pop("wall_time_s")
    assert actual == expected
    for name in ("policy", "group_encoder", "critic", "target_critic"):
        D._assert_equal(getattr(legacy, name).state_dict(), getattr(explicit, name).state_dict())
    D._assert_equal(legacy.actor_optimizer.state_dict(), explicit.actor_optimizer.state_dict())
    D._assert_equal(legacy.critic_optimizer.state_dict(), explicit.critic_optimizer.state_dict())


@pytest.mark.parametrize("mix", [0.25, 1.0])
@pytest.mark.parametrize("scale,floor", [(7.0, 1.0), (7.0, 10000.0), (20000.0, 10000.0)])
def test_real_mix_uses_raw_units_scale_floor_and_independent_mean(mix, scale, floor):
    provider = Provider()
    trainer = _trainer(provider=provider, real_return_mix=mix, real_return_scale=floor)
    trainer._return_scale = lambda returns: scale
    real_prediction = trainer.critic(provider.batch.states).detach()
    expected_real = ((real_prediction - provider.batch.returns) / max(scale, floor)).square().mean()
    observed = {}
    imagine = trainer.imagine
    def capture(*args, **kwargs):
        result = imagine(*args, **kwargs)
        # The actor update changes neither these rollout features nor critic weights.
        prediction = torch.stack([torch.stack([trainer.critic({k: v.detach() for k, v in s.items()})
                                               for s in states[:-1]])
                                  for states in result["states"]])
        observed["imagined_loss"] = float(((prediction - result["returns"]) / max(scale, floor)).square().mean().detach())
        return result
    trainer.imagine = capture
    target_before = copy.deepcopy(trainer.target_critic.state_dict())
    critic_before = copy.deepcopy(trainer.critic.state_dict())
    metrics = trainer.update()
    assert provider.calls == [({"batch_size": 2, "device": torch.device("cpu"), "update": 1}, False)]
    assert metrics["critic_real_loss"] == pytest.approx(float(expected_real))
    expected = mix * float(expected_real) + (1 - mix) * observed["imagined_loss"]
    assert metrics["critic_loss"] == pytest.approx(expected)
    assert metrics["real_return_units"] == RETURN_UNITS
    assert metrics["real_return_count"] == 3
    assert metrics["real_return_mean"] == -40.0
    assert metrics["critic_imagined_loss"] == (None if mix == 1 else observed["imagined_loss"])
    assert trainer.critic.bias < critic_before["bias"]
    for key, value in trainer.target_critic.state_dict().items():
        tau = trainer.config.target_tau
        torch.testing.assert_close(value, (1 - tau) * target_before[key]
                                   + tau * trainer.critic.state_dict()[key])


def test_real_provider_and_graph_bearing_targets_cannot_add_actor_or_wm_gradients():
    torch.manual_seed(99)
    baseline = _trainer()
    baseline.update()
    torch.manual_seed(99)
    provider = Provider()
    trainer = _trainer(provider=provider, real_return_mix=1.0)
    # Explicit graph creation even though the provider will be called under no_grad.
    state = D._state(3)
    source = torch.tensor(0.1, requires_grad=True)
    state["v_mps"] = state["v_mps"] + source + trainer.policy.mean
    target = -torch.arange(1., 4.) * 20 + source + trainer.group_encoder.signal
    provider.batch = RealReturnBatch(state, target, 1.0)
    seen = []
    hook = trainer.critic.register_forward_pre_hook(
        lambda module, args: seen.append({k: v.requires_grad for k, v in args[0].items()}))
    trainer.update()
    hook.remove()
    assert source.grad is None
    assert seen and not any(any(flags.values()) for flags in seen)
    for name in ("policy", "group_encoder"):
        D._assert_equal(getattr(baseline, name).state_dict(), getattr(trainer, name).state_dict())
        for left, right in zip(getattr(baseline, name).parameters(), getattr(trainer, name).parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
    assert all(p.grad is None for model in trainer.world_models for p in model.parameters())
    assert all(p.grad is None for p in trainer.target_critic.parameters())


@pytest.mark.parametrize("replacement,match", [
    ({"returns": torch.tensor([1., -2.])}, "nonpositive"),
    ({"returns": torch.tensor([math.nan, -2.])}, "finite"),
    ({"returns": torch.tensor([-math.inf, -2.])}, "finite"),
    ({"returns": torch.tensor([[-1.], [-2.]])}, r"\[B\]"),
    ({"returns": torch.tensor([-1, -2])}, "floating"),
    ({"returns": torch.tensor([])}, "nonempty"),
    ({"states": {}}, "nonempty mapping"),
    ({"states": {"v_mps": torch.ones(3, 2)}}, "batch dimension"),
    ({"states": {"v_mps": torch.tensor(2.)}}, "batch dimension"),
    ({"states": {"v_mps": [[1.], [2.]]}}, "batch dimension"),
    ({"gamma": 0.99}, "gamma"),
    ({"gamma": True}, "gamma"),
    ({"gamma": math.nan}, "gamma"),
    ({"units": "per_vehicle_seconds"}, "units"),
])
def test_invalid_real_batches_fail_before_any_optimizer_step(replacement, match):
    fields = dict(states=D._state(2), returns=torch.tensor([-1., -2.]), gamma=1.0)
    fields.update(replacement)
    trainer = _trainer(provider=Provider(RealReturnBatch(**fields)), real_return_mix=0.5)
    before = copy.deepcopy(trainer.policy.state_dict())
    with pytest.raises(ValueError, match=match):
        trainer.update()
    D._assert_equal(before, trainer.policy.state_dict())
    assert trainer.update_count == 0 and not trainer.actor_optimizer.state


def test_provider_must_return_explicit_batch_and_gamma_can_be_discounted():
    trainer = _trainer(provider=lambda **kw: ({}, torch.tensor([-2.])), real_return_mix=1.0)
    with pytest.raises(ValueError, match="RealReturnBatch"):
        trainer.update()
    provider = Provider(RealReturnBatch(D._state(2), torch.tensor([-10., -15.]), 0.99))
    trainer = _trainer(provider=provider, real_return_mix=0.5, gamma=0.99)
    assert math.isfinite(trainer.update()["critic_real_loss"])


def test_component_audit_hand_computed_weighted_actor_norms():
    def reward(state, nxt, accel, weights):
        result = D.speed_reward(state, nxt, accel, weights)
        result["cost_ttc"] = nxt["v_mps"][:, 0] / 20
        return result
    trainer = D._trainer(horizon=1, gamma=0.9, lambda_=1.0, actor_gradient_audit=True,
                         lagrange_init=2.0, lagrange_scaled=True, entropy_weight=0.25,
                         reward_fn=reward)
    imagine = trainer.imagine
    trainer.imagine = lambda *args, **kwargs: imagine(*args, deterministic=True, **kwargs)
    trainer._return_scale = lambda returns: 2.0
    metrics = trainer.update()
    norms = metrics["actor_grad_norms"]
    assert set(norms) == {"reward", "critic", "constraint", "entropy"}
    assert norms["reward"] == pytest.approx(math.sqrt(2) * 1.5 / 2)
    assert norms["critic"] == pytest.approx(math.sqrt(2) * 0.9 * 0.05 / 2)
    assert norms["constraint"] == pytest.approx(math.sqrt(2) * 0.075)
    assert norms["entropy"] == pytest.approx(0.25)
    assert metrics["actor_grad_norm"] == pytest.approx(
        math.sqrt(2 * (-0.75 - 0.0225 + 0.075) ** 2 + 0.25 ** 2))


@pytest.mark.parametrize("checkpoint", [False, True])
def test_audit_is_observational_with_ensemble_and_checkpointing(checkpoint):
    def run(audit):
        torch.manual_seed(111)
        trainer = D._trainer(horizon=2, ensemble=2, actor_gradient_audit=audit,
                             uncertainty_weight=0.5, checkpoint_imagination=checkpoint,
                             reward_fn=D.hdv_reward, entropy_weight=0.1)
        metrics = trainer.update()
        return trainer, metrics
    base, expected = run(False)
    audited, actual = run(True)
    assert "uncertainty" in actual["actor_grad_norms"]
    assert all(math.isfinite(v) for v in actual["actor_grad_norms"].values())
    for key in ("actor_grad_norms", "action_audit", "wall_time_s"):
        actual.pop(key)
    expected.pop("wall_time_s")
    assert actual == expected
    for name in ("policy", "group_encoder", "critic", "target_critic"):
        D._assert_equal(getattr(base, name).state_dict(), getattr(audited, name).state_dict())


def test_action_audit_uses_active_pre_safety_actions_and_physical_density():
    trainer = _trainer(actor_gradient_audit=True)
    trainer.policy.accel_min, trainer.policy.accel_max = -3.0, 2.6
    # [E,H,N,K]: row 1 is a counterfactual; last slot of row 0 is inactive.
    imagined = {"n_base": 1, "action_audit": {
        "audit_action": torch.tensor([[[[-2.99, 2.59, 0., 2.6], [2.6, 2.6, 2.6, 2.6]]]]),
        "audit_acting": torch.tensor([[[[True, True, True, False], [True] * 4]]]),
        "audit_log_prob": torch.tensor([[[[-math.log(5.6)] * 3 + [100.], [100.] * 4]]]),
    }}
    metrics = trainer._action_audit_metrics(imagined)
    assert metrics["active_actions"] == 3
    assert metrics["entropy_nats"] == pytest.approx(math.log(5.6))
    assert metrics["entropy_effective_width_mps2"] == pytest.approx(5.6)
    assert metrics["saturation_low_fraction"] == pytest.approx(1 / 3)
    assert metrics["saturation_high_fraction"] == pytest.approx(1 / 3)
    assert metrics["saturation_fraction"] == pytest.approx(2 / 3)
    assert metrics["saturation_margin_mps2"] == pytest.approx(0.056)
    imagined["action_audit"]["audit_acting"].zero_()
    empty = trainer._action_audit_metrics(imagined)
    assert empty["active_actions"] == empty["entropy_effective_width_mps2"] == 0
    assert empty["saturation_fraction"] == 0


def test_action_audit_records_pre_safety_command_in_imagination():
    trainer = _trainer(actor_gradient_audit=True)
    trainer.safety = lambda state, action: torch.zeros_like(action)
    result = trainer.imagine(trainer.start_states.sample(2), deterministic=True)
    assert result["actions"].count_nonzero() == 0
    assert result["action_audit"]["audit_action"][..., 0].count_nonzero() > 0
    assert trainer._action_audit_metrics(result)["active_actions"] == 4


def test_lane_and_counterfactual_audit_preserves_full_actor_gradient():
    from wms.control.bridge import make_cav_controller
    from wms.control.counterfactual import CounterfactualConfig
    from wms.control.lanes import LaneRules
    from wms.control.obs import build_observations
    from wms.control.policy import CavPolicy, RsuGroupEncoder, TeamCritic
    from wms.control.rsu_field import build_rsu_field
    from wms.control.safety import safe_accel

    class Group(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = RsuGroupEncoder(hidden_dim=16)

        def forward(self, obs, state, field=None):
            if isinstance(state, dict):
                return self.encoder(obs, state["s_m"], build_rsu_field(state))
            return self.encoder(obs, state, field)

    def run(audit):
        torch.manual_seed(53)
        trainer = DreamerTrainer(
            [D.FakeWorldModel()], CavPolicy(hidden_dim=16), Group(), TeamCritic(hidden_dim=16),
            D.FakeSampler(), observation_builder=build_observations, reward_fn=_negative_j_reward,
            safety=safe_accel, controller_factory=make_cav_controller, lane_rules=LaneRules(),
            counterfactual=CounterfactualConfig(enabled=True, n_individual=1, n_group=1,
                                                n_flip=1, group_state_fraction=0.0),
            config=DreamerConfig(batch_size=2, horizon=2, actor_gradient_audit=audit,
                                 checkpoint_every=0, validation_every=0))
        metrics = trainer.update()
        return trainer, metrics
    baseline, expected = run(False)
    trainer, actual = run(True)
    assert actual["actor_grad_norms"]["lane"] > 0
    assert actual["lane_decisions_with_options"] > 0
    assert actual["cf_rows"] > 0
    assert actual["action_audit"]["active_actions"] == 4
    for left, right in zip(baseline.actor_parameters, trainer.actor_parameters):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
    assert actual["actor_loss"] == expected["actor_loss"]


def test_resume_requires_reinjected_provider_and_keeps_scalar_switches(tmp_path, monkeypatch):
    monkeypatch.setattr("wms.control.dreamer.checked_path", lambda p, **kw: Path(p))
    original = _trainer(provider=Provider(), real_return_mix=0.5, actor_gradient_audit=True)
    original.update()
    path = tmp_path / "critic.pt"
    original.save_checkpoint(path)
    with pytest.raises(ValueError, match="requires a real_return_provider"):
        _trainer().load_checkpoint(path)
    resumed = _trainer(provider=Provider())
    resumed.load_checkpoint(path)
    assert resumed.config.real_return_mix == 0.5
    assert resumed.config.actor_gradient_audit is True
    expected = original.update()
    resumed.load_checkpoint(path)
    actual = resumed.update()
    expected.pop("wall_time_s")
    actual.pop("wall_time_s")
    assert expected == actual


@pytest.fixture
def npz_workspace(tmp_path, monkeypatch):
    # Exercise the real shared path guard while confining all synthetic I/O to
    # pytest's temporary root, which need not be inside the repo on Windows.
    from wms.control import real_returns14 as real
    guard = real.checked_path
    monkeypatch.setattr(real, "checked_path", lambda path: guard(path, workspace=tmp_path))
    return tmp_path


def _npz_states(n=3):
    return {**D._state(n), "entry_backlog": torch.ones(n, 2)}


def _npz_payload(n=3):
    return {**{f"state__{key}": value.numpy() for key, value in _npz_states(n).items()},
            "returns": -np.arange(1., n + 1) * 20, "gamma": np.asarray(1.0),
            "units": np.asarray(RETURN_UNITS), "split": np.asarray("TRAIN"),
            "source_kind": np.asarray("native"), "requested_population": np.arange(n) + 10,
            "horizon_s": np.full(n, 300.0), "accounting": np.asarray(ACCOUNTING),
            "terminal_semantics": np.asarray(TERMINAL_SEMANTICS)}


def _save_payload(root, data, filename="targets.npz"):
    path = root / filename
    np.savez(path, **data)
    return path


@pytest.mark.parametrize("steps", [600, 1200])
def test_npz_writer_roundtrip_charges_all_supplied_costs_without_normalization(npz_workspace, steps):
    states = _npz_states()
    # No admitted vehicles in the first window. Source costs must still count.
    states["present"][0] = False
    road = np.zeros((steps, 3))
    road[0] = [1., 2., 3.]
    road[-100:] += [1., 2., 4.]  # includes road delay of later entrants
    source = np.full((steps, 3), [0.5, 1.0, 2.0])
    population = np.array([11, 21, 31])  # exogenous, not the admitted vehicle count
    path = write_real_returns_npz(
        npz_workspace / "continuations.npz", states, road, source, population,
        source_kind="current_policy", policy_checkpoint="policy_001.pt")
    expected = -(road.sum(0) + source.sum(0))
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["returns"], expected)
        np.testing.assert_array_equal(data["horizon_s"], np.full(3, steps * 0.5))
        np.testing.assert_array_equal(data["requested_population"], population)
        np.testing.assert_array_equal(data["road_delay_steps"], road)
        np.testing.assert_array_equal(data["source_wait_steps"], source)
        assert data["physics_dt"] == 0.5 and data["gamma"] == 1
        assert data["accounting"] == ACCOUNTING
        assert data["terminal_semantics"] == TERMINAL_SEMANTICS
    provider = NPZRealReturnProvider(path, seed=7)
    assert provider.provenance["ledger_recomputed"] is True
    assert provider.provenance["policy_checkpoint"] == "policy_001.pt"
    batch = provider(batch_size=20, device=torch.device("cpu"), update=1)
    indices = np.random.default_rng(8).integers(3, size=20)
    np.testing.assert_array_equal(batch.returns.numpy(), expected[indices])
    assert batch.gamma == 1 and batch.units == RETURN_UNITS
    for name in states:
        torch.testing.assert_close(batch.states[name], states[name][indices])
        assert not batch.states[name].requires_grad


def test_npz_sampling_is_seed_plus_update_stateless_and_leaves_global_rngs_alone(npz_workspace):
    path = _save_payload(npz_workspace, _npz_payload())
    provider = NPZRealReturnProvider(path, seed=42)
    numpy_rng, torch_rng = np.random.get_state(), torch.get_rng_state()
    first = provider(batch_size=32, device="cpu", update=1)
    later = provider(batch_size=32, device="cpu", update=9)
    repeated = provider(batch_size=32, device="cpu", update=1)
    restarted = NPZRealReturnProvider(path, seed=42)(batch_size=32, device="cpu", update=9)
    torch.testing.assert_close(first.returns, repeated.returns, rtol=0, atol=0)
    torch.testing.assert_close(later.returns, restarted.returns, rtol=0, atol=0)
    assert not torch.equal(first.returns, later.returns)
    D._assert_equal(numpy_rng, np.random.get_state())
    torch.testing.assert_close(torch_rng, torch.get_rng_state(), rtol=0, atol=0)
    # Caller mutation cannot alter the provider cache.
    first.states["v_mps"].zero_()
    first.returns.zero_()
    again = provider(batch_size=32, device="cpu", update=1)
    torch.testing.assert_close(again.returns, repeated.returns, rtol=0, atol=0)
    assert again.states["v_mps"].count_nonzero() > 0
    assert provider.provenance["ledger_recomputed"] is False


@pytest.mark.parametrize("name", ["s_m", "v_mps", "a_mps2", "lane", "present", "is_cav",
                                 "length_m", "entry_backlog"])
def test_npz_validates_every_state_field_including_padding(npz_workspace, name):
    data = _npz_payload()
    key = f"state__{name}"
    # Nonfinite values in absent slots must not bypass validation.
    value = data[key].astype(float)
    value[0, -1] = math.nan
    data[key] = value
    with pytest.raises(ValueError, match=key):
        NPZRealReturnProvider(_save_payload(npz_workspace, data))
    data = _npz_payload()
    data[key] = data[key][:, :1]
    with pytest.raises(ValueError, match="state__"):
        NPZRealReturnProvider(_save_payload(npz_workspace, data))


def test_npz_accepts_present_junction_lane_minus_one_as_team_critic_input(npz_workspace):
    data = _npz_payload()
    data["state__lane"][:, 0] = -1
    assert data["state__present"][:, 0].all()
    provider = NPZRealReturnProvider(_save_payload(npz_workspace, data))
    batch = provider(batch_size=3, device="cpu", update=1)
    assert (batch.states["lane"][:, 0] == -1).all()
    assert batch.states["present"][:, 0].all()
    critic = ScaledTeamCritic(hidden_dim=8, n_heads=1, value_scale=10000)
    assert torch.isfinite(critic(batch.states)).all()


@pytest.mark.parametrize("changes,match", [
    ({"returns": [0., -1., -2.]}, "negative"),
    ({"returns": [-1., math.inf, -2.]}, "finite"),
    ({"returns": [[-1.], [-2.], [-3.]]}, "negative"),
    ({"gamma": 0.99}, "gamma"),
    ({"gamma": True}, "gamma"),
    ({"gamma": [1.]}, "scalar"),
    ({"units": "per_vehicle_seconds"}, "units"),
    ({"split": "VAL"}, "split"),
    ({"split": ["TRAIN"]}, "scalar"),
    ({"accounting": "network_only"}, "accounting"),
    ({"terminal_semantics": "zero_bootstrap"}, "terminal_semantics"),
    ({"source_kind": "designed_controller"}, "source_kind"),
    ({"source_kind": "current_policy"}, "policy_checkpoint"),
    ({"policy_checkpoint": " "}, "nonempty"),
    ({"policy_checkpoint": ["a.pt"]}, "scalar"),
    ({"requested_population": [1, 0, 3]}, "requested_population"),
    ({"requested_population": [1, 2.5, 3]}, "requested_population"),
    ({"requested_population": [1, 2]}, "shape"),
    ({"horizon_s": [299.5, 300, 600]}, "horizon_s"),
    ({"horizon_s": [300, 600.5, 600]}, "horizon_s"),
    ({"horizon_s": [300, math.nan, 600]}, "finite"),
    ({"state__lane": [[3, 1, -1]] * 3}, "state__lane"),
    ({"state__present": [[1, 2, 0]] * 3}, "binary"),
    ({"state__is_cav": [[True, math.nan, False]] * 3}, "binary"),
    ({"state__v_mps": [[-1., 1., 0.]] * 3}, "nonnegative"),
    ({"state__length_m": [[0., 5., 0.]] * 3}, "positive length"),
    ({"state__entry_backlog": [[-1., 0.]] * 3}, "nonnegative"),
    ({"state__a_mps2": [[1e300, 0., 0.]] * 3}, "finite float32"),
    ({"state__unexpected": [[0.]] * 3}, "unknown"),
], ids=[f"case{i}" for i in range(29)])
def test_npz_strict_schema_and_provenance_validation(npz_workspace, changes, match):
    data = _npz_payload()
    data.update(changes)
    with pytest.raises(ValueError, match=match):
        NPZRealReturnProvider(_save_payload(npz_workspace, data))


def test_npz_requires_all_schema_keys(npz_workspace):
    for name in _npz_payload():
        data = _npz_payload()
        del data[name]
        with pytest.raises(ValueError, match="missing"):
            NPZRealReturnProvider(_save_payload(npz_workspace, data))


def test_npz_optional_seed_metadata_refuses_evaluation_range(npz_workspace):
    for name in ("seed", "seeds"):
        for value in (964000, 964199, np.array([4107, 964123, 4108])):
            data = {**_npz_payload(), name: np.asarray(value)}
            with pytest.raises(ValueError, match="forbidden traffic seed"):
                NPZRealReturnProvider(_save_payload(npz_workspace, data))
        for value in (963999, 964200, np.array([4107, 4108, 4109])):
            data = {**_npz_payload(), name: np.asarray(value)}
            assert name in NPZRealReturnProvider(_save_payload(npz_workspace, data)).provenance
        for value in (True, 4107.0, "4107", [-1, 1, 2], [4107, 4108]):
            with pytest.raises(ValueError, match="metadata"):
                NPZRealReturnProvider(_save_payload(npz_workspace, {**_npz_payload(), name: value}))


def test_npz_paths_are_guarded_before_any_io_and_provenance_cannot_hide_tokens(npz_workspace, monkeypatch):
    from wms.control import real_returns14 as real
    tokens = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1", "hetdata_truth")
    # These paths are strings only; none of these directories/files are created.
    def no_load(*args, **kwargs):
        pytest.fail("restricted path reached np.load")
    with monkeypatch.context() as guarded:
        guarded.setattr(real.np, "load", no_load)
        for token in tokens:
            path = npz_workspace / token.upper() / "never_open.npz"
            with pytest.raises(ValueError, match="restricted"):
                NPZRealReturnProvider(path)
            with pytest.raises(ValueError, match="restricted"):
                write_real_returns_npz(path, {}, [], [], [], source_kind="native")
        with pytest.raises(ValueError, match="outside workspace"):
            NPZRealReturnProvider(npz_workspace.parent / "never_open.npz")
        # Simulate a shared guard resolving an otherwise benign path to a denied
        # target, without creating a symlink or opening any prohibited location.
        guarded.setattr(real, "checked_path", lambda path: npz_workspace / "hetdata_truth.npz")
        with pytest.raises(ValueError, match="restricted"):
            NPZRealReturnProvider(npz_workspace / "safe.npz")
    for token in tokens:
        data = {**_npz_payload(), "policy_checkpoint": f"model/{token.upper()}/actor.pt"}
        with pytest.raises(ValueError, match="restricted"):
            NPZRealReturnProvider(_save_payload(npz_workspace, data))


def test_npz_never_enables_pickle(npz_workspace, monkeypatch):
    from wms.control import real_returns14 as real
    path = _save_payload(npz_workspace, {**_npz_payload(), "returns": np.array([object()], dtype=object)})
    load = real.np.load
    seen = []
    def checked_load(*args, **kwargs):
        seen.append(kwargs)
        return load(*args, **kwargs)
    monkeypatch.setattr(real.np, "load", checked_load)
    with pytest.raises(ValueError, match="allow_pickle=False"):
        NPZRealReturnProvider(path)
    assert seen == [{"allow_pickle": False}]


@pytest.mark.parametrize("mutate", ["targets", "road_negative", "source_nan", "source_shape",
                                    "horizon", "dt", "missing_source", "missing_dt"])
def test_npz_raw_ledgers_are_recomputed_and_validated(npz_workspace, mutate):
    data = _npz_payload()
    data.update(road_delay_steps=np.ones((600, 3)), source_wait_steps=np.ones((600, 3)),
                physics_dt=0.5, returns=np.full(3, -1200.))
    assert NPZRealReturnProvider(_save_payload(npz_workspace, data)).provenance["ledger_recomputed"]
    if mutate == "targets":
        data["returns"][0] /= 2
    elif mutate == "road_negative":
        data["road_delay_steps"][0, 0] = -1
    elif mutate == "source_nan":
        data["source_wait_steps"][0, 0] = math.nan
    elif mutate == "source_shape":
        data["source_wait_steps"] = np.ones((599, 3))
    elif mutate == "horizon":
        data["horizon_s"][0] = 600
    elif mutate == "dt":
        data["physics_dt"] = 1
    elif mutate == "missing_source":
        del data["source_wait_steps"]
    else:
        del data["physics_dt"]
    with pytest.raises(ValueError):
        NPZRealReturnProvider(_save_payload(npz_workspace, data))


@pytest.mark.parametrize("steps,dt", [(599, 0.5), (1201, 0.5), (600, 1.0), (600, True)])
def test_npz_writer_rejects_invalid_horizons_before_creating_file(npz_workspace, steps, dt):
    path = npz_workspace / "invalid.npz"
    with pytest.raises(ValueError):
        write_real_returns_npz(path, _npz_states(), np.ones((steps, 3)), np.ones((steps, 3)),
                               [10, 20, 30], physics_dt=dt, source_kind="native")
    assert not path.exists()


def test_npz_writer_requires_source_claim_and_validates_before_creating_file(npz_workspace):
    path = npz_workspace / "invalid.npz"
    args = (path, _npz_states(), np.ones((600, 3)), np.ones((600, 3)), [10, 20, 30])
    with pytest.raises(TypeError, match="source_kind"):
        write_real_returns_npz(*args)
    for kwargs in ({"source_kind": "designed"}, {"source_kind": "current_policy"}):
        with pytest.raises(ValueError):
            write_real_returns_npz(*args, **kwargs)
        assert not path.exists()
    zero = np.zeros((600, 3))
    with pytest.raises(ValueError, match="negative"):
        write_real_returns_npz(path, _npz_states(), zero, zero, [10, 20, 30], source_kind="native")
    assert not path.exists()


def test_npz_provider_validates_sampling_arguments(npz_workspace):
    path = _save_payload(npz_workspace, _npz_payload())
    for seed in (-1, True, 1.5, "0"):
        with pytest.raises(ValueError, match="seed"):
            NPZRealReturnProvider(path, seed=seed)
    provider = NPZRealReturnProvider(path)
    for name in ("batch_size", "update"):
        for bad in (0, -1, True, 1.5, "1"):
            kwargs = dict(batch_size=2, update=1, device="cpu")
            kwargs[name] = bad
            with pytest.raises(ValueError, match=name):
                provider(**kwargs)


@pytest.mark.parametrize("value", [0, -1, True, False, None, "10000", math.inf, math.nan])
def test_scaled_critic_validates_scale(value):
    with pytest.raises(ValueError, match="value_scale"):
        ScaledTeamCritic(value_scale=value)


@pytest.mark.parametrize("scale", [1.0, 10000.0])
def test_scaled_critic_has_subclass_only_buffer_and_scaled_values_and_state_gradients(scale):
    from wms.control.policy import TeamCritic
    torch.manual_seed(15)
    base = TeamCritic(hidden_dim=8, n_heads=1, n_layers=1, entry_backlog=True)
    torch.manual_seed(15)
    scaled = ScaledTeamCritic(hidden_dim=8, n_heads=1, n_layers=1, entry_backlog=True,
                              value_scale=scale)
    assert "value_scale" not in base.state_dict()
    assert set(scaled.state_dict()) == set(base.state_dict()) | {"value_scale"}
    assert "value_scale" in dict(scaled.named_buffers())
    assert "value_scale" not in dict(scaled.named_parameters())
    state = _npz_states(2)
    state["v_mps"].requires_grad_(True)
    value = base(state)
    scaled_value = scaled(state)
    torch.testing.assert_close(scaled_value, value * scale, rtol=0, atol=0)
    grad, = torch.autograd.grad(value.sum(), state["v_mps"])
    scaled_grad, = torch.autograd.grad(scaled_value.sum(), state["v_mps"])
    torch.testing.assert_close(scaled_grad, grad * scale, rtol=1e-5, atol=1e-5)
    restored = ScaledTeamCritic(hidden_dim=8, n_heads=1, n_layers=1, entry_backlog=True)
    restored.load_state_dict(scaled.state_dict())
    assert float(restored.value_scale) == scale
    torch.testing.assert_close(restored(state), scaled_value, rtol=0, atol=0)


def test_warm_start_scaled_critic_32_steps_is_critic_only_and_exactly_copies_target():
    provider = Provider(RealReturnBatch(D._state(3), torch.full((3,), -60000.), 1.0))
    trainer = _trainer(provider=provider, real_return_mix=0.0, real_return_scale=10000.,
                       critic=ScaledTeamCritic(hidden_dim=8, n_heads=1, n_layers=0,
                                               value_scale=10000.), critic_lr=0.01)
    trainer.update_count = 17  # warmup provider indices must still start at 1
    trainer.return_low, trainer.return_high = -1e7, 1e7
    trainer.policy.eval()
    trainer.critic.eval()
    trainer.critic.head[0].train()  # mixed module modes must also be preserved
    modes = [module.training for module in trainer.critic.modules()]
    for p in trainer.actor_parameters:
        p.grad = torch.full_like(p, 0.125)
    modules = [trainer.policy, trainer.group_encoder, *trainer.world_models]
    module_before = [copy.deepcopy(module.state_dict()) for module in modules]
    actor_grad_before = [p.grad.clone() for p in trainer.actor_parameters]
    actor_optimizer = copy.deepcopy(trainer.actor_optimizer.state_dict())
    critic_before = copy.deepcopy(trainer.critic.state_dict())
    rng = torch.get_rng_state()
    log = warm_start_real_critic(trainer, updates=32, batch_size=5)
    assert log["updates"] == 32 and log["batch_size"] == 5 and log["finite"] is True
    assert log["real_return_scale"] == 10000
    assert log["last_mse_raw"] < log["first_mse_raw"]
    assert log["first_mse_scaled"] < 100
    assert log["last_grad_norm"] < trainer.config.grad_clip
    for prefix in ("first", "last"):
        assert math.isfinite(log[f"{prefix}_mse_raw"])
        assert log[f"{prefix}_mse_scaled"] == pytest.approx(log[f"{prefix}_mse_raw"] / 1e8)
    assert [call[0]["update"] for call in provider.calls] == list(range(1, 33))
    assert all(call[0]["batch_size"] == 5 and call[1] is False for call in provider.calls)
    D._assert_equal(trainer.critic.state_dict(), trainer.target_critic.state_dict())
    assert any(not torch.equal(value, critic_before[key])
               for key, value in trainer.critic.state_dict().items())
    assert [module.training for module in trainer.critic.modules()] == modes
    for before, module in zip(module_before, modules):
        D._assert_equal(before, module.state_dict())
    for before, p in zip(actor_grad_before, trainer.actor_parameters):
        torch.testing.assert_close(p.grad, before, rtol=0, atol=0)
    D._assert_equal(actor_optimizer, trainer.actor_optimizer.state_dict())
    assert trainer.update_count == 17 and trainer.start_states.draws == 0
    assert (trainer.return_low, trainer.return_high) == (-1e7, 1e7)
    assert trainer.lagrange_multiplier == trainer.config.lagrange_init
    assert trainer.policy.training is False
    assert all(p.grad is None for model in trainer.world_models for p in model.parameters())
    assert all(not p.requires_grad and p.grad is None for p in trainer.target_critic.parameters())
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


def test_warm_start_uses_only_configured_scale_and_detaches_provider_graph():
    trainer = _trainer(provider=Provider(), real_return_scale=10000., target_tau=0.001)
    source = torch.tensor(0.5, requires_grad=True)
    state = D._state(3)
    state["v_mps"] = state["v_mps"] + source + trainer.policy.mean
    target = torch.full((3,), -60000.) + source + trainer.group_encoder.signal
    trainer.real_return_provider.batch = RealReturnBatch(state, target, 1.0)
    expected = float((trainer.critic(state).detach() - target.detach()).double().square().mean())
    # A large imagined scale must not influence warmup or its normalization stats.
    trainer.return_low, trainer.return_high = -1e9, 1e9
    log = warm_start_real_critic(trainer, 1)
    assert log["batch_size"] == trainer.config.batch_size
    assert log["first_mse_raw"] == pytest.approx(expected)
    assert log["first_mse_scaled"] == pytest.approx(expected / 1e8)
    assert source.grad is None
    assert all(p.grad is None for p in trainer.actor_parameters)
    D._assert_equal(trainer.critic.state_dict(), trainer.target_critic.state_dict())


def test_warm_start_npz_sampling_is_reproducible_without_provider_state(npz_workspace):
    data = _npz_payload()
    data["returns"] *= 1000
    path = _save_payload(npz_workspace, data)
    def run():
        torch.manual_seed(19)
        trainer = _trainer(provider=NPZRealReturnProvider(path, seed=3), real_return_scale=10000.,
                           critic=ScaledTeamCritic(hidden_dim=8, n_heads=1, n_layers=0,
                                                   value_scale=10000.))
        with torch.no_grad():  # helper explicitly enables the critic-fitting graph
            log = warm_start_real_critic(trainer, 3)
        return trainer, log
    first, first_log = run()
    second, second_log = run()
    assert first_log == second_log
    D._assert_equal(first.critic.state_dict(), second.critic.state_dict())
    D._assert_equal(first.critic_optimizer.state_dict(), second.critic_optimizer.state_dict())


def test_warm_start_zero_updates_is_noop_without_provider():
    trainer = _trainer()
    before = [copy.deepcopy(module.state_dict()) for module in (trainer.critic, trainer.target_critic)]
    log = warm_start_real_critic(trainer, 0)
    assert log["updates"] == 0 and log["first_mse_raw"] is log["last_mse_raw"] is None
    for value, module in zip(before, (trainer.critic, trainer.target_critic)):
        D._assert_equal(value, module.state_dict())
    assert not trainer.critic_optimizer.state


def test_warm_start_fails_on_nonfinite_mse_and_restores_modes_without_copying_target():
    trainer = _trainer(provider=Provider())
    trainer.critic.eval()
    with torch.no_grad():
        trainer.critic.bias.fill_(math.nan)
    target_before = copy.deepcopy(trainer.target_critic.state_dict())
    with pytest.raises(ValueError, match="nonfinite"):
        warm_start_real_critic(trainer, 1)
    assert trainer.critic.training is False
    D._assert_equal(target_before, trainer.target_critic.state_dict())
    assert not trainer.critic_optimizer.state


def test_warm_start_validates_counts_and_requires_provider():
    trainer = _trainer()
    for updates in (-1, True, 0.5, "1"):
        with pytest.raises(ValueError, match="updates"):
            warm_start_real_critic(trainer, updates)
    for size in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="batch_size"):
            warm_start_real_critic(trainer, 1, batch_size=size)
    with pytest.raises(ValueError, match="real_return_provider"):
        warm_start_real_critic(trainer, 1)
