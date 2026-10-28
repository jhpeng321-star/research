"""CPU-only objective/gradient/update proofs; synthetic starts, no simulator/data."""
import copy
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from wms.control.counterfactual import CounterfactualConfig
from wms.control.dreamer import DreamerConfig, DreamerTrainer, MEMORY_CONFIG_FIELDS
from wms.control.start_states import StartBatch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA saved-storage sharing proof')
def test_saved_spill_shares_parameter_views_without_changing_gradients():
    from wms.control.imagination_memory import _saved_tensor_spill
    torch.cuda.set_per_process_memory_fraction(.45)
    torch.manual_seed(4101)
    model = nn.Linear(64, 64, bias=False).cuda()
    initial = torch.randn(2, 64, device='cuda', requires_grad=True)
    owner = SimpleNamespace(world_models=[model], policy=model, group_encoder=model,
                            critic=model, device=torch.device('cuda'))
    def run(context):
        with context:
            x = initial
            for _ in range(8):
                x = torch.tanh(model(x))
            loss = x.square().sum()
            return torch.autograd.grad(loss, (initial, model.weight))
    expected = run(torch.autograd.graph.save_on_cpu(pin_memory=True))
    actual = run(_saved_tensor_spill(owner))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert owner.memory_spill_stats['reuses'] >= 7


def scene(batch=3, slots=4):
    pos = torch.arange(slots).float()[None].expand(batch, -1) * 30 + 200
    return dict(s_m=pos.clone(), v_mps=torch.full_like(pos, 10.),
                a_mps2=torch.zeros_like(pos),
                lane=torch.zeros_like(pos, dtype=torch.long),
                present=torch.ones_like(pos, dtype=torch.bool),
                is_cav=torch.ones_like(pos, dtype=torch.bool), length_m=torch.full_like(pos, 5.))


def observe(state):
    return SimpleNamespace(cav_mask=state["present"] & state["is_cav"],
                           speed=state["v_mps"] / 30)


class Group(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.1))

    def forward(self, obs, state):
        return obs.speed * self.weight


class Policy(nn.Module):
    supports_noise = True

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.2))

    def forward(self, obs, signal, deterministic=False, noise=None):
        eps = (torch.zeros_like(signal) if deterministic else
               torch.randn_like(signal) if noise is None else noise.accel)
        action = torch.tanh(self.weight * obs.speed + signal + .1 * eps)
        return action, -(action - self.weight).square()


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 12), nn.Tanh(), nn.Linear(12, 1))

    def forward(self, state):
        x = torch.stack((state["v_mps"].mean(-1) / 30, state["s_m"].mean(-1) / 1000), -1)
        return self.net(x).squeeze(-1)


class World(nn.Module):
    """Differentiable recurrent WM with real continuation and large hidden activations."""
    def __init__(self, width=128, offset=0.):
        super().__init__()
        self.register_buffer("features", torch.linspace(-1, 1, width))
        self.offset = offset
        self.physics = SimpleNamespace(dt_s=.5, exit_at_road_end=False)
        self.checkpoint_forward = False
        self.cav_exec_cap = "off"
        self.grad_decay_per_step = 1.

    def rollout(self, state, *, n_steps, k0=0, history=None, history_mask=None,
                cav_controller=None, **kwargs):
        current = dict(state)
        log = {k: [] for k in ("positions", "speeds", "lanes", "masks", "accel")}
        for t in range(n_steps):
            command = cav_controller(current)
            hidden = torch.sin(current["v_mps"].unsqueeze(-1) * .01 + self.features)
            delta = hidden.square().mean(-1) * .01 + self.offset
            speed = command[..., 2] + delta
            pos = command[..., 0] + .5 * delta
            current = {**current, "s_m": pos, "v_mps": speed, "a_mps2": command[..., 3]}
            for key, value in zip(log, (pos, speed, current["lane"], current["present"], command[..., 3])):
                log[key].append(value)
        return {**{k: torch.stack(v, 1) for k, v in log.items()}, "state": current,
                "history": current["v_mps"], "history_mask": current["present"]}


def reward(state, nxt, accel, weights):
    delay = (30 - nxt["v_mps"]).mean(-1) * .5
    return dict(reward=-delay - .03 * accel.square().mean(-1), delay=delay,
                cost_ttc=.01 * nxt["v_mps"].square().mean(-1))


def queue_reward(state, nxt, accel, weights):
    result = reward(state, nxt, accel, weights)
    source = .5 * state.get("entry_queue", torch.zeros_like(result["delay"]))
    return {**result, "reward": result["reward"] - source, "source_wait": source}


class Starts:
    def __init__(self, batch=3, slots=4):
        self.batch = StartBatch(scene(batch, slots), metadata=[{} for _ in range(batch)])

    def sample(self, size, device):
        return copy.deepcopy(self.batch)


def trainer(micro=0, segment=0, offload=False, horizon=5, batch=3, slots=4, width=128,
            counterfactual=False, real_world=False, **options):
    torch.manual_seed(1729)
    worlds = [World(width, .001), World(width, -.001)]
    if real_world:
        from wms.model.wms import WmSModel
        from wms.model.spec import ModelSpec
        worlds = [WmSModel(ModelSpec(history_hidden=8, k_hidden=8, grad_decay_per_step=.9))]
    config = DreamerConfig(batch_size=batch, horizon=horizon, imagine_entries=False,
                           checkpoint_imagination=True, lagrange_init=.03,
                           imagination_microbatch=micro, imagination_checkpoint_segment=segment,
                           imagination_offload_futures=offload,
                           imagination_offload_saved_tensors=offload, **options)
    cf = CounterfactualConfig(enabled=counterfactual, max_starts=2, n_individual=2,
                              group_min_size=2, group_radius_m=100)
    return DreamerTrainer(worlds, Policy(), Group(), Critic(), Starts(batch, slots),
                          observation_builder=observe, reward_fn=reward,
                          safety=lambda state, command, **kw: command,
                          config=config, counterfactual=cf)


def assert_tree_close(a, b, atol=2e-6, rtol=2e-5):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_close(a[key], b[key], atol, rtol)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_close(x, y, atol, rtol)
    elif isinstance(a, float):
        assert a == pytest.approx(b, abs=atol, rel=rtol)
    else:
        assert a == b


@pytest.mark.parametrize("micro,segment,offload", [(1, 0, False), (2, 0, True),
                                                   (0, 3, False), (1, 3, True), (2, 4, False)])
@pytest.mark.parametrize("cf", [False, True])
def test_full_update_and_gradients_identical(micro, segment, offload, cf):
    baseline = trainer(counterfactual=cf, entropy_target=.5)
    memory = trainer(micro, segment, offload, counterfactual=cf, entropy_target=.5)
    for _ in range(2):
        torch.manual_seed(91 + baseline.update_count)
        a = baseline.update()
        rng = torch.get_rng_state()
        torch.manual_seed(91 + memory.update_count)
        b = memory.update()
        assert torch.equal(rng, torch.get_rng_state())
        for key in ("actor_loss", "critic_loss", "returns", "return_scale", "entropy",
                    "actor_grad_norm", "critic_grad_norm", "lagrange_multiplier"):
            assert_tree_close(a[key], b[key])
        for left, right in ((baseline.actor_parameters, memory.actor_parameters),
                            (list(baseline.critic.parameters()), list(memory.critic.parameters()))):
            for x, y in zip(left, right):
                assert_tree_close(x.grad, y.grad)
                assert_tree_close(x, y)
        assert_tree_close(baseline.actor_optimizer.state_dict(), memory.actor_optimizer.state_dict())
        assert_tree_close(baseline.critic_optimizer.state_dict(), memory.critic_optimizer.state_dict())
        assert_tree_close(baseline.target_critic.state_dict(), memory.target_critic.state_dict())
        assert_tree_close(baseline.alpha_optimizer.state_dict(), memory.alpha_optimizer.state_dict())
        assert baseline._training_imagination is memory._training_imagination is False


def test_real_wm_continuation_gradients_and_updates():
    baseline, memory = trainer(real_world=True), trainer(2, 3, True, real_world=True)
    torch.manual_seed(31)
    a = baseline.update()
    torch.manual_seed(31)
    b = memory.update()
    for key in ("actor_loss", "critic_loss", "actor_grad_norm", "critic_grad_norm"):
        assert_tree_close(a[key], b[key])
    for x, y in zip(baseline.actor_parameters, memory.actor_parameters):
        assert_tree_close(x.grad, y.grad)
        assert_tree_close(x, y)


def test_heterogeneous_correlated_exploration():
    from wms.control.coopw import ExplorationConfig
    baseline = trainer(counterfactual=True, slots=6)
    memory = trainer(1, 3, True, counterfactual=True, slots=6)
    for t in (baseline, memory):
        t.exploration = ExplorationConfig(enabled=True, rho=[.5])
        t.start_states.batch.state["s_m"][0] = torch.tensor([0., 10., 20., 400., 410., 420.])
    torch.manual_seed(60)
    a = baseline.update()
    torch.manual_seed(60)
    b = memory.update()
    assert_tree_close(a["actor_loss"], b["actor_loss"])
    for x, y in zip(baseline.actor_parameters, memory.actor_parameters):
        assert_tree_close(x.grad, y.grad)


def test_global_masked_safety_and_gradient_audit():
    options = dict(counterfactual=True, ttc_budget_mode="native_relative", lagrange_scaled=True,
                   actor_gradient_audit=True)
    baseline, memory = trainer(**options), trainer(2, 3, True, **options)
    # One start has no counterfactual native reference; global dual reduction must stay global.
    torch.manual_seed(334)
    a = baseline.update()
    torch.manual_seed(334)
    b = memory.update()
    for key in ("actor_loss", "actor_grad_norm", "lagrange_multiplier", "ttc_excess",
                "actor_grad_norms"):
        assert_tree_close(a[key], b[key])
    for x, y in zip(baseline.actor_parameters, memory.actor_parameters):
        assert_tree_close(x.grad, y.grad)


def test_streamed_critic_preserves_frozen_parameters():
    baseline, memory = trainer(), trainer(2, 3, True)
    for t in (baseline, memory):
        t.critic.net[0].weight.requires_grad_(False)
    torch.manual_seed(50)
    a = baseline.update()
    torch.manual_seed(50)
    b = memory.update()
    assert_tree_close(a["critic_loss"], b["critic_loss"])
    for x, y in zip(baseline.critic.parameters(), memory.critic.parameters()):
        assert_tree_close(x.grad, y.grad)
        assert_tree_close(x, y)


@pytest.mark.parametrize("lane,residual,entries,samples", [(True, False, True, 2),
                                                   (False, True, False, 2),
                                                   (False, True, False, 8)])
def test_production_actor_bridge_futures_and_entries(lane, residual, entries, samples):
    from wms.control.bridge import make_cav_controller
    from wms.control.lanes import LaneRules
    from wms.control.obs import build_observations
    from wms.control.policy import CavPolicy, RsuGroupEncoder
    from wms.model.spec import ModelSpec
    from wms.model.wms import WmSModel

    def make(micro, segment):
        torch.manual_seed(701)
        starts = Starts(3, 4)
        starts.batch.state["is_cav"][:, -1] = False
        starts.batch.state["lane"][:, 1] = 1
        if entries:
            # Entrance rate/speed/CAV-share features, generated in memory.
            from wms.model.entries import EntranceHistory
            import numpy as np
            hist = EntranceHistory.defaults()
            hist.rate[:] = .5
            starts.batch.entry_feat = torch.tensor(np.stack([hist.to_features()] * 3)).float()
        cfg = DreamerConfig(batch_size=3, horizon=3, imagine_entries=entries, entry_pad_max=2,
                            source_wait_grad=entries,
                            future_samples=samples, pessimistic_credit=samples == 8,
                            future_driver_scale=.05, future_accel_noise=True,
                            imagination_microbatch=micro, imagination_checkpoint_segment=segment,
                            imagination_offload_futures=True if micro else False,
                            imagination_offload_saved_tensors=True if micro else False)
        spec = ModelSpec(history_hidden=8, k_hidden=8, rollout_mode="hybrid",
                         grad_decay_per_step=.9, **({"lane_model": "hazard_v5",
                         "merge_decision": "first_passage", "lane_decision": "sample",
                         "merge_passage_law": "hybrid"} if residual else {}))
        return DreamerTrainer(
            [WmSModel(spec)],
            CavPolicy(hidden_dim=8, n_heads=2, lane_head=lane, residual_on_native=residual),
            RsuGroupEncoder(hidden_dim=8, n_heads=2), Critic(), starts,
            observation_builder=build_observations, reward_fn=queue_reward,
            safety=lambda state, command, **kw: command, config=cfg,
            controller_factory=(partial(make_cav_controller, native_lane_change=True,
                                        native_ramp_merge="wm_law", symmetric_exec_cap=True)
                                if residual else make_cav_controller),
            lane_rules=LaneRules() if lane else None,
            counterfactual=CounterfactualConfig(enabled=True, max_starts=2, n_individual=2,
                                                group_min_size=2, group_radius_m=100))

    baseline, memory = make(0, 0), make(2, 3)
    torch.manual_seed(313)
    a = baseline.update()
    rng = torch.get_rng_state()
    torch.manual_seed(313)
    b = memory.update()
    assert torch.equal(rng, torch.get_rng_state())
    for key in ("actor_loss", "critic_loss", "actor_grad_norm", "critic_grad_norm", "returns"):
        assert_tree_close(a[key], b[key])
    for left, right in ((baseline.actor_parameters, memory.actor_parameters),
                        (list(baseline.critic.parameters()), list(memory.critic.parameters()))):
        for x, y in zip(left, right):
            assert_tree_close(x.grad, y.grad)
            assert_tree_close(x, y)


@pytest.mark.parametrize("field,value", [("imagination_microbatch", -1),
                                         ("imagination_microbatch", True),
                                         ("imagination_checkpoint_segment", 1.5),
                                         ("imagination_offload_futures", 1)])
def test_flags_validate(field, value):
    with pytest.raises(ValueError):
        DreamerConfig(**{field: value})


def test_default_off():
    cfg = DreamerConfig()
    assert cfg.imagination_microbatch == cfg.imagination_checkpoint_segment == 0
    assert cfg.imagination_offload_futures is False
    assert cfg.imagination_offload_saved_tensors is False


def test_enable_memory_on_full_resume_of_legacy_checkpoint(tmp_path):
    baseline = trainer()
    baseline.update()
    path = tmp_path / "legacy.pt"
    baseline.save_checkpoint(path)
    payload = torch.load(path, weights_only=True)
    for key in MEMORY_CONFIG_FIELDS:
        del payload["config"][key]
    torch.save(payload, path)
    memory = trainer(2, 3, True)
    memory.load_checkpoint(path)
    assert memory.config.imagination_microbatch == 2
    assert memory.config.imagination_checkpoint_segment == 3
    assert memory.config.imagination_offload_futures
    assert memory.config.imagination_offload_saved_tensors
    assert_tree_close(baseline.actor_optimizer.state_dict(), memory.actor_optimizer.state_dict())
    torch.manual_seed(419)
    a = baseline.update()
    torch.manual_seed(419)
    b = memory.update()
    assert_tree_close(a["actor_loss"], b["actor_loss"])
    for x, y in zip(baseline.actor_parameters, memory.actor_parameters):
        assert_tree_close(x.grad, y.grad)
        assert_tree_close(x, y)
    memory.save_checkpoint(path)
    restored = trainer()
    restored.load_checkpoint(path)
    for key in MEMORY_CONFIG_FIELDS:
        assert getattr(restored.config, key) == getattr(memory.config, key)
