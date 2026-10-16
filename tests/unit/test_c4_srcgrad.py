"""Entry-queue gradient for the C4 Dreamer (SRCGRAD_NOTES.md).

CPU only, tiny randomly initialised WM-S; no data or checkpoint access."""
from __future__ import annotations

import copy

import pytest
import torch

from wms.control import dreamer as dreamer_module
from wms.control.bridge import make_cav_controller
from wms.control.contract import PHYSICS_DT_S, RewardWeights
from wms.control.counterfactual import CounterfactualConfig
from wms.control.dreamer import DreamerConfig, DreamerTrainer
from wms.control.lanes import LaneRules
from wms.control.obs import build_observations
from wms.control.policy import CavPolicy, RsuGroupEncoder, TeamCritic
from wms.control.reward import team_reward
from wms.control.safety import safe_accel
from wms.control.srcgrad import (lane_block_probability, queue_surrogate, straight_through)
from wms.control.train_cli import GroupSignal
from wms.model.entries import ENTRY_S_M, EntranceHistory, EntryConfig
from wms.model.residual_net import HdvConfig
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel

from .test_c4_srcwait_imagination import Starts


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _world_model():
    torch.manual_seed(11)
    model = WmSModel(ModelSpec(rollout_mode="hybrid", history_hidden=16, k_hidden=8),
                     hdv=HdvConfig(d_model=16, n_heads=2, n_layers=2, ffn=32))
    return model.eval()


def _trainer(model, starts, *, grad=False, critic_backlog=False, counterfactual=False, horizon=4,
             checkpoint_imagination=True):
    torch.manual_seed(3)
    settings = dict(batch_size=2, horizon=horizon, checkpoint_every=0, validation_every=0,
                    actor_lr=1e-3, critic_lr=1e-3, checkpoint_imagination=checkpoint_imagination,
                    source_wait_grad=grad, critic_entry_backlog=critic_backlog)
    return DreamerTrainer(
        [copy.deepcopy(model)], CavPolicy(hidden_dim=16), GroupSignal(RsuGroupEncoder(hidden_dim=16)),
        TeamCritic(hidden_dim=16, entry_backlog=critic_backlog), starts,
        observation_builder=build_observations, reward_fn=team_reward, safety=safe_accel,
        controller_factory=make_cav_controller, config=DreamerConfig(**settings),
        reward_weights=RewardWeights(), lane_rules=LaneRules(),
        counterfactual=CounterfactualConfig(enabled=counterfactual, n_individual=2, n_group=2,
                                            n_flip=2, group_max_size=4))


def _grads(trainer):
    return [None if p.grad is None else p.grad.clone() for p in trainer.actor_parameters]


# --------------------------------------------------------------------------- switch off
def test_switch_off_never_builds_the_surrogate_and_update_is_reproducible(monkeypatch):
    """Off: the surrogate is never evaluated, the queue is a constant, and the update equals an
    independent second run bit for bit (the cross-commit golden check is in SRCGRAD_NOTES.md)."""
    def boom(*args, **kwargs):
        raise AssertionError("surrogate evaluated with source_wait_grad=false")

    monkeypatch.setattr(dreamer_module, "_queue_surrogate_fn", boom)
    model = _world_model()
    results = []
    for _ in range(2):
        trainer = _trainer(model, Starts(True), counterfactual=True)
        trainer.update_count = 1500
        torch.manual_seed(7)
        metrics = trainer.update()
        metrics.pop("wall_time_s")
        results.append((metrics, _grads(trainer),
                        [p.detach().clone() for p in trainer.critic.parameters()]))
    assert results[0][0] == results[1][0]
    for a, b in zip(results[0][1], results[1][1]):
        assert (a is None and b is None) or torch.equal(a, b)
    for a, b in zip(results[0][2], results[1][2]):
        assert torch.equal(a, b)
    assert results[0][0]["source_wait"] > 0


def test_reward_with_constant_queue_is_unchanged_by_the_straight_through_path():
    g = torch.Generator().manual_seed(0)
    b, k = 3, 6
    state = {"s_m": torch.randn(b, k, generator=g) * 20, "v_mps": torch.rand(b, k, generator=g) * 30,
             "lane": torch.randint(0, 3, (b, k), generator=g),
             "present": torch.rand(b, k, generator=g) > 0.3, "a_mps2": torch.zeros(b, k)}
    nxt = {**state, "v_mps": torch.rand(b, k, generator=g) * 30, "s_m": state["s_m"] + 10}
    accel = torch.randn(b, k, generator=g)
    queue = torch.tensor([0.0, 3.0, 10.0])
    plain = team_reward(state, nxt, accel, RewardWeights(), entry_queue=queue)
    sur = (queue * 0.3).requires_grad_()
    st = team_reward(state, nxt, accel, RewardWeights(), entry_queue=straight_through(queue, sur))
    for key in plain:
        assert torch.equal(plain[key], st[key].detach()), key
    st["reward"].sum().backward()
    # d reward / d queue (source_wait only, count fixed) = -w_delay * dt / count.
    count = state["present"].sum(-1).float() + queue
    torch.testing.assert_close(sur.grad, -RewardWeights().delay * PHYSICS_DT_S / count)


# ------------------------------------------------------------------- value is exact backlog
def test_surrogate_queue_value_is_exact_backlog_and_carries_gradient():
    model = _world_model()
    off = _trainer(model, Starts(True), horizon=5)
    on = _trainer(model, Starts(True), grad=True, horizon=5)
    imagined = []
    for trainer in (off, on):
        torch.manual_seed(7)
        imagined.append(trainer.imagine(trainer._entry_slots(trainer.start_states.sample(2))))
    q_off = [s["entry_queue"] for s in imagined[0]["states"][0][:-1]]
    q_on = [s["entry_queue"] for s in imagined[1]["states"][0][:-1]]
    for t, (a, b) in enumerate(zip(q_off, q_on)):
        # Step 0 is the start state (no actor dependence); later steps carry the surrogate.
        assert not a.requires_grad and b.requires_grad == (t > 0)
        assert torch.equal(a, b.detach())
    for key in ("rewards", "returns", "costs"):
        assert torch.equal(imagined[0][key], imagined[1][key].detach()), key
    for key, value in imagined[0]["components"].items():
        assert torch.equal(value, imagined[1]["components"][key].detach()), key
    assert float(q_on[-1].detach().min()) > 0
    source = imagined[1]["components"]["source_wait"]
    assert source.requires_grad and not imagined[0]["components"]["source_wait"].requires_grad
    # The source-wait term alone now reaches the actor.
    params = [p for p in on.actor_parameters if p.requires_grad]
    grads = torch.autograd.grad(source.sum(), params, allow_unused=True)
    assert sum(float(g.abs().sum()) for g in grads if g is not None) > 0


def test_update_with_surrogate_checkpointing_parity():
    model = _world_model()
    results = []
    for checkpointing in (False, True):
        trainer = _trainer(model, Starts(True), grad=True, critic_backlog=True, counterfactual=True,
                           checkpoint_imagination=checkpointing)
        trainer.update_count = 1500
        torch.manual_seed(7)
        metrics = trainer.update()
        results.append((metrics, _grads(trainer)))
        assert metrics["source_wait"] > 0
    for key in ("actor_loss", "returns", "source_wait", "D_all"):
        assert results[1][0][key] == pytest.approx(results[0][0][key], rel=1e-5, abs=1e-8), key
    for a, b in zip(results[0][1], results[1][1]):
        if a is not None:
            torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-7)


# ------------------------------------------------------------------------- gradient sign
def _ramp_scene(accel, n_steps, s_gap=8.0, v0=2.0):
    """One CAV on the ramp lane (2) just downstream of its entrance + two far mainline HDVs.
    The CAV holds ``accel`` (ballistic, as the trainer's controller); states t = 0..n_steps."""
    states = []
    for t in range(n_steps + 1):
        time = t * PHYSICS_DT_S
        v_cav = (v0 + accel * time).clamp_min(0.0)
        s_cav = ENTRY_S_M[2] + 5.0 + s_gap + v0 * time + 0.5 * accel * time ** 2
        s = torch.stack((s_cav, torch.tensor(-1000.0), torch.tensor(-900.0)))[None]
        v = torch.stack((v_cav, torch.tensor(25.0), torch.tensor(25.0)))[None]
        states.append({"s_m": s, "v_mps": v, "lane": torch.tensor([[2, 0, 1]]),
                       "present": torch.ones(1, 3, dtype=torch.bool),
                       "a_mps2": torch.stack((accel, torch.tensor(0.0), torch.tensor(0.0)))[None],
                       "is_cav": torch.tensor([[True, False, False]]),
                       "length_m": torch.full((1, 3), 5.0)})
    return states


def _hist_tensors(h):
    return {name: torch.as_tensor(getattr(h, name), dtype=torch.float32)[None]
            for name in ("rate", "speed_mean", "gap_mean", "gap_std")}


def test_slowing_an_entrance_cav_increases_cost():
    hist = _hist_tensors(EntranceHistory.defaults())
    cfg = EntryConfig(max_pending=10 ** 9)
    pending = torch.tensor([[0.0, 3.0]])           # three vehicles queued at the ramp
    accel = torch.tensor(0.0, requires_grad=True)
    states = _ramp_scene(accel, 8)
    total = 0.0
    p_blocks = []
    for t in range(8):
        q_sur, p_block = queue_surrogate(states[t], states[t]["present"], pending, hist, ENTRY_S_M,
                                         cfg, PHYSICS_DT_S)
        p_blocks.append(p_block)
        queue = straight_through(torch.tensor([3.0]), q_sur)
        out = team_reward(states[t], states[t + 1], states[t]["a_mps2"], RewardWeights(),
                          entry_queue=queue)
        total = total + out["reward"].sum()
    assert 0.05 < float(p_blocks[0][0, 1].detach()) < 0.95       # ramp in the transition region
    assert float(p_blocks[0][0, 0].detach()) == 0.0               # mainline lanes start empty -> free
    (grad,) = torch.autograd.grad(total, accel)
    # More acceleration clears the entrance -> less waiting -> higher reward; slowing -> cost.
    assert float(grad) > 0
    # And only through the surrogate: without it the reward gradient w.r.t. accel is the
    # smoothness/delay part, which favours braking-free low accel; compare against it.
    accel2 = torch.tensor(0.0, requires_grad=True)
    states2 = _ramp_scene(accel2, 8)
    base = sum(team_reward(states2[t], states2[t + 1], states2[t]["a_mps2"], RewardWeights(),
                           entry_queue=torch.tensor([3.0]))["reward"].sum() for t in range(8))
    (grad_base,) = torch.autograd.grad(base, accel2)
    assert float(grad) > float(grad_base)


def test_lane_block_probability_limits():
    cfg = EntryConfig()
    zero = torch.zeros(1)
    state = {"s_m": torch.tensor([[ENTRY_S_M[2] + 5.5, ENTRY_S_M[2] + 300.0]]),
             "v_mps": torch.tensor([[0.0, 20.0]]), "lane": torch.tensor([[2, 2]]),
             "length_m": torch.full((1, 2), 5.0), "present": torch.ones(1, 2, dtype=torch.bool)}
    p = lane_block_probability(state, state["present"], 2, ENTRY_S_M[2], torch.ones(1),
                               torch.full((1,), 10.0), zero + 5.0, cfg)
    assert float(p) > 0.95                         # stopped vehicle 0.5 m past the lane start
    far = {**state, "s_m": torch.tensor([[ENTRY_S_M[2] + 200.0, ENTRY_S_M[2] + 300.0]])}
    assert float(lane_block_probability(far, far["present"], 2, ENTRY_S_M[2], torch.ones(1),
                                        torch.full((1,), 10.0), zero + 5.0, cfg)) < 1e-3
    empty = lane_block_probability(state, state["present"], 1, ENTRY_S_M[1], torch.ones(1),
                                   torch.full((1,), 10.0), zero + 5.0, cfg)
    assert float(empty) == 0.0


# ---------------------------------------------------------------------------- critic input
def test_critic_backlog_input_shape_and_loading():
    critic = TeamCritic(hidden_dim=16, entry_backlog=True)
    assert critic.head[0].in_features == 18
    state = {"s_m": torch.randn(3, 5) * 50, "v_mps": torch.rand(3, 5) * 20, "a_mps2": torch.zeros(3, 5),
             "lane": torch.randint(0, 3, (3, 5)), "present": torch.ones(3, 5, dtype=torch.bool),
             "is_cav": torch.zeros(3, 5, dtype=torch.bool)}
    without = critic(state)
    assert without.shape == (3,)
    torch.testing.assert_close(critic({**state, "entry_backlog": torch.zeros(3, 2)}), without)
    assert not torch.equal(critic({**state, "entry_backlog": torch.full((3, 2), 20.0)}), without)
    with pytest.raises(RuntimeError):
        critic.load_state_dict(TeamCritic(hidden_dim=16).state_dict())
    assert TeamCritic(hidden_dim=16).head[0].in_features == 16


def test_trainer_feeds_backlog_to_critic_and_checks_the_critic():
    model = _world_model()
    with pytest.raises(ValueError):
        DreamerTrainer([copy.deepcopy(model)], CavPolicy(hidden_dim=16),
                       GroupSignal(RsuGroupEncoder(hidden_dim=16)), TeamCritic(hidden_dim=16),
                       Starts(True), observation_builder=build_observations, reward_fn=team_reward,
                       safety=safe_accel, controller_factory=make_cav_controller,
                       config=DreamerConfig(critic_entry_backlog=True), lane_rules=LaneRules())
    trainer = _trainer(model, Starts(True), grad=True, critic_backlog=True, horizon=3)
    torch.manual_seed(7)
    imagined = trainer.imagine(trainer._entry_slots(trainer.start_states.sample(2)))
    states = imagined["states"][0]
    assert all(s["entry_backlog"].shape == (2, 2) for s in states)
    for s in states[:-1]:
        torch.testing.assert_close(s["entry_backlog"].sum(-1), s["entry_queue"].detach())
    assert float(states[-1]["entry_backlog"].sum()) > 0
    metrics = trainer.update()
    assert metrics["critic_grad_norm"] > 0
