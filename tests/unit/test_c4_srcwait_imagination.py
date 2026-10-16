"""Entry queue in the C4 reward and imagination (SRCWAIT_NOTES.md).

CPU only, tiny randomly initialised WM-S; no data or checkpoint access."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from wms.control.bridge import make_cav_controller
from wms.control.contract import PHYSICS_DT_S, RewardWeights
from wms.control.counterfactual import CounterfactualConfig
from wms.control.dreamer import DreamerConfig, DreamerTrainer, with_entry_slots
from wms.control.lanes import LaneRules
from wms.control.obs import build_observations
from wms.control.policy import CavPolicy, RsuGroupEncoder, TeamCritic
from wms.control.reward import episode_summary, team_reward
from wms.control.safety import safe_accel
from wms.control.start_states import StartBatch
from wms.control.train_cli import GroupSignal
from wms.model.entries import (ENTRY_S_M, EntryConfig, TorchEntryProcessV2,
                               build_entrance_history)
from wms.model.residual_net import HdvConfig
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


# ------------------------------------------------------------------------------ reward
def _reward_states(seed=0, b=3, k=6):
    g = torch.Generator().manual_seed(seed)
    state = {"s_m": torch.randn(b, k, generator=g) * 20, "v_mps": torch.rand(b, k, generator=g) * 30,
             "lane": torch.randint(0, 3, (b, k), generator=g),
             "present": torch.rand(b, k, generator=g) > 0.3, "a_mps2": torch.zeros(b, k)}
    nxt = {**state, "v_mps": torch.rand(b, k, generator=g) * 30, "s_m": state["s_m"] + 10}
    return state, nxt, torch.randn(b, k, generator=g)


def test_reward_without_queue_is_unchanged_and_zero_queue_is_identical():
    state, nxt, accel = _reward_states()
    w = RewardWeights()
    base = team_reward(state, nxt, accel, w)
    zero = team_reward(state, nxt, accel, w, entry_queue=torch.zeros(3))
    assert set(base) == {"reward", "cost_ttc", "delay", "throughput", "smooth", "hard_brake",
                         "fairness_delay", "source_wait"}
    assert torch.equal(base["source_wait"], torch.zeros(3))
    for key in base:
        torch.testing.assert_close(zero[key], base[key], rtol=0, atol=0)
    # Reference: the pre-2026-10-13 formula, recomputed independently.
    present = state["present"]
    count = present.sum(-1).clamp_min(1).float()
    expected = ((-w.delay * base["delay"] + w.throughput * base["throughput"]
                 - w.smooth * base["smooth"] - w.hard_brake * base["hard_brake"]) / count
                - w.fairness * base["fairness_delay"])
    torch.testing.assert_close(base["reward"], expected)


def test_reward_queue_term_definition():
    state, nxt, accel = _reward_states()
    w = RewardWeights()
    queue = torch.tensor([0.0, 3.0, 10.0])
    base = team_reward(state, nxt, accel, w)
    out = team_reward(state, nxt, accel, w, entry_queue=queue)
    torch.testing.assert_close(out["source_wait"], PHYSICS_DT_S * queue)
    count = state["present"].sum(-1).float() + queue
    expected = ((-w.delay * (out["delay"] + out["source_wait"]) + w.throughput * out["throughput"]
                 - w.smooth * out["smooth"] - w.hard_brake * out["hard_brake"]) / count.clamp_min(1)
                - w.fairness * out["fairness_delay"])
    torch.testing.assert_close(out["reward"], expected)
    assert out["reward"][0] == base["reward"][0]
    assert bool((out["reward"][1:] < base["reward"][1:]).all())
    # Queued vehicles carry the maximal per-step delay dt -> the tail delay cannot drop.
    assert bool((out["fairness_delay"] >= base["fairness_delay"] - 1e-6).all())
    # state["entry_queue"] is the same input.
    via_state = team_reward({**state, "entry_queue": queue}, nxt, accel, w)
    torch.testing.assert_close(via_state["reward"], out["reward"])
    summary = episode_summary([out, out])
    assert summary["source_wait"] == pytest.approx(2 * float(out["source_wait"].mean()))


def test_weighted_cvar_equals_replicated_vehicles():
    from wms.control.reward import _soft_cvar
    delay = torch.tensor([[0.1, 0.3, 0.5, 0.0]])
    present = torch.tensor([[True, True, True, False]])
    weight = torch.tensor([[1.0, 3.0, 2.0, 0.0]])
    replicated = torch.tensor([[0.1, 0.3, 0.3, 0.3, 0.5, 0.5]])
    torch.testing.assert_close(_soft_cvar(delay, present, weight),
                               _soft_cvar(replicated, torch.ones(1, 6, dtype=torch.bool)))


# ------------------------------------------------------------------------- imagination
def _history(mainline_per_bin=10.0, ramp_per_bin=4.0):
    counter = np.zeros((2, 30), np.float32)
    counter[0, :] = mainline_per_bin
    counter[1, :] = ramp_per_bin
    return build_entrance_history(counter=counter)


def _state(block_entrance):
    s = [-900.0, -860.0, -820.0, -700.0, -880.0, -790.0, -760.0, -930.0]
    lane = [0, 1, 0, 1, 1, 1, 0, 0]
    v = [25.0, 24.0, 26.0, 22.0, 20.0, 23.0, 21.0, 24.0]
    cav = [True, True, True, False, False, True, False, True]
    if block_entrance:      # stopped vehicles right at every lane start: nobody can enter
        s += [ENTRY_S_M[0] + 6.0, ENTRY_S_M[1] + 6.0, ENTRY_S_M[2] + 6.0]
        lane += [0, 1, 2]
        v += [0.0, 0.0, 0.0]
        cav += [False, False, False]
    else:                   # far downstream fillers keep K equal
        s += [100.0, 110.0, 120.0]
        lane += [0, 1, 1]
        v += [25.0, 25.0, 25.0]
        cav += [False, False, False]
    k = len(s)
    return {"s_m": torch.tensor([s]), "v_mps": torch.tensor([v]), "a_mps2": torch.zeros(1, k),
            "lane": torch.tensor([lane]), "present": torch.ones(1, k, dtype=torch.bool),
            "is_cav": torch.tensor([cav]), "length_m": torch.full((1, k), 5.0)}


class Starts:
    def __init__(self, block_entrance, with_entries=True):
        self.x = _state(block_entrance)
        self.feat = torch.as_tensor(_history().to_features()).float()
        self.with_entries = with_entries

    def sample(self, batch_size, device="cpu"):
        state = {k: v.expand(batch_size, -1).clone() for k, v in self.x.items()}
        feat = self.feat.expand(batch_size, -1).clone() if self.with_entries else None
        return StartBatch(state, {}, [{} for _ in range(batch_size)], feat)


def _world_model():
    torch.manual_seed(11)
    model = WmSModel(ModelSpec(rollout_mode="hybrid", history_hidden=16, k_hidden=8),
                     hdv=HdvConfig(d_model=16, n_heads=2, n_layers=2, ffn=32))
    return model.eval()


def _trainer(model, starts, *, checkpoint_imagination=True, entries=True, counterfactual=True,
             horizon=4):
    torch.manual_seed(3)
    settings = dict(batch_size=2, horizon=horizon, checkpoint_every=0, validation_every=0,
                    actor_lr=1e-3, critic_lr=1e-3, checkpoint_imagination=checkpoint_imagination,
                    imagine_entries=entries)
    return DreamerTrainer(
        [copy.deepcopy(model)], CavPolicy(hidden_dim=16), GroupSignal(RsuGroupEncoder(hidden_dim=16)),
        TeamCritic(hidden_dim=16), starts, observation_builder=build_observations,
        reward_fn=team_reward, safety=safe_accel, controller_factory=make_cav_controller,
        config=DreamerConfig(**settings), reward_weights=RewardWeights(), lane_rules=LaneRules(),
        counterfactual=CounterfactualConfig(enabled=counterfactual, n_individual=2, n_group=2,
                                            n_flip=2, group_max_size=4))


def test_entry_slots_padding_shapes():
    batch = Starts(True).sample(2)
    padded = with_entry_slots(batch, horizon_s=10.0, pad_max=256)
    k0 = batch.state["s_m"].shape[1]
    spec = padded.entry_spec
    assert spec["k0"] == k0 and padded.state["s_m"].shape == (2, k0 + spec["pad"])
    assert not bool(padded.state["present"][:, k0:].any())
    assert bool((padded.state["lane"][:, k0:] == -1).all())
    assert spec["seeds"][0] != spec["seeds"][1]
    assert with_entry_slots(Starts(True, with_entries=False).sample(2), 10.0, 256).entry_spec is None


def test_backlog_accumulates_when_entrance_blocked_and_gradients_flow():
    model = _world_model()
    blocked = _trainer(model, Starts(True), counterfactual=False, horizon=5)
    torch.manual_seed(7)
    batch = blocked._entry_slots(blocked.start_states.sample(2))
    imagined = blocked.imagine(batch)
    queue = [state["entry_queue"] for state in imagined["states"][0][:-1]]
    assert all(q.shape == (2,) for q in queue)
    assert float(queue[-1].min()) > float(queue[0].min()) >= 0   # backlog grows
    assert float(imagined["components"]["source_wait"].sum()) > 0
    loss = -imagined["returns"].mean()
    loss.backward()
    grads = [p.grad for p in blocked.actor_parameters if p.grad is not None]
    assert grads and sum(float(g.abs().sum()) for g in grads) > 0

    open_ = _trainer(model, Starts(False), counterfactual=False, horizon=5)
    torch.manual_seed(7)
    with torch.no_grad():
        free = open_.imagine(open_._entry_slots(open_.start_states.sample(2)))
    assert float(free["components"]["source_wait"].sum()) < float(
        imagined["components"]["source_wait"].sum())
    inserted = free["states"][0][-1]["present"][:, 11:]
    assert bool(inserted.any())                                    # entrants really appear


def test_entries_off_matches_old_imagination():
    model = _world_model()
    on = _trainer(model, Starts(False, with_entries=False), counterfactual=False)
    off = _trainer(model, Starts(False), entries=False, counterfactual=False)
    torch.manual_seed(7)
    a = on.update()
    torch.manual_seed(7)
    b = off.update()
    for key in ("actor_loss", "returns", "reward", "delay"):
        assert a[key] == pytest.approx(b[key], rel=1e-6, abs=1e-9)
    assert a["source_wait"] == 0.0 and a["entries"] is False


def test_update_with_entries_counterfactuals_and_checkpointing():
    model = _world_model()
    results = []
    for checkpointing in (False, True):
        trainer = _trainer(model, Starts(True), checkpoint_imagination=checkpointing)
        trainer.update_count = 1500
        torch.manual_seed(7)
        metrics = trainer.update()
        grads = [p.grad.clone() for p in trainer.actor_parameters if p.grad is not None]
        results.append((metrics, grads))
        assert metrics["entries"] is True and metrics["source_wait"] > 0
        assert metrics["total_wait"] == pytest.approx(metrics["delay"] + metrics["source_wait"])
    for key in ("actor_loss", "returns", "source_wait", "D_all"):
        assert results[1][0][key] == pytest.approx(results[0][0][key], rel=1e-5, abs=1e-8), key
    for a, b in zip(results[0][1], results[1][1]):
        torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-7)
    detail = trainer.validate_detail(Starts(True))
    assert detail["total_wait"] == pytest.approx(detail["delay"] + detail["source_wait"])
    assert detail["total_wait_vs_native"] is not None


def test_entry_process_row_seeds_give_common_arrivals():
    hist = _history()
    k = 40
    state = {"s_m": torch.zeros(2, k), "v_mps": torch.zeros(2, k), "a_mps2": torch.zeros(2, k),
             "lane": torch.full((2, k), -1), "length_m": torch.full((2, k), 5.0),
             "present": torch.zeros(2, k, dtype=torch.bool), "is_cav": torch.zeros(2, k, dtype=torch.bool)}
    proc = TorchEntryProcessV2(histories=[hist, hist], dt_s=0.5, batch=2,
                               cfg=EntryConfig(max_pending=10 ** 9), row_seeds=[5, 5])
    for _ in range(10):
        proc.insert(state)
    assert torch.equal(state["present"][0], state["present"][1])
    assert torch.equal(state["s_m"][0], state["s_m"][1])
    assert proc.backlog().shape == (2,)
