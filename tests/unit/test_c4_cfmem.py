"""Memory-saving imagination (CFMEM_NOTES.md): activation checkpointing must leave the C4
objective and every gradient unchanged and must never replay the stateful CAV controller.

CPU only, tiny randomly initialised WM-S; no data or checkpoint access."""
from __future__ import annotations

import copy

import pytest
import torch

from wms.control.bridge import make_cav_controller
from wms.control.contract import RewardWeights
from wms.control.counterfactual import CounterfactualConfig
from wms.control.dreamer import DreamerConfig, DreamerTrainer
from wms.control.lanes import LaneRules
from wms.control.obs import build_observations
from wms.control.policy import CavPolicy, RsuGroupEncoder, TeamCritic
from wms.control.reward import team_reward
from wms.control.safety import safe_accel
from wms.control.start_states import StartBatch
from wms.control.train_cli import GroupSignal
from wms.model.residual_net import HdvConfig
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _state():
    s = [-900.0, -860.0, -820.0, -700.0, -880.0, -790.0, -760.0, -930.0]
    lane = [0, 1, 0, 1, 1, 1, 0, 0]
    v = [25.0, 24.0, 26.0, 22.0, 20.0, 23.0, 21.0, 24.0]
    cav = [True, True, True, False, False, True, False, True]
    k = len(s)
    return {"s_m": torch.tensor([s]), "v_mps": torch.tensor([v]), "a_mps2": torch.zeros(1, k),
            "lane": torch.tensor([lane]), "present": torch.ones(1, k, dtype=torch.bool),
            "is_cav": torch.tensor([cav]), "length_m": torch.full((1, k), 5.0)}


class Starts:
    def __init__(self, x):
        self.x = x

    def sample(self, batch_size, device="cpu"):
        return StartBatch({k: v.expand(batch_size, -1).clone() for k, v in self.x.items()}, {})


def _world_model(mode):
    torch.manual_seed(11)
    model = WmSModel(ModelSpec(rollout_mode=mode, history_hidden=16, k_hidden=8),
                     hdv=HdvConfig(d_model=16, n_heads=2, n_layers=2, ffn=32))
    return model.eval()


def _trainer(model, checkpoint_imagination, lanes=True, counterfactual=True):
    torch.manual_seed(3)
    settings = dict(batch_size=2, horizon=3, checkpoint_every=0, validation_every=0,
                    actor_lr=1e-3, critic_lr=1e-3, checkpoint_imagination=checkpoint_imagination)
    return DreamerTrainer(
        [copy.deepcopy(model)], CavPolicy(hidden_dim=16), GroupSignal(RsuGroupEncoder(hidden_dim=16)),
        TeamCritic(hidden_dim=16), Starts(_state()), observation_builder=build_observations,
        reward_fn=team_reward, safety=safe_accel, controller_factory=make_cav_controller,
        config=DreamerConfig(**settings), reward_weights=RewardWeights(),
        lane_rules=LaneRules() if lanes else None,
        counterfactual=CounterfactualConfig(enabled=counterfactual, n_individual=2, n_group=2,
                                            n_flip=2, group_max_size=4))


def _update_with_grads(trainer):
    torch.manual_seed(7)
    trainer.update_count = 1500                    # every row kind carries weight
    metrics = trainer.update()
    actor = [p.grad.clone() if p.grad is not None else None for p in trainer.actor_parameters]
    critic = [p.grad.clone() if p.grad is not None else None for p in trainer.critic.parameters()]
    return metrics, actor, critic


@pytest.mark.parametrize("mode,lanes,counterfactual", [("hybrid", True, True),
                                                       ("residual", True, True),
                                                       ("residual", False, False)])
def test_checkpointed_imagination_has_identical_objective_and_gradients(mode, lanes, counterfactual):
    model = _world_model(mode)
    reference = _update_with_grads(_trainer(model, False, lanes, counterfactual))
    checkpointed = _update_with_grads(_trainer(model, True, lanes, counterfactual))
    for key in ("actor_loss", "critic_loss", "returns", "reward", "entropy", "cost_ttc"):
        assert checkpointed[0][key] == pytest.approx(reference[0][key], rel=1e-5, abs=1e-8), key
    if counterfactual:
        for key in ("D_all", "D_i", "D_G", "C_G", "M_i", "cf_rows"):
            assert checkpointed[0][key] == pytest.approx(reference[0][key], rel=1e-5, abs=1e-8), key
    nonzero = 0
    for grads_ref, grads_ckpt in zip(reference[1:], checkpointed[1:]):
        assert len(grads_ref) == len(grads_ckpt)
        for a, b in zip(grads_ref, grads_ckpt):
            assert (a is None) == (b is None)
            if a is not None:
                torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-8)
                nonzero += int(a.abs().sum() > 0)
    assert nonzero > 4                             # gradients really flow (actor and critic)


def _saved_bytes(trainer):
    """Bytes autograd stores outside checkpoints during one imagination (unique storages)."""
    seen = {}

    def pack(tensor):
        storage = tensor.untyped_storage()
        seen[(storage.data_ptr(), storage.nbytes())] = storage.nbytes()
        return tensor

    batch = trainer.start_states.sample(2)
    from wms.control.counterfactual import build_variants
    variants = build_variants(batch.state, trainer.counterfactual)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        trainer.imagine(batch, variants=variants)
    return sum(seen.values())


def test_checkpointing_cuts_stored_activations_and_restores_the_wm_switch():
    model = _world_model("hybrid")
    plain, checkpointed = _trainer(model, False), _trainer(model, True)
    torch.manual_seed(1)
    stored_plain = _saved_bytes(plain)
    torch.manual_seed(1)
    stored_checkpointed = _saved_bytes(checkpointed)
    assert stored_checkpointed < 0.5 * stored_plain
    assert checkpointed.world_models[0].checkpoint_forward is False
    with torch.no_grad():                          # no graph: the switch is a no-op
        checkpointed.validate(checkpointed.start_states)
    assert DreamerConfig().checkpoint_imagination is True       # memory-safe default
    with pytest.raises(ValueError):
        DreamerConfig(checkpoint_imagination=1)


def test_wm_forward_checkpoint_never_replays_the_controller():
    model = _world_model("hybrid")
    x = {k: v.expand(2, -1).clone() for k, v in _state().items()}
    x["v_mps"].requires_grad_(True)
    calls = []

    def run(flag):
        calls.clear()
        policy = CavPolicy(hidden_dim=8, lane_head=False)
        torch.manual_seed(5)
        controller = make_cav_controller(policy, RsuGroupEncoder(hidden_dim=8), safe_accel)

        def counted(state):
            calls.append(1)
            return controller(state)

        model.checkpoint_forward = flag
        try:
            out = model.rollout(x, n_steps=4, cav_controller=counted)
            loss = out["speeds"].square().sum() + out["positions"].sum()
            grad, = torch.autograd.grad(loss, x["v_mps"])
        finally:
            model.checkpoint_forward = False
        return out["speeds"].detach(), grad, len(calls)

    torch.manual_seed(2)
    speeds_ref, grad_ref, calls_ref = run(False)
    torch.manual_seed(2)
    speeds, grad, n_calls = run(True)
    assert calls_ref == n_calls == 4               # exactly once per physics step, no replay
    torch.testing.assert_close(speeds, speeds_ref)
    torch.testing.assert_close(grad, grad_ref, rtol=1e-5, atol=1e-8)
