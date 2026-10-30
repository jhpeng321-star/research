"""CPU regression for execution-cap callback state after the last CAV exits."""
from dataclasses import replace
from functools import partial

import pytest
import torch

from tests.unit.test_longmem import Critic, Starts, assert_tree_close, reward
from wms.control.bridge import make_cav_controller
from wms.control.dreamer import DreamerConfig, DreamerTrainer
from wms.control.obs import build_observations
from wms.control.policy import CavPolicy, RsuGroupEncoder
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.physics.config import PhysicsConfig


def make_trainer(monkeypatch, *, exits, memory=True):
    monkeypatch.setenv('C4_EXECUTION_PARITY', '2')
    monkeypatch.setenv('C4_SUMO_LEADER_RULE', '1')
    monkeypatch.setenv('WMS_LM300_RECOMPUTE_REWARD', '2')
    monkeypatch.setenv('WMS_LM300_PAIRBIAS_CKPT', '1')
    monkeypatch.setenv('WMS_SPEED_FUTURE_WORKERS', '1')
    monkeypatch.delenv('C4_EXECUTION_DRIVER', raising=False)
    torch.manual_seed(701)
    starts = Starts(1, 4)
    state = starts.batch.state
    state['is_cav'][:] = torch.tensor([[True, True, False, False]])
    state['s_m'][:] = torch.tensor([[550., 545., 100., 50.]])
    state['lane'][:] = torch.tensor([[1, 0, 1, 0]])
    # Both CAVs leave; HDVs remain. Exit is not a no-leader condition.
    spec = ModelSpec(history_hidden=8, k_hidden=8, rollout_mode='hybrid',
                     lane_model='hazard_v5', merge_decision='first_passage',
                     lane_decision='sample', merge_passage_law='hybrid')
    model = WmSModel(spec, physics=PhysicsConfig(exit_at_road_end=exits))
    cfg = DreamerConfig(batch_size=1, horizon=4, imagine_entries=False,
        cav_exec_cap='idm', future_samples=1, imagination_microbatch=1 if memory else 0,
        imagination_checkpoint_segment=2 if memory else 0,
        imagination_offload_futures=memory, imagination_skip_zero_vjp=memory,
        imagination_replay_dense_prefix=memory, checkpoint_imagination=False)
    policy = CavPolicy(hidden_dim=8, n_heads=2, lane_head=False, residual_on_native=True)
    trainer = DreamerTrainer([model], policy, RsuGroupEncoder(hidden_dim=8, n_heads=2),
        Critic(), starts, observation_builder=build_observations, reward_fn=reward,
        safety=lambda state, command, **kw: command, config=cfg,
        controller_factory=partial(make_cav_controller, native_lane_change=True,
            native_ramp_merge='wm_law', symmetric_exec_cap=True))
    return trainer


@pytest.mark.parametrize('memory', [False, True])
@pytest.mark.parametrize('reenter', [False, True])
def test_imagine_execution_cap_survives_last_cav_exit(monkeypatch, memory, reenter):
    trainer = make_trainer(monkeypatch, exits=True, memory=memory)
    model = trainer.world_models[0]
    seen = []
    original = model._apply_cav_overrides

    if reenter:
        class Entrants:
            def __init__(self, tick):
                self.tick = tick

            def insert(self, state):
                new = torch.zeros_like(state['present'])
                if self.tick == 3:
                    new[:, 0] = True
                    state['present'][:, 0] = True
                    state['is_cav'][:, 0] = True
                    state['s_m'][:, 0] = 90.
                    state['v_mps'][:, 0] = 8.
                    state['lane'][:, 0] = 1
                self.tick += 1
                return new

        # A fresh process is needed on each full-forward/replay rollout.
        original_rollout = model.rollout

        def rollout(*args, k0=0, history=None, **kwargs):
            return original_rollout(*args, k0=k0, history=history,
                                    **kwargs, entries=Entrants(k0))

        monkeypatch.setattr(model, 'rollout', rollout)

    def observe(*args, **kwargs):
        cav = args[2]
        seen.append((bool(torch.is_grad_enabled()), bool(cav.any()), kwargs['exec_cap']))
        return original(*args, **kwargs)

    monkeypatch.setattr(model, '_apply_cav_overrides', observe)
    imagined = trainer.imagine(trainer.start_states.batch, deterministic=True)
    assert torch.isfinite(imagined['rewards']).all()
    imagined['returns'].sum().backward()
    assert any(not cav for _, cav, _ in seen)
    assert all(cap is not None and torch.isfinite(cap).all() for _, _, cap in seen)
    if memory:
        assert any(not grad and not cav for grad, cav, _ in seen)
        assert any(grad and not cav for grad, cav, _ in seen)
    assert model.cav_exec_cap == 'off'  # trainer restores the member switch
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in trainer.actor_parameters)


@pytest.mark.parametrize('entrant_only', [False, True])
def test_initial_empty_and_entrant_only_cap_is_real(monkeypatch, entrant_only):
    from wms.model.cav import EXEC_CAP_KEY
    trainer = make_trainer(monkeypatch, exits=False, memory=False)
    state = trainer.start_states.batch.state
    state['entrant'] = state['is_cav'].clone() if entrant_only else torch.zeros_like(state['is_cav'])
    if not entrant_only:
        state['is_cav'].zero_()
    model = trainer.world_models[0]
    model.cav_exec_cap = 'idm'
    seen = []

    def controller(current):
        cap = current[EXEC_CAP_KEY]
        params = current['inferred_driver_parameters']
        from wms.model.driver_encoder import PARAMETER_NAMES
        from wms.control.execution import inferred_follow_cap
        expected = inferred_follow_cap(current, {name: params[..., i]
                                      for i, name in enumerate(PARAMETER_NAMES)})
        assert torch.equal(cap, expected)
        seen.append(True)
        return model.hold_teacher(current)

    model.rollout(state, n_steps=2, cav_controller=controller)
    assert len(seen) == 2


def test_nonempty_cav_forward_and_actor_gradients_bitwise_legacy(monkeypatch):
    baseline = make_trainer(monkeypatch, exits=False)
    actual = make_trainer(monkeypatch, exits=False)
    old_cap = baseline.world_models[0]._exec_cap

    def legacy_cap(out, state, params):
        # Every step in the previously working domain takes the old cap branch.
        assert (state['present'] & state['is_cav']).any()
        return old_cap(out, state, params)

    monkeypatch.setattr(baseline.world_models[0], '_exec_cap', legacy_cap)
    outputs = []
    for trainer in (baseline, actual):
        torch.manual_seed(29)
        result = trainer.imagine(trainer.start_states.batch, deterministic=True)
        result['returns'].sum().backward()
        outputs.append(result)
    assert_tree_close(outputs[0], outputs[1], 0, 0)
    for x, y in zip(baseline.actor_parameters, actual.actor_parameters):
        assert (x.grad is None and y.grad is None) or torch.equal(x.grad, y.grad)


def test_disabled_cap_stays_disabled(monkeypatch):
    trainer = make_trainer(monkeypatch, exits=True, memory=False)
    trainer.config = replace(trainer.config, cav_exec_cap='off')
    with pytest.raises(KeyError, match='cav_exec_cap_mps2'):
        trainer.imagine(trainer.start_states.batch, deterministic=True)
