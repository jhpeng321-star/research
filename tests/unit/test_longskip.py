"""CPU-only exact replay skips: real Dreamer outputs, synthetic starts, no data."""
from dataclasses import replace
import json

import pytest
import torch

from tests.unit.test_longgrad import make_long_trainer
from tests.unit.test_longmem import assert_tree_close


FLAGS = ('imagination_skip_zero_vjp', 'imagination_replay_dense_prefix')


def make(tmp_path, monkeypatch, *, flags, balance, gate_on, segment=2):
    from tests.unit.test_gated300 import cct2_authority, enable_gain_only
    t = make_long_trainer(tmp_path, monkeypatch, memory=True, balance=balance)
    enable_gain_only(t, cct2_authority(tmp_path, hg=300))
    t.config = replace(t.config, imagination_checkpoint_segment=segment,
                       **{k: k in flags for k in FLAGS})
    if not gate_on:
        original = t.reward_fn
        def reward(*args):
            out = original(*args)
            return {**out, 'reward': -out['reward']}
        t.reward_fn = reward
    return t


def test_default_off_and_boolean_validation():
    from wms.control.dreamer import DreamerConfig
    for key in FLAGS:
        assert getattr(DreamerConfig(), key) is False
        for bad in (0, 1, None, 'true'):
            with pytest.raises(ValueError, match=key):
                DreamerConfig(**{key: bad})


def test_combined_preset_only_adds_exact_skip_flags_and_binds_source():
    from pathlib import Path
    from tools.candA.authority import MANIFEST_FILES, effective_config
    from wms.control.train_cli import _read_config
    root = Path(__file__).resolve().parents[2]
    baseline = _read_config(root / 'configs/c4_policy14_candA_longgrad.toml')
    combined = _read_config(root / 'configs/c4_policy14_candA_longgrad_skip.toml')
    baseline['trainer'].update({key: True for key in FLAGS})
    assert baseline == combined
    assert effective_config(root, 'configs/c4_policy14_candA_longgrad_skip.toml') == combined
    assert 'src/wms/control/longskip.py' in MANIFEST_FILES


@pytest.mark.parametrize('flags', [(FLAGS[0],), (FLAGS[1],), FLAGS])
@pytest.mark.parametrize('balance', [False, True])
@pytest.mark.parametrize('gate_on', [False, True])
@pytest.mark.parametrize('mode', ['0', '2'])
def test_parameter_gradients_updates_rng_and_registered_credits_bitwise(
        tmp_path, monkeypatch, flags, balance, gate_on, mode):
    monkeypatch.setenv('WMS_LM300_RECOMPUTE_REWARD', mode)
    baseline = make(tmp_path, monkeypatch, flags=(), balance=balance, gate_on=gate_on)
    actual = make(tmp_path, monkeypatch, flags=flags, balance=balance, gate_on=gate_on)
    torch.manual_seed(29)
    a = baseline.update()
    rng = torch.get_rng_state()
    torch.manual_seed(29)
    b = actual.update()
    assert torch.equal(rng, torch.get_rng_state())
    for key in ('actor_loss', 'returns', 'return_scale', 'gated300_active_rows',
                'gated300_rows_on_share', 'gated300_long_credit',
                'gated300_long_output_cotangent', 'grad_norm_dense', 'grad_norm_long',
                'grad_long_scale', 'actor_grad_norm'):
        assert a[key] == b[key], key
    for x, y in zip(baseline.actor_parameters, actual.actor_parameters):
        assert (x.grad is None and y.grad is None) or torch.equal(x.grad, y.grad)
        assert torch.equal(x, y)
    assert_tree_close(baseline.actor_optimizer.state_dict(), actual.actor_optimizer.state_dict(), 0, 0)
    assert_tree_close(baseline.critic_optimizer.state_dict(), actual.critic_optimizer.state_dict(), 0, 0)
    assert a['futures_replayed'] == (16 if balance else 24)
    long = [r for r in b['replay_horizon_s'] if r['phase'] == 'long']
    if FLAGS[0] in flags:
        assert sum(bool(r['replay_horizon_s']) for r in long) <= (2 if gate_on else 0)
    if FLAGS[1] in flags:
        dense = [r for r in b['replay_horizon_s'] if r['phase'] == 'dense']
        assert len(dense) == 8
        assert all(r['replay_horizon_s'] == 3 for r in dense)  # 2 + next-input boundary
    assert b['replay_seconds_saved_estimate'] >= 0
    json.dumps(b, allow_nan=False)


@pytest.mark.parametrize('forward', [False, True])
def test_actual_worker_vjp_and_forward_transport_bitwise(tmp_path, monkeypatch, forward):
    from wms.control import parallel_futures as pf
    baseline = make(tmp_path, monkeypatch, flags=(), balance=True, gate_on=True)
    actual = make(tmp_path, monkeypatch, flags=FLAGS, balance=True, gate_on=True)
    torch.manual_seed(29)
    a = baseline.update()
    class CpuPool:
        def __init__(self, trainer, common):
            self.trainer, self.common = pf._snapshot(trainer), common
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def map(self, tasks):
            for task in tasks:
                yield (pf._forward(self.trainer, self.common, task) if self.common['kind'] == 'forward'
                       else pf._vjp(self.trainer, self.common, task))
    monkeypatch.setattr(pf, 'FuturePool', CpuPool)
    monkeypatch.setattr(pf, 'worker_count', lambda *args: 2)
    monkeypatch.setattr(pf, 'parallel_forward', lambda: forward)
    torch.manual_seed(29)
    b = actual.update()
    for key in ('actor_loss', 'grad_norm_dense', 'grad_norm_long', 'actor_grad_norm'):
        assert a[key] == b[key]
    for x, y in zip(baseline.actor_parameters, actual.actor_parameters):
        assert torch.equal(x.grad, y.grad) and torch.equal(x, y)


@pytest.mark.parametrize('prior_longgrad', [False, True])
def test_new_phase_requires_exact_skip_keys_and_plain_resume_restores(
        tmp_path, monkeypatch, prior_longgrad):
    monkeypatch.setenv('C4_ALLOWED_ROOTS', str(tmp_path))
    old = make(tmp_path, monkeypatch, flags=(), balance=prior_longgrad, gate_on=False)
    old.config = replace(old.config, long_grad_reset_actor_adam=prior_longgrad)
    old.update()
    checkpoint = tmp_path / 'before.pt'
    old.save_checkpoint(checkpoint)
    if not prior_longgrad:
        historical = torch.load(checkpoint, weights_only=True)
        for key in (*FLAGS, 'long_grad_balance', 'long_grad_reset_actor_adam'):
            historical['config'].pop(key)  # actual pre-longgrad checkpoint defaults
        torch.save(historical, checkpoint)
    new = make(tmp_path, monkeypatch, flags=FLAGS, balance=True, gate_on=False)
    new.config = replace(new.config, long_grad_reset_actor_adam=True)
    with pytest.raises(ValueError, match='imagination_'):
        new.load_checkpoint(checkpoint, new_phase=True)
    allow = ['trainer.' + k for k in FLAGS]
    if not prior_longgrad:
        allow += ['trainer.long_grad_balance', 'trainer.long_grad_reset_actor_adam']
    new.load_checkpoint(checkpoint, new_phase=True, resume_allow=allow)
    assert set(new.resume_config_receipt['differences']) == set(allow)
    path = tmp_path / 'after.pt'
    new.save_checkpoint(path)
    restored = make(tmp_path, monkeypatch, flags=(), balance=True, gate_on=False)
    restored.load_checkpoint(path)
    assert all(getattr(restored.config, key) for key in FLAGS)


def test_planner_checks_each_future_and_microbatch_and_fails_closed(tmp_path, monkeypatch):
    from torch.utils._pytree import tree_flatten
    from wms.control.longskip import replay_horizon
    t = make(tmp_path, monkeypatch, flags=FLAGS, balance=True, gate_on=False, segment=8)
    t._training_imagination = True
    batch = t.start_states.batch
    cfg = replace(t.config, horizon=10)
    values = {'returns': torch.zeros(8, 2, 3),
              'gated300_long_difference': torch.zeros(8, 1, 3)}
    def plan(future, rows):
        vectors, spec = tree_flatten(values)
        return replay_horizon(t, batch, cfg, {'items': [None] * 8}, range(len(vectors)),
                              vectors, spec, future, torch.tensor(rows))
    values['returns'][1, :, 1] = 1
    assert plan(0, [1]) == 0
    assert plan(1, [0]) == 0
    assert plan(1, [1]) == 4  # align to original 8-physics-step segments
    values['gated300_long_difference'][1, 0, 2] = 1
    assert plan(1, [1]) == 4
    assert plan(1, [2]) == 10
    values['gated300_long_difference'][1, 0, 2] = float('nan')
    assert plan(1, [2]) == 10
    values['new_unproved_output'] = torch.ones(())
    assert plan(0, [0]) == 10


@pytest.mark.parametrize('unused', [False, True])
def test_all_zero_backward_preserves_materialized_zero_vs_unused_none(
        tmp_path, monkeypatch, unused):
    from wms.control.counterfactual import build_variants
    from wms.control.longgrad import actor_backward
    def run(flags):
        t = make(tmp_path, monkeypatch, flags=flags, balance=True, gate_on=False)
        if unused:
            t.actor_parameters.append(torch.nn.Parameter(torch.ones(())))
        t._training_imagination = True
        batch = t.start_states.batch
        variants = build_variants(batch.state, t.counterfactual)
        torch.manual_seed(29)
        out = t.imagine(batch, variants=variants)
        zero = out['returns'].sum() * 0
        metrics = actor_backward(t, zero, None, zero)
        return [p.grad for p in t.actor_parameters], metrics
    a, _ = run(())
    b, metrics = run(FLAGS)
    for x, y in zip(a, b):
        assert (x is None and y is None) or torch.equal(x, y)
    assert metrics['futures_replayed'] == (8 if unused else 1)
    if unused:
        assert b[-1] is None  # no false zero gradient / unintended Adam update


def test_hybrid_wm_residual_actor_entries_and_next_state_boundary_bitwise(tmp_path, monkeypatch):
    from functools import partial
    import numpy as np
    from tests.unit.test_gated300 import cct2_authority, enable_gain_only
    from tests.unit.test_longmem import Starts, Critic, queue_reward
    from wms.control import gated300
    from wms.control.bridge import make_cav_controller
    from wms.control.counterfactual import CounterfactualConfig
    from wms.control.dreamer import DreamerConfig, DreamerTrainer
    from wms.control.obs import build_observations
    from wms.control.policy import CavPolicy, RsuGroupEncoder
    from wms.model.entries import EntranceHistory
    from wms.model.spec import ModelSpec
    from wms.model.wms import WmSModel
    monkeypatch.setattr(gated300, 'STEP_S', 30.)
    monkeypatch.setenv('WMS_LM300_RECOMPUTE_REWARD', '2')
    monkeypatch.setenv('WMS_LM300_PAIRBIAS_CKPT', '1')
    def build(flags):
        torch.manual_seed(701)
        starts = Starts(1, 4)
        starts.batch.state['is_cav'][:, -1] = False
        starts.batch.state['s_m'][:, -1] = 10.
        starts.batch.state['lane'][:, 1] = 1
        hist = EntranceHistory.defaults()
        hist.rate[:] = 1.
        starts.batch.entry_feat = torch.tensor(np.stack([hist.to_features()])).float()
        cfg = DreamerConfig(batch_size=1, horizon=2, imagine_entries=True, entry_pad_max=3,
            source_wait_grad=True, entrant_control=True, deterministic_crn=True,
            native_residual_train_group=True, native_residual_train_trunk=True,
            future_samples=8, pessimistic_credit=True, future_driver_scale=.05,
            future_accel_noise=True, imagination_microbatch=1, imagination_checkpoint_segment=2,
            imagination_offload_futures=True, imagination_offload_saved_tensors=True,
            long_grad_balance=True, **{k: k in flags for k in FLAGS})
        spec = ModelSpec(history_hidden=8, k_hidden=8, rollout_mode='hybrid',
            grad_decay_per_step=.9, lane_model='hazard_v5', merge_decision='first_passage',
            lane_decision='sample', merge_passage_law='hybrid')
        policy = CavPolicy(hidden_dim=8, n_heads=2, lane_head=False, residual_on_native=True)
        with torch.no_grad():
            policy.residual_head.weight.normal_(std=.01)
        t = DreamerTrainer([WmSModel(spec)], policy, RsuGroupEncoder(hidden_dim=8, n_heads=2),
            Critic(), starts, observation_builder=build_observations, reward_fn=queue_reward,
            safety=lambda state, command, **kw: command, config=cfg,
            controller_factory=partial(make_cav_controller, native_lane_change=True,
                native_ramp_merge='wm_law', symmetric_exec_cap=True),
            counterfactual=CounterfactualConfig(enabled=True, max_starts=1, n_individual=0,
                n_group=1, n_flip=0, group_min_size=3, group_max_size=3,
                w_team=[0., 0.]))
        t.long_credit_requested = True
        enable_gain_only(t, cct2_authority(tmp_path, hg=300, entrant_control=True))
        return t
    baseline, actual = build(()), build(FLAGS)
    torch.manual_seed(313)
    a = baseline.update()
    rng = torch.get_rng_state()
    torch.manual_seed(313)
    b = actual.update()
    assert a['entries'] and b['entries']
    assert a['gated300_rows_on_share'] == b['gated300_rows_on_share'] == 0
    assert torch.equal(rng, torch.get_rng_state())
    for key in ('actor_loss', 'returns', 'grad_norm_dense', 'grad_norm_long', 'actor_grad_norm'):
        assert a[key] == b[key], key
    for x, y in zip(baseline.actor_parameters, actual.actor_parameters):
        assert (x.grad is None and y.grad is None) or torch.equal(x.grad, y.grad)
        assert torch.equal(x, y)
    for module in (actual.policy.residual_head, actual.policy.own_encoder,
                   actual.policy.report_encoder, actual.policy.attention, actual.group_encoder):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert_tree_close(baseline.actor_optimizer.state_dict(), actual.actor_optimizer.state_dict(), 0, 0)
    assert b['futures_replayed'] == 8
    assert all(r['replay_horizon_s'] == (3 if r['phase'] == 'dense' else 0)
               for r in b['replay_horizon_s'])
