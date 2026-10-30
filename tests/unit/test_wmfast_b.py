"""Exact ES batching proofs: synthetic inputs only; no optimizer or training run."""
import copy
from dataclasses import replace

import pytest
import torch
from torch.utils._pytree import tree_flatten

from tests.unit.test_eslong import make, capture
from wms.control import eslong, gated300
from wms.control.dreamer import _rng_state


def tensor_leaves(value):
    return [x for x in tree_flatten(value)[0] if torch.is_tensor(x)]


def common_context(trainer):
    out, context = capture(trainer)
    dimension = sum(p.numel() for p in eslong.head_parameters(trainer))
    directions = torch.randn(2, dimension, generator=torch.Generator().manual_seed(7000027))
    return out, {**context, 'kind': 'forward', 'eslong': True,
                 'directions': directions, 'sigma': .08}


@pytest.mark.parametrize('sides', [2, 4])
@pytest.mark.parametrize('starts', [1, 2])
def test_all_long_tensors_and_estimator_bitwise(tmp_path, monkeypatch, sides, starts):
    from tests.unit.test_longmem import Starts
    trainer = make(tmp_path, monkeypatch)
    trainer.start_states = Starts(starts, 4)
    trainer.config = replace(trainer.config, batch_size=starts)
    trainer.counterfactual = replace(trainer.counterfactual, max_starts=starts)
    imagined, common = common_context(trainer)
    original_prefix, full = gated300.prefix, []
    def prefix(owner, out, dense):
        full.append([x.detach().clone() for x in tensor_leaves(out)])
        return original_prefix(owner, out, dense)
    monkeypatch.setattr(gated300, 'prefix', prefix)
    order = [(i, sign) for i in range(2) for sign in (1, -1)]
    expected = [eslong.forward_task(trainer, common, (*side, future), full_output=True)
                for side in order for future in range(8)]
    reference_full = full[:]
    full.clear()
    actual = []
    reordered_full = []
    for start in range(0, 4, sides):
        group = [eslong.forward_stacked_task(trainer, common, (order[start:start+sides], k),
                                            full_output=True) for k in range(8)]
        actual.extend(group[k][s] for s in range(sides) for k in range(8))
        offset = start * 8
        reordered_full.extend(full[offset + k*sides+s] for s in range(sides) for k in range(8))
    assert len(reference_full) == len(reordered_full) == 32
    for reference, candidate in zip(reference_full, reordered_full):
        assert len(reference) == len(candidate)
        for index, (x, y) in enumerate(zip(reference, candidate)):
            assert torch.equal(x, y), (index, x.shape, float((x.double()-y.double()).abs().max()))
    for reference, candidate in zip(expected, actual):
        for x, y in zip(tensor_leaves(reference), tensor_leaves(candidate)):
            assert torch.equal(x, y)
    from wms.control.counterfactual import row_coefficients
    coeff = row_coefficients(imagined['variants'], trainer.counterfactual.weights(0),
                             imagined['returns'].shape[-1], imagined['n_base'])
    def estimator(outputs):
        deltas, receipts = [], []
        for i in range(2):
            plus, minus = [torch.cat([x['gated300_long_difference']
                for x in outputs[(2*i+s)*8:(2*i+s+1)*8]]) for s in range(2)]
            delta, receipt = eslong.paired_loss_difference(trainer, plus, minus, imagined, coeff, 1.)
            deltas.append(delta)
            receipts.append(receipt)
        return eslong.estimate(common['directions'], torch.stack(deltas), .08), receipts
    g, receipt = estimator(expected)
    other, other_receipt = estimator(actual)
    assert torch.equal(g, other) and receipt == other_receipt


@pytest.mark.parametrize('sides', [2, 4])
def test_gradient_rng_parameters_and_context_restoration(tmp_path, monkeypatch, sides):
    trainer = make(tmp_path, monkeypatch)
    out, context = capture(trainer)
    saved = copy.deepcopy(trainer.policy.state_dict())
    rng = _rng_state(trainer.device)
    monkeypatch.setenv('WMS_SPEED_ES_STACK_SIDES', '0')
    expected, metrics = eslong.gradient(trainer, out, trainer.counterfactual.weights(0), 1.)
    trainer._eslong_context = context
    monkeypatch.setenv('WMS_SPEED_ES_STACK_SIDES', str(sides))
    actual, stacked = eslong.gradient(trainer, out, trainer.counterfactual.weights(0), 1.)
    for x, y in zip(expected, actual):
        assert (x is None and y is None) or torch.equal(x, y)
    for key in metrics:
        assert metrics[key] == stacked[key]
    assert stacked['eslong_stacked_forward_calls'] == 32 // sides
    assert stacked['eslong_accepted_pairs'] > 0
    assert any(x is not None and bool(x.any()) for x in actual)
    assert trainer._eslong_context is None
    assert all(torch.equal(saved[k], v) for k, v in trainer.policy.state_dict().items())
    assert all(torch.equal(x, y) for x, y in zip(tensor_leaves(rng),
                                               tensor_leaves(_rng_state(trainer.device))))


def test_stack_validation_and_exception_restores_head(tmp_path, monkeypatch):
    trainer = make(tmp_path, monkeypatch)
    _, common = common_context(trainer)
    for bad in ('1', '3', '-2', 'true'):
        monkeypatch.setenv('WMS_SPEED_ES_STACK_SIDES', bad)
        with pytest.raises(ValueError, match='0, 2 or 4'):
            eslong.stack_sides()
    head = trainer.policy.residual_head
    variants, _, owners, selections = eslong.stacked_variants(common['kwargs']['variants'], 2)
    with pytest.raises(RuntimeError, match='injected'):
        with eslong.row_heads(trainer, common, [(0, 1), (0, -1)], selections, owners):
            raise RuntimeError('injected')
    assert trainer.policy.residual_head is head


def test_actual_stacked_worker_transport(tmp_path, monkeypatch):
    from wms.control.parallel_futures import FuturePool
    trainer = make(tmp_path, monkeypatch)
    trainer.output_dir = tmp_path / 'pool'
    _, common = common_context(trainer)
    common['eslong_stack'] = True
    tasks = [([(0, 1), (0, -1), (1, 1), (1, -1)], k) for k in (0, 7)]
    expected = [eslong.forward_stacked_task(trainer, common, task) for task in tasks]
    monkeypatch.setenv('WMS_SPEED_FORWARD_WORKERS', '2')
    with FuturePool(trainer, common) as pool:
        actual = list(pool.map(tasks))
    for x, y in zip(tensor_leaves(expected), tensor_leaves(actual)):
        assert torch.equal(x, y)


def real_fixture(tmp_path, monkeypatch, device='cpu'):
    """Small real hybrid WM, synthetic histories, actual births and learned mix."""
    import numpy as np
    from torch.utils._pytree import tree_map
    from tests.unit.test_speedprof import real_trainer
    from wms.control.coopw import ExplorationConfig
    from wms.control.dreamer import with_entry_slots
    from wms.model.entries import EntranceHistory
    trainer = real_trainer(tmp_path, monkeypatch)
    monkeypatch.setattr(gated300, 'STEP_S', 30.)  # 10 real seconds, registered K8 algebra
    trainer.gated300['config'].update(estimator='antithetic', es_sigma=.08, es_directions=2)
    trainer.config = replace(trainer.config, imagination_offload_futures=True,
        imagination_checkpoint_segment=2, imagine_entries=True, entry_pad_max=2,
        entrant_learned_mix=True, future_driver_scale=.05, future_accel_noise=True)
    trainer.exploration = ExplorationConfig(enabled=True, mode='additive_accel', rho=[.1])
    trainer.policy.residual_additive_exploration = True
    batch = trainer.start_states.batch
    batch.state['is_cav'][:, -1] = False
    batch.state['s_m'][:, -1] = 10.
    batch.state['lane'][:, 1] = 1
    history = EntranceHistory.defaults()
    history.rate[:] = 1.
    batch.entry_feat = torch.tensor(np.stack([history.to_features()])).float()
    model = trainer.world_models[0]
    batch.rollout_kwargs.update(
        driver_history=torch.zeros(1, 4, model.history_len, model.history_features),
        driver_history_mask=torch.ones(1, 4, model.history_len, dtype=torch.bool))
    trainer.start_states.batch = with_entry_slots(batch, 10., 2,
        torch.Generator().manual_seed(41))
    if device == 'cuda':
        trainer.device = torch.device('cuda')
        for module in (*trainer.world_models, trainer.policy, trainer.group_encoder,
                       trainer.critic, trainer.target_critic):
            module.to('cuda')
        batch = trainer.start_states.batch
        batch.state = tree_map(lambda x: x.to('cuda'), batch.state)
        batch.rollout_kwargs = tree_map(lambda x: x.to('cuda'), batch.rollout_kwargs)
    return trainer


@pytest.mark.parametrize('sides', [2, 4])
def test_real_wm_entrants_full_outputs(tmp_path, monkeypatch, sides):
    trainer = real_fixture(tmp_path, monkeypatch)
    _, common = common_context(trainer)
    order = [(i, sign) for i in range(2) for sign in (1, -1)]
    original_prefix, full = gated300.prefix, []
    def prefix(owner, out, dense):
        full.append([x.detach().clone() for x in tensor_leaves(out)])
        return original_prefix(owner, out, dense)
    monkeypatch.setattr(gated300, 'prefix', prefix)
    expected = [eslong.forward_task(trainer, common, (*side, 7), full_output=True) for side in order]
    reference = full[:]
    full.clear()
    actual = []
    for start in range(0, 4, sides):
        actual.extend(eslong.forward_stacked_task(trainer, common, (order[start:start+sides], 7),
                                                  full_output=True))
    for left, right in zip(reference, full):
        assert len(left) == len(right)
        for x, y in zip(left, right):
            # Changing WM batch GEMM geometry can reassociate float32 operations.
            # This is bounded explicitly; discrete outcomes must remain bitwise.
            if x.is_floating_point():
                torch.testing.assert_close(x, y, atol=2e-5, rtol=2e-6)
            else:
                assert torch.equal(x, y)
    for left, right in zip(expected, actual):
        for key in ('gated300_long_difference', 'gated300_long_J'):
            torch.testing.assert_close(left[key], right[key], atol=2e-5, rtol=2e-6)
    assert any(bool(s['entrant'].any()) for s in expected[0]['states'][0] if 'entrant' in s)


def test_300_seconds_full_tensor_parity(tmp_path, monkeypatch):
    trainer = make(tmp_path, monkeypatch)
    monkeypatch.setattr(gated300, 'STEP_S', 1.)  # Actual 300 controls / 600 physics ticks
    _, common = common_context(trainer)
    original_prefix, full = gated300.prefix, []
    def prefix(owner, out, dense):
        full.append([x.detach().clone() for x in tensor_leaves(out)])
        return original_prefix(owner, out, dense)
    monkeypatch.setattr(gated300, 'prefix', prefix)
    order = [(i, sign) for i in range(2) for sign in (1, -1)]
    expected = [eslong.forward_task(trainer, common, (*side, 7), full_output=True) for side in order]
    reference = full[:]
    full.clear()
    actual = eslong.forward_stacked_task(trainer, common, (order, 7), full_output=True)
    for left, right in zip(reference, full):
        assert len(left) == len(right)
        assert all(torch.equal(x, y) for x, y in zip(left, right))
    for x, y in zip(tensor_leaves(expected), tensor_leaves(actual)):
        assert torch.equal(x, y)


def test_execution_switch_across_checkpoint_preserves_adam_and_gradient(tmp_path, monkeypatch):
    from tests.unit.test_longmem import assert_tree_close
    trainer = make(tmp_path, monkeypatch)
    monkeypatch.setenv('C4_ALLOWED_ROOTS', str(tmp_path))
    # Non-empty known moments without performing a training/optimizer step.
    for p in trainer.actor_parameters:
        trainer.actor_optimizer.state[p] = dict(step=torch.tensor(7.),
            exp_avg=torch.full_like(p, .1), exp_avg_sq=torch.full_like(p, .2))
    path = tmp_path / 'resume.pt'
    trainer.save_checkpoint(path)
    restored = make(tmp_path, monkeypatch, 'pathwise')
    monkeypatch.setenv('WMS_SPEED_ES_STACK_SIDES', '4')
    restored.load_checkpoint(path)
    assert trainer.gated300 == restored.gated300
    assert trainer.update_count == restored.update_count
    assert_tree_close(trainer.actor_optimizer.state_dict(), restored.actor_optimizer.state_dict(), 0, 0)
    monkeypatch.setenv('WMS_SPEED_ES_STACK_SIDES', '0')
    torch.manual_seed(79)
    original, _ = capture(trainer)
    expected, metrics = eslong.gradient(trainer, original, trainer.counterfactual.weights(0), 1.)
    monkeypatch.setenv('WMS_SPEED_ES_STACK_SIDES', '4')
    torch.manual_seed(79)
    out, _ = capture(restored)
    actual, receipt = eslong.gradient(restored, out, restored.counterfactual.weights(0), 1.)
    for x, y in zip(expected, actual):
        assert (x is None and y is None) or torch.equal(x, y)
    assert receipt['eslong_pairs'] == metrics['eslong_pairs']
    assert_tree_close(trainer.actor_optimizer.state_dict(), restored.actor_optimizer.state_dict(), 0, 0)
