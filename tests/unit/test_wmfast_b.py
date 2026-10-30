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
