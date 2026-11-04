"""Regression for source-conditioned imitation, not unconditional label frequencies."""
from copy import deepcopy
import json

import pytest
import torch

from tools.coopmwm.exact_grade import _seal
from tools.coopmwm.rerun_tracks_bank import training_batch, batch_receipt
from tools.coopmwm.rerun_tracks_runtime import Runtime, state_sha
from tools.coopmwm.test_t2c_runtime_cpu import prepare, Features, Move


@pytest.mark.parametrize('swap', [False, True])
def test_state_target_association_drives_gradients_and_choices_in_runtime(tmp_path, monkeypatch, swap):
    cfg, reg, backend = prepare(tmp_path, monkeypatch)
    examples = backend.bank_examples('B', 1, cfg=cfg)[:2]
    for e in examples:
        e['weight'] = 1.
        if swap:
            e['plan']['family'] = 'none' if e['plan']['family'] == 'gap_open' else 'gap_open'
            e['effective_sample']['plan_sha256'] = _seal(dict(plan=e['plan']), 'sha256')['sha256']
    backend.bank_examples = lambda *a, **kw: deepcopy(examples)
    rt = Runtime(cfg, reg, backend, smoke=True)
    comp = rt.components['B']
    base_sha = state_sha(rt.models['B'])
    comp['group_credit'].estimate = lambda *a: 0.
    comp['dense_teacher'].label = lambda obs, proposals, **kw: dict(target=proposals[0],
        calibrated_probability=.1, imitation_eligible=False,
        provenance=dict(kind='planner_calibration', receipt_sha256='a'*64))
    batch = training_batch(examples, [], arm='B', encoder=Features())
    layer, trigger = comp['coop_layer'], comp['trigger_head']
    move_loss, effective = layer.bank_losses(batch, trigger)
    move_gradient = torch.autograd.grad(move_loss, trigger.logits)[0]
    assert move_gradient[1, 1].item() == pytest.approx(.5 if swap else -.5)
    assert move_gradient[:, 0].abs().sum() == 0  # identical label frequencies in both batches
    effective_grad = torch.autograd.grad(effective, (layer.correction, layer.effective_slope))
    assert all(g.abs() > 0 for g in effective_grad)
    rt.update('B', 1, episodes=2)
    assert state_sha(rt.models['B']) == base_sha
    assert all(p.grad is None for p in rt.models['B'].parameters())
    assert layer.effective_slope.item() != 0
    assert torch.sign(trigger.logits[1, 1]).item() == (-1 if swap else 1)
    class Midpoint:
        def random(self): return .5
    for e in examples:
        chosen = trigger.choose(e['observation'], Features().encode(e['observation']),
                               [Move('none'), Move('gap_open')], Midpoint())
        assert chosen.family == e['plan']['family']
    learned = layer.last_bank_batch
    assert [r['source_id'] for r in learned] == ['B_0', 'B_1', 'B_0', 'B_1']
    assert [r['weight'] for r in learned] == [1.]*4
    assert learned[0]['features']['values'] != learned[1]['features']['values']
    receipt = json.loads((rt.root/'components/u001_B_bank_learning.json').read_text())
    assert receipt['batch_sha256'] == batch_receipt(learned)['batch_sha256']
    assert receipt['effective_rows'] == receipt['move_rows'] == 4


def test_effective_limits_and_weights_reach_only_trainable_layer(tmp_path, monkeypatch):
    cfg, reg, backend = prepare(tmp_path, monkeypatch)
    rt = Runtime(cfg, reg, backend, smoke=True)
    batch = training_batch(backend.bank_examples('B', 1, cfg=cfg), [], arm='B', encoder=Features())
    layer, trigger = rt.components['B']['coop_layer'], rt.components['B']['trigger_head']
    # A bound target's raw dead-zone value must have no effect on the reviewed
    # Tobit objective, while its recorded binding limit and AWR weight must matter.
    batch = [batch[1]]
    loss = layer.bank_losses(batch, trigger)[1]
    changed = deepcopy(batch)
    changed[0]['effective_sample']['action'] = [80., 80.]
    assert torch.equal(loss, layer.bank_losses(changed, trigger)[1])
    changed[0]['weight'] *= 2
    assert layer.bank_losses(changed, trigger)[1].item() == pytest.approx(2*loss.item())
    original_grad = torch.autograd.grad(loss, layer.effective_slope)[0]
    changed[0]['effective_sample']['binding_limits'] = dict(c1=.1, c2=.1)
    changed_loss = layer.bank_losses(changed, trigger)[1]
    new_grad = torch.autograd.grad(changed_loss, layer.effective_slope)[0]
    assert not torch.isclose(original_grad, new_grad)
    assert all(not p.requires_grad and p.grad is None for p in rt.models['B'].parameters())


def test_missing_effective_learning_receipt_refused(tmp_path, monkeypatch):
    cfg, reg, backend = prepare(tmp_path, monkeypatch)
    rt = Runtime(cfg, reg, backend, smoke=True)
    layer = rt.components['B']['coop_layer']
    update = layer.update
    def incomplete(**kwargs):
        result = update(**kwargs)
        result['bank_receipt']['effective_rows'] = 0
        return result
    layer.update = incomplete
    with pytest.raises(ValueError, match='complete bank/effective supervision'):
        rt.update('B', 1, episodes=2)
    assert not (rt.root/'B_checkpoint_001.pt').exists()


@pytest.mark.parametrize('fault', ['observation', 'association', 'weight', 'provenance',
                                   'limits', 'plan', 'foreign', 'duplicate', 'future'])
def test_invalid_bank_supervision_refused_before_episode_dispatch(tmp_path, monkeypatch, fault):
    cfg, reg, backend = prepare(tmp_path, monkeypatch)
    bank = backend.bank_examples('B', 1, cfg=cfg)
    row = bank[0]
    if fault == 'observation': del row['observation']
    if fault == 'association': row['effective_sample']['context']['observation']['own_sensing']['queue'] = 99
    if fault == 'weight': row['statistics'] = 'pooled'
    if fault == 'provenance': row['provenance'].pop('weight_sha256')
    if fault == 'limits': row['effective_sample']['binding_limits'].pop('c1')
    if fault == 'plan': row['plan']['depth'] += .1
    if fault == 'foreign': row['source_id'] = 'R_0'
    if fault == 'duplicate': bank[1]['id'] = row['id']
    if fault == 'future': row['observation']['v2v_reports'].append(dict(time_s=0))
    backend.bank_examples = lambda *a, **kw: bank
    rt = Runtime(cfg, reg, backend, smoke=True)
    with pytest.raises(ValueError, match='REFUSED'):
        rt.update('B', 1, episodes=2)
    assert not backend.plants and not backend.branch_calls
