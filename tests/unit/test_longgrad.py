"""CPU proofs for term balancing, unchanged OFF update, replay and worker transport."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from wms.control.dreamer import DreamerConfig
from wms.control.longgrad import actor_backward, combine, norm, StateTrace


def owner(balance=False):
    policy = nn.Linear(2, 1, bias=False)
    group = nn.Linear(1, 1, bias=False)
    params = list(policy.parameters()) + list(group.parameters())
    return SimpleNamespace(policy=policy, group_encoder=group, actor_parameters=params,
        actor_optimizer=torch.optim.Adam(params), config=DreamerConfig(long_grad_balance=balance))


def test_flag_defaults_off_and_rejects_non_boolean():
    assert DreamerConfig().long_grad_balance is False
    assert DreamerConfig().long_grad_reset_actor_adam is False
    for bad in (0, 1, 'true', None):
        with pytest.raises(ValueError, match='long_grad_balance'):
            DreamerConfig(long_grad_balance=bad)
        with pytest.raises(ValueError, match='long_grad_reset_actor_adam'):
            DreamerConfig(long_grad_reset_actor_adam=bad)


def test_opt_in_preset_only_changes_gradient_controls_and_disables_zero_restart():
    from pathlib import Path
    from wms.control.train_cli import _read_config
    root = Path(__file__).resolve().parents[2]
    base = _read_config(root / 'configs/c4_policy14_candA_long300_serial_coloc.toml')
    flagged = _read_config(root / 'configs/c4_policy14_candA_longgrad.toml')
    base['trainer'].update(long_grad_balance=True, long_grad_reset_actor_adam=True)
    base['policy14']['zero_residual_output'] = False
    assert base == flagged


@pytest.mark.parametrize('balance', [False, True])
def test_known_orthogonal_terms_unused_parameters_and_exact_ratio(balance):
    t = owner(balance)
    w = t.policy.weight.flatten()
    dense, long = 3 * w[0], 4e8 * w[1]
    before = torch.get_rng_state()
    metrics = actor_backward(t, dense, long, dense + long)
    assert metrics['grad_norm_dense'] == 3
    assert metrics['grad_norm_long'] == 4e8
    assert metrics['grad_cosine'] == 0
    assert metrics['grad_long_scale'] == (3 / 4e8 if balance else 1)
    torch.testing.assert_close(t.policy.weight.grad, torch.tensor([[3., 3. if balance else 4e8]]))
    assert t.group_encoder.weight.grad is None
    assert torch.equal(before, torch.get_rng_state())
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize('dense,long,expected', [
    ([3., 0.], [0., 2.], [3., 2.]),
    ([0., 0.], [1., 2.], [0., 0.]),
    ([3., 0.], [0., 0.], [3., 0.]),
    ([1., 2.], [-10., -20.], [0., 0.]),
])
def test_no_amplification_zero_norms_and_opposite_terms(dense, long, expected):
    out, metrics = combine([torch.tensor(dense)], [torch.tensor(long)], balance=True)
    torch.testing.assert_close(out[0], torch.tensor(expected))
    assert metrics['grad_norm_long_scaled'] <= metrics['grad_norm_dense'] + 1e-14


def test_float64_norm_avoids_float32_reduction_overflow():
    out, metrics = combine([torch.tensor([3., 4.])], [torch.tensor([1e30, 0.])], balance=True)
    assert metrics['grad_norm_dense'] == 5
    assert norm([out[0] - torch.tensor([3., 4.])]) == pytest.approx(5)


@pytest.mark.parametrize('which', ['dense', 'long'])
@pytest.mark.parametrize('value', [float('inf'), float('nan')])
def test_nonfinite_terms_fail_before_optimizer_step(which, value):
    t = owner(True)
    w = t.policy.weight.flatten()
    dense, long = w[0], w[1]
    if which == 'dense':
        dense = dense * value
    else:
        long = long * value
    with pytest.raises(ValueError, match='nonfinite'):
        actor_backward(t, dense, long, dense + long)
    assert not t.actor_optimizer.state


def test_off_is_bitwise_legacy_combined_backward_even_under_cancellation():
    t = owner(False)
    w = t.policy.weight
    dense = w.sin().sum()
    long = (w * 1e8).sum() - (w * 1e8).sum() + w.square().sum()
    total = dense + long
    expected = torch.autograd.grad(total, t.actor_parameters, retain_graph=True, allow_unused=True)
    actor_backward(t, dense, long, total)
    for p, grad in zip(t.actor_parameters, expected):
        assert (p.grad is None and grad is None) or torch.equal(p.grad, grad)


def test_passive_state_trace_reports_amplification_without_changing_gradient():
    def run(traced):
        trace = StateTrace()
        x = torch.ones(2, 3, requires_grad=True)
        current = x
        for step in range(10):
            if traced:
                trace.watch({'v_mps': current}, step)
            current = 2 * current
        grad, = torch.autograd.grad(current.sum(), x)
        return grad, trace.summary()
    expected, _ = run(False)
    actual, metrics = run(True)
    assert torch.equal(actual, expected)
    assert metrics['peak_step'] == 0
    assert metrics['max_norm'] == pytest.approx(1024 * 6 ** .5)
    assert metrics['row_max_norms'] == pytest.approx([1024 * 3 ** .5] * 2)


@pytest.mark.parametrize('gap', [-1e-8, 0., 1e-8, .001, 1.])
def test_mode2_actual_reward_ttc_tiny_gaps_and_unused_cost_branch_are_exact(gap):
    from wms.control.dreamer import _ImmediateRecompute
    from wms.control.reward import team_reward
    from wms.control.contract import RewardWeights
    from tests.unit.test_longmem import scene
    state = scene(1, 3)
    state['s_m'] = torch.tensor([[0., 5. + gap, 40.]], requires_grad=True)
    state['v_mps'] = torch.tensor([[10., 9.99999, 10.]], requires_grad=True)
    params = (state['s_m'], state['v_mps'])
    def run(mode):
        call = _ImmediateRecompute(team_reward, mode)
        out = call(state, state, state['a_mps2'], RewardWeights())
        # TTC is a report-only output: its zero cotangent must not inject NaNs.
        grads = torch.autograd.grad(out['reward'].sum() + 0 * out['cost_ttc'].sum(),
                                    params, allow_unused=True)
        return out, grads
    baseline, actual = run(False), run(True)
    for key in baseline[0]:
        assert torch.equal(baseline[0][key], actual[0][key])
    for a, b in zip(baseline[1], actual[1]):
        assert a is None and b is None or (torch.isfinite(b).all() and torch.equal(a, b))


def make_long_trainer(tmp_path, monkeypatch, *, memory, balance):
    from tests.unit.test_gated300 import trainer, enable_gain_only, cct2_authority
    from wms.control import gated300
    monkeypatch.setattr(gated300, 'STEP_S', 30.)  # short CPU graph, same K/gate/tail algebra
    t = trainer(memory=memory)
    t.config = replace(t.config, horizon=2, long_grad_balance=balance)
    original = t.reward_fn
    def reward(*args):
        out = original(*args)
        return {**out, 'reward': -out['reward']}
    t.reward_fn = reward
    enable_gain_only(t, cct2_authority(tmp_path))
    return t


def test_sole_gain_only_negative_credit_identifies_flip_row_and_two_tail_futures():
    from tests.unit.test_gated300 import trainer
    from wms.control import gated300
    from wms.control.counterfactual import build_variants, row_coefficients
    t = trainer()
    t.gated300 = {'config': {'mode': 'gain_only', 'w_long': 1.}}
    variants = build_variants(t.start_states.batch.state, t.counterfactual)
    coeff = row_coefficients(variants, t.counterfactual.weights(0), variants.n_rows, 1)
    flip = next(iter(variants.groups[0]['flips'].values()))
    samples = torch.zeros(8, 1, variants.n_rows)
    samples[:, 0, flip] = torch.arange(8.) * 10 + 500
    samples.requires_grad_(True)
    term, metrics = gated300.actor_term(t, dict(n_base=1, variants=variants,
        gated300_long_difference=samples, gated300_horizon_s=300), coeff, samples * 0, 1.)
    assert coeff[flip] == pytest.approx(-10.3)
    assert metrics['gated300_rows_on_share'] == pytest.approx(.2)
    assert metrics['gated300_long_credit'] < 0 and float(term.detach()) > 0
    vector = torch.tensor(metrics['gated300_long_output_cotangent'])
    assert torch.count_nonzero(vector) == 2
    assert vector.sum() == pytest.approx(10.3)  # K multiplier exactly cancels mean(K)


@pytest.mark.parametrize('memory', [False, True])
@pytest.mark.parametrize('mode', ['0', '2'])
def test_registered_k8_gain_only_objective_unchanged_and_balance_bound(
        tmp_path, monkeypatch, memory, mode):
    monkeypatch.setenv('WMS_LM300_RECOMPUTE_REWARD', mode)
    legacy = make_long_trainer(tmp_path, monkeypatch, memory=memory, balance=False)
    balanced = make_long_trainer(tmp_path, monkeypatch, memory=memory, balance=True)
    torch.manual_seed(29)
    a = legacy.update()
    rng = torch.get_rng_state()
    torch.manual_seed(29)
    b = balanced.update()
    for key in ('actor_loss', 'returns', 'gated300_long_credit', 'gated300_rows_on_share',
                'gated300_active_rows', 'gated300_long_output_cotangent'):
        assert a[key] == b[key]
    assert b['gated300_rows_on_share'] > 0
    assert b['grad_norm_long'] > b['grad_norm_dense'] > 0
    assert b['grad_long_scale'] < 1
    assert b['grad_norm_long_scaled'] == pytest.approx(b['grad_norm_dense'])
    assert b['actor_grad_norm'] <= 2 * b['grad_norm_dense'] * (1 + 1e-6)
    assert torch.equal(rng, torch.get_rng_state())
    assert len(b['grad_long_futures']) == (8 if memory else 0)
    if memory:
        assert any(f['observed_state_cotangent']['max_norm'] > 0 for f in b['grad_long_futures'])
    json.dumps(b, allow_nan=False)


@pytest.mark.parametrize('forward', [False, True])
def test_worker_snapshot_and_vjps_retain_long_receipt_and_curriculum(tmp_path, monkeypatch, forward):
    from wms.control import parallel_futures as pf
    from wms.control.coopw import HorizonConfig
    serial = make_long_trainer(tmp_path, monkeypatch, memory=True, balance=True)
    pooled = make_long_trainer(tmp_path, monkeypatch, memory=True, balance=True)
    for t in (serial, pooled):
        t.horizon_curriculum = HorizonConfig(enabled=True, horizons_s=[1, 2], curriculum_updates=[0, 1])
    snapshot = pf._snapshot(pooled)
    assert snapshot.gated300 == pooled.gated300
    assert snapshot.world_model_ids == pooled.world_model_ids
    assert snapshot.horizon_curriculum is pooled.horizon_curriculum

    # Actual worker _vjp using the production snapshot/payload, CPU in-process;
    # avoids launching a GPU or creating/deleting worker scratch files.
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

    torch.manual_seed(29)
    a = serial.update()
    monkeypatch.setattr(pf, 'worker_count', lambda *args: 2)
    monkeypatch.setattr(pf, 'parallel_forward', lambda: forward)
    monkeypatch.setattr(pf, 'FuturePool', CpuPool)
    torch.manual_seed(29)
    b = pooled.update()
    for key in ('actor_loss', 'grad_norm_dense', 'grad_norm_long', 'grad_cosine',
                'grad_long_futures', 'actor_grad_norm'):
        assert a[key] == b[key]
    for x, y in zip(serial.actor_parameters, pooled.actor_parameters):
        assert torch.equal(x, y) and torch.equal(x.grad, y.grad)


def test_new_phase_requires_only_balance_key_and_retains_adam(tmp_path, monkeypatch):
    monkeypatch.setenv('C4_ALLOWED_ROOTS', str(tmp_path))
    old = make_long_trainer(tmp_path, monkeypatch, memory=True, balance=False)
    old.update()
    checkpoint = tmp_path / 'policy.pt'
    old.save_checkpoint(checkpoint)
    new = make_long_trainer(tmp_path, monkeypatch, memory=True, balance=True)
    with pytest.raises(ValueError, match='trainer.long_grad_balance'):
        new.load_checkpoint(checkpoint, new_phase=True)
    new.load_checkpoint(checkpoint, new_phase=True, resume_allow=['trainer.long_grad_balance'])
    assert new.config.long_grad_balance
    assert new.resume_config_receipt['differences'] == {
        'trainer.long_grad_balance': {'checkpoint': False, 'requested': True}}
    from tests.unit.test_longmem import assert_tree_close
    assert_tree_close(old.actor_optimizer.state_dict(), new.actor_optimizer.state_dict(), 0, 0)
    new.update()


def test_explicit_new_phase_adam_reset_keeps_weights_and_is_not_repeated_on_resume(tmp_path, monkeypatch):
    monkeypatch.setenv('C4_ALLOWED_ROOTS', str(tmp_path))
    old = make_long_trainer(tmp_path, monkeypatch, memory=True, balance=False)
    old.update()
    checkpoint = tmp_path / 'before.pt'
    old.save_checkpoint(checkpoint)
    new = make_long_trainer(tmp_path, monkeypatch, memory=True, balance=True)
    new.config = replace(new.config, long_grad_reset_actor_adam=True)
    new.load_checkpoint(checkpoint, new_phase=True, resume_allow=[
        'trainer.long_grad_balance', 'trainer.long_grad_reset_actor_adam'])
    assert not new.actor_optimizer.state
    assert new.resume_config_receipt['actor_adam_reset']['cleared_parameters'] > 0
    assert old.update_count == new.update_count
    for a, b in zip(old.actor_parameters, new.actor_parameters):
        assert torch.equal(a, b)
    new.update()
    checkpoint2 = tmp_path / 'after.pt'
    new.save_checkpoint(checkpoint2)
    restored = make_long_trainer(tmp_path, monkeypatch, memory=True, balance=True)
    restored.load_checkpoint(checkpoint2)
    from tests.unit.test_longmem import assert_tree_close
    assert_tree_close(new.actor_optimizer.state_dict(), restored.actor_optimizer.state_dict(), 0, 0)
