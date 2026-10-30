"""CPU synthetic proofs: no simulator, production data or accelerator required."""
from dataclasses import replace
from functools import wraps
import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from tests.unit.test_gated300 import cct2_authority, enable_gain_only
from tests.unit.test_longmem import assert_tree_close
from tests.unit.test_longgrad import make_long_trainer
from wms.control import eslong, gated300
from wms.control.counterfactual import build_variants, row_coefficients
from wms.control.dreamer import _rng_state, _restore_rng


class HeadPolicy(nn.Module):
    supports_noise = True
    accel_min, accel_max = -3., 1.2

    def __init__(self):
        super().__init__()
        self.residual_head = nn.Linear(2, 1)
        nn.init.constant_(self.residual_head.weight, .02)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, obs, signal, deterministic=False, noise=None):
        features = torch.stack((obs.speed, signal), -1)
        latent = self.residual_head(features).squeeze(-1)
        eps = 0. if deterministic or noise is None else .1 * noise.accel
        out = 1.2 * latent.tanh() + eps
        return out, out * 0


def make(tmp_path, monkeypatch, estimator='antithetic', **options):
    t = make_long_trainer(tmp_path, monkeypatch, memory=True, balance=True)
    t.policy = HeadPolicy()
    t.actor_parameters = list(t.policy.parameters()) + list(t.group_encoder.parameters())
    t.actor_optimizer = torch.optim.Adam(t.actor_parameters, lr=t.config.actor_lr)
    # Keep the complete K8/gate algebra; 300 registered seconds map to 10 toy steps.
    enable_gain_only(t, cct2_authority(tmp_path, hg=300), estimator=estimator,
                     es_sigma=.08, es_directions=2, **options)
    t.config = replace(t.config, imagination_checkpoint_segment=2)
    return t


def test_defaults_validation_and_config_no_rng():
    before = torch.get_rng_state()
    assert gated300.LongCreditConfig().estimator == 'pathwise'
    for kwargs in (dict(estimator='bad'), dict(es_directions=0), dict(es_directions=True),
                   dict(es_sigma=-1.), dict(es_sigma=float('nan'))):
        with pytest.raises(ValueError):
            gated300.LongCreditConfig(**kwargs)
    assert torch.equal(before, torch.get_rng_state())


def test_quadratic_unbiased_gaussian_estimator_and_rejected_denominator():
    # E[eps eps^T]=I: antithetic central differences are exact on any quadratic.
    dtype = torch.float64
    theta = torch.tensor([.3, -1.1, .7], dtype=dtype)
    a = torch.tensor([[2., .3, -.2], [.3, 3., .1], [-.2, .1, 1.]], dtype=dtype)
    b = torch.tensor([1., -.4, .9], dtype=dtype)
    directions = torch.randn(100000, 3, dtype=dtype, generator=torch.Generator().manual_seed(31))
    def loss(x):
        return .5 * ((x @ a) * x).sum(-1) + x @ b
    sigma = .13
    diff = loss(theta + sigma * directions) - loss(theta - sigma * directions)
    actual = eslong.estimate(directions, diff, sigma)
    torch.testing.assert_close(actual, a @ theta + b, atol=.035, rtol=.02)
    assert eslong.estimate(torch.ones(2, 1), torch.tensor([2., 0.]), 1.).item() == .5


def test_registered_gain_gate_pair_gate_strict_margin_reversal_and_joint_tail(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch)
    variants = build_variants(t.start_states.batch.state, t.counterfactual)
    coeff = row_coefficients(variants, t.counterfactual.weights(0), variants.n_rows, 1)
    imagined = dict(variants=variants, n_base=1, gated300_horizon_s=300)
    plus = torch.full((8, 1, variants.n_rows), 500.)
    minus = plus.clone()
    delta = torch.zeros_like(plus)
    eligible = torch.nonzero(coeff).flatten()
    delta[..., eligible[0]] = 200.
    delta[..., eligible[1]] = -200.
    delta[..., eligible[2]] = 2 * gated300.SIGMA[300]  # strict boundary
    delta[:2, :, eligible[3]] = -30.
    delta[2:, :, eligible[3]] = 200.  # median large but q10 inconsistent
    plus += delta
    plus[..., eligible[2]] = 2 * gated300.SIGMA[300]
    minus[..., eligible[2]] = 0.
    value, receipt = eslong.paired_loss_difference(t, plus, minus, imagined, coeff, 2.)
    reverse, reversed_receipt = eslong.paired_loss_difference(t, minus, plus, imagined, coeff, 2.)
    assert receipt['accepted'] == 2
    assert receipt['on'] == reversed_receipt['on']
    assert value == -reverse
    expected = -((200 * coeff[eligible[0]] - 200 * coeff[eligible[1]]) / 2)
    torch.testing.assert_close(value, expected)
    # Both evaluations predict harm: original gain-only gate must still exclude them.
    value, _ = eslong.paired_loss_difference(t, -plus, -minus, imagined, coeff, 1.)
    assert value == 0


def test_default_pathwise_is_bitwise_unchanged(tmp_path, monkeypatch):
    baseline = make(tmp_path, monkeypatch, 'pathwise')
    actual = make(tmp_path, monkeypatch, 'pathwise')
    baseline.gated300['config'].pop('estimator')  # historical default path
    torch.manual_seed(47)
    a = baseline.update()
    rng = _rng_state(baseline.device)
    torch.manual_seed(47)
    b = actual.update()
    for key in ('actor_loss', 'returns', 'grad_norm_dense', 'grad_norm_long', 'actor_grad_norm',
                'gated300_active_rows', 'gated300_long_credit'):
        assert a[key] == b[key]
    for x, y in zip(baseline.actor_parameters, actual.actor_parameters):
        assert torch.equal(x, y) and torch.equal(x.grad, y.grad)
    assert_tree_close(baseline.actor_optimizer.state_dict(), actual.actor_optimizer.state_dict(), 0, 0)
    assert_tree_close(rng, _rng_state(actual.device), 0, 0)


def test_full_update_no_long_autograd_dense_identical_crn_rng_and_finite(tmp_path, monkeypatch):
    legacy = make(tmp_path, monkeypatch, 'pathwise')
    actual = make(tmp_path, monkeypatch)
    legacy.config = replace(legacy.config, imagination_replay_dense_prefix=True,
                            imagination_skip_zero_vjp=True)
    calls, differences = [], []
    original = actual.world_models[0].rollout
    @wraps(original)
    def rollout(*args, **kwargs):
        calls.append((torch.is_grad_enabled(), kwargs['n_steps']))
        return original(*args, **kwargs)
    actual.world_models[0].rollout = rollout
    original_pair = eslong.paired_loss_difference
    def pair(*args, **kwargs):
        value, receipt = original_pair(*args, **kwargs)
        differences.append(receipt)
        return value, receipt
    monkeypatch.setattr(eslong, 'paired_loss_difference', pair)
    torch.manual_seed(49)
    a = legacy.update()
    rng = _rng_state(legacy.device)
    torch.manual_seed(49)
    b = actual.update()
    assert b['grad_norm_dense'] == a['grad_norm_dense']
    assert b['eslong_forwards_per_update'] == 40
    assert sum(not grad and steps == 20 for grad, steps in calls) == 40
    assert all(steps <= 6 for grad, steps in calls if grad)  # 2 + next-state boundary
    assert b['grad_long_futures'] == []
    assert b['grad_norm_long_scaled'] <= b['grad_norm_dense']
    assert len(differences) == 2
    assert_tree_close(rng, _rng_state(actual.device), 0, 0)
    json.dumps(b, allow_nan=False)


def capture(t):
    t._training_imagination = True
    batch = t.start_states.batch
    variants = build_variants(batch.state, t.counterfactual)
    out = t.imagine(batch, variants=variants)
    assert not out['gated300_long_difference'].requires_grad
    return out, t._eslong_context


def test_crn_tape_exact_plus_minus_future_and_parameter_restoration(tmp_path, monkeypatch):
    from wms.control.imagination_memory import _RandomTape
    t = make(tmp_path, monkeypatch)
    _, common = capture(t)
    common.update(directions=torch.ones(1, 3), sigma=.08)
    saved = copy.deepcopy(t.policy.state_dict())
    observed = []
    original = _RandomTape.__torch_dispatch__
    def dispatch(self, func, types, args=(), kwargs=None):
        out = original(self, func, types, args, kwargs)
        if func._schema.name.split('::')[-1] in self.names:
            observed.append((str(func), out.detach().clone()))
        return out
    monkeypatch.setattr(_RandomTape, '__torch_dispatch__', dispatch)
    eslong.forward_task(t, common, (0, 1, 3))
    first = observed[:]
    observed.clear()
    eslong.forward_task(t, common, (0, -1, 3))
    assert len(first) == len(observed) > 10
    for (name, x), (other, y) in zip(first, observed):
        assert name == other and torch.equal(x, y)
    assert_tree_close(saved, t.policy.state_dict(), 0, 0)
    with pytest.raises(RuntimeError, match='injected'):
        with eslong.perturb(t, torch.ones(3), .1):
            raise RuntimeError('injected')
    assert_tree_close(saved, t.policy.state_dict(), 0, 0)


def test_auto_sigma_matches_local_exploration_and_reports_scale(tmp_path, monkeypatch):
    from wms.control.coopw import ExplorationConfig
    t = make(tmp_path, monkeypatch)
    t.gated300['config']['es_sigma'] = 0.
    t.exploration = ExplorationConfig(enabled=True, mode='additive_accel', rho=[.1])
    _, common = capture(t)
    sigma, receipt = eslong.choose_sigma(t, common['sensitivity'])
    assert sigma > 0
    assert receipt['eslong_predicted_command_rms'] == pytest.approx(.1)
    assert receipt['eslong_sigma_relative_head_scale'] > 0


def test_checkpoint_resume_next_update_and_historical_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv('C4_ALLOWED_ROOTS', str(tmp_path))
    t = make(tmp_path, monkeypatch)
    t.update()
    path = tmp_path / 'es.pt'
    t.save_checkpoint(path)
    saved = torch.load(path, weights_only=True)
    expected = t.update()
    expected_rng = _rng_state(t.device)
    restored = make(tmp_path, monkeypatch, 'pathwise')
    restored.load_checkpoint(path)
    assert eslong.active(restored)
    result = restored.update()
    for key in ('grad_norm_dense', 'grad_norm_long_es', 'actor_loss', 'eslong_pairs',
                'eslong_loss_differences', 'eslong_sigma'):
        assert result[key] == expected[key]
    assert_tree_close(t.actor_optimizer.state_dict(), restored.actor_optimizer.state_dict(), 0, 0)
    assert_tree_close(expected_rng, _rng_state(t.device), 0, 0)
    for key in ('estimator', 'es_sigma', 'es_directions'):
        saved['gated300']['config'].pop(key)
    torch.save(saved, path)
    restored.load_checkpoint(path)
    assert not eslong.active(restored)


def test_reference_hook_is_transported_to_worker_snapshot(tmp_path, monkeypatch):
    from wms.control.parallel_futures import _snapshot
    t = make(tmp_path, monkeypatch)
    def reference(owner, imagined, totals, base):
        return torch.zeros_like(totals) - 123.
    t._long_credit_reference = reference
    out, _ = capture(t)
    worker = _snapshot(t)
    assert worker._long_credit_reference is reference
    assert worker.gated300 == t.gated300
    assert worker.world_model_ids == t.world_model_ids
    assert worker.horizon_curriculum is t.horizon_curriculum
    assert out['gated300_long_difference'].shape[0] == 8


def test_dense_prefix_enforced_with_flags_off_and_large_segment_refused(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch)
    t.config = replace(t.config, imagination_microbatch=0, imagination_offload_futures=False,
        imagination_replay_dense_prefix=False, imagination_skip_zero_vjp=False)
    result = t.update()
    assert all(r['replay_horizon_s'] <= 3 for r in result['replay_horizon_s'])
    t.config = replace(t.config, imagination_checkpoint_segment=20)
    with pytest.raises(ValueError, match='boundary must be shorter'):
        t.update()


def test_zero_dense_unknown_parameter_support_never_replays_long(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch)
    unused = nn.Parameter(torch.ones(()))
    t.actor_parameters.append(unused)
    t._longskip_records = []
    out, _ = capture(t)
    eslong.backward(t, out['returns'].sum() * 0, (None,) * len(t.actor_parameters))
    assert unused.grad is None
    assert len(t._longskip_records) == 8
    assert all(r['replay_horizon_s'] <= 3 for r in t._longskip_records)


def test_pooled_base_and_all_es_directions_match_serial_update(tmp_path, monkeypatch):
    from wms.control import parallel_futures as pf
    serial = make(tmp_path, monkeypatch)
    parallel = make(tmp_path, monkeypatch)
    torch.manual_seed(19)
    expected = serial.update()
    rng = _rng_state(serial.device)
    tasks_seen = []
    class CpuPool:
        def __init__(self, trainer, common):
            self.trainer, self.common = pf._snapshot(trainer), common
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def map(self, tasks):
            tasks = list(tasks)
            tasks_seen.append((self.common['kind'], bool(self.common.get('eslong')), len(tasks)))
            for task in tasks:
                if self.common.get('eslong'):
                    assert all(set(item) == {'tape_end'} for item in self.common['cache']['items'])
                    yield eslong.forward_task(self.trainer, self.common, task)
                elif self.common['kind'] == 'forward':
                    yield pf._forward(self.trainer, self.common, task)
                else:
                    yield pf._vjp(self.trainer, self.common, task)
    monkeypatch.setattr(pf, 'FuturePool', CpuPool)
    monkeypatch.setattr(pf, 'worker_count', lambda *args: 2)
    monkeypatch.setattr(pf, 'parallel_forward', lambda: True)
    torch.manual_seed(19)
    result = parallel.update()
    assert ('forward', False, 8) in tasks_seen
    assert ('forward', True, 32) in tasks_seen
    assert result['grad_norm_dense'] == expected['grad_norm_dense']
    assert result['grad_norm_long_es'] == expected['grad_norm_long_es']
    assert result['eslong_pairs'] == expected['eslong_pairs']
    assert_tree_close(serial.actor_optimizer.state_dict(), parallel.actor_optimizer.state_dict(), 0, 0)
    assert_tree_close(rng, _rng_state(parallel.device), 0, 0)


def test_real_residual_wm_entrants_crn_and_dense_trunk_gradients(tmp_path, monkeypatch):
    from functools import partial
    import numpy as np
    from tests.unit.test_longmem import Starts, Critic, queue_reward
    from wms.control.bridge import make_cav_controller
    from wms.control.coopw import ExplorationConfig
    from wms.control.counterfactual import CounterfactualConfig
    from wms.control.dreamer import DreamerConfig, DreamerTrainer
    from wms.control.obs import build_observations
    from wms.control.policy import CavPolicy, RsuGroupEncoder
    from wms.model.entries import EntranceHistory
    from wms.model.spec import ModelSpec
    from wms.model.wms import WmSModel
    from wms.control import dreamer
    entry_draws = []
    original_entry = dreamer._entry_process
    def entry(*args, **kwargs):
        process = original_entry(*args, **kwargs)
        if not torch.is_grad_enabled():
            draws = dict(seeds=process._pair_seeds, renewal=[], candidates=[])
            entry_draws.append(draws)
            original_headway, original_lane = process._headway, process._try_lane
            def headway(b, e):
                value = original_headway(b, e)
                draws['renewal'].append((b, e, value))
                return value
            def lane(work, b, e, ln, cav, speed_ratio, gap, vehicle_class):
                draws['candidates'].append((b, e, ln, cav, speed_ratio, gap, vehicle_class))
                return original_lane(work, b, e, ln, cav, speed_ratio, gap, vehicle_class)
            process._headway, process._try_lane = headway, lane
        return process
    monkeypatch.setattr(dreamer, '_entry_process', entry)
    monkeypatch.setattr(gated300, 'STEP_S', 30.)
    torch.manual_seed(701)
    starts = Starts(1, 4)
    starts.batch.state['is_cav'][:, -1] = False
    starts.batch.state['s_m'][:, -1] = 10.
    starts.batch.state['lane'][:, 1] = 1
    hist = EntranceHistory.defaults()
    hist.rate[:] = .2
    starts.batch.entry_feat = torch.tensor(np.stack([hist.to_features()])).float()
    cfg = DreamerConfig(batch_size=1, horizon=2, imagine_entries=True, entry_pad_max=1,
        source_wait_grad=True, entrant_control=True, deterministic_crn=True,
        native_residual_train_group=True, native_residual_train_trunk=True,
        future_samples=8, pessimistic_credit=True, future_driver_scale=.05,
        future_accel_noise=True, imagination_microbatch=1, imagination_checkpoint_segment=2,
        imagination_offload_futures=True, long_grad_balance=True, wm_safety_report_only=True)
    spec = ModelSpec(history_hidden=8, k_hidden=8, rollout_mode='hybrid',
        grad_decay_per_step=.9, lane_model='hazard_v5', merge_decision='first_passage',
        lane_decision='sample', merge_passage_law='hybrid')
    policy = CavPolicy(hidden_dim=8, n_heads=2, lane_head=False, residual_on_native=True)
    policy.residual_additive_exploration = True
    with torch.no_grad():
        policy.residual_head.weight.normal_(std=.01)
    t = DreamerTrainer([WmSModel(spec)], policy, RsuGroupEncoder(hidden_dim=8, n_heads=2),
        Critic(), starts, observation_builder=build_observations, reward_fn=queue_reward,
        safety=lambda state, command, **kw: command, config=cfg,
        exploration=ExplorationConfig(enabled=True, mode='additive_accel', rho=[.1]),
        controller_factory=partial(make_cav_controller, native_lane_change=True,
            native_ramp_merge='wm_law', symmetric_exec_cap=True),
        counterfactual=CounterfactualConfig(enabled=True, max_starts=1, n_individual=0,
            n_group=1, n_flip=0, group_min_size=3, group_max_size=3, w_team=[0., 0.]))
    t.long_credit_requested = True
    enable_gain_only(t, cct2_authority(tmp_path, hg=300, entrant_control=True),
                     estimator='antithetic', es_directions=1)
    result = t.update()
    assert result['entries'] and result['eslong_dimension'] == 25
    assert result['eslong_predicted_command_rms'] == pytest.approx(.1)
    assert result['eslong_forwards_per_update'] == 24
    assert result['grad_long_futures'] == []
    assert len(entry_draws) == 24
    assert any(draws['candidates'] for draws in entry_draws)
    for k in range(8):
        assert entry_draws[k] == entry_draws[8 + k] == entry_draws[16 + k]
    assert all(r['replay_horizon_s'] <= 3 for r in result['replay_horizon_s'])
    for module in (policy.residual_head, policy.own_encoder, policy.report_encoder,
                   policy.attention, t.group_encoder):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    json.dumps(result, allow_nan=False)


def test_actual_forward_process_pool_matches_serial_and_uses_all_tasks(tmp_path, monkeypatch):
    from wms.control.parallel_futures import FuturePool
    monkeypatch.setenv('C4_ALLOWED_ROOTS', str(tmp_path))
    t = make(tmp_path, monkeypatch)
    t.output_dir = tmp_path / 'pool'
    _, context = capture(t)
    common = {**context, 'kind': 'forward', 'eslong': True,
              'directions': torch.ones(1, 3), 'sigma': .08}
    tasks = [(0, sign, future) for sign in (1, -1) for future in (0, 7)]
    before = _rng_state(t.device)
    expected = [eslong.forward_task(t, common, task) for task in tasks]
    monkeypatch.setenv('WMS_SPEED_FORWARD_WORKERS', '2')
    with FuturePool(t, common) as pool:
        actual = list(pool.map(tasks))
    _restore_rng(before, t.device)
    assert_tree_close(expected, actual, 0, 0)


def test_preset_extends_requested_parent_and_is_hash_bound():
    from tools.candA.authority import MANIFEST_FILES, effective_config
    from wms.control.train_cli import _read_config
    root = Path(__file__).resolve().parents[2]
    parent = _read_config(root / 'configs/c4_policy14_candA_longgrad_skip.toml')
    actual = _read_config(root / 'configs/c4_policy14_candA_es.toml')
    parent['long_credit'].update(estimator='antithetic', es_directions=2, es_sigma=0.)
    assert actual == parent
    assert actual == effective_config(root, 'configs/c4_policy14_candA_es.toml')
    assert 'src/wms/control/eslong.py' in MANIFEST_FILES
