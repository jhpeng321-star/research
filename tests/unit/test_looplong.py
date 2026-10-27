"""CPU-only boundary, uncertainty, parity and objective-preservation regressions."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/c4'))
import loop14 as loop  # noqa: E402
import loop14_long as long  # noqa: E402
from wms.control.long_credit import (fit_cqr, world_role, bounded_difference,  # noqa: E402
    sample_location_width, pessimistic_difference, hold_metrics)
from wms.control.coopw import ExplorationConfig, exploration_plan  # noqa: E402
from wms.data.window_cache import WindowCache  # noqa: E402


def rows():
    return [{'world': w, 'horizon_s': 120, 'samples': list(np.arange(8.) + w + delta),
             'real_benefit': float(w + 1. + delta)} for w in range(40) for delta in (-10., 10.)]


def test_flags_default_off():
    assert not long.enabled({})
    assert long.wm_sets({}, {'cycles': [{'cycle': 2}]}) == []
    with pytest.raises(ValueError):
        long.validate({'looplong': {'enabled': 'true'}}, ValueError)


def test_cycle_boundary_wm_flags_and_resolved_config():
    from wms.config.load import load_train_config
    cfg = {'looplong': {'enabled': True, 'wm_loss': True, 'enable_cycle': 3}}
    assert not long.wm_sets(cfg, {'cycles': [{'cycle': 1}]})
    sets = long.wm_sets(cfg, {'cycles': [{'cycle': 2}]})
    resolved = load_train_config(str(ROOT / 'configs/c4_wm_loop14_r16_ft.toml'), sets)
    resolved.validate_for_training()
    assert resolved.loss.longwm_enabled
    assert tuple(resolved.curriculum.stage_steps) == (40,)
    assert tuple(resolved.curriculum.unroll_s) == (300.,)
    assert resolved.optim.freeze_prefixes


@pytest.mark.parametrize('bad_h', [10, 30, 60])
def test_short_nonregression_cannot_be_bypassed(bad_h):
    doc = {'threshold': .8, 'horizons': {str(h): {'sign_agreement': .9, 'verdict': 'PASS'} for h in long.HORIZONS}}
    assert long.horizon_selection(doc, 60) == (True, 60, 300)
    doc['horizons'][str(bad_h)]['sign_agreement'] = .79
    assert long.horizon_selection(doc, 60) == (False, 10, 10)


def test_largest_passing_horizon_and_no_grandfathering():
    doc = {'threshold': .8, 'horizons': {str(h): {'sign_agreement': .9, 'verdict': 'PASS'} for h in long.HORIZONS}}
    doc['horizons']['300']['verdict'] = 'FAIL'
    doc['horizons']['90']['verdict'] = 'FAIL'
    assert long.horizon_selection(doc, 90) == (True, 10, 180)


def test_cqr_world_split_and_measured_scale():
    artifact = fit_cqr(rows(), {'wm_sha256': 'wm'})
    model = artifact['horizons']['120']
    assert model['qualified']
    assert not set(model['fit_worlds']) & set(model['calibration_worlds'])
    assert all(world_role(w) == 'fit' for w in model['fit_worlds'])
    assert model['lower_scale'] < 0
    assert artifact['k'] == 8
    same = rows() + [dict(rows()[0])]
    assert fit_cqr(same, {})['horizons']['120']['fit_worlds'] == model['fit_worlds']


def test_cqr_insufficient_worlds_never_fakes_confidence():
    selected = [next(r for r in rows() if world_role(r['world']) == role)
                for role in ('fit', 'calibration')]
    model = fit_cqr(selected, {})['horizons']['120']
    assert not model['qualified'] and model['correction_scale'] is None
    assert not bounded_difference(torch.randn(8, 2, 1), model, torch.ones(1)).any()


@pytest.mark.parametrize('k', [1, 2, 7, 9])
def test_fake_or_wrong_k_refused(k):
    with pytest.raises(ValueError, match='K=8'):
        sample_location_width(np.zeros((3, k)))


def test_positive_mean_uncertain_benefit_has_no_positive_long_credit():
    model = {'qualified': True, 'lower_scale': -.5, 'upper_scale': .5, 'correction_scale': 1.}
    x = torch.arange(8.).reshape(8, 1, 1).requires_grad_()
    bound = bounded_difference(x, model, torch.ones(1))
    assert float(x.detach().mean()) > 0 and float(bound.detach().mean()) < 0
    bound.mean().backward()
    assert torch.isfinite(x.grad).all()
    # Identical paired worlds retain native identity, despite numerical width floor.
    assert not bounded_difference(torch.zeros_like(x), model, torch.ones(1)).any()


def test_negative_cooperation_coefficient_uses_upper_endpoint():
    model = {'qualified': True, 'lower_scale': -1., 'upper_scale': 1., 'correction_scale': .2}
    x = torch.arange(8.).reshape(8, 1, 1)
    lo = bounded_difference(x, model, torch.ones(1))
    hi = bounded_difference(x, model, -torch.ones(1))
    assert float(hi.mean()) > float(lo.mean())


def test_long_credit_leaves_short_horizon_byte_identical():
    trainer = SimpleNamespace(long_credit={})
    difference = torch.randn(8, 60, 2)
    actual = pessimistic_difference(trainer, {'imagination_horizon_s': 60}, difference, None, None, None)
    assert actual is difference


def test_optimistic_critic_tail_cannot_bypass_long_cqr():
    trainer = SimpleNamespace(long_credit={'horizons': {'120': {'qualified': False, 'correction_scale': None}}},
        config=SimpleNamespace(gamma=1., lambda_=1.))
    rewards = torch.zeros(8, 120, 2)
    rewards[..., 0] = 1.
    imagined = {'imagination_horizon_s': 120, 'n_base': 1, 'rewards': rewards,
                'values': torch.ones(8, 121, 2) * 1e9}
    result = pessimistic_difference(trainer, imagined, imagined['values'][:, :-1],
                                     torch.tensor([1, 1]), torch.tensor([True, False]), torch.ones(2))
    assert result[:, 0, 0].mean() == 60
    assert result[:, 60:, 0].count_nonzero() == 0


def test_cached_records_are_bounded_and_never_mutated():
    cache = WindowCache(limit=64)
    calls = []
    def load():
        calls.append(1)
        return {'x': np.ones(4)}
    first = cache.get(0, load)
    first['x'][0] = 7
    assert cache.get(0, load)['x'][0] == 1
    for i in range(10):
        cache.get(i, load)
    assert cache.size <= cache.limit and len(calls) == 10


def test_full_graph_speed_removes_probe_without_changing_objective():
    from tools.looplong.dry_run import actual_ft_speed_probe
    result = actual_ft_speed_probe()
    assert result['False']['rollout_calls'] == 2 and result['True']['rollout_calls'] == 1
    assert result['objective_checkpoint_rng_byte_equal']


def test_exploration_long_persistence_active_and_crn_plan():
    cfg = ExplorationConfig(enabled=True, rho=[.5], long_persistence=True)
    assert cfg.persistence_at(0) == 30 and cfg.persistence_at(150) == 60
    state = {'s_m': torch.tensor([[-30., -20.]]), 'present': torch.ones(1, 2, dtype=torch.bool),
             'lane': torch.zeros(1, 2, dtype=torch.long),
             'is_cav': torch.ones(1, 2, dtype=torch.bool)}
    a = exploration_plan(state, 100., 60, 60., torch.Generator().manual_seed(7))
    b = exploration_plan(state, 100., 60, 60., torch.Generator().manual_seed(7))
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_metering_hold_counts_have_explicit_denominator():
    image = {'actions': -torch.ones(8, 60, 1, 2), 'action_mask': torch.ones(8, 60, 1, 2, dtype=torch.bool),
             'states': [[{'s_m': -torch.ones(1, 2)} for _ in range(61)] for _ in range(8)], 'n_base': 1}
    result = hold_metrics(image)
    assert result['metering_like_holds_ge30s'] == result['metering_like_sampled_starts'] == 8
    assert result['metering_like_max_hold_s'] == 60


def boundary_files(tmp_path):
    wm = tmp_path / 'wm.pt'
    wm.write_bytes(b'new accepted model')
    cfg = tmp_path / 'wm.toml'
    cfg.write_text('name="wm"')
    from wms.control.looplong import digest
    gate = tmp_path / 'gate.json'
    gate.write_text(json.dumps({'wm_sha256': digest(wm), 'wm_config_sha256': digest(cfg),
        'threshold_source': {'sha256': 'threshold'}, 'threshold': .8,
        'horizons': {str(h): {'verdict': 'PASS', 'sign_agreement': .9} for h in (10, 30, 60)}}))
    accepted = tmp_path / 'ACCEPT.json'
    accepted.write_text(json.dumps({'decision': 'ACCEPT', 'inputs': {
        'candidate_sha256': digest(wm), 'config_sha256': digest(cfg)}}))
    doc = {'schema': 'looplong_boundary/1', 'wm': {'checkpoint': str(wm), 'config': str(cfg), 'sha256': digest(wm)},
        'accepted_after_cycle': 2,
        'gate_file': str(gate), 'gate_sha256': digest(gate), 'short_nonregression': True,
        'threshold_sha256': 'threshold', 'acceptance': str(accepted), 'acceptance_sha256': digest(accepted),
        'horizon_s': 60, 'previous_horizon_s': 60, 'flags': {}}
    path = tmp_path / 'swap.json'
    path.write_text(json.dumps(doc))
    return path, doc


def test_swap_checks_acceptance_candidate_and_hash(tmp_path):
    from wms.control.looplong import read_boundary
    path, doc = boundary_files(tmp_path)
    assert read_boundary(path, swap=True)['wm'] == doc['wm']
    Path(doc['wm']['checkpoint']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='binding changed'):
        read_boundary(path, swap=True)


def test_hot_swap_preserves_actor_optimizer_and_records_update(tmp_path):
    from wms.control.looplong import maybe_swap
    path, doc = boundary_files(tmp_path)
    trainer = SimpleNamespace(device=torch.device('cpu'), config=SimpleNamespace(imagination_grad_decay=1.),
        update_count=17, world_models=[torch.nn.Linear(1, 1)], actor_optimizer=object(),
        policy=torch.nn.Linear(1, 1), _curriculum_position=lambda advance=False: 17)
    initial = copy.deepcopy(trainer.policy.state_dict())
    optimizer = trainer.actor_optimizer
    def load(*args):
        return [torch.nn.Linear(1, 1)], [{'physics': {}}]
    result = maybe_swap(trainer, path, {'wm': {'sha256': 'old'}, 'accepted_after_cycle': 1}, {}, tmp_path, load)
    assert result[2]['swap_update'] == 17 and trainer.update_count == 17
    assert trainer.actor_optimizer is optimizer
    assert all(torch.equal(v, initial[k]) for k, v in trainer.policy.state_dict().items())
    assert (tmp_path / 'wm_swaps.jsonl').is_file()
    assert maybe_swap(trainer, path, result[0], {}, tmp_path, load) is None
    del trainer._looplong_swap_request_sha256
    assert maybe_swap(trainer, path, {'wm': {'sha256': 'newer'}, 'accepted_after_cycle': 3},
                      {}, tmp_path, load) is None


def test_tiny_full_pathwise_ft_and_frozen_core():
    from tools.looplong.dry_run import tiny_ft
    rows = tiny_ft(updates=1)
    assert len(rows) == 1 and np.isfinite(rows[0]['loss'])
    receipt = rows[0]['longwm']
    assert receipt and any(n.startswith('paired_') for n in receipt['weights'])
    assert sum(receipt['weights'][n] * receipt['gradient_norms'][n] for n in receipt['weights']) == pytest.approx(receipt['base_gradient_norm'])


def test_horizon300_features_keep_the_long_path_gradient():
    from wms.train.longwm import horizon_features
    x = torch.ones(2, 600, 3, requires_grad=True)
    view = {n: x for n in ('density', 'speed', 'queue', 'flow', 'delay')}
    view['valid'] = torch.ones(2, 600, dtype=torch.bool)
    result = horizon_features(view, [60, 120, 300], .5)
    result['delay'][:, -1].sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


@pytest.mark.parametrize('passes', [False, True])
def test_acceptance_and_short_gate_are_and_composed(tmp_path, monkeypatch, passes):
    _, doc = boundary_files(tmp_path)
    state = {'accepted_wm': {'sha256': 'old', 'checkpoint': 'old.pt'},
             'consecutive_rejects': 0, 'cycles': [{'cycle': 2}]}
    cfg = {'loop_root': str(tmp_path), 'wm': {'config': doc['wm']['config']},
           'looplong': {'enabled': True, 'auto_horizon': True}}
    boundary = {'short_nonregression': passes, 'wm': doc['wm'], 'horizon_s': 60,
                'gate_file': doc['gate_file'], 'gate_sha256': doc['gate_sha256']}
    monkeypatch.setattr(long, 'check_candidate', lambda *args: boundary)
    published = []
    monkeypatch.setattr(long, 'publish', lambda *args: published.append(args))
    monkeypatch.setattr(loop, 'long_module', lambda: long)
    ft = {'candidate': doc['wm']['checkpoint'], 'candidate_sha256': doc['wm']['sha256']}
    receipt = loop.apply_verdict(cfg, state, ft, {'decision': 'ACCEPT'}, 0, tmp_path, 2)
    assert receipt['decision'] == ('ACCEPT' if passes else 'REJECT')
    assert len(published) == int(passes)
    assert state['accepted_wm']['sha256'] == (doc['wm']['sha256'] if passes else 'old')


def test_failed_existing_acceptance_never_calls_new_gate(tmp_path, monkeypatch):
    cfg = {'looplong': {'enabled': True, 'auto_horizon': True}}
    monkeypatch.setattr(long, 'check_candidate', lambda *args: pytest.fail('must not gate rejected WM'))
    monkeypatch.setattr(loop, 'long_module', lambda: long)
    state = {'accepted_wm': {'sha256': 'old'}, 'consecutive_rejects': 0}
    assert loop.apply_verdict(cfg, state, {}, {'decision': 'REJECT'}, 3, tmp_path, 2)['decision'] == 'REJECT'


def test_calibration_labels_do_not_fit_quantile_scales():
    data = rows()
    before = fit_cqr(data, {})['horizons']['120']
    for r in data:
        if world_role(r['world']) == 'calibration':
            r['real_benefit'] += 10000
    after = fit_cqr(data, {})['horizons']['120']
    assert before['lower_scale'] == after['lower_scale']
    assert before['upper_scale'] == after['upper_scale']
    assert after['correction_scale'] > before['correction_scale']


def test_longwm_pair_marker_survives_collation():
    from tools.looplong.dry_run import batch
    from wms.data.collate import collate
    b = batch()
    records = [{n: b[n][i].numpy() for n in ('states', 'present', 'slot_index')}
                for i in range(2)]
    assert 'mix_source' not in collate(records)
    for r in records:
        r.update(longwm_pair=True, mix_source='loop14_pairs')
    assert collate(records)['mix_source'] == ['loop14_pairs', 'loop14_pairs']


def test_acceptance_window_major_is_identical_and_decodes_half(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location('accept_looplong_test', ROOT / 'tools/loop14_checks/accept.py')
    acc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(acc)
    calls = []
    class Dataset:
        def __getitem__(self, i):
            calls.append(i)
            return i
    monkeypatch.setattr(acc, 'free_rollout', lambda m, s, r, n, seed: r)
    monkeypatch.setattr(acc, 'lane1_hard_brake', lambda r: (r + 10, r % 2))
    monkeypatch.setattr(acc, 'safety_metrics', lambda r: {
        'blowup': False, 'overlap_pair_steps': r, 'ttc_lt_1_5_rows': r, 'ttc_eligible_rows': r + 10})
    arms = {name: (None, None, None) for name in ('candidate', 'reference')}
    monkeypatch.delenv('WMS_LOOPLONG_SPEED', raising=False)
    off = acc.rollouts(arms, [Dataset()], [(0, i) for i in range(3)])
    assert len(calls) == 6
    calls.clear()
    monkeypatch.setenv('WMS_LOOPLONG_SPEED', '1')
    on = acc.rollouts(arms, [Dataset()], [(0, i) for i in range(3)])
    assert on == off and len(calls) == 3


def test_current_registered_exploration_is_active_at_c3():
    from wms.control.train_cli import _read_config
    raw = _read_config(str(ROOT / 'configs/c4_policy14_c2h60.toml'))
    exp = ExplorationConfig(**raw['exploration'])
    assert exp.enabled and exp.long_persistence and exp.rho_at(383) > 0
    assert 30 <= exp.persistence_at(383) <= 60


def test_real_trainer_updates_with_long_bounds_and_persistent_exploration():
    from tools.looplong.dry_run import policy_probe
    rows = policy_probe()
    assert len(rows) == 3 and all(r['imagination_horizon_s'] == 90 for r in rows)
    assert all(r['exploration_correlated_active'] for r in rows)


def test_enable_commands_are_default_off_and_twins_have_identical_flags(tmp_path, monkeypatch):
    from tools.looplong import prepare
    base = {'fastloop': {'overlap_policy': True}, 'resumeprep': {'enabled': True}}
    monkeypatch.setattr(prepare.loop14, 'load_config', lambda _: copy.deepcopy(base))
    threshold = tmp_path / 'registered.json'
    threshold.write_text(json.dumps({'schema': 'coopw_horizon_gate/1', 'threshold': .868421053}))
    disabled = prepare.prepare('base.json', tmp_path / 'disabled.json', 'pairs300', threshold)
    assert not disabled['looplong']['enabled']
    main = prepare.prepare('base.json', tmp_path / 'main.json', 'pairs300', threshold, enable=True)
    twin = prepare.prepare('base.json', tmp_path / 'twin.json', 'pairs300', threshold, enable=True, coop_credit=False)
    expected = copy.deepcopy(main)
    expected['looplong']['coop_credit'] = False
    assert expected == twin


def test_k8_is_only_allocated_beyond_known_small_error_horizon():
    from wms.control.dreamer import DreamerTrainer, DreamerConfig
    from wms.control.coopw import UncertaintyConfig
    batch = SimpleNamespace(state={'present': torch.ones(1, 2, dtype=torch.bool),
                                  'is_cav': torch.ones(1, 2, dtype=torch.bool)})
    trainer = SimpleNamespace(config=DreamerConfig(horizon=60), _training_imagination=True,
        horizon_curriculum=None, uncertainty=UncertaintyConfig(enabled=True),
        long_credit_requested=True, _curriculum_position=lambda advance=False: 383)
    assert DreamerTrainer._imagination_config(trainer, batch).future_samples == 2
    trainer.config = DreamerConfig(horizon=120)
    long_cfg = DreamerTrainer._imagination_config(trainer, batch)
    assert long_cfg.future_samples == 8 and long_cfg.uncertainty_weight == 0


def test_real_acceptance_rollout_and_safety_metrics_match_after_speed_fix():
    from tools.looplong.dry_run import acceptance_probe
    result = acceptance_probe()
    assert result['metrics_identical'] and result['window_decodes'] == {'False': 4, 'True': 2}


def test_handoff_word_budget():
    assert len((ROOT / 'LOOPLONG.md').read_text().split()) <= 500


def test_longwm_source_uses_only_policy_native_not_extra_group_tapes(tmp_path, monkeypatch):
    from wms.data import c4_pairs
    doc = {'allow_designed': True, 'episodes': [
        {'pair_key': 'world@0', 'world': 'world', 'split_role': 'train', 'warmup_s': 0.,
         'episode': name, 'dir': name, 'branch': {'kind': name}}
        for name in ('B', 'GROUP', 'POLICY')]}
    class Episode:
        def array(self, key):
            return {'time_s': np.arange(601)*.5, 'control_time_s': np.arange(300),
                'veh_mask': np.ones((601, 3), bool), 'vid': np.tile(np.arange(3), (601, 1)),
                'veh_kin': np.zeros((601, 3, 7), np.float32)}[key]
    monkeypatch.setattr(c4_pairs, 'read_source_manifest', lambda root: doc)
    monkeypatch.setattr(c4_pairs, '_open_episode', lambda path: Episode())
    source = c4_pairs.PairedBranchSource(tmp_path, horizon_steps=600, arm_kinds=('B', 'POLICY'))
    assert [e['branch']['kind'] for e in source.dataset._loader.entries] == ['B', 'POLICY']
    assert source.groups[0]['has_native_ref'] and len(source.groups[0]['rows']) == 2


def test_wm_pairs_bind_this_cycle_policy_and_return_bytes(tmp_path):
    policy, native = tmp_path / 'policy.npz', tmp_path / 'native.npz'
    policy.write_bytes(b'current paired real policy returns')
    native.write_bytes(b'current paired real native returns')
    receipt = tmp_path / 'pairing.json'
    doc = {'schema': 'paircrit_pairs/1', 'split': 'TRAIN', 'policy_checkpoint_sha256': 'current',
        'policy_sha256': loop.sha256(policy), 'native_sha256': loop.sha256(native),
        'pairs': [{'horizon_s': 300, 'same_saved_start': True, 'observable_start_sha256': 'start'}]}
    receipt.write_text(json.dumps(doc))
    conv = {'returns': str(policy), 'native_returns': str(native), 'pairing_receipt': str(receipt),
            'pairing_receipt_sha256': loop.sha256(receipt)}
    state = {'policy': {'sha256': 'current'}}
    assert long.verify_pairs(loop, state, conv) == loop.sha256(receipt)
    state['policy']['sha256'] = 'previous-cycle'
    with pytest.raises(loop.Refused, match='policy/role mismatch'):
        long.verify_pairs(loop, state, conv)
    state['policy']['sha256'] = 'current'
    policy.write_bytes(b'changed')
    with pytest.raises(loop.Refused, match='bytes changed'):
        long.verify_pairs(loop, state, conv)


def test_long_credit_overlap_uses_current_cohort_even_without_paircrit(tmp_path, monkeypatch):
    import loop14_fast as fast
    files = {}
    for name in ('returns', 'native_returns', 'pairing_receipt', 'old_returns'):
        path = tmp_path / name
        path.write_bytes(name.encode())
        files[name] = str(path)
    cfg = {'policy': {'paircrit': False}, 'looplong': {'enabled': True, 'long_credit': True}}
    cycle = loop.new_cycle(2)
    cycle['phases']['convert']['receipt'] = {**files, 'pairs': str(tmp_path / 'pairs')}
    state = {'cycles': [cycle], 'real_returns': files['old_returns'],
             'accepted_wm': {'sha256': 'wm'}, 'policy': {'sha256': 'actor', 'checkpoint': 'actor.pt'}}
    def run_policy(cfg_, snapshot, phase, root):
        assert snapshot['real_returns'] == files['returns']
        assert snapshot['native_real_returns'] == files['native_returns']
        assert snapshot['pairing_receipt'] == files['pairing_receipt']
        return {'synthetic': True}
    monkeypatch.setattr(loop, 'run_policy', run_policy)
    result = fast.overlap(loop, cfg, state, tmp_path)
    assert result['dependencies']['real_returns'] == files['returns']
    assert state['real_returns'] == files['old_returns']  # owned snapshot only


def test_zero_acceleration_metering_hold_is_counted_against_paired_native():
    state = {'s_m': -torch.ones(2, 2), 'v_mps': torch.tensor([[5., 5.], [8., 8.]]),
             'present': torch.ones(2, 2, dtype=torch.bool), 'is_cav': torch.ones(2, 2, dtype=torch.bool)}
    image = {'actions': torch.zeros(2, 60, 2, 2),
             'action_mask': torch.ones(2, 60, 2, 2, dtype=torch.bool), 'n_base': 1,
             'variants': SimpleNamespace(base_row={0: 1}), 'states': [[state]*61 for _ in range(2)]}
    result = hold_metrics(image)
    assert result['metering_like_holds_ge30s'] == 2
    assert result['metering_like_max_hold_s'] == 60


def test_conditional_cqr_preserves_old_development_partition():
    import hashlib
    for world in range(100):
        if int(hashlib.sha256(f'965140:{world}'.encode()).hexdigest(), 16) % 5 == 0:
            assert world_role(world) == 'validation'
    model = fit_cqr(rows(), {})['horizons']['120']
    assert model['condition_edges'] and len(model['conditional_lower']) > 1


def test_missing_fit_worlds_is_unqualified_not_a_fake_finite_bound():
    model = fit_cqr([r for r in rows() if world_role(r['world']) == 'validation'], {})['horizons']['120']
    assert model['correction_scale'] is None and not model['qualified']


def test_calibrated_extension_is_separate_from_trusted_gate(tmp_path, monkeypatch):
    from wms.control.looplong import configure
    from wms.control import long_credit, looplong_real
    from wms.control.coopw import HorizonConfig
    path, doc = boundary_files(tmp_path)
    doc.update(imagination_horizon_s=120, source_policy_checkpoint=str(tmp_path / 'actor.pt'),
               flags={'auto_horizon': True, 'long_credit': True})
    policy, group = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
    torch.save({'policy': policy.state_dict(), 'group_encoder': group.state_dict()}, doc['source_policy_checkpoint'])
    from wms.control.looplong import digest
    doc['source_policy_sha256'] = digest(doc['source_policy_checkpoint'])
    trainer = SimpleNamespace(config=SimpleNamespace(entry_pad_max=32, horizon=10),
        device=torch.device('cpu'), update_count=383, policy=policy, group_encoder=group,
        _curriculum_position=lambda advance=False: 383)
    monkeypatch.setattr(looplong_real.RealPairCredit, 'from_boundary',
                        lambda _: SimpleNamespace(excluded_worlds=set()))
    calls = []
    def fit(*args):
        calls.append(args[-2])
        return {'horizons': {'120': {'qualified': False, 'correction_scale': None}}}
    monkeypatch.setattr(long_credit, 'fit_from_loop', fit)
    doc.update(pairs='pairs', pairing_receipt='proof', policy_returns='policy', native_returns='native')
    configure(trainer, doc, {}, tmp_path)
    assert calls == [[120]] and trainer.long_credit_trusted_s == 60
    assert trainer.long_credit_on_policy and trainer.long_credit_bound_update == 383
    assert isinstance(trainer.horizon_curriculum, HorizonConfig)
    assert trainer.horizon_curriculum.horizons_s == [10, 60, 120]
    assert trainer.horizon_curriculum.allowed_stage == 2
    assert trainer.horizon_curriculum.gate_record['trusted_horizon_s'] == 60


def test_updated_actor_and_unvalidated_cooperation_are_pessimistic_only():
    trainer = SimpleNamespace(long_credit={'horizons': {'120': {
        'qualified': True, 'lower_scale': 0., 'upper_scale': 0., 'correction_scale': 0.}}},
        long_credit_trusted_s=90, long_credit_on_policy=True,
        long_credit_bound_update=10, update_count=10)
    rewards = torch.zeros(8, 120, 3)
    rewards[:, 90:, :2] = 1.
    image = {'imagination_horizon_s': 120, 'rewards': rewards, 'n_base': 1,
             'deterministic_policy': True}
    args = (image, rewards, torch.tensor([2, 2, 2]), torch.tensor([True, True, False]), torch.ones(3))
    result = pessimistic_difference(trainer, *args)
    assert result[..., 0].min() == 30 and result[..., 1:].count_nonzero() == 0
    image['deterministic_policy'] = False
    assert pessimistic_difference(trainer, *args).count_nonzero() == 0
    image['deterministic_policy'] = True
    trainer.update_count += 1
    assert pessimistic_difference(trainer, *args).count_nonzero() == 0


def test_between_world_levels_and_correct_signs_cannot_qualify_action_ranking():
    from wms.control.long_credit import validate_model
    worlds = np.repeat(np.arange(1, 7), 2)
    # Almost-perfect between-world correlation; all policy/native signs right,
    # but the model reverses EVERY within-world action-effect ordering.
    real = worlds * 100. + np.tile([-1., 1.], 6)
    mean = worlds * 100. + np.tile([1., -1.], 6)
    model = {'lower_scale': -3., 'upper_scale': 3., 'correction_scale': 0.}
    result = validate_model(worlds, mean, np.ones(12), real, model)
    assert result['world_coverage'] == 1 and result['ranking_lower95'] == 0
    assert not result['qualified']
    singleton = validate_model(worlds[::2], mean[::2], np.ones(6), real[::2], model)
    assert singleton['ranking_lower95'] is None and not singleton['qualified']


def test_batched_parity_preserves_weights_and_final_gradient(monkeypatch):
    from wms.train.longwm import gradient_parity
    results = []
    for on in ('0', '1'):
        monkeypatch.setenv('WMS_LOOPLONG_SPEED', on)
        p = torch.tensor([.2, -.7, 1.3], dtype=torch.float64, requires_grad=True)
        unused = torch.ones(2, dtype=torch.float64, requires_grad=True)
        shared = p.sin().exp()
        base = shared.sum().square()
        components = {'a': shared.square().mean(), 'b': (shared[0] - shared[1]).square(),
                      'constant': p.new_tensor(1.), 'zero': (shared * 0.).sum()}
        weights, receipt = gradient_parity(base, components, [p, unused])
        loss = base + sum(weights[k] * v for k, v in components.items())
        loss.backward()
        results.append((weights, receipt['gradient_norms'], p.grad))
        assert weights['constant'] == weights['zero'] == 0 and unused.grad is None
    assert results[0][0] == pytest.approx(results[1][0], rel=1e-12)
    assert results[0][1] == pytest.approx(results[1][1], rel=1e-12)
    torch.testing.assert_close(results[0][2], results[1][2], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('stochastic', [False, True])
def test_courtesy_replay_cache_preserves_stochastic_draws(monkeypatch, stochastic):
    from wms.train.losses import follower_courtesy as courtesy
    monkeypatch.setenv('WMS_LOOPLONG_SPEED', '1')
    calls = []
    def replay(*args, **kwargs):
        calls.append(1)
        value = torch.rand(1) if stochastic else torch.ones(1)
        return None, {key: value for key in courtesy.REPLAY_KEYS}
    monkeypatch.setattr(courtesy, 'one_step_replay', replay)
    targets = SimpleNamespace(states=torch.zeros(1), prepared={})
    first = courtesy.replay_outputs(None, targets)
    second = courtesy.replay_outputs(None, targets)
    assert len(calls) == (2 if stochastic else 1)
    assert (first is second) is not stochastic


def test_validation_outcomes_never_refit_conditional_quantiles():
    baseline = fit_cqr(rows(), {})['horizons']['120']
    altered = [dict(row, real_benefit=-1e9) if world_role(row['world']) == 'validation'
               else row for row in rows()]
    model = fit_cqr(altered, {})['horizons']['120']
    for key in ('condition_edges', 'conditional_lower', 'conditional_upper', 'correction_scale'):
        assert baseline[key] == model[key]
    assert baseline['qualified'] and not model['qualified']


@pytest.mark.parametrize('benefit', [-100., 100.])
def test_direct_real_300s_credit_moves_toward_beneficial_observed_contrast(benefit):
    from wms.control.looplong_real import secant_loss
    action = torch.tensor([[.3, .1]], requires_grad=True)
    native = torch.zeros_like(action)
    loss, count = secant_loss(action, native, torch.tensor([[-.5, 3.]]),
        torch.tensor([benefit]), torch.tensor([[True, False]]), 10.)
    loss.backward()
    assert count == 1 and action.grad[0, 0] == pytest.approx(benefit / 5.)
    assert action.grad[0, 1] == 0  # unobserved/uncontrolled rows get no credit
    action.grad.zero_()
    loss, count = secant_loss(action, native, torch.zeros_like(action),
        torch.tensor([benefit]), torch.ones_like(action, dtype=torch.bool), 10.)
    loss.backward()
    assert count == 0 and torch.isfinite(loss) and action.grad.count_nonzero() == 0


def test_batched_vjps_support_checkpointed_r16_first_passage_graph(monkeypatch):
    from tools.looplong.dry_run import batch
    from dataclasses import replace
    from wms.model.history import LONG_HISTORY_F
    from wms.cli.assemble import build_model, build_physics
    from wms.config.load import load_train_config
    from wms.train.inputs import prepare_inputs
    from wms.train.settings import TrainSettings
    from wms.train.longwm import gradient_parity
    cfg = load_train_config(str(ROOT / 'configs/c4_wm_loop14_r16_ft.toml'))
    # Exercise the production architecture/physics switches with synthetic
    # parameters, without resolving or opening any traffic assets.
    cfg = replace(cfg, model=replace(cfg.model, physics_params=None))
    m = build_model(cfg, build_physics(cfg, resolver=lambda _: pytest.fail('asset read')))
    m.checkpoint_forward = True
    for name, p in m.named_parameters():
        p.requires_grad_(not name.startswith(tuple(cfg.optim.freeze_prefixes)))
    settings = TrainSettings.from_config(cfg)
    probe = batch(4)
    probe['states'][..., 1] = 0.
    probe['states'][..., 2] = 1.
    probe['states'][:, :, 0, 2] = 0.
    probe['states'][:, :, 0, 3] = 1.
    probe['states'][..., 0] = (torch.tensor([-10., -25., -55.])[None, None, :]
                                 + torch.arange(5)[None, :, None] * 6.)
    probe['cav_teacher'][..., 0] = probe['states'][..., 0]
    probe['cav_teacher'][..., 1] = probe['states'][..., 1:4].argmax(-1).float()
    probe['driver_hist'] = torch.zeros(2, 3, m.driver_history_len, LONG_HISTORY_F)
    probe['driver_hist_mask'] = torch.zeros(2, 3, m.driver_history_len, dtype=torch.bool)
    inp = prepare_inputs(probe, m, settings, torch.device('cpu'), 4)
    parameters = [p for p in m.parameters() if p.requires_grad]
    initial_rng = torch.get_rng_state()
    results = []
    for on in ('0', '1'):
        monkeypatch.setenv('WMS_LOOPLONG_SPEED', on)
        torch.set_rng_state(initial_rng)
        roll = m.rollout(inp.initial, n_steps=4, cav_teacher=inp.teacher,
            present_seq=inp.present_seq, is_cav_seq=inp.is_cav,
            history=inp.hist, history_mask=inp.hist_mask,
            driver_history=inp.observed_hist, driver_history_mask=inp.observed_hist_mask)
        base = roll['positions'].square().mean()
        terms = {'speed': roll['speeds'].square().mean(),
                 'paired': (roll['speeds'][1] - roll['speeds'][0]).square().mean()}
        rng = torch.get_rng_state()
        weights, _ = gradient_parity(base, terms, parameters)
        loss = base + sum(weights[k] * value for k, value in terms.items())
        grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        assert torch.equal(rng, torch.get_rng_state())
        assert any(g is not None for g in grads)
        assert all(torch.isfinite(g).all() for g in grads if g is not None)
        assert any(bool(g.abs().sum() > 0) for g in grads if g is not None)
        results.append((weights, grads, roll['positions'].detach(), rng))
    assert results[0][0] == pytest.approx(results[1][0], rel=1e-5, abs=1e-7)
    assert torch.equal(results[0][2], results[1][2]) and torch.equal(results[0][3], results[1][3])
    for serial, batched in zip(results[0][1], results[1][1]):
        assert (serial is None) == (batched is None)
        if serial is not None:
            torch.testing.assert_close(serial, batched, rtol=1e-5, atol=1e-6)


def test_missing_real_cost_clocks_give_zero_beyond_trusted_credit(tmp_path, monkeypatch):
    from wms.control.looplong import configure, digest
    from wms.control import long_credit, looplong_real
    path, doc = boundary_files(tmp_path)
    doc.update(imagination_horizon_s=120, source_policy_checkpoint=str(tmp_path / 'actor.pt'),
               flags={'auto_horizon': True, 'long_credit': True})
    policy, group = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
    torch.save({'policy': policy.state_dict(), 'group_encoder': group.state_dict()}, doc['source_policy_checkpoint'])
    doc['source_policy_sha256'] = digest(doc['source_policy_checkpoint'])
    trainer = SimpleNamespace(config=SimpleNamespace(entry_pad_max=32, horizon=10),
        device=torch.device('cpu'), update_count=383, policy=policy, group_encoder=group,
        _curriculum_position=lambda advance=False: 383)
    monkeypatch.setattr(looplong_real.RealPairCredit, 'from_boundary',
                        lambda _: SimpleNamespace(excluded_worlds=set()))

    def fit(*args):
        raise long_credit.MissingCostClocks('converted real pairs have no entry_backlog clocks')
    monkeypatch.setattr(long_credit, 'fit_from_loop', fit)
    doc.update(pairs='pairs', pairing_receipt='proof', policy_returns='policy', native_returns='native')
    configure(trainer, doc, {}, tmp_path)
    assert trainer.long_credit is None and trainer.long_credit_requested
    assert trainer.long_credit_refusal['qualified'] is False
    assert trainer.horizon_curriculum.horizons_s == [10, 60]
    assert list(tmp_path.glob('CQR_*_REFUSED.json'))
