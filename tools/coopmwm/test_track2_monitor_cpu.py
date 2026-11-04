"""P3 contracts with CPU tensors and a fake runner/accountant; no physics/remote."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from . import track2_monitor as monitor
from .exact_grade import verify_sealed
from .test_policy_synergy_cpu import fixture


def metrics(**changes):
    value = dict(group_credit_wm=2., group_credit_checked=None, dense_ce=.8,
        dense_accuracy=.5, dense_filtered_share=.25, dense_checked_accuracy=None,
        bank_move_ce=.7, layer_kl=.002, merge_missing_share=.1, reward=11.,
        decisions=4, credit_rows=4, credit_checked_rows=0, dense_labels=4,
        dense_eligible_labels=3, dense_checked_labels=0, augmented_examples=2,
        weighted_starts=1, families=['none', 'gap_open', 'ease_off'],
        trigger_counts=dict(none=2, gap_open=1, ease_off=1),
        safety=dict.fromkeys(monitor.SAFETY, 0), safety_stop=False)
    return dict(value, **changes)


def wm_config(root):
    # The superseded profile is now explicit, rather than a default-on component.
    return dict(run_root=str(root), group_credit=dict(enabled=True), dense_teacher=dict(enabled=True))


def test_monitor_complete_metrics_hash_chain_summary_and_duplicate(tmp_path, capsys, monkeypatch):
    cfg = wm_config(tmp_path)
    a = monitor.monitor_update(cfg, 'B', 1, dict(metrics=metrics(), sources=dict(labels='receipt-1')))
    b = monitor.monitor_update(cfg, 'B', 2, metrics(group_credit_wm=3.))
    assert a['trigger_rates'] == dict(none=.5, gap_open=.25, ease_off=.25)
    assert b['previous_monitor_sha256'] == a['monitor_sha256']
    assert b['action'] == 'CONTINUE'
    verify_sealed(a, 'monitor_sha256')
    assert monitor.monitor_update(cfg, 'B', 2, metrics(group_credit_wm=3.)) == b
    with pytest.raises(ValueError, match='conflicting'):
        monitor.monitor_update(cfg, 'B', 2, metrics(group_credit_wm=4.))
    summary = (tmp_path/'CLEAN1_STATUS.md').read_text()
    assert 'B u2: CONTINUE' in summary and 'Safety:' in summary and 'Trigger rates:' in summary
    assert 'dense CE/accuracy=0.8/0.5' in summary
    monkeypatch.setattr('sys.argv', ['monitor', 'status', '--root', str(tmp_path)])
    monitor.main()
    status = json.loads(capsys.readouterr().out)
    assert status['updates'] == dict(A=0, B=2, R=0)


def test_diagnose_only_learning_and_explicit_safety_pause(tmp_path):
    cfg = wm_config(tmp_path)
    for u in (1, 2, 3):
        row = monitor.monitor_update(cfg, 'R', u,
            metrics(layer_kl=0., trigger_counts=dict(none=4, gap_open=0, ease_off=0)))
    assert set(row['diagnose']) == {'trigger_saturated', 'layer_kl_near_zero',
        'group_credit_not_rising_3_updates', 'dense_accuracy_flat_3_updates'}
    assert row['action'] == 'DIAGNOSE'
    row = monitor.monitor_update(cfg, 'R', 4, metrics(safety_stop=True))
    assert row['action'] == 'PAUSE_SAFETY'


def test_flat_accuracy_not_imputed_when_all_labels_filtered(tmp_path):
    cfg = wm_config(tmp_path)
    for u in (1, 2, 3):
        row = monitor.monitor_update(cfg, 'B', u, metrics(dense_eligible_labels=0,
            dense_filtered_share=1., dense_ce=None, dense_accuracy=None, group_credit_wm=u))
    assert row['metrics']['dense_ce'] is None
    assert row['diagnose'] == ['no_eligible_dense_labels']


def test_plain_a_not_applicable_and_sampled_subset(tmp_path):
    cfg = wm_config(tmp_path)
    m = metrics()
    for k in monitor.METRICS:
        if k != 'reward':
            m[k] = None
    for k in monitor.COUNTS:
        m[k] = 0
    m['trigger_counts'] = dict.fromkeys(m['families'], 0)
    row = monitor.monitor_update(cfg, 'A', 1, m)
    assert row['action'] == 'CONTINUE' and set(row['trigger_rates'].values()) == {None}
    row = monitor.monitor_update(cfg, 'B', 1, metrics(credit_checked_rows=1,
        group_credit_checked=-2., dense_checked_labels=1, dense_checked_accuracy=0.))
    assert row['metrics']['group_credit_checked'] == -2.


@pytest.mark.parametrize('change', [dict(reward=float('nan')), dict(layer_kl=-1),
    dict(dense_accuracy=1.1), dict(dense_checked_labels=5), dict(credit_checked_rows=True),
    dict(trigger_counts=dict(none=2, gap_open=0, ease_off=0)), dict(decisions=0),
    dict(dense_filtered_share=0), dict(safety={'collisions': 0}), dict(safety_stop=None),
    dict(families=['none', 'none']), dict(group_credit_checked=None, credit_checked_rows=1),
    dict(group_credit_checked=0.), dict(dense_checked_accuracy=0.)])
def test_reject_invalid_or_incomplete_component_metrics(tmp_path, change):
    with pytest.raises(ValueError):
        monitor.monitor_update(wm_config(tmp_path), 'B', 1, metrics(**change))
    assert not (tmp_path/'learning_monitor/B/u001.json').exists()


def test_missing_metric_skips_and_tampered_history_refused(tmp_path):
    cfg = wm_config(tmp_path)
    m = metrics(); del m['bank_move_ce']
    with pytest.raises(ValueError, match='missing'):
        monitor.monitor_update(cfg, 'B', 1, m)
    with pytest.raises(ValueError, match='contiguous'):
        monitor.monitor_update(cfg, 'B', 2, metrics())
    monitor.monitor_update(cfg, 'B', 1, metrics())
    p = tmp_path/'learning_monitor/B/u001.json'
    row = json.loads(p.read_text()); row['metrics']['reward'] = 999
    p.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='edited'):
        monitor.read_monitor(tmp_path)


@pytest.mark.parametrize('suffix', ['sealed', 'holdout', 'heldout', 'cal_split', 'chunk5',
    'j4b', 'inc1', 'hetdata_truth', 'vType_truth.json', 'driver_truth.json', 'vehicle_params_truth.json'])
def test_forbidden_paths_refused_before_open(tmp_path, suffix):
    # No restricted directory/file is created or opened.
    with pytest.raises(ValueError, match='restricted'):
        monitor.read_monitor(tmp_path/suffix)


def test_guard_output_escape_without_reading_link_target(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    original = Path.resolve
    target = root/'learning_monitor/B'
    monkeypatch.setattr(Path, 'resolve', lambda p, *a, **k:
        root.parent/'outside' if p == target else original(p, *a, **k))
    with pytest.raises(ValueError, match='escapes'):
        monitor.monitor_update(wm_config(root), 'B', 1, metrics())


def test_concurrent_all_arm_status_preserves_latest(tmp_path):
    cfg = wm_config(tmp_path)
    # B/R may publish concurrently with the diagnostics thread.
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(lambda a: monitor.monitor_update(cfg, a, 1, metrics()), ('B', 'R')))
    assert len(rows) == 2
    assert monitor.read_monitor(tmp_path)['updates'] == dict(A=0, B=1, R=1)
    summary = (tmp_path/'CLEAN1_STATUS.md').read_text()
    assert 'B u1' in summary and 'R u1' in summary


def real_reward_config(root):
    return dict(run_root=str(root), group_credit=dict(enabled=False),
        dense_teacher=dict(enabled=False), track2_reward=True, track2_reward_weight=1,
        track2_reward_provenance='MAIN02:30 registered reward weight', track2_reward_arms=['B', 'R'])


def reward_receipt(arm):
    # Native P2 training_episodes/layer_records receipt fields, not WM advantages.
    return dict(weight=1., provenance='MAIN02:30 registered reward weight', arms=['B', 'R'],
        arm=arm, scope='cooperation layer only', wm_used=False, global_reward_used=False,
        terminal_fallback_steps=0, steps=4, cav_outcomes=8, empty_cav_steps=0,
        empty_neighbourhoods=1, neighbour_outcomes=10, partner_reports=2,
        unavailable_partner_reports=1, own_reward_sum=-2., neighbourhood_reward_sum=-3.,
        team_addition_sum=-3., reward_sum=-5., layer_records=4,
        reward_source='SUMO observed speed deficit; local sensing + timestamped V2V only',
        aggregation='equal acting-CAV mean; empty neighbourhood = 0',
        outcome_rule='negative measured speed deficit, vehicle-seconds per control tick')


def real_metrics(arm, **changes):
    m = metrics(track2_reward=reward_receipt(arm))
    for name in monitor.COMPONENT_METRICS:
        m.update(dict.fromkeys(monitor.COMPONENT_METRICS[name], None))
        m.update(dict.fromkeys(monitor.COMPONENT_COUNTS[name], 0))
    return dict(m, **changes)


@pytest.mark.parametrize('arm', ['B', 'R'])
def test_redesigned_profile_three_updates_native_reward_history_and_status(tmp_path, arm, capsys, monkeypatch):
    cfg = real_reward_config(tmp_path)
    rows = [monitor.monitor_update(cfg, arm, u,
        dict(metrics=real_metrics(arm), sources=dict(reward=f'p2-{arm}-{u}'))) for u in (1, 2, 3)]
    assert all(r['action'] == 'CONTINUE' and r['diagnose'] == [] for r in rows)
    assert rows[-1]['components']['track2_reward_weight'] == 1.
    assert rows[-1]['metrics']['track2_reward'] == reward_receipt(arm)
    assert rows[-1]['metrics']['reward'] == 11.  # base/evaluation reward preserved
    assert monitor.monitor_history(tmp_path, arm) == rows  # no cfg needed to reconstruct
    for r in rows:
        assert r['metrics']['group_credit_wm'] is None and r['metrics']['dense_filtered_share'] is None
        verify_sealed(r, 'monitor_sha256')
    assert monitor.monitor_update(cfg, arm, 3,
        dict(metrics=real_metrics(arm), sources=dict(reward=f'p2-{arm}-3'))) == rows[-1]
    summary = (tmp_path/'CLEAN1_STATUS.md').read_text()
    assert 'group WM=disabled (N/A)' in summary and 'dense teacher=disabled (N/A)' in summary
    assert 'Real observable team reward:' in summary and 'reward_sum=-5.0' in summary
    assert 'cav_outcomes=8' in summary and 'provenance=MAIN02:30' in summary
    assert 'not_rising' not in summary and 'dense_accuracy_flat' not in summary
    monkeypatch.setattr('sys.argv', ['monitor', 'status', '--root', str(tmp_path)])
    monitor.main()
    status = json.loads(capsys.readouterr().out)
    assert status['updates'][arm] == 3 and status['latest'][arm] == rows[-1]


@pytest.mark.parametrize('arm', ['B', 'R'])
def test_disabled_components_are_default_off_and_never_imputed(tmp_path, arm):
    row = monitor.monitor_update(dict(run_root=str(tmp_path)), arm, 1,
        real_metrics(arm, track2_reward=None))
    assert row['metrics']['track2_reward'] is None
    assert row['diagnose'] == []


@pytest.mark.parametrize('component', ['group_credit', 'dense_teacher'])
@pytest.mark.parametrize('arm', ['B', 'R'])
def test_enabled_components_remain_strict_and_independent(tmp_path, component, arm):
    cfg = real_reward_config(tmp_path)
    cfg[component]['enabled'] = True
    with pytest.raises(ValueError, match='invalid|missing learning component'):
        monitor.monitor_update(cfg, arm, 1, real_metrics(arm))
    m = real_metrics(arm)
    legacy = metrics()
    for key in monitor.COMPONENT_METRICS[component]+monitor.COMPONENT_COUNTS[component]:
        m[key] = legacy[key]
    for u in (1, 2, 3):
        row = monitor.monitor_update(cfg, arm, u, m)
    assert row['diagnose'] == (['group_credit_not_rising_3_updates'] if component == 'group_credit'
                               else ['dense_accuracy_flat_3_updates'])
    assert monitor.read_monitor(tmp_path)['latest'][arm] == row


@pytest.mark.parametrize('component', ['group_credit', 'dense_teacher'])
def test_enabled_component_counts_cannot_be_zero_even_with_numeric_scalars(tmp_path, component):
    cfg = wm_config(tmp_path)
    m = metrics()
    for key in monitor.COMPONENT_COUNTS[component]:
        m[key] = 0
    with pytest.raises(ValueError, match='missing learning component'):
        monitor.monitor_update(cfg, 'B', 1, m)


@pytest.mark.parametrize('change', [dict(group_credit=True), dict(dense_teacher=dict(enabled=1)),
    dict(track2_reward='true'), dict(track2_reward_weight=0), dict(track2_reward_weight=True),
    dict(track2_reward_provenance=' '), dict(track2_reward_arms=['B'])])
def test_component_registration_is_explicit_and_validated_before_persistence(tmp_path, change):
    with pytest.raises(ValueError):
        monitor.monitor_update(dict(real_reward_config(tmp_path), **change), 'B', 1, real_metrics('B'))
    assert not (tmp_path/'learning_monitor/B/u001.json').exists()


@pytest.mark.parametrize('change', [dict(group_credit_wm=0.), dict(credit_rows=1),
    dict(group_credit_checked=0.), dict(dense_labels=1), dict(dense_ce=0.),
    dict(dense_filtered_share=0.), dict(dense_checked_accuracy=0.)])
def test_disabled_components_refuse_fabricated_measurements(tmp_path, change):
    with pytest.raises(ValueError, match='disabled'):
        monitor.monitor_update(real_reward_config(tmp_path), 'B', 1, real_metrics('B', **change))
    assert not (tmp_path/'learning_monitor/B/u001.json').exists()


@pytest.mark.parametrize('change', [dict(weight=2.), dict(provenance='different'), dict(arm='R'),
    dict(wm_used=True), dict(global_reward_used=True), dict(terminal_fallback_steps=1),
    dict(steps=0), dict(cav_outcomes=0), dict(empty_neighbourhoods=9),
    dict(own_reward_sum=float('nan')), dict(reward_sum=None), dict(team_addition_sum=-4.),
    dict(neighbour_outcomes=True), dict(reward_sum=0.)])
def test_real_reward_requires_registered_finite_consistent_p2_evidence(tmp_path, change):
    m = real_metrics('B', track2_reward=dict(reward_receipt('B'), **change))
    with pytest.raises(ValueError, match='track2_reward'):
        monitor.monitor_update(real_reward_config(tmp_path), 'B', 1, m)
    assert not (tmp_path/'learning_monitor/B/u001.json').exists()


def test_real_reward_cannot_be_missing_or_reported_for_disabled_or_a(tmp_path):
    cfg = real_reward_config(tmp_path)
    for m in (real_metrics('B', track2_reward=None),
              {k: v for k, v in real_metrics('B').items() if k != 'track2_reward'}):
        with pytest.raises(ValueError, match='native P2'):
            monitor.monitor_update(cfg, 'B', 1, m)
    with pytest.raises(ValueError, match='disabled track2_reward'):
        monitor.monitor_update(dict(run_root=str(tmp_path)), 'B', 1, real_metrics('B'))
    a = real_metrics('A')
    with pytest.raises(ValueError, match='disabled track2_reward'):
        monitor.monitor_update(cfg, 'A', 1, a)
    a['track2_reward'] = None
    assert monitor.monitor_update(cfg, 'A', 1, a)['action'] == 'CONTINUE'


def test_registration_bound_to_history_and_rehashed_component_edit_refused(tmp_path):
    cfg = real_reward_config(tmp_path)
    monitor.monitor_update(cfg, 'B', 1, real_metrics('B'))
    with pytest.raises(ValueError, match='registration changed'):
        monitor.monitor_update(dict(cfg, track2_reward_weight=2.), 'B', 2, real_metrics('B'))
    path = tmp_path/'learning_monitor/B/u001.json'
    row = json.loads(path.read_text())
    row['components']['dense_teacher']['enabled'] = True
    path.write_text(json.dumps(monitor._seal(row, 'monitor_sha256')))
    with pytest.raises(ValueError):
        monitor.read_monitor(tmp_path)


def test_legacy_v1_history_keeps_original_contract(tmp_path):
    cfg = wm_config(tmp_path)
    folder = tmp_path/'learning_monitor/B'
    folder.mkdir(parents=True)
    history = []
    for u in (1, 2, 3):
        diagnoses = ['group_credit_not_rising_3_updates', 'dense_accuracy_flat_3_updates'] if u == 3 else []
        row = monitor._seal(dict(schema='coopmwm.track2.learning_monitor.v1', arm='B', update=u,
            metrics=metrics(), trigger_rates=dict(none=.5, gap_open=.25, ease_off=.25),
            diagnose=diagnoses, sources={}, settings=dict(monitor.MONITOR_DEFAULTS),
            action='DIAGNOSE' if u == 3 else 'CONTINUE',
            previous_monitor_sha256=history[-1]['monitor_sha256'] if history else None), 'monitor_sha256')
        (folder/f'u{u:03}.json').write_text(json.dumps(row))
        history.append(row)
    assert monitor.monitor_history(tmp_path, 'B') == history
    row = monitor.monitor_update(cfg, 'B', 4, metrics())
    assert row['schema'].endswith('.v2') and row['previous_monitor_sha256'] == history[-1]['monitor_sha256']
    assert monitor.read_monitor(tmp_path)['latest']['B'] == row


class Backend:
    """Real snapshot/spec expansion with fake accounting and branch grades."""
    def __init__(self):
        self.states = {}
        self.calls = []
        self.fail_branch = False
        self.safety_regression = False
        self.resolve_calls = []
        self.weight_calls = []
        self.weights_by_arm = {}
        self.base_shas = []
        self.weight_evidence = []
        from .sumo_verify import _state_sha
        for rep in range(5):
            values = []
            for i in range(4):
                s, r, new, old = fixture()
                s['episode']['role'] = 'train'
                s['episode']['world']['seed'] = 986048+rep
                r['verification_id'] = f'state{rep}_{i}'
                values.append((s, r))
            self.states[rep] = values
        self.new, self.old = new, old
        self.current = dict(A={'w': torch.full((2,), 2.)}, B=new, R=new)
        self.base_sha = _state_sha(old)

    def resolve_states(self, job):
        self.resolve_calls.append(job)
        return deepcopy(self.states[job['replicate']])

    def weights(self, arm, u):
        self.weight_calls.append((arm, u))
        return self.current[arm], self.old

    def reserve(self, jobs, phase):
        self.calls.append(('reserve', jobs, phase))
        return dict(reserved=True, phase=phase, sim_s=sum(j['sim_s'] for j in jobs),
                    job_ids=[j['id'] for j in jobs])

    def run_branch(self, spec, snapshot):
        from .sumo_verify import _state_sha
        assert self.calls[0][0] == 'reserve'
        self.calls.append(('branch', deepcopy(spec)))
        if self.fail_branch:
            return dict(status='failed')
        cfg = snapshot['policy_synergy']
        verify_sealed(spec, 'branch_sha256')
        assert cfg['background_old'] is True
        assert cfg['intervention_start_s'] == snapshot['t0']+1
        assert cfg['u3_state_sha256'] == self.base_sha
        expected = self.old if spec['branch'] == 'reference' else self.current[cfg['arm']]
        assert _state_sha(snapshot['weights']) == _state_sha(expected)
        assert cfg['new_state_sha256'] == _state_sha(expected)
        self.weight_evidence.append(dict(arm=cfg['arm'], branch=spec['branch'],
            current_sha=_state_sha(snapshot['weights']), reference_sha=_state_sha(snapshot['u3_weights']),
            members=cfg['new_members'], group=cfg['group'], background_old=cfg['background_old']))
        safety = dict.fromkeys(monitor.SAFETY, 0)
        if self.safety_regression and cfg['arm'] == 'B' and spec['branch'] == 'all':
            safety['collisions'] = 1
        J = {'all': 70. if cfg['arm'] == 'B' else 80., 'solo0': 90.,
             'solo1': 85., 'reference': 100.}[spec['branch']]
        if cfg['arm'] == 'A':
            J = dict(all=75., solo0=90., solo1=85., reference=100.)[spec['branch']]
        return dict(status='graded', J_veh_s=J, safety=safety)


def diagnostic_config(tmp_path):
    backend = Backend()
    rows = [dict(replicate=i, world_seed=986048+i,
        **monitor.diagnostic_identity(backend.states[i])) for i in range(5)]
    cfg = dict(run_root=str(tmp_path), monitor_backend=backend,
        isolated_registration=dict(role='train', metric='policy_cooperation_gain_veh_s', replicates=rows),
        budget=dict(phases=dict(early_probes=1_000_000, isolated_replicated=1_000_000)),
        early_probes=dict(arms=['A', 'B', 'R'], states=3, sim_s_per_arm=32610*3),
        isolated_test=dict(replicates=5, sim_s_per_replicate=32610, bootstrap=dict(monitor.BOOTSTRAP)))
    return cfg, backend


def test_replication_dry_real_specs_full_windows_and_no_charge(tmp_path):
    cfg, backend = diagnostic_config(tmp_path)
    row = monitor.replicated(cfg, 6)
    assert row['dry'] and row['cells'] == 10 and row['branches'] == 160
    assert row['quoted_sim_s'] == 10*4*4*840
    assert row['new_sim_s'] == 0 and not backend.calls
    assert {a for a, _ in backend.weight_calls} == {'B', 'R'}
    assert len(backend.resolve_calls) == 10
    assert monitor.replicated(cfg, 6) == row
    assert len(backend.resolve_calls) == 10
    assert not (tmp_path/'track2_diagnostics/claims').exists()


@pytest.mark.parametrize('u', [2, 4])
def test_quick_probes_one_run_three_states_per_arm(tmp_path, u):
    cfg, backend = diagnostic_config(tmp_path)
    cfg['early_probes'].pop('arms')  # exercise the default A/B/R wave
    cfg['monitor_dry_run'] = False
    row = monitor.probe(cfg, u)
    assert row['cells'] == 3 and row['branches'] == 3*3*4
    assert row['quoted_sim_s'] == 3*3*4*840
    assert len(backend.resolve_calls) == 3
    assert all(j['replicate'] == 0 for j in backend.resolve_calls)
    assert all(len(r['entries']) == 3 for r in row['records'])
    assert [r['metric'] for r in row['records']] == [10., 15., 5.]
    assert 'comparisons' not in row
    assert backend.calls[0][2] == 'early_probes'
    # Both scheduled probes must evaluate trained A, including when arms are omitted.
    from .sumo_verify import _state_sha
    current_sha = _state_sha(backend.current['A'])
    assert current_sha != backend.base_sha
    evidence = [e for e in backend.weight_evidence if e['arm'] == 'A']
    assert len(evidence) == 12
    for e in evidence:
        assert e['reference_sha'] == backend.base_sha and e['background_old']
        assert e['current_sha'] == (backend.base_sha if e['branch'] == 'reference' else current_sha)
        if e['branch'] in ('all', 'reference'):
            assert e['members'] == e['group']
        else:
            assert len(e['members']) == 1


def test_live_replication_paired_world_bootstrap_correct_and_immutable(tmp_path):
    cfg, backend = diagnostic_config(tmp_path)
    cfg['monitor_dry_run'] = False
    row = monitor.replicated(cfg, 24)
    assert row['status'] == 'complete' and row['verdict'] == 'pass'
    assert row['comparisons']['B-common_base']['n'] == 5
    assert row['comparisons']['B-common_base']['mean'] == 15.
    assert row['comparisons']['R-common_base']['mean'] == 5.
    assert row['comparisons']['B-R']['mean'] == 10.
    assert row['comparisons']['B-R']['ci_lo'] == row['comparisons']['B-R']['ci_hi'] == 10.
    assert row['safety_ok']
    ids = [c[1]['verification_id'] for c in backend.calls if c[0] == 'branch']
    assert len(ids) == len(set(ids)) == 160
    jobs = backend.calls[0][1]
    assert [j['id'] for j in jobs] == ids
    assert sum(j['sim_s'] for j in jobs) == row['new_sim_s']
    assert all(c[1]['window']['end_s'] == 840 for c in backend.calls if c[0] == 'branch')
    before = len(backend.calls)
    assert monitor.replicated(cfg, 24) == row and len(backend.calls) == before
    verify_sealed(row, 'receipt_sha256')
    assert 'verdict=pass' in (tmp_path/'CLEAN1_STATUS.md').read_text()


def test_replicated_safety_regression_fails_primary_not_secondary(tmp_path):
    cfg, backend = diagnostic_config(tmp_path)
    cfg['monitor_dry_run'] = False
    backend.safety_regression = True
    row = monitor.replicated(cfg, 24)
    assert row['verdict'] == 'fail' and not row['safety_ok']
    row = monitor.replicated(cfg, 12)
    assert row['verdict'] == 'descriptive'


def test_interrupted_failed_wave_never_recharged_or_retried(tmp_path):
    cfg, backend = diagnostic_config(tmp_path)
    cfg['monitor_dry_run'] = False
    backend.fail_branch = True
    with pytest.raises(ValueError, match='ungraded'):
        monitor.replicated(cfg, 6)
    assert len(backend.calls) == 2 and backend.calls[0][0] == 'reserve'
    with pytest.raises(ValueError, match='never replay'):
        monitor.replicated(cfg, 6)
    assert len(backend.calls) == 2
    failed = json.loads((tmp_path/'track2_diagnostics/replicated_u006_live.json').read_text())
    assert failed['status'] == 'failed' and failed['new_sim_s'] is None
    branch = json.loads(next((tmp_path/'track2_diagnostics/branches').glob('*.json')).read_text())
    assert branch['result']['status'] == 'failed'
    assert 'replicated u6: failed' in (tmp_path/'CLEAN1_STATUS.md').read_text()


def test_budget_denial_before_branch_leaves_nonreplayable_claim(tmp_path):
    cfg, backend = diagnostic_config(tmp_path)
    cfg['monitor_dry_run'] = False
    def refuse(*a):
        raise ValueError('budget denied')
    backend.reserve = refuse
    with pytest.raises(ValueError, match='budget denied'):
        monitor.probe(cfg, 2)
    with pytest.raises(ValueError, match='never replay'):
        monitor.probe(cfg, 2)
    assert not backend.calls


def test_incomplete_accounting_receipt_cannot_dispatch(tmp_path):
    cfg, backend = diagnostic_config(tmp_path)
    cfg['monitor_dry_run'] = False
    backend.reserve = lambda *a: dict(reserved=False)
    with pytest.raises(ValueError, match='accounting receipt'):
        monitor.probe(cfg, 2)
    assert not backend.calls


def test_branch_exception_retained_and_never_replayed(tmp_path):
    cfg, backend = diagnostic_config(tmp_path)
    cfg['monitor_dry_run'] = False
    def fail(*a):
        raise RuntimeError('fake plant interrupted')
    backend.run_branch = fail
    with pytest.raises(RuntimeError, match='interrupted'):
        monitor.probe(cfg, 2)
    evidence = json.loads(next((tmp_path/'track2_diagnostics/branches').glob('*.json')).read_text())
    assert evidence['result']['status'] == 'failed'
    assert 'fake plant interrupted' in evidence['result']['error']
    with pytest.raises(ValueError, match='never replay'):
        monitor.probe(cfg, 2)


@pytest.mark.parametrize('mutation', ['phase_cap', 'cell_cap', 'state', 'window',
    'seed', 'role', 'replicate', 'n', 'bootstrap', 'few_states'])
def test_registration_or_quote_errors_before_any_accounting(tmp_path, mutation):
    cfg, backend = diagnostic_config(tmp_path)
    cfg['monitor_dry_run'] = False
    reg = cfg['isolated_registration']
    if mutation == 'phase_cap':
        cfg['budget']['phases']['isolated_replicated'] = 1
    elif mutation == 'cell_cap':
        cfg['isolated_test']['sim_s_per_replicate'] = 1
    elif mutation == 'state':
        backend.states[0][0][1]['verification_id'] = 'changed'
    elif mutation == 'window':
        backend.states[0][0][1]['window']['end_s'] -= 1
    elif mutation == 'seed':
        reg['replicates'][0]['world_seed'] = reg['replicates'][1]['world_seed']
    elif mutation == 'role':
        reg['role'] = 'eval'
    elif mutation == 'replicate':
        reg['replicates'][0]['replicate'] = True
    elif mutation == 'n':
        reg['replicates'].pop()
    elif mutation == 'bootstrap':
        cfg['isolated_test']['bootstrap']['reps'] = 100
    elif mutation == 'few_states':
        backend.states[0] = backend.states[0][:2]
        reg['replicates'][0].update(monitor.diagnostic_identity(backend.states[0]))
    with pytest.raises(ValueError):
        (monitor.probe if mutation == 'few_states' else monitor.replicated)(cfg, 2 if mutation == 'few_states' else 6)
    assert not backend.calls
    assert not (tmp_path/'track2_diagnostics/claims').exists()


def test_unscheduled_calls_are_zero_side_effect():
    assert monitor.probe({}, 1)['status'] == 'not_scheduled'
    assert monitor.replicated({}, 18)['status'] == 'not_scheduled'
    assert monitor.replicated({}, 2)['new_sim_s'] == 0


def test_registered_role_and_pairing_validated_at_snapshot_boundary(tmp_path):
    cfg, backend = diagnostic_config(tmp_path)
    states = deepcopy(backend.states[0])
    job = dict(id='test', arm='B', update=6, world_seed=986048, **monitor.diagnostic_identity(states))
    states[0][0]['episode']['role'] = 'eval'
    job.update(monitor.diagnostic_identity(states))
    with pytest.raises(ValueError, match='TRAIN'):
        monitor.isolated_expansion(job, states, backend.new, backend.old)


@pytest.mark.parametrize('different_common', [True, False])
def test_checkpoint_must_stay_fixed_and_common_across_cells(tmp_path, different_common):
    cfg, backend = diagnostic_config(tmp_path)
    calls = []
    def changing_weights(arm, u):
        calls.append(arm)
        if len(calls) == 2:
            return (backend.new, backend.new) if different_common else (backend.old, backend.old)
        return backend.new, backend.old
    backend.weights = changing_weights
    with pytest.raises(ValueError, match='checkpoint changed'):
        monitor.replicated(cfg, 6)
    assert not backend.calls


def test_bootstrap_pairs_worlds_and_not_individual_states():
    rows = []
    for i in range(5):
        for arm in ('B', 'R'):
            rows.append(dict(arm=arm, replicate=i, world_seed=986048+i,
                state_set_sha256=str(i), window_set_sha256=str(i), metric=100*i+(3 if arm == 'B' else 0),
                safety=dict.fromkeys(monitor.SAFETY, 0), reference_safety=dict.fromkeys(monitor.SAFETY, 0)))
    row = monitor._analyse(rows, 24, monitor.BOOTSTRAP)
    assert row['comparisons']['B-R']['ci_lo'] == row['comparisons']['B-R']['ci_hi'] == 3
    assert row['comparisons']['B-R']['n'] == 5
    with pytest.raises(ValueError, match='incomplete'):
        monitor._analyse(rows[:-1], 24, monitor.BOOTSTRAP)
    rows[-1]['world_seed'] += 1
    with pytest.raises(ValueError, match='unpaired'):
        monitor._analyse(rows, 24, monitor.BOOTSTRAP)
