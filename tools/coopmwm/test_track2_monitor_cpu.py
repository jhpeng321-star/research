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


def test_monitor_complete_metrics_hash_chain_summary_and_duplicate(tmp_path, capsys, monkeypatch):
    cfg = dict(run_root=str(tmp_path))
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
    cfg = dict(run_root=str(tmp_path))
    for u in (1, 2, 3):
        row = monitor.monitor_update(cfg, 'R', u,
            metrics(layer_kl=0., trigger_counts=dict(none=4, gap_open=0, ease_off=0)))
    assert set(row['diagnose']) == {'trigger_saturated', 'layer_kl_near_zero',
        'group_credit_not_rising_3_updates', 'dense_accuracy_flat_3_updates'}
    assert row['action'] == 'DIAGNOSE'
    row = monitor.monitor_update(cfg, 'R', 4, metrics(safety_stop=True))
    assert row['action'] == 'PAUSE_SAFETY'


def test_flat_accuracy_not_imputed_when_all_labels_filtered(tmp_path):
    cfg = dict(run_root=str(tmp_path))
    for u in (1, 2, 3):
        row = monitor.monitor_update(cfg, 'B', u, metrics(dense_eligible_labels=0,
            dense_filtered_share=1., dense_ce=None, dense_accuracy=None, group_credit_wm=u))
    assert row['metrics']['dense_ce'] is None
    assert row['diagnose'] == ['no_eligible_dense_labels']


def test_plain_a_not_applicable_and_sampled_subset(tmp_path):
    cfg = dict(run_root=str(tmp_path))
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
    dict(families=['none', 'none']), dict(group_credit_checked=None, credit_checked_rows=1)])
def test_reject_invalid_or_incomplete_component_metrics(tmp_path, change):
    with pytest.raises(ValueError):
        monitor.monitor_update(dict(run_root=str(tmp_path)), 'B', 1, metrics(**change))
    assert not (tmp_path/'learning_monitor/B/u001.json').exists()


def test_missing_metric_skips_and_tampered_history_refused(tmp_path):
    cfg = dict(run_root=str(tmp_path))
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
        monitor.monitor_update(dict(run_root=str(root)), 'B', 1, metrics())


def test_concurrent_all_arm_status_preserves_latest(tmp_path):
    cfg = dict(run_root=str(tmp_path))
    # B/R may publish concurrently with the diagnostics thread.
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(lambda a: monitor.monitor_update(cfg, a, 1, metrics()), ('B', 'R')))
    assert len(rows) == 2
    assert monitor.read_monitor(tmp_path)['updates'] == dict(A=0, B=1, R=1)
    summary = (tmp_path/'CLEAN1_STATUS.md').read_text()
    assert 'B u1' in summary and 'R u1' in summary


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
        self.base_sha = _state_sha(old)

    def resolve_states(self, job):
        self.resolve_calls.append(job)
        return deepcopy(self.states[job['replicate']])

    def weights(self, arm, u):
        self.weight_calls.append((arm, u))
        return (self.old if arm == 'A' else self.new), self.old

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
        safety = dict.fromkeys(monitor.SAFETY, 0)
        if self.safety_regression and cfg['arm'] == 'B' and spec['branch'] == 'all':
            safety['collisions'] = 1
        J = {'all': 70. if cfg['arm'] == 'B' else 80., 'solo0': 90.,
             'solo1': 85., 'reference': 100.}[spec['branch']]
        if cfg['arm'] == 'A':
            assert _state_sha(snapshot['weights']) == self.base_sha
            J = 100.
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
    cfg['monitor_dry_run'] = False
    row = monitor.probe(cfg, u)
    assert row['cells'] == 3 and row['branches'] == 3*3*4
    assert row['quoted_sim_s'] == 3*3*4*840
    assert len(backend.resolve_calls) == 3
    assert all(j['replicate'] == 0 for j in backend.resolve_calls)
    assert all(len(r['entries']) == 3 for r in row['records'])
    assert [r['metric'] for r in row['records']] == [0., 15., 5.]
    assert 'comparisons' not in row
    assert backend.calls[0][2] == 'early_probes'


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
    assert not (tmp_path/'track2_diagnostics/replicated_u006_live.json').exists()


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
