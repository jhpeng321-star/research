"""C2 ACCEPT/budget boundaries and real configuration integration."""
import copy
import json
from pathlib import Path

import pytest

from tools.loopresume.c2prep_common import BUDGET_STOP, POLICY_SHA, budget_need, policy_boundary, resource_fits


def state():
    phases = {n: {'status': 'done'} for n in ('policy', 'collect', 'convert', 'wm_ft', 'check')}
    phases['wm_ft']['receipt'] = {'candidate_sha256': 'wm1'}
    phases['check']['receipt'] = {'decision': 'ACCEPT'}
    return {'sim_s_charged': 176473, 'policy': {'sha256': POLICY_SHA}, 'stopped': BUDGET_STOP,
            'accepted_wm': {'sha256': 'wm1'}, 'cycles': [{}, {'cycle': 1, 'phases': phases}]}


def test_policy_only_boundary_preserves_stop_and_charges():
    original = state()
    before = copy.deepcopy(original)
    s = policy_boundary(original)
    assert original == before
    assert s['sim_s_charged'] == 176473
    assert s['stopped'] is None and s['c2prep_prior_budget_stop'] == BUDGET_STOP
    assert s['cycles'][2]['phases']['collect'] == {'status': 'pending', 'attempt': 0}
    assert budget_need() == dict(charged=176473, cap=210000, remaining=33527,
        collection_need=36933, projected_total=213406, additional_cap_needed=3406)


@pytest.mark.parametrize('problem', ['reject', 'running', 'scientific_stop', 'wrong_wm', 'wrong_policy', 'charges'])
def test_boundary_fails_closed(problem):
    s = state()
    if problem == 'reject':
        s['cycles'][1]['phases']['check']['receipt']['decision'] = 'REJECT'
    elif problem == 'running':
        s['cycles'][1]['phases']['check']['status'] = 'running'
    elif problem == 'scientific_stop':
        s['stopped'] += '; SUMO safety'
    elif problem == 'wrong_wm':
        s['accepted_wm']['sha256'] = 'wm0'
    elif problem == 'wrong_policy':
        s['policy']['sha256'] = 'other'
    else:
        s['sim_s_charged'] += 1
    with pytest.raises(AssertionError):
        policy_boundary(s)


@pytest.mark.parametrize('problem', [None, 'oom', 'time', 'no60', 'incomplete', 'over60'])
def test_measured_resources(problem):
    smoke = dict(status='PASS', horizon_s=60, updates=[{'seconds': 5}]*3,
                 observed_training_horizons=[60], peak_reserved_bytes=20, allocator_cap_bytes=30,
                 wall_seconds=100)
    if problem == 'oom':
        smoke['peak_reserved_bytes'] = 31
    elif problem == 'time':
        smoke['updates'] = [{'seconds': 100}]*3
    elif problem == 'no60':
        smoke['observed_training_horizons'] = [10]
    elif problem == 'incomplete':
        smoke['updates'] = []
    elif problem == 'over60':
        smoke.update(horizon_s=90, observed_training_horizons=[90])
    assert resource_fits(smoke, 150, 10800) is (problem is None)


def test_resolved_config_and_horizon_gate(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(root)
    monkeypatch.setenv('C4_ALLOWED_ROOTS', str(root)+':'+str(tmp_path))
    from wms.control.policy14_cli import _read_config
    from wms.control.coopw import HorizonConfig
    raw = _read_config('configs/c4_policy14_c2prep.toml')
    prior = _read_config('configs/c4_policy14_loopresume.toml')
    expected = copy.deepcopy(prior)
    expected['run']['gpu_memory_fraction'] = 0.65
    expected['horizon']['curriculum_updates'] = [0, 150, 190]
    assert raw == expected
    assert raw['paircrit'] == {'enabled': True, 'warmup_updates': 100}
    assert not raw['longcritic']['enabled'] and not raw['breakdown']['enabled']
    assert not raw['real_share']['enabled']
    assert raw['trainer']['horizon'] == 10
    path = tmp_path/'gate.json'
    path.write_text(json.dumps({'schema': 'coopw_horizon_gate/1', 'wm_sha256': 'wm1',
                               'horizons': {'30': {'verdict': 'PASS'}, '60': {'verdict': 'PASS'}}}))
    horizon = HorizonConfig(**{**raw['horizon'], 'gate_file': str(path)})
    assert horizon.bind(['wm1'])['allowed_stage'] == 2
    assert horizon.horizons_s[horizon.stage(180)] == 30
    assert horizon.horizons_s[horizon.stage(190)] == 60
    assert horizon.horizons_s[horizon.stage(10000)] == 60
    assert horizon.bind(['wm0'])['allowed_stage'] == 0


def test_nonresource_smoke_failure_cannot_fallback():
    with pytest.raises(ValueError, match='no fallback'):
        resource_fits({'status': 'FAIL', 'error': 'paired source hash mismatch'}, 150, 10800)
    assert not resource_fits({'status': 'FAIL', 'resource_failure': True}, 150, 10800)


def test_driver_has_no_collection_dispatch():
    import ast
    source = (Path(__file__).resolve().parents[2]/'tools/loopresume/c2prep_box.py').read_text()
    calls = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr in ('run_phase', 'retry')]
    assert len(calls) == 1 and calls[0].args[1].value == 'policy'
