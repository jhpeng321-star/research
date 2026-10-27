"""Operational recovery cannot reset the curve, replace the WM, or bypass vetoes."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wms.control import policy14_recovery as recovery


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'checked_path', Path)
    files = {}
    for name in ('config', 'gate_panel', 'checkpoint', 'wm', 'wm_config', 'returns'):
        p = tmp_path/name
        p.write_text(name)
        files[name] = str(p)
    def digest(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    prior = {'status': 'failed', 'error': 'OutOfMemoryError: CUDA out of memory',
             'real_share': {'source_sha256': digest(files['returns'])},
             'seed': 965141, 'resumed_from': {'checkpoint': 'original.pt', 'update_count': 30},
             'wm_members': [{'checkpoint_sha256': digest(files['wm']),
                             'config_sha256': digest(files['wm_config'])}],
             'native_curve': [{'update': u, 'passed': True, 'mean_delta': -u,
                               'panel_sha256': recovery.PANEL_SHA256} for u in range(30, 171, 10)],
             'updates': [{'update': u} for u in range(31, 175)], 'gate_panel': {'passed': True}}
    summary = tmp_path/'summary.json'
    summary.write_text(json.dumps(prior))
    files['summary'] = str(summary)
    receipt = {k: files[k] for k in ('config', 'gate_panel', 'checkpoint', 'summary')}
    receipt.update({k+'_sha256': digest(v) for k, v in receipt.copy().items()})
    receipt.update(retry_outputs=[str(tmp_path/'retry')], saved_update=170, seed=965141, phase_updates=150)
    path = tmp_path/'receipt.json'
    path.write_text(json.dumps(receipt))
    args = SimpleNamespace(native_init=True, seed=965141, max_updates=150,
                           config=files['config'], gate_panel=files['gate_panel'], resume='original.pt',
                           output_dir=str(tmp_path/'retry'), real_returns=files['returns'],
                           wm_ckpt=[files['wm']], wm_config=[files['wm_config']])
    return {'run': {'operational_recovery': str(path)}}, args, files


def test_remaining_updates_and_full_curve(case):
    raw, args, files = case
    row = recovery.load_recovery(raw, args)
    assert (args.resume, args.max_updates, args.critic_warmup_updates) == (files['checkpoint'], 10, 0)
    assert row['phase_start'] == 30 and len(row['updates']) == 140
    assert row['curve'][-1]['update'] == 170


@pytest.mark.parametrize('change', ['wm', 'checkpoint', 'config', 'gate_panel', 'returns'])
def test_changed_pinned_artifact_refused(case, change):
    raw, args, files = case
    Path(files[change]).write_text('changed')
    with pytest.raises(ValueError, match='mismatch'):
        recovery.load_recovery(raw, args)


def test_later_cycle_does_not_replay_recovery(case):
    raw, args, _ = case
    args.output_dir += '_later_cycle'
    assert recovery.load_recovery(raw, args) is None


def test_native_gate_cannot_be_disabled(case):
    raw, args, _ = case
    args.native_init = False
    with pytest.raises(ValueError, match='identity mismatch'):
        recovery.load_recovery(raw, args)


def test_non_round_phase_start_without_share_selection(case):
    raw, args, files = case
    prior = json.loads(Path(files['summary']).read_text())
    source_sha = prior.pop('real_share')['source_sha256']
    prior['resumed_from']['update_count'] = 233
    prior['native_curve'] = [dict(prior['native_curve'][-1], update=240)]
    prior['updates'] = [{'update': u} for u in range(234, 250)]
    Path(files['summary']).write_text(json.dumps(prior))
    path = Path(raw['run']['operational_recovery'])
    receipt = json.loads(path.read_text())
    receipt.update(saved_update=240, real_returns_sha256=source_sha,
                   summary_sha256=hashlib.sha256(Path(files['summary']).read_bytes()).hexdigest())
    path.write_text(json.dumps(receipt))
    result = recovery.load_recovery(raw, args)
    assert args.max_updates == 143 and result['phase_start'] == 233
    assert [r['update'] for r in result['updates']] == list(range(234, 241))


def test_missing_returns_binding_without_share_refused(case):
    raw, args, files = case
    prior = json.loads(Path(files['summary']).read_text())
    prior.pop('real_share')
    Path(files['summary']).write_text(json.dumps(prior))
    path = Path(raw['run']['operational_recovery'])
    receipt = json.loads(path.read_text())
    receipt['summary_sha256'] = hashlib.sha256(Path(files['summary']).read_bytes()).hexdigest()
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='returns hash mismatch'):
        recovery.load_recovery(raw, args)


def test_saved_stop_retains_latest_state_and_separate_native_best(case):
    raw, args, files = case
    summary_path = Path(files['summary'])
    cause = summary_path.with_name('cause.json')
    cause.write_bytes(summary_path.read_bytes())
    prior = json.loads(summary_path.read_text())
    prior['error'] = 'RuntimeError: own-run STOP requested; checkpoint saved'
    prior['resumed_from'] = {'checkpoint': 'best170.pt', 'update_count': 170}
    prior['updates'] = [{'update': u} for u in range(31, 172)]
    summary_path.write_text(json.dumps(prior))
    path = Path(raw['run']['operational_recovery'])
    receipt = json.loads(path.read_text())
    receipt.update(saved_update=171, best_update=170, best_checkpoint=files['checkpoint'],
                   best_checkpoint_sha256=receipt['checkpoint_sha256'], cause_summary=str(cause),
                   cause_summary_sha256=hashlib.sha256(cause.read_bytes()).hexdigest(),
                   summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest())
    path.write_text(json.dumps(receipt))
    args.resume = 'best170.pt'
    result = recovery.load_recovery(raw, args)
    assert args.max_updates == 9 and result['phase_start'] == 30
    assert result['updates'][-1]['update'] == 171 and result['curve'][-1]['update'] == 170


def test_phase_length_checked_before_recovery_remainder(monkeypatch, tmp_path):
    from wms.control import nativeinit, policy14_cli as cli
    panel = tmp_path / 'panel.json'
    panel.write_text('{}')
    monkeypatch.setattr(cli, 'checked_path', Path)
    monkeypatch.setattr(nativeinit, 'PANEL_SHA256', hashlib.sha256(panel.read_bytes()).hexdigest())
    monkeypatch.setattr(cli, '_read_config', lambda _: {'policy14': {}, 'trainer': {}})
    def load(raw, args):
        args.max_updates = 143
        return {'phase_updates': 150}
    monkeypatch.setattr(recovery, 'load_recovery', load)
    def reached_config(**kwargs):
        raise ValueError('reached trainer after original phase-length validation')
    monkeypatch.setattr(cli, 'DreamerConfig', reached_config)
    printed = []
    monkeypatch.setattr('builtins.print', lambda text: printed.append(json.loads(text)))
    assert cli.main(['--config', 'unused.toml', '--wm-ckpt', 'unused.pt', '--wm-config', 'unused.toml',
                     '--real-returns', 'unused.npz', '--output-dir', str(tmp_path / 'out'),
                     '--seed', '965142', '--max-updates', '150', '--native-init',
                     '--gate-panel', str(panel), '--compact']) == 1
    assert 'reached trainer' in printed[-1]['error']


@pytest.fixture
def long_resume(tmp_path, monkeypatch):
    import torch
    from wms.control.coopw import ExplorationConfig, GATE_SCHEMA
    from wms.control.dreamer import _rng_state
    from wms.control.looplong_real import RealPairCredit
    monkeypatch.setattr(recovery, 'checked_path', Path)
    monkeypatch.setattr('wms.control.start_states.checked_path', Path)
    monkeypatch.setattr(RealPairCredit, 'from_boundary', lambda d: SimpleNamespace(excluded_worlds=[]))
    gate = tmp_path / 'gate.json'
    gate.write_text(json.dumps({'schema': GATE_SCHEMA, 'wm_sha256': 'frozen-wm',
        'horizons': {str(h): {'verdict': 'PASS'} for h in (10, 60, 90)}}))
    boundary = {'wm': {'sha256': 'frozen-wm'}, 'horizon_s': 90, 'flags': {'long_credit': True}}
    horizon = {'enabled': True, 'horizons_s': [10., 60., 90.],
        'curriculum_updates': [0., 380., 463.], 'entry_pad_max': [32, 192, 288],
        'gate_file': str(gate), 'allowed_stage': 2}
    refusal = tmp_path / 'CQR_frozen-wm_REFUSED.json'
    refusal.write_text('{"qualified": false, "reason": "missing clocks"}')
    payload = {'update_count': 514, 'sampler': None, 'rng': _rng_state(torch.device('cpu')),
        'looplong': {'boundary': boundary, 'cqr': None, 'horizon': horizon,
                     'exploration': ExplorationConfig(enabled=True, long_persistence=True).as_dict()}}
    path = tmp_path / 'policy_000514.pt'
    torch.save(payload, path)
    receipt = {'checkpoint': str(path), 'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'saved_update': 514, 'prior': {'looplong_long_credit': {'qualified': False,
            'reason': 'missing clocks', 'artifact_sha256': hashlib.sha256(refusal.read_bytes()).hexdigest()}}}
    out = tmp_path / 'retry'
    out.mkdir()
    return SimpleNamespace(device=torch.device('cpu')), boundary, receipt, out, payload, gate


def test_looplong_recovery_preserves_90s_schedule_rng_and_credit_refusal(long_resume):
    import torch
    trainer, boundary, receipt, out, payload, _ = long_resume
    torch.rand(10)
    proof = recovery.restore_looplong(trainer, boundary, receipt, out)
    assert proof['horizon'] == payload['looplong']['horizon']
    assert trainer.horizon_curriculum.stage(514) == 2
    assert trainer.horizon_curriculum.horizons_s[trainer.horizon_curriculum.stage(514)] == 90
    assert torch.equal(torch.get_rng_state(), payload['rng']['torch'])
    assert trainer.long_credit is None and trainer.long_credit_refusal['qualified'] is False
    assert (out / 'CQR_frozen-wm_REFUSED.json').read_bytes() == (out.parent / 'CQR_frozen-wm_REFUSED.json').read_bytes()


def test_looplong_recovery_changed_gate_refused(long_resume):
    trainer, boundary, receipt, out, _, gate = long_resume
    data = json.loads(gate.read_text())
    data['horizons']['90']['verdict'] = 'FAIL'
    gate.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='trusted horizon gate changed'):
        recovery.restore_looplong(trainer, boundary, receipt, out)


def test_looplong_recovery_boundary_change_refused(long_resume):
    trainer, boundary, receipt, out, _, _ = long_resume
    with pytest.raises(ValueError, match='saved boundary mismatch'):
        recovery.restore_looplong(trainer, {**boundary, 'horizon_s': 60}, receipt, out)
