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
