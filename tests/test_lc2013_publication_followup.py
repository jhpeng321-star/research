"""Publication observer integrity only; no worker/fitter/simulator launch."""
from datetime import datetime
import hashlib
import json

import pytest

from tools.rblane import watch_fit_precision as watch


def test_publication_requires_explicit_native_raw_fit_directory(tmp_path, monkeypatch):
    state = tmp_path / 'watch.json'
    monkeypatch.setattr(watch, 'FIT_ROOT', tmp_path / 'fit')
    monkeypatch.setattr(watch, 'FIT_STATE', state)
    state.write_text(json.dumps({'latest_publication': {'directory': str(tmp_path / 'foreign')}}))
    assert watch.latest() is None
    state.write_text(json.dumps({'latest_publication': {'directory': str(tmp_path / 'fit/lane_directional_v3')}}))
    assert watch.latest() is None


def test_completed_publication_hash_checked_before_use(tmp_path, monkeypatch):
    directory = tmp_path / 'lane_raw_0123456789ab'
    directory.mkdir()
    verified = directory / 'VERIFIED.json'
    verified.write_text('{"status":"PASS"}')
    expected = hashlib.sha256(verified.read_bytes()).hexdigest()
    (directory / 'PUBLICATION.json').write_text(json.dumps(dict(status='PASS', verified_sha256=expected)))
    state = tmp_path / 'watch.json'
    state.write_text(json.dumps({'latest_publication': {'directory': str(directory)}}))
    monkeypatch.setattr(watch, 'FIT_ROOT', tmp_path)
    monkeypatch.setattr(watch, 'FIT_STATE', state)
    assert watch.latest() == (directory, expected)
    verified.write_text('{"status":"CHANGED"}')
    with pytest.raises(AssertionError):
        watch.latest()


def test_unrelated_work_blocks_handoff_and_git_staging(monkeypatch):
    monkeypatch.setattr(watch, 'git', lambda *args: '?? user_work.py')
    monkeypatch.setattr(watch.subprocess, 'run', lambda *args, **kw: pytest.fail('must not stage or commit'))
    with pytest.raises(AssertionError, match='unrelated work'):
        watch.publish([], 'new report', 'unused')


def test_only_marked_handoff_block_is_replaced():
    original = 'source and user notes\n' + watch.BEGIN + '\nold\n' + watch.END + '\nrest\n'
    assert watch.replace_block(original, 'new') == (
        'source and user notes\n' + watch.BEGIN + '\nnew\n' + watch.END + '\nrest\n')
    with pytest.raises(AssertionError):
        watch.replace_block(original + watch.BEGIN, 'new')


def test_launch_past_deadline_never_starts_worker(monkeypatch):
    monkeypatch.setattr(watch.subprocess, 'Popen', lambda *args, **kw: pytest.fail('must not launch'))
    with pytest.raises(AssertionError, match='window already closed'):
        watch.launch(datetime.fromisoformat('2020-01-01T09:00:00+08:00'))


def test_duplicate_dispatch_receipt_blocks_worker(tmp_path, monkeypatch):
    dispatch = tmp_path / 'dispatch.json'
    dispatch.write_text('{"status":"DISPATCHED","pid":123}')
    monkeypatch.setattr(watch, 'DISPATCH', dispatch)
    monkeypatch.setattr(watch, 'RUNTIME', tmp_path / 'new_runtime')
    monkeypatch.setattr(watch, 'git', lambda *args: '')
    monkeypatch.setattr(watch.subprocess, 'Popen', lambda *args, **kw: pytest.fail('duplicate worker'))
    with pytest.raises(AssertionError):
        watch.launch(datetime.fromisoformat('2030-01-01T09:00:00+08:00'))
    assert dispatch.read_text() == '{"status":"DISPATCHED","pid":123}'
