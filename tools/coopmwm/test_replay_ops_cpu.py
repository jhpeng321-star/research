import json
from types import SimpleNamespace
from . import replay_ops


def test_absent_ready_does_not_dispatch(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(replay_ops.ops, 'LOCAL', tmp_path / 'coopnative')
    def forbidden(*args, **kwargs):
        raise AssertionError('readiness failure must not contact a host')
    monkeypatch.setattr(replay_ops.ops.ssh, 'remote', forbidden)
    replay_ops.qualify()
    assert json.loads(capsys.readouterr().out)['ready'] is False


def test_ready_but_unintegrated_does_not_dispatch(monkeypatch, tmp_path, capsys):
    import subprocess
    delegated = tmp_path / 'c4-harden-int'
    delegated.mkdir()
    (delegated / 'HARDEN_INT_READY.json').write_text(json.dumps(dict(commit='abc123')))
    monkeypatch.setattr(replay_ops.ops, 'LOCAL', tmp_path / 'coopnative')
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=1))
    def forbidden(*args, **kwargs):
        raise AssertionError('unintegrated release must not contact a host')
    monkeypatch.setattr(replay_ops.ops.ssh, 'remote', forbidden)
    replay_ops.qualify()
    value = json.loads(capsys.readouterr().out)
    assert value['ready'] is True and value['integrated'] is False
