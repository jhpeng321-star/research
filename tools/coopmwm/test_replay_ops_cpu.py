import json
from types import SimpleNamespace
from . import replay_ops


def test_durable_cpu_runner_compiles():
    p = replay_ops.ops.LOCAL/'tools/coopmwm/ls_cpu_test.py'
    compile(p.read_text(encoding='utf-8'), str(p), 'exec')


def test_integrated_release_payload_compiles_without_remote_dispatch(monkeypatch):
    monkeypatch.setattr(replay_ops,'release_status',lambda:dict(ready=True,integrated=True,release=dict(commit='abc')))
    monkeypatch.setattr(replay_ops,'checked',lambda p:SimpleNamespace(read_text=lambda **kw:'# CPU stub'))
    def remote(host,code,**kwargs):
        if host=='gpuA':
            assert 'phys_code_22cc32f65a80' in code
            assert "'wms/control/alwayscmd.py' in files" in code
            return json.dumps({'wms/control/alwayscmd.py':'# frozen source'})
        assert host=='cpuQ'
        compile(code,'<ready-release-qualification>','exec')
        assert 'DELEGATED_RELEASE.json' in code
        return '{}'
    monkeypatch.setattr(replay_ops.ops.ssh,'remote',remote)
    replay_ops.qualify()


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
