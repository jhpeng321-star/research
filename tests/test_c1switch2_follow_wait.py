"""Regression for the real pre-dispatch SSH closure; no scientific retry allowed."""
import json

import pytest

from tools.c1switch2 import modelops, watch


def test_release_observer_retries_transport_and_keeps_original_memory_gate(tmp_path, monkeypatch):
    calls, sleeps = [], []
    responses = iter([RuntimeError('Connection closed'),
        dict(wm_finished=True, free_mib=8191), dict(wm_finished=True, free_mib=9000),
        dict(wm_finished=True, free_mib=8192), dict(wm_finished=True, free_mib=9000)])

    def ssh(host, code, timeout):
        calls.append(host)
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(modelops, 'ssh', ssh)
    monkeypatch.setattr(modelops.time, 'sleep', sleeps.append)
    modelops.wait_for_release(dict(gpus=dict(gpuA='/root/own_a', gpuR='/root/own_r')), tmp_path)
    assert calls == ['gpuA', 'gpuA', 'gpuR', 'gpuA', 'gpuR']
    assert sleeps == [60, 60]
    assert 'Connection closed' in json.loads((tmp_path/'WAIT_TRANSPORT_ERROR.json').read_text())['error']
    assert json.loads((tmp_path/'WAIT.json').read_text())['states']['gpuA']['free_mib'] == 8192


def test_wait_resume_refuses_live_predecessor(tmp_path, monkeypatch):
    monkeypatch.setattr(modelops, 'EVIDENCE', tmp_path)
    local = tmp_path/'BB_FOLLOW'
    local.mkdir()
    config, after = tmp_path/'config.json', tmp_path/'after.json'
    config.write_text(json.dumps(dict(engine=dict(model='bb'))))
    after.write_text('{}')
    (local/'START.json').write_text(json.dumps(dict(pid=43404,
        config_sha256=modelops.sha(config), after_sha256=modelops.sha(after))))
    monkeypatch.setattr(watch, 'local_process', lambda pid: dict(state='LIVE'))
    with pytest.raises(ValueError, match='has not exited'):
        modelops.resume_follow_wait(config, after)
    assert not (local/'WAIT_RESUME_ONCE.json').exists()


def test_wait_resume_refuses_an_existing_correction(tmp_path, monkeypatch):
    monkeypatch.setattr(modelops, 'EVIDENCE', tmp_path)
    local = tmp_path/'BB_FOLLOW'
    local.mkdir()
    (local/'WAIT_RESUME_ONCE.json').write_text('{}')
    config, after = tmp_path/'config.json', tmp_path/'after.json'
    config.write_text(json.dumps(dict(engine=dict(model='bb'))))
    after.write_text('{}')
    with pytest.raises(ValueError, match='never duplicate'):
        modelops.resume_follow_wait(config, after)
