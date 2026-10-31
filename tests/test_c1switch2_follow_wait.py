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


def test_offline_first_cpu_does_not_hide_later_cpu_stop(monkeypatch):
    codes = []
    config = dict(cpus=dict(cpuQ='/home/ubuntu/own_cpu', gpuR='/root/own_cpu'),
                  gpus=dict(gpuA='/root/own_gpu', gpuR='/root/own_gpu'), engine=dict(model='bb'))

    def ssh(host, code, timeout):
        codes.append((host, code))
        if host == 'cpuQ':
            raise RuntimeError('Connection closed')
        if 'expected=' in code:
            return dict(stop=dict(reason='child failed'), episodes=2, expected=4, complete=False)
        return dict(stop=None, finished=None)

    monkeypatch.setattr(modelops, 'ssh', ssh)
    monkeypatch.setattr(modelops, 'transfer', lambda *args: pytest.fail('STOP must prevent transfer'))
    result = modelops.relay_cycle(config)
    assert result['cpus']['cpuQ']['unknown'] is True
    assert result['failed'] is True and result['episodes_done'] is False
    assert len(codes) == 4
    assert all('if True and not path.exists()' in code for _, code in codes[2:])


def test_offline_first_cpu_does_not_starve_live_cpu_immutable_routes(monkeypatch):
    routes = []
    config = dict(cpus=dict(cpuQ='/home/ubuntu/own_cpu', gpuR='/root/own_cpu'),
                  gpus=dict(gpuA='/root/own_gpu', gpuR='/root/own_gpu'), engine=dict(model='bb'))

    def ssh(host, code, timeout):
        if host == 'cpuQ':
            raise RuntimeError('Connection closed')
        if 'expected=' in code:
            return dict(stop=None, episodes=2, expected=4, complete=False)
        return dict(stop=None, finished=None)

    def transfer(*args):
        routes.append(args)
        return 1

    monkeypatch.setattr(modelops, 'ssh', ssh)
    monkeypatch.setattr(modelops, 'transfer', transfer)
    result = modelops.relay_cycle(config)
    assert [args[2] for args in routes] == ['gpuA', 'gpuR']
    assert all(args[0] == 'gpuR' for args in routes)
    assert result['transferred'] == 2 and result['failed'] is False
    assert result['episodes_done'] is False
