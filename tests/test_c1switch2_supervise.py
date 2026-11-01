"""Supervisor observes resource pressure and never repeats an admitted attempt."""
import json

import pytest

from tools.c1switch2 import supervise


def setup(tmp_path,monkeypatch,free):
    monkeypatch.setattr(supervise,'EVIDENCE',tmp_path)
    monkeypatch.setattr(supervise,'SCOPE',tmp_path)
    monkeypatch.setattr(supervise,'ssh',lambda *args,**kwargs:dict(load=1,free_bytes=free))
    monkeypatch.setattr(supervise.bbcontinue,'CLAIM',tmp_path/'BB_CLAIM.json')
    monkeypatch.setattr(supervise.bbcontinue,'BOUND',tmp_path/'BOUND.json')
    (tmp_path/'BOUND.json').write_text('{}')


def test_disk_pressure_causes_zero_claims(tmp_path,monkeypatch):
    setup(tmp_path,monkeypatch,100000000)
    monkeypatch.setattr(supervise.selectorops,'dispatch',lambda *args:pytest.fail('disk-blocked dispatch'))
    monkeypatch.setattr(supervise.bbcontinue,'start',lambda:pytest.fail('disk-blocked GPU allocation'))
    assert supervise.admit()['status']=='WAIT_CPU_DISK_NO_CLAIM'
    assert not (tmp_path/'FLEET_ATTEMPT_ONCE.json').exists()
    assert not (tmp_path/'BB_ATTEMPT_ONCE.json').exists()


def test_admitted_scopes_are_each_attempted_only_once(tmp_path,monkeypatch):
    setup(tmp_path,monkeypatch,3*1024**3)
    calls = []
    monkeypatch.setattr(supervise.selectorops,'dispatch',lambda *args:calls.append('fleet'))
    monkeypatch.setattr(supervise.bbcontinue,'resources',lambda config:({},True))
    monkeypatch.setattr(supervise.bbcontinue,'start',lambda:calls.append('bb'))
    supervise.admit();supervise.admit()
    assert calls==['fleet','bb']
    assert json.loads((tmp_path/'BB_ATTEMPT_ONCE.json').read_text())['states']=={}


def test_failed_science_is_not_retried(tmp_path,monkeypatch):
    setup(tmp_path,monkeypatch,3*1024**3)
    (tmp_path/'SELECTOR_DISPATCH_fleet_ppo_gpuR.json').write_text('{}')
    monkeypatch.setattr(supervise.bbcontinue,'resources',lambda config:({},True))
    calls = []
    def fail():
        calls.append('bb')
        raise RuntimeError('terminal startup')
    monkeypatch.setattr(supervise.bbcontinue,'start',fail)
    with pytest.raises(RuntimeError,match='terminal startup'):
        supervise.admit()
    supervise.admit()
    assert calls==['bb']
