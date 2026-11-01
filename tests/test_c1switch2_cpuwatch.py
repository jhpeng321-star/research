"""CPU resource waits never replay an attempted scope or admit a GPU follower."""
import json

import pytest

from tools.c1switch2 import cpuwatch


def setup(tmp_path,monkeypatch,**changes):
    monkeypatch.setattr(cpuwatch,'EVIDENCE',tmp_path)
    monkeypatch.setattr(cpuwatch,'SCOPE',tmp_path)
    (tmp_path/'SELECTOR_PREPARED_cav_mappo_cpuQ.json').write_text(json.dumps(dict(root='/home/ubuntu/c1switch2_run_test')))
    states = dict(load=1,free_bytes=3*1024**3,claimed=False,stopped=False,jobs=False,ledger=False)
    states.update(changes)
    monkeypatch.setattr(cpuwatch,'ssh',lambda *a,**k:states)
    return states


@pytest.mark.parametrize('changes',[dict(load=28),dict(free_bytes=2*1024**3-1),dict(claimed=True),
                                   dict(stopped=True),dict(jobs=True),dict(ledger=True)])
def test_refused_cpu_scope_never_claims_or_dispatches(tmp_path,monkeypatch,changes):
    setup(tmp_path,monkeypatch,**changes)
    monkeypatch.setattr(cpuwatch.selectorops,'dispatch',lambda *a:pytest.fail('inadmissible dispatch'))
    cpuwatch.admit_cpu()
    assert not (tmp_path/'CPU_ATTEMPT_ONCE.json').exists()


def test_only_original_cpu_scope_once(tmp_path,monkeypatch):
    setup(tmp_path,monkeypatch)
    calls=[]
    monkeypatch.setattr(cpuwatch.selectorops,'dispatch',lambda *a:calls.append(a))
    cpuwatch.admit_cpu();cpuwatch.admit_cpu()
    assert calls==[('cpuQ',1,'cav_mappo')]


def test_failed_attempt_never_automatically_retried(tmp_path,monkeypatch):
    setup(tmp_path,monkeypatch)
    calls=[]
    def fail(*args):
        calls.append(args)
        raise RuntimeError('terminal admission exception')
    monkeypatch.setattr(cpuwatch.selectorops,'dispatch',fail)
    with pytest.raises(RuntimeError):cpuwatch.admit_cpu()
    assert cpuwatch.admit_cpu()['status']=='ADMITTED_OR_ATTEMPTED_NO_RETRY'
    assert len(calls)==1
