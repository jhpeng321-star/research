"""Independent bounded budget replacement tests; no simulator or launcher retry."""
import json
from pathlib import Path

import pytest

from traffic_shepherd import budget_guard as bg


LIMITS=dict(categories={'test':dict(max_allocations=2,max_seconds=20)},total_seconds=20)


def denied(code):
    exc=PermissionError(13,'injected access denial')
    if code is not None:
        exc.winerror=code
    return exc


def ledger(tmp_path):
    path=tmp_path/'budget.json'
    data=dict(schema=bg.LEDGER_SCHEMA,campaigns={},reservations=[])
    path.write_text(json.dumps(data),encoding='utf-8')
    return bg.BudgetGuard(path),path


@pytest.mark.parametrize('code',[5,32,33])
def test_retry_same_payload_under_lock_then_success(tmp_path,monkeypatch,code):
    guard,path=ledger(tmp_path)
    original=bg.os.replace
    attempts=[]
    sleeps=[]
    def replace(src,dst):
        assert Path(str(path)+bg.LOCK_SUFFIX).exists()
        attempts.append((Path(src),Path(dst),Path(src).read_bytes()))
        if len(attempts)==1:
            raise denied(code)
        return original(src,dst)
    monkeypatch.setattr(bg.os,'replace',replace)
    monkeypatch.setattr(bg.time,'sleep',sleeps.append)
    guard.stop_campaign('c','unchanged reason')
    assert len(attempts)==2 and attempts[0]==attempts[1]
    assert sleeps==[.05]
    assert json.loads(path.read_text())['campaigns']['c']['stopped']
    assert not Path(str(path)+bg.LOCK_SUFFIX).exists()


def test_exhaustion_preserves_old_ledger_and_raises_original(tmp_path,monkeypatch):
    guard,path=ledger(tmp_path)
    original=path.read_bytes()
    error=denied(5)
    attempts=[]
    sleeps=[]
    def replace(src,dst):
        attempts.append(Path(src).read_bytes())
        assert path.read_bytes()==original
        assert Path(str(path)+bg.LOCK_SUFFIX).exists()
        raise error
    monkeypatch.setattr(bg.os,'replace',replace)
    monkeypatch.setattr(bg.time,'sleep',sleeps.append)
    with pytest.raises(PermissionError) as got:
        guard.stop_campaign('c','reason')
    assert got.value is error
    assert len(attempts)==5 and len(set(attempts))==1
    assert sleeps==[.05]*4
    assert path.read_bytes()==original and not Path(str(path)+bg.LOCK_SUFFIX).exists()


@pytest.mark.parametrize('error',[denied(None),denied(2),OSError(5,'unrelated')])
def test_unrelated_replace_fault_not_retried(tmp_path,monkeypatch,error):
    guard,path=ledger(tmp_path)
    original=path.read_bytes()
    calls=[]
    def replace(*args):
        calls.append(args)
        raise error
    monkeypatch.setattr(bg.os,'replace',replace)
    monkeypatch.setattr(bg.time,'sleep',lambda _:pytest.fail('unrelated fault slept'))
    with pytest.raises(type(error)) as got:
        guard.stop_campaign('c','reason')
    assert got.value is error and len(calls)==1 and path.read_bytes()==original


def test_temporary_write_failure_not_retried(tmp_path,monkeypatch):
    guard,path=ledger(tmp_path)
    original=Path.write_text
    calls=[]
    error=denied(5)
    def write(p,*args,**kwargs):
        if p==path.with_suffix('.tmp'):
            calls.append(p)
            raise error
        return original(p,*args,**kwargs)
    monkeypatch.setattr(Path,'write_text',write)
    monkeypatch.setattr(bg.os,'replace',lambda *a:pytest.fail('replace after failed temporary write'))
    monkeypatch.setattr(bg.time,'sleep',lambda _:pytest.fail('temporary write retried'))
    with pytest.raises(PermissionError) as got:
        guard.stop_campaign('c','reason')
    assert got.value is error and len(calls)==1


def test_failed_reservation_never_invokes_launcher(tmp_path,monkeypatch):
    guard,path=ledger(tmp_path)
    error=denied(32)
    attempts=[]
    def replace(*args):
        attempts.append(args)
        raise error
    monkeypatch.setattr(bg.os,'replace',replace)
    monkeypatch.setattr(bg.time,'sleep',lambda _:None)
    with pytest.raises(PermissionError):
        guard.reserve(campaign='c',category='test',amount_s=10,target_dir=tmp_path/'run',
            ceilings=LIMITS,launcher=lambda:pytest.fail('launcher invoked before durable charge'))
    assert len(attempts)==5
    assert json.loads(path.read_text())['reservations']==[]


def test_progress_retry_one_launcher_one_charge_and_stop_preserved(tmp_path,monkeypatch):
    guard,path=ledger(tmp_path)
    original=bg.os.replace
    failures=[]
    launched=[]
    def replace(src,dst):
        data=json.loads(Path(src).read_text())
        if data['reservations'] and data['reservations'][0].get('completed_simulated_s')==.5 and not failures:
            failures.append(True)
            raise denied(33)
        return original(src,dst)
    monkeypatch.setattr(bg.os,'replace',replace)
    monkeypatch.setattr(bg.time,'sleep',lambda _:None)
    def launch():
        launched.append(True)
        guard.mark_latest_progress('c','test',.5)
        guard.stop_campaign('c','sticky stop')
        return 'one invocation'
    reservation,result=guard.reserve(campaign='c',category='test',amount_s=10,
        target_dir=tmp_path/'run',ceilings=LIMITS,launcher=launch)
    data=json.loads(path.read_text())
    assert result=='one invocation' and launched==[True] and failures==[True]
    assert len(data['reservations'])==1 and data['reservations'][0]['amount_s']==10
    assert data['reservations'][0]['completed_simulated_s']==.5
    assert data['campaigns']['c']['stopped']
    with pytest.raises(bg.BlockedByStop):
        guard.reserve(campaign='c',category='test',amount_s=10,target_dir=tmp_path/'other',
            ceilings=LIMITS,launcher=lambda:pytest.fail('STOP bypass'))
    assert json.loads(path.read_text())==data
