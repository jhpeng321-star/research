import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('chain',Path(__file__).with_name('clean2_chain.py'))
chain=importlib.util.module_from_spec(spec);spec.loader.exec_module(chain)


@pytest.mark.parametrize('failed',['QUALIFIED','PREPARE','SMOKE','GATE','LAUNCH'])
def test_any_failure_stops_without_retry_or_later_dispatch(failed):
    calls=[];events=[]
    names=['QUALIFIED','PREPARE','SMOKE','GATE','LAUNCH']
    def action(name):
        def run():
            calls.append(name)
            if name==failed:raise ValueError('receipt/environment failure')
            return {'ok':True}
        return run
    with pytest.raises(ValueError):
        chain.sequence([(n,action(n)) for n in names],lambda *a:events.append(a))
    assert calls==names[:names.index(failed)+1]
    assert events[-1][:2]==(failed,'FAILED')
    assert all(calls.count(n)==1 for n in calls)


def test_complete_chain_is_ordered_and_passes_once():
    calls=[];events=[]
    names=['QUALIFIED','PREPARE','SMOKE','GATE','LAUNCH']
    chain.sequence([(n,lambda n=n:calls.append(n)) for n in names],lambda *a:events.append(a))
    assert calls==names
    assert [(e[0],e[1]) for e in events]==[(n,s) for n in names for s in ('RUNNING','PASS')]


def test_status_preserves_events_and_stops_final_root_on_fault(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir();control=tmp_path/'control';control.mkdir()
    root=tmp_path/'clean2';root.mkdir()
    chain.publish(source/'SOURCE_MANIFEST.json',{'commit':'abc','files':{}})
    monkeypatch.setattr(chain,'ROOT',root)
    worker=chain.Chain(source,control)
    worker.status('GATE','RUNNING')
    worker.status('GATE','FAILED',{'error':'incomplete'})
    assert len(list(control.glob('STATUS_*.json')))==2
    assert chain.read(root/'CHAIN_STATUS.json')['status']=='FAILED'
    assert chain.read(root/'STOP.json')['step']=='GATE'
    with pytest.raises(FileExistsError):chain.publish(root/'STOP.json',{})
