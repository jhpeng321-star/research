"""Adopt opaque A48 precompute; complete fixed PRIMARY A24/B24/R24 once."""
import copy
from pathlib import Path
import time
from . import final_runtime as f,final_discovery as d
from .accel_storage import read,write
from .protocol import sha


def ready(root,arms=('A','B','R')):
    """A48 completion certifies its registered A24 checkpoint exists; never select."""
    for arm in arms:
        p=root/f'WARM_{arm}_PAUSED.json'
        if not p.exists() or not (root/f'{arm}_checkpoint_024.pt').is_file():return False
        u=read(p)['update']
        if (arm=='A' and u!=48) or (arm!='A' and u!=24):
            raise ValueError('fixed primary endpoint certificate mismatch')
    return True


def configure(root,config):
    from .warm_protocol import install_accounting
    primary=copy.deepcopy(config['budget']);primary['updates']=dict(A=24,B=24,R=24)
    f.ID_ALIASES={'A':'A24v2'}
    f.budget=lambda:copy.deepcopy(primary)
    f.endpoints_ready=ready
    d.budget=lambda:copy.deepcopy(primary)
    d.install_accounting=lambda:install_accounting(root,budget_override=config['budget'])
    install_accounting(root,budget_override=config['budget'])


def secondary(root,b,plan):
    from .accel_runner import panel_report
    from .branch_eval import analyze,window
    eps=[]
    for w in f.worlds('final_test'):
        for a in ('mappo','native','A','B','R'):
            result=read(root/'jobs'/f'fixedfinal_{a}_{w["seed"]}'/'RESULT.json')
            eps.append({k:result[k] for k in ('id','world','arm','metrics')})
    report=panel_report(root,'FINAL_SECONDARY_A48_EPISODES.json',
        [e for e in eps if e['arm'] in ('A','B','R')],[e for e in eps if e['arm'] in ('mappo','native')],'final_test')
    write(root/'FINAL_SECONDARY_A48_LABEL.json',dict(checkpoints=dict(A=48,B=24,R=24),
        role='secondary longer-training A48; primary A24 results separate'))
    records=[]
    for w in f.worlds('final_test'):
        ref=read(root/'jobs'/f'fixedfinal_mappo_{w["seed"]}'/'RESULT.json')
        for t in f.TIMES:
            for a in ('MAPPO','A','B'):
                ep=ref if a=='MAPPO' else read(root/'jobs'/f'fixedbranch_{a}_{w["seed"]}_t{t}'/'RESULT.json')
                m=window(ep,t,300)
                records.append(dict(seed=w['seed'],branch_time_s=t,arm=a,J=m['team_delay_vs'],window=m))
    write(root/'FINAL_SECONDARY_A48_BRANCHES.json',dict(checkpoints=dict(A=48,B=24),analysis=analyze(records,plan)))


def adopt_completed_stage(root,items):
    """Validate opaque completion metadata only; never read result values."""
    for spec in items:
        meta=read(root/'final_receipts'/(spec['id']+'.json'))
        assert meta['id']==spec['id'] and meta['outcomes_closed']
        assert meta['checkpoint_update']==spec['checkpoint_update']==24
        assert meta['checkpoint_sha256']==sha(root/'A_checkpoint_024.pt')
        for name,info in meta['files'].items():
            assert name in ('RESULT.json','RESULT.json.gz','observable.jsonl.gz','CLAIM.json','CLOSE.json')
            assert sha(root/'jobs'/spec['id']/name)==info['sha256']
    return len(items)


def run(root,config):
    import fcntl
    import os
    root=Path(root);lock=(root/'FINAL_PRECOMPUTE.claim').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    suffix=config.get('receipt_suffix','')
    assert suffix in ('','_V2','_RESOURCE1651'),'unregistered final attempt suffix'
    started=root/('FINAL_ENDPOINT_STARTED'+suffix+'.json')
    assert not started.exists(),'adopt existing owner; no duplicate final dispatch'
    assert config['amendment_sha256'] in (root/'AUTHORITY.md').read_text()
    for n,h in config['source_hashes'].items():assert sha(Path(config['source_root'])/n)==h
    complete=read(root/'FINAL_PRECOMPUTE_COMPLETE.json')
    assert complete['outcomes_closed'] and complete['completed_jobs']==673
    assert read(root/'FINAL_MAPPO_REUSE_PASS.json')['pass_']
    assert not (root/'FINAL_OUTCOMES_OPENED.json').exists()
    configure(root,config)
    b=dict(read(root/'WARM_BINDING.json'),budget=config['budget'])
    from .resource_config import clocks
    b=clocks(root,b)
    write(started,dict(unix=time.time(),primary=dict(A=24,B=24,R=24),
        secondary=dict(A=48),adopted_opaque_jobs=673,outcomes_closed=True))
    try:
        # Frozen A24 only, lowest priority; no old A48/MAPPO/native rerun.
        a_items=f.specs(('A',))+f.specs(('A',),branches=True)
        if suffix=='_RESOURCE1651':assert adopt_completed_stage(root,a_items)==480
        else:f.run_stage(root,b,a_items,config['workers'])
        while not ready(root):
            if f.blocked(root):raise ValueError('durable final STOP')
            time.sleep(30)
        f.run_stage(root,b,f.specs(('B','R')),config['workers'])
        f.run_stage(root,b,f.specs(('B',),branches=True),config['workers'])
        f.analyze_complete(root,b,config['branch_plan'])
        secondary(root,b,config['branch_plan'])
        # Same previously qualified departed-member replay executor/detectors.
        d.run(root,dict(config,source_hashes=config['source_hashes']))
    except BaseException as error:
        write(root/('FINAL_ENDPOINT_FAILED'+suffix+'.json'),dict(unix=time.time(),error_type=type(error).__name__,
            physics_retry=False,outcomes_closed=not (root/'FINAL_OUTCOMES_OPENED.json').exists()))
        raise
    finally:lock.close()


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('config',type=Path)
    a=p.parse_args();run(a.root,read(a.config))
