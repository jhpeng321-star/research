"""Outcome-blind repair: retain failed attempts, adopt only never-claimed jobs.

No failed traffic episode is replayed. Replacement TRAIN worlds are selected by
same-cell ascending registered seed, preserving the exact 12focus/12other mix.
"""
import argparse
import copy
import json
from pathlib import Path
import types
from .protocol import sha
from .warm_spool import immutable


def replacement_plan(original,train,claimed):
    used={w['seed'] for w in original};result=[];replaced=[];adopt=[]
    for w in original:
        if w['seed'] not in claimed:
            result.append(w);adopt.append(f'warm_u001_B_{w["seed"]}');continue
        options=sorted((v for v in train if v['cell']==w['cell'] and v['seed'] not in used),key=lambda v:v['seed'])
        if not options:raise ValueError('no unattempted same-cell TRAIN replacement')
        v=options[0];used.add(v['seed']);result.append(v)
        replaced.append(dict(failed=w,replacement=v))
    return dict(worlds=sorted(result,key=lambda w:w['seed']),replaced=replaced,precharged_unattempted=adopt,
        additional_sim_s=1200*len(replaced),failed_jobs=[f'warm_u001_B_{w["seed"]}' for w in original if w['seed'] in claimed])


def repair_budget(original,additional):
    b=copy.deepcopy(original);b['phases']['train']+=additional
    b['allocated']+=additional;b['unallocated']-=additional
    if b['unallocated']<0 or b['allocated']>b['total']:raise ValueError('repair exceeds approved cap')
    return b


def binding(root):
    root=Path(root);b=json.loads((root/'MAIN0508_REPAIR_BINDING.json').read_text())
    assert b['amendment_sha256'] in (root/'AUTHORITY.md').read_text()
    assert b['cpu']['pass_'] and b['cpu']['sim_s']==b['cpu']['skips']==0
    source=Path(b['source_root']);assert Path(__file__).resolve()==source/'pytools/tools/coopmwm/capacity_recovery.py'
    for name,h in b['cpu']['source_hashes'].items():assert sha(source/name)==h,name
    return b


def install(root):
    import tools.coopmwm.warm_runner as wr
    from .warm_protocol import batch_worlds
    from . import accel_storage
    from tools.coopnative import storage
    from .accel_protocol import SUPERSEDED_CHARGE
    b=binding(root)
    accel_storage._reserve=types.FunctionType(storage.reserve.__code__,
        dict(storage.reserve.__globals__,CAP=b['budget']['total']-SUPERSEDED_CHARGE,CAPS=b['budget']['phases']))
    wr.budget=lambda:copy.deepcopy(b['budget'])
    wr.batch_worlds=lambda u:copy.deepcopy(b['repair_plan']['worlds']) if u==1 else batch_worlds(u)

    def verify(r,arm):
        from .warm_health import assert_running
        assert_running(r,arm)
        original=json.loads((r/'EXACT_OPERATIONAL_BINDING.json').read_text())
        operational=json.loads((r/'WARM_B_RUNTIME_BINDING_V4.json').read_text())['operational']
        awr=json.loads((r/'AWR_BINDING.json').read_text())['config']
        for w in b['repair_plan']['worlds']:
            ident=f'warm_u001_{arm}_{w["seed"]}'
            assert not (r/'jobs'/ident/'CLAIM.json').exists(),'repair job was already attempted'
        return dict(original,**operational,budget=b['budget'],awr=awr,resume_update=0,
            runtime_receipt_suffix='_V5',workers=24 if arm=='B' else 4,
            precharged_unattempted=b['repair_plan']['precharged_unattempted'] if arm=='B' else [])
    wr.verify=verify
    return wr,b


def run(root,component):
    root=Path(root);wr,b=install(root)
    if component in ('B','R'):return wr.run_arm(root,component)
    if component=='grader':
        from .exact_runtime import supervise
        return supervise(root,4)
    if component=='broker':
        from .capacity_broker import run
        return run(root,root/'CAPACITY_BROKER_CONFIG_V2.json')
    raise ValueError(component)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--component',choices=('B','R','grader','broker'),required=True)
    a=p.parse_args();run(a.root,a.component)
