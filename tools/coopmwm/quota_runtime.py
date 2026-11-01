"""Boundary-only handover adopting complete u1 results and original Adam/RNG."""
import argparse
import copy
import json
from pathlib import Path
import types
from .protocol import sha


def binding(root):
    root=Path(root);v=json.loads((root/'MAIN0548_RUNTIME_BINDING.json').read_text())
    assert v['amendment_sha256'] in (root/'AUTHORITY.md').read_text()
    assert v['plan_amendment_sha256'] in (root/'AUTHORITY.md').read_text()
    assert v['cpu']['pass_'] and v['cpu']['skips']==v['cpu']['sim_s']==0
    source=Path(v['source_root'])
    assert Path(__file__).resolve()==source/'pytools/tools/coopmwm/quota_runtime.py'
    for name,h in v['cpu']['source_hashes'].items():assert sha(source/name)==h,name
    return v


def install(root):
    from . import warm_runner as wr,accel_storage
    from tools.coopnative import storage
    from .accel_protocol import SUPERSEDED_CHARGE
    from .warm_protocol import batch_worlds
    root=Path(root);v=binding(root)
    previous=json.loads((root/'MAIN0508_REPAIR_BINDING_V2.json').read_text())
    accel_storage._reserve=types.FunctionType(storage.reserve.__code__,
        dict(storage.reserve.__globals__,CAP=previous['budget']['total']-SUPERSEDED_CHARGE,CAPS=previous['budget']['phases']))
    wr.budget=lambda:copy.deepcopy(previous['budget'])
    wr.batch_worlds=lambda u:copy.deepcopy(previous['repair_plan']['worlds']) if u==1 else batch_worlds(u)

    def verify(r,arm):
        from .warm_health import assert_running
        assert_running(r,arm)
        original=json.loads((r/'EXACT_OPERATIONAL_BINDING.json').read_text())
        operational=json.loads((r/'WARM_B_RUNTIME_BINDING_V4.json').read_text())['operational']
        awr=json.loads((r/'AWR_BINDING.json').read_text())['config']
        result=dict(original);result.update(operational)
        result.update(budget=previous['budget'],awr=awr,resume_update=0,
            runtime_receipt_suffix='_Q1',workers=24 if arm=='B' else 4,
            quota_aware_admission=True,precharged_unattempted=[])
        # All existing B-u1 attempts are COMPLETE and will be read, never rerun.
        for w in previous['repair_plan']['worlds']:
            p=r/'jobs'/f'warm_u001_{arm}_{w["seed"]}'
            if arm=='B':assert (p/'RESULT.json').exists(),'wait for current physics once'
            else:assert not (p/'CLAIM.json').exists(),'R already attempted; inspect'
        assert not (r/f'{arm}_UPDATE_001.json').exists(),'optimizer boundary already passed'
        return result
    wr.verify=verify
    return wr,v


def run(root,component):
    root=Path(root)
    if component=='broker':
        binding(root)
        from .capacity_broker import run as broker
        return broker(root,root/'QUOTA_BROKER_CONFIG.json')
    wr,_=install(root)
    return wr.run_arm(root,component)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--component',choices=('B','R','broker'),required=True)
    a=p.parse_args();run(a.root,a.component)
