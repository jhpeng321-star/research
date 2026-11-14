"""Exact checkpoint-boundary continuation for the compute-only group fan-out."""
import argparse
import copy
import json
from pathlib import Path
import types
import numpy as np
from .protocol import sha


def restore_progress(root,arm,resume,worlds,seed):
    """Reconstruct diagnostic-only history/RNG without requeueing old diagnostics.

    Policy/Adam/torch RNG come from the original checkpoint in warm_runner.
    Trust and library are the exact immutable previous-update receipts.
    """
    from .accel_learner import evidence
    from .accel_storage import read
    root=Path(root);running=[];rng=np.random.default_rng(seed);count=0
    for u in range(1,resume+1):
        episodes=[read(root/'jobs'/f'warm_u{u:03}_{arm}_{w["seed"]}'/'RESULT.json') for w in worlds(u)]
        episodes.sort(key=lambda e:(e['world']['seed'],e['arm']))
        running.extend(evidence(episodes))
        if arm=='B':
            candidates=[s for e in episodes for s in e['steps'] if s['time_s']<=600 and s['sample'].get('behaviour',{}).get('plan_start')]
            count+=len(rng.permutation(len(candidates))[:max(0,12-count)])
    update=json.loads((root/f'{arm}_UPDATE_{resume:03}.json').read_text())
    learned=json.loads((root/f'B_SEARCH_LEARNING_{resume:03}.json').read_text()) if arm=='B' else None
    return dict(running=running,beta=update['trust'],learned=learned,
        exact_rng_state=copy.deepcopy(rng.bit_generator.state),exact_count=count)


def busy_report(root,update,begin,end,slots):
    """Per-slot union of completed inference intervals; pending work is censored."""
    root=Path(root);intervals={s:[] for s in slots};pending=0
    for p in (root/'quota_requests'/f'u{update:03}').glob('*_REQUEST.json'):
        v=json.loads(p.read_text())
        for record in v.get('records',[v]):
            completed=root/'broker_completed'/(record['state_id']+'.json')
            if not completed.exists():pending+=1;continue
            c=json.loads(completed.read_text());slot=c['slot'];stop=min(end,c['unix'])
            start=max(begin,c['unix']-c['inference_wall_s'])
            if stop>start:intervals.setdefault(slot,[]).append((start,stop))
    totals={}
    for slot,parts in intervals.items():
        merged=[]
        for start,stop in sorted(parts):
            if merged and start<=merged[-1][1]:merged[-1]=(merged[-1][0],max(stop,merged[-1][1]))
            else:merged.append((start,stop))
        seconds=sum(stop-start for start,stop in merged)
        totals[slot]=dict(completed_inference_s=seconds,busy_fraction=seconds/(end-begin) if end>begin else None)
    return dict(begin_unix=begin,end_unix=end,wall_s=end-begin,slots=totals,pending_queries=pending,
        basis='completed B-update reserved-query inference intervals, clipped/unioned per slot; pending/nonquota/other-update work censored; not nvidia-smi utilisation')


def binding(root):
    root=Path(root);v=json.loads((root/'TAIL_RUNTIME_BINDING.json').read_text())
    assert v['amendment_sha256'] in (root/'AUTHORITY.md').read_text()
    assert v['cpu']['pass_'] and v['cpu']['sim_s']==v['cpu']['skips']==0
    source=Path(v['source_root'])
    assert Path(__file__).resolve()==source/'pytools/tools/coopmwm/tail_runtime.py'
    for name,h in v['cpu']['source_hashes'].items():assert sha(source/name)==h,name
    return v


def run(root):
    from . import warm_runner as wr,accel_storage
    from .accel_protocol import SUPERSEDED_CHARGE
    from .warm_protocol import batch_worlds
    from tools.coopnative import storage
    from .warm_health import assert_running
    root=Path(root);v=binding(root)
    previous=json.loads((root/'MAIN0508_REPAIR_BINDING_V2.json').read_text())
    accel_storage._reserve=types.FunctionType(storage.reserve.__code__,
        dict(storage.reserve.__globals__,CAP=previous['budget']['total']-SUPERSEDED_CHARGE,CAPS=previous['budget']['phases']))
    wr.budget=lambda:copy.deepcopy(previous['budget'])
    wr.batch_worlds=lambda u:copy.deepcopy(previous['repair_plan']['worlds']) if u==1 else batch_worlds(u)
    def verify(r,arm):
        assert arm=='B';assert_running(r,arm)
        original=json.loads((r/'EXACT_OPERATIONAL_BINDING.json').read_text())
        operational=json.loads((r/'WARM_B_RUNTIME_BINDING_V4.json').read_text())['operational']
        result=dict(original);result.update(operational)
        result.update(budget=previous['budget'],awr=json.loads((r/'AWR_BINDING.json').read_text())['config'],
            resume_update=v['resume_update'],runtime_receipt_suffix='_T1',workers=24,
            quota_aware_admission=True,parallel_group_queries=True,precharged_unattempted=[])
        # Recovered wave may already be complete, but must never be partially rerun.
        u=v['adopt_update']
        for w in wr.batch_worlds(u):assert (r/'jobs'/f'warm_u{u:03}_B_{w["seed"]}'/'RESULT.json').exists(),'adopt completed results only'
        assert not (r/f'B_UPDATE_{u:03}.json').exists(),'optimizer already applied; inspect before launch'
        result['resume_progress']=restore_progress(r,arm,v['resume_update'],wr.batch_worlds,wr.SEED)
        return result
    wr.verify=verify
    wr.run_arm(root,'B')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)
