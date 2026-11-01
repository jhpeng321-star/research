"""FIT-only imagined descendants and batched paired-future GPU teacher.

The existing v3 supervisor is immutable. This is a separately registered
continuation: new states only, inherited exact quality before any scale fitting.
No sampled generator latents are saved or exposed to the labelling/student ABI.
"""
import copy
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import time

from tools.coopbandit.protocol import digest, sha
from tools.wmteach.labels import actions, label, helpful_quality

PARENT=os.environ.get('WMTEACH_PARENT_ROOT','/root/autodl-tmp/wmteach20261101_v3')
BACKEND=os.environ.get('WMTEACH_BACKEND','FULL-v4')
WORLDS=tuple(range(969400,969432))
TIMES=tuple(300+7.5*i for i in range(1,81))
TARGET=5120


def roster_key(record):
    return digest(dict(world=record['world_id'],time=record['time_s'],rows=record['rows']))


def validate_record(r):
    if r['role']!='TRAIN' or r['world_id'] not in WORLDS or r['time_s'] not in TIMES:
        raise ValueError('unregistered source world/clock/role')
    if r['setting'] not in range(4) or not r['rows']:
        raise ValueError('invalid setting/empty imagined state')
    if not all(math.isfinite(float(row[k])) for row in r['rows'] for k in ('s_m','v_mps','length_m')):
        raise ValueError('nonfinite observable state')
    if r['state_id']!=f"w{r['world_id']}_line{r['lineage']}_t{int(r['time_s']*2)}":
        raise ValueError('state identity mismatch')
    if r['lineage'] not in (0,1):
        raise ValueError('unknown lineage')
    if len({v['vid'] for v in r['rows']})!=len(r['rows']):
        raise ValueError('duplicate observable vehicle ID')
    if len(r['backlog'])!=2 or len(r['request_total'])!=2 or any(
            not math.isfinite(float(v)) or v<0 or int(v)!=v for v in r['backlog']+r['request_total']):
        raise ValueError('finite nonnegative integer source counters required')
    return r


def quality_receipt(parent):
    """Read only the already-completed quality estimate, never rerun its futures."""
    rows=json.loads((parent/'QUALITY_RECORDS.json').read_text())
    if len(rows)!=24 or len({(r['state_id'],r['horizon_s']) for r in rows})!=24:
        raise ValueError('complete original 12-state/two-horizon quality required')
    return dict(source_root=str(parent),quality_sha256=sha(parent/'QUALITY.json'),
                records_sha256=sha(parent/'QUALITY_RECORDS.json'),
                helpful=helpful_quality([r for r in rows if r['horizon_s']==300]))


def source_for(rt,record):
    """Observable roster and source backlog only; no generator driver parameters."""
    import torch
    source=copy.deepcopy(next(b for b in rt.batches if b.metadata[0]['seed']==record['world_id']))
    rows=record['rows'];n=len(rows)
    def tensor(values,dtype=torch.float32):return torch.tensor([values],dtype=dtype)
    source.state=dict(s_m=tensor([r['s_m'] for r in rows]),v_mps=tensor([r['v_mps'] for r in rows]),
        lane=tensor([r['lane'] for r in rows],torch.long),present=tensor([True]*n,torch.bool),
        is_cav=tensor([r['is_cav'] for r in rows],torch.bool),length_m=tensor([r['length_m'] for r in rows]),
        a_mps2=tensor([r.get('a_mps2',0.) for r in rows]))
    source.rollout_kwargs={k:torch.zeros((1,n,*v.shape[2:]),dtype=v.dtype)
                           for k,v in source.rollout_kwargs.items()}
    source.metadata=[dict(seed=record['world_id'],t0=record['time_s'],
                          initial_state_sha256=digest(rows))]
    source.entry_queue=source.entry_queue.clone()
    source.entry_queue[0,:2]=tensor(record['backlog'])[0].to(source.entry_queue)
    source.entry_queue[0,2:4]=tensor([n/record['time_s'] for n in record['request_total']])[0].to(source.entry_queue)
    return source


def futures(rt,record,indices=tuple(range(8))):
    """Batch all K futures × available options in ONE physical rollout.

    Virtual row keys independently address futures; each key repeats across all
    actions, so complete hidden/arrival realizations stay paired. At descendants
    use the frozen joint FIT population (no saved-ID lookup, no generator latents)
    and causal observed request totals/exposure/backlog. Cold motion/controller
    history is explicit; this is NOT the native motion-conditioned v4 contract.
    """
    import torch
    import numpy as np
    from wms.physics.batched import road_end_exit_bound
    from wms.control.nativeinit import pair_merge_streams
    from tools.rbint.departure import rollout_after_movement
    from tools.wmteach.runtime import prepare_batch,option_controller
    from tools.wmteach.marginal import plans,bind
    acts=actions(record['rows'],record['setting']);na=len(acts)
    b=prepare_batch(rt,source_for(rt,record));nr=na*len(indices)
    # Input queues are measured at this imagined clock; history is only a prior
    # for entrance placement/mix. Arrival intensity uses actual causal counters.
    q=b.entry_spec['queues'][0]
    q=replace(q,backlog=np.array(record['backlog'],dtype=float),
              demand_rate=np.array(record['request_total'],float)/record['time_s'])
    initial=rt.frozen.expand(b.state,nr)
    pair_merge_streams(initial,torch.tensor([i for i in indices for _ in acts],device='cuda'),8,190)
    initial['exit_s_m']=torch.full_like(initial['s_m'],road_end_exit_bound(rt.model.physics))
    keys=[record['world_id']*100+i for i in indices for _ in acts]
    observation=dict(total=record['request_total'],exposure_s=record['time_s'],
                     final_backlog=record['backlog'])
    kw=rt.frozen.expand(b.rollout_kwargs,nr)
    sampling_seed=190+int(digest(record['state_id'])[:8],16)
    plan=plans(rt,initial,kw,n_samples=1,seed=sampling_seed,row_keys=keys,
        histories=[b.entry_spec['histories'][0]]*nr,queues=[q]*nr,
        t0_s=record['time_s'],observations={str(k):observation for k in keys})[0]
    kw,entries,_=bind(rt,initial,kw,plan)
    cb=option_controller(rt.mp,acts*len(indices),record['rows'])
    with torch.no_grad():
        roll=rollout_after_movement(rt.model,initial,n_steps=600,cav_controller=cb,**kw)
        score=rt.frozen.costs(roll,entries,b.entry_spec['k0'])
        audit=entries.audit_population(initial,roll)
    out={str(h):{a['id']:[] for a in acts} for h in (120,300)}
    for j,k in enumerate(indices):
        draw=digest(dict(state=record['state_id'],future=k,seed=sampling_seed,
                         observation=observation,source=BACKEND+'_population_cold_descendant'))
        for h in (120,300):
            for i,a in enumerate(acts):
                row=j*na+i
                value=score[str(h)]['real_J'][row]
                road=score[str(h)]['real_road'][row];source=score[str(h)]['real_source'][row]
                if not math.isfinite(value) or abs(value-road-source)>1e-6:
                    raise ValueError('nonfinite/source-inclusive team accounting failed')
                out[str(h)][a['id']].append(dict(draw_id=draw,team_delay=value,breakdown=None))
    return dict(futures=out,population_audit=audit,sampling=plan.receipt,
                indices=list(indices),observation=observation,
                contract=BACKEND+' physics/FIT-joint-population/cold-motion-and-controller/causal-source-counters')


def generate(rt,source,lineage):
    """Two independent O0 imagined paths; yield observable descendants only."""
    import torch
    from tools.wmteach.runtime import roster,move,option_controller
    from wms.control.dreamer import with_entry_slots
    from tools.wmteach.marginal import plans,bind
    from tools.rbint.departure import rollout_after_movement
    from wms.physics.batched import road_end_exit_bound
    from wms.control.nativeinit import pair_merge_streams
    seed=source.metadata[0]['seed']
    b=with_entry_slots(copy.deepcopy(source),600,1600,torch.Generator().manual_seed(191000573),queue_state=True)
    b.state,b.rollout_kwargs=move(b.state),move(b.rollout_kwargs)
    current=rt.frozen.expand(b.state,1)
    pair_merge_streams(current,torch.zeros(1,dtype=torch.long,device='cuda'),1,190+lineage)
    current['exit_s_m']=torch.full_like(current['s_m'],road_end_exit_bound(rt.model.physics))
    original=rt.model._wmphys_arrival_fit.artifact['worlds'][str(seed)]
    kw=rt.frozen.expand(b.rollout_kwargs,1)
    plan=plans(rt,current,kw,row_keys=[seed],histories=[b.entry_spec['histories'][0]],
        queues=[b.entry_spec['queues'][0]],observations={str(seed):original},
        t0_s=300.,seed=190,n_samples=2)[lineage]
    kw,entries,receipt=bind(rt,current,kw,plan)
    # No team costs are needed for unlabelled path generation; the raw source
    # process carries queues/ordinals. Avoid retaining an entire600s observer tape.
    kw['entries']=entries.process
    cb=option_controller(rt.mp,[dict(id='O0',available=True)],roster(current))
    for j,t in enumerate(TIMES):
        kw['k0']=j*15
        with torch.no_grad():
            roll=rollout_after_movement(rt.model,current,n_steps=15,cav_controller=cb,**kw)
        current=roll['state']
        kw.update(history=roll['history'],history_mask=roll['history_mask'],
                  driver_params=roll['driver_params_after_insertion'])
        for name in ('driver_history','driver_history_mask','driver_defaults'):kw.pop(name,None)
        rows=roster(current)
        for r in rows:r['a_mps2']=float(current['a_mps2'][0,int(r['vid'])])
        total=(entries.process.due_total[0]+original['total']).tolist()
        record=dict(state_id=f'w{seed}_line{lineage}_t{int(t*2)}',world_id=seed,lineage=lineage,
            role='TRAIN',time_s=t,setting=(j+lineage)%4,rows=rows,
            backlog=entries.process.pending[0].tolist(),request_total=total,
            backend=BACKEND,generator=BACKEND+'-population-marginal-O0',generator_future=lineage,
            parent_state_sha256=source.metadata[0]['initial_state_sha256'])
        yield validate_record(record)


def execute(root):
    import fcntl
    import gc
    import os
    import torch
    from tools.wmteach.run import save,read
    from tools.wmteach.runtime import bootstrap
    root=Path(root);parent=Path(PARENT)
    lock=(root/'WORKER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity=read(root/'IDENTITY.json')
    def guard():
        if (root/'STOP').exists():raise RuntimeError('own STOP; preserve all completed work')
        for p,s in identity['sources'].items():
            if sha(p)!=s:raise ValueError('immutable source changed: '+p)
    guard();os.nice(19)
    # Parent runs quality first. Waiting does not allocate CUDA or duplicate it.
    while not (parent/'QUALITY.json').is_file():
        guard()
        owner=read(parent/'DISPATCH.json');p=Path(f'/proc/{owner["pid"]}/stat')
        st=p.read_text().split() if p.is_file() else None
        if st is None or st[21]!=owner['start_ticks'] or st[2]=='Z':
            raise RuntimeError('initial quality owner ended incomplete; no redispatch')
        time.sleep(30)
    provenance=quality_receipt(parent)
    save(root/'QUALITY.json',read(parent/'QUALITY.json'))
    save(root/'QUALITY_RECORDS.json',read(parent/'QUALITY_RECORDS.json'))
    save(root/'HELPFUL_QUALITY.json',provenance['helpful'])
    save(root/'QUALITY_PROVENANCE.json',provenance)
    print(json.dumps(dict(event='QUALITY_FIRST',**provenance)),flush=True)
    rt=bootstrap()
    save(root/'RUNTIME.json',dict(model=rt.receipt,export=rt.export,starts=rt.provenance))
    def compute(r,canary=False):
        dest=root/'labels'/r['state_id']
        if (dest/'LABEL_H300.json').is_file() and (dest/'LABEL_H120.json').is_file():
            return  # Completed bounded first-state canary is reused exactly once.
        dest.mkdir()
        start=time.monotonic();result=futures(rt,r)
        if canary:
            single=futures(rt,r,(0,))
            worst=max(abs(v[0]['team_delay']-single['futures'][h][a][0]['team_delay'])
                      for h,values in result['futures'].items() for a,v in values.items())
            if worst>1e-3:raise ValueError(f'batched-vs-serial canary mismatch {worst}')
            save(root/'CANARY_PASS.json',dict(max_team_cost_error=worst,paired_complete_futures=8,
                population_audit=result['population_audit'],first_result_reused=True,new_sumo_sim_s=0))
        save(dest/'FUTURES.json',dict(result,wall_s=time.monotonic()-start,identity_sha256=sha(root/'IDENTITY.json')))
        for h in (120,300):save(dest/f'LABEL_H{h}.json',label(dict(r,horizon_s=h),result['futures'][str(h)]))
        gc.collect();torch.cuda.empty_cache()
    for lineage in (0,1):
        for source in rt.batches:
            guard();seed=source.metadata[0]['seed']
            folder=root/'states'/f'w{seed}_line{lineage}';folder.mkdir()
            # A lineage is generated once; its completed observable states persist.
            # No automatic retry of partial/failed lineages or label tasks.
            for record in generate(rt,source,lineage):
                guard();save(folder/(record['state_id']+'.json'),record)
                if not (root/'CANARY_PASS.json').exists():
                    # Qualify batching immediately, before generating the full corpus.
                    compute(record,canary=True)
                    print(json.dumps(dict(event='CANARY_PASS',state=record['state_id'])),flush=True)
            save(folder/'COMPLETE.json',dict(states=80))
            print(json.dumps(dict(event='GENERATED',world=seed,lineage=lineage,states=80)),flush=True)
    paths=sorted((root/'states').glob('*/*_t*.json'))
    records=[read(p) for p in paths]
    if len(records)!=TARGET or len({roster_key(r) for r in records})!=TARGET:
        raise ValueError('5120 distinct observable decision states required before labels')
    save(root/'CORPUS.json',dict(states=TARGET,worlds=list(WORLDS),times=list(TIMES),lineages=2,
        hashes={str(p.relative_to(root)):sha(p) for p in paths},new_sumo_sim_s=0))
    for i,r in enumerate(records):
        guard();start=time.monotonic();compute(r)
        print(json.dumps(dict(event='LABELLED',state=r['state_id'],complete=i+1,target=TARGET,
                              wall_s=time.monotonic()-start)),flush=True)
    from tools.wmteach.student import train_records
    labels=[read(root/'labels'/r['state_id']/'LABEL_H300.json') for r in records]
    train_records(root,root/'student_scale_5120',labels)
    save(root/'COMPLETE.json',dict(states=TARGET,labels=TARGET*2,K=8,new_sumo_sim_s=0,
                                  helpful_quality_pass=provenance['helpful']['pass_']))


if __name__=='__main__':
    import argparse
    import traceback
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    root=parser.parse_args().root
    try:execute(root)
    except BaseException:
        error=traceback.format_exc()
        with (root/'FAILURE.json').open('x') as f:
            json.dump(dict(unix=time.time(),traceback=error,new_sumo_sim_s=0,automatic_retry=False),f)
        raise
