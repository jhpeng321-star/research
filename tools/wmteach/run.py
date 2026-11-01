"""Exclusive resumable GPU teacher job; a completed task is never repeated."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import time

from tools.coopbandit.protocol import checked, digest, sha
from tools.wmteach.labels import label, compare, quality


def save(path,value):
    with checked(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,sort_keys=True,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())


def read(path): return json.loads(checked(path).read_text())


def execute(root,canary=False,quality_only=False):
    import torch
    root=checked(root)
    assert root.is_dir()
    lock=(root/'WORKER.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.nice(19)
    pid=os.getpid()
    save(root/f'PROCESS_{pid}.json',dict(pid=pid,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        start_ticks=Path(f'/proc/{pid}/stat').read_text().split()[21],canary=canary,started_unix=time.time()))
    identity=read(root/'IDENTITY.json')
    for p,s in identity['sources'].items():
        if sha(p)!=s: raise ValueError('frozen source changed: '+p)
    from tools.wmteach.runtime import bootstrap,fit_futures,roster
    rt=bootstrap()
    if not (root/'RUNTIME.json').exists():
        save(root/'RUNTIME.json',dict(model=rt.receipt,export=rt.export,starts=rt.provenance))
    records=read(root/'QUALITY_INPUTS.json')
    def compute(record,source):
        record=dict(record,backend=identity['backend'])
        path=root/'tasks'/record['state_id']
        path.mkdir(exist_ok=True)
        for future in range(8):
            target=path/f'k{future}.json'
            if target.exists(): continue
            if (root/'STOP').exists(): raise RuntimeError('own cooperative STOP')
            for p,s in identity['sources'].items():
                if sha(p)!=s: raise ValueError('frozen source changed: '+p)
            start=time.monotonic()
            result=fit_futures(rt,source,record,future)
            result.update(identity_sha256=sha(root/'IDENTITY.json'),future=future,
                          state_id=record['state_id'],wall_s=time.monotonic()-start)
            save(target,result)
            print(json.dumps(dict(state=record['state_id'],future=future,wall_s=result['wall_s'])),flush=True)
            gc.collect(); torch.cuda.empty_cache()
            if canary: return None
        labels=[]
        for horizon in (120,300):
            out=path/f'LABEL_H{horizon}.json'
            if not out.exists():
                futures={}
                for k in range(8):
                    task=read(path/f'k{k}.json')
                    if task['identity_sha256']!=sha(root/'IDENTITY.json') or task['future']!=k:
                        raise ValueError('task identity mismatch')
                    for a,f in task['results'][str(horizon)].items(): futures.setdefault(a,[]).append(f)
                save(out,label(dict(record,horizon_s=horizon),futures))
            labels.append(read(out))
        return labels
    # Exact diagnostic targets are opened only after WM predictions have been saved.
    for r in records:
        result=compute(r,None)
        if canary: return
    if not (root/'QUALITY.json').exists():
        exact={(r['state_id'],r['horizon_s']):r for r in read(root/'EXACT_QUALITY_ONLY.json')}
        rows=[]
        for r in records:
            for horizon in (120,300):
                wm=read(root/'tasks'/r['state_id']/f'LABEL_H{horizon}.json')
                rows.append(compare(wm,exact[r['state_id'],horizon]))
        save(root/'QUALITY_RECORDS.json',rows)
        save(root/'QUALITY.json',{str(h):quality([r for r in rows if r['horizon_s']==h],300) for h in (120,300)})
    if quality_only:
        save(root/'QUALITY_COMPLETE.json',dict(states=12,futures=96,new_sumo_sim_s=0,
                                             identity_sha256=sha(root/'IDENTITY.json')))
        return
    # FIT-only sources; NEVER exact EXAM targets or planner DEV outcomes.
    for setting in range(4):
        for i,source in enumerate(rt.batches):
            seed=source.metadata[0]['seed']
            record=dict(state_id=f'w{seed}_t300_setting{setting}',world_id=seed,role='TRAIN',
                        time_s=300,setting=setting,rows=roster(source.state))
            compute(record,source)
        # Complete-world snapshots only; same shared student implementation as S-DISTILL.
        target=root/f'student_setting0_{setting}'
        if not target.exists():
            from tools.wmteach.student import train
            train(root,target,setting)
    save(root/'COMPLETE.json',dict(contexts=128,labels=256,K=8,new_sumo_sim_s=0,
                                  identity_sha256=sha(root/'IDENTITY.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('root',type=Path); parser.add_argument('--canary',action='store_true')
    parser.add_argument('--quality-only',action='store_true')
    args=parser.parse_args(); execute(args.root,args.canary,args.quality_only)
