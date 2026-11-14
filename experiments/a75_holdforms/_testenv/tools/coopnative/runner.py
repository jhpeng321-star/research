"""Approval-bound parallel native PPO. No launch path exists without MAIN approval."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import re
from pathlib import Path
import time
import torch
from .protocol import (approval_line, sha, worlds, batch_worlds, UPDATES, CHECK_EVERY,
                       REGISTRY, CAP, CAPS, budget)
from .storage import read, write, reserve, ledger, reconciled
from .policy import Policy
from .learner import save, load, update


def verify(root, registry):
    line = approval_line(Path(registry).read_text(encoding='utf-8'))
    cap = re.search(r'cap\s*([\d,]+)',line,re.I)
    if cap and int(cap[1].replace(',','')) < CAP:
        raise ValueError('approval cap below registered design; amend prospectively')
    binding = read(root/'BINDING.json')
    if binding['approval_line'] != line or binding['budget'] != budget():
        raise ValueError('approval/budget binding mismatch')
    if not binding['cpu']['pass_'] or binding['cpu']['sim_s'] != 0 or binding['cpu']['skips']:
        raise ValueError('CPU gate absent')
    if not binding['exposure']['pass_']:
        raise ValueError('exposure audit absent')
    for filename, expected in binding['sources'].items():
        if sha(root/'pytools'/filename) != expected:
            raise ValueError('source binding mismatch')
    if sha(root/'COOPNATIVE_DESIGN.md') != binding['design_sha256']:
        raise ValueError('design mismatch')
    if sha(root/'MAPPO_DEV.json') != binding['baseline_sha256']:
        raise ValueError('baseline cache binding mismatch')
    if (root/'STOP.json').exists():
        raise ValueError('durable STOP')
    return binding


def admission(binding):
    import shutil
    while True:
        if os.getloadavg()[0] <= binding['load_max'] and shutil.disk_usage('.').free >= 2*1024**3:
            mem = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
            if int(mem['MemAvailable'].split()[0])*1024 >= 8*1024**3:
                return
        time.sleep(30)


def wave(root,binding,policy,world_list,arm,role,ident,phase):
    jobs, results = [], []
    weights = {k:v.detach().cpu() for k,v in policy.state_dict().items()}
    for w in world_list:
        job_id = f'{ident}_{arm}_{w["seed"]}'
        path = root/'jobs'/job_id
        result = path/'RESULT.json'
        if result.exists():
            results.append(read(result))
        else:
            jobs.append(dict(id=job_id,path=str(path),world=w,role=role,arm=arm,weights=weights,
                             approval_line=binding['approval_line'],
                             runtime=binding['runtime'],comparators=binding['comparators'],
                             noise_seed=20261101+w['seed']+sum(map(ord,ident))))
    if jobs:
        reserve(root,jobs,phase)
        admission(binding)
        from .plant import episode
        # Standard context manager waits for owned in-flight workers; no process-name signals.
        with ProcessPoolExecutor(max_workers=binding['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
            results.extend(pool.map(episode,jobs))
    return sorted(results,key=lambda r:r['world']['seed'])


def run(root,registry):
    binding = verify(root,registry)  # BEFORE any simulator import/construction.
    if (root/'TRAIN_COMPLETE.json').exists():
        return read(root/'TRAIN_COMPLETE.json')
    os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    if binding['workers'] not in (1,2) or binding['load_max'] > 12:
        raise ValueError('initial host allocation: cpuQ<=2 workers/load<=12')
    root.mkdir(parents=True,exist_ok=True)
    lock = root/'SUPERVISOR.claim'
    import fcntl
    claim = lock.open('a')
    fcntl.flock(claim,fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(1)
    try:
        charged = reconciled(root)
        checkpoints = sorted(root.glob('checkpoint_*.pt'))
        if checkpoints:
            policy,optimizer,generator,payload = load(checkpoints[-1])
            start = max(1,payload['update'])
        else:
            torch.manual_seed(20261101)
            policy = Policy()
            optimizer = torch.optim.Adam(policy.parameters(),lr=3e-4)
            generator = torch.Generator().manual_seed(20261101)
            start = 1
            save(root/'checkpoint_000.pt',policy,optimizer,generator,0,charged)
        controls = []
        for arm in ('native','ctrl0'):
            controls += wave(root,binding,Policy(),worlds('dev'),arm,'dev','control','dev')
        zero = Policy()
        canary = wave(root,binding,zero,worlds('dev')[:1],'learned','dev','zero_canary','mechanics')[0]
        native = next(r for r in controls if r['arm']=='native' and r['world']==canary['world'])
        if canary['physical_sha256'] != native['physical_sha256'] or canary['metrics'] != native['metrics']:
            raise ValueError('zero-action physical canary differs from native')
        check_controls = []
        for arm in ('native','mappo'):
            check_controls += wave(root,binding,zero,worlds('check'),arm,'check','control','checks')
        baselines = read(root/'MAPPO_DEV.json')['records']
        from .evaluation import paired, behaviour
        for u in range(start,UPDATES+1):
            checkpoint = root/f'checkpoint_{u:03}.pt'
            if not checkpoint.exists():
                episodes = wave(root,binding,policy,batch_worlds(u),'train','train',f'u{u:03}','train')
                metrics = update(policy,optimizer,episodes,generator)
                charged = sum(r['sim_s'] for r in ledger(root))
                save(checkpoint,policy,optimizer,generator,u,charged,ppo=metrics)
            if not (root/f'UPDATE_{u:03}.json').exists():
                payload = torch.load(checkpoint,map_location='cpu',weights_only=False)
                write(root/f'UPDATE_{u:03}.json',dict(update=u,total_charged_sim_s=payload['charged'],
                      training_sim_s=u*28800,ppo=payload['extra']['ppo']))
            if u % CHECK_EVERY == 0:
                learned = wave(root,binding,policy,worlds('check'),'learned','check',f'check{u:03}','checks')
                if not (root/f'CHECK_{u:03}.json').exists():
                    write(root/f'CHECK_{u:03}.json',dict(update=u,total_charged_sim_s=sum(r['sim_s'] for r in ledger(root)),
                      native=paired(learned+check_controls,'native','check'),
                      mappo=paired(learned+check_controls,'mappo','check'),
                      behaviour=[behaviour(r,root/'jobs'/r['id']/'observable.jsonl.gz') for r in learned]))
            if u in (8,UPDATES):
                learned = wave(root,binding,policy,worlds('dev'),'learned','dev',f'dev{u:03}','dev')
                if not (root/f'DEV_{u:03}.json').exists():
                    write(root/f'DEV_{u:03}.json',dict(update=u,comparisons={arm:paired(learned+controls+baselines,arm) for arm in ('native','ctrl0','mappo')}))
        from .discovery import discover
        discover(root,binding,policy,learned)
        write(root/'TRAIN_COMPLETE.json',dict(updates=UPDATES,charged=reconciled(root),cap=CAP,phases=CAPS,
              discovery='complete; inspect DISCOVERY.json for exact group coverage'))
    except BaseException as exc:
        if not (root/'STOP.json').exists():
            write(root/'STOP.json',dict(error=repr(exc),charged=sum(r['sim_s'] for r in ledger(root))))
        raise
    finally:
        claim.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--registry',default=REGISTRY)
    args = parser.parse_args()
    run(args.root,args.registry)
