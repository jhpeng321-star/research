"""COOPFREE exact-twin plant and unchanged MAPPO safety/ownership executor."""
import gzip
import os
from pathlib import Path
import torch
from .protocol import guard,install_seed_scope,canonical
from .storage import write
from .policy import Policy,Controller


def episode(job):
    from tools.leverprobe import backend as exact
    from tools.coopfree import backend as cf
    path=Path(job['path']);path.mkdir(parents=True,exist_ok=True)
    write(path/'CLAIM.json',dict(pid=os.getpid(),id=job['id'],charged_sim_s=job['sim_s']))
    # Same setup, full-prefix fingerprint and bit-pattern hash as COOPFREE.
    exact.setup(job['runtime'],job['checkpoint'])
    install_seed_scope(); guard(job['world'],job['role'])
    exact.guard=lambda w:guard(w,job['role'])
    p=Policy();p.load_state_dict(job['weights']);p.eval()
    g=torch.Generator().manual_seed(job['noise_seed'])
    controller=Controller(p,g,deterministic=job['deterministic'],
                          samples=job.get('samples'),branch=job.get('branch'))
    b=exact.Backend(path,job['world'],job['sim_s'],job['checkpoint'])
    steps=[];metrics={};prefixes={};horizon_totals={};requests=overrides=0
    branch=job.get('branch'); stop=job.get('stop',1200)
    try:
        with gzip.open(path/'observable.jsonl.gz','xt',encoding='utf-8') as trace:
            for t in range(300,stop,10):
                prefixes[str(t)]=b.fingerprint()
                if branch and t<=branch['time_s'] and prefixes[str(t)]!=branch['prefixes'][str(t)]:
                    raise ValueError('exact full-prefix mismatch before intervention')
                before=b.totals['team_delay_vs']; n=len(controller.events)
                for _ in range(10):
                    audit=cf.tick(b,controller,trace)
                    requests+=len(audit);overrides+=sum(x['clamped'] for x in audit)
                if len(controller.events)>n:
                    event=controller.events[-1]
                    steps.append(dict(event,reward=before-b.totals['team_delay_vs']))
                if t+10 in (900,1200):metrics[str(t+10-300)]=dict(b.totals)
                if branch and t+10-branch['time_s'] in job.get('horizons',[]):
                    horizon_totals[str(t+10-branch['time_s'])]=dict(b.totals)
        result=dict(id=job['id'],world=job['world'],arm=job['arm'],steps=steps,
            metrics=metrics,prefixes=prefixes,final=b.fingerprint(),
            totals=dict(b.totals),horizon_totals=horizon_totals,actual_sim_s=b.actual,charged_sim_s=job['sim_s'],
            requests=requests,overrides=overrides)
        write(path/'RESULT.json',result)
        return result
    finally:
        b.close()
