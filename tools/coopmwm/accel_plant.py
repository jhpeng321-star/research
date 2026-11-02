"""One-second residual episodes using the original COOPFREE exact-twin plant."""
import gzip
import os
from pathlib import Path
import torch
from .accel_protocol import guard,install_seed_scope
from .accel_storage import write
from .acceleration import Policy,Controller


def episode(job):
    if job.get('native_reference'):
        import types
        from . import plant as original
        run=types.FunctionType(original.episode.__code__,dict(original.episode.__globals__,
            guard=guard,install_seed_scope=install_seed_scope))
        return run(job)
    path=Path(job['path']);path.mkdir(parents=True,exist_ok=True)
    write(path/'CLAIM.json',dict(pid=os.getpid(),id=job['id'],charged_sim_s=job['sim_s']))
    from tools.leverprobe import backend as exact
    from tools.coopfree import backend as cf
    exact.setup(job['runtime'],job['checkpoint']);install_seed_scope();guard(job['world'],job['role'])
    exact.guard=lambda w:guard(w,job['role'])
    policy=Policy();policy.load_state_dict(job['weights']);policy.eval()
    generator=torch.Generator().manual_seed(job['noise_seed'])
    from .plan_client import guide
    controller=Controller(policy,generator,job['deterministic'],
        guide=guide(job,generator) if not job['deterministic'] and not job.get('b_replay') else None,beta=job.get('beta',1.),
        replay=job.get('samples'),branch=job.get('branch'))
    backend=exact.Backend
    if job.get('warm_history'):
        from .warm_plant import observed_backend
        backend=observed_backend(backend)
    b=backend(path,job['world'],job['sim_s'],job['checkpoint'])
    if job.get('warm_history') and controller.guide:
        controller.guide.prime([f for f in b.causal_history.snapshot(300) if f['time_s']<300])
    steps=[];metrics={};prefixes={};horizon_totals={};requested=overrides=0
    if job.get('collect'):  # B-REPLAY collector: snapshot + non-blocking request, policy always continues
        from .replay_bank import collector_guide
        controller.guide=collector_guide(job,generator,controller,prefixes)
    if job.get('sumo_verify') and not job.get('b_replay'):  # owner option 1: blocking exact top-k verification (default off)
        from .sumo_verify import install
        install(controller,job,prefixes,generator)
    branch=job.get('branch')
    try:
        with gzip.open(path/'observable.jsonl.gz','xt',encoding='utf-8') as trace:
            for t in range(300,job.get('stop',1200)):
                if job.get('health_guards'):
                    from .warm_health import assert_running
                    assert_running(Path(job['queue_root']),job['arm'])
                prefixes[str(t)]=b.fingerprint()
                if branch and t<=branch['time_s'] and prefixes[str(t)]!=branch['prefixes'][str(t)]:
                    raise ValueError('exact full-prefix mismatch before acceleration intervention')
                before=b.totals['team_delay_vs'];n=len(controller.events)
                audit=cf.tick(b,controller,trace);requested+=len(audit);overrides+=sum(x['clamped'] for x in audit)
                if job.get('health_guards') and job.get('guided'):
                    from .warm_health import live_safety
                    live_safety(Path(job['queue_root']),job['arm'],b.totals,job['world']['seed'],job['safety_reference'])
                if len(controller.events)!=n+1:raise ValueError('expected one acceleration action per1s')
                event=controller.events[-1]
                if job.get('guided') and job['arm']=='B' and event['sample'].get('behaviour',{}).get('plan_start'):
                    # Recorded only AFTER the real SUMO step succeeded. This
                    # controls queue backpressure, not the final closed-job gate.
                    write(Path(job['queue_root'])/'guided_starts'/f'u{job["update"]:03}'/(event['proposal']['record_sha256']+'.json'),
                        dict(job=job['id'],time_s=t,behaviour=event['sample']['behaviour']))
                steps.append(dict(controller.events[-1],reward=before-b.totals['team_delay_vs']))
                if t+1 in (900,1200):metrics[str(t+1-300)]=dict(b.totals)
                if branch and t+1-branch['time_s'] in (60,120,300,600):
                    horizon_totals[str(t+1-branch['time_s'])]=dict(b.totals)
        result=dict(id=job['id'],world=job['world'],arm=job['arm'],steps=steps,metrics=metrics,
            prefixes=prefixes,final=b.fingerprint(),totals=dict(b.totals),horizon_totals=horizon_totals,
            actual_sim_s=b.actual,charged_sim_s=job['sim_s'],requests=requested,overrides=overrides)
        if job.get('sumo_verify'):
            result['sumo_verification_root']=job['sumo_verify']['root']
        write(path/'RESULT.json',result);return result
    finally:b.close()
