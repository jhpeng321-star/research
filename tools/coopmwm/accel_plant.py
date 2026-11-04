"""One-second residual episodes using the original COOPFREE exact-twin plant."""
import gzip
import os
from pathlib import Path
import torch
from .accel_protocol import guard,install_seed_scope
from .accel_storage import write
from .acceleration import Policy,Controller


def episode(job, *, controller_hook=None, policy_loader=None):
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
    v2v = bool(job.get('v2v')) or os.environ.get('C4NM_V2V') == '1'
    if policy_loader is not None:
        policy=policy_loader(job['weights'])
    elif v2v:
        from .v2v_runtime import from_weights
        policy=from_weights(job['weights'], enabled=True)
    else:
        policy=Policy();policy.load_state_dict(job['weights'])
    policy.eval()
    generator=torch.Generator().manual_seed(job['noise_seed'])
    from .plan_client import guide
    controller_type=Controller
    if job.get('v2v'):
        from .v2v_runtime import controller_class
        controller_type=controller_class(controller_type)
    if job.get('practice'):
        # practice_starts (owner 2026-11-04 proposal, default absent = untouched code path):
        # restart from a bank example's decision snapshot through the REGISTERED VerifyController
        # ('policy continues': prefix replay, snapshot t0 sample, CRN generator restore, committed
        # snapshot plans continue) with the CURRENT training policy weights the job already
        # carries; v2v jobs get the same v2v observation wrapper as every other v2v controller.
        from .sumo_verify import verify_controller_class as _practice_base
        _cls = _practice_base()
        if job.get('v2v'):
            from .v2v_runtime import controller_class as _practice_v2v_class
            _cls = _practice_v2v_class(_cls)
        _snapshot = job['practice']['snapshot']
        _plan = job['practice'].get('plan')
        def controller_type(*args, **kwargs):
            return _cls(args[0], args[1], _snapshot, _plan)
    controller=controller_type(policy,generator,job['deterministic'],
        guide=guide(job,generator) if not job['deterministic'] and not job.get('b_replay') else None,beta=job.get('beta',1.),
        replay=job.get('samples'),branch=job.get('branch'))
    steps=[];metrics={};prefixes={};horizon_totals={};requested=overrides=0
    if controller_hook is not None:
        controller_hook(controller, job, prefixes)
    if job.get('ls_collect'):
        from .ls_collection import install
        install(controller,job,prefixes,generator)
    if job.get('collect'):
        from .replay_bank import collector_guide
        controller.guide=collector_guide(job,generator,controller,prefixes)
    backend=exact.Backend
    if job.get('warm_history'):
        from .warm_plant import observed_backend
        backend=observed_backend(backend)
    b=backend(path,job['world'],job['sim_s'],job['checkpoint'])
    if job.get('warm_history') and controller.guide:
        controller.guide.prime([f for f in b.causal_history.snapshot(300) if f['time_s']<300])
    if job.get('sumo_verify') and not job.get('b_replay'):  # owner option 1: blocking exact top-k verification (default off)
        from .sumo_verify import install
        install(controller,job,prefixes,generator)
    branch=job.get('branch')
    try:
        with gzip.open(path/'observable.jsonl.gz','xt',encoding='utf-8') as trace:
            if v2v and hasattr(controller,'observe_issued'):
                from .v2v_runtime import ObservedTrace
                trace=ObservedTrace(trace,controller)
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
                if (job.get('guided') and job['arm']=='B' and event['sample'].get('behaviour',{}).get('plan_start')
                        and event.get('proposal') is not None):
                    # Replayed snapshot ticks (practice/replay) record plan_start behaviour with
                    # proposal None - there is no new proposal record to queue; only a real
                    # executed proposal controls backpressure. Recorded only AFTER the real SUMO
                    # step succeeded.
                    write(Path(job['queue_root'])/'guided_starts'/f'u{job["update"]:03}'/(event['proposal']['record_sha256']+'.json'),
                        dict(job=job['id'],time_s=t,behaviour=event['sample']['behaviour']))
                steps.append(dict(controller.events[-1],reward=before-b.totals['team_delay_vs']))
                if t+1 in (900,1200):metrics[str(t+1-300)]=dict(b.totals)
                if branch and t+1-branch['time_s'] in (60,120,300,600):
                    horizon_totals[str(t+1-branch['time_s'])]=dict(b.totals)
        result=dict(id=job['id'],world=job['world'],arm=job['arm'],steps=steps,metrics=metrics,
            prefixes=prefixes,final=b.fingerprint(),totals=dict(b.totals),horizon_totals=horizon_totals,
            actual_sim_s=b.actual,charged_sim_s=job['sim_s'],requests=requested,overrides=overrides)
        if job.get('practice'):
            # practice receipt flag (never the snapshot itself)
            result['practice']={k:v for k,v in job['practice'].items() if k!='snapshot'}
        if job.get('sumo_verify'):
            result['sumo_verification_root']=job['sumo_verify']['root']
        write(path/'RESULT.json',result);return result
    finally:b.close()
