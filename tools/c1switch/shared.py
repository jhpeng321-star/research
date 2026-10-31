"""Shared-owner-GPU recovery: one allocation for BOTH frozen model adapters.

No per-model timer or quota. Old physics, forecasts, parity gates and abandoned
charges remain immutable. Scheduling alone changes; the scientific episode and
full-rerun implementation are reused verbatim.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from . import resume
from .storage import canonical, exclusive, write

CPU = '/home/ubuntu/c1switch20261031/shared_v10'
GPU = '/root/c1switch20261031/shared_v10'
TOTAL = 499680
DROPPED = tuple(range(969118,969128))


def configure():
    resume.ROOT = Path(CPU)
    resume.TOTAL = TOTAL
    resume.DROPPED = DROPPED


def physics_done(path):
    if not (path/'RESULT.json').exists() and not (path/'PHYSICS.json').exists():
        return False
    close = json.loads((path/'CLOSE.json').read_text())
    if close != dict(actual_sim_s=900.,charged_sim_s=900.):
        raise ValueError('completed physics accounting mismatch')
    return True


def supervisor():
    configure()
    root = Path(CPU);phase=root/'recovery'
    with exclusive(phase):
        if (phase/'START.json').exists() or (root/'STOP').exists():
            raise ValueError('existing shared recovery dispatch or STOP')
        write(phase/'START.json',dict(pid=os.getpid(),unix=time.time()))
        episodes = [(a,w) for a,w in resume.tasks() if not physics_done(root/'jobs'/f'{a}_s{w["seed"]}_episode')]
        priority = {'wm':0,'bb':1}
        episodes.sort(key=lambda t:(priority.get(t[0],2),t[1]['seed'],t[0]))
        active,done,queued = {},[],set()
        branches = []
        def enqueue_branches():
            for p in sorted((root/'specs').glob('*.json')):
                spec=json.loads(p.read_text())
                ep=root/'jobs'/f'{spec["arm"]}_s{spec["world"]["seed"]}_episode'
                if physics_done(ep) and p.stem not in queued and not (root/'jobs'/p.stem/'RESULT.json').exists():
                    branches.append((p.stem,['branch','--spec',str(p)],None))
                    queued.add(p.stem)
        enqueue_branches()
        while episodes or branches or active:
            if (root/'STOP').exists():
                return
            while len(active)<14:
                wm_active=sum(x['arm']=='wm' for x in active.values())
                ix=next((i for i,(a,w) in enumerate(episodes) if a!='wm' or wm_active<2),None)
                if ix is not None and (episodes[ix][0] in ('wm','bb') or not branches):
                    arm,w=episodes.pop(ix);ident=f'{arm}_s{w["seed"]}_episode'
                    args=['episode','--arm',arm,'--seed',str(w['seed'])]
                elif branches:
                    ident,args,arm=branches.pop(0)
                elif ix is not None:
                    arm,w=episodes.pop(ix);ident=f'{arm}_s{w["seed"]}_episode'
                    args=['episode','--arm',arm,'--seed',str(w['seed'])]
                else:
                    break
                with (phase/(ident+'.log')).open('x') as log:
                    p=subprocess.Popen([sys.executable,'-B','-u','-m','tools.c1switch.shared']+args,stdout=log,stderr=subprocess.STDOUT)
                active[ident]=dict(process=p,arm=arm)
                write(phase/('OWN_'+ident+'.json'),dict(pid=p.pid,id=ident,arm=arm))
            for ident,row in list(active.items()):
                code=row['process'].poll()
                if code is None:
                    continue
                if code:
                    write(root/'STOP',dict(reason='shared recovery child failed',id=ident,code=code))
                    return
                done.append(ident);del active[ident]
                enqueue_branches()
            for arm in ('wm','bb'):
                if all((root/'jobs'/f'{arm}_s{w["seed"]}_episode'/'pick_j9.json').exists() for w in resume.worlds('main')):
                    if not (root/'rpc'/arm/'COMPLETE').exists():
                        write(root/'rpc'/arm/'COMPLETE',dict(all160_picks=True,unix=time.time()))
            write(phase/'STATUS.json',dict(pending_episodes=len(episodes),pending_branches=len(branches),done_count=len(done),
                active={k:dict(pid=v['process'].pid,arm=v['arm']) for k,v in active.items()}))
            time.sleep(.5)
        for arm,world in resume.tasks():
            resume.finalize(arm,world)
        write(phase/'COMPLETE.json',dict(unix=time.time(),episodes=118,branches=480))


def serve():
    """Single process, single wall-time debit/alarm over BB+WM together."""
    from .adapter import create
    from .core import medians
    from .design import sha
    root=Path(GPU);allocation=json.loads((root/'SHARED_BUDGET.json').read_text())
    cap=allocation['cap_s'];prior=allocation['prior_gpu_s']
    if cap<=0 or cap+prior>14400:
        raise ValueError('shared owner GPU cap')
    began=time.monotonic()
    with exclusive(root/'allocation'):
        if (root/'GPU_STARTED.json').exists():
            raise ValueError('shared allocation already started')
        write(root/'GPU_STARTED.json',dict(pid=os.getpid(),unix=time.time(),cap_s=cap,prior_gpu_s=prior))
        def expired(*args):
            raise TimeoutError('combined owner GPU budget exhausted')
        signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,cap)
        try:
            bindings={a:json.loads((root/(a+'_BINDING.json')).read_text()) for a in ('bb','wm')}
            models={}
            for a,binding in bindings.items():
                if sha(Path(__file__).with_name('adapter.py'))!=binding['adapter_sha256']:
                    raise ValueError('frozen adapter changed')
                models[a]=create(binding,device='cuda')
                old=json.loads((root/(a+'_ATTESTATION.json')).read_text())
                got=models[a].attestation
                for n,h in old['sources'].items():
                    if got['sources'].get(n)!=h:raise ValueError('frozen dependency changed: '+n)
                for k in ('checkpoint_sha256','wm_config_sha256','candidate_sha256','mappo_sha256'):
                    if got[k]!=old[k]:raise ValueError('frozen inference binding changed: '+k)
            write(root/'READY.json',dict(pid=os.getpid(),unix=time.time(),original_host_parity_reused=True,
                attestations={a:m.attestation for a,m in models.items()}))
            while not all((root/'rpc'/a/'COMPLETE').exists() for a in ('bb','wm')):
                if (root/'STOP').exists() or any((root/'rpc'/a/'STOP').exists() for a in ('bb','wm')):
                    raise ValueError('shared GPU STOP')
                for a in ('bb','wm'):
                    # Process at most one/model each round; BB and WM share time,
                    # not overlapping independent idle wall-clock allocations.
                    pending=[p for p in sorted((root/'rpc'/a).glob('*.request.json')) if not p.with_name(p.name.replace('.request.','.result.')).exists()]
                    if not pending:continue
                    p=pending[0];req=json.loads(p.read_text());key=req.pop('request_sha256')
                    if canonical(req)!=key or req['binding']!=bindings[a] or 'validation_seed' in req:
                        raise ValueError('shared RPC identity mismatch')
                    samples=models[a].predict(req['belief'],K=8,horizon_s=60)
                    medians(samples)
                    write(p.with_name(p.name.replace('.request.','.result.')),dict(request_sha256=key,binding=bindings[a],samples=samples,
                        elapsed_gpu_s=time.monotonic()-began))
                time.sleep(.1)
        except BaseException as exc:
            for a in ('wm','bb'):write(root/'rpc'/a/'STOP',dict(reason=str(exc),pid=os.getpid()))
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            elapsed=time.monotonic()-began
            write(root/'GPU_FINISHED.json',dict(pid=os.getpid(),elapsed_s=elapsed,cumulative_gpu_s=prior+elapsed,unix=time.time()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('supervisor','episode','branch','gpu'))
    parser.add_argument('--arm');parser.add_argument('--seed',type=int);parser.add_argument('--spec')
    args=parser.parse_args()
    if args.mode=='gpu':serve();return
    configure()
    if args.mode=='supervisor':supervisor();return
    # Reuse the same fail-closed CPU job implementation, only its accounting
    # constants/root and prospectively registered random subset are rebound.
    sys.argv=[sys.argv[0],args.mode]+(['--spec',args.spec] if args.mode=='branch' else ['--arm',args.arm,'--seed',str(args.seed)])
    resume.main()


if __name__=='__main__':main()
