"""Observable WM request queue and exact per-regime/duration random-plan matching."""
import copy
from pathlib import Path
import time
import numpy as np
import torch
from .plans import group,family_seeds,population
from .protocol import digest
from .storage import read,write
from .exploration import regime


def worker_slots(root):
    admissions=sorted(root.glob('WORKER_ADMISSION_*.json'))
    return read(admissions[-1] if admissions else root/'WORKERS.json')['workers']


def used_tickets(match):
    return {read(x)['ticket'] for x in match.glob('*_START.json')
        if not (match/x.name.replace('_START','_END')).exists()
        or read(match/x.name.replace('_START','_END')).get('executed')}


def guide(job,generator):
    root=Path(job['queue_root']);bounds=tuple(job['accel_bounds'])
    quota=read(Path(job['random_quota'])) if job.get('random_quota') else None
    def call(rows,commands,now,sample):
        if int(now)%10 or now>1140:return None
        members=group(rows)
        if len(members)!=2:return None
        from tools.wmteach.labels import actions
        if not any(a['id']!='O0' and set(a['members'])<=set(members) for setting in range(4) for a in actions(rows,setting)):
            return None
        traffic=regime(rows);selected_ticket=None
        if job['arm']=='A':
            sample['behaviour']=dict(route='policy',sample_probability=0.,mixture_mass=0.,
                regime=traffic,executed=False,beta=0.,q=0.,eligible_query=True)
            return None
        if quota is None:
            # Queue occupancy is infrastructure-only, never predicted/SUMO gain.
            while True:
                pending=sum(not (root/'responses'/p.name).exists() for p in (root/'requests').glob('*.json'))
                workers=worker_slots(root)
                if pending<2*len(workers):break
                starts=root/'guided_starts'/f'u{job["update"]:03}'
                if sum(1 for _ in starts.glob('*.json'))>=job['N_exec']:break
                if (root/'STOP.json').exists():raise ValueError('STOP while waiting for GPU slot')
                if time.time()>job['gpu_deadline']:raise ValueError('registered GPU deadline exceeded')
                time.sleep(.5)
            p=.95 if pending<2*len(workers) else 0.
        else:
            import fcntl
            match=root/'random_match'/f'u{job["update"]:03}';match.mkdir(parents=True,exist_ok=True)
            with (match/'MATCH.claim').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                used=used_tickets(match)
                remaining=any(t['regime']==traffic and t['id'] not in used for t in quota['tickets'])
            p=quota['sampling'][traffic] if remaining else 0.
        queried=float(torch.rand((),generator=generator))<p
        b=dict(route='wm' if queried else 'policy',sample_probability=p,mixture_mass=p,
            regime=traffic,executed=False,beta=0.,q=0.,eligible_query=True)
        sample['behaviour']=b
        if not queried:return None
        deltas={v:float(a[0]) for v,a in zip(sample['context']['members'],sample['own_action'])}
        r=dict(world_id=job['world']['seed'],time_s=int(now),rows=rows,group=members,
            commands=commands,policy_residuals=deltas)
        r['state_id']=digest(dict(record=r,job=job['id']))
        if quota is None:
            request=root/'requests'/(r['state_id']+'.json');response=root/'responses'/request.name
            write(request,r);start=time.monotonic()
            while not response.exists():
                if (root/'STOP.json').exists():raise ValueError('STOP during plan imagination')
                if time.time()>job['gpu_deadline']:raise ValueError('registered GPU deadline exceeded')
                time.sleep(.5)
            result=read(response)
            if result['record_sha256']!=digest(r):raise ValueError('plan response identity mismatch')
            result['query_wall_s']=time.monotonic()-start
        else:
            import fcntl
            match=root/'random_match'/f'u{job["update"]:03}';match.mkdir(parents=True,exist_ok=True)
            with (match/'MATCH.claim').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                used=used_tickets(match)
                selected_ticket=next((t for t in quota['tickets'] if t['regime']==traffic and t['id'] not in used),None)
                if selected_ticket is not None:
                    write(match/(r['state_id']+'_START.json'),dict(ticket=selected_ticket['id']))
            if selected_ticket is None:
                return dict(accepted=False,reason='matched quota exhausted',behaviour=b)
            seeds=family_seeds(rows,members,commands,bounds)
            rng=np.random.default_rng(int(torch.randint(2**31,(),generator=generator)))
            plan=population(r,rng,bounds,seeds,1)[0]
            plan['duration_s']=selected_ticket['duration_s']
            result=dict(accepted=True,plan=plan,option=plan['family'],expected_saving=None,
                known_family=False,record_sha256=digest(r),random_match=dict(path=str(match),key=r['state_id']))
        b.update(queried_option=result.get('option'),proposal_accepted=result['accepted'],
            query_wall_s=result.get('query_wall_s',0.))
        if result['accepted']:
            b.update(option=result['option'],proposal_members=result['plan']['members'],
                expected_saving=result.get('expected_saving'),predicted_rank=result.get('predicted_rank'),
                known_family=result.get('known_family'),duration_s=result['plan']['duration_s'],
                candidates_searched=result.get('candidates_searched'),iterations=result.get('iterations'))
        result['behaviour']=b
        return result
    def completed(proposal,executed):
        m=proposal.get('random_match')
        if m:
            import fcntl
            with (Path(m['path'])/'MATCH.claim').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                write(Path(m['path'])/(m['key']+'_END.json'),dict(executed=executed))
    call.completed=completed
    return call
