"""Observable WM request queue and exact per-regime/duration random-plan matching."""
import copy
from pathlib import Path
import time
import numpy as np
import torch
from .mppi_search import group_components,sample_unscored
from .protocol import digest
from .storage import read,write
from .exploration import regime
from .warm_health import assert_running


def worker_slots(root):
    admissions=sorted(root.glob('WORKER_ADMISSION_*.json'))
    return read(admissions[-1] if admissions else root/'WORKERS.json')['workers']


def used_tickets(match):
    retired=match.parent.parent/'INCIDENT_RETIRED_JOBS.json'
    excluded=set(read(retired).get('R_ticket_keys',[])) if retired.exists() else set()
    return {read(x)['ticket'] for x in match.glob('*_START.json')
        if x.name.removesuffix('_START.json') not in excluded and
        (not (match/x.name.replace('_START','_END')).exists()
        or read(match/x.name.replace('_START','_END')).get('executed'))}


def guide(job,generator):
    from .warm_history import ObservableHistory
    history=ObservableHistory()
    root=Path(job['queue_root']);bounds=tuple(job['accel_bounds'])
    quota=read(Path(job['random_quota'])) if job.get('random_quota') else None
    def call(rows,commands,now,sample):
        if int(now)%10 or now>900:return None
        from .committed import members as committed_members
        committed=sample.get('committed_plans',[])
        groups=group_components(rows,exclude=committed_members(committed))
        if not groups:return None
        chosen_group=groups[0];members=chosen_group['members']
        traffic=regime(rows);selected_ticket=None
        if job['arm']=='A':
            sample['behaviour']=dict(route='policy',sample_probability=0.,mixture_mass=0.,
                regime=traffic,executed=False,beta=0.,q=0.,eligible_query=True)
            return None
        if job.get('parallel_group_queries') and job.get('quota_aware_admission') and quota is None:
            from .parallel_groups import query
            result=query(job,generator,history,rows,commands,now,sample,groups,traffic,worker_slots(root))
            if result is not None:return result
        reservation=None;admission=None
        if quota is None:
            # Queue occupancy is infrastructure-only, never predicted/SUMO gain.
            pending=sum(not (root/'responses'/p.name).exists() for p in (root/'requests').glob('*.json'))
            workers=worker_slots(root)
            # Availability is observed at this decision, never waited for. The
            # broker resolves simultaneous clients racing for the final slot.
            if job.get('quota_aware_admission'):
                from .quota_admission import choose
                p,queried,admission,reservation=choose(root,job,now,
                    lambda:float(torch.rand((),generator=generator)),pending,len(workers))
            else:p=.95 if pending<len(workers) else 0.
        else:
            import fcntl
            match=root/'random_match'/f'u{job["update"]:03}';match.mkdir(parents=True,exist_ok=True)
            with (match/'MATCH.claim').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                used=used_tickets(match)
                available=[t for t in quota['tickets'] if t['regime']==traffic and t['id'] not in used]
                chosen_group=next((g for g in groups if any(t['group_size']==g['size'] for t in available)),None)
                remaining=chosen_group is not None
                if remaining:members=chosen_group['members']
            p=quota['sampling'][traffic] if remaining else 0.
        if admission is None:queried=float(torch.rand((),generator=generator))<p
        b=dict(route='wm' if queried else 'policy',sample_probability=p,mixture_mass=p,
            regime=traffic,executed=False,beta=0.,q=0.,eligible_query=True)
        sample['behaviour']=b
        if quota is None:
            b['capacity_admission']=admission or dict(pending=pending,slots=len(workers),available=p>0.,
                query_probability=p,nonblocking=True)
        if not queried:return None
        deltas={v:float(a[0]) for v,a in zip(sample['context']['members'],sample['own_action'])}
        r=dict(world_id=job['world']['seed'],time_s=int(now),rows=rows,group=members,
              commands=commands,policy_residuals=deltas,history=history.snapshot(now))
        if committed:r['committed_plans']=copy.deepcopy(committed)
        if job.get('policy_state') is not None:r['policy_state']=job['policy_state']
        if job.get('search_learning') is not None and quota is None:r['search_learning']=job['search_learning']
        r['state_id']=digest(dict(record=r,job=job['id']))
        if quota is None:
            if reservation:
                from .quota_admission import bind
                bind(reservation,r)
            request=root/'requests'/(r['state_id']+'.json');response=root/'responses'/request.name
            from .warm_spool import immutable
            immutable(request,r);start=time.monotonic()
            while not response.exists():
                assert_running(root,job['arm'])
                if time.time()>job['gpu_deadline']:raise ValueError('registered GPU deadline exceeded')
                time.sleep(.5)
            result=read(response)
            if result['record_sha256']!=digest(r):raise ValueError('plan response identity mismatch')
            result['query_wall_s']=time.monotonic()-start
            if reservation:result['quota_reservation']=reservation
        else:
            import fcntl
            match=root/'random_match'/f'u{job["update"]:03}';match.mkdir(parents=True,exist_ok=True)
            with (match/'MATCH.claim').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                used=used_tickets(match)
                selected_ticket=next((t for t in quota['tickets'] if t['regime']==traffic and t['group_size']==len(members) and t['id'] not in used),None)
                if selected_ticket is not None:
                    write(match/(r['state_id']+'_START.json'),dict(ticket=selected_ticket['id']))
            if selected_ticket is None:
                return dict(accepted=False,reason='matched quota exhausted',behaviour=b)
            # Same CURRENT-policy trajectory extraction as B, no candidate scoring.
            # The model is used only for this shared nuisance-centre calculation.
            # Random choice/covariance/library remain independent of WM rankings.
            from .centre_client import trajectory
            centre,centre_receipt=trajectory(root,r,job['gpu_deadline'])
            plan=sample_unscored(chosen_group,centre,bounds,traffic,
                int(torch.randint(2**31,(),generator=generator)),duration_s=selected_ticket['duration_s'])
            result=dict(accepted=True,plan=plan,option=plan['family'],expected_saving=None,
                known_family=False,record_sha256=digest(r),random_match=dict(path=str(match),key=r['state_id']),
                centre_contract='same warm K1/H300 current-policy trajectory as B; zero candidate WM scores',
                centre_receipt=centre_receipt)
        b.update(queried_option=result.get('option'),proposal_accepted=result['accepted'],
            query_wall_s=result.get('query_wall_s',0.))
        if result['accepted']:
            result['history']=r['history']
            b.update(option=result['option'],proposal_members=result['plan']['members'],
                expected_saving=result.get('expected_saving'),predicted_rank=result.get('predicted_rank'),
                known_family=result.get('known_family'),duration_s=result['plan']['duration_s'],
                candidates_searched=result.get('candidates_searched'),iterations=result.get('iterations'))
            b['group_size']=len(result['plan']['members'])
        result['behaviour']=b
        return result
    def completed(proposal,executed):
        if proposal.get('quota_reservation'):
            from .quota_admission import resolved
            resolved(proposal['quota_reservation'],executed)
        m=proposal.get('random_match')
        if m:
            import fcntl
            with (Path(m['path'])/'MATCH.claim').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                write(Path(m['path'])/(m['key']+'_END.json'),dict(executed=executed))
    call.completed=completed
    call.observe=lambda rows,now:history.push(now,rows)
    def prime(frames):
        for frame in frames:history.push(frame['time_s'],frame['rows'],frame['entrance_counts'])
    call.prime=prime
    return call
