"""Update-boundary learning from same-arm/seed SUMO-positive executed plans."""
import numpy as np
from .mppi_search import CovarianceModel,PlanLibrary,learn,default_sigma


def empty(arm,seed,bounds):
    return dict(arm=arm,seed=seed,covariance=CovarianceModel(default_sigma(bounds)).to_json(),
                library=PlanLibrary().to_json())


def exact_update(previous,receipts,load_episode,arm,seed):
    """Learn each exact-positive, same-arm/seed executed perturbation once."""
    from .exact_grade import verify_sealed
    if arm!='B' or previous['arm']!=arm or previous['seed']!=seed:
        raise ValueError('search learning requires same B arm/training seed')
    cov=CovarianceModel.from_json(previous['covariance']);lib=PlanLibrary.from_json(previous['library'])
    consumed=set(previous.get('exact_grade_ids',[]));entries=[]
    for r in sorted(receipts,key=lambda r:r['grade_id']):
        verify_sealed(r,'receipt_sha256')
        if r['arm']!=arm or r.get('training_seed')!=seed or r['status']!='graded' or r['grade_veh_s']<=0 or r['grade_id'] in consumed:continue
        ep=load_episode(r['result_path'])
        if ep['arm']!=arm:raise ValueError('cross-arm library episode')
        step=next(s for s in ep['steps'] if s['time_s']==r['plan']['start_s'])
        proposal=step['proposal']
        if r['plan']['duration_s']!=proposal['plan']['duration_s']:
            # The unexecuted tail has no exact label. Keep it out of library and
            # covariance rather than labelling the original longer plan positive.
            consumed.add(r['grade_id'])
            entries.append(dict(grade_id=r['grade_id'],learned=False,reason='truncated plan; unexecuted tail has no exact label'))
            continue
        result=learn(proposal['plan'],r['grade_veh_s'],proposal['descriptor'],cov,lib)
        consumed.add(r['grade_id'])
        entries.append(dict(grade_id=r['grade_id'],receipt_sha256=r['receipt_sha256'],exact_gain_veh_s=r['grade_veh_s'],**result))
    return dict(arm=arm,seed=seed,covariance=cov.to_json(),library=lib.to_json(),entries=entries,
        exact_grade_ids=sorted(consumed),favoured_directions={key:cov.favoured_directions(key.split('|')[0],int(key.split('|')[1])) for key in cov.stats})


def update(previous,episodes,records,arm,seed):
    if arm!='B' or previous['arm']!=arm or previous['seed']!=seed:
        raise ValueError('search learning requires same B arm/training seed')
    cov=CovarianceModel.from_json(previous['covariance']);lib=PlanLibrary.from_json(previous['library'])
    entries=[]
    for ep in episodes:
        if ep['arm']!=arm:raise ValueError('cross-arm library input')
        ev={r['time_s']:r for r in records if r['world']==ep['world']['seed']}
        for step in ep['steps']:
            b=step['sample'].get('behaviour',{});proposal=step.get('proposal')
            r=ev.get(step['time_s'])
            if not (b.get('plan_start') and b.get('executed') and proposal and r):continue
            own=[x for x in records if x['regime']==r['regime'] and x['route']=='policy' and x['p']<1]
            if not own:continue
            weights=np.array([1/(1-x['p']) for x in own])
            baseline=float(np.average([x['advantage'] for x in own],weights=weights))
            advantage=r['advantage']-baseline
            result=learn(proposal['plan'],advantage,proposal['descriptor'],cov,lib)
            entries.append(dict(plan=proposal['plan']['id'],world=r['world'],time_s=r['time_s'],
                sumo_critic_residual=r['advantage'],own_residual_baseline=baseline,
                sumo_realized_advantage=advantage,**result))
    return dict(arm=arm,seed=seed,covariance=cov.to_json(),library=lib.to_json(),entries=entries,
        favoured_directions={key:cov.favoured_directions(key.split('|')[0],int(key.split('|')[1]))
            for key in cov.stats})
