"""Update-boundary learning from same-arm/seed SUMO-positive executed plans."""
import numpy as np
from .mppi_search import CovarianceModel,PlanLibrary,learn,default_sigma


def empty(arm,seed,bounds):
    return dict(arm=arm,seed=seed,covariance=CovarianceModel(default_sigma(bounds)).to_json(),
                library=PlanLibrary().to_json())


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
