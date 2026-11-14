"""MAIN02:49 all-active-plan recheck on one shared K8 CRN batch."""
import copy
import numpy as np
from .protocol import digest


def recheck(record,result,score):
    active=record.get('committed_plans',[])
    if not result['accepted'] or not active:return result
    plan=result['plan'];H=result['final']['H']
    # ALL active groups interact by registration: no behavioural distance cutoff.
    keep=dict(plan,id='overlap_keep_all',committed_plans=copy.deepcopy(active))
    drop=dict(id='overlap_drop_new',members=[],residuals=record['policy_residuals'],
        window=600,committed_plans=copy.deepcopy(active))
    actions=[keep,drop];ended=[None,None]
    for i,c in enumerate(active):
        actions.append(dict(plan,id=f'overlap_end_{i}',
            committed_plans=copy.deepcopy(active[:i]+active[i+1:])))
        ended.append(c['plan']['id'])
    values=score(record,actions,8,H)
    costs=np.asarray([[f['team_delay'] for f in values[a['id']]] for a in actions],float)
    if costs.shape!=(len(actions),8) or not np.isfinite(costs).all():
        raise ValueError('overlap recheck incomplete K8 futures')
    rng=np.random.default_rng(int(digest(dict(state=record['state_id'],overlap=True))[:15],16))
    draws=rng.integers(0,8,size=8);means=costs[:,draws].mean(axis=1)
    chosen=int(np.argmin(means));out=dict(result)
    out['overlap_recheck']=dict(rule='all active groups',K=8,H=H,
        active_plan_ids=[c['plan']['id'] for c in active],bootstrap_indices=draws.tolist(),
        options=[dict(id=a['id'],mean_cost=float(costs[i].mean()),sampled_cost=float(means[i])) for i,a in enumerate(actions)],
        chosen=actions[chosen]['id'],truncated_plan_ids=[] if ended[chosen] is None else [ended[chosen]],
        predicted_gain_vs_keep_all=float(means[0]-means[chosen]))
    if chosen==1:
        out.update(accepted=False,reason='overlap recheck keeps existing plans; drops new plan')
    return out
