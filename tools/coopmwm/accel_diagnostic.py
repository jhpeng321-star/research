"""Executed-plan E2: exact member reversion and conditional CRN noise, never reward."""
import copy
import numpy as np
from .protocol import digest
from .credit import paired,noise,wilson


def evaluate(rt,record,*,speed2=False,memory=None,transfers=False):
    from .plan_wm import evaluate as futures
    plan=record['plan'];members=plan['members']
    acts=[]
    for name,revert in [('joint',[]),*[(f'revert:{i}',[v]) for i,v in enumerate(members)],('all',members)]:
        keep=[i for i,v in enumerate(members) if v not in revert]
        # The whole observable background follows MAPPO in all four arms.
        # Reversion does not decode as native/IDM, and the commitment ends once.
        p=dict(plan,id=name,members=[members[i] for i in keep],knots=[plan['knots'][i] for i in keep])
        acts.append(p)
    r=dict(record,policy_residuals={})
    result=futures(rt,r,acts,8,600,chunk_actions=4,trim=False,pad_actions=True,horizons=(60,120,300,600),speed2=speed2,memory=memory,transfers=transfers)
    credits={};variances={}
    for h,fs in result.items():
        draws=[{a:values[k] for a,values in fs.items()} for k in range(8)]
        credits[h]=paired(draws);variances[h]=noise(draws)
    return dict(record_sha256=digest(record),credits=credits,noise=variances,futures=result,
        observation_contract='observable roster/cold MAPPO memory/uniform32FIT source prior; same MAPPO background for all arms')


def agreement(records,draws=10000):
    """Cluster whole worlds, preserving multiple sampled windows of each world."""
    worlds=sorted({r['world']['seed'] for r in records})
    if not worlds:return dict(windows=0,worlds=0)
    rng=np.random.default_rng(20261101)
    indices=rng.integers(len(worlds),size=(draws,len(worlds)))
    clusters={w:[r for r in records if r['world']['seed']==w] for w in worlds}
    def fit(rs,h):
        x=np.array([v for r in rs for v in (*r['wm'][h]['D_i'],r['wm'][h]['C_G'])])
        y=np.array([v for r in rs for v in (*r['exact'][h]['D_i'],r['exact'][h]['C_G'])])
        return max(0.,float(x@y/(x@x))) if x@x else 0.
    checks={}
    for h in ('60','120','300','600'):
        result={}
        for kind in ('D_i','C_G'):
            def counts(rs):
                ps=[(a,b) for r in rs for a,b in (zip(r['wm'][h][kind],r['exact'][h][kind]) if kind=='D_i' else [(r['wm'][h][kind],r['exact'][h][kind])])]
                nonzero=[(a,b) for a,b in ps if a!=0 and b!=0]
                return sum((a>0)==(b>0) for a,b in nonzero),len(nonzero),len(ps)-len(nonzero)
            hits,n,ties=counts(records)
            boot=[]
            for ix in indices:
                a,b,_=counts([r for i in ix for r in clusters[worlds[int(i)]]]);boot.append(a/b if b else 0.)
            result[kind]=dict(sign_agreement=hits/n if n else None,n=n,ties=ties,
                ci95=np.quantile(boot,[.025,.975]).tolist() if len(worlds)>1 else [0.,1.],wilson_ci95=wilson(hits,n))
        samples=[fit([r for i in ix for r in clusters[worlds[int(i)]]],h) for ix in indices]
        result.update(scale=fit(records,h),scale_ci95=np.quantile(samples,[.025,.975]).tolist(),
            pass_=len(worlds)>1 and all(result[k]['ci95'][0]>.5 for k in ('D_i','C_G')))
        checks[h]=result
    eligible={h:min(v[k]['sign_agreement'] for k in ('D_i','C_G')) for h,v in checks.items() if v['pass_']}
    return dict(windows=len(records),worlds=len(worlds),horizons=checks,
        weights={h:v/sum(eligible.values()) for h,v in eligible.items()},
        primary_horizon_s=300 if '300' in eligible else None,diagnostic_only=True,training_gate=False,
        bootstrap='10000 whole-world clusters; seed20261101; ties separate; one-world CI uninformative')
