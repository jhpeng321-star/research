"""One-second SUMO-only PPO with logged AR state and singular plan-route correction."""
import math
import numpy as np
import torch
import types
from tools.coopfree.policy import gaussian_log_prob
from .acceleration import RHO
from .acceleration import Policy
from tools.coopnative import learner as original
save=types.FunctionType(original.save.__code__,dict(original.save.__globals__,Policy=Policy))
load=types.FunctionType(original.load.__code__,dict(original.load.__globals__,Policy=Policy))

UNIT=10000.
GAMMA=math.exp(-1/600)
LAMBDA=math.exp(-1/1200)


def advantages(steps):
    carry=0.;out=[]
    for i in reversed(range(len(steps))):
        s=steps[i];v=s['sample']['value']
        nxt=steps[i+1]['sample']['value'] if i+1<len(steps) else 0.
        delta=s['reward']/UNIT+GAMMA*nxt-v
        carry=delta+GAMMA*LAMBDA*carry
        out.append((carry,carry+v))
    return list(reversed(out))


def ratio(logp,sample):
    b=sample.get('behaviour',{})
    # The plan/WM query route is an atom in the augmented behaviour measure.
    # Its target policy-route mass is exactly zero, never a fictitious density.
    if b.get('route') in ('wm','plan'):return logp*0
    p=b.get('sample_probability',0.)
    if not 0<=p<1:raise ValueError('own-route mass must remain positive')
    return (logp-sample['logp']-math.log1p(-p)).exp()


def imitation(policy,sample):
    b=sample.get('behaviour',{});mean,_=policy(sample['context'])
    if not b.get('executed'):return mean.sum()*0
    indices=[sample['context']['members'].index(v) for v in b['proposal_members'] if v in sample['context']['members']]
    if not indices:return mean.sum()*0
    action=torch.tensor(sample['action']).reshape(-1,1)
    return -b['beta']*gaussian_log_prob(action[indices],mean[indices],RHO,policy.log_std.exp())/len(indices)


def update(policy,optimizer,episodes,generator,epochs=4):
    records=[(s,a,r) for ep in episodes for s,(a,r) in zip(ep['steps'],advantages(ep['steps']))]
    adv=torch.tensor([a for _,a,_ in records]);adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-8)
    metrics=[]
    for _ in range(epochs):
        for idx in torch.randperm(len(records),generator=generator).split(128):
            samples=[records[int(i)][0]['sample'] for i in idx]
            scores=[policy.score(s) for s in samples];logps,values=map(torch.stack,zip(*scores))
            ratios=torch.stack([ratio(lp,s) for lp,s in zip(logps,samples)])
            mask=torch.tensor([s.get('behaviour',{}).get('route') not in ('wm','plan') for s in samples])
            a=adv[idx];actor=-(torch.minimum(ratios*a,ratios.clamp(.8,1.2)*a)*mask).mean()
            returns=torch.tensor([records[int(i)][2] for i in idx]);old=torch.tensor([s['value'] for s in samples])
            clipped=old+(values-old).clamp(-.2,.2)
            vloss=torch.maximum((values-returns).square(),(clipped-returns).square()).mean()
            auxiliary=torch.stack([imitation(policy,s) for s in samples]).mean()
            loss=actor+.5*vloss+auxiliary
            if not bool(torch.isfinite(loss)):raise ValueError('nonfinite acceleration PPO')
            optimizer.zero_grad();loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),.5,error_if_nonfinite=True);optimizer.step()
            delta=logps-torch.tensor([s['logp'] for s in samples])
            kl=float((((delta.exp()-1-delta)*mask).sum()/mask.sum().clamp_min(1)).detach())
            metrics.append(dict(loss=float(loss.detach()),kl=kl,gradient_norm=float(norm),imitation=float(auxiliary.detach())))
            if kl>.045:return metrics
    return metrics


def evidence(episodes):
    out=[]
    for ep in episodes:
        steps=ep['steps']
        for i,s in enumerate(steps):
            if i+300>len(steps):continue
            b=s['sample'].get('behaviour',{})
            if b.get('executed') and not b.get('plan_start'):continue
            # Compare at the same ex-ante eligible observable decision rule.
            if not b:continue
            value=sum(x['reward'] for x in steps[i:i+300])
            value+=(steps[i+300]['sample']['value'] if i+300<len(steps) else 0)*UNIT
            value-=s['sample']['value']*UNIT
            out.append(dict(world=ep['world']['seed'],cell=ep['world']['cell'],
                route=b['route'],executed=b.get('executed',False),q=b.get('q',0.),
                p=b.get('sample_probability',0.),regime=b['regime'],advantage=value,
                option=b.get('option'),time_s=s['time_s'],rank=b.get('predicted_rank'),
                known_family=b.get('known_family'),expected_saving=b.get('expected_saving')))
    return out


def viability(records):
    """Cluster bootstrap descriptive selected-plan versus own-action critic residuals."""
    from .exploration import trust
    pooled=trust(1.,records)
    by_rank={}
    for rank in sorted({r['rank'] for r in records if r.get('rank') is not None}):
        subset=[r for r in records if r['route']=='policy' or r.get('rank')==rank]
        by_rank[str(rank)]=trust(1.,subset)
    predictions=[r for r in records if r['executed']]
    return dict(overall=pooled,by_rank=by_rank,
        new_plans=sum(r.get('known_family') is False for r in predictions),
        inherited_plans=sum(r.get('known_family') is True for r in predictions),
        predicted_vs_realized=[{k:r.get(k) for k in ('world','option','rank','expected_saving','advantage')} for r in predictions],
        interpretation='SUMO H300 critic-baselined selected-plan vs own actions; selection conditional, not causal same-state twins')
