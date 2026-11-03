"""MAIN02:01: own-action PPO, SUMO-positive completed-plan AWR, all-step critic."""
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


def own_step(sample):
    return not sample.get('behaviour',{}).get('executed',False)


def ratio(logp,sample):
    if not own_step(sample):raise ValueError('no policy ratio exists for an executed plan step')
    return (logp-sample['logp']).exp()


def awr_config(controls):
    """Freeze temperature from existing MAPPO TRAIN-check 300s delay intervals."""
    delays=[float(e['metrics']['900']['team_delay_vs'])-float(e['metrics']['600']['team_delay_vs']) for e in controls]
    if not delays or not np.isfinite(delays).all():raise ValueError('invalid MAPPO AWR scale data')
    temperature=float(np.sqrt(np.mean(np.square(delays))))/UNIT
    if temperature<=0:raise ValueError('MAPPO AWR temperature must be positive')
    return dict(temperature=temperature,weight_clip=20.,coefficient=1.,return_unit_veh_s=UNIT,
        scale_source='RMS source-inclusive MAPPO TRAIN-check J900-J600; frozen before B/R outcomes',
        worlds=[e['world']['seed'] for e in controls])


def awr_weight(advantage,config):
    if not math.isfinite(advantage):raise ValueError('nonfinite plan advantage')
    if advantage<=0:return 0.
    temperature=config['temperature'];clip=config['weight_clip']
    if temperature<=0 or clip<1:raise ValueError('invalid registered AWR settings')
    return math.exp(min(advantage/temperature,math.log(clip)))


def plan_targets(episode,config):
    """Only completed commitments with frozen rollout values and SUMO returns.

    No WM score, beta, or unexecuted sample becomes a target. Plans extending
    past the observed episode are excluded, not truncated wins.
    """
    steps=episode['steps'];targets={}
    for i,s in enumerate(steps):
        b=s['sample'].get('behaviour',{})
        if not (b.get('executed') and b.get('plan_start')):continue
        duration=int(s.get('proposal',{}).get('plan',{}).get('duration_s',b.get('duration_s',0)))
        if duration<=0:raise ValueError('executed plan duration missing')
        end=i+duration
        if end>len(steps):continue
        segment=steps[i:end]
        if any(x['time_s']!=s['time_s']+j or not x['sample'].get('behaviour',{}).get('executed') or
            x['sample']['behaviour'].get('plan_id')!=b['plan_id'] for j,x in enumerate(segment)):
            raise ValueError('incomplete committed plan tape')
        advantage=sum(GAMMA**j*x['reward']/UNIT for j,x in enumerate(segment))
        advantage+=GAMMA**duration*(steps[end]['sample']['value'] if end<len(steps) else 0.)
        advantage-=s['sample']['value']
        weight=awr_weight(advantage,config)
        for j in range(i,end):targets[j]=dict(advantage=advantage,weight=weight,plan_id=b['plan_id'])
    return targets


def imitation(policy,sample,weight=0.):
    b=sample.get('behaviour',{});mean,_=policy(sample['context'])
    if not b.get('executed') or weight<=0:return mean.sum()*0
    indices=[sample['context']['members'].index(v) for v in b['proposal_members'] if v in sample['context']['members']]
    if not indices:return mean.sum()*0
    action=torch.tensor(sample['action']).reshape(-1,1)
    return -weight*gaussian_log_prob(action[indices],mean[indices],RHO,policy.log_std.exp())/len(indices)


def update(policy,optimizer,episodes,generator,epochs=4,awr=None):
    if awr is None and any(s['sample'].get('behaviour',{}).get('executed') for ep in episodes for s in ep['steps']):
        raise ValueError('executed plans require pre-registered AWR scale')
    records=[]
    for ep in episodes:
        targets=plan_targets(ep,awr) if awr is not None else {}
        records.extend((s,a,r,targets.get(i,{}).get('weight',0.)) for i,(s,(a,r)) in enumerate(zip(ep['steps'],advantages(ep['steps']))))
    adv=torch.tensor([a for _,a,_,_ in records]);own=torch.tensor([own_step(s['sample']) for s,_,_,_ in records])
    if own.any():adv=(adv-adv[own].mean())/adv[own].std(unbiased=False).clamp_min(1e-8)
    from .v2v_intent import ppo_setup
    intent=ppo_setup(policy,episodes)  # None unless an IntentPolicy (v2v_intent ON)
    metrics=[]
    for _ in range(epochs):
        for idx in torch.randperm(len(records),generator=generator).split(128):
            samples=[records[int(i)][0]['sample'] for i in idx]
            values=torch.stack([policy(s['context'])[1] for s in samples])
            own_indices=[j for j,s in enumerate(samples) if own_step(s)]
            scores=[policy.score(samples[j])[0] for j in own_indices]
            if scores:
                logps=torch.stack(scores);ratios=torch.stack([ratio(lp,samples[j]) for lp,j in zip(logps,own_indices)])
                a=adv[idx[own_indices]];actor=-torch.minimum(ratios*a,ratios.clamp(.8,1.2)*a).mean()
                delta=logps-torch.tensor([samples[j]['logp'] for j in own_indices])
                kl=float((delta.exp()-1-delta).mean().detach())
            else:actor=values.sum()*0;kl=0.
            returns=torch.tensor([records[int(i)][2] for i in idx]);old=torch.tensor([s['value'] for s in samples])
            clipped=old+(values-old).clamp(-.2,.2)
            vloss=torch.maximum((values-returns).square(),(clipped-returns).square()).mean()
            weights=[records[int(i)][3] for i in idx]
            auxiliary=torch.stack([imitation(policy,s,w) for s,w in zip(samples,weights)]).mean()
            auxiliary=auxiliary*(awr['coefficient'] if awr is not None else 1.)
            loss=actor+.5*vloss+auxiliary
            if intent is not None:loss=loss+intent.term(idx)
            if not bool(torch.isfinite(loss)):raise ValueError('nonfinite acceleration PPO')
            optimizer.zero_grad();loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),.5,error_if_nonfinite=True);optimizer.step()
            metrics.append(dict(loss=float(loss.detach()),kl=kl,gradient_norm=float(norm),imitation=float(auxiliary.detach()),
                own_policy_steps=len(own_indices),critic_steps=len(samples),awr_positive_steps=sum(w>0 for w in weights),
                loss_rule='MAIN0201 own PPO; positive completed-plan AWR; all-step critic'))
            if kl>.045:return metrics if intent is None else intent.close(metrics)
    return metrics if intent is None else intent.close(metrics)


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
    def snr(xs):
        values=[r['advantage'] for r in xs]
        variance=float(np.var(values,ddof=1)) if len(values)>1 else None
        mean=float(np.mean(values)) if values else None
        return dict(windows=len(values),mean_advantage_veh_s=mean,variance_veh_s_squared=variance,
            snr_squared=mean*mean/variance if variance else None)
    return dict(overall=pooled,by_rank=by_rank,
        cooperative_move_noise=dict(eligible_group=snr(records),executed_plan=snr(predictions),
            horizon_s=300,scope='empirical SUMO H300 critic residual; eligible RSU groups not certified cooperation'),
        new_plans=sum(r.get('known_family') is False for r in predictions),
        inherited_plans=sum(r.get('known_family') is True for r in predictions),
        predicted_vs_realized=[{k:r.get(k) for k in ('world','option','rank','expected_saving','advantage')} for r in predictions],
        interpretation='SUMO H300 critic-baselined selected-plan vs own actions; selection conditional, not causal same-state twins')
