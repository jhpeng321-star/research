"""MAIN09:36 full-bank post-PPO AWR passes, bounded by the pre-update policy KL.

IMITFIX (MAIN, default OFF): imitation_trust='own_pass' gives imitation its OWN trust
region anchored at the policy as it was right after PPO (pre-imitation snapshot), so
PPO's KL spend never consumes the imitation budget. The FIRST pass over the bank is
then guaranteed complete: pairs are ordered cooperative-first and the per-step learning
rate is halved (with full parameter/optimizer rollback of the rejected trial step)
whenever the projected post-step full-bank KL would exceed the limit, instead of
stopping. Additional passes beyond the first may still early-stop on KL. With the
option off, behaviour is byte-identical to the registered MAIN09:36 rule.
"""
import copy
import math
import torch
from .acceleration import PHI,RHO
from .accel_learner import imitation


def indices(sample):
    members=sample['context']['members']
    return [members.index(v) for v in sample.get('behaviour',{}).get('proposal_members',[]) if v in members]


@torch.no_grad()
def action_error(policy,pairs):
    total=0.;count=0
    for sample,weight in pairs:
        if weight<=0:continue
        js=indices(sample)
        if not js:continue
        mean,_=policy(sample['context']);target=torch.as_tensor(sample['action']).reshape(-1,1)
        total+=float((mean[js]-target[js]).abs().sum());count+=len(js)
    return total/count if count else None


@torch.no_grad()
def joint_kl(policy,reference,samples):
    """Exact Gaussian KL(reference || policy), same AR1/shared covariance as PPO."""
    values=[]
    for sample in samples:
        mean,_=policy(sample['context']);old,_=reference(sample['context']);n=mean.numel()
        if not n:continue
        var=policy.log_std.exp().square()*(1-PHI**2)
        oldvar=reference.log_std.exp().square()*(1-PHI**2)
        d=(old-mean).reshape(-1)
        quadratic=(d.square().sum()-RHO/(1-RHO+n*RHO)*d.sum().square())/(1-RHO)
        value=.5*(n*(oldvar/var-1+torch.log(var/oldvar))+quadratic/var)
        value=float(value)
        if not math.isfinite(value):raise ValueError('nonfinite full-bank KL')
        values.append(max(0.,value))
    return sum(values)/len(values) if values else 0.


def _cooperative_split(pairs,cooperative):
    """Cooperative-first ordering plus the per-group pair lists (option order preserved)."""
    flags=[bool(f) for f in cooperative]
    if len(flags)!=len(pairs):raise ValueError('cooperative flags must align with the positive-weight bank pairs')
    coop=[p for f,p in zip(flags,pairs) if f];other=[p for f,p in zip(flags,pairs) if not f]
    return flags,coop,other


@torch.no_grad()
def _projected_step(policy,optimizer,reference,samples,limit,lr_scale,base_lr):
    """Commit the pending gradient step at the largest trial scale whose projected
    post-step full-bank KL stays within the limit; roll back parameters AND optimizer
    state and halve the scale otherwise. Returns (committed scale, projected KL, trials)."""
    groups=optimizer.param_groups
    params=[p for g in groups for p in g['params']]
    saved=[p.detach().clone() for p in params]
    state=copy.deepcopy(optimizer.state_dict())
    trials=0
    while True:
        trials+=1
        for g,lr in zip(groups,base_lr):g['lr']=lr*lr_scale
        optimizer.step()
        kl=joint_kl(policy,reference,samples)
        if kl<=limit:return lr_scale,kl,trials
        with torch.no_grad():
            for p,s in zip(params,saved):p.copy_(s)
        optimizer.load_state_dict(state)
        lr_scale=lr_scale/2.


def run(policy,optimizer,pairs,generator,reference,coefficient=1.,passes=5,
        imitation_trust=None,cooperative=None):
    if int(passes)!=passes or passes<5:raise ValueError('MAIN09:36 requires at least five passes')
    if imitation_trust not in (None,'own_pass'):raise ValueError('unknown imitation_trust option; expected None or own_pass')
    pairs=[(s,float(w)) for s,w in pairs if w>0]
    before=action_error(policy,pairs)
    flags=None
    if imitation_trust=='own_pass' and cooperative is not None:
        flags,_,_=_cooperative_split(pairs,cooperative)
    report=dict(requested_epochs=int(passes),epochs_completed=0,epochs_started=0,
        pairs_total=len(pairs),pairs_seen=0,member_actions_seen=0,optimizer_steps=0,
        critic_steps=0,own_policy_steps=0,plan_ratios_computed=0,
        action_error_before_mps2=before,action_error_before_update_mps2=action_error(reference,pairs),
        kl_limit=.045,kl_reference='frozen pre-update policy',last_loss=0.,last_gradient_norm=0.,
        rule='all bank pairs; inherited AWR x beta and cooperative oversampling unchanged')
    samples=[s for s,_ in pairs if indices(s)]
    if not samples:
        report.update(stop_reason='empty bank or no present member targets',kl=0.,action_error_after_mps2=before)
        return report
    if imitation_trust=='own_pass':
        return _own_pass(policy,optimizer,pairs,generator,coefficient,passes,flags,reference,samples,before,report)
    stop=False;kl=joint_kl(policy,reference,samples)
    if kl>.045:stop=True;report['stop_reason']='pre-imitation KL already exceeds existing bound'
    for epoch in range(int(passes)):
        if stop:break
        report['epochs_started']=epoch+1
        for ids in torch.randperm(len(pairs),generator=generator).split(128):
            batch=[pairs[int(i)] for i in ids];report['pairs_seen']+=len(batch)
            active=[(s,w) for s,w in batch if indices(s)]
            report['member_actions_seen']+=sum(len(indices(s)) for s,_ in active)
            if not active:continue
            loss=torch.stack([imitation(policy,s,w) for s,w in active]).mean()*coefficient
            if not bool(torch.isfinite(loss)):raise ValueError('nonfinite full-bank imitation')
            optimizer.zero_grad();loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),.5,error_if_nonfinite=True)
            optimizer.step();report['optimizer_steps']+=1
            report.update(last_loss=float(loss.detach()),last_gradient_norm=float(norm))
            kl=joint_kl(policy,reference,[s for s,_ in active])
            if kl>.045:
                report['stop_reason']='imitation minibatch KL exceeds existing bound';stop=True;break
        if not stop:
            report['epochs_completed']=epoch+1
            kl=joint_kl(policy,reference,samples)
            if kl>.045:report['stop_reason']='full-bank KL exceeds existing bound';stop=True
    report.update(kl=joint_kl(policy,reference,samples),action_error_after_mps2=action_error(policy,pairs))
    if not stop:report['stop_reason']='all registered imitation epochs complete'
    if report['action_error_after_mps2']>=before:
        report['error_not_decreased_explanation']=('KL bound prevented further fitting' if stop else
            'AWR likelihood improved objective need not reduce unweighted L1 action error; no outcome-based adjustment')
    return report


def _own_pass(policy,optimizer,pairs,generator,coefficient,passes,flags,pre_update,samples,before,report):
    """imitation_trust='own_pass': own post-PPO trust region, guaranteed complete first pass."""
    limit=report['kl_limit']
    reference=copy.deepcopy(policy)          # frozen pre-imitation snapshot = own trust region
    report['kl_reference']='frozen post-PPO pre-imitation policy (own_pass)'
    report['imitation_trust']='own_pass'
    report['rule']='all bank pairs; inherited AWR x beta and cooperative oversampling unchanged; own post-PPO KL trust region, cooperative-first guaranteed first pass, per-step learning-rate halving instead of stopping'
    if flags is not None:
        coop,other=_cooperative_split(pairs,flags)[1:]
        report['cooperative_pairs']=len(coop)
        report['action_error_before_cooperative_mps2']=action_error(policy,coop) if coop else None
        report['action_error_before_other_mps2']=action_error(policy,other) if other else None
    else:coop,other=None,None
    base_lr=[float(g['lr']) for g in optimizer.param_groups]
    lr_scale=1.;schedule=[];halvings=0
    for epoch in range(int(passes)):
        report['epochs_started']=epoch+1
        first=epoch==0
        if first:sequence=(coop+other if flags is not None else list(pairs))
        else:sequence=[pairs[int(i)] for i in torch.randperm(len(pairs),generator=generator)]
        for start in range(0,len(sequence),128):
            batch=sequence[start:start+128];report['pairs_seen']+=len(batch)
            active=[(s,w) for s,w in batch if indices(s)]
            report['member_actions_seen']+=sum(len(indices(s)) for s,_ in active)
            if not active:continue
            loss=torch.stack([imitation(policy,s,w) for s,w in active]).mean()*coefficient
            if not bool(torch.isfinite(loss)):raise ValueError('nonfinite full-bank imitation')
            optimizer.zero_grad();loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),.5,error_if_nonfinite=True)
            lr_scale,kl_step,trials=_projected_step(policy,optimizer,reference,samples,limit,lr_scale,base_lr)
            halvings+=trials-1
            schedule.append(dict(step=len(schedule)+1,scale=lr_scale,kl=kl_step,trials=trials))
            report['optimizer_steps']+=1
            report.update(last_loss=float(loss.detach()),last_gradient_norm=float(norm))
        report['epochs_completed']=epoch+1
        report['passes_completed']=epoch+1
        if first:report['kl_after_pass1']=joint_kl(policy,reference,samples)
    report.update(kl=joint_kl(policy,reference,samples),action_error_after_mps2=action_error(policy,pairs))
    if flags is not None:
        report['action_error_after_cooperative_mps2']=action_error(policy,coop) if coop else None
        report['action_error_after_other_mps2']=action_error(policy,other) if other else None
    report['lr_schedule']=dict(rule='halve the per-step learning rate, rolling back the trial step, whenever the projected post-step full-bank KL would exceed the limit',
        base_lr=base_lr,final_scale=lr_scale,total_halvings=halvings,per_step=schedule)
    report['stop_reason']='all registered imitation epochs complete'
    if report['action_error_after_mps2']>=before:
        report['error_not_decreased_explanation']=('KL bound prevented further fitting' if stop else
            'AWR likelihood improved objective need not reduce unweighted L1 action error; no outcome-based adjustment')
    return report
