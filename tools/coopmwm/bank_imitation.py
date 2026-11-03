"""MAIN09:36 full-bank post-PPO AWR passes, bounded by the pre-update policy KL."""
import math
import copy
import torch
from .acceleration import PHI,RHO
from .accel_learner import imitation


def indices(sample):
    members=sample['context']['members']
    return [members.index(v) for v in sample.get('behaviour',{}).get('proposal_members',[]) if v in members]


def intent_terms(policy,batch):
    """Masked plan-future intent losses for pairs carrying context['intent_target'].

    Each target ({vid: {target, mask}} from bank_v2v.plan_intent_targets) is the
    winning plan's absolute next-H commands; the loss is the masked mean squared
    error against the intent head plan of the pair's own members. [] when the
    batch carries no intent targets."""
    from .v2v_model import planned_commands
    terms=[]
    for sample,_ in batch:
        pack=(sample.get('context') or {}).get('intent_target') or {}
        if not pack:continue
        members=sample['context']['members']
        planned=planned_commands(policy,sample['context'])
        losses=[]
        for vid,item in pack.items():
            if vid not in members:continue
            mask=torch.as_tensor(item['mask'],dtype=torch.float32)
            total=float(mask.sum())
            if total<=0:continue
            target=torch.as_tensor(item['target'],dtype=torch.float32)
            losses.append(((planned[members.index(vid)]-target).square()*mask).sum()/total)
        if losses:terms.append(torch.stack(losses).mean())
    return terms


@torch.no_grad()
def intent_error(policy,pairs):
    """Masked mean absolute plan-future command error (m/s^2); None with no targets."""
    from .v2v_model import planned_commands
    total=0.;count=0.
    for sample,_ in pairs:
        pack=(sample.get('context') or {}).get('intent_target') or {}
        if not pack:continue
        members=sample['context']['members']
        planned=planned_commands(policy,sample['context'])
        for vid,item in pack.items():
            if vid not in members:continue
            mask=torch.as_tensor(item['mask'],dtype=torch.float32)
            weight=float(mask.sum())
            if weight<=0:continue
            target=torch.as_tensor(item['target'],dtype=torch.float32)
            total+=float(((planned[members.index(vid)]-target).abs()*mask).sum());count+=weight
    return total/count if count else None


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


def run(policy,optimizer,pairs,generator,reference,coefficient=1.,passes=5):
    if int(passes)!=passes or passes<5:raise ValueError('MAIN09:36 requires at least five passes')
    pairs=[(s,float(w)) for s,w in pairs if w>0]
    before=action_error(policy,pairs)
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


def run_separate(policy,optimizer,pairs,generator,reference,coefficient=1.,passes=5,intent=None):
    """MAIN17:27: transactional full passes, KL relative to post-PPO policy.

    Backtracking changes only the optimizer step size, never targets or AWR
    weights. A rejected pass restores BOTH parameters and Adam state. Its fixed
    permutation is retried without consuming additional sampling RNG. No partial
    pass can become the published policy. The inherited .045 KL budget is separate
    from PPO. Legacy non-V2V run() remains byte-for-byte unchanged.

    With ``intent`` (DEFAULT OFF: dict(weight, horizon) from v2v_model.intent_config)
    the pairs' plan-future-command intent targets add a supervised loss with the
    registered weight to every minibatch, and the report carries
    report['intent'] with the intent error before/after the passes."""
    if int(passes)!=passes or passes<5:raise ValueError('at least five requested passes')
    pairs=[(s,float(w)) for s,w in pairs if w>0 and indices(s)]
    centre=copy.deepcopy(policy).eval()
    before=action_error(policy,pairs)
    intent_before=intent_error(policy,pairs) if intent is not None else None
    report=dict(requested_epochs=int(passes),epochs_completed=0,epochs_started=0,
        pairs_total=len(pairs),pairs_seen=0,member_actions_seen=0,optimizer_steps=0,
        critic_steps=0,own_policy_steps=0,plan_ratios_computed=0,backtracks=0,
        action_error_before_mps2=before,action_error_before_update_mps2=action_error(reference,pairs),
        kl_limit=.045,kl_reference='frozen pre-imitation (post-PPO) policy',
        rule='full-bank transactional passes; separate imitation KL; AWR weights unchanged')
    if intent is not None:
        report['intent']=dict(weight=float(intent['weight']),horizon=int(intent['horizon']),
            error_before_mps2=intent_before,error_after_mps2=None,
            rule='bank plan-future-command intent targets, masked, same registered weight path as PPO')
    if not pairs:
        if intent is not None:
            report['intent']=dict(report['intent'],error_after_mps2=intent_error(policy,pairs))
        return dict(report,stop_reason='empty statistically qualified bank',kl=0.,action_error_after_mps2=None)
    samples=[s for s,_ in pairs]
    original_lrs=[g['lr'] for g in optimizer.param_groups]
    try:
        for epoch in range(int(passes)):
            report['epochs_started']=epoch+1
            order=torch.randperm(len(pairs),generator=generator).split(128)
            state=copy.deepcopy(policy.state_dict());adam=copy.deepcopy(optimizer.state_dict())
            accepted=False
            for attempt in range(25):
                if attempt:
                    policy.load_state_dict(state);optimizer.load_state_dict(adam)
                    report['backtracks']+=1
                for g,lr in zip(optimizer.param_groups,original_lrs):g['lr']=lr*(.5**attempt)
                steps=0
                for ids in order:
                    batch=[pairs[int(i)] for i in ids]
                    loss=torch.stack([imitation(policy,s,w) for s,w in batch]).mean()*coefficient
                    if intent is not None:
                        terms=intent_terms(policy,batch)
                        if terms:loss=loss+intent['weight']*torch.stack(terms).mean()
                    if not bool(torch.isfinite(loss)):raise ValueError('nonfinite full-bank imitation')
                    optimizer.zero_grad();loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(),.5,error_if_nonfinite=True)
                    optimizer.step();steps+=1
                kl=joint_kl(policy,centre,samples)
                if kl<=.045:
                    accepted=True;break
            if not accepted:
                policy.load_state_dict(state);optimizer.load_state_dict(adam)
                if not report['epochs_completed']:
                    raise RuntimeError('cannot fit one complete imitation pass inside separate KL')
                report['stop_reason']='separate imitation KL prevents another complete pass';break
            report['epochs_completed']+=1;report['pairs_seen']+=len(pairs)
            report['member_actions_seen']+=sum(len(indices(s)) for s,_ in pairs)
            report['optimizer_steps']+=steps
            report['last_step_scale']=.5**attempt
        else:report['stop_reason']='all registered imitation epochs complete'
    finally:
        for g,lr in zip(optimizer.param_groups,original_lrs):g['lr']=lr
    report.update(kl=joint_kl(policy,centre,samples),action_error_after_mps2=action_error(policy,pairs))
    if intent is not None:
        report['intent']=dict(report['intent'],error_after_mps2=intent_error(policy,pairs))
    if report['action_error_after_mps2']>=before:
        report['error_not_decreased_explanation']='AWR likelihood and unweighted L1 differ; full-pass coverage and separate KL recorded'
    return report
