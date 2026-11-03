"""MAIN09:36 full-bank post-PPO AWR passes, bounded by the pre-update policy KL."""
import math
import copy
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


MIN_ADDITIONAL_PASS_SCALE=1/64   # documented numeric floor for passes after pass 1 (fix2)
MAX_ATTEMPTS=25


def trainable(sample,weight):
    """A bank pair can train only if positive-weight, executed and with present members."""
    return weight>0 and bool(sample.get('behaviour',{}).get('executed')) and bool(indices(sample))


@torch.no_grad()
def awr_loss(policy,pairs,coefficient=1.):
    """Full-bank AWR imitation loss (same per-pair term as the minibatch objective)."""
    if not pairs:return None
    return float(torch.stack([imitation(policy,s,w) for s,w in pairs]).mean()*coefficient)


def _finite(x):
    return x is not None and math.isfinite(x)


def run_separate(policy,optimizer,pairs,generator,reference,coefficient=1.,passes=5):
    """MAIN17:27: transactional full passes, KL relative to post-PPO policy.

    Backtracking changes only the optimizer step size, never targets or AWR
    weights. A rejected attempt restores BOTH parameters and Adam state (from a
    deep copy, so later attempts never mutate the snapshot). Its fixed
    permutation is retried without consuming additional sampling RNG. No partial
    pass can become the published policy. A non-finite loss, gradient norm or KL
    rejects the attempt; if every pass-1 attempt is non-finite the untouched
    policy is returned with imitation_failed=True. Passes after the first may not
    use a step scale below MIN_ADDITIONAL_PASS_SCALE of the base learning rate.
    Pairs with executed=False are not trainable and are filtered before counting.
    The inherited .045 KL budget is separate from PPO. Legacy non-V2V run()
    remains byte-for-byte unchanged.
    """
    if int(passes)!=passes or passes<5:raise ValueError('at least five requested passes')
    offered=[(s,float(w)) for s,w in pairs]
    unexecuted=sum(1 for s,w in offered if w>0 and not s.get('behaviour',{}).get('executed'))
    pairs=[(s,w) for s,w in offered if trainable(s,w)]
    centre=copy.deepcopy(policy).eval()
    before=action_error(policy,pairs)
    report=dict(requested_epochs=int(passes),epochs_completed=0,epochs_started=0,
        pairs_total=len(pairs),pairs_filtered_unexecuted=unexecuted,pairs_seen=0,member_actions_seen=0,
        optimizer_steps=0,rejected_optimizer_steps=0,
        critic_steps=0,own_policy_steps=0,plan_ratios_computed=0,backtracks=0,
        action_error_before_mps2=before,action_error_before_update_mps2=action_error(reference,pairs),
        kl_limit=.045,kl_reference='frozen pre-imitation (post-PPO) policy',
        last_loss=0.,last_gradient_norm=0.,epochs=[],
        pass1_step_scale=None,pass1_loss_drop=None,pass1_effective=False,imitation_failed=False,
        min_additional_pass_step_scale=MIN_ADDITIONAL_PASS_SCALE,
        rule='full-bank transactional passes; separate imitation KL; AWR weights unchanged')
    if not pairs:
        return dict(report,stop_reason='empty statistically qualified bank',kl=0.,action_error_after_mps2=None)
    samples=[s for s,_ in pairs]
    original_lrs=[g['lr'] for g in optimizer.param_groups]

    def attempt_pass(order):
        """One full pass -> (status, kl, steps, last_loss, last_norm). Never raises on non-finite values."""
        steps=0;last_loss=last_norm=None
        for ids in order:
            batch=[pairs[int(i)] for i in ids]
            loss=torch.stack([imitation(policy,s,w) for s,w in batch]).mean()*coefficient
            if not bool(torch.isfinite(loss)):return 'nonfinite_loss',None,steps,last_loss,last_norm
            optimizer.zero_grad();loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),.5)
            if not bool(torch.isfinite(norm)):return 'nonfinite_gradient',None,steps,last_loss,last_norm
            optimizer.step();steps+=1
            last_loss=float(loss.detach());last_norm=float(norm)
        try:kl=joint_kl(policy,centre,samples)
        except ValueError:return 'nonfinite_kl',None,steps,last_loss,last_norm
        return ('accepted' if kl<=.045 else 'kl_exceeded'),kl,steps,last_loss,last_norm

    try:
        for epoch in range(int(passes)):
            report['epochs_started']=epoch+1
            order=torch.randperm(len(pairs),generator=generator).split(128)
            state=copy.deepcopy(policy.state_dict());adam=copy.deepcopy(optimizer.state_dict())
            loss_before=awr_loss(policy,pairs,coefficient);error_before=action_error(policy,pairs)
            # pass 1 keeps the guaranteed 25 halvings; later passes stop at scale 1/64 (attempt 6)
            max_attempts=MAX_ATTEMPTS if epoch==0 else round(-math.log2(MIN_ADDITIONAL_PASS_SCALE))+1
            accepted=False;statuses=[];rejected=0;kl=None
            for attempt in range(max_attempts):
                if attempt:
                    policy.load_state_dict(state);optimizer.load_state_dict(copy.deepcopy(adam))
                    report['backtracks']+=1
                for g,lr in zip(optimizer.param_groups,original_lrs):g['lr']=lr*(.5**attempt)
                status,kl,steps,last_loss,last_norm=attempt_pass(order)
                statuses.append(status)
                if status=='accepted':
                    accepted=True;break
                rejected+=steps
            scale=.5**attempt
            row=dict(epoch=epoch+1,accepted=accepted,step_scale=scale if accepted else None,
                     smallest_step_scale_tried=scale,attempts=len(statuses),attempt_status=statuses,kl=kl,
                     awr_loss_before=loss_before,l1_action_error_before=error_before,
                     rejected_optimizer_steps=rejected)
            report['rejected_optimizer_steps']+=rejected
            if not accepted:
                policy.load_state_dict(state);optimizer.load_state_dict(copy.deepcopy(adam))
                row.update(awr_loss_after=loss_before,l1_action_error_after=error_before)
                report['epochs'].append(row)
                if not report['epochs_completed']:
                    if all(x.startswith('nonfinite') for x in statuses):
                        report.update(imitation_failed=True,
                            failure_reason='every pass-1 attempt was non-finite; policy and Adam state untouched',
                            stop_reason='imitation failed: every pass-1 attempt non-finite')
                        break
                    raise RuntimeError('cannot fit one complete imitation pass inside separate KL')
                report['stop_reason']=(f'pass {epoch+1} not accepted at any step scale >= '
                    f'{MIN_ADDITIONAL_PASS_SCALE:g} of base (statuses {statuses}); additional passes stopped')
                break
            row.update(awr_loss_after=awr_loss(policy,pairs,coefficient),l1_action_error_after=action_error(policy,pairs))
            report['epochs'].append(row)
            report['epochs_completed']+=1;report['pairs_seen']+=len(pairs)
            report['member_actions_seen']+=sum(len(indices(s)) for s,_ in pairs)
            report['optimizer_steps']+=steps
            report['last_step_scale']=scale
            report.update(last_loss=last_loss,last_gradient_norm=last_norm)
            if epoch==0:
                drop=(row['awr_loss_before']-row['awr_loss_after']
                      if _finite(row['awr_loss_before']) and _finite(row['awr_loss_after']) else None)
                report.update(pass1_step_scale=scale,pass1_loss_drop=drop,
                              pass1_effective=bool(drop is not None and drop>0))
            if scale<MIN_ADDITIONAL_PASS_SCALE and epoch+1<int(passes):
                report['stop_reason']=(f'accepted step scale {scale:g} below {MIN_ADDITIONAL_PASS_SCALE:g} '
                    'of base; additional passes stopped')
                break
        else:report['stop_reason']='all registered imitation epochs complete'
    finally:
        for g,lr in zip(optimizer.param_groups,original_lrs):g['lr']=lr
    report.update(kl=joint_kl(policy,centre,samples),action_error_after_mps2=action_error(policy,pairs))
    after=report['action_error_after_mps2']
    if after is not None and before is not None and after>=before:
        report['error_not_decreased_explanation']='AWR likelihood and unweighted L1 differ; full-pass coverage and separate KL recorded'
    return report
