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
            if getattr(policy,'intent_horizon',0):  # v2v_intent ON only: plan intent targets
                from .v2v_intent import bank_term;loss=loss+bank_term(policy,active)
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
STEP_MODES=('fixed','kl_targeted')   # c4/imit-strength: registered imitation-strength modes
STEP_SCALE_CAP=16   # kl_targeted: a pass's step scale never exceeds 16x the base learning rate
STEPS_CAP=4         # kl_targeted: optimizer steps per pass never exceed 4x (minibatch 128//mult)
KL_LIMIT=.045       # the registered separate-imitation bound (report kl_limit; never exceeded)
KL_LIMITS=(.045,.06,.09)   # c4/imit-kl-option: registered imitation_kl_limit alternatives
PPO_STOP_KL=.045    # accel_learner stops PPO as soon as its minibatch KL exceeds .045; the
                    # report-only total-KL flag threshold is PPO_STOP_KL + the runtime limit

RUNTIME_STEP_MODE='fixed'       # bound from the deployed runtime by v2v_live.hooks (B and R alike)
RUNTIME_INTENT_BANK_TERM=True   # bound likewise; False drops the bank intent term from the loss
RUNTIME_KL_LIMIT=KL_LIMIT       # bound likewise; the runtime's registered imitation_kl_limit


def install(step_mode='fixed',intent_bank_term=True,kl_limit=None):
    """Bind the runtime's registered imitation-strength keys process-wide (v2v_live.hooks).

    One key serves B and R (v2v_launch3 refuses any B/R mismatch before this is called); explicit
    run_separate arguments still win, and the process default stays 'fixed'/True/.045 (byte-identical).
    kl_limit must be a registered alternative (KL_LIMITS); None keeps the current binding (the
    module default is .045, identical to every receipt published before c4/imit-kl-option).
    """
    global RUNTIME_STEP_MODE,RUNTIME_INTENT_BANK_TERM,RUNTIME_KL_LIMIT
    if step_mode not in STEP_MODES:raise ValueError('unknown imitation_step_mode '+repr(step_mode))
    if not isinstance(intent_bank_term,bool):raise ValueError('v2v_intent_bank_term must be a bool')
    if kl_limit is None:kl_limit=RUNTIME_KL_LIMIT
    if type(kl_limit) not in (int,float):raise ValueError('unknown imitation_kl_limit '+repr(kl_limit))
    kl_limit=float(kl_limit)
    if kl_limit not in KL_LIMITS:raise ValueError('unknown imitation_kl_limit '+repr(kl_limit))
    RUNTIME_STEP_MODE,RUNTIME_INTENT_BANK_TERM,RUNTIME_KL_LIMIT=step_mode,bool(intent_bank_term),kl_limit


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


PASS1_BOOTSTRAP=10000        # MAIN 2026-11-04 09:00: statistical pass-1 effectiveness (report)
PASS1_BOOTSTRAP_SEED=20261101


@torch.no_grad()
def pair_losses(policy,pairs,coefficient=1.):
    """Per-pair AWR imitation loss (the same term awr_loss averages)."""
    return [float(imitation(policy,s,w))*coefficient for s,w in pairs]


def pass1_loss_change(before,after,resamples=PASS1_BOOTSTRAP,seed=PASS1_BOOTSTRAP_SEED):
    """Paired per-pair pass-1 loss change (after - before): mean and percentile bootstrap 95% CI
    over imitation pairs (fixed seed). 'worse' only when the whole CI is above 0, i.e. pass 1
    made the imitation loss significantly WORSE; a non-significant change is reported only."""
    import numpy as np
    d=np.asarray(after,dtype=float)-np.asarray(before,dtype=float)
    if not len(d) or not np.all(np.isfinite(d)):
        return dict(n=int(len(d)),mean=None,ci95=None,significantly_worse=None,significantly_better=None)
    idx=np.random.default_rng(seed).integers(0,len(d),(int(resamples),len(d)))
    lo,hi=(float(x) for x in np.quantile(d[idx].mean(1),[.025,.975]))
    return dict(n=int(len(d)),mean=float(d.mean()),ci95=[lo,hi],significantly_worse=bool(lo>0),
        significantly_better=bool(hi<0),bootstrap=int(resamples),seed=int(seed),
        rule='paired per-pair pass-1 AWR loss change after-before; percentile bootstrap over imitation pairs')


def run_separate(policy,optimizer,pairs,generator,reference,coefficient=1.,passes=5,
                 step_mode=None,intent_bank_term=None,kl_limit=None):
    """MAIN17:27 transactional full passes, KL relative to post-PPO policy.

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

    c4/imit-strength: step_mode 'fixed' (default, byte-identical) keeps the
    halving-only backtracking above. 'kl_targeted' additionally doubles the NEXT
    pass's step scale (capped at STEP_SCALE_CAP x base lr) and its number of
    optimizer steps (capped at STEPS_CAP x, via smaller minibatches of the same
    fixed permutation) whenever the ACCEPTED pass ends with KL (vs the post-PPO
    snapshot, same joint_kl measurement) below half the registered KL limit;
    the existing halving backtracking is unchanged and the limit is never
    exceeded. Per accepted pass the report records step_scale_mult, steps_mult
    and optimizer_steps. A None step_mode falls back to the runtime binding
    installed by v2v_live.hooks (identical for B and R). intent_bank_term=False
    (runtime key v2v_intent_bank_term, default True = current) drops the
    v2v_intent bank term from the imitation loss; the TRAINED-subset (w>0,
    trainable pairs) bank intent loss is reported by exact_update's v2v_intent
    block as before_trained/after_trained alongside the all-pairs values.

    c4/imit-kl-option: the separate-imitation trust region is the registered
    option imitation_kl_limit (default .045 = every earlier receipt,
    byte-identical). A None kl_limit falls back to the runtime binding
    (RUNTIME_KL_LIMIT, installed by v2v_live.hooks for B and R alike); an
    explicit argument wins and only registered alternatives (KL_LIMITS) are
    accepted. The limit bounds every acceptance check here and the kl_targeted
    growth threshold (limit/2); the report records it as kl_limit. Per-update
    safety note: when an imitation option deviates from its registered default
    (kl_targeted or a non-default limit), the report adds
    total_kl_vs_pre_update, the exact joint KL of the published (post-PPO +
    imitation) policy vs the FROZEN PRE-UPDATE policy (the caller's reference
    snapshot) over the same bank samples, i.e. the PPO + imitation total drift;
    report-only, no separate threshold. When the runtime limit EXCEEDS the
    .045 default, the report additionally carries the boolean
    total_kl_above_ppo_stop_plus_limit (True when total_kl_vs_pre_update
    exceeds PPO_STOP_KL + kl_limit, i.e. the PPO stopping bound plus this
    runtime's imitation limit); report-only, surfaced by the learningwatch
    report, never a pause criterion. The 'fixed'/.045 report stays
    byte-identical to the branch point.
    """
    if int(passes)!=passes or passes<5:raise ValueError('at least five requested passes')
    step_mode=RUNTIME_STEP_MODE if step_mode is None else str(step_mode)
    if step_mode not in STEP_MODES:raise ValueError('unknown imitation_step_mode '+repr(step_mode))
    kl_limit=RUNTIME_KL_LIMIT if kl_limit is None else kl_limit
    if type(kl_limit) not in (int,float) or float(kl_limit) not in KL_LIMITS:
        raise ValueError('unknown imitation_kl_limit '+repr(kl_limit))
    kl_limit=float(kl_limit)
    if intent_bank_term is None:intent_bank_term=RUNTIME_INTENT_BANK_TERM
    intent_bank_term=bool(intent_bank_term)
    targeted=step_mode=='kl_targeted'
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
        kl_limit=kl_limit,kl_reference='frozen pre-imitation (post-PPO) policy',
        last_loss=0.,last_gradient_norm=0.,epochs=[],
        pass1_step_scale=None,pass1_loss_drop=None,pass1_effective=False,imitation_failed=False,
        min_additional_pass_step_scale=MIN_ADDITIONAL_PASS_SCALE,
        rule='full-bank transactional passes; separate imitation KL; AWR weights unchanged')
    if targeted:report.update(step_mode=step_mode,step_scale_cap=STEP_SCALE_CAP,steps_cap=STEPS_CAP)
    if not intent_bank_term:report['intent_bank_term']=False
    # c4/imit-kl-option: the per-update safety note (PPO + imitation total KL vs the frozen
    # PRE-UPDATE policy) is carried only when an imitation option deviates from its registered
    # default, keeping the 'fixed'/.045 report byte-identical to the branch point.
    if targeted or kl_limit != KL_LIMIT:
        report['safety_note']='total_kl_vs_pre_update is the exact joint KL of the published policy vs the frozen pre-update policy (PPO + imitation combined); report-only'
    if not pairs:
        extra = dict(total_kl_vs_pre_update=0.,
                     total_kl_reference='frozen pre-update policy (PPO + imitation combined)') \
            if (targeted or kl_limit != KL_LIMIT) else {}
        if kl_limit > KL_LIMIT:   # gate option: the report-only flag joins the empty-bank note too
            extra['total_kl_above_ppo_stop_plus_limit']=False
        return dict(report,stop_reason='empty statistically qualified bank',kl=0.,
                    action_error_after_mps2=None,**extra)
    samples=[s for s,_ in pairs]
    original_lrs=[g['lr'] for g in optimizer.param_groups]

    intent_on=bool(getattr(policy,'intent_horizon',0));shares=[]  # v2v_intent ON: grad share log

    def attempt_pass(order):
        """One full pass -> (status, kl, steps, last_loss, last_norm). Never raises on non-finite values."""
        steps=0;last_loss=last_norm=None;shares.clear()
        for ids in order:
            batch=[pairs[int(i)] for i in ids]
            loss=torch.stack([imitation(policy,s,w) for s,w in batch]).mean()*coefficient
            if intent_on and intent_bank_term:  # v2v_intent ON only: plan intent targets
                from .v2v_intent import bank_term;loss=loss+bank_term(policy,batch)
            if not bool(torch.isfinite(loss)):return 'nonfinite_loss',None,steps,last_loss,last_norm
            optimizer.zero_grad();loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),.5)
            if not bool(torch.isfinite(norm)):return 'nonfinite_gradient',None,steps,last_loss,last_norm
            if intent_on:
                from .v2v_intent import grad_share;shares.append(grad_share(policy))
            optimizer.step();steps+=1
            last_loss=float(loss.detach());last_norm=float(norm)
        try:kl=joint_kl(policy,centre,samples)
        except ValueError:return 'nonfinite_kl',None,steps,last_loss,last_norm
        return ('accepted' if kl<=kl_limit else 'kl_exceeded'),kl,steps,last_loss,last_norm

    try:
        mult=1.;steps_mult=1   # kl_targeted growth multipliers; both stay 1 in fixed mode
        for epoch in range(int(passes)):
            report['epochs_started']=epoch+1
            order=torch.randperm(len(pairs),generator=generator).split(
                max(1,128//(steps_mult if targeted else 1)))
            state=copy.deepcopy(policy.state_dict());adam=copy.deepcopy(optimizer.state_dict())
            loss_before=awr_loss(policy,pairs,coefficient);error_before=action_error(policy,pairs)
            if epoch==0 and targeted:pass1_before=pair_losses(policy,pairs,coefficient)   # report only (no RNG, no grad)
            # pass 1 keeps the guaranteed 25 halvings; later passes stop at scale 1/64 (attempt 6)
            max_attempts=MAX_ATTEMPTS if epoch==0 else round(-math.log2(MIN_ADDITIONAL_PASS_SCALE))+1
            accepted=False;statuses=[];rejected=0;kl=None
            for attempt in range(max_attempts):
                if attempt:
                    policy.load_state_dict(state);optimizer.load_state_dict(copy.deepcopy(adam))
                    report['backtracks']+=1
                scale=(mult if targeted else 1.)*(.5**attempt)
                for g,lr in zip(optimizer.param_groups,original_lrs):g['lr']=lr*scale
                status,kl,steps,last_loss,last_norm=attempt_pass(order)
                statuses.append(status)
                if status=='accepted':
                    accepted=True;break
                rejected+=steps
            row=dict(epoch=epoch+1,accepted=accepted,step_scale=scale if accepted else None,
                     smallest_step_scale_tried=scale,attempts=len(statuses),attempt_status=statuses,kl=kl,
                     awr_loss_before=loss_before,l1_action_error_before=error_before,
                     rejected_optimizer_steps=rejected)
            if targeted:row.update(step_scale_mult=mult,steps_mult=steps_mult,optimizer_steps=steps)
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
            if intent_on:row['intent_grad_shares']=list(shares)
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
                # statistical pass-1 report (MAIN 09:00) in kl_targeted mode only: the 'fixed'
                # report stays byte-identical to its branch point (test_option_off_...)
                if targeted:report['pass1_loss_change']=pass1_loss_change(pass1_before,pair_losses(policy,pairs,coefficient))
            # kl_targeted: an accepted pass that ended below half the runtime's registered bound
            # lets the next pass push harder (step scale x2 capped 16, optimizer steps x2 capped 4).
            if targeted and epoch+1<int(passes) and kl is not None and kl<kl_limit/2:
                mult=min(float(STEP_SCALE_CAP),mult*2.);steps_mult=min(int(STEPS_CAP),steps_mult*2)
            if scale<MIN_ADDITIONAL_PASS_SCALE and epoch+1<int(passes):
                report['stop_reason']=(f'accepted step scale {scale:g} below {MIN_ADDITIONAL_PASS_SCALE:g} '
                    'of base; additional passes stopped')
                break
        else:report['stop_reason']='all registered imitation epochs complete'
    finally:
        for g,lr in zip(optimizer.param_groups,original_lrs):g['lr']=lr
    report.update(kl=joint_kl(policy,centre,samples),action_error_after_mps2=action_error(policy,pairs))
    if targeted or kl_limit != KL_LIMIT:   # see the safety-note comment above (byte-identical default)
        report['total_kl_vs_pre_update']=joint_kl(policy,reference,samples)
        report['total_kl_reference']='frozen pre-update policy (PPO + imitation combined)'
    if kl_limit > KL_LIMIT:   # c4/imit-kl-option gate: report-only flag vs the PPO stopping bound
        report['total_kl_above_ppo_stop_plus_limit']=bool(
            report.get('total_kl_vs_pre_update',0.) > PPO_STOP_KL+kl_limit)
    after=report['action_error_after_mps2']
    if after is not None and before is not None and after>=before:
        report['error_not_decreased_explanation']='AWR likelihood and unweighted L1 differ; full-pass coverage and separate KL recorded'
    return report
