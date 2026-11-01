"""Registered WM proposal routing; all grading is observable SUMO team return.

The routing indicator is part of the recorded behaviour measure. Its policy
component has mass (1-p*beta); its WM component can have deterministic atoms.
The target measure puts mass one on the policy route and zero on the WM route.
This avoids assigning a fictitious Gaussian density to deterministic proposals.
Rejected WM proposals also retain their route and zero policy importance weight.
"""
import copy
import math
import numpy as np
import torch
from tools.coopnative.learner import REWARD_UNIT, advantages
from tools.coopfree.policy import gaussian_log_prob

# GPU-bound prospective sampling rates, fixed before physical outcomes.
GROUP_RATE = 1/180
ORDINARY_RATE = 1/720


def rate(rows):
    from tools.coopbandit.protocol import group
    return GROUP_RATE if len(group(rows)) == 2 else ORDINARY_RATE


def route(sample, rows, beta, generator):
    if not 0 <= beta <= 1:
        raise ValueError('beta outside probability interval')
    p = rate(rows)
    guided = float(torch.rand((), generator=generator)) < p*beta
    sample['behaviour'] = dict(route='wm' if guided else 'policy', beta=beta,
        sample_probability=p, mixture_mass=p*beta, executed=False)
    return guided


def mixture_ratio(logp, sample):
    b = sample.get('behaviour')
    if not b:
        return (logp-sample['logp']).exp()
    if b['route'] == 'wm':
        return logp*0  # Singular proposals never acquire a made-up policy density.
    return (logp-sample['logp']-math.log1p(-b['mixture_mass'])).exp()


def apply_proposal(sample, proposal):
    out = copy.deepcopy(sample)
    if not proposal['accepted']:
        return out
    members = out['context']['members']
    for vid, target in proposal['targets'].items():
        i = members.index(vid)
        anchor = out['context']['anchors'][i]
        out['action'][i] = [target['speed']-anchor[0], math.log(target['tau'])-anchor[1]]
    out['behaviour'].update(executed=True, option=proposal['option'],
        proposal_members=list(proposal['targets']), expected_saving=proposal['expected_saving'])
    return out


def imitation(policy, sample):
    b = sample.get('behaviour', {})
    mean, _ = policy(sample['context'])
    if not b.get('executed'):
        return mean.sum()*0
    indices = [sample['context']['members'].index(v) for v in b['proposal_members']]
    action = torch.tensor(sample['action'], dtype=mean.dtype).reshape(-1, 2)
    return -b['beta']*gaussian_log_prob(action[indices], mean[indices],
        sample['rho'], policy.log_std.exp())/max(1, len(indices))


def update(policy, optimizer, episodes, generator, epochs=4):
    records = [(s,a,r) for ep in episodes for s,(a,r) in zip(ep['steps'],advantages(ep['steps']))]
    adv = torch.tensor([a for _,a,_ in records])
    adv = (adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-8)
    metrics = []
    for _ in range(epochs):
        for indices in torch.randperm(len(records), generator=generator).split(128):
            samples = [records[int(i)][0]['sample'] for i in indices]
            scores = [policy.score(s) for s in samples]
            logps, values = map(torch.stack, zip(*scores))
            ratio = torch.stack([mixture_ratio(lp,s) for lp,s in zip(logps,samples)])
            a = adv[indices]
            # WM-route target density is identically zero, including the clipped term.
            mask = torch.tensor([s.get('behaviour',{}).get('route') != 'wm' for s in samples])
            actor = -(torch.minimum(ratio*a, ratio.clamp(.8,1.2)*a)*mask).mean()
            returns = torch.tensor([records[int(i)][2] for i in indices])
            oldv = torch.tensor([s['value'] for s in samples])
            clipped = oldv+(values-oldv).clamp(-.2,.2)
            value_loss = torch.maximum((values-returns).square(), (clipped-returns).square()).mean()
            auxiliary = torch.stack([imitation(policy,s) for s in samples]).mean()
            loss = actor+.5*value_loss+auxiliary
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite mixture PPO objective')
            optimizer.zero_grad();loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), .5, error_if_nonfinite=True)
            optimizer.step()
            # KL is between policy distributions, not between a density and an atom.
            delta = logps-torch.tensor([s['logp'] for s in samples])
            kl = float(((delta.exp()-1-delta)*mask).sum().detach()/mask.sum().clamp_min(1))
            metrics.append(dict(loss=float(loss.detach()), kl=kl, gradient_norm=float(norm),
                imitation=float(auxiliary.detach()), wm_routes=int((~mask).sum())))
            if kl > .045:
                return metrics
    return metrics


def evidence(episodes):
    """H300 critic-based SUMO return; right-censored windows excluded prospectively."""
    out = []
    for ep in episodes:
        steps = ep['steps']
        for i,s in enumerate(steps):
            if i+30 > len(steps):
                continue
            b = s['sample'].get('behaviour')
            if not b:
                continue
            value = sum(x['reward'] for x in steps[i:i+30])
            value += (steps[i+30]['sample']['value'] if i+30 < len(steps) else 0)*REWARD_UNIT
            value -= s['sample']['value']*REWARD_UNIT
            out.append(dict(world=ep['world']['seed'], cell=ep['world']['cell'],
                route=b['route'], executed=b['executed'], q=b['mixture_mass'],
                advantage=value, option=b.get('option'), time_s=s['time_s']))
    return out


def trust(previous, records, bootstraps=10000):
    """Monotone, scale-free 95% evidence rule; no behavioural cutoffs.

    IPW randomized intent-to-guide contrast (rejections included) is the primary
    trust statistic. Executed-proposal residuals are separately descriptive, not
    a claim of observing two real futures from one physical window.
    """
    worlds = sorted({r['world'] for r in records})
    values = []
    for w in worlds:
        rs = [r for r in records if r['world'] == w and r['q'] > 0]
        if rs:
            values.append(float(np.mean([r['advantage']*(1/r['q'] if r['route']=='wm'
                else -1/(1-r['q'])) for r in rs])))
    proposed = [r['advantage'] for r in records if r['executed']]
    own = [r['advantage'] for r in records if r['route']=='policy']
    if len(values)<2 or not any(r['route']=='wm' for r in records):
        return dict(beta=previous, ci95=None, reason='insufficient randomized WM-route evidence',
            worlds=len(values), executed=len(proposed), own=len(own))
    rng = np.random.default_rng(20261101)
    v = np.asarray(values);mean = float(v.mean())
    boot = np.array([v[rng.integers(len(v),size=len(v))].mean() for _ in range(bootstraps)])
    lo,hi = map(float,np.quantile(boot,[.025,.975]))
    uncertainty = max(abs(mean-lo),abs(hi-mean))
    denominator = abs(mean)+uncertainty
    beta = min(previous, max(0.,hi)/denominator) if denominator else 0.
    return dict(beta=beta, previous=previous, mean=mean, ci95=[lo,hi], worlds=len(values),
        executed=len(proposed), own=len(own), executed_advantage_mean=float(np.mean(proposed)) if proposed else None,
        own_advantage_mean=float(np.mean(own)) if own else None,
        estimand='world-cluster IPW H300 SUMO critic-residual intent-to-guide contrast')
