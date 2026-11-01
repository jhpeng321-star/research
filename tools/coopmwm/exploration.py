"""Registered WM proposal routing; all grading is observable SUMO team return.

The query indicator is part of the recorded behaviour measure. Its policy
component has mass (1-p); the WM query component can have deterministic atoms
and a within-query Bernoulli execution decision with bucket-specific beta.
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
GROUP_RATE = 1/900
ORDINARY_RATE = 1/3600
OPTIONS = ('O1', 'O2', 'O3', 'O4')
REGIMES = ('merge-dense', 'ordinary')
BUCKETS = tuple(o+'|'+r for o in OPTIONS for r in REGIMES)
# LEVERPROBE RECORDED_QUANTILES.json: original four TRAIN-cell upper quartiles.
# Observable projection is the UNION of the inherited high bands, without true
# demand/share labels. These are recorded data, not new behavioural thresholds.
HIGH_BANDS = ((4.,34.19972640218879),(4.,39.89968080255359),
              (7.,37.99969600243199),(8.,41.799665602675184))


def regime(rows):
    from tools.leverprobe.selection import features
    f = features(rows)
    dense = f['ramp'] and f['pair'] and any(
        f['cav_count'] >= c and f['density'] >= d for c,d in HIGH_BANDS)
    return 'merge-dense' if dense else 'ordinary'


def bucket_beta(previous, option, traffic):
    if isinstance(previous, (float, int)):
        return float(previous)
    return previous.get('buckets', {}).get(option+'|'+traffic, {}).get('beta', previous['beta'])


def rate(rows):
    return GROUP_RATE if regime(rows) == 'merge-dense' else ORDINARY_RATE


def route(sample, rows, beta, generator):
    if any(not 0 <= bucket_beta(beta,o,regime(rows)) <= 1 for o in OPTIONS):
        raise ValueError('beta outside probability interval')
    p = rate(rows)
    # Query independent of trust: the winning type is unknown until inference.
    # The recorded query indicator is the augmented behaviour measure; its
    # entire WM route has target mass zero, including declined/rejected proposals.
    guided = float(torch.rand((), generator=generator)) < p
    sample['behaviour'] = dict(route='wm' if guided else 'policy', beta=0.,
        regime=regime(rows), sample_probability=p, mixture_mass=p, q=0., executed=False)
    return guided


def execute(sample, proposal, previous, generator):
    b = sample['behaviour']
    if not proposal['accepted']:
        b['rejection'] = proposal.get('reason', 'nonpositive saving or unavailable')
        return sample
    beta = bucket_beta(previous, proposal['option'], b['regime'])
    probability=proposal.get('option_probability',1.)
    b.update(beta=beta, option=proposal['option'], q=b['sample_probability']*beta*probability,
              option_probability=probability,available_options=proposal.get('available_options'),
              proposal_accepted=True)
    if float(torch.rand((), generator=generator)) < beta:
        return apply_proposal(sample, proposal)
    return sample


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
                route=b['route'], executed=b['executed'], q=b.get('q',b['mixture_mass']),
                p=b['sample_probability'], regime=b['regime'],
                advantage=value, option=b.get('option'), time_s=s['time_s']))
    return out


def decay(previous, mean, ci):
    lo,hi = ci
    e = max(abs(mean-lo),abs(hi-mean))
    return min(previous, max(0.,hi)/(abs(mean)+e)) if abs(mean)+e else 0.


def trust(previous, records, bootstraps=10000):
    """World-bootstrap normal-means empirical Bayes, all evidence from SUMO.

    Executed proposals' H300 critic residuals minus own-route residuals in the
    same observable regime (Hajek inverse-route weights). The winner-type label
    is missing for nonqueries: bucket comparisons are critic-based, selection-
    conditional diagnostics, NOT exact or randomized type-specific causal effects.
    Between-bucket variance is method-of-moments, with no tuned prior strength.
    """
    prior = previous if isinstance(previous,dict) else dict(beta=float(previous))
    keys = list(BUCKETS); ws = sorted({r['world'] for r in records})
    # Columns: numerator/denominator per proposal bucket, then own per regime.
    x = np.zeros((len(ws),20)); wi = {w:i for i,w in enumerate(ws)}
    treated_worlds = {k:set() for k in keys}
    for r in records:
        traffic = r.get('regime','ordinary'); i = wi[r['world']]
        if r['executed'] and r['q']>0:
            key = (r.get('option') or 'O1')+'|'+traffic
            j = 2*keys.index(key); weight = 1/r['q']
            treated_worlds[key].add(r['world'])
        elif r['route']=='policy':
            j = 16+2*REGIMES.index(traffic); weight = 1/(1-r.get('p',r['q']))
        else:continue
        x[i,j] += weight*r['advantage'];x[i,j+1] += weight
    out = dict(beta=prior['beta'], buckets={}, worlds=len(ws),
        estimand='H300 SUMO critic residual; selected proposal-type vs own in same observable regime; not exact causal credit',
        executed=sum(r['executed'] for r in records), own=sum(r['route']=='policy' for r in records))
    valid = [i for i,k in enumerate(keys) if len(treated_worlds[k])>=2 and x[:,17+2*(i%2)].sum()>0]
    if len(ws)<2 or not valid:
        out.update(ci95=None,reason='insufficient proposal evidence in two independent worlds')
        out['buckets']={k:dict(beta=bucket_beta(prior,*k.split('|')),ci95=None,shrinkage_to_local=0.) for k in keys}
        return out
    rng = np.random.default_rng(20261101)
    counts = rng.multinomial(len(ws),np.full(len(ws),1/len(ws)),size=bootstraps)
    totals = x.sum(0); draws = counts@x
    def contrasts(t):
        z = t.reshape(*t.shape[:-1],10,2)
        means = np.divide(z[...,0],z[...,1],out=np.full(z.shape[:-1],np.nan),where=z[...,1]>0)
        local = means[...,:8]-means[...,8:][...,np.arange(8)%2]
        # Pooled contrast keeps the observed proposal regime composition.
        den = z[...,:8,1].sum(-1)
        controls = np.nan_to_num(means[...,8:][...,np.arange(8)%2])*z[...,:8,1]
        a,b = z[...,:8,0].sum(-1),controls.sum(-1)
        delta = a-b
        # Cancellation at machine precision is a numerical zero, independent
        # of reward units; never turn roundoff around zero into a trust decision.
        delta = np.where(abs(delta)<=64*np.finfo(float).eps*(abs(a)+abs(b)),0.,delta)
        pooled = np.divide(delta,den,
            out=np.full(den.shape,np.nan),where=den>0)
        return local,pooled
    local,pooled = contrasts(totals); raw,pooled_boot = contrasts(draws)
    pool_draws = pooled_boot[np.isfinite(pooled_boot)]
    ci = list(map(float,np.quantile(pool_draws,[.025,.975])))
    out.update(mean=float(pooled),ci95=ci,beta=decay(prior['beta'],float(pooled),ci))
    se2 = np.nanvar(raw[:,valid],axis=0,ddof=1)
    tau2 = max(0.,float(np.var(local[valid],ddof=1)-np.mean(se2))) if len(valid)>1 else 0.
    out['between_bucket_variance'] = tau2
    for i,k in enumerate(keys):
        old = bucket_beta(prior,*k.split('|'))
        weight = tau2/(tau2+se2[valid.index(i)]) if i in valid and tau2 else 0.
        mu = weight*local[i]+(1-weight)*pooled if weight else pooled
        posterior = weight*raw[:,i]+(1-weight)*pooled_boot if weight else pooled_boot
        posterior = posterior[np.isfinite(posterior)]
        interval = list(map(float,np.quantile(posterior,[.025,.975])))
        out['buckets'][k] = dict(beta=decay(old,float(mu),interval),previous=old,
            mean=float(mu),ci95=interval,shrinkage_to_local=float(weight),
            proposal_worlds=len(treated_worlds[k]),raw_mean=float(local[i]) if i in valid else None)
    return out
