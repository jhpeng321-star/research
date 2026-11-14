"""Exact deterministic credit and team-sharing objective; no behaviour bonuses."""
import math
import numpy as np
import torch
from .policy import log_prob
from .protocol import MIN_WINDOWS


def credit(baseline, joint, solos, w_ind=1., w_group=1., w_coop=1.):
    weights = (w_ind,w_group,w_coop)
    if not all(math.isfinite(w) and w >= 0 for w in weights) or w_coop > w_ind:
        raise ValueError('nonnegative anti-gaming weights required')
    if not solos or not all(math.isfinite(v) for v in [baseline,joint,*solos]):
        raise ValueError('finite complete member counterfactuals required')
    dg = baseline-joint
    di = [baseline-s for s in solos]
    cg = dg-sum(di)
    objective = w_ind*sum(di)+w_group*dg+w_coop*cg
    return dict(D_G=dg, D_i=di, sum_D_i=sum(di), C_G=cg, objective=objective,
                member_team_credit=[w_ind*d+w_group*dg+w_coop*cg for d in di])


def update(policy, optimizer, records):
    if len(records) < MIN_WINDOWS or len({r['world']['seed'] for r in records}) < 12:
        raise ValueError('>=150 windows across >=12 worlds required before every update')
    # A common fixed unit preserves the registered objective including negative returns.
    optimizer.zero_grad()
    loss = torch.stack([-log_prob(policy,r['sample'])*r['credit']['600']['objective']/10000.
                        for r in records]).mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError('nonfinite score loss')
    loss.backward()
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in policy.parameters()):
        raise ValueError('nonfinite gradient')
    grads = torch.cat([p.grad.flatten() for p in policy.parameters() if p.grad is not None])
    optimizer.step()
    out = dict(loss=float(loss.detach()),gradient_norm=float(grads.norm()),windows=len(records),
               worlds=len({r['world']['seed'] for r in records}))
    for key in ('D_G','sum_D_i','C_G'):
        values = np.asarray([r['credit']['600'][key] for r in records])
        out[key] = dict(mean=float(values.mean()),sd=float(values.std()),
                        quantiles=np.quantile(values,[0,.25,.5,.75,1]).tolist())
    out['positive_C_G_share'] = sum(r['credit']['600']['C_G'] > 0 for r in records)/len(records)
    out['executed_change_share'] = sum(r['execution']['motion_changed'] for r in records)/len(records)
    issued = sum(r['execution']['issued_changed'] for r in records)
    out['issued_change_share'] = issued/len(records)
    out['executed_vs_issued_share'] = sum(r['execution']['motion_changed'] and r['execution']['issued_changed']
                                          for r in records)/issued if issued else None
    requested = sum(r['execution']['requests'] for r in records)
    out['safety_override_share'] = sum(r['execution']['overrides'] for r in records)/requested if requested else 0.
    return out


def next_sigma(sigma, executed_share, enlargements):
    if not 0 <= executed_share <= 1:
        raise ValueError('execution share outside [0,1]')
    grow = executed_share <= .5 and enlargements < 2
    return ([v*math.sqrt(2) for v in sigma] if grow else list(sigma)), enlargements+int(grow)
