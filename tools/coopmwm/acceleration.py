"""One-second MAPPO acceleration residuals; historical set-point ABI is untouched."""
import copy
import math
import torch
from torch import nn
from tools.coopnative.policy import Policy as Network, encode
from tools.coopfree.policy import gaussian_log_prob

PHI = math.exp(-1/5)  # Registered five-second AR(1) correlation time, all arms.
RHO = .5


class Policy(Network):
    def __init__(self):
        super().__init__()
        self.actor[-1] = nn.Linear(64, 1)
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        self.log_std = nn.Parameter(torch.tensor([math.log(.5)]))

    @torch.no_grad()
    def sample(self, context, generator, deterministic=False, memory=None):
        memory = memory or {}
        mean, value = self(context)
        previous = torch.tensor([[memory.get(v, 0.)] for v in context['members']]).reshape(-1, 1)
        # All members use the same conditional innovation scale, including entrants.
        # Entrants start with known zero noise, not an unlogged stationary draw.
        sigma = self.log_std.exp()*math.sqrt(1-PHI**2)
        innovation = math.sqrt(RHO)*torch.randn((1, 1), generator=generator)
        innovation = innovation+math.sqrt(1-RHO)*torch.randn(mean.shape, generator=generator)
        noise = previous*PHI+sigma*innovation
        action = mean if deterministic else mean+noise
        conditional = mean if deterministic else mean+previous*PHI
        logp = gaussian_log_prob(action, conditional, RHO, sigma) if len(mean) else mean.sum()
        sample = dict(context=context, action=action.tolist(), own_action=action.tolist(),
            previous_noise=previous.tolist(), deterministic=deterministic, rho=RHO,
            window=1, logp=float(logp), value=float(value))
        return sample, dict(zip(context['members'], noise.flatten().tolist()))

    def score(self, sample):
        mean, value = self(sample['context'])
        conditional = mean+PHI*torch.tensor(sample['previous_noise']).reshape(-1, 1)
        sigma = self.log_std.exp()*math.sqrt(1-PHI**2)
        action = torch.tensor(sample['own_action']).reshape(-1, 1)
        return (gaussian_log_prob(action, conditional, RHO, sigma) if len(mean) else mean.sum()), value


def requests(sample, commands):
    """Only raw pre-safety commands; never write TraCI or bypass ownership."""
    out = {}
    for vid, (delta,) in zip(sample['context']['members'], sample['action']):
        if not math.isfinite(delta):
            raise ValueError('nonfinite acceleration residual')
        if delta != 0.:
            out[vid] = dict(requested_raw=commands[vid]+delta, acceleration_residual=delta)
    for vid,raw in sample.get('absolute_accels',{}).items():
        if vid in commands:out[vid]=dict(requested_raw=raw,acceleration_residual=raw-commands[vid])
    return out


class Controller:
    def __init__(self, policy, generator, deterministic=False, guide=None, beta=1.,
                 replay=None, branch=None):
        self.policy, self.generator, self.deterministic = policy, generator, deterministic
        self.guide, self.beta, self.replay, self.branch = guide, beta, replay, branch
        self.events, self.memory = [], {}
        self.plan, self.plan_start = None, None

    def apply(self, rows, commands, now):
        from .plans import acceleration_at
        from .exploration import regime, bucket_beta
        context = encode(rows)
        if self.replay is not None:
            sample = copy.deepcopy(self.replay[str(int(now))])
        else:
            sample, self.memory = self.policy.sample(context, self.generator, self.deterministic, self.memory)
        from .plans import group as observable_group
        observed_group=observable_group(rows)
        if self.replay is None and not self.deterministic and int(now)%10==0 and len(observed_group)==2:
            sample['behaviour']=dict(route='policy',executed=False,sample_probability=0.,q=0.,regime=regime(rows))
        proposal = None
        if self.plan and now >= self.plan_start+self.plan['duration_s']:
            self.plan = self.plan_start = None
        if self.replay is None and self.guide and not self.deterministic and self.plan is None:
            proposal = self.guide(rows, commands, now, sample)
            if proposal is not None:
                b = proposal['behaviour']
                sample['behaviour'] = b
                if proposal['accepted']:
                    b['beta'] = bucket_beta(self.beta, proposal['option'], b['regime'])
                    b['q'] = b.get('sample_probability',1.)*b['beta']
                    if float(torch.rand((), generator=self.generator)) < b['beta']:
                        self.plan, self.plan_start = proposal['plan'], now
                        b.update(executed=True, plan_start=True)
                if hasattr(self.guide,'completed'):
                    self.guide.completed(proposal,b.get('executed',False))
        if self.plan is not None and self.replay is None:
            if 'behaviour' not in sample:
                sample['behaviour'] = dict(route='plan', executed=True, plan_start=False,
                    beta=bucket_beta(self.beta, self.plan['family'], regime(rows)),
                    option=self.plan['family'], regime=regime(rows))
            b = sample['behaviour']
            b.update(plan_id=self.plan['id'], proposal_members=self.plan['members'])
            sample['absolute_accels']=acceleration_at(self.plan,now-self.plan_start)
            for vid, raw in sample['absolute_accels'].items():
                if vid in context['members']:
                    sample['action'][context['members'].index(vid)] = [raw-commands[vid]]
        desired = requests(sample, commands)
        # Exact twins release only the selected group's whole committed plan;
        # all non-group sampled actions are byte-identical on the replayed prefix.
        if self.branch and now >= self.branch['time_s']:
            if self.branch.get('mappo_background'):
                allowed=set(self.branch['group'])-set(self.branch['revert']) if now<self.branch['until'] else set()
                desired={v:r for v,r in desired.items() if v in allowed}
            release=(self.branch.get('group',[]) if now>=self.branch.get('until',float('inf')) else self.branch['revert'])
            for vid in release:
                desired.pop(vid, None)
        self.events.append(dict(time_s=int(now), sample=sample, rows=rows,
            commands=commands, group=(self.plan['members'] if self.plan else observed_group),
            proposal=proposal, requests=desired))
        return desired
