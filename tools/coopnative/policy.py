"""Observable CTDE actor/critic with exact shared-component exploration density."""
import math
import torch
from torch import nn
from tools.coopfree.policy import gaussian_log_prob, leader_gap
from .protocol import WINDOW

FIELDS = {'vid', 's_m', 'v_mps', 'lane', 'is_cav', 'length_m'}
FEATURES = 29


def encode(rows):
    if any(set(r) != FIELDS for r in rows):
        raise ValueError('sensor whitelist violation')
    if len({r['vid'] for r in rows}) != len(rows):
        raise ValueError('duplicate sensor identities')
    if any(not all(math.isfinite(float(r[k])) for k in ('s_m', 'v_mps', 'length_m')) for r in rows):
        raise ValueError('nonfinite sensor values')
    rows = sorted(rows, key=lambda r: r['vid'])
    members, x, anchors = [], [], []
    rsu = []
    for lane in range(3):
        local = [r for r in rows if r['lane'] == lane and -100 <= r['s_m'] <= 75.44]
        rsu.extend([len(local)/10, sum(r['v_mps'] for r in local)/max(1,len(local))/30,
                    sum(r['s_m'] for r in local)/max(1,len(local))/100,
                    sum(r['is_cav'] for r in local)/10])
    for own in rows:
        if not own['is_cav']:
            continue
        members.append(own['vid'])
        f = [own['s_m']/1000, own['v_mps']/30, *[float(own['lane'] == l) for l in range(3)]]
        sensed = [r for r in rows if abs(r['s_m']-own['s_m']) <= 100]
        for lane in range(3):
            for direction in (-1, 1):
                candidates = [r for r in sensed if r['vid'] != own['vid'] and r['lane'] == lane
                              and direction*(r['s_m']-own['s_m']) > 0]
                r = min(candidates, key=lambda r: (abs(r['s_m']-own['s_m']), r['vid'])) if candidates else None
                f.extend([(r['s_m']-own['s_m'])/100, r['v_mps']/30, float(r['is_cav']), 1] if r else [0]*4)
        x.append(f)
        gap = leader_gap(own, sensed)
        anchors.append([own['v_mps'], math.log(max(1e-6, gap/max(own['v_mps'],1e-6))) if gap is not None else 0.])
    return dict(members=members, x=x, rsu=rsu, anchors=anchors)


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.broadcast = nn.Sequential(nn.Linear(12,32), nn.Tanh(), nn.Linear(32,4))
        self.actor = nn.Sequential(nn.Linear(FEATURES+4,64), nn.Tanh(), nn.Linear(64,64), nn.Tanh(), nn.Linear(64,2))
        self.critic = nn.Sequential(nn.Linear(FEATURES+12,128), nn.Tanh(), nn.Linear(128,64), nn.Tanh(), nn.Linear(64,1))
        self.log_std = nn.Parameter(torch.tensor([math.log(4.5), math.log(.5)]))
        for head in (self.broadcast[-1], self.actor[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, context):
        x = torch.tensor(context['x'], dtype=torch.float32).reshape(-1,FEATURES)
        rsu = torch.tensor(context['rsu'], dtype=torch.float32)
        latent = self.broadcast(rsu).expand(len(x),-1)
        mean = self.actor(torch.cat((x,latent),-1))
        pooled = x.mean(0) if len(x) else torch.zeros(FEATURES)
        value = self.critic(torch.cat((pooled,rsu))).squeeze(-1)
        return mean, value

    @torch.no_grad()
    def sample(self, context, generator, deterministic=False):
        mean, value = self(context)
        rho = .5  # Fixed joint covariance, not a behaviour prescription.
        sigma = self.log_std.exp()
        noise = math.sqrt(rho)*torch.randn((1,2),generator=generator) + math.sqrt(1-rho)*torch.randn(mean.shape,generator=generator)
        action = mean if deterministic else mean+noise*sigma
        logp = gaussian_log_prob(action,mean,rho,sigma) if len(mean) else torch.tensor(0.)
        return dict(context=context, action=action.tolist(), rho=rho, window=WINDOW,
                    logp=float(logp), value=float(value))

    def score(self, sample):
        mean, value = self(sample['context'])
        action = torch.tensor(sample['action']).reshape(-1,2)
        logp = gaussian_log_prob(action,mean,sample['rho'],self.log_std.exp()) if len(mean) else mean.sum()
        return logp, value


def targets(sample):
    out = {}
    for vid, action, anchor in zip(sample['context']['members'],sample['action'],sample['context']['anchors']):
        if not all(math.isfinite(v) for v in action):
            raise ValueError('nonfinite action')
        # Exact zero is an abstention for the ENTIRE window, not a speed hold.
        if action == [0.,0.]:
            continue
        out[vid] = dict(speed=max(0.,anchor[0]+action[0]),
                        tau=math.exp(min(20.,max(-20.,anchor[1]+action[1]))))
    return out
