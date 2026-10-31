"""Observable per-CAV set-points and an exactly scored shared Gaussian search law."""
import math
import torch
from torch import nn
from .protocol import ZONE, WINDOWS, group

TRACK_S = 5.  # Existing leverprobe command tracking time, not a behaviour detector.
SIGMA = (4.5, .5)
FEATURES = 43


def leader_gap(own, rows):
    front = [r for r in rows if r['lane'] == own['lane'] and r['s_m'] > own['s_m']]
    if not front:
        return None
    leader = min(front, key=lambda r:(r['s_m'], r['vid']))
    return max(0., leader['s_m']-own['s_m']-leader.get('length_m', 5.)-2.)


def encode(rows, commands, members=None):
    """Whitelist input fields: no seeds, scenario assignments, demand or driver truth."""
    members = group(rows) if members is None else members
    byid = {r['vid']:r for r in rows}
    sensed = [r for r in rows if ZONE[0] <= r['s_m'] <= ZONE[1]]
    rsu = []
    for lane in range(3):
        lane_rows = [r for r in sensed if r['lane'] == lane]
        n = len(lane_rows)
        rsu.extend((n/10., sum(r['is_cav'] for r in lane_rows)/10.,
                    sum(r['v_mps'] for r in lane_rows)/max(1,n)/30.,
                    sum(r['s_m'] for r in lane_rows)/max(1,n)/100.))
    inputs, anchors = [], []
    for vid in members:
        own = byid[vid]
        v_target = max(0., own['v_mps']+TRACK_S*commands[vid])
        gap = leader_gap(own, rows)
        tau = 1. if gap is None else max(1e-6, gap/max(v_target, 1e-6))
        anchor = [v_target, math.log(tau)]
        x = [own['s_m']/100., own['v_mps']/30.,
             *[float(own['lane'] == lane) for lane in range(3)],
             v_target/30., anchor[1]/3.]
        for lane in range(3):
            for direction in (-1, 1):
                neighbours = [r for r in rows if r['vid'] != vid and r['lane'] == lane
                              and direction*(r['s_m']-own['s_m']) > 0]
                if neighbours:
                    r = min(neighbours, key=lambda r:(abs(r['s_m']-own['s_m']), r['vid']))
                    x.extend(((r['s_m']-own['s_m'])/100., r['v_mps']/30., float(r['is_cav']), 1.))
                else:
                    x.extend((0., 0., 0., 0.))
        inputs.append(x+rsu)
        anchors.append(anchor)
    return dict(members=members, x=inputs, rsu=rsu, anchors=anchors)


class SetPointPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.broadcast = nn.Sequential(nn.Linear(12,32), nn.Tanh(), nn.Linear(32,4))
        self.body = nn.Sequential(nn.Linear(FEATURES+4,64), nn.Tanh(), nn.Linear(64,64), nn.Tanh())
        self.mean = nn.Linear(64,2)
        self.window = nn.Linear(64,3)
        for head in (self.broadcast[-1], self.mean, self.window):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, context):
        x = torch.as_tensor(context['x'], dtype=torch.float32).reshape(-1,FEATURES)
        rsu = torch.as_tensor(context['rsu'], dtype=x.dtype)
        latent = self.broadcast(rsu).expand(len(x),-1)
        h = self.body(torch.cat((x,latent),dim=-1))
        anchor = torch.as_tensor(context['anchors'], dtype=x.dtype).reshape(-1,2)
        return anchor+self.mean(h), self.window(h).mean(0)

    @torch.no_grad()
    def sample(self, context, generator, sigma=SIGMA, deterministic=False):
        if not context['members']:
            raise ValueError('cannot sample an empty group')
        mean, logits = self(context)
        if deterministic:
            rho, sample = 0., mean
            index = int(logits.argmax())
        else:
            rho = float(torch.rand((), generator=generator))*.95
            shared = torch.randn((1,2), generator=generator)
            solo = torch.randn(mean.shape, generator=generator)
            noise = math.sqrt(rho)*shared+math.sqrt(1-rho)*solo
            sample = mean+noise*torch.as_tensor(sigma)
            index = int(torch.multinomial(logits.softmax(0),1,generator=generator))
        return dict(context=context, action=sample.tolist(), rho=rho, sigma=list(sigma),
                    window=WINDOWS[index], window_index=index)


def gaussian_log_prob(action, mean, rho, sigma):
    """Sherman-Morrison density for C=(1-rho)I+rho 11'; sum both dimensions."""
    if not 0 <= rho < 1:
        raise ValueError('nonsingular shared mixing required')
    scale = torch.as_tensor(sigma, dtype=mean.dtype, device=mean.device)
    if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
        raise ValueError('finite positive sigma required')
    z = (action.detach()-mean)/scale
    n = len(mean)
    if not n:
        raise ValueError('empty search density')
    a, b = 1-rho, 1-rho+n*rho
    quadratic = z.square().sum(0)/a-rho*z.sum(0).square()/(a*b)
    logdet = (n-1)*math.log(a)+math.log(b)
    return -.5*(quadratic+logdet+n*math.log(2*math.pi)+2*n*scale.log()).sum()


def log_prob(policy, sample):
    mean, logits = policy(sample['context'])
    action = torch.as_tensor(sample['action'],dtype=mean.dtype)
    return gaussian_log_prob(action,mean,sample['rho'],sample['sigma']) + \
        torch.log_softmax(logits,0)[sample['window_index']]


def decode(sample):
    """Numerical exp limit only; acceleration/speed constraints belong to plant safety."""
    return {vid:dict(speed=max(0.,float(a[0])), tau=math.exp(min(20.,max(-20.,float(a[1])))))
            for vid,a in zip(sample['context']['members'],sample['action'])}


def requests(targets, rows):
    requested = {}
    for own in rows:
        if own['vid'] not in targets or not ZONE[0] <= own['s_m'] <= ZONE[1]:
            continue
        target = targets[own['vid']]
        speed = target['speed']
        gap = leader_gap(own,rows)
        if gap is not None:
            speed = min(speed,gap/target['tau'])
        requested[own['vid']] = dict(target_mps=speed, target_tau_s=target['tau'],
                                     requested_raw=(speed-own['v_mps'])/TRACK_S)
    return requested
