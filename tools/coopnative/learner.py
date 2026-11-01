"""Full-episode long-credit PPO and complete deterministic checkpoint/resume."""
import math
import random
import numpy as np
import torch
from .policy import Policy

GAMMA = math.exp(-10/600)
LAMBDA = math.exp(-10/1200)  # Combined GAE e-folding 400 s, beyond 300 s.
REWARD_UNIT = 10000.


def advantages(steps, bootstrap=0.):
    out, tail, next_value = [], 0., bootstrap
    for step in reversed(steps):
        delta = step['reward']/REWARD_UNIT + GAMMA*next_value-step['sample']['value']
        tail = delta + GAMMA*LAMBDA*tail
        out.append((tail,tail+step['sample']['value']))
        next_value = step['sample']['value']
    return list(reversed(out))


def update(policy, optimizer, episodes, generator, epochs=4):
    records = [(s,a,r) for ep in episodes for s,(a,r) in zip(ep['steps'],advantages(ep['steps']))]
    adv = torch.tensor([a for _,a,_ in records])
    adv = (adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-8)
    metrics = []
    for _ in range(epochs):
        for indices in torch.randperm(len(records),generator=generator).split(128):
            logps, values = zip(*(policy.score(records[int(i)][0]['sample']) for i in indices))
            logps, values = torch.stack(logps), torch.stack(values)
            old = torch.tensor([records[int(i)][0]['sample']['logp'] for i in indices])
            returns = torch.tensor([records[int(i)][2] for i in indices])
            oldv = torch.tensor([records[int(i)][0]['sample']['value'] for i in indices])
            ratio = (logps-old).exp()
            a = adv[indices]
            actor = -torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a).mean()
            clipped = oldv+(values-oldv).clamp(-.2,.2)
            value_loss = torch.maximum((values-returns).square(),(clipped-returns).square()).mean()
            loss = actor+.5*value_loss
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite PPO objective')
            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(),.5,error_if_nonfinite=True)
            optimizer.step()
            # Joint KL, not independent per-member pseudo likelihood.
            kl = float(((ratio-1)-(logps-old)).mean().detach())
            metrics.append(dict(loss=float(loss.detach()),kl=kl,gradient_norm=float(norm)))
            if kl > .045:
                return metrics
    return metrics


def save(path, policy, optimizer, generator, update_id, charged, **extra):
    path.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive immutable checkpoint, never overwrite/delete a prior milestone.
    with path.open('xb') as stream:
        torch.save(dict(schema='coopnative/1',policy=policy.state_dict(),optimizer=optimizer.state_dict(),
                        generator=generator.get_state(),torch_rng=torch.get_rng_state(),
                        numpy_rng=np.random.get_state(),python_rng=random.getstate(),
                        update=update_id,charged=charged,extra=extra),stream)
        stream.flush()
        import os
        os.fsync(stream.fileno())


def load(path):
    payload = torch.load(path,map_location='cpu',weights_only=False)
    if payload['schema'] != 'coopnative/1':
        raise ValueError('native-only checkpoint required')
    policy = Policy()
    policy.load_state_dict(payload['policy'])
    optimizer = torch.optim.Adam(policy.parameters(),lr=3e-4)
    optimizer.load_state_dict(payload['optimizer'])
    generator = torch.Generator()
    generator.set_state(payload['generator'])
    torch.set_rng_state(payload['torch_rng'])
    np.random.set_state(payload['numpy_rng'])
    random.setstate(payload['python_rng'])
    return policy, optimizer, generator, payload
