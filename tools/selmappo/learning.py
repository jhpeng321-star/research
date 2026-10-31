"""Categorical selector PPO, reusing the existing MAPPO optimization primitives."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from wms.control.mappo.buffer import ReturnScaler, compute_gae
from wms.control.mappo.ppo import (PPOConfig, clipped_surrogate, clipped_value_loss,
                                   sample_normalized)

from .observations import ACTOR_DIM, SHARED_DIM, Observation
from .protocol import LIBRARY, TIMES


def mlp(n, out):
    return nn.Sequential(nn.Linear(n, 128), nn.Tanh(), nn.Linear(128, 128),
                         nn.Tanh(), nn.Linear(128, out))


class Selector(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = mlp(ACTOR_DIM, len(LIBRARY))
        self.critic = mlp(SHARED_DIM, 1)

    def distribution(self, actors):
        return Categorical(logits=self.actor(actors))

    @torch.no_grad()
    def act(self, observation, *, deterministic=False, generator=None):
        x = torch.tensor(observation.actors, dtype=torch.float32).reshape(-1, ACTOR_DIM)
        if not len(x):
            return {}, torch.empty(0)
        dist = self.distribution(x)
        actions = (dist.logits.argmax(-1) if deterministic else
                   torch.multinomial(dist.probs, 1, generator=generator).squeeze(-1))
        return dict(zip(observation.keys, actions.tolist())), dist.log_prob(actions)


@dataclass
class Transition:
    observation: Observation
    actions: dict
    old_logp: torch.Tensor
    reward: float


class Updater:
    def __init__(self, model):
        self.model = model
        self.cfg = PPOConfig(gamma=1., gae_lambda=1., epochs=4, minibatch_frames=256)
        self.actor_opt = torch.optim.Adam(model.actor.parameters(), lr=self.cfg.lr_actor)
        self.critic_opt = torch.optim.Adam(model.critic.parameters(), lr=self.cfg.lr_critic)
        self.scaler = ReturnScaler()

    def update(self, episodes):
        if not episodes or any(len(ep) != len(TIMES) for ep in episodes):
            raise ValueError('PPO requires complete thirty-window episodes')
        frames = [t for episode in episodes for t in episode]
        shared = torch.tensor([t.observation.shared for t in frames], dtype=torch.float32)
        counts = torch.tensor([len(t.observation.keys) for t in frames])
        with torch.no_grad():
            old_value = self.model.critic(shared).squeeze(-1)
            raw_value = self.scaler.unscale(old_value)
        advantages, returns, offset = [], [], 0
        for episode in episodes:
            a, r = compute_gae(torch.tensor([t.reward for t in episode]), raw_value[offset:offset+len(TIMES)],
                               gamma=1., lam=1., bootstrap_value=0.)
            advantages.append(a)
            returns.append(r)
            offset += len(TIMES)
        returns = torch.cat(returns)
        adv = sample_normalized(torch.cat(advantages), counts) if counts.sum() else torch.zeros(len(frames))
        target = self.scaler.scale(returns)
        losses, stopped = [], False
        for _ in range(self.cfg.epochs):
            for indices in torch.randperm(len(frames)).split(self.cfg.minibatch_frames):
                value = self.model.critic(shared[indices]).squeeze(-1)
                loss = self.cfg.value_coef * clipped_value_loss(value, old_value[indices], target[indices],
                                                               self.cfg.clip_value_eps).mean()
                xs, actions, logps, aa = [], [], [], []
                for i in indices.tolist():
                    t = frames[i]
                    if t.observation.keys:
                        xs.extend(t.observation.actors)
                        actions.extend(t.actions[k] for k in t.observation.keys)
                        logps.append(t.old_logp.detach())
                        aa.extend([adv[i]] * len(t.observation.keys))
                kl = torch.tensor(0.)
                if xs:
                    dist = self.model.distribution(torch.tensor(xs, dtype=torch.float32))
                    old = torch.cat(logps)
                    new = dist.log_prob(torch.tensor(actions))
                    ratio = (new-old).clamp(-20, 20).exp()
                    loss = loss-clipped_surrogate(ratio, torch.stack(aa), self.cfg.clip_eps).mean()
                    loss = loss-self.cfg.entropy_coef*dist.entropy().mean()
                    kl = (old-new).mean().detach()
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite PPO loss')
                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                for module in (self.model.actor, self.model.critic):
                    nn.utils.clip_grad_norm_(module.parameters(), self.cfg.max_grad_norm, error_if_nonfinite=True)
                self.actor_opt.step()
                self.critic_opt.step()
                losses.append(float(loss.detach()))
                if kl > 1.5*self.cfg.target_kl:
                    stopped = True
                    break
            if stopped:
                break
        self.scaler.update(returns)
        return dict(loss=sum(losses)/len(losses), frames=len(frames), samples=int(counts.sum()),
                    kl_stopped=stopped, mean_return=float(returns.mean()))
