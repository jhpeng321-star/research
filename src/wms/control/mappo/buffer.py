"""Rollout storage, GAE and return normalisation for the C4 MAPPO baseline.

Team-level credit: one shared team-reward stream and one centralised value per
episode (per-CAV rewards = team reward, shared), so the per-CAV advantage at a
control step is the team advantage broadcast to every acting CAV slot.  CAV
slots entering/leaving the episode are handled by the ``cav_mask``, not by
episode dones; an episode only ends at its fixed horizon (time-limit
truncation, bootstrapped) or on a reported termination (no bootstrap).

GAE recursion ported from ``archive/circles_era/traffic_shepherd/ppo_train.py``
(``compute_gae``); the Welford running-statistics pattern is ported from
``ppo_features.RunningNormalizer`` and applied to returns (value normalisation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import torch
from torch import Tensor

from wms.control.contract import CONTROL_DT_S


@dataclass
class EpisodeData:
    """One team episode on the CavEnv 1 s control grid.

    ``states`` holds T+1 WM-frame state dicts (batch dim 1): s_0..s_T, where
    s_T is the state after the last step (kept for the truncation bootstrap).
    ``actions``/``logp`` are the PRE-safety-layer policy action and its
    log-probability (the safety layer is part of the environment from the
    learner's point of view).
    """

    states: list[Mapping[str, Tensor]]
    actions: Tensor  # [T, K]
    logp: Tensor  # [T, K]
    cav_mask: Tensor  # [T, K] bool: slots that acted at t
    rewards: Tensor  # [T] team reward
    costs: Tensor  # [T] team cost_ttc exposure (>= 0)
    terminated: bool
    meta: dict = field(default_factory=dict)

    @property
    def steps(self) -> int:
        return int(self.rewards.shape[0])

    @property
    def sumo_seconds(self) -> float:
        return self.steps * CONTROL_DT_S


@dataclass
class FlatBatch:
    """Episodes of one iteration flattened frame-first, keeping segment info."""

    states: dict[str, Tensor]  # [N, K] per state key (T frames per episode)
    final_states: dict[str, Tensor]  # [E, K] one bootstrap state per episode
    actions: Tensor  # [N, K]
    logp: Tensor  # [N, K]
    cav_mask: Tensor  # [N, K]
    rewards: Tensor  # [N]
    costs: Tensor  # [N]
    terminated: Tensor  # [E] bool
    lengths: list[int]  # frames per episode

    @property
    def n_frames(self) -> int:
        return int(self.rewards.shape[0])

    @property
    def n_episodes(self) -> int:
        return len(self.lengths)


def flatten_episodes(episodes: Iterable[EpisodeData]) -> FlatBatch:
    episodes = list(episodes)
    if not episodes:
        raise ValueError("flatten_episodes needs at least one episode")
    keys = sorted(set().union(*[set(ep.states[0]) for ep in episodes]))
    frames = [frame for ep in episodes for frame in ep.states[: ep.steps]]
    return FlatBatch(
        states={k: torch.cat([f[k].detach().clone() for f in frames], dim=0) for k in keys},
        final_states={
            k: torch.cat([ep.states[-1][k].detach().clone() for ep in episodes], dim=0)
            for k in keys
        },
        actions=torch.cat([ep.actions for ep in episodes], dim=0),
        logp=torch.cat([ep.logp for ep in episodes], dim=0),
        cav_mask=torch.cat([ep.cav_mask for ep in episodes], dim=0),
        rewards=torch.cat([ep.rewards for ep in episodes], dim=0),
        costs=torch.cat([ep.costs for ep in episodes], dim=0),
        terminated=torch.tensor([ep.terminated for ep in episodes], dtype=torch.bool),
        lengths=[ep.steps for ep in episodes],
    )


def compute_gae(
    rewards: Tensor, values: Tensor, *, gamma: float, lam: float, bootstrap_value: Tensor | float
) -> tuple[Tensor, Tensor]:
    """GAE over one episode segment; ``values`` = V(s_0)..V(s_{T-1}).

    ``bootstrap_value`` is V(s_T) for time-limit truncation and 0 for
    termination.  Returns (advantages [T], returns [T] = advantages + values).
    """
    rewards = rewards.detach().float()
    values = values.detach().float()
    if rewards.shape != values.shape or rewards.dim() != 1:
        raise ValueError("compute_gae needs matching 1-D rewards/values")
    advantages = torch.zeros_like(rewards)
    last = torch.zeros((), dtype=rewards.dtype)
    next_value = torch.as_tensor(bootstrap_value, dtype=rewards.dtype)
    for t in range(rewards.shape[0] - 1, -1, -1):
        delta = rewards[t] + gamma * next_value - values[t]
        last = delta + gamma * lam * last
        advantages[t] = last
        next_value = values[t]
    return advantages, advantages + values


def discounted_returns(rewards: Tensor, gamma: float, bootstrap: Tensor | float = 0.0) -> Tensor:
    """Discounted return-to-go of one segment (used for the cost_ttc stream)."""
    rewards = rewards.detach().float()
    out = torch.zeros_like(rewards)
    run = torch.as_tensor(bootstrap, dtype=rewards.dtype)
    for t in range(rewards.shape[0] - 1, -1, -1):
        run = rewards[t] + gamma * run
        out[t] = run
    return out


def episode_gae(
    batch: FlatBatch, values_raw: Tensor, finals_raw: Tensor, *, gamma: float, lam: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-iteration GAE over all segments of ``batch``.

    ``values_raw``/``finals_raw`` are RAW (unscaled) centralised values of the
    T frames and the E terminal states.  Returns (advantages [N], returns [N],
    cost_returns [N]); the cost stream gets no truncation bootstrap
    (conservative: unseen future TTC exposure is not credited away).
    """
    n = batch.n_frames
    adv = torch.zeros(n)
    ret = torch.zeros(n)
    cost = torch.zeros(n)
    start = 0
    for e, length in enumerate(batch.lengths):
        sl = slice(start, start + length)
        boot = torch.zeros(()) if bool(batch.terminated[e]) else finals_raw[e]
        a, r = compute_gae(
            batch.rewards[sl], values_raw[sl], gamma=gamma, lam=lam, bootstrap_value=boot
        )
        adv[sl], ret[sl] = a, r
        cost[sl] = discounted_returns(batch.costs[sl], gamma)
        start += length
    return adv, ret, cost


class ReturnScaler:
    """Welford running mean/std of returns; the critic predicts scaled returns.

    ``scale``/``unscale`` are identity until at least two returns were seen.
    Stats are frozen while an iteration's PPO update runs and refreshed with
    that iteration's returns afterwards, so rollout values and update targets
    always agree on one scaler state.
    """

    def __init__(self, min_count: int = 2, eps: float = 1e-4):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min_count = min_count
        self.eps = eps

    @property
    def std(self) -> float:
        if self.count < self.min_count:
            return 1.0
        var = self.m2 / (self.count - 1)
        return float(max(var, 0.0) ** 0.5) if var > self.eps * self.eps else 1.0

    def update(self, returns: Tensor) -> None:
        values = returns.detach().float().reshape(-1)
        batch = int(values.numel())
        if batch == 0:
            return
        b_mean = values.mean()
        b_m2 = ((values - b_mean) ** 2).sum()
        if self.count == 0:
            self.count, self.mean, self.m2 = batch, float(b_mean), float(b_m2)
            return
        total = self.count + batch
        delta = float(b_mean) - self.mean
        new_mean = self.mean + delta * batch / total
        self.m2 = self.m2 + float(b_m2) + delta * delta * self.count * batch / total
        self.mean = new_mean
        self.count = total

    def scale(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std

    def unscale(self, x: Tensor) -> Tensor:
        return x * self.std + self.mean

    def state_dict(self) -> dict:
        return {
            "count": self.count,
            "mean": self.mean,
            "m2": self.m2,
            "min_count": self.min_count,
            "eps": self.eps,
        }

    @classmethod
    def from_state(cls, state: Mapping) -> "ReturnScaler":
        out = cls(min_count=state["min_count"], eps=state["eps"])
        out.count, out.mean, out.m2 = state["count"], state["mean"], state["m2"]
        return out
