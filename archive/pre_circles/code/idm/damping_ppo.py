"""Time-major PPO for a shared local actor and a training-only team critic.

The actor loss sums active CAV contributions per global timestep; padding,
birth/death and active-population changes do not change the reward definition.
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from idm.damping_credit import TeamCostScales, TeamIntervalCost, team_advantages, team_reward


@dataclass(frozen=True)
class PPOContract:
    learning_rate: float = 1e-4
    clip_ratio: float = 0.2
    epochs: int = 2
    time_minibatch: int = 128
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    gamma: float = 1.0
    gae_lambda: float = 1.0
    service_budget_s_per_vehicle: float = 2.0
    service_violation_penalty: float = 2.0

    def manifest(self):
        return {
            "schema": "sensor_local_team_ppo/2",
            **asdict(self),
            "advantage_time": "global_fixed_clock",
            "actor_loss": "sum_agents_mean_times",
            "advantage_normalization": "none_preserve_terminal_episode_cost",
            "reward": "same_requests_policy_minus_cached_native_cost",
        }


class TeamCritic(nn.Module):
    def __init__(self, observation_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, states):
        if not torch.isfinite(states).all():
            raise ValueError("nonfinite critic input")
        return self.net(states).squeeze(-1)


def centered_reward_series(
    rollout: dict, native: dict, scales: TeamCostScales, contract: PPOContract = PPOContract()
) -> tuple[np.ndarray, dict]:
    own, reference = rollout["summary"], native["summary"]
    for field in (
        "scenario_identity",
        "seed",
        "population",
        "global_control_steps",
        "team_terminal",
    ):
        if own[field] != reference[field]:
            raise ValueError(f"paired reference mismatch: {field}")
    if (
        len(rollout["intervals"]) != own["global_control_steps"]
        or len(native["intervals"]) != own["global_control_steps"]
    ):
        raise ValueError("incomplete reward series")
    if scales.requested_population != own["population"]:
        raise ValueError("reward denominator differs from the requested population")
    parts = {key: [] for key in ("braking", "energy", "service", "hard_events", "reward")}
    for actual, baseline in zip(rollout["intervals"], native["intervals"]):
        actual_r = team_reward(TeamIntervalCost(**actual), scales)
        baseline_r = team_reward(TeamIntervalCost(**baseline), scales)
        for field in parts:
            parts[field].append(actual_r[field] - baseline_r[field])
    delay_delta = (
        own["system_seconds_per_requested_vehicle"]
        - reference["system_seconds_per_requested_vehicle"]
    )
    arrival_delta = own["arrived"] - reference["arrived"]
    violation = bool(
        own["team_terminal"]
        and (delay_delta > contract.service_budget_s_per_vehicle or arrival_delta < -1)
    )
    rewards = np.asarray(parts["reward"], dtype=np.float64)
    if violation:
        rewards[-1] -= contract.service_violation_penalty
    totals = {name: float(sum(values)) for name, values in parts.items() if name != "reward"}
    expected = sum(totals.values()) - contract.service_violation_penalty * violation
    if not np.isfinite(rewards).all() or not np.isclose(rewards.sum(), expected):
        raise ValueError("reward/component audit failed")
    return rewards, {
        "components": totals,
        "reward_sum": float(rewards.sum()),
        "service_violation": violation,
        "system_time_delta_s_per_vehicle": delay_delta,
        "arrived_delta": arrival_delta,
    }


def prepare_batch(
    rollout: dict, native: dict, scales: TeamCostScales, contract: PPOContract = PPOContract()
) -> dict:
    if not rollout["summary"]["team_terminal"]:
        raise ValueError("this trainer requires complete fixed-clock team episodes")
    rewards, audit = centered_reward_series(rollout, native, scales, contract)
    ends = np.zeros(len(rewards), dtype=bool)
    ends[-1] = True
    advantages, returns = team_advantages(
        rewards, rollout["old_values"], ends, gamma=contract.gamma, gae_lambda=contract.gae_lambda
    )
    times = np.asarray(rollout["actor_times"], dtype=np.int64)
    if times.ndim != 1 or (len(times) and (times.min() < 0 or times.max() >= len(rewards))):
        raise ValueError("invalid actor-to-team-time mapping")
    if any(
        len(rollout[key]) != len(times)
        for key in ("actor_observations", "actions", "old_logp", "actor_ids")
    ):
        raise ValueError("actor sample alignment mismatch")
    # One full episode is one update batch. Subtracting its time-average
    # advantage would erase constant terminal failure costs from every actor
    # decision. The frozen native cost scales and learned critic already
    # provide normalization/baselining; retain the actual signed advantages.
    return {
        **rollout,
        "rewards": rewards,
        "returns": returns.astype(np.float32),
        "team_advantages": advantages.astype(np.float32),
        "actor_advantages": advantages[times].astype(np.float32),
        "reward_audit": audit,
    }


def update_ppo(
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    batch: dict,
    *,
    rng: np.random.Generator,
    contract: PPOContract = PPOContract(),
) -> dict:
    actor.train()
    critic.train()
    n_times = len(batch["rewards"])
    observations = torch.as_tensor(batch["actor_observations"])
    actions = torch.as_tensor(batch["actions"])
    old_logp = torch.as_tensor(batch["old_logp"])
    advantages = torch.as_tensor(batch["actor_advantages"])
    states = torch.as_tensor(batch["critic_observations"])
    returns = torch.as_tensor(batch["returns"])
    times = np.asarray(batch["actor_times"])
    history = []
    actor_before = torch.cat([p.detach().flatten() for p in actor.parameters()]).clone()
    early_stop = False
    for _ in range(contract.epochs):
        permutation = rng.permutation(n_times)
        for start in range(0, n_times, contract.time_minibatch):
            selected_times = permutation[start : start + contract.time_minibatch]
            index = np.flatnonzero(np.isin(times, selected_times))
            predicted = critic(states[selected_times])
            critic_loss = 0.5 * (predicted - returns[selected_times]).square().mean()
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_grad = nn.utils.clip_grad_norm_(
                critic.parameters(), contract.max_grad_norm, error_if_nonfinite=True
            )
            critic_optimizer.step()
            row = {
                "critic_loss": float(critic_loss.detach()),
                "critic_grad_norm": float(critic_grad),
            }
            if len(index):
                logp = actor.log_prob(observations[index], actions[index])
                log_ratio = logp - old_logp[index]
                ratio = log_ratio.exp()
                loss = -torch.minimum(
                    ratio * advantages[index],
                    ratio.clamp(1 - contract.clip_ratio, 1 + contract.clip_ratio)
                    * advantages[index],
                ).sum() / len(selected_times)
                kl = (ratio - 1 - log_ratio).mean()
                if not torch.isfinite(loss) or not torch.isfinite(kl):
                    raise FloatingPointError("nonfinite actor update")
                if float(kl.detach()) > contract.target_kl:
                    early_stop = True
                    break
                actor_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                actor_grad = nn.utils.clip_grad_norm_(
                    actor.parameters(), contract.max_grad_norm, error_if_nonfinite=True
                )
                actor_optimizer.step()
                row.update(
                    actor_loss=float(loss.detach()),
                    approximate_kl=float(kl.detach()),
                    actor_grad_norm=float(actor_grad),
                    actor_samples=len(index),
                )
            history.append(row)
        if early_stop:
            break
    actor_after = torch.cat([p.detach().flatten() for p in actor.parameters()])
    result = {
        key: float(np.mean([row[key] for row in history if key in row]))
        for key in sorted({k for row in history for k in row})
    }
    result.update(
        actor_parameter_change_l2=float(torch.linalg.vector_norm(actor_after - actor_before)),
        kl_early_stop=early_stop,
        optimizer_minibatches=len(history),
        global_times=n_times,
        actor_samples=len(times),
    )
    if not all(np.isfinite(value) for value in result.values()):
        raise FloatingPointError("nonfinite learner metrics")
    return result
