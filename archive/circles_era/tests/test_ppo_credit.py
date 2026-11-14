"""Delayed-credit verification through the REAL trainer functions.

Places an otherwise identical penalty at +0 s, +100 s, +450 s and near the
2400 s horizon after an early wall decision, and measures the early
decision's advantage and critic target under (a) the superseded
GAE(lambda=0.95) trace and (b) the registered Monte Carlo return-with-value-
baseline estimator - both through traffic_shepherd.ppo_train's own functions
and the actual update() path (masks and stored log probabilities included).

Decision cadence is 5 s, so GAE steps are DECISION TRANSITIONS: a 100 s
delay is 20 steps, 450 s is 90 steps. Rewards attached to deterministic
assembly/exit windows are already folded into the preceding decision's
window by the environment, so every physical tick reward is accounted for
exactly once at decision granularity.
"""

import pytest
import torch

from traffic_shepherd import ppo_contract as C
from traffic_shepherd.ppo_features import RunningNormalizer
from traffic_shepherd.ppo_net import ActorCritic, seed_everything
from traffic_shepherd.ppo_train import (build_batch, compute_gae,
                                        compute_monte_carlo_advantages,
                                        compute_monte_carlo_returns, update)

LENGTH = 480            # decisions per 2400 s episode at the 5 s cadence
EARLY = 10              # early wall decision index (t = 50 s)
PENALTY = -1.0
DELAY_INDICES = {"+0s": 10, "+100s": 30, "+450s": 100, "+2345s(near end)": 479}


def synthetic_episode(penalty_index, length=LENGTH):
    # each decision carries a distinctive observation (its normalized time),
    # so the critic CAN represent different return-to-go per state
    rows = []
    for i in range(length):
        marker = i / length
        rows.append(dict(action=0, logp=-0.693147, value=0.0,
                         reward=PENALTY if i == penalty_index else 0.0,
                         done=i == length - 1,
                         mask=[True, True],
                         obs_stacked_raw=[marker] * C.OBS_DIM))
    return rows


def zeros_like_rollout(rollout):
    return [dict(r, value=0.0) for r in rollout]


@pytest.mark.parametrize("delay,delay_index", sorted(DELAY_INDICES.items(), key=lambda kv: kv[1]))
def test_gae_before_attenuates_delayed_credit(delay, delay_index):
    """BEFORE: with an uninformative critic, the GAE trace attenuates the
    advantage at the causing decision by lambda^(delay/5s)."""
    rollout = synthetic_episode(delay_index)
    rewards = torch.tensor([r["reward"] for r in rollout])
    dones = [r["done"] for r in rollout]
    values = torch.zeros(len(rollout))
    advantages = compute_gae(rewards, dones, values, gamma=1.0, lam=0.95)
    expected_attenuation = 0.95 ** (delay_index - EARLY)
    assert advantages[EARLY] == pytest.approx(-expected_attenuation, rel=1e-4)
    # the critic target inherits the same attenuation (returns = A + V)
    assert (advantages[EARLY] + values[EARLY]) == pytest.approx(-expected_attenuation, rel=1e-4)


@pytest.mark.parametrize("delay,delay_index", sorted(DELAY_INDICES.items(), key=lambda kv: kv[1]))
def test_monte_carlo_after_preserves_exact_credit(delay, delay_index):
    """AFTER: the registered estimator gives the causing decision the EXACT
    remaining physical return as critic target and a full-magnitude
    advantage against the baseline, at every delay."""
    rollout = synthetic_episode(delay_index)
    rewards = torch.tensor([r["reward"] for r in rollout])
    dones = [r["done"] for r in rollout]
    values = torch.zeros(len(rollout))
    returns = compute_monte_carlo_returns(rewards, dones, values)
    advantages = compute_monte_carlo_advantages(rewards, dones, values)
    assert advantages[EARLY] == pytest.approx(PENALTY, abs=1e-12)
    assert returns[EARLY] == pytest.approx(PENALTY, abs=1e-12)   # critic target
    assert returns[0] == pytest.approx(PENALTY, abs=1e-12)
    # the complete physical return is accounted for exactly once
    assert float(rewards.sum()) == pytest.approx(PENALTY, abs=1e-12)
    assert float(returns[0]) == pytest.approx(float(rewards.sum()), abs=1e-12)


def test_truncated_segment_without_bootstrap_is_rejected():
    rewards = torch.tensor([0.0, -1.0])
    dones = [False, False]
    with pytest.raises(ValueError, match="truncated"):
        compute_monte_carlo_returns(rewards, dones, torch.zeros(2))


def test_mixed_decision_counts_across_policies_in_one_batch():
    """Two policies with different decision counts form correct per-segment
    returns (terminal at each episode's done row)."""
    ep_a = synthetic_episode(50, length=100)    # short policy episode
    ep_b = synthetic_episode(300, length=LENGTH)
    rollouts = [ep_a, ep_b]
    normalizer = RunningNormalizer(C.OBS_DIM)
    batch, lengths = build_batch(rollouts, normalizer)
    returns = compute_monte_carlo_returns(batch["rewards"], batch["dones"], batch["values_old"])
    assert lengths == [100, LENGTH]
    assert float(returns[0]) == pytest.approx(PENALTY, abs=1e-12)      # ep_a own total
    assert float(returns[100]) == pytest.approx(PENALTY, abs=1e-12)    # ep_b own total
    assert float(returns[99]) == pytest.approx(0.0, abs=1e-12)         # no cross-episode leak


def test_update_uses_stored_masks_and_log_probabilities():
    seed_everything(0)
    net = ActorCritic()
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
    rollout = synthetic_episode(30)
    for i, row in enumerate(rollout):
        row["mask"] = [True, i % 2 == 0]   # action 1 unavailable on odd rows
    normalizer = RunningNormalizer(C.OBS_DIM)
    normalizer.update([0.01] * C.OBS_DIM)
    # the stored log probabilities/values are the behavior policy's own
    from traffic_shepherd.ppo_worker import stacked_obs
    with torch.no_grad():
        for i, row in enumerate(rollout):
            obs = torch.tensor([normalizer.normalize(
                stacked_obs([r["obs_stacked_raw"] for r in rollout], i))],
                dtype=torch.float32)
            d = net.distribution(obs, torch.tensor([row["mask"]]))
            row["logp"] = float(d.log_prob(torch.tensor([row["action"]])))
            row["value"] = float(net.value(obs))
    batch, _ = build_batch([rollout], normalizer)
    assert batch["masks"].dtype == torch.bool
    # stored log probabilities are consumed as the behavior log-prob
    with torch.no_grad():
        logp_recomputed, _entropy, _value = net.evaluate(
            batch["obs"][:1], batch["masks"][:1], batch["actions"][:1])
    assert float(logp_recomputed[0]) == pytest.approx(batch["logp_old"][0].item(), abs=1e-5)
    diagnostics = update(net, optimizer, batch, [len(rollout)])
    assert diagnostics["steps"] == len(rollout)
    # masked actions have exactly zero probability under the update path
    with torch.no_grad():
        distribution = net.distribution(batch["obs"][1:2], batch["masks"][1:2])
    assert float(distribution.probs[0, 1]) == 0.0


def test_critic_target_reaches_early_state_exactly_and_is_learnable():
    """Through build_batch + the registered estimator, the EARLY state's
    critic target is the exact delayed return; and through the real update,
    the critic moves toward it (convergence speed is optimizer dynamics,
    not the credit contract)."""
    rollout = synthetic_episode(100)   # penalty 450 s after the early decision
    normalizer = RunningNormalizer(C.OBS_DIM)
    normalizer.update([0.01] * C.OBS_DIM)
    batch, _ = build_batch([rollout], normalizer)
    target_early = float((compute_monte_carlo_advantages(
        batch["rewards"], batch["dones"], batch["values_old"])
        + batch["values_old"])[EARLY])
    assert target_early == pytest.approx(-1.0, abs=1e-12)
    seed_everything(3)
    net = ActorCritic()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    with torch.no_grad():
        value_before = float(net.value(batch["obs"][EARLY:EARLY + 1])[0])
    for _ in range(60):
        update(net, optimizer, batch, [len(rollout)])
    with torch.no_grad():
        value_after = float(net.value(batch["obs"][EARLY:EARLY + 1])[0])
    assert value_after < value_before          # moved toward -1
    assert value_after < 0.0                   # crossed toward the target


def test_ppo_update_moves_early_action_logp_with_correct_sign():
    """Sign-correct credit through the real update path: a harmful delayed
    consequence SUPPRESSES the causing action; a beneficial one REINFORCES
    it. Masks and stored log probabilities are consumed as stored."""
    rollout_harm = synthetic_episode(100)
    rollout_good = synthetic_episode(100)
    for rows, reward in ((rollout_harm, -1.0), (rollout_good, +1.0)):
        for row in rows:
            row["action"] = 1 if row is rows[EARLY] else 0
            row["mask"] = [True, True]
            row["reward"] = reward if row is rows[100] else 0.0
    results = {}
    for name, rows in (("harmful", rollout_harm), ("beneficial", rollout_good)):
        seed_everything(4)
        net = ActorCritic()
        optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
        normalizer = RunningNormalizer(C.OBS_DIM)
        normalizer.update([0.01] * C.OBS_DIM)
        batch, _ = build_batch([rows], normalizer)
        with torch.no_grad():
            before, _e, _v = net.evaluate(batch["obs"][EARLY:EARLY + 1],
                                          batch["masks"][EARLY:EARLY + 1],
                                          batch["actions"][EARLY:EARLY + 1])
        for _ in range(30):
            update(net, optimizer, batch, [len(rows)])
        with torch.no_grad():
            after, _e, _v = net.evaluate(batch["obs"][EARLY:EARLY + 1],
                                         batch["masks"][EARLY:EARLY + 1],
                                         batch["actions"][EARLY:EARLY + 1])
        results[name] = (float(before), float(after))
    assert results["harmful"][1] < results["harmful"][0] - 1e-3   # suppressed
    assert results["beneficial"][1] > results["beneficial"][0] + 1e-3  # reinforced
