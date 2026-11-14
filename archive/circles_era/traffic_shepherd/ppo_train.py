"""Bounded PPO trainer: 32 chargeable episodes, one declared configuration.

The trainer spawns one worker process per episode (limited parallelism,
episode assignment fixed by index, sampling seeds fixed by index), collects
on-policy rollouts from the frozen round-start checkpoint, then performs the
registered PPO update. No development outcome is ever consulted; the
evaluated checkpoint is the final completed update.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from . import ppo_contract as C
from .ppo_features import RunningNormalizer
from .ppo_net import ActorCritic, seed_everything

ROOT = Path(__file__).resolve().parents[1]


def load_rollouts(training_dir, episode_indices):
    rows = []
    for index in episode_indices:
        path = Path(training_dir) / f"episode_{index:03d}" / "rollout.json"
        rollout = json.loads(path.read_text(encoding="utf-8"))
        if any(("reward" not in r) or (r["reward"] != r["reward"]) for r in rollout):
            raise ValueError(f"non-finite or missing reward in episode {index}")
        rows.append(rollout)
    return rows


def build_batch(rollouts, normalizer):
    obs, actions, logp_old, values, rewards, dones, masks = [], [], [], [], [], [], []
    lengths = []
    for rollout in rollouts:
        for row in rollout:
            stacked = row.get("obs_stacked_raw")
            if stacked is None:
                stacked = ppo_features_stacked(rollout, row)
            obs.append(normalizer.normalize(stacked))
            actions.append(row["action"])
            logp_old.append(row["logp"])
            values.append(row["value"])
            rewards.append(row["reward"])
            dones.append(bool(row.get("done")))
            masks.append(list(row["mask"]))
        lengths.append(len(rollout))
    tensors = dict(
        obs=torch.tensor(obs, dtype=torch.float32),
        actions=torch.tensor(actions, dtype=torch.long),
        logp_old=torch.tensor(logp_old, dtype=torch.float32),
        values_old=torch.tensor(values, dtype=torch.float32),
        rewards=torch.tensor(rewards, dtype=torch.float32),
        dones=torch.tensor(dones),
        masks=torch.tensor(masks, dtype=torch.bool))
    return tensors, lengths


def ppo_features_stacked(rollout, row):
    """Fallback stacking when the worker did not embed the stacked vector."""
    index = rollout.index(row)
    history = [r["features"] for r in rollout]
    parts = []
    for k in range(C.HISTORY_STACK - 1, -1, -1):
        j = index - k
        parts.append(history[j] if j >= 0 else [0.0] * len(C.FEATURE_NAMES))
    return [x for part in parts for x in part]


def compute_monte_carlo_returns(rewards, dones, values_old=None):
    """Undiscounted return-to-go per episode segment (gamma = 1).

    The pilot's episodes always terminate at the registered 2400 s horizon
    (no truncation), so a segment whose final row is not marked done cannot
    be valued and is rejected instead of silently bootstrapped. This
    estimator gives every decision the EXACT remaining physical return as
    its critic target and an exact advantage against the value baseline,
    independent of how far ahead the consequence lies.
    """
    dones = [bool(d) for d in dones]
    if dones and not dones[-1]:
        raise ValueError("episode segment is truncated without bootstrap; "
                         "cannot value the tail")
    returns = torch.zeros_like(rewards)
    run = 0.0
    for t in reversed(range(len(rewards))):
        run = float(rewards[t]) + (0.0 if dones[t] else run)
        returns[t] = run
    return returns


def compute_monte_carlo_advantages(rewards, dones, values_old):
    """Exact return-to-go minus the value baseline (the learned critic)."""
    return compute_monte_carlo_returns(rewards, dones, values_old) - values_old


def compute_gae(rewards, dones, values_old, gamma, lam):
    dones = [bool(d) for d in dones]
    advantages = torch.zeros_like(rewards)
    last = 0.0
    for t in reversed(range(len(rewards))):
        next_value = 0.0 if dones[t] or t + 1 >= len(rewards) else float(values_old[t + 1])
        delta = rewards[t] + gamma * next_value - float(values_old[t])
        last = delta + gamma * lam * (0.0 if dones[t] else 1.0) * last
        advantages[t] = last
    return advantages


def update(net, optimizer, batch, lengths, advantage_fn=None):
    cfg = C.PPO_CONFIG
    # Advantage estimator: undiscounted Monte Carlo return with a value
    # baseline (registered as advantage_estimator in the contract). GAE with
    # lambda=0.95 attenuates direct credit by 0.95^steps; at the registered
    # cadence (5 s) a consequence 450 s late would reach the causing decision
    # with ~1% of its advantage and a horizon-distant consequence with ~1e-8,
    # so the lambda-trace estimator fails the pilot's credit contract.
    advantage_fn = advantage_fn or compute_monte_carlo_advantages
    advantages = advantage_fn(batch["rewards"], batch["dones"], batch["values_old"])
    returns = advantages + batch["values_old"]
    # batch normalization of advantages is skipped when it would degenerate
    # (a single row or zero variance produces NaN via torch.std)
    if advantages.numel() > 1 and bool(torch.isfinite(advantages.std())):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    diagnostics = dict(pi_loss=0.0, value_loss=0.0, entropy=0.0, clip_fraction=0.0,
                       approx_kl=0.0, steps=int(len(batch["rewards"])))
    count = 0
    index = torch.randperm(len(batch["rewards"]),
                           generator=torch.Generator().manual_seed(cfg["optimizer_seed"]))
    for _ in range(cfg["epochs"]):
        for start in range(0, len(index), cfg["minibatch_size"]):
            mb = index[start:start + cfg["minibatch_size"]]
            logp, entropy, value = net.evaluate(batch["obs"][mb], batch["masks"][mb],
                                                batch["actions"][mb])
            ratio = torch.exp(logp - batch["logp_old"][mb])
            surrogate = ratio * advantages[mb]
            clipped = torch.clamp(ratio, 1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * advantages[mb]
            pi_loss = -torch.min(surrogate, clipped).mean()
            value_loss = (value - returns[mb]).pow(2).mean()
            loss = pi_loss + cfg["value_coef"] * value_loss - cfg["entropy_coef"] * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["max_grad_norm"])
            optimizer.step()
            with torch.no_grad():
                diagnostics["pi_loss"] += float(pi_loss)
                diagnostics["value_loss"] += float(value_loss)
                diagnostics["entropy"] += float(entropy.mean())
                diagnostics["approx_kl"] += float((batch["logp_old"][mb] - logp).mean())
                diagnostics["clip_fraction"] += float(
                    ((ratio - 1).abs() > cfg["clip_eps"]).float().mean())
                count += 1
    for key in ("pi_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"):
        diagnostics[key] /= max(1, count)
    return diagnostics


def _episode_complete(out):
    """A previously completed episode is reused, never re-simulated.

    The authoritative completion marker is the manifest status (the paired
    summary schema carries no status field)."""
    out = Path(out)
    summary, manifest = out / "summary.json", out / "manifest.json"
    if not summary.exists() or not manifest.exists():
        return False
    try:
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        json.loads(summary.read_text(encoding="utf-8"))
        return (manifest_data.get("status") == "complete"
                and (out / "rollout.json").exists())
    except Exception:
        return False


def train(registration_path, parallel=1):
    registration = json.loads(Path(registration_path).read_text(encoding="utf-8"))
    campaign = Path(registration["campaign_root"])
    training_dir = campaign / "training"
    cfg = C.PPO_CONFIG
    seed_everything(cfg["optimizer_seed"])
    net = ActorCritic()
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    normalizer = RunningNormalizer(C.OBS_DIM)
    (campaign / "normalizer.json").write_text(
        json.dumps(normalizer.state_dict()), encoding="utf-8")
    torch.save(net.state_dict(), campaign / "checkpoints" / "ckpt_init.pt")
    diagnostics_all = []
    episode_time = []
    episodes_per_update = cfg["episodes_per_update"]
    for update_index in range(cfg["updates"]):
        round_start = time.monotonic()
        checkpoint = campaign / "checkpoints" / f"ckpt_u{update_index}.pt"
        if not checkpoint.exists():
            torch.save(net.state_dict(), checkpoint)
        # The workers normalize with the round-start normalizer state; the
        # batch for this round's update uses exactly that same state, and the
        # live normalizer absorbs the new data only for the next round.
        batch_normalizer = RunningNormalizer.from_state(normalizer.state_dict())
        indices = list(range(update_index * episodes_per_update,
                             (update_index + 1) * episodes_per_update))
        pending = [i for i in indices
                   if not _episode_complete(training_dir / f"episode_{i:03d}")]
        while pending:
            chunk, pending = pending[:parallel], pending[parallel:]
            procs = []
            for episode_index in chunk:
                out = training_dir / f"episode_{episode_index:03d}"
                started = time.monotonic()
                procs.append((episode_index, started, subprocess.Popen(
                    [sys.executable, "-m", "traffic_shepherd.ppo_worker",
                     "--registration", str(registration_path),
                     "--episode-index", str(episode_index),
                     "--checkpoint", str(checkpoint),
                     "--normalizer", str(campaign / "normalizer.json"),
                     "--cohorts", str(campaign / "cohorts.json"),
                     "--out", str(out)],
                    cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True)))
            for episode_index, started, proc in procs:
                _out, err = proc.communicate()
                episode_time.append(time.monotonic() - started)
                if proc.returncode != 0:
                    raise RuntimeError(f"episode {episode_index} failed:\n{err[-2000:]}")
        rollouts = load_rollouts(training_dir, indices)
        for rollout in rollouts:
            for row in rollout:
                normalizer.update(row.get("obs_stacked_raw")
                                  or ppo_features_stacked(rollout, row))
        (campaign / "normalizer.json").write_text(
            json.dumps(normalizer.state_dict()), encoding="utf-8")
        batch, _lengths = build_batch(rollouts, batch_normalizer)
        diagnostics = update(net, optimizer, batch, [len(r) for r in rollouts])
        diagnostics.update(update_index=update_index,
            mean_return=float(sum(sum(r["reward"] for r in r_) for r_ in rollouts) / len(rollouts)),
            round_wall_s=time.monotonic() - round_start,
            mean_episode_wall_s=sum(episode_time[-len(indices):]) / len(indices))
        diagnostics_all.append(diagnostics)
        torch.save(net.state_dict(), campaign / "checkpoints" / f"ckpt_u{update_index + 1}.pt")
        (campaign / "training_diagnostics.json").write_text(
            json.dumps(diagnostics_all, indent=1), encoding="utf-8")
        print(json.dumps(diagnostics), flush=True)
    (campaign / "FINAL_CHECKPOINT.txt").write_text(
        f"ckpt_u{cfg['updates']}.pt\n{C.CHECKPOINT_RULE}\n", encoding="utf-8")
    return diagnostics_all


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", default="runs/active/ppo_pilot_v1/registration.json")
    parser.add_argument("--parallel", type=int, default=1)
    args = parser.parse_args()
    train(args.registration, parallel=args.parallel)
