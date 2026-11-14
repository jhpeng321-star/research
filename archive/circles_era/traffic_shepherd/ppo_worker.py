"""One-episode PPO worker: chargeable rollout with a frozen policy snapshot.

Invoked by the trainer as a separate process (SUMO/TraCI own their process).
The worker never mutates shared state except its own output directory and
its own budget-ledger reservation, which happens at the real launch boundary
inside PpoEpisode.run().
"""
import argparse
import json
from pathlib import Path

import torch

from . import ppo_contract as C
from . import ppo_features
from .ppo_env import PpoEpisode, build_pilot_job
from .ppo_net import ActorCritic, seed_everything


def load_net(path):
    net = ActorCritic()
    net.load_state_dict(torch.load(path, weights_only=True))
    net.eval()
    return net


def stacked_obs(history_raw, index):
    """Zero-padded HISTORY_STACK window ending at index (causal)."""
    if not 0 <= index < len(history_raw):
        raise ValueError("stack index outside the recorded history")
    dim = len(history_raw[index])
    parts = []
    for k in range(C.HISTORY_STACK - 1, -1, -1):
        j = index - k
        parts.append(history_raw[j] if j >= 0 else [0.0] * dim)
    return [x for part in parts for x in part]


SAMPLING_VERSION = "dedicated_generator/2"


class SamplingActor:
    """Stochastic deployment actor with a DEDICATED sampling stream.

    v1 (executed in ppo_run_v1) created a torch.Generator but sampled via
    Categorical.sample(), which draws from the GLOBAL torch RNG: the declared
    episode seed never governed sampling and episodes were reproducible only
    up to shared-stream correlation. v2 draws actions with
    torch.multinomial(..., generator=self.rng) from the masked categorical
    probabilities; the stored log-probability still comes from the same
    masked distribution, so sampling and evaluation remain consistent.
    """

    def __init__(self, net, normalizer_state, sample_seed):
        self.net = net
        self.sampling_version = SAMPLING_VERSION
        self.normalizer = ppo_features.RunningNormalizer.from_state(normalizer_state)
        self.rng = torch.Generator()
        self.rng.manual_seed(int(sample_seed))
        self.history = []
        self.rows = []

    def __call__(self, mask, features, phase, now):
        self.history.append(features)
        obs = torch.tensor([self.normalizer.normalize(
            stacked_obs(self.history, len(self.history) - 1))], dtype=torch.float32)
        with torch.no_grad():
            distribution = self.net.distribution(obs, torch.tensor([mask]))
            probs = distribution.probs[0]
            action = torch.multinomial(probs, num_samples=1, generator=self.rng)
            logp = distribution.log_prob(action)
            value = self.net.value(obs)
        self.rows.append((int(action.item()), float(logp.item()), float(value.item())))
        return int(action.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--normalizer", required=True)
    parser.add_argument("--cohorts", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    registration = json.loads(Path(args.registration).read_text(encoding="utf-8"))
    cohorts = json.loads(Path(args.cohorts).read_text(encoding="utf-8"))
    schedule = registration["training_schedule"]
    spec = next(e for e in schedule if e["episode_index"] == args.episode_index)
    seed, loading = spec["seed"], spec["loading"]
    binding = cohorts[str(seed)]
    if binding["loading"] != loading:
        raise ValueError("cohort/scenario loading mismatch")
    member_ids, required_lanes = binding["member_ids"], binding["required_lanes"]
    no_cohort = member_ids is None
    if no_cohort:
        # Registered no-cohort scenario: no feasible pair exists at the
        # binding tick, so activation is infeasible BY CONSTRUCTION. The
        # episode runs as matched damping with WAIT-only decisions (the
        # mask keeps ACTIVATE unavailable); recorded as such in artifacts.
        member_ids = ["ghost_a", "ghost_b"]

    net = load_net(args.checkpoint)
    normalizer_state = json.loads(Path(args.normalizer).read_text(encoding="utf-8"))
    seed_everything(C.PPO_CONFIG["optimizer_seed"])
    actor = SamplingActor(net, normalizer_state,
                          C.PPO_CONFIG["optimizer_seed"] + args.episode_index)
    job = build_pilot_job(seed=seed, loading=loading, horizon=C.HORIZON_S,
                          case_id=f"train{args.episode_index:03d}",
                          policy_spec=dict(member_ids=list(member_ids),
                                           required_lanes=list(required_lanes),
                                           no_cohort=no_cohort))
    episode = PpoEpisode(job=job, registration=registration, member_ids=member_ids,
        required_lanes=required_lanes, policy_mode="ppo", actor=actor,
        output=Path(args.out),
        budget=dict(registration["budget"], category="training",
                    amount_s=float(C.HORIZON_S), target_dir=registration["campaign_root"],
                    campaign_stop_on_failure=True))
    summary, rollout = episode.run()
    if len(rollout) != len(actor.rows):
        raise ValueError("rollout/actor decision count mismatch")
    for index, row in enumerate(rollout):
        action, logp, value = actor.rows[index]
        if action != row["action"]:
            raise ValueError("rollout/actor action mismatch")
        row["logp"] = logp
        row["value"] = value
        row["obs_stacked_raw"] = stacked_obs(actor.history, index)
    (Path(args.out) / "rollout.json").write_text(
        json.dumps(rollout, allow_nan=False), encoding="utf-8")
    manifest = json.loads((Path(args.out) / "manifest.json").read_text(encoding="utf-8"))
    print(json.dumps(dict(job_id=manifest["job_id"], status=manifest["status"],
                          completed_simulated_seconds=summary["completed_simulated_seconds"
                          if "completed_simulated_seconds" in summary else "duration_s"])))


if __name__ == "__main__":
    main()
