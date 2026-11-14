"""Frozen partitions and budgets for standalone native-IDM damping training."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


SCHEMA = "native_damping_campaign/1"
TRAINING_SEEDS = tuple(range(910001, 910017))
DEVELOPMENT_SEEDS = tuple(range(911001, 911009))
ACCEPTANCE_SEEDS = tuple(range(912001, 912033))
TRAINING_LOADS = (
    1900,
    2000,
    2100,
    2600,
    2200,
    1900,
    2000,
    2700,
    2100,
    2200,
    1900,
    2600,
    2000,
    2100,
    2200,
    2700,
)
DEVELOPMENT_LOADS = (2000, 2000, 2200, 2200, 2600, 2600, 2700, 2700)
TRAINING_CASES = tuple(zip(TRAINING_SEEDS, TRAINING_LOADS))
DEVELOPMENT_CASES = tuple(zip(DEVELOPMENT_SEEDS, DEVELOPMENT_LOADS))
ACCEPTANCE_CASES = tuple(
    zip(ACCEPTANCE_SEEDS, (2000,) * 8 + (2200,) * 8 + (2600,) * 8 + (2700,) * 8)
)
OPTIMIZER_SEEDS = (71, 72, 73)
CHECKPOINT_EPISODES = (20, 62, 100)
SEED_AUDIT = (
    Path(__file__).resolve().parents[1]
    / "runs/probe/native_damping_campaign_seed_audit_2026-09-05_v2.json"
)


def validate_case(seed, loading, role):
    allowed = {"training": TRAINING_CASES, "development": DEVELOPMENT_CASES}.get(role)
    if allowed is None:
        raise ValueError(
            "campaign generation permits training/development only; acceptance stays sealed"
        )
    if (seed, loading) not in allowed:
        raise ValueError("seed/loading pair is outside the registered campaign role")


def validate_seed_audit(path=SEED_AUDIT):
    evidence = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = dict(
        training=list(TRAINING_SEEDS),
        development=list(DEVELOPMENT_SEEDS),
        acceptance_reserved=list(ACCEPTANCE_SEEDS),
    )
    if (
        not evidence.get("passed")
        or evidence.get("hits")
        or evidence.get("errors")
        or evidence["blocks"] != expected
    ):
        raise ValueError("campaign requires the matching successful pre-registration seed audit")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class CampaignBudget:
    episode_seconds: int = 2400
    initial_optimized_episodes: int = 4
    previous_allocated_policy_seconds: int = 19200
    episodes_per_lineage: int = 100
    total_policy_cap: int = 750000

    def new_episodes(self, optimizer_seed):
        if optimizer_seed not in OPTIMIZER_SEEDS:
            raise ValueError("unregistered optimizer seed")
        return self.episodes_per_lineage - (
            self.initial_optimized_episodes if optimizer_seed == 71 else 0
        )

    def manifest(self):
        new = sum(self.new_episodes(seed) for seed in OPTIMIZER_SEEDS) * self.episode_seconds
        allocated = new + self.previous_allocated_policy_seconds
        if allocated > self.total_policy_cap:
            raise ValueError("campaign policy simulation budget exceeded")
        return {
            **asdict(self),
            "new_policy_seconds": new,
            "allocated_policy_seconds": allocated,
            "unused_policy_budget_seconds": self.total_policy_cap - allocated,
            "new_training_reference_seconds": len(TRAINING_CASES) * self.episode_seconds,
            "development_reference_and_analytic_rollouts": 2 * len(DEVELOPMENT_CASES),
            "development_checkpoint_rollouts": len(OPTIMIZER_SEEDS)
            * len(CHECKPOINT_EPISODES)
            * len(DEVELOPMENT_CASES),
        }


def campaign_manifest():
    return dict(
        schema=SCHEMA,
        training_cases=[list(case) for case in TRAINING_CASES],
        development_cases=[list(case) for case in DEVELOPMENT_CASES],
        acceptance_reserved_cases=[list(case) for case in ACCEPTANCE_CASES],
        optimizer_seeds=list(OPTIMIZER_SEEDS),
        checkpoint_episodes=list(CHECKPOINT_EPISODES),
        checkpoint_optimized_seconds=[episode * 2400 for episode in CHECKPOINT_EPISODES],
        budget=CampaignBudget().manifest(),
        seed_audit_sha256=validate_seed_audit(),
        initial_training_state="verified_repaired_v2_four_updates_for_lineage71_only",
        wall_generation="paused_until_accepted_damping",
    )
