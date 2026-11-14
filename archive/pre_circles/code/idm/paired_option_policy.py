"""Small decentralized actor-critic for the paired IDM option problem."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from idm.paired_option_dataset import PHYSICAL_OBSERVATION_DIM, PairedOptionRecord


class BinaryOptionActorCritic(nn.Module):
    """Flat causal-observation encoder with a binary actor and scalar critic."""

    def __init__(
        self,
        observation_mean: Sequence[float],
        observation_scale: Sequence[float],
        hidden: int = 128,
    ):
        super().__init__()
        mean = torch.as_tensor(observation_mean, dtype=torch.float32).reshape(-1)
        scale = torch.as_tensor(observation_scale, dtype=torch.float32).reshape(-1)
        if mean.numel() != PHYSICAL_OBSERVATION_DIM or scale.numel() != mean.numel():
            raise ValueError("normalization vectors do not match the 134-field contract")
        if torch.any(~torch.isfinite(mean)) or torch.any(~torch.isfinite(scale)):
            raise ValueError("normalization vectors must be finite")
        self.hidden = int(hidden)
        self.register_buffer("observation_mean", mean)
        self.register_buffer("observation_scale", torch.clamp(scale, min=1e-4))
        self.encoder = nn.Sequential(
            nn.Linear(PHYSICAL_OBSERVATION_DIM, self.hidden),
            nn.LayerNorm(self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, self.hidden),
            nn.Tanh(),
        )
        self.actor = nn.Linear(self.hidden, 2)
        self.critic = nn.Linear(self.hidden, 1)
        # Exactly 50/50 before seeing paired evidence.
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.nan_to_num(observation.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if values.dim() == 1:
            values = values.unsqueeze(0)
        normalized = (values - self.observation_mean) / self.observation_scale
        encoded = self.encoder(torch.clamp(normalized, -10.0, 10.0))
        return self.actor(encoded), self.critic(encoded).squeeze(-1)

    @torch.no_grad()
    def activation_probability(self, observation: np.ndarray) -> np.ndarray:
        logits, _ = self(torch.as_tensor(observation, dtype=torch.float32))
        return torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()


def normalization(records: Sequence[PairedOptionRecord]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray([row.observation for row in records], dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != PHYSICAL_OBSERVATION_DIM:
        raise ValueError("records do not match the physical observation contract")
    return values.mean(axis=0).astype(np.float32), values.std(axis=0).astype(np.float32)


def evaluate_binary_policy(
    model: BinaryOptionActorCritic,
    records: Sequence[PairedOptionRecord],
    threshold: float,
    *,
    deadband: float = 0.02,
) -> dict:
    """Evaluate a deterministic deployment threshold on paired ground truth."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    observations = np.asarray([row.observation for row in records], dtype=np.float32)
    probabilities = model.activation_probability(observations)
    actions = probabilities >= float(threshold)
    rewards = np.asarray([row.activate_reward for row in records], dtype=np.float64)
    effects = np.asarray([row.physical_utility for row in records], dtype=np.float64)
    constraints = np.asarray([row.constraints_pass for row in records], dtype=bool)
    selected_rewards = rewards * actions
    selected_effects = effects * actions
    helpful = actions & constraints & (effects > deadband)
    harmful = actions & ((~constraints) | (effects < -deadband))
    activation_count = int(actions.sum())
    denom = max(activation_count, 1)
    by_penetration = {}
    for penetration in sorted({float(row.penetration) for row in records}):
        selected = np.asarray(
            [abs(float(row.penetration) - penetration) <= 1e-12 for row in records]
        )
        by_penetration[f"{penetration:.2f}"] = {
            "count": int(selected.sum()),
            "activation_coverage": float(actions[selected].mean()),
            "mean_paired_effect": float(selected_effects[selected].mean()),
            "mean_reward": float(selected_rewards[selected].mean()),
        }
    always_tail = float(np.percentile(rewards, 10.0))
    selected_tail = float(np.percentile(selected_rewards, 10.0))
    return {
        "count": len(records),
        "threshold": float(threshold),
        "mean_activation_probability": float(probabilities.mean()),
        "minimum_activation_probability": float(probabilities.min()),
        "maximum_activation_probability": float(probabilities.max()),
        "activation_count": activation_count,
        "abstention_count": int(len(records) - activation_count),
        "activation_coverage": float(actions.mean()),
        "mean_reward": float(selected_rewards.mean()),
        "mean_paired_effect": float(selected_effects.mean()),
        "helpful_activation_count": int(helpful.sum()),
        "harmful_activation_count": int(harmful.sum()),
        "neutral_activation_count": int(activation_count - helpful.sum() - harmful.sum()),
        "helpful_activation_fraction": float(helpful.sum() / denom),
        "harmful_activation_fraction": float(harmful.sum() / denom),
        "helpful_minus_harmful_fraction": float((helpful.sum() - harmful.sum()) / denom),
        "constraint_failure_activation_count": int((actions & ~constraints).sum()),
        "tenth_percentile_reward": selected_tail,
        "always_abstain_mean_reward": 0.0,
        "always_activate_mean_reward": float(rewards.mean()),
        "always_activate_mean_paired_effect": float(effects.mean()),
        "always_activate_tenth_percentile_reward": always_tail,
        "improves_over_always_activate_mean_or_tail": bool(
            selected_rewards.mean() > rewards.mean() + 1e-12
            or selected_tail > always_tail + 1e-12
        ),
        "by_penetration": by_penetration,
    }


def select_deployment_threshold(
    model: BinaryOptionActorCritic,
    records: Sequence[PairedOptionRecord],
    *,
    minimum_coverage: float = 0.20,
) -> tuple[float, dict]:
    """Choose a conservative threshold using training rows only.

    Feasible thresholds satisfy the already frozen coverage/harm criteria.
    If none do, the returned diagnostic explicitly marks the fallback; this
    never silently turns a failed smoke into an accepted policy.
    """

    probabilities = model.activation_probability(
        np.asarray([row.observation for row in records], dtype=np.float32)
    )
    candidates = sorted(
        {
            0.50,
            *[float(value) for value in np.linspace(0.05, 0.95, 37)],
            *[float(value) for value in probabilities],
        }
    )
    rows = [evaluate_binary_policy(model, records, threshold) for threshold in candidates]
    feasible = [
        row
        for row in rows
        if row["activation_coverage"] >= minimum_coverage
        and row["harmful_activation_fraction"] <= 0.25
        and row["helpful_minus_harmful_fraction"] >= 0.15
    ]
    pool = feasible or [row for row in rows if row["activation_coverage"] >= minimum_coverage]
    if not pool:
        pool = rows
    best = max(
        pool,
        key=lambda row: (
            row["mean_reward"],
            row["mean_paired_effect"],
            -row["harmful_activation_fraction"],
            row["activation_coverage"],
        ),
    )
    return float(best["threshold"]), {
        "selection_split": "train",
        "frozen_gate_feasible": bool(feasible),
        "candidate_count": len(rows),
        "selected": best,
    }
