"""Deterministic CPU stub simulator + screener for S-SUMOES tests. No SUMO import.

J(theta; seed, update) = offset(seed) + <b(seed), theta> + 0.5 * c * ||theta - star||^2
with fixed per-seed draws (common random numbers): within an antithetic pair the
per-seed offset and linear term cancel exactly in J_plus - J_minus for the SHARED
world seed, so the stub exposes CRN pairing, gradient sign and panel behaviour.
"""
from __future__ import annotations

import torch


class StubSimulator:
    def __init__(self, dimension: int = 137, curvature: float = 1.0,
                 seed: int = 20261031) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.star = torch.randn(dimension, generator=generator).double() * 0.01
        self.b = torch.randn(dimension, generator=generator).double() * 10.0
        self.curvature = float(curvature)
        self.episodes = 0

    def offset(self, world_seed: int) -> float:
        generator = torch.Generator().manual_seed(1_000_003 * world_seed)
        return float(torch.randn((), generator=generator).double()) * 500.0

    def episode(self, theta: torch.Tensor, world_seed: int, update: int) -> dict:
        self.episodes += 1
        diff = theta.detach().cpu().double() - self.star
        j600 = (self.offset(world_seed) + float(self.b @ theta.detach().cpu().double())
                + 0.5 * self.curvature * float(diff @ diff))
        return dict(job=f"s{world_seed}_u{update}_stub", J600=j600, actual_sim_s=900,
                    safety_clean=True, initial_state_sha256=f"stubstate{world_seed}")

    def evaluate(self, theta: torch.Tensor, world_seed: int, update: int, tag: str) -> dict:
        return self.episode(theta, world_seed, update) | dict(job=f"s{world_seed}_{tag}")


class RankingScreener:
    """Test screener: keeps the top-k directions by a fixed predicted ranking.

    The real screener replaces predicted ranking with frozen-WM 60 s paired
    rollouts (confident-gain screen); the driver interface is identical.
    """

    def __init__(self, simulator: StubSimulator, keep: int, probe_sigma: float = 1e-3) -> None:
        self.simulator, self.keep, self.probe_sigma = simulator, keep, probe_sigma
        self.screenings = 0

    def screen(self, theta: torch.Tensor, sigma: float, directions: torch.Tensor,
               update: int) -> dict:
        self.screenings += 1
        gains = []
        for direction in directions:
            plus = self.simulator.episode(theta + sigma * direction, -1, update)["J600"]
            minus = self.simulator.episode(theta - sigma * direction, -2, update)["J600"]
            gains.append(minus - plus)
        order = sorted(range(len(directions)), key=lambda i: gains[i], reverse=True)
        kept = sorted(order[:self.keep])
        return dict(kept=kept, predicted_gain=[gains[i] for i in kept],
                    all_gains=gains,
                    rejected=[i for i in range(len(directions)) if i not in kept])
