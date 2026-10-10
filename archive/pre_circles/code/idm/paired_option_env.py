"""One-decision contextual-bandit environment for paired IDM options."""

from __future__ import annotations

from typing import Any, Sequence

import gymnasium as gym
import numpy as np

from idm.paired_option_dataset import PHYSICAL_OBSERVATION_DIM, PairedOptionRecord


class PairedIDMOptionEnv(gym.Env):
    """Replay treatment-independent contexts with paired option rewards.

    Action 0 retains native IDM and receives exactly zero.  Action 1 applies
    the already executed P30 option and receives its frozen paired reward.
    Every episode is one decision; outcomes never enter the observation.
    """

    metadata = {"render_modes": []}

    def __init__(self, records: Sequence[PairedOptionRecord], seed: int = 0):
        if not records:
            raise ValueError("at least one paired option record is required")
        self.records = tuple(records)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(PHYSICAL_OBSERVATION_DIM,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(2)
        self._rng = np.random.default_rng(seed)
        self._current: PairedOptionRecord | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        requested = None if options is None else options.get("index")
        index = int(self._rng.integers(len(self.records))) if requested is None else int(requested)
        if not 0 <= index < len(self.records):
            raise IndexError(index)
        self._current = self.records[index]
        return np.asarray(self._current.observation, dtype=np.float32), {
            "sample_id": self._current.sample_id,
            "demand": self._current.demand,
            "penetration": self._current.penetration,
            "seed": self._current.seed,
        }

    def step(self, action: int):
        if self._current is None:
            raise RuntimeError("reset must be called before step")
        value = int(action)
        if not self.action_space.contains(value):
            raise ValueError(f"invalid binary option action {action!r}")
        row = self._current
        reward = float(row.activate_reward if value == 1 else 0.0)
        self._current = None
        return (
            np.zeros(PHYSICAL_OBSERVATION_DIM, dtype=np.float32),
            reward,
            True,
            False,
            {
                "sample_id": row.sample_id,
                "executed_action": value,
                "activate_reward": row.activate_reward,
                "physical_utility": row.physical_utility,
                "constraints_pass": row.constraints_pass,
                "dose_steps": row.dose_steps,
            },
        )
