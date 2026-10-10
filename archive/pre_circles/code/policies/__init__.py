"""Eval-time policies: abstract interfaces, rule-based baselines, RLlib loading.

- `Policy` / `DirectControlPolicy` — abstract interfaces (defined here)
- baselines.py                     — no_op, random, constant_speed, rule_based
- rllib_policy.py                  — load trained RLlib checkpoints
- traci_control.py                 — TraCI helpers for direct control
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class Policy(ABC):
    """Abstract base class for any TrafficShepherd controller.

    A policy receives observations from the multi-agent environment and returns
    an action for each active CAV. The action is a continuous acceleration for
    Selfish / Flat-MARL or a discrete Shepherd mode for Shepherd.

    Direct-control policies (see ``DirectControlPolicy``) may return an empty
    dict and apply TraCI commands themselves inside ``act()``.
    """

    direct_control: bool = False

    def reset(self, env: Any, seed: int = None) -> None:
        """Reset the policy at the start of a new episode.

        Args:
            env: The TrafficShepherdMultiAgentEnv instance.
            seed: Optional RNG seed for reproducibility.
        """
        pass

    @abstractmethod
    def act(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, object]:
        """Return one environment action for each active CAV.

        Args:
            obs_dict: mapping from CAV vehicle ID to observation vector.

        Returns:
            A dict mapping CAV IDs to either:
              - a scalar continuous acceleration in [-3, +3] m/s², or
              - an integer Shepherd mode in [0, 4].
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release any resources held by the policy."""
        pass


class DirectControlPolicy(Policy):
    """Policy that applies TraCI speed control directly and returns no discrete modes."""

    direct_control: bool = True

    @abstractmethod
    def act(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, object]:
        """Apply TraCI control and return ``{}`` so the env skips discrete modes."""
        raise NotImplementedError


from policies.baselines import (  # noqa: E402
    ConstantSpeedPolicy,
    ForcedModePolicy,
    NoOpPolicy,
    RandomPolicy,
    RuleBasedMergePolicy,
)

from policies.rule_shepherd import RuleBasedShepherdPolicy  # noqa: E402

try:
    from policies.rllib_policy import RLlibPolicy
except ImportError:
    RLlibPolicy = None

__all__ = [
    "Policy",
    "DirectControlPolicy",
    "NoOpPolicy",
    "RandomPolicy",
    "ConstantSpeedPolicy",
    "RuleBasedMergePolicy",
    "RuleBasedShepherdPolicy",
    "ForcedModePolicy",
    "RLlibPolicy",
]
