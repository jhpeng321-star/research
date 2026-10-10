"""Native-IDM no-op base policy (sustained-wall pilot Arm C).

Arm C must leave CAVs under native IDM with no commands at all: the existing
``NoOpPolicy`` returns a mode-0 action per CAV, which the Shepherd engine is
entitled to apply, so it is not a "no commands" control. This policy returns
an empty action dict and issues no TraCI vehicle commands.

The one deliberate exception: ``act`` reads the merge-zone health once per
step via ``env.metastable_health_detail()``. That accessor advances its EMA
once per call, and the treatment arms make the same call inside their own
``act`` before the environment step. Keeping the call pattern identical
across arms is what makes the prefix health series and replay digests
comparable; it is an observation read, not a vehicle command.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from policies import DirectControlPolicy


class NoOpNativeIDMPolicy(DirectControlPolicy):
    """Direct-control policy that issues no TraCI commands at all."""

    direct_control = True

    def __init__(self) -> None:
        super().__init__()
        self._env: Any = None
        self.last_audit: Dict[str, object] = {}

    def reset(self, env: Any, seed: int = None) -> None:
        self._env = env
        self.last_audit = {}

    def act(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, object]:
        # Health-EMA discipline (see module docstring): one pre-step call.
        if self._env is not None:
            try:
                self._env.metastable_health_detail()
            except Exception:
                pass
        self.last_audit = {"treatment_state": "DAMPING", "wall_active": False}
        return {}
