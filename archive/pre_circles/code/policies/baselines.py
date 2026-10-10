"""Rule-based / trivial baseline policies (no learning).

- `NoOpPolicy`          — always mode 0 (let SUMO drive)
- `RandomPolicy`        — random Shepherd mode each step
- `ConstantSpeedPolicy` — non_intervention (mode 0); equivalent to no_op
  under shepherd control mode
- `RuleBasedMergePolicy`— yield to adjacent-lane merge intent via kinematic decel
"""

from typing import Dict

import numpy as np
import traci

# Imported from the partially initialized package: the ABCs are defined at the
# top of policies/__init__.py, before this module is imported.
from policies import DirectControlPolicy, Policy
from policies.traci_control import (
    apply_kinematic_speed,
    compute_a_req,
    configure_cav,
    find_merge_intent,
    release_to_sumo,
)


class NoOpPolicy(Policy):
    """Baseline that always selects mode 0 (let SUMO drive)."""

    def reset(self, env, seed: int = None) -> None:
        pass

    def act(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, int]:
        return {cav_id: 0 for cav_id in obs_dict.keys()}


class RandomPolicy(Policy):
    """Baseline that samples a random Traffic Shepherd mode for each CAV."""

    def __init__(self, seed: int = None):
        self._rng = np.random.default_rng(seed)

    def reset(self, env, seed: int = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._num_modes = int(env.action_space.n)

    def act(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, int]:
        return {cav_id: int(self._rng.integers(0, self._num_modes)) for cav_id in obs_dict.keys()}


class ConstantSpeedPolicy(Policy):
    """Baseline that selects non_intervention (mode 0) for every CAV.

    Under shepherd control mode this is equivalent to no_op: control is
    handed back to SUMO's car-following model, which is the closest honest
    "steady driving" reference available without overriding SUMO. For a true
    constant-speed semantic, use continuous mode with a P controller
    (not implemented — YAGNI).
    """

    def reset(self, env, seed: int = None) -> None:
        pass

    def act(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, int]:
        return {cav_id: 0 for cav_id in obs_dict.keys()}


class ForcedModePolicy(Policy):
    """Ablation controller: force one Shepherd mode on (a subset of) CAVs.

    Every CAV whose distance to the merge junction is <= ``radius`` meters
    receives ``mode``; all other CAVs get mode 0 (non_intervention). With
    ``radius=None`` every active CAV is forced. Used to measure the system
    effect of a single intervention type without any learned policy.
    """

    def __init__(self, mode: int, radius: float = None):
        self.mode = int(mode)
        self.radius = radius

    def reset(self, env, seed: int = None) -> None:
        pass

    def act(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, int]:
        from env.features import distance_to_merge

        actions = {}
        for cav_id in obs_dict.keys():
            if self.radius is None:
                actions[cav_id] = self.mode
            else:
                d = distance_to_merge(cav_id)
                actions[cav_id] = self.mode if 0.0 <= d <= self.radius else 0
        return actions


class RuleBasedMergePolicy(DirectControlPolicy):
    """Yield when an adjacent-lane vehicle wants to merge; otherwise SUMO drives."""

    lateral_radius: float = 50.0
    max_decel: float = 4.5
    decel_threshold: float = 0.3
    normal_tau: float = 1.0
    simulation_step: float = 1.0

    def reset(self, env, seed: int = None) -> None:
        self.simulation_step = getattr(env, "simulation_step", 1.0)

    def act(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, int]:
        for cav_id in obs_dict:
            if cav_id not in traci.vehicle.getIDList():
                continue

            configure_cav(cav_id)
            v_current = traci.vehicle.getSpeed(cav_id)

            intent = find_merge_intent(cav_id, self.lateral_radius)
            if intent is not None:
                v_target = min(v_current, intent.v_merging)
                a_req = compute_a_req(v_current, v_target, intent.delta_x)
                if a_req < -self.decel_threshold:
                    apply_kinematic_speed(
                        cav_id, v_current, a_req, self.max_decel, dt=self.simulation_step
                    )
                    continue

            release_to_sumo(cav_id, tau=self.normal_tau)

        return {}
