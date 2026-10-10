"""Rule-based Traffic Shepherd baseline — same observation and action space as
the learned shepherd, but with hand-crafted mode-selection rules.
"""

from typing import Dict

import numpy as np

from policies import Policy


def _obs_ttc(ego_speed: float, nbrs: np.ndarray, n_count: int) -> float:
    """Estimate ego TTC from the closest neighbour ahead (purely from obs)."""
    best_ttc = float("inf")
    for i in range(n_count):
        rel_dist = float(nbrs[i, 0])
        rel_spd = float(nbrs[i, 1])
        if rel_dist <= 0:
            continue
        nbr_speed = ego_speed + rel_spd
        closing = ego_speed - nbr_speed
        if closing <= 0.01:
            continue
        ttc = rel_dist / closing
        if ttc < best_ttc:
            best_ttc = ttc
    return best_ttc


class RuleBasedShepherdPolicy(Policy):
    """Select gated shepherd modes with hand-crafted rules from the obs vector.

    Observation layout (v3, see ``env.features``):
      [0:9]    ego:   [lane, speed/35, accel, time_to_merge, density_id,
                       leader_gap, leader_rel_speed, queue_ahead, queue_behind]
      [9:21]   zone:  [pre_occ, pre_spd/35, merge_occ, merge_spd/35,
                       ramp_occ, ramp_spd/35, ramp_queue, merge_outflow,
                       delay_excess, fair_gap_excess, main_delay, ramp_delay]
    """

    MAX_SPEED: float = 35.0
    EGO_DIM: int = 9
    ZONE_DIM: int = 12
    MAX_NEIGHBORS: int = 20

    ttc_conflict: float = 2.0
    meter_pre_spd: float = 14.0
    recover_pre_spd: float = 12.0
    recover_merge_spd: float = 10.0
    vacate_ttm: float = 0.15  # normalized time_to_merge threshold

    hysteresis_steps: int = 2

    MODE_COAST = 0
    MODE_METER_INFLOW = 1
    MODE_RECOVER_SPEED = 2
    MODE_VACATE_DROP = 3
    MODE_YIELD_CONFLICT = 4

    def reset(self, env, seed: int = None) -> None:
        self._hyst: Dict[str, tuple] = {}
        self._hyst_committed: Dict[str, int] = {}

    def act(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, int]:
        actions: Dict[str, int] = {}
        nbr_start = self.EGO_DIM + self.ZONE_DIM
        for cav_id, obs in obs_dict.items():
            lane = float(obs[0])
            ego_speed = float(obs[1]) * self.MAX_SPEED
            ttm = float(obs[3])
            pre_spd = float(obs[self.EGO_DIM + 1]) * self.MAX_SPEED
            merge_spd = float(obs[self.EGO_DIM + 3]) * self.MAX_SPEED

            nbrs = obs[nbr_start : nbr_start + self.MAX_NEIGHBORS * 5].reshape(-1, 5)
            nbrs = nbrs.copy()
            nbrs[:, 1] *= self.MAX_SPEED
            ego_ttc = _obs_ttc(ego_speed, nbrs, self.MAX_NEIGHBORS)

            # Priority: conflict yield → vacate drop → meter → recover → coast
            if ego_ttc < self.ttc_conflict:
                selected = self.MODE_YIELD_CONFLICT
            elif lane < 0.5 and ttm < self.vacate_ttm:
                # lane feature is typically 0 for rightmost / drop-side
                selected = self.MODE_VACATE_DROP
            elif pre_spd < self.meter_pre_spd:
                selected = self.MODE_METER_INFLOW
            elif (
                ego_speed < self.recover_pre_spd
                and merge_spd >= self.recover_merge_spd
                and pre_spd < self.meter_pre_spd + 4.0
            ):
                selected = self.MODE_RECOVER_SPEED
            else:
                selected = self.MODE_COAST

            prev_mode, count = self._hyst.get(cav_id, (selected, 0))
            if selected == prev_mode:
                count += 1
            else:
                count = 1
            self._hyst[cav_id] = (selected, count)
            if count >= self.hysteresis_steps:
                actions[cav_id] = selected
            else:
                actions[cav_id] = self._hyst_committed.get(cav_id, self.MODE_COAST)

        self._hyst_committed = dict(actions)
        return actions
