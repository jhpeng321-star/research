"""Wall-over-champion composition (2026-08-23, TTC ablation probe).

The two mechanisms on their own layers: the rule-triggered wall handles the
metastable collapse (throttling, global), while the frozen pruned champion's
damping modes (recover/guard) drive every CAV OUTSIDE the wall band (local
smoothing) — including inside the queue that forms behind the wall. The
question: does the champion's damping reduce the residual near-misses (TTC)
that remain inside the wall's slow region, without disturbing the wall's
protection?

Division of labor per step:
  - trigger active  -> in-band CAVs: the wall (direct TraCI, probe-identical);
                       out-of-band CAVs: the champion's modes (env applies).
  - trigger idle    -> the champion drives everything.
"""

from __future__ import annotations

from typing import Any, Dict

import traci

from env.features import distance_to_merge
from policies import DirectControlPolicy
from policies.inflow_control import WallHoldPolicy
from policies.rllib_policy import RLlibPolicy


class WallOverChampionPolicy(DirectControlPolicy):
    # NOT a pure direct-control policy: the champion's actions must be
    # applied by the env (hybrid mode), while the wall TraCIs the in-band
    # CAVs. So direct_control stays False and the returned dict is real.
    direct_control = False

    def __init__(self, ckpt: str, wall_params: dict):
        super().__init__()
        self.ckpt = ckpt
        self.wall_params = dict(wall_params)
        self._wall = None
        self._champ = None

    def reset(self, env, seed: int = None) -> None:
        self._env = env
        self._wall = WallHoldPolicy(**self.wall_params)
        self._wall.reset(env, seed)
        if self._champ is None:
            self._champ = RLlibPolicy(self.ckpt)
        self._champ.reset(env, seed=seed)

    def act(self, obs_dict: Dict[str, Any]) -> Dict[str, object]:
        champ_actions = self._champ.act(obs_dict)
        active = self._wall._wall_active()  # trigger hysteresis (h arm/release)
        if not active:
            return champ_actions
        inband = {}
        live = traci.vehicle.getIDList()
        for cav_id in obs_dict.keys():
            if cav_id not in live:
                continue
            try:
                d = distance_to_merge(cav_id)
            except Exception:
                continue
            if self._wall.d_lo <= d <= self._wall.d_hi:
                inband[cav_id] = obs_dict[cav_id]
        if inband:
            self._wall.act(inband)  # probe-identical wall on the band members
        # the champion drives every CAV NOT in the wall band
        return {c: a for c, a in champ_actions.items() if c not in inband}

    def close(self) -> None:
        if self._champ is not None:
            self._champ.close()
