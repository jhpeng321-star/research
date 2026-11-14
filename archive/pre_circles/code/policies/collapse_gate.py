"""Collapse-risk gate policy (v4 track 2, 2026-08-22).

The learned alternative to the handwritten health trigger: a small MLP
predicts P(sustained collapse) from the 16-dim corridor state (evaluated
with the wall state forced OFF — the unprotected risk), and the wall engages
when risk > thr_on, releases when risk < thr_off (hysteresis, same shape as
the trigger's). Execution reuses WallHoldPolicy (rung 1 = the measured
w10_t30 operating point), so gate episodes are bit-comparable to the probe
cells.

Direct-control policy for the probe harness (returns {} so the env skips
the per-CAV mode machinery).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import traci

from policies import DirectControlPolicy
from policies.inflow_control import WallHoldPolicy
from policies.traci_control import release_to_sumo

from env.wall_supervisor import RUNGS, corridor_state


class RiskGatePolicy(DirectControlPolicy):
    direct_control = True

    def __init__(self, model_path: str, thr_on: float = 0.5, thr_off: float = 0.3, rung: int = 1):
        super().__init__()
        self.model_path = model_path
        self.thr_on = float(thr_on)
        self.thr_off = float(thr_off)
        self.rung = int(rung)
        self._net = None
        self._mu = self._sd = None
        self._wall = None
        self._on = False
        self._env = None
        self._prev_h = None
        self._h_hist: list = []
        self._step_i = 0
        self._inflow_tracker: dict = {}

    def reset(self, env, seed: int = None) -> None:
        self._env = env
        self._prev_h = None
        self._h_hist = []
        self._step_i = 0
        self._inflow_tracker = {}
        self._on = False
        self._wall = WallHoldPolicy(**RUNGS[self.rung])
        self._wall.reset(env, seed)
        if self._net is None:
            import torch

            blob = torch.load(self.model_path, weights_only=False)
            d = int(blob["in_dim"])
            net = torch.nn.Sequential(
                torch.nn.Linear(d, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1),
            )
            net.load_state_dict(blob["state_dict"])
            net.eval()
            self._net = net
            self._mu = np.array(blob["mu"], dtype=np.float32)
            self._sd = np.array(blob["sd"], dtype=np.float32)

    def _risk(self, obs: np.ndarray) -> float:
        import torch

        x = (obs - self._mu) / self._sd
        # Predict the UNPROTECTED risk: force the rung feature to its raw 0 —
        # in standardized units that's (0 − mu)/sd, NOT 0 (the 2026-08-22
        # bug: x[8]=0 post-standardization = the MEAN rung = partial
        # protection, which biased risk down and the gate never engaged).
        x[8] = (0.0 - self._mu[8]) / self._sd[8]
        with torch.no_grad():
            return float(torch.sigmoid(self._net(torch.tensor(x, dtype=torch.float32))).item())

    def act(self, obs_dict: Dict[str, Any]) -> Dict[str, object]:
        obs = corridor_state(
            self._env,
            self._prev_h,
            self._rung_state(),
            0,
            self._step_i,
            self._inflow_tracker,
            self._h_hist,
        )
        self._prev_h = float(obs[0])
        self._h_hist.append(float(obs[0]))
        risk = self._risk(obs)
        if not self._on and risk > self.thr_on:
            self._on = True
        elif self._on and risk < self.thr_off:
            self._on = False
        if self._on:
            self._wall.act(obs_dict)
        else:
            live = traci.vehicle.getIDList()
            for cav_id in obs_dict.keys():
                if cav_id not in live:
                    continue
                try:
                    release_to_sumo(cav_id)
                except Exception:
                    pass
        self._step_i += 1
        return {}

    def _rung_state(self) -> int:
        return self.rung if self._on else 0
