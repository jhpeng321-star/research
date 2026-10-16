"""CPU fake of ``traffic_shepherd.c4.CavSumoEnv`` for MAPPO unit tests and ``--smoke``.

Random WM-frame states on three lanes (deterministic given the reset seed), simple
kinematics, CAV accelerations applied, lane targets (``step(accel, lanes=...)``, -1 = stay)
executed.  Everything else (observations, RSU field, actor, safety, lanes, reward) is the
REAL ``wms.control`` code.  ``fail_seeds`` raise in ``reset`` (worker retry tests) and
``exit_seeds`` terminate the process (worker-death tests).  ``entry_queue`` (optional float)
makes ``step`` emit ``info["entry_queue_steps"]`` / ``["source_wait_steps"]`` like the real env:
a deterministic counter pattern ``entry_queue + (step index % 3)`` (no RNG use, so the random
stream of existing tests is unchanged); ``None`` (default) emits nothing.
"""

from __future__ import annotations

import os

import torch
from torch import Tensor

from wms.control.contract import (ACCEL_MIN_MPS2, CONTROL_DT_S, DIRECT_ACCEL_MAX_MPS2, MERGE_S1_M,
                                  N_LANES, SPEED_LIMIT_MPS)

VEHICLE_LENGTH_M = 5.0


class FakeCavEnv:
    """``CavEnv`` protocol on random states; ``step(accel [1,K], lanes=[1,K] | None)``."""

    def __init__(self, k_slots: int = 24, min_vehicles: int = 12, churn_prob: float = 0.1,
                 fail_seeds=(), exit_seeds=(), entry_queue: float | None = None):
        self.k_slots = int(k_slots)
        self.min_vehicles = int(min_vehicles)
        self.churn_prob = float(churn_prob)
        self.fail_seeds = {int(s) for s in fail_seeds}
        self.exit_seeds = {int(s) for s in exit_seeds}
        self.entry_queue = None if entry_queue is None else float(entry_queue)
        self._queue_tick = 0
        self._gen: torch.Generator | None = None
        self.done = True
        self.lane_counts = {"issued": 0, "completed": 0}

    def _rand(self, *shape: int, lo: float = 0.0, hi: float = 1.0) -> Tensor:
        return lo + (hi - lo) * torch.rand(shape, generator=self._gen)

    def _state(self) -> dict[str, Tensor]:
        p = self._present
        return {"s_m": torch.where(p, self._s, torch.zeros_like(self._s))[None],
                "v_mps": torch.where(p, self._v, torch.zeros_like(self._v))[None],
                "a_mps2": torch.where(p, self._a, torch.zeros_like(self._a))[None],
                "lane": torch.where(p, self._lane, torch.full_like(self._lane, -1))[None],
                "present": p.clone()[None], "is_cav": (self._is_cav & p)[None],
                "length_m": torch.full((1, self.k_slots), VEHICLE_LENGTH_M)}

    def reset(self, *, seed: int, demand_vph: int, penetration: float, warmup_s: float,
              horizon_s: float) -> dict[str, Tensor]:
        if int(seed) in self.exit_seeds:
            os._exit(17)
        if int(seed) in self.fail_seeds:
            raise RuntimeError(f"fake SUMO failure for seed {seed}")
        self._gen = torch.Generator().manual_seed(int(seed) * 7919 + int(demand_vph))
        self._penetration = float(penetration)
        k = self.k_slots
        n = int(torch.randint(self.min_vehicles, k + 1, (1,), generator=self._gen))
        present = torch.zeros(k, dtype=torch.bool)
        present[torch.randperm(k, generator=self._gen)[:n]] = True
        self._lane = torch.randint(0, N_LANES, (k,), generator=self._gen)
        # spread vehicles so most gaps are positive; ramp vehicles near the merge zone
        self._s = self._rand(k, lo=-400.0, hi=MERGE_S1_M + 100.0)
        self._s = torch.where(self._lane == 2, self._rand(k, lo=-60.0, hi=MERGE_S1_M), self._s)
        self._v = self._rand(k, lo=8.0, hi=28.0)
        self._a = torch.zeros(k)
        is_cav = present & (self._rand(k) < max(self._penetration, 0.25))
        if not bool(is_cav.any()):
            is_cav[torch.nonzero(present)[0]] = True
        self._present, self._is_cav = present, is_cav
        self._steps_left = int(round(float(horizon_s) / CONTROL_DT_S))
        self.done = False
        return self._state()

    def step(self, accel_by_slot=None, *, lanes=None, fallback: str = "native"):
        if self.done:
            raise RuntimeError("episode is done; call reset()")
        dt = CONTROL_DT_S
        cmd = torch.as_tensor(accel_by_slot, dtype=torch.float32).reshape(-1)
        # Same env-level DIRECT ceiling as CavSumoEnv (ACCELCAP_NOTES.md); the actor's bound binds first.
        cmd = torch.nan_to_num(cmd, nan=0.0).clamp(ACCEL_MIN_MPS2, DIRECT_ACCEL_MAX_MPS2)
        cav = self._is_cav & self._present
        hdv = self._rand(self.k_slots, lo=-0.5, hi=0.5)
        self._a = torch.where(cav, cmd, hdv)
        self._v = (self._v + self._a * dt).clamp(0.0, SPEED_LIMIT_MPS)
        self._s = self._s + self._v * dt
        issued = 0
        if lanes is not None:
            target = torch.as_tensor(lanes).reshape(-1).long()
            move = cav & (target >= 0) & (target <= 1) & (target != self._lane)
            issued = int(move.sum())
            self._lane = torch.where(move, target, self._lane)
        self.lane_counts["issued"] += issued
        self.lane_counts["completed"] += issued
        if float(torch.rand((1,), generator=self._gen)) < self.churn_prob:
            slot = int(torch.nonzero(self._present)[0])
            self._s[slot] = float(self._rand(1, lo=-400.0, hi=-300.0)[0])
            self._v[slot] = float(self._rand(1, lo=8.0, hi=28.0)[0])
            self._is_cav[slot] = bool(torch.rand((1,), generator=self._gen) < self._penetration)
        self._steps_left -= 1
        done = self._steps_left <= 0
        info = {"terminated": False, "done": done,
                "lane_commands": {"issued": issued, "completed": issued}}
        if self.entry_queue is not None:
            queue = self.entry_queue + float(self._queue_tick % 3)
            self._queue_tick += 1
            info["entry_queue_steps"] = [queue, queue]
            info["source_wait_steps"] = [0.5 * queue, 0.5 * queue]
        if done:
            self.done = True
            info["summary"] = {"collisions": 0, "teleports": 0,
                               "lane_commands": dict(self.lane_counts)}
        return self._state(), info

    def close(self) -> None:
        self.done = True
