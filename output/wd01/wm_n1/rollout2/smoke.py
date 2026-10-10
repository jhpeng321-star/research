"""P3 60 s J4-world CPU smoke contract (integration fixture).

The frozen integration smoke is a 60 s (120 x 0.5 s) CPU rollout on a small
synthetic J4-world batch: ``batch=2`` worlds, ``slots=32``, one native CAV +
one damping CAV per world, no file reads and no SUMO.  It must run on CPU with
torch 2.5.1-compatible ops only and return finite positions/speeds with all
validity counts reported.

The world geometry (lane offsets/lengths, merge zone) comes from the frozen
``partA_v2g/params_nominal.json`` (read-only); vehicles are synthetic
kinematics-only rows so the fixture needs no run data.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from . import baseline

# slot layout per world (kinematics-only rows; the J4 topology):
#   0 main_right platoon (slot 0 = native CAV, slot 1 = damping CAV)
#   1 main_left pair
#   2 accel-lane pair inside the merge zone (u > 0 at s=430)


def j4_smoke_world(*, batch: int = 2, slots: int = 32, seed: int = 0) -> Dict:
    """Build the small CPU J4-world batch (kinematics-only)."""
    del seed  # the fixture is a fixed deterministic scene
    params = baseline.load_params()
    geom = params["geometry"]
    zone_s1 = float(params["merge"]["zone_s1_m"])
    rows = [
        # lane, s, v, a, is_cav, action
        (1, 2350.0, 29.0, 0.0, True, 0),    # native CAV (no command)
        (1, 2380.0, 29.0, 0.0, True, 1),    # damping CAV
        (1, 2410.0, 28.5, 0.0, False, -1),
        (1, 2450.0, 28.0, 0.0, False, -1),
        (0, 2400.0, 29.5, 0.0, False, -1),
        (0, 2320.0, 29.0, 0.0, False, -1),
        (2, 395.0, 22.0, 0.0, False, -1),
        (2, 430.0, 20.0, 0.0, False, -1),
    ]
    assert rows[7][1] <= zone_s1      # accel-lane pair inside the zone
    b_n = int(batch)
    k_n = int(slots)
    kin = np.zeros((b_n, k_n, 7), dtype=np.float32)
    mask = np.zeros((b_n, k_n), dtype=np.int64)
    is_cav = np.zeros((b_n, k_n), dtype=np.int64)
    slot_id = np.tile(np.arange(k_n, dtype=np.int64), (b_n, 1))
    action = np.full((b_n, k_n), -1, dtype=np.int64)
    shift = (2644.22 - 2380.0) + 40.0   # world-1 phase shift on lane 1
    for b in range(b_n):
        for j, (ln, s, v, a, cav, act) in enumerate(rows):
            if b == 1 and ln == 1:
                s = s + shift if s < 2500 else s
            kin[b, j] = (s, ln == 0, ln == 1, ln == 2, v, a, 5.0)
            mask[b, j] = 1
            is_cav[b, j] = int(cav)
            action[b, j] = act
    return dict(
        veh_kin=torch.from_numpy(kin),
        veh_mask=torch.from_numpy(mask),
        is_cav=torch.from_numpy(is_cav),
        slot_id=torch.from_numpy(slot_id),
        action_code=torch.from_numpy(action),
        counter=torch.zeros((b_n, 2, 30)),
        t0_s=0.0,
        lane_names=tuple(geom["lane_names"]),
    )


def j4_smoke_contract(*, n_steps: int = 120) -> dict:
    """The frozen smoke parameters (60 s = 120 ticks)."""
    return dict(batch=2, slots=32, n_steps=int(n_steps), dt_s=0.5,
                device="cpu", n_samples=1, seed=0)
