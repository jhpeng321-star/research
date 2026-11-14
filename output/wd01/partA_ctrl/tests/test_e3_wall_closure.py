"""E3b: batched wall-v2.1 closure law == WallGroupSession.tick (1e-9),
assembly and hold; constants pinned to the reference module."""
from types import SimpleNamespace

import numpy as np
import torch

import emulators
import refload

refload.load_reference()

from traffic_shepherd import wall_group as wg  # noqa: E402
from traffic_shepherd.wall_group import WallGroupSession  # noqa: E402


def _frames(ids, speeds, pos, now):
    return {v: SimpleNamespace(vehicle_id=v, timestamp_s=now, speed_mps=float(speeds[i]),
                               distance_to_merge_m=-float(pos[i]), lane_index=i)
            for i, v in enumerate(ids)}


def test_constants_pinned_to_reference():
    assert emulators.WALL_CLOSE_MAX_DELTA_MPS == wg.ASSEMBLY_CLOSE_MAX_DELTA_MPS
    assert emulators.WALL_CLOSE_GAIN_PER_S == wg.ASSEMBLY_CLOSE_GAIN_PER_S
    assert emulators.WALL_ASSEMBLY_TIMEOUT_S == wg.ASSEMBLY_TIMEOUT_S
    assert emulators.WALL_ASSEMBLY_SPAN_TOL_M == wg.ASSEMBLY_SPAN_TOL_M
    assert emulators.WALL_LAW_VERSION == wg.WALL_LAW_VERSION


def test_random_batches_parity_assembly_and_hold():
    rng = np.random.default_rng(2026)
    worst = 0.0
    n_checked = 0
    for trial in range(300):
        k = int(rng.integers(2, 5))
        ids = [f"m{i}" for i in range(k)]
        v_wall = float(rng.uniform(3.0, 30.0))
        speeds = rng.uniform(0.0, 32.0, k)
        pos = rng.uniform(0.0, 75.0, k) + 1000.0
        dt = float(rng.choice([0.5, 1.0]))
        hold = bool(trial % 3 == 0)
        s = WallGroupSession("w", members=frozenset(ids), wall_speed_mps=v_wall)
        if hold:
            s.phase = wg.PHASE_HOLD
        cmds = s.tick(_frames(ids, speeds, pos, 0.0), 0.0, control_dt_s=dt)
        assert set(cmds) == set(ids)
        p = torch.tensor(pos, dtype=torch.float64)
        gap = emulators.wall_gap_to_front(
            p, torch.zeros(k, dtype=torch.int64),
            closure=torch.full((k,), not hold))
        acc = emulators.wall_pace(torch.full((k,), v_wall, dtype=torch.float64),
                                  torch.tensor(speeds, dtype=torch.float64), dt,
                                  gap_to_front_m=gap).numpy()
        for i, v in enumerate(ids):
            worst = max(worst, abs(cmds[v] - float(acc[i])))
            n_checked += 1
    assert n_checked > 700
    assert worst <= 1e-9, worst


def test_multi_group_front_is_per_group():
    pos = torch.tensor([10.0, 40.0, 0.0, 5.0, 999.0], dtype=torch.float64)
    grp = torch.tensor([0, 0, 1, 1, -1])
    gap = emulators.wall_gap_to_front(pos, grp)
    assert gap.tolist() == [30.0, 0.0, 5.0, 0.0, 0.0]
    gap2 = emulators.wall_gap_to_front(pos, grp, closure=torch.tensor(
        [True, True, False, False, True]))
    assert gap2.tolist() == [30.0, 0.0, 0.0, 0.0, 0.0]
