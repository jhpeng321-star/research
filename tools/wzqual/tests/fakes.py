"""Synthetic wz3to2 tapes and trajectories for CPU tests (no SUMO, no WM)."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from tools.wzqual import frame as F
from tools.wzqual.traj import Traj


def synthetic_world(*, seed=989201, cell="D3200_p60", t_end=1200.0, n_veh=60, slow_after=None,
                    slow_speed=3.0, cav_every=3, headway_s=4.0, v0=28.0):
    """SUMO-frame arrays of a simple world: vehicles inserted every ``headway_s`` at x=0 in lane
    ``i % 3`` with speed v0; inside [X_B-300, X_B] they slow to ``slow_speed`` after ``slow_after``
    (creates a breakdown); closing-lane vehicles switch to lane 1 at the drop; exit at x=4000."""
    times = np.round(np.arange(1, int(t_end / F.DT) + 1) * F.DT, 6)
    rows = {k: [] for k in ("step", "vid", "edge", "lane", "x", "v", "a")}
    backlog = np.zeros(len(times), np.int32)
    inserted = np.zeros(len(times), np.int32)
    names, depart = [], []
    x = {}
    lane = {}
    for k, t in enumerate(times):
        nv = int(t // headway_s)
        while len(names) < min(nv, n_veh):
            i = len(names)
            names.append(f"main.{i}")
            depart.append(i * headway_s)
            x[i] = 0.0
            lane[i] = i % 3
        for i in list(x):
            in_zone = F.X_B - 300.0 <= x[i] <= F.X_B
            v = slow_speed if (slow_after is not None and t >= slow_after and in_zone) else v0
            x[i] += v * F.DT
            if x[i] >= F.X_B and lane[i] == 0:
                lane[i] = 1
            if x[i] > F.ROAD_END_X:
                del x[i]
                continue
            edge = 0 if x[i] < F.X_B else 1
            rows["step"].append(k)
            rows["vid"].append(i)
            rows["edge"].append(edge)
            rows["lane"].append(lane[i] if edge == 0 else (0 if lane[i] == 1 else 1))
            rows["x"].append(x[i])
            rows["v"].append(v)
            rows["a"].append(0.0)
        inserted[k] = len(names)
    d, p = F.parse_cell(cell)
    meta = dict(schema="wzqual_native_tape/1", map="wz3to2", seed=seed, demand={"main": d},
                penetration=p, cell=cell, t_end=t_end, split="WMQUAL", task="WM-D")
    return dict(time_s=times, step=np.asarray(rows["step"], np.int32), vid=np.asarray(rows["vid"], np.int32),
                edge=np.asarray(rows["edge"], np.int16), lane=np.asarray(rows["lane"], np.int16),
                x=np.asarray(rows["x"]), v=np.asarray(rows["v"]), a=np.asarray(rows["a"]),
                backlog=backlog, inserted=inserted, ids=np.asarray(names),
                is_cav=np.asarray([i % cav_every == 0 for i in range(len(names))]),
                depart_s=np.asarray(depart, float), edges=np.asarray(["up", "wz"]),
                meta_json=np.asarray(json.dumps(meta)))


def write_tape(path: Path, **kw) -> Path:
    arrays = synthetic_world(**kw)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    path.write_bytes(buf.getvalue())
    return path


def line_traj(speeds, *, t0=0.0, steps=10, lane=1, s0=0.0, cav=None):
    """Vehicles at s0 + 10 m spacing moving at constant ``speeds`` for ``steps`` 0.5 s steps."""
    n = len(speeds)
    cav = cav or [False] * n
    rows = dict(step=[], vid=[], s=[], lane=[], v=[])
    for k in range(steps + 1):
        for i, v in enumerate(speeds):
            rows["step"].append(k)
            rows["vid"].append(i)
            rows["s"].append(s0 - 10.0 * i + v * F.DT * k)
            rows["lane"].append(lane)
            rows["v"].append(v)
    return Traj(t0 + F.DT * np.arange(steps + 1), rows["step"], rows["vid"], rows["s"], rows["lane"],
                rows["v"], [f"v{i}" for i in range(n)], cav, None, {})
