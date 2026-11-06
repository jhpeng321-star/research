"""Backend protocol for the qualification suite + the keep-speed baseline + test backends.

A backend turns a :class:`~tools.wzqual.traj.StartState` and named arms (member specs for
:class:`~tools.wzqual.moves.MoveExecutor`; ``[]`` = base, all CAVs native) into predicted
trajectories::

    backend.rollout(start, {"base": [], "group": [...], "solo:main.7": [...]},
                    horizon_s=300, n_samples=8, seed=...) -> {arm: [Traj per sample]}

Contract (checked by run_qual): every Traj starts with the t0 frame and has 0.5 s steps to t0+H;
sample ``i`` of every arm uses the SAME random draws (common random numbers); members run closed
loop (commands from the executor on the backend's own predicted state, never a recorded path);
non-member CAVs are native; ``backlog`` is the source backlog per step or None (source not
modelled: J is then compared on the t0 cohort, see WM_QUAL_GATE.md section 4).
``supported_roles`` lists the move roles the backend can execute; candidates with other roles
are excluded from the action-response check and counted as unsupported.
"""
from __future__ import annotations

from typing import Mapping, Protocol

import numpy as np

from . import frame as F
from .moves import MoveExecutor
from .traj import StartState, Traj


class Backend(Protocol):
    name: str
    stochastic: bool
    models_entries: bool      # vehicles entering after t0 are predicted (required for D1-D4)
    closed_loop: bool         # members are driven by the executor on predicted states
    supported_roles: frozenset

    def rollout(self, start: StartState, arms: Mapping[str, list], *, horizon_s: float,
                n_samples: int, seed: int) -> dict:
        ...


def _traj_from_arrays(t0, steps, ids, is_cav, S, L, V, P, meta) -> Traj:
    k_idx, v_idx = np.nonzero(P)
    return Traj(t0 + F.DT * np.arange(steps + 1), k_idx, v_idx, S[k_idx, v_idx], L[k_idx, v_idx],
                V[k_idx, v_idx], ids, is_cav, None, meta)


class KeepSpeedBackend:
    """Persistence baseline: every vehicle keeps its t0 speed and lane (no interaction); members
    execute their commands kinematically (accel for 1 s, lane switch next step); closing-lane
    vehicles move to WM lane 1 at the drop; vehicles leave at the road end.  Entries: the
    observed entry rate of the last 300 s before t0 continues as evenly spaced arrivals at the
    road entry (lanes round-robin, speed = median t0 speed within 500 m of the entry, else 28).
    Deterministic; no source backlog."""

    name = "keep_speed"
    stochastic = False
    models_entries = True
    closed_loop = True
    supported_roles = frozenset({"hold", "pulse", "lane_to", "merge_early", "zip_merge", "zip_yield"})

    def _entries(self, start: StartState, horizon_s: float):
        et = np.asarray(start.entry_times_s, float)
        rate = float(((et > start.t0 - 300.0) & (et <= start.t0)).sum()) / 300.0
        fr = start.frame
        near = np.asarray(fr["s"]) < F.ENTRY_S + 500.0
        v_in = float(np.median(np.asarray(fr["v"])[near])) if near.any() else 28.0
        if rate <= 0:
            return [], v_in
        times = start.t0 + (np.arange(int(rate * horizon_s)) + 0.5) / rate
        return [float(t) for t in times if t <= start.t0 + horizon_s], v_in

    def rollout(self, start: StartState, arms, *, horizon_s, n_samples=1, seed=0):
        fr = start.frame
        arrivals, v_in = self._entries(start, horizon_s)
        ids = list(fr["ids"]) + [f"__entry_{j}" for j in range(len(arrivals))]
        n0, n = len(fr["ids"]), len(ids)
        is_cav = np.concatenate([np.asarray(fr["is_cav"], bool), np.zeros(n - n0, bool)])
        steps = int(round(horizon_s / F.DT))
        arr_step = {int(np.ceil((t - start.t0) / F.DT - 1e-9)): j for j, t in enumerate(arrivals)}
        out = {}
        for name, members in arms.items():
            ex = MoveExecutor(members, t0=start.t0)
            idx = {vid: i for i, vid in enumerate(ids)}
            watch = set(ex.ids) | {m["params"].get(k) for m in ex.members for k in ("anchor", "partner")}
            s = np.concatenate([np.asarray(fr["s"], np.float64), np.full(n - n0, F.ENTRY_S)])
            v = np.concatenate([np.asarray(fr["v"], np.float64), np.full(n - n0, v_in)])
            lane = np.concatenate([np.asarray(fr["lane"], np.int64), np.arange(n - n0) % 3])
            live = np.concatenate([np.ones(n0, bool), np.zeros(n - n0, bool)])
            accel = np.zeros(n)
            S = np.zeros((steps + 1, n))
            V = np.zeros((steps + 1, n))
            L = np.zeros((steps + 1, n), np.int64)
            P = np.zeros((steps + 1, n), bool)
            S[0], V[0], L[0], P[0] = s, v, lane, live
            for k in range(steps):
                if k % 2 == 0:
                    view = {ids[i]: (s[i], lane[i], v[i]) for i in np.flatnonzero(live) if ids[i] in watch}
                    for vid, cmds in ex.control(k // 2, view).items():
                        i = idx[vid]
                        for kind, val in cmds:
                            if kind == "accel":
                                accel[i] = val
                            elif kind == "lane":
                                lane[i] = int(val)
                            elif kind == "release":
                                accel[i] = 0.0
                nv = np.where(live, np.clip(v + accel * F.DT, 0.0, F.SPEED_LIMIT), v)
                s = np.where(live, s + 0.5 * (v + nv) * F.DT, s)
                v = nv
                lane = np.where((lane == F.CLOSING_LANE) & (s >= F.DROP_S), 1, lane)
                live &= s <= F.ROAD_END_S
                j = arr_step.get(k + 1)
                if j is not None:
                    live[n0 + j] = True
                S[k + 1], V[k + 1], L[k + 1], P[k + 1] = s, v, lane, live
            traj = _traj_from_arrays(start.t0, steps, ids, is_cav, S, L, V, P,
                                     dict(backend=self.name, arm=name))
            out[name] = [traj] * max(1, int(n_samples))
        return out


class ReplayBackend:
    """TEST/ORACLE ONLY: returns the recorded truth for every arm (perfect base prediction, no
    action response).  Used to validate metric plumbing; never a qualification candidate."""

    name = "replay_oracle"
    stochastic = False
    models_entries = True
    closed_loop = False
    supported_roles = frozenset({"hold", "pulse", "tau", "lane_to", "merge_early", "zip_merge", "zip_yield"})

    def __init__(self, truth_by_world: Mapping[str, Traj]):
        self.truth = dict(truth_by_world)

    def rollout(self, start: StartState, arms, *, horizon_s, n_samples=1, seed=0):
        tr = self.truth[start.world].window(start.t0, start.t0 + horizon_s)
        return {name: [tr] * max(1, int(n_samples)) for name in arms}
