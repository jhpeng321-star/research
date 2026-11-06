"""Proposer-path backend (SECONDARY, amendment 1): WM-A's ``wms.planapi`` (load/observe/evaluate),
the law WM-C ranks with.  J-only (no trajectories), hold roles only.

``observability='proposer'`` feeds exactly WM-C's moment input (all CAV self-states at t0, no
neighbours, no entrance counter); ``'full'`` adds every t0 vehicle as a detected neighbour and the
entrance counter (upper bound).  Rows are given in SUMO coordinates (edge, SUMO lane, global x),
re-projected by planapi with the bound road.
"""
from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

from .moves import sumo_view
from .traj import StartState


class PlanApiBackend:
    stochastic = True
    closed_loop = True
    supported_roles = frozenset({"hold"})

    def __init__(self, checkpoint: str, *, device: str = "cpu", map_spec: Optional[str] = None,
                 observability: str = "proposer", max_rows: int = 64):
        from wms import planapi
        if observability not in ("proposer", "full"):
            raise ValueError(observability)
        self.api = planapi
        self.h = planapi.load(checkpoint, map_spec=map_spec, device=device, max_rows=max_rows)
        self.observability = observability
        self.name = f"planapi_{observability}"
        self.models_entries = observability == "full"

    def moment(self, start: StartState) -> dict:
        fr = start.frame
        rows = []
        for vid, s, ln, v, c in zip(fr["ids"], fr["s"], fr["lane"], fr["v"], fr["is_cav"]):
            edge, sl, x = sumo_view(float(s), int(ln))
            rows.append(dict(id=vid, edge=edge, lane=sl, x=x, v=float(v), is_cav=bool(c)))
        m = dict(map="wz3to2", t0=start.t0, cavs=[r for r in rows if r["is_cav"]])
        if self.observability == "full":
            m["vehicles"] = rows
            m["counter"] = start.counter().tolist()
        return m

    def evaluate_J(self, start: StartState, arms: Mapping[str, list], *, horizon_s: float,
                   n_samples: int, seed: int) -> dict:
        state = self.api.observe(self.h, self.moment(start))
        actions = [dict(id=name, kind="base" if not members else "group",
                        members=[dict(id=m["id"], role=m.get("role", "hold"), params=dict(m["params"]))
                                 for m in members])
                   for name, members in arms.items()]
        J = self.api.evaluate(self.h, state, actions, K=int(n_samples), H=float(horizon_s), seed=int(seed))
        return {name: np.asarray(J[name], float) for name in arms}
