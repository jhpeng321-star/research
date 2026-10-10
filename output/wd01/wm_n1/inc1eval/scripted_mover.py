"""WM-side adapter for the shared R5/R6 generators (ADDENDUM_2 A2.2/A2.3).

R5 and R6 live in ONE shared package (``inc1_gen``, gated by
``INC1_GENERATORS.json``).  This module is the WM call site: a
``ctrl.mover.CavMover``-compatible mover that renders the rollout's tensor
state through the shared WM-side adapter
(:func:`inc1_gen.adapters.states_from_wm_arrays`) and returns the generator's
per-tick acceleration as a commanded target speed, so the Idea A and the Part A
v2g engines both drive the SAME in-the-loop CAV mover.

Nothing here re-implements a rule; the speed/headway targets, the window and
the membership live only in ``inc1_gen.generators``.  Numpy/torch only.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from ctrl.mover import CavStepOutput


def _nan_col(values, n):
    return [math.nan if v is None else float(v) for v in values][:n]


def wm_frame_from_states(states) -> tuple[list[str], dict]:
    """Render ``CavState``/``VehState``-like records into WM array columns.

    The reverse of :func:`inc1.scripted.to_traci_rows`; both render the SAME
    physical truth, so the shared generator must see identical records.
    """
    ids = [str(getattr(s, "vehicle_id")) for s in states]
    return ids, {
        "s_down": [float(getattr(s, "s_down_m", getattr(s, "s_down", 0.0)))
                   for s in states],
        "v": [float(getattr(s, "speed_mps", getattr(s, "v", 0.0)))
              for s in states],
        "a": [float(getattr(s, "accel_mps2", getattr(s, "a", 0.0)) or 0.0)
              for s in states],
        "lane_phys": [int(getattr(s, "lane_phys")) for s in states],
        "is_cav": [bool(getattr(s, "is_cav", True)) for s in states],
        "present": [bool(getattr(s, "present", True)) for s in states],
        "native_accel": _nan_col(
            [getattr(s, "native_accel_mps2", None) for s in states], len(ids)),
        "leader_gap": _nan_col(
            [getattr(s, "leader_gap_m", None) for s in states], len(ids)),
        "leader_v": _nan_col(
            [getattr(s, "leader_v_mps", None) for s in states], len(ids)),
    }


class ScriptedCavMover:
    """A ``CavMover``-compatible R5/R6 mover backed by the shared generator.

    ``step`` renders the batched :class:`ctrl.mover.CavStepInput` into the WM
    adapter's array frame, latches the generator membership on the first call
    (the t0 state by construction) and returns the generator's acceleration as
    a commanded target speed ``v + a * dt``.  Vehicles the generator releases
    (or an infeasible generator) return NaN -> native.
    """

    def __init__(self, arm: str, obstacle_s_m, *, t0_s: float = 0.0,
                 control_dt_s: float = 1.0,
                 control_t_s: float | None = None) -> None:
        if arm not in ("R5", "R6"):
            raise ValueError("ScriptedCavMover arm must be 'R5' or 'R6'")
        self.arm = str(arm)
        self.obstacle_s_m = float(obstacle_s_m)
        self.t0_s = float(t0_s)
        self.control_dt_s = float(control_dt_s)
        self.control_t_s = control_t_s
        self.generator = None

    # -- adapter ---------------------------------------------------------
    @staticmethod
    def _to_numpy(t: Tensor):
        return t.detach().to("cpu").numpy()

    def frame_from_inp(self, inp, *, leader_gap=None, leader_v=None,
                       native_accel=None) -> tuple[list[str], dict]:
        """Build the WM array frame from a batched ``CavStepInput``.

        Slot ids are ``slot{slot_id}``.  ``leader_gap`` / ``leader_v`` /
        ``native_accel`` may be supplied explicitly (a list aligned with the
        flattened slots); otherwise leaders are derived from the same-lane
        geometry and ``native_accel`` is NaN (unknown -> never accelerate).
        """
        s = self._to_numpy(inp.s_m).reshape(-1)
        v = self._to_numpy(inp.v_mps).reshape(-1)
        a = self._to_numpy(inp.a_mps2).reshape(-1) if getattr(
            inp, "a_mps2", None) is not None else None
        lane = self._to_numpy(inp.lane).reshape(-1).astype(int)
        present = self._to_numpy(inp.present).reshape(-1).astype(bool)
        is_cav = self._to_numpy(inp.is_cav).reshape(-1).astype(bool)
        slot_id = (self._to_numpy(inp.slot_id).reshape(-1).astype(int)
                   if getattr(inp, "slot_id", None) is not None
                   else list(range(s.size)))
        ids = [f"slot{int(sid)}" for sid in slot_id]
        if leader_gap is None:
            leader_gap, leader_v = self._derive_leaders(s, v, lane, present)
        return ids, {
            "s_down": s.tolist(), "v": v.tolist(),
            "a": (a.tolist() if a is not None else [0.0] * s.size),
            "lane_phys": lane.tolist(), "is_cav": is_cav.tolist(),
            "present": present.tolist(),
            "native_accel": (native_accel if native_accel is not None
                             else [math.nan] * s.size),
            "leader_gap": leader_gap, "leader_v": leader_v,
        }

    @staticmethod
    def _derive_leaders(s, v, lane, present):
        """Nearest same-lane leader ahead of each slot (bumper gap, speed)."""
        n = s.size
        gap = [math.nan] * n
        lv = [math.nan] * n
        for i in range(n):
            if not present[i]:
                continue
            ahead = [j for j in range(n)
                     if j != i and present[j] and lane[j] == lane[i]
                     and s[j] > s[i]]
            if not ahead:
                continue
            j = min(ahead, key=lambda k: s[k])
            gap[i] = float(max(0.0, s[j] - s[i]))
            lv[i] = float(v[j])
        return gap, lv

    # -- CavMover interface ---------------------------------------------
    def init_state(self, batch: int, slots: int = 128, *,
                   device=None) -> dict:
        return {"generator": None}

    @torch.no_grad()
    def step(self, state, inp, actions=None, *, wall_speed_mps=None):
        from inc1_gen.generators import make_generator
        from inc1_gen.states import ObstacleRecord

        ids, kwargs = self.frame_from_inp(inp)
        if self.generator is None:
            states = self._render(ids, kwargs)
            obstacle = ObstacleRecord(
                obstacle_id="incident.obstacle", lane_phys=1,
                s_down=self.obstacle_s_m)
            self.generator = make_generator(
                self.arm, states, obstacle, t0_s=self.t0_s,
                control_dt_s=self.control_dt_s)
            state["generator"] = self.generator
        now_s = (float(inp.control_t_s) if self.control_t_s is None
                 and getattr(inp, "control_t_s", None) is not None
                 else float(inp.t_s))
        step = self.generator.step(now_s, self._render(ids, kwargs))

        shape = tuple(inp.s_m.shape)
        target = torch.full(shape, float("nan"), dtype=torch.float64,
                            device=inp.s_m.device)
        accel = torch.full(shape, float("nan"), dtype=torch.float64,
                           device=inp.s_m.device)
        reason = torch.full(shape, 1, dtype=torch.int64, device=inp.s_m.device)
        v_flat = self._to_numpy(inp.v_mps).reshape(-1)
        for i, vid in enumerate(ids):
            if vid not in step.commands:
                continue
            a_cmd = float(step.commands[vid])
            accel.view(-1)[i] = a_cmd
            target.view(-1)[i] = float(v_flat[i]) + a_cmd * self.control_dt_s
            reason.view(-1)[i] = 0
        return CavStepOutput(target_speed_mps=target, accel_cmd_mps2=accel,
                             reason_code=reason, state=state)

    @staticmethod
    def _render(ids, kwargs):
        from inc1_gen.adapters import states_from_wm_arrays
        return states_from_wm_arrays(ids, **kwargs)

    def commands_from_states(self, now_s: float, states) -> dict:
        """WM-side call site for an explicit state sequence (identity test).

        Renders ``states`` through the WM array adapter and returns the shared
        generator's ``{vehicle_id: accel}`` at this tick.  This is the exact
        rendering :meth:`step` uses once the engine hands it a tick.
        """
        from inc1_gen.generators import make_generator
        from inc1_gen.states import ObstacleRecord

        ids, kwargs = wm_frame_from_states(states)
        if self.generator is None:
            obstacle = ObstacleRecord(obstacle_id="incident.obstacle",
                                      lane_phys=1, s_down=self.obstacle_s_m)
            self.generator = make_generator(
                self.arm, self._render(ids, kwargs), obstacle,
                t0_s=self.t0_s, control_dt_s=self.control_dt_s)
        step = self.generator.step(float(now_s), self._render(ids, kwargs))
        return dict(step.commands), tuple(step.native)


def feasibility_from_world(arm: str, world, obstacle_s_m, *,
                           t0_s: float = 0.0):
    """The R5/R6 generator feasibility on a rollout world's t0 state.

    Returns ``(feasible, reason, members)`` using the SAME shared generator the
    mover runs, so the predictor's arm set matches the executed response.
    """
    from inc1_gen.adapters import states_from_wm_arrays
    from inc1_gen.generators import make_generator
    from inc1_gen.states import ObstacleRecord

    ids = [f"slot{int(x)}" for x in
           world.slot_id.reshape(-1).detach().to("cpu").numpy()]
    kwargs = {
        "s_down": world.s_m.reshape(-1).detach().to("cpu").numpy().tolist(),
        "v": world.v_mps.reshape(-1).detach().to("cpu").numpy().tolist(),
        "a": world.a_mps2.reshape(-1).detach().to("cpu").numpy().tolist(),
        "lane_phys": world.lane.reshape(-1).detach().to("cpu").numpy().tolist(),
        "is_cav": world.is_cav.reshape(-1).detach().to("cpu").numpy().tolist(),
        "present": world.present.reshape(-1).detach().to("cpu").numpy().tolist(),
        "native_accel": [math.nan] * len(ids),
    }
    states = states_from_wm_arrays(ids, **kwargs)
    obstacle = ObstacleRecord(obstacle_id="incident.obstacle", lane_phys=1,
                              s_down=float(obstacle_s_m))
    gen = make_generator(arm, states, obstacle, t0_s=float(t0_s))
    members = tuple(getattr(gen, "_members", ()))
    return bool(gen.feasible), gen.infeasible_reason, members
