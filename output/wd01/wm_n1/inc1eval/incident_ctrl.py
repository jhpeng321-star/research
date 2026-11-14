"""R1-R4 incident response mirrors (registration section 4).

R1 and R2 are the two NEW controllers, implemented as batched target-speed
kernels in the ``partA_ctrl`` mirror style (explicit state, pure tensors, a
scalar reference used by the parity test).  R3 (CIRCLES damping) and R4 (wall
v2.2 merge-release) are NOT re-implemented: they select the upstream CAVs and
hand the existing, already-verified ``ctrl`` mirrors the same action codes a
normal Idea A run uses.

Arm definitions (registration section 4, verbatim):

* ``R1`` nearest-CAV stop: the nearest CAV upstream of the obstacle, in either
  mainline lane, within 500 m, brakes at ``-2.0 m/s^2`` to a stop, holds for
  30 s, then goes native.  Other CAVs stay native.  No qualifying CAV => the
  arm equals R0 (logged).
* ``R2`` early-slow: every CAV with ``s_down in [obstacle - 1000,
  obstacle - 200)`` at t0 tracks ``15 m/s`` (decel <= ``1.5 m/s^2``) for 120 s,
  then goes native.
* ``R3`` damping-upstream: every CAV upstream of the obstacle at t0 runs the
  CIRCLES damping controller for the 300 s.
* ``R4`` wall-v2.2-upstream: the wall law v2.2 assembled from the two nearest
  qualifying upstream CAVs; infeasible assembly is logged and the arm still
  runs native-equivalent.

Zero SUMO.  Nothing here reads an outcome.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from ctrl.config import ACTION_CODES
from ctrl.mover import CavStepOutput
from ctrl.wall_hook import MERGE_START_S_ROAD_M
from ctrl.wall_v22 import check_pair

# ---------------------------------------------------------------- constants
R1_DECEL_MPS2 = -2.0
R1_HOLD_S = 30.0
R1_RANGE_M = 500.0
R2_TARGET_MPS = 15.0
R2_DECEL_MAX_MPS2 = 1.5
R2_WINDOW_S = 120.0
R2_LOWER_M = -1000.0
R2_UPPER_M = -200.0
R4_WALL_SPEED_FACTOR = 0.85     # J4 "mid" factor, the highest registered one
R4_N_MEMBERS = 2

MAINLINE_LANES = (0, 1)
_NATIVE_REASON = 1              # partA_ctrl.REASON_STRINGS[1] = selected_native
_ISSUED_REASON = 0

ARMS = ("R0", "R1", "R2", "R3", "R4", "R5", "R6")


# ------------------------------------------------------------- scalar refs
def r1_scalar_target(v0_mps: float, elapsed_s: float, *,
                     decel_mps2: float = R1_DECEL_MPS2,
                     hold_s: float = R1_HOLD_S) -> float:
    """Scalar R1 reference: ``-decel`` ramp to 0, hold, then native (NaN)."""
    v0 = float(v0_mps)
    decel = abs(float(decel_mps2))
    t_stop = (v0 / decel) if decel > 0 else 0.0
    if elapsed_s < t_stop - 1e-12:
        return max(0.0, v0 - decel * float(elapsed_s))
    if elapsed_s < t_stop + float(hold_s) - 1e-12:
        return 0.0
    return float("nan")


def r2_scalar_target(v0_mps: float, elapsed_s: float, *,
                     target_mps: float = R2_TARGET_MPS,
                     decel_mps2: float = R2_DECEL_MAX_MPS2,
                     window_s: float = R2_WINDOW_S) -> float:
    """Scalar R2 reference: ramp to the target at <= decel, hold, then native."""
    if elapsed_s >= float(window_s) - 1e-12:
        return float("nan")
    ramp = float(v0_mps) - abs(float(decel_mps2)) * float(elapsed_s)
    return max(float(target_mps), ramp)


# ------------------------------------------------------------- selections
def _present_cav(world) -> Tensor:
    return world.present.to(torch.bool) & world.is_cav.to(torch.bool)


@torch.no_grad()
def r1_select(world, obstacle_s_m, *, range_m: float = R1_RANGE_M
              ) -> Tensor:
    """``[B, K]`` bool: the nearest mainline CAV upstream within ``range_m``."""
    s = world.s_m.to(torch.float64)
    lane = world.lane.to(torch.int64)
    is_cav = _present_cav(world)
    obs = torch.as_tensor(obstacle_s_m, dtype=torch.float64,
                            device=s.device).reshape(-1, 1)
    if obs.shape[0] == 1 and s.shape[0] > 1:
        obs = obs.expand(s.shape[0], 1)
    mainline = (lane == MAINLINE_LANES[0]) | (lane == MAINLINE_LANES[1])
    upstream = is_cav & mainline & (s <= obs) & ((obs - s) <= float(range_m))
    masked = torch.where(upstream, s, torch.full_like(s, -1e9))
    best = masked.max(dim=1, keepdim=True).values
    return upstream & (masked >= best - 1e-9) & (best > -1e8)


@torch.no_grad()
def r2_select(world, obstacle_s_m, *, lower_m: float = R2_LOWER_M,
              upper_m: float = R2_UPPER_M) -> Tensor:
    """``[B, K]`` bool: CAVs with ``s_down in [obstacle+lower, obstacle+upper)``."""
    s = world.s_m.to(torch.float64)
    is_cav = _present_cav(world)
    obs = torch.as_tensor(obstacle_s_m, dtype=torch.float64,
                            device=s.device).reshape(-1, 1)
    if obs.shape[0] == 1 and s.shape[0] > 1:
        obs = obs.expand(s.shape[0], 1)
    rel = s - obs
    return is_cav & (rel >= float(lower_m)) & (rel < float(upper_m))


@torch.no_grad()
def upstream_select(world, obstacle_s_m) -> Tensor:
    """``[B, K]`` bool: every CAV strictly upstream of the obstacle at t0."""
    s = world.s_m.to(torch.float64)
    is_cav = _present_cav(world)
    obs = torch.as_tensor(obstacle_s_m, dtype=torch.float64,
                            device=s.device).reshape(-1, 1)
    if obs.shape[0] == 1 and s.shape[0] > 1:
        obs = obs.expand(s.shape[0], 1)
    return is_cav & (s < obs)


# ------------------------------------------------------------- R1/R2 mover
@dataclass
class IncidentCtrlState:
    """Persistent per-slot state of an incident response (t0 latch)."""

    active: Optional[Tensor] = None      # bool [B, K]
    t0_s: Optional[Tensor] = None        # float64 [B, K], NaN before latch
    v0_mps: Optional[Tensor] = None      # float64 [B, K]


class IncidentCavMover:
    """A ``CavMover``-compatible R1/R2 response controller.

    ``kind`` is ``"R1"`` or ``"R2"``.  The selection is latched on the first
    control tick (which is the t0 state by construction); every later tick
    returns the arm's target-speed schedule.  ``state`` is mutated in place so
    the t0 latch survives the engine's state handoff.
    """

    def __init__(self, kind: str, obstacle_s_m, *, range_m: float = R1_RANGE_M,
                 lower_m: float = R2_LOWER_M, upper_m: float = R2_UPPER_M,
                 device: Optional[torch.device] = None) -> None:
        if kind not in ("R1", "R2"):
            raise ValueError("IncidentCavMover kind must be 'R1' or 'R2'")
        self.kind = kind
        self.obstacle_s_m = obstacle_s_m
        self.range_m = float(range_m)
        self.lower_m = float(lower_m)
        self.upper_m = float(upper_m)
        self.device = device or torch.device("cpu")

    def init_state(self, batch: int, slots: int = 128, *,
                   device: Optional[torch.device] = None) -> IncidentCtrlState:
        dev = device or self.device
        shape = (int(batch), int(slots))
        return IncidentCtrlState(
            active=torch.zeros(shape, dtype=torch.bool, device=dev),
            t0_s=torch.full(shape, float("nan"), dtype=torch.float64,
                            device=dev),
            v0_mps=torch.zeros(shape, dtype=torch.float64, device=dev))

    def _select(self, inp) -> Tensor:
        from types import SimpleNamespace

        world = SimpleNamespace(s_m=inp.s_m, lane=inp.lane, present=inp.present,
                                is_cav=inp.is_cav)
        if self.kind == "R1":
            return r1_select(world, self.obstacle_s_m, range_m=self.range_m)
        return r2_select(world, self.obstacle_s_m, lower_m=self.lower_m,
                         upper_m=self.upper_m)

    @torch.no_grad()
    def step(self, state: IncidentCtrlState, inp, actions=None, *,
             wall_speed_mps=None) -> CavStepOutput:
        shape = tuple(inp.s_m.shape)
        dev = inp.s_m.device
        if state.active is None:
            state.active = torch.zeros(shape, dtype=torch.bool, device=dev)
            state.t0_s = torch.full(shape, float("nan"), dtype=torch.float64,
                                    device=dev)
            state.v0_mps = torch.zeros(shape, dtype=torch.float64, device=dev)
        t = float(inp.control_t_s if inp.control_t_s is not None else inp.t_s)
        if not bool(torch.isfinite(state.t0_s).any()):
            sel = self._select(inp)
            state.active = sel
            state.v0_mps = torch.where(
                sel, inp.v_mps.to(torch.float64),
                torch.zeros_like(inp.v_mps, dtype=torch.float64))
            state.t0_s = torch.where(
                sel, torch.full_like(state.t0_s, t), state.t0_s)
        elapsed = t - state.t0_s
        v0 = state.v0_mps
        if self.kind == "R1":
            decel = abs(R1_DECEL_MPS2)
            t_stop = v0 / decel
            target = torch.where(
                elapsed < t_stop,
                (v0 - decel * elapsed).clamp(min=0.0),
                torch.where(elapsed < t_stop + R1_HOLD_S,
                            torch.zeros_like(v0),
                            torch.full_like(v0, float("nan"))))
        else:
            ramp = v0 - R2_DECEL_MAX_MPS2 * elapsed
            target = torch.where(
                elapsed < R2_WINDOW_S, ramp.clamp(min=R2_TARGET_MPS),
                torch.full_like(v0, float("nan")))
        target = torch.where(state.active, target,
                             torch.full_like(target, float("nan")))
        speed = inp.v_mps.to(torch.float64)
        zeros = torch.zeros_like(speed)
        accel_cmd = torch.where(torch.isnan(target), zeros,
                                (target - speed) / 1.0)
        accel_cmd = torch.where(torch.isnan(target),
                                torch.full_like(accel_cmd, float("nan")),
                                accel_cmd)
        reason = torch.where(torch.isnan(target),
                             torch.full_like(inp.lane, _NATIVE_REASON),
                             torch.full_like(inp.lane, _ISSUED_REASON))
        return CavStepOutput(
            target_speed_mps=target.reshape(shape),
            accel_cmd_mps2=accel_cmd.reshape(shape),
            reason_code=reason,
            state=state)


# ------------------------------------------------------------- R3 / R4 arms
def r3_action_codes(world, obstacle_s_m) -> Tensor:
    """``[B, K]`` action codes: damping on every upstream CAV, native else."""
    sel = upstream_select(world, obstacle_s_m)
    codes = torch.full_like(world.lane, ACTION_CODES["native"])
    codes = torch.where(sel, torch.full_like(codes, ACTION_CODES["damping"]),
                        codes)
    return codes


def r4_members(world, obstacle_s_m, *, n_members: int = R4_N_MEMBERS):
    """The two nearest qualifying upstream CAVs, one per mainline lane.

    Returns ``(slots, refusal)`` where ``slots`` is a list of ``(lane, idx)``
    in the chosen lanes (empty when a lane lacks an upstream CAV) and
    ``refusal`` is the ``check_pair`` record when assembly is infeasible.
    """
    if int(world.batch) != 1:
        raise ValueError("r4_members expects a single-world batch")
    s = world.s_m[0].to(torch.float64)
    lane = world.lane[0].to(torch.int64)
    is_cav = _present_cav(world)[0]
    obs = float(torch.as_tensor(obstacle_s_m).reshape(-1)[0])
    chosen = []
    for ln in MAINLINE_LANES:
        cand = is_cav & (lane == ln) & (s < obs)
        if not bool(cand.any()):
            return [], {"reason": "no_upstream_cav_in_lane", "lane": int(ln)}
        s_masked = torch.where(cand, s, torch.full_like(s, -1e18))
        idx = int(torch.argmax(s_masked).item())
        chosen.append((int(ln), idx))
    return chosen, None


def r4_assembly(world, obstacle_s_m, wall_speed_mps: float):
    """``(members, feasible, refusal)`` for the R4 v2.2 wall proposal."""
    members, refusal = r4_members(world, obstacle_s_m)
    if refusal is not None:
        return [], False, refusal
    lanes = [ln for ln, _ in members]
    pos_road = [float(world.s_m[0, idx]) + MERGE_START_S_ROAD_M
                for _, idx in members]
    gate = check_pair(lanes, pos_road, float(wall_speed_mps))
    feasible = bool(gate.get("feasible"))
    return members, feasible, gate.get("refusal")


def r4_wall_speed(world, members, *, factor: float = R4_WALL_SPEED_FACTOR,
                  free_flow_mps: Optional[float] = None) -> float:
    """The registered-factor wall target speed for the chosen members.

    ``free_flow_mps`` (the segment limit at the lead member) is used when
    given; otherwise the mean t0 member speed is used as the reference.  The
    factor itself is a lock-recorded parameter (never tuned on outcomes).
    """
    if not members:
        raise ValueError("r4_wall_speed needs at least one member")
    if free_flow_mps is not None:
        ref = float(free_flow_mps)
    else:
        ref = float(sum(float(world.v_mps[0, idx]) for _, idx in members)
                    / len(members))
    return max(1.0, float(factor) * ref)


def r4_action_codes(world, members) -> Tensor:
    """``[B, K]`` action codes: wall on the members, native elsewhere."""
    codes = torch.full_like(world.lane, ACTION_CODES["native"])
    for _, idx in members:
        codes[0, idx] = ACTION_CODES["wall"]
    return codes


def r4_wall_speeds(world, members, wall_speed_mps: float) -> Tensor:
    """``[B, K]`` wall target speeds (NaN off the wall members)."""
    speeds = torch.full_like(world.s_m, float("nan"), dtype=torch.float64)
    for _, idx in members:
        speeds[0, idx] = float(wall_speed_mps)
    return speeds


def is_finite_scalar(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):                       # pragma: no cover
        return False
