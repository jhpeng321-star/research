"""P1 batched per-step CAV mover over the rollout state (Idea A step (a)).

Wraps the verified ``partA_ctrl`` mirrors (native / CIRCLES damping / PI
external-U / wall law) into one batched control tick:

* native -> the returned target speed is NaN (no command); the baseline IDM
  then drives the CAV exactly like an HDV;
* damping -> the CIRCLES ONNX policy on the (ego, leader, bumper gap) triple
  through the direct-target bridge;
* pi -> the PI external-U recurrence with explicit per-slot state;
* wall -> the wall law through the direct-target bridge.  ``wall-v2.1``
  delegates to the exact J4 mirrors (``WALL_LAW_VERSION =
  "wall-v2.1-closure"``); ``wall-v2.2`` (``ctrl.wall_v22``, the frozen
  merge-release transcription) adds the merge-release rule: in HOLD only,
  when the LEAD member's edge-aware ``s_road`` reaches the merge start
  ``2644.2 m`` (X = 0), the wall is released to native ONCE per wall and
  the ``merge_release`` event is emitted on
  :attr:`CavStepOutput.merge_release_events`; the hold is capped there
  (:func:`ctrl.wall_v22.hold_cap_s`) and the proposer refuses pairs whose
  lead is within ``D_min = 70 * v_wall`` of the merge start
  (:func:`ctrl.wall_v22.dmin_refusal`, reason ``dmin_unreachable``).

``step`` is a control tick (1 s); the returned target speed is executed by the
engine with :func:`execution_speed` once per control tick and held for the
two 0.5 s physics steps, exactly as ``partA_v2g.engine`` does.

No controller is learned or approximated: every strategy path is a call
into the E1-E5-validated ``partA_ctrl`` mirrors, batched over the ``[B, K]``
slot grid.  The mirror module is imported read-only (see ``_bridge.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .config import NATIVE, NONE, WALL, CtrlConfig
from .wall_hook import MERGE_START_S_ROAD_M, WALL_LAW_V21, resolve_wall_law

# wall phases carried by the engine on CavStepInput.wall_phase
WALL_PHASE_NONE = 0
WALL_PHASE_ASSEMBLY = 1
WALL_PHASE_HOLD = 2

# batched-world lane ends (frozen network; partA_v2g params geometry
# lane_lengths_m = [3199.0, 3197.55, 454.1] for main_left/main_right/accel)
_LANE_END_M = (3199.0, 3197.55, 454.1)

_REASON_SELECTED_NATIVE = 1  # partA_ctrl.emulators.REASON_STRINGS[1]


@dataclass
class CavStepInput:
    """Batched world view handed to one control tick.

    ``segment_lo_m`` / ``segment_hi_m`` / ``segment_limit_mps`` carry the
    CAV's current speed-limit segment (the segment-local sensing convention of
    partA_ctrl).  All tensors are ``[B, K]``; ``lane`` is 0/1/2 or -1.

    Optional explicit ideal-sensor fields (``front_*`` / ``rear_*``) mirror
    ``partA_ctrl.emulators.FrameBatch`` per slot; when they are absent the
    mover derives same-lane neighbours from the batched state itself (the
    model-world approximation; recorded-run parity passes the explicit
    SUMO lane-local frames).  ``wall_phase`` is the engine-carried wall
    lifecycle per slot (0 none / 1 assembly / 2 hold).
    """

    s_m: Tensor                 # float32/float64, merge-relative downstream
    v_mps: Tensor
    lane: Tensor                # int64
    length_m: Tensor
    present: Tensor             # bool
    is_cav: Tensor              # bool
    wall_group: Tensor          # int64, -1 = not a wall member
    segment_lo_m: Tensor
    segment_hi_m: Tensor
    segment_limit_mps: Tensor
    t_s: float
    control_t_s: Optional[float] = None
    # -- optional explicit ideal-sensor fields (all [B, K]) ---------------
    front_present: Optional[Tensor] = None
    front_gap_m: Optional[Tensor] = None
    front_rel_mps: Optional[Tensor] = None
    front_coverage_m: Optional[Tensor] = None
    rear_present: Optional[Tensor] = None
    rear_gap_m: Optional[Tensor] = None
    rear_rel_mps: Optional[Tensor] = None
    rear_coverage_m: Optional[Tensor] = None
    frame_present: Optional[Tensor] = None
    allowed_speed_mps: Optional[Tensor] = None
    wall_phase: Optional[Tensor] = None


@dataclass
class CtrlState:
    """Persistent per-slot controller state between control ticks.

    Shapes are ``[B, K]``.  ``wall_assembled_at_s`` / ``wall_safety_abort`` are
    per (world, wall-id) and carried by the engine, not here.
    ``merge_released`` latches the v2.2 merge-release handback per slot
    (the release fires ONCE: a merge-released slot is never re-enslaved
    even if the engine's phase report flips back); always ``False`` under
    wall v2.1 (no release rule).
    """

    pi_raw_prev: Tensor         # float64, last committed raw command
    pi_last_ts: Tensor          # float64, nan when invalidated
    pi_active: Tensor           # bool
    merge_released: Optional[Tensor] = None  # bool [B, K], v2.2 latch


@dataclass
class CavStepOutput:
    """One control tick's commands.

    ``target_speed_mps`` is NaN where no command was issued (native, fallback
    or out-of-contract sensing).  ``accel_cmd_mps2`` is the issued commanded
    acceleration of the direct-target bridge, ``(target - speed) / 1 s``
    (NaN with the target).  ``reason_code`` mirrors
    ``partA_ctrl.emulators.REASON_STRINGS``; ``None`` (0) means issued.
    ``merge_release_events`` lists the v2.2 ``merge_release`` wall events
    FIRED THIS TICK (once per (row, wall); see
    :func:`ctrl.wall_v22.merge_release_event`), ``None`` otherwise.
    """

    target_speed_mps: Tensor    # float64 [B, K], nan = no command
    accel_cmd_mps2: Tensor      # float64 [B, K], nan = no command
    reason_code: Tensor         # int64 [B, K]
    state: CtrlState
    merge_release_events: Optional[list] = None


@torch.no_grad()
def execution_speed(target_speed_mps: Tensor, speed_mps: Tensor,
                    idm_accel_mps2: Tensor, *,
                    config: Optional[CtrlConfig] = None) -> Tensor:
    """The measured Part A SUMO execution rule (partA_v2g/engine.py:549).

    ``v' = max(0, min(clip(T, v - b*dt, v + a*dt), v + idm_accel*dt))`` with
    ``b = -accel_min_mps2`` and ``a = accel_max_mps2``.  Vectorized; no state.
    """
    cfg = config or CtrlConfig()
    dt = float(cfg.dt_s)
    a_kin = float(cfg.accel_max_mps2)
    b_kin = -float(cfg.accel_min_mps2)
    kin = torch.clamp(target_speed_mps,
                      min=torch.clamp(speed_mps - b_kin * dt, min=0.0),
                      max=speed_mps + a_kin * dt)
    return torch.clamp(torch.minimum(kin, speed_mps + idm_accel_mps2 * dt),
                       min=0.0)


def _flatten(t: Tensor) -> Tensor:
    return t.reshape(-1)


class CavMover:
    """Batched per-CAV mover mirroring ``partA_ctrl.emulators``.

    Constructed with the CIRCLES model payload (sha-pinned) only when a
    damping/PI slot actually needs it, so native-only rollouts stay cheap.
    """

    def __init__(self, config: Optional[CtrlConfig] = None, *,
                 device: Optional[torch.device] = None,
                 wall_law=None) -> None:
        self.config = config or CtrlConfig()
        self.device = device or torch.device("cpu")
        self.wall_law = wall_law
        self._emu = None

    # ------------------------------------------------------------- mirrors
    def _emulators(self):
        """Lazily construct the partA_ctrl ControllerEmulators (ONNX load)."""
        if self._emu is None:
            from ._bridge import emulators, model_onnx

            self._emu = emulators().ControllerEmulators(
                str(model_onnx()), model_sha256=self.config.model_sha256,
                device=self.device)
        return self._emu

    def init_state(self, batch: int, slots: int = 128, *,
                   device: Optional[torch.device] = None) -> CtrlState:
        """Fresh per-slot controller state (all PI slots invalidated)."""
        dev = device or self.device
        shape = (int(batch), int(slots))
        return CtrlState(
            pi_raw_prev=torch.zeros(shape, dtype=torch.float64, device=dev),
            pi_last_ts=torch.full(shape, float("nan"), dtype=torch.float64,
                                  device=dev),
            pi_active=torch.zeros(shape, dtype=torch.bool, device=dev),
            merge_released=torch.zeros(shape, dtype=torch.bool, device=dev))

    # ------------------------------------------------------------- sensing
    def _auto_frames(self, inp: CavStepInput, dev: torch.device):
        """Same-lane ideal-sensor frames derived from the batched state.

        Model-world approximation: neighbours are the nearest same-``lane``
        slots ahead/behind within the 100 m coverage
        (``min(100, lane_end - s)`` front, ``min(100, s - length)`` rear).
        Recorded-run parity passes the explicit SUMO lane-local frames via
        :class:`CavStepInput` instead.
        """
        s = inp.s_m.to(torch.float64)
        v = inp.v_mps.to(torch.float64)
        length = inp.length_m.to(torch.float64)
        lane = inp.lane.to(torch.int64)
        present = inp.present.to(torch.bool)
        sensor = float(self.config.sensor_range_m)
        lane_end = torch.tensor(_LANE_END_M, dtype=torch.float64, device=dev)
        front_cov = (lane_end[lane.clamp(min=0)] - s).clamp(0.0, sensor)
        rear_cov = (s - length).clamp(0.0, sensor)
        best_fg = torch.full_like(s, float("inf"))
        best_fv = torch.zeros_like(s)
        best_rg = torch.full_like(s, float("inf"))
        best_rv = torch.zeros_like(s)
        for j in range(int(s.shape[1])):
            sj = s[:, j:j + 1]                       # [B, 1] candidate
            vj = v[:, j:j + 1]
            lj = length[:, j:j + 1]
            pj = present[:, j:j + 1]
            same = (lane == lane[:, j:j + 1]) & pj   # candidate is live
            gap_f = sj - lj - s                      # [B, K]
            gap_r = s - length - sj
            upd_f = same & (gap_f >= 0) & (gap_f <= front_cov) \
                & (gap_f < best_fg)
            best_fg = torch.where(upd_f, gap_f, best_fg)
            best_fv = torch.where(upd_f, vj.expand_as(v), best_fv)
            upd_r = same & (gap_r >= 0) & (gap_r <= rear_cov) \
                & (gap_r < best_rg)
            best_rg = torch.where(upd_r, gap_r, best_rg)
            best_rv = torch.where(upd_r, vj.expand_as(v), best_rv)
        has_f = torch.isfinite(best_fg)
        has_r = torch.isfinite(best_rg)
        zeros = torch.zeros_like(s)
        return dict(
            front_present=has_f,
            front_gap_m=torch.where(has_f, best_fg, zeros),
            front_rel_mps=torch.where(has_f, best_fv - v, zeros),
            front_coverage_m=front_cov,
            rear_present=has_r,
            rear_gap_m=torch.where(has_r, best_rg, zeros),
            rear_rel_mps=torch.where(has_r, best_rv - v, zeros),
            rear_coverage_m=rear_cov,
        )

    # ---------------------------------------------------------------- step
    @torch.no_grad()
    def step(self, state: CtrlState, inp: CavStepInput, actions: Tensor, *,
             wall_speed_mps: Optional[Tensor] = None) -> CavStepOutput:
        """One batched control tick.

        Parameters
        ----------
        state:
            :class:`CtrlState` from :meth:`init_state` / a previous tick.
        inp:
            :class:`CavStepInput` world view.
        actions:
            ``int64 [B, K]`` per-slot action codes (:data:`ctrl.config.
            ACTION_CODES`); only ``is_cav`` slots are read.  ``native`` and
            non-CAV slots get NaN targets.
        wall_speed_mps:
            ``float`` ``[B, K]`` wall target speed for wall slots (NaN / None
            elsewhere).

        Returns
        -------
        :class:`CavStepOutput`.
        """
        dev = inp.s_m.device
        from ._bridge import emulators as _emulators_mod

        emu_mod = _emulators_mod()
        shape = tuple(inp.s_m.shape)
        n = inp.s_m.numel()
        zeros = torch.zeros(n, dtype=torch.float64, device=dev)
        act = _flatten(actions).to(torch.int64)
        is_cav = _flatten(inp.present.to(torch.bool) &
                          inp.is_cav.to(torch.bool))
        speed = _flatten(inp.v_mps).to(torch.float64)
        allowed = (
            _flatten(inp.allowed_speed_mps).to(torch.float64)
            if inp.allowed_speed_mps is not None
            else _flatten(inp.segment_limit_mps).to(torch.float64))

        # ---- frames (explicit or derived) -> partA_ctrl FrameBatch --------
        if inp.front_present is not None:
            f = dict(
                front_present=_flatten(inp.front_present).to(torch.bool),
                front_gap_m=_flatten(inp.front_gap_m).to(torch.float64),
                front_rel_mps=_flatten(inp.front_rel_mps).to(torch.float64),
                front_coverage_m=_flatten(inp.front_coverage_m).to(torch.float64),
                rear_present=_flatten(inp.rear_present).to(torch.bool),
                rear_gap_m=_flatten(inp.rear_gap_m).to(torch.float64),
                rear_rel_mps=_flatten(inp.rear_rel_mps).to(torch.float64),
                rear_coverage_m=_flatten(inp.rear_coverage_m).to(torch.float64),
            )
        else:
            f = {k: _flatten(t) for k, t
                 in self._auto_frames(inp, dev).items()}
        fp_present = (_flatten(inp.frame_present).to(torch.bool)
                      if inp.frame_present is not None
                      else torch.ones(n, dtype=torch.bool, device=dev))
        batch = emu_mod.FrameBatch(
            now_s=torch.full((n,), float(inp.t_s), dtype=torch.float64,
                             device=dev),
            ts_s=torch.full((n,), float(inp.t_s), dtype=torch.float64,
                            device=dev),
            speed_mps=speed,
            allowed_speed_mps=allowed,
            front_coverage_m=f["front_coverage_m"],
            rear_coverage_m=f["rear_coverage_m"],
            front_present=f["front_present"],
            front_gap_m=f["front_gap_m"],
            front_rel_mps=f["front_rel_mps"],
            rear_present=f["rear_present"],
            rear_gap_m=f["rear_gap_m"],
            rear_rel_mps=f["rear_rel_mps"],
            frame_present=fp_present,
        )

        # ---- damping + PI rows through the verified emulate() -------------
        kind = torch.where(is_cav, act, torch.full_like(act, NATIVE))
        kind = torch.where(kind == WALL, torch.full_like(kind, NATIVE), kind)
        kind = torch.where((kind == NONE) | (kind == NATIVE),
                           torch.full_like(kind, NATIVE), kind)
        needs_emu = bool(((kind == 1) | (kind == 2)).any())
        if needs_emu:
            emu = self._emulators()
            pi_state = emu_mod.ControllerState(pi=emu_mod.PIState(
                raw_prev=_flatten(state.pi_raw_prev).to(torch.float64),
                last_ts=_flatten(state.pi_last_ts).to(torch.float64),
                active=_flatten(state.pi_active).to(torch.bool)))
            out = emu.emulate(kind, pi_state, batch)
            target = out["target_speed_mps"]
            codes = out["reason_code"]
            new_pi = out["new_state"].pi
        else:
            target = torch.full((n,), float("nan"), dtype=torch.float64,
                                device=dev)
            codes = torch.full((n,), _REASON_SELECTED_NATIVE, dtype=torch.int64,
                               device=dev)
            new_pi = emu_mod.PIState(
                raw_prev=_flatten(state.pi_raw_prev).to(torch.float64),
                last_ts=_flatten(state.pi_last_ts).to(torch.float64),
                active=_flatten(state.pi_active).to(torch.bool))

        # ---- wall rows (v2.1 mirrors or the registered v2.2 law) ----------
        is_wall = is_cav & (act == WALL)
        merge_events: Optional[list] = None
        latch = _flatten(state.merge_released).to(torch.bool)
        new_latch = latch
        if is_wall.any():
            if wall_speed_mps is None:
                raise ValueError("wall slots require wall_speed_mps")
            w_speed = _flatten(wall_speed_mps).to(torch.float64)
            group = _flatten(inp.wall_group).to(torch.int64)
            if inp.wall_phase is not None:
                phase = _flatten(inp.wall_phase).to(torch.int64)
            else:
                phase = torch.full((n,), WALL_PHASE_HOLD, dtype=torch.int64,
                                   device=dev)
            hold = phase == WALL_PHASE_HOLD
            assembly = phase == WALL_PHASE_ASSEMBLY
            s_road = _flatten(inp.s_m).to(torch.float64) + MERGE_START_S_ROAD_M
            fresh = torch.zeros_like(hold)
            if self.config.wall_law == WALL_LAW_V21:
                gap = emu_mod.wall_gap_to_front(s_road, group,
                                                closure=assembly)
                accel = emu_mod.wall_pace(w_speed, speed, dt_s=1.0,
                                          gap_to_front_m=gap)
            else:
                law = self.wall_law
                if law is None:
                    law = resolve_wall_law(self.config.wall_law)
                gap = law.gap_to_front(s_road, group, closure=assembly)
                accel = law.pace(w_speed, speed, dt_s=1.0, gap_to_front_m=gap)
                release_fn = getattr(law, "release_mask", None)
                if release_fn is not None:
                    # v2.2 merge-release: HOLD-only, fires ONCE per wall
                    fresh = release_fn(s_road, group, hold).to(torch.bool)
                    newly = is_wall & fresh & ~latch
                    if bool(newly.any()):
                        from .wall_v22 import merge_release_event

                        k = shape[1]
                        merge_events = []
                        seen = set()
                        for i in torch.nonzero(
                                newly, as_tuple=False).reshape(-1).tolist():
                            key = (i // k, int(group[i].item()))
                            if key in seen:
                                continue
                            seen.add(key)
                            merge_events.append(merge_release_event(
                                wall_id=key[1],
                                timestamp_s=float(inp.t_s)))
                    new_latch = latch | (is_wall & fresh)
            released = is_wall & (fresh | latch)
            fp_codes = emu_mod.frame_problem_codes(batch)
            _, w_tgt = emu_mod.direct_target(speed, allowed, accel)
            wall_ok = (is_wall & (fp_codes == 0) & ~torch.isnan(w_tgt)
                       & ~released)
            target = torch.where(wall_ok, w_tgt, target)
            codes = torch.where(wall_ok, torch.zeros_like(codes), codes)
            wall_fb = is_wall & ~wall_ok
            codes = torch.where(wall_fb & (fp_codes != 0), fp_codes, codes)
            codes = torch.where(
                wall_fb & (fp_codes == 0) & ~released,
                torch.full_like(codes, 10), codes)  # infeasible_speed_bounds
            # released slots are handed back to native (merge-release rule)
            codes = torch.where(is_wall & released,
                                torch.full_like(codes, _REASON_SELECTED_NATIVE),
                                codes)

        # native rows keep selected_native / no command (emulate default)
        accel_cmd = torch.where(torch.isnan(target), zeros,
                                (target - speed) / float(
                                    self.config.control_dt_s))
        accel_cmd = torch.where(torch.isnan(target),
                                torch.full_like(accel_cmd, float("nan")),
                                accel_cmd)
        return CavStepOutput(
            target_speed_mps=target.reshape(shape),
            accel_cmd_mps2=accel_cmd.reshape(shape),
            reason_code=codes.reshape(shape),
            state=CtrlState(
                pi_raw_prev=new_pi.raw_prev.reshape(shape),
                pi_last_ts=new_pi.last_ts.reshape(shape),
                pi_active=new_pi.active.reshape(shape),
                merge_released=new_latch.reshape(shape)),
            merge_release_events=merge_events)
