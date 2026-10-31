"""S-RB-CF: faithful SUMO 1.27.1 ``MSCFModel_IDM`` car-following parity.

Every formula below is a line-by-line tensor transcription of SUMO 1.27.1
``src/microsim/cfmodels/MSCFModel_IDM.cpp`` / ``MSCFModel_IDM.h`` and of the
generic step composition in ``MSCFModel.cpp::finalizeSpeed`` /
``MSVehicle.cpp::planMoveInternal``+``accelerate`` (read-only source copies at
``E:/Temp/opencode/sumo_src`` and ``E:/Temp/opencode/sumo-1.27.1-rb-lane``;
no downloaded code is executed).  Sources and line anchors:

* ``MSCFModel_IDM.cpp:44``    ``myIterations = MAX2(1, int(TS / idmStepping + .5))``
  with the vType default ``cfParameter(cf.idmStepping) = 0.25`` s
  (``TS = 0.5`` s here -> 2 internal iterations; ``TS = 1.0`` -> 4).
* ``MSCFModel_IDM.cpp:45``    ``myTwoSqrtAccelDecel = 2*sqrt(myAccel*myDecel)``.
* ``MSCFModel_IDM.cpp:225-269`` ``_v``: the iterated IDM safe speed.  The SUMO
  caller passes the leader gap WITH ``minGap`` already subtracted
  (``MSVehicle.cpp:3209-3211``) and ``respectMinGap=true`` adds it back, so the
  interaction term is evaluated on the BUMPER gap while ``s*`` keeps ``+s0``
  (``stopSpeed`` calls ``_v`` with ``respectMinGap=false``: no ``s0`` at all).
  Per iteration: ``acc = a(1 - (v/desSpeed)^delta - (s/gap)^2)``,
  ``newSpeed = max(0, newSpeed + acc*TS/iterations)``, then the gap is advanced
  with the NEW speed difference ``gap -= max(0, (newSpeed-predSpeed)*TS/it)``.
* ``MSCFModel_IDM.cpp:80-89``  ``minNextSpeed`` (Euler): floor at
  ``max(0, v - max(myDecel, min(myEmergencyDecel, 1.5))*TS)``.
* ``MSCFModel.cpp:341-348``   ``minNextSpeedEmergency``: ``max(0, v - emergencyDecel*TS)``.
* ``MSCFModel.cpp:199-232``   ``finalizeSpeed`` composite with no stops/dawdling/
  startup delay: ``vMin = min(minNextSpeed, max(vPos, vMinEmergency))``,
  ``vMax = min(max(laneMaxSpeed, vPos), maxNextSpeed, vStop)`` -> the effective
  next speed is ``clamp(vSafe, vMin, min(vSafe, v + a*TS, vTypeMax))`` (with the
  default ``actionStepLength == TS`` the aMax term degenerates to
  ``max(laneMaxSpeed, vSafe)``).
* ``MSCFModel.cpp:324-326``   ``maxNextSpeed = min(v + a*TS, vTypeMaxSpeed)``.
* ``MSCFModel_IDM.cpp:179-200`` ``stopSpeed``: ``gap < 0.01 -> 0``; standing
  vehicles (``speed < NUMERICAL_EPS`` and result ~0) fall back to
  ``maximumSafeStopSpeed``; the result never exceeds ``gap/TS``.
* ``MSCFModel.cpp:860-884``   ``maximumSafeStopSpeedEuler`` (exact translation).
* ``MSCFModel_IDM.cpp:138-162`` ``insertionFollowSpeed`` (equilibrium insertion).
* ``MSCFModel_IDM.cpp:105-127`` ``freeSpeed`` (speed-limit step ahead).

Conventions/limits (documented gaps, no silent deviation):

* Semi-implicit Euler update (``--step-method.ballistic`` off, the SUMO default).
  The WM integrator (``WmSModel.step``: ``v' = max(0, v + a dt)``,
  ``s' = s + (v + v')/2 dt``) reproduces the SUMO ballistic position rule and
  the Euler speed rule from the acceleration returned here.
* ``actionStepLength == stepLength`` (the SUMO default): the model is
  re-evaluated every 0.5 s step.  A larger action step would hold the last
  acceleration between action steps (``MSVehicle.cpp:4654-4663``); the
  ``action_step_*`` helpers implement that hold, but the WM rollout does not
  carry their state yet (S-RB-INT wiring; not needed for parity with default
  SUMO runs).
* Driver perception errors (``applyHeadwayAndSpeedDifferencePerceptionErrors``),
  IDMM adaptation, ``patchSpeedBeforeLC`` dawdling, startup delay, stops and
  junction link speeds are not part of car-following parity (IDM defaults are
  no-ops for them; junctions/insertion policy belong to S-RB-MERGE).
* ``emergencyDecel`` defaults to **9 m/s^2** (the SUMO passenger-vClass
  default, ``MSCFModel.cpp:59-60`` + docs; NOT decel — that is apparentDecel's
  default), so IDM hard braking is floored at 9 m/s^2 per step exactly like a
  default SUMO passenger vType.

Pure tensor functions; differentiable in the inputs where natural; no RNG.
Reconciled with the rb-shared ``idmspec`` reference (spec.md, ref_idm.py,
20 test vectors; NUMERICAL_EPS=0.001 from ``src/config.h.cmake:236`` and
emergencyDecel=9 confirmed independently by S-RB-FIT).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional

import torch
from torch import Tensor

#: SUMO ``NUMERICAL_EPS`` (src/config.h.cmake:236 of tag v1_27_1).
NUMERICAL_EPS = 0.001
#: SUMO vType ``carFollowModel`` parameter ``cf.idmStepping`` default (s).
DEFAULT_CF_STEPPING_S = 0.25
#: IDM ``delta`` exponent default (``SUMO_ATTR_CF_IDM_DELTA``).
DEFAULT_IDM_DELTA = 4.0
#: SUMO passenger-vClass default ``emergencyDecel`` (m/s^2).
DEFAULT_EMERGENCY_DECEL = 9.0
#: ``EMERGENCY_DECEL_AMPLIFIER`` (MSCFModel.h:35).
EMERGENCY_DECEL_AMPLIFIER = 1.2
#: Cap of the comfortable-deceleration relaxations in ``minNextSpeed``
#: (``MIN2(myEmergencyDecel, 1.5)`` of MSCFModel_IDM.cpp:82).
MIN_NEXT_SPEED_EMERGENCY_SLACK = 1.5


def idm_iterations(dt_s: float, stepping_s: float = DEFAULT_CF_STEPPING_S) -> int:
    """``MAX2(1, int(TS / idmStepping + .5))`` (MSCFModel_IDM.cpp:44)."""
    dt_s, stepping_s = float(dt_s), float(stepping_s)
    if not (math.isfinite(dt_s) and dt_s > 0.0):
        raise ValueError("dt_s must be finite and > 0")
    if not (math.isfinite(stepping_s) and stepping_s > 0.0):
        raise ValueError("stepping_s must be finite and > 0")
    return max(1, int(dt_s / stepping_s + 0.5))


@dataclass(frozen=True)
class SumoIDMParams:
    """Per-driver SUMO IDM parameters (the vType car-following set).

    ``des_speed`` is ``min(lane speed limit * speedFactor, vType maxSpeed)``;
    the WM's driver ``v0`` (encoder-inferred or the population default) is that
    value.  ``vtype_max_speed`` caps acceleration like SUMO's ``maxNextSpeed``
    (``None`` = unbounded, i.e. the desired speed dominates as on this road).
    ``emergency_decel`` ``None`` means the SUMO passenger-vClass default 9
    m/s^2 (``DEFAULT_EMERGENCY_DECEL``).
    """

    accel: float | Tensor
    decel: float | Tensor
    headway_time: float | Tensor
    min_gap: float | Tensor
    des_speed: float | Tensor
    delta: float | Tensor = DEFAULT_IDM_DELTA
    emergency_decel: Optional[float | Tensor] = None
    vtype_max_speed: Optional[float | Tensor] = None

    @classmethod
    def from_driver_params(cls, params: Mapping[str, Tensor],
                           *, des_speed: Optional[float | Tensor] = None) -> "SumoIDMParams":
        """SUMO parameters from the encoder's per-driver settings.

        ``params`` is the driver-encoder mapping (``v0``, ``time_headway``,
        ``min_gap``, ``accel``, ``decel``); ``v0`` IS the desired speed
        (speedFactor semantics), so ``des_speed`` defaults to it.
        """
        return cls(accel=params["accel"].clamp_min(1.0e-3),
                   decel=params["decel"].clamp_min(1.0e-3),
                   headway_time=params["time_headway"].clamp_min(0.0),
                   min_gap=params["min_gap"].clamp_min(0.0),
                   des_speed=params["v0"].clamp_min(1.0e-3) if des_speed is None
                   else des_speed)

    def effective_emergency_decel(self):
        """``emergency_decel`` resolved (None -> the SUMO passenger default 9)."""
        return DEFAULT_EMERGENCY_DECEL if self.emergency_decel is None \
            else self.emergency_decel


def _pair(x: float | Tensor, like: Tensor) -> Tensor:
    return x if isinstance(x, Tensor) else torch_full(float(x), like)


def torch_full(value: float, like: Tensor) -> Tensor:
    return torch.full_like(like, value)


def sumo_idm_v(v: Tensor, gap_bumper: Tensor, leader_v: Tensor, des_speed: Tensor,
               p: SumoIDMParams, *, dt_s: float, iterations: int,
               respect_min_gap: bool = True) -> Tensor:
    """``MSCFModel_IDM::_v`` (MSCFModel_IDM.cpp:225-269): the iterated safe speed.

    ``gap_bumper`` is the front-to-rear gap of the configuration the caller
    judged (SUMO passes net gap + minGap back together = bumper gap when
    ``respect_min_gap``); with ``respect_min_gap=False`` (SUMO ``stopSpeed``)
    it is the raw obstacle distance and ``s0`` is never added.
    Returns the next speed (m/s), ``>= 0``.
    """
    like = v
    accel = _pair(p.accel, like)
    decel = _pair(p.decel, like)
    two_sqrt_ad = (2.0 * (accel * decel).clamp_min(1.0e-12).sqrt()).clamp_min(1.0e-12)
    delta = _pair(p.delta, like)
    s0 = _pair(p.min_gap, like)
    headway = _pair(p.headway_time, like)
    new_speed = v.clone()
    gap = gap_bumper.clone()
    for _ in range(int(iterations)):
        delta_v = new_speed - leader_v
        s = (new_speed * headway
             + new_speed * delta_v / two_sqrt_ad).clamp_min(0.0)
        if respect_min_gap:
            s = s + s0
        gap = gap.clamp_min(NUMERICAL_EPS)
        free = (new_speed / des_speed.clamp_min(NUMERICAL_EPS)).clamp_min(0.0)
        acc = accel * (1.0 - free.pow(delta)
                       - (s * s) / (gap * gap))
        new_speed = (new_speed + acc * float(dt_s) / float(iterations)).clamp_min(0.0)
        gap = gap - ((new_speed - leader_v) * float(dt_s) / float(iterations)).clamp_min(0.0)
    return new_speed.clamp_min(0.0)


def follow_speed(v: Tensor, gap_bumper: Tensor, leader_v: Tensor, des_speed: Tensor,
                 p: SumoIDMParams, *, dt_s: float, iterations: int) -> Tensor:
    """``MSCFModel_IDM::followSpeed`` == ``_v(respectMinGap=true)`` (no perception error)."""
    return sumo_idm_v(v, gap_bumper, leader_v, des_speed, p, dt_s=dt_s,
                      iterations=iterations, respect_min_gap=True)


def min_next_speed(v: Tensor, p: SumoIDMParams, *, dt_s: float) -> Tensor:
    """IDM ``minNextSpeed`` (Euler): ``max(0, v - max(decel, min(emerg, 1.5))*dt)``."""
    like = v
    decel = _pair(p.decel, like)
    emerg = _pair(p.effective_emergency_decel(), like)
    decel_eff = torch_maximum(decel, torch_minimum(
        emerg, torch_full(MIN_NEXT_SPEED_EMERGENCY_SLACK, like)))
    return (v - decel_eff * float(dt_s)).clamp_min(0.0)


def min_next_speed_emergency(v: Tensor, p: SumoIDMParams, *, dt_s: float) -> Tensor:
    """Base ``minNextSpeedEmergency`` (Euler): ``max(0, v - emergencyDecel*dt)``."""
    return (v - _pair(p.effective_emergency_decel(), v) * float(dt_s)).clamp_min(0.0)


def max_next_speed(v: Tensor, p: SumoIDMParams, *, dt_s: float) -> Tensor:
    """``MSCFModel::maxNextSpeed``: ``min(v + accel*dt, vTypeMaxSpeed)``."""
    cap = (v + _pair(p.accel, v) * float(dt_s))
    if p.vtype_max_speed is not None:
        cap = torch_minimum(cap, _pair(p.vtype_max_speed, v))
    return cap


def torch_maximum(a: Tensor, b: Tensor) -> Tensor:
    return a.maximum(b) if a.dtype == b.dtype else a.maximum(b.to(a.dtype))


def torch_minimum(a: Tensor, b: Tensor) -> Tensor:
    return a.minimum(b) if a.dtype == b.dtype else a.minimum(b.to(a.dtype))


def sumo_idm_step(v: Tensor, gap_bumper: Tensor, leader_v: Tensor, lane_max: Tensor,
                  p: SumoIDMParams, *, dt_s: float, iterations: int) -> tuple[Tensor, Tensor]:
    """One full SUMO Euler step of the IDM composition.

    ``lane_max`` is ``veh->getLane()->getVehicleMaxSpeed(veh)`` (the desired
    speed ``des_speed`` of ``_v`` is exactly this).  The returned
    ``(v_next, accel)`` follows ``planMoveInternal`` -> ``adaptToLeader`` ->
    ``finalizeSpeed`` with default ``actionStepLength == TS`` and no stops.
    Exact chain (idmspec spec.md section 4):

    * ``laneCap = max(lane_max, minNextSpeed(v))``   (speed limits are not
      emergencies, MSVehicle.cpp:2289-2290);
    * ``vPos = min(vSafe, laneCap, maxNextSpeed(v))`` (planMove initialises
      v = maximumLaneSpeedCF(maxNextSpeed, laneCap), adaptToLeader min-es the
      leader safe speed in);
    * ``vMin = min(minNextSpeed(v), max(vPos, minNextSpeedEmergency(v)))``;
    * ``vNext = max(vMin, min(vPos, maxNextSpeed(v)))`` (finalizeSpeed:
      ``vMax = MAX2(vMin, vMax)``, ``patchSpeedBeforeLC`` base returns vMax).

    ``accel`` is ``(v_next - v) / dt``; the WM integrator reproduces the SUMO
    speed/position update from it.
    """
    v_safe = follow_speed(v, gap_bumper, leader_v, lane_max, p, dt_s=dt_s,
                          iterations=iterations)
    floor_c = min_next_speed(v, p, dt_s=dt_s)
    lane_cap = torch_maximum(lane_max, floor_c)
    v_pos = torch_minimum(torch_minimum(v_safe, lane_cap), max_next_speed(v, p, dt_s=dt_s))
    v_cap = torch_minimum(v_pos, max_next_speed(v, p, dt_s=dt_s))
    v_min = torch_minimum(floor_c,
                          torch_maximum(v_pos, min_next_speed_emergency(v, p, dt_s=dt_s)))
    v_next = torch_maximum(v_min, v_cap).clamp_min(0.0)
    return v_next, (v_next - v) / float(dt_s)


def maximum_safe_stop_speed_euler(gap: Tensor, decel: Tensor, headway_time: Tensor,
                                  *, dt_s: float) -> Tensor:
    """``MSCFModel::maximumSafeStopSpeedEuler`` (MSCFModel.cpp:860-884), exact."""
    g = gap - NUMERICAL_EPS
    b = decel * float(dt_s)
    t = headway_time
    s = float(dt_s)
    g = g.clamp_min(0.0)
    b = b.clamp_min(1.0e-12)
    root = (s * s + 4.0 * (s * (2.0 * g / b - t) + t * t)).clamp_min(0.0).sqrt()
    n = (0.5 - ((t + root * -0.5) / s)).floor().clamp_min(0.0)
    h = 0.5 * n * (n - 1.0) * b * s + n * b * t
    r = (g - h) / (n * s + t).clamp_min(1.0e-12)
    x = n * b + r
    return torch.where((gap - NUMERICAL_EPS) < 0.0,
                       torch.zeros_like(x), x.clamp_min(0.0))


def calculate_emergency_deceleration(gap: Tensor, ego_speed: Tensor, pred_speed: Tensor,
                                     pred_max_decel: float,
                                     p: SumoIDMParams) -> Tensor:
    """``MSCFModel::calculateEmergencyDeceleration`` (MSCFModel.cpp:1037-1086)."""
    emerg = _pair(p.effective_emergency_decel(), ego_speed)
    pred_brake = 0.5 * pred_speed * pred_speed / float(pred_max_decel)
    b1 = 0.5 * ego_speed * ego_speed / (gap + pred_brake).clamp_min(1.0e-12)
    b2 = 0.5 * (ego_speed * ego_speed - pred_speed * pred_speed) / gap.clamp_min(1.0e-12)
    out = torch.where(b1 <= float(pred_max_decel), b1, b2)
    return torch.where(gap <= 0.0, emerg, out)


def maximum_safe_stop_speed(gap: Tensor, decel: Tensor, current_speed: Tensor,
                            p: SumoIDMParams, headway: Tensor, *,
                            dt_s: float) -> Tensor:
    """``MSCFModel::maximumSafeStopSpeed`` (MSCFModel.cpp:806-856, Euler branch)
    including the default ``relaxEmergency=true`` relaxation of :815-844."""
    vsafe = maximum_safe_stop_speed_euler(gap, decel, headway, dt_s=dt_s)
    emerg = _pair(p.effective_emergency_decel(), current_speed)
    relax = torch.not_equal(_pair(p.decel, current_speed), emerg)
    orig_safe_decel = (current_speed - vsafe) / float(dt_s)
    need = orig_safe_decel > _pair(p.decel, current_speed) + NUMERICAL_EPS
    safe_decel = EMERGENCY_DECEL_AMPLIFIER * calculate_emergency_deceleration(
        gap, current_speed, torch.zeros_like(current_speed), 1.0, p)
    safe_decel = torch_maximum(safe_decel, _pair(p.decel, current_speed))
    safe_decel = torch_minimum(safe_decel, orig_safe_decel)
    relaxed = (current_speed - safe_decel * float(dt_s)).clamp_min(0.0)
    return torch.where(relax & need, relaxed, vsafe)


def stop_speed(v: Tensor, gap: Tensor, p: SumoIDMParams, *, dt_s: float,
               iterations: int, action_step_s: Optional[float] = None) -> Tensor:
    """``MSCFModel_IDM::stopSpeed`` (MSCFModel_IDM.cpp:179-200).

    ``gap`` is the raw distance to the standing obstacle (no ``minGap``).
    ``gap < 0.01`` stops immediately; standing vehicles that the IDM cannot
    resolve fall back to ``maximumSafeStopSpeed`` (Euler + emergency
    relaxation) with the vehicle's ACTION-STEP length as the headway argument
    (the SUMO call passes ``actionStepLengthSecs``, i.e. ``dt_s`` at the
    default), and the result never exceeds ``gap/TS``.
    """
    like = v
    des = _p_lane_max(p, like)
    result = torch.where(gap < 0.01, torch.zeros_like(v),
                         sumo_idm_v(v, gap, torch.zeros_like(v), des, p, dt_s=dt_s,
                                    iterations=iterations, respect_min_gap=False))
    standing = (v < NUMERICAL_EPS) & (result < NUMERICAL_EPS) & (gap >= 0.01)
    if bool(standing.any()):
        decel = _pair(p.decel, like)
        headway = like.new_full((), float(dt_s if action_step_s is None
                                          else float(action_step_s))).expand_as(like)
        fallback = maximum_safe_stop_speed(gap, decel, v, p, headway, dt_s=dt_s)
        result = torch.where(standing, fallback, result)
    return torch_minimum(result, gap / float(dt_s)).clamp_min(0.0)


def _p_lane_max(p: SumoIDMParams, like: Tensor) -> Tensor:
    """Desired speed of an IDM evaluation: the model's own ``des_speed``."""
    return _pair(p.des_speed, like)


def free_speed(v: Tensor, seen: Tensor, max_speed: Tensor, p: SumoIDMParams, *,
               dt_s: float, iterations: int) -> Tensor:
    """``MSCFModel_IDM::freeSpeed`` (MSCFModel_IDM.cpp:105-127).

    ``seen`` is the distance to the speed-limit change, ``max_speed`` the new
    limit.  Accelerating uses the free ``_v`` (gap 1e6); braking for a lower
    limit relaxes the gap to ``max(seen, secureGap)`` against a standing
    transition point; close to the change the result is capped at the new limit.
    """
    like = v
    lane_max = _p_lane_max(p, like)
    decel = _pair(p.decel, like)
    headway = _pair(p.headway_time, like)
    two_sqrt = (2.0 * (_pair(p.accel, like) * decel).clamp_min(1.0e-12).sqrt())
    two_sqrt = two_sqrt.clamp_min(1.0e-12)
    # getSecureGap(veh, nullptr, maxSpeed, 0, myDecel): the secure gap of driving
    # at the NEW limit against the standing transition point.
    sec_gap = (max_speed * headway + max_speed * max_speed / two_sqrt).clamp_min(0.0)
    huge = like.new_full((), 1.0e6).expand_as(v)
    v_acc = sumo_idm_v(v, huge, max_speed, lane_max, p, dt_s=dt_s,
                       iterations=iterations, respect_min_gap=False)
    v_dec = sumo_idm_v(v, torch_maximum(seen, sec_gap), torch.zeros_like(v), lane_max,
                       p, dt_s=dt_s, iterations=iterations, respect_min_gap=False)
    v_safe = torch.where(v <= max_speed, v_acc, v_dec)
    # if (seen < secGap) vSafe = MIN2(vSafe, maxSpeed)  (overshoot guard)
    return torch.where(seen < sec_gap, torch_minimum(v_safe, max_speed), v_safe)


def insertion_follow_speed(v: float, gap_bumper: float, leader_v: float,
                           p: SumoIDMParams, *, dt_s: float, iterations: int,
                           max_rounds: int = 10) -> float:
    """``MSCFModel_IDM::insertionFollowSpeed`` (scalar, recursive in SUMO).

    Equilibrium insertion speed: when the gap covers the moving equilibrium
    ``s = max(0, v*T + v*(v-vL)/(2*sqrt(a*b)))`` the desired speed is kept;
    otherwise ``followSpeed`` is applied to the CANDIDATE speed until the
    iteration settles (``speed2 - speed3 < accel*dt``), at most ``max_rounds``
    recursions (SUMO recurses unboundedly; 10 is its own style of bound
    elsewhere).  Reference implementation for S-RB-MERGE.
    """
    like = scalar_tensor(v)
    two_sqrt = math.sqrt(float(_pair(p.accel, like)) * float(_pair(p.decel, like))) * 2.0
    s = max(0.0, v * float(_pair(p.headway_time, like))
            + v * (v - leader_v) / max(two_sqrt, 1.0e-12))
    if gap_bumper >= s:
        return v
    speed = v
    for _ in range(int(max_rounds)):
        speed2 = float(follow_speed(scalar_tensor(speed), scalar_tensor(gap_bumper),
                                    scalar_tensor(leader_v), _pair(p.des_speed, like),
                                    p, dt_s=dt_s, iterations=iterations))
        speed3 = float(follow_speed(scalar_tensor(speed2), scalar_tensor(gap_bumper),
                                    scalar_tensor(leader_v), _pair(p.des_speed, like),
                                    p, dt_s=dt_s, iterations=iterations))
        if speed2 - speed3 < 1.0 * float(dt_s):
            return speed2
        speed = speed2
    return speed


def scalar_tensor(value: float) -> Tensor:
    return torch.tensor(float(value), dtype=torch.float64)


def action_step_enabled(action_step_s: Optional[float], dt_s: float) -> bool:
    """Whether ``actionStepLength > TS`` holds accelerations between action steps."""
    if action_step_s is None:
        return False
    if not (math.isfinite(float(action_step_s)) and float(action_step_s) >= float(dt_s)):
        raise ValueError("action_step_s must be >= dt_s (SUMO multiples of the step)")
    return float(action_step_s) > float(dt_s) + 1.0e-12


def action_step_update(v: Tensor, held_accel: Optional[Tensor], p: SumoIDMParams, *,
                       dt_s: float) -> Tensor:
    """Non-action-step continuation (``MSVehicle.cpp:4654-4657``):
    ``vSafe = v + accel_last * TS`` capped by the speed bounds, no re-planning.

    The caller carries ``held_accel`` (the last action-step acceleration) per
    vehicle; the WM rollout does not persist it yet (documented gap).
    """
    accel = torch.zeros_like(v) if held_accel is None else held_accel
    v_next = (v + accel * float(dt_s)).clamp_min(0.0)
    v_next = torch_minimum(v_next, max_next_speed(v, p, dt_s=dt_s))
    v_min = min_next_speed(v, p, dt_s=dt_s)
    return torch_maximum(v_next, torch_minimum(v_min, v_next))


#: S-RB-CF pure-physics preset (physics-side overrides for ``load_backend`` /
#: :class:`~wms.physics.config.PhysicsConfig`): the SUMO IDM core with every
#: learned/hand ACCELERATION correction off.  The model-side learned families
#: (residual, temporal/slow, courtesy, negotiation, ramp adapt, mainline
#: residual, merge response, learned yield) are neutralised by the runtime
#: switch ``WmSModel.learned_accel_off`` (``RolloutFlags.pure_physics``), not by
#: these physics keys.  Lane/merge model switches are NOT touched (S-RB-LANE /
#: S-RB-MERGE own them); ``entry_learned`` (a learned insertion rule) is off.
PURE_SUMO_PHYSICS = {
    "cf_model": "sumo_idm",
    "emergency_braking": False,
    "emergency_decel_mps2": None,
    "merge_cal": False,
    "mainline_merge_yield": False,
    "merge_end_wait": False,
    "merge_impatience": False,
    "entry_learned": False,
}


# ------------------------------------------------------- WM baseline dispatch
def sumo_idm_baseline_accel(kin: Tensor, present: Tensor, physics,
                            driver_params: Optional[Mapping[str, Tensor]] = None
                            ) -> Tensor:
    """The SUMO IDM core as the WM physics baseline ``[B, K]`` acceleration.

    Mirrors :func:`wms.physics.accel.baseline_accel`'s leader rule (same
    :func:`~wms.physics.neighbours.lead_neighbour`, including the accel-lane-end
    standing obstacle and the ``C4_SUMO_LEADER_RULE`` chain coordinates) but
    evaluates the SUMO composition: a real same-lane leader goes through
    :func:`sumo_idm_step` (followSpeed + finalizeSpeed bounds); the accel-lane
    end obstacle goes through SUMO ``stopSpeed`` semantics.  The legacy
    ``emergency_decel``/``emergency_braking`` envelope and the hand merge-yield
    rules are NOT applied (SUMO parity is the point of ``cf_model='sumo_idm'``;
    lane/merge interaction physics composes outside the CF core).

    ``driver_params`` (encoder-inferred) map 1:1; without them the population
    defaults of the :class:`~wms.physics.config.PhysicsConfig` apply.
    """
    from .idm import IDMParams
    from .neighbours import (MIN_GAP_M, lane_neighbour, lane_of,
                             lead_neighbour)

    gap, leader_v = lead_neighbour(kin, present, physics)
    v = kin[..., 4]
    if driver_params is None:
        defaults: IDMParams = physics.idm_params()
        like = v
        p = SumoIDMParams(
            accel=torch_full(float(defaults.a_max), like),
            decel=torch_full(float(defaults.b_comf), like),
            headway_time=torch_full(float(defaults.time_headway), like),
            min_gap=torch_full(float(defaults.s0), like),
            des_speed=torch_full(float(defaults.v0), like),
            delta=torch_full(float(defaults.delta), like),
            emergency_decel=(None if physics.sumo_cf_emergency_decel_mps2 is None
                             else torch_full(float(physics.sumo_cf_emergency_decel_mps2), like)),
            vtype_max_speed=(None if physics.sumo_cf_vtype_max_mps is None
                             else torch_full(float(physics.sumo_cf_vtype_max_mps), like)))
        lane_max = p.des_speed
    else:
        p = SumoIDMParams.from_driver_params(driver_params)
        if physics.sumo_cf_emergency_decel_mps2 is not None:
            p = SumoIDMParams(
                **{**p.__dict__, "emergency_decel": torch_full(
                    float(physics.sumo_cf_emergency_decel_mps2), v)})
        if physics.sumo_cf_vtype_max_mps is not None:
            p = SumoIDMParams(
                **{**p.__dict__, "vtype_max_speed": torch_full(
                    float(physics.sumo_cf_vtype_max_mps), v)})
        lane_max = p.des_speed
    dt_s = float(physics.dt_s)
    iterations = idm_iterations(dt_s, float(physics.sumo_cf_stepping_s))
    _, accel = sumo_idm_step(v, gap, leader_v, lane_max, p, dt_s=dt_s,
                             iterations=iterations)
    # The accel-lane end is SUMO's stopSpeed case (standing obstacle at the dead
    # end).  lead_neighbour returns the end gap already merged into `gap`, so the
    # obstacle is re-derived from the RAW same-lane leader gap exactly like
    # lead_neighbour selects it.
    lane = lane_of(kin)
    from .lane_chain import leader_rule_enabled
    if leader_rule_enabled():
        # Same chain-coordinate end obstacle as lead_neighbour (C4_SUMO_LEADER_RULE).
        from .lane_chain import MERGE_CONNECTOR_RAMP_LEN_M, chain_s
        chain = chain_s(lane, kin[..., 0])
        lane_gap, _ = lane_neighbour(kin, present, lane, physics, ahead=True)
        end_gap = (float(physics.merge_zone_s1_m) + float(MERGE_CONNECTOR_RAMP_LEN_M)
                   - chain).clamp_min(MIN_GAP_M)
    else:
        lane_gap, _ = lane_neighbour(kin, present, lane, physics, ahead=True)
        end_gap = (float(physics.merge_zone_s1_m) - kin[..., 0]).clamp_min(MIN_GAP_M)
    obstacle = (lane == int(physics.merge_lane)) & (end_gap < lane_gap)
    if bool(getattr(physics, "merge_cal", False)):
        obstacle = obstacle & (end_gap <= float(physics.end_obstacle_range_m))
    if bool(obstacle.any()):
        stop_next = stop_speed(v, end_gap, p, dt_s=dt_s, iterations=iterations,
                               action_step_s=getattr(physics, "sumo_cf_action_step_s", None))
        accel = torch.where(obstacle, (stop_next - v) / dt_s, accel)
    return torch.where(present, accel, torch.zeros_like(accel))
