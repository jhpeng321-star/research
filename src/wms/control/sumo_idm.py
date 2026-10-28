"""S-EXECPARITY2 shared exact executor law: SUMO 1.27.1 IDM on our ``cav_typical`` vType.

The deployment executor's base law for a commanded CAV is
:func:`wms.control.nativeinit.sumo_native_accel`: a TraCI ``getFollowSpeed``
query that SUMO evaluates with the vehicle's own car-following model. That
model and its parameters are OUR controller design (``traffic_shepherd.c4.
population`` vType ``cav_typical``: ``carFollowModel="IDM"`` with the
``TEXTBOOK_CAV_PARAMS`` population medians), so the exact function is
``MSCFModel_IDM::_v`` of eclipse-sumo v1_27_1 with those parameters:

* ``myIterations = max(1, int(TS / 0.25 + .5))`` = 2 at the frozen step length
  ``TS = 0.5 s`` (``SUMO_ATTR_CF_IDM_STEPPING`` default 0.25);
* ``myTwoSqrtAccelDecel = 2*sqrt(accel*decel)``; ``myDelta`` = 4;
* ``NUMERICAL_EPS`` = 1e-10 (SUMO ``StdDefs.h``);
* ``respectMinGap=True``: ``gap2pred`` arrives with minGap subtracted (the
  TraCI ``getLeader`` convention) and ``_v`` adds it back once.

The executor additionally applies the frozen legacy composer conventions:
``getFollowSpeed`` is fed the leader gap PLUS minGap again (the double-add
retained by S-CTRLARM CTRL0 bit-identity), the target is clipped to
``getAllowedSpeed`` (lane speed limit x speedFactor), and the base
acceleration is ``(max(0, target) - speed) / dt``.

This module is the single shared implementation of that exact law (S-EXECPARITY2,
REGISTRATION.md 02:40): the imagination path
(:func:`wms.control.alwayscmd.inferred_base` / :func:`wms.control.alwayscmd.executed`)
calls it with the WM state of the CAV and its leader, the offline replay and
the independent scalar fixture call it, and identity with the deployment
executor is verified offline against the recorded per-tick ledgers of the 00:20
always-commanded SUMO runs (``tools/execparity/execparity2_base_validation.json``).

Geometry conventions (bumper gap ``g`` as supplied by the WM frame /
``native_cav.leader_gap_dv``; SUMO's getLeader gap is ``g - minGap``):

* base law (executor, legacy double-add): ``_v(gap2pred = g, respectMinGap=True)``
  -> interaction gap ``g + minGap``;
* safe-speed cap (SUMO-internal single add): ``_v(gap2pred = g - minGap,
  respectMinGap=True)`` -> interaction gap ``g``.

Lane speed limits (registered merge network
``assets/research/highway_merge_shepherd_long/osm.net.xml.gz``): mainline
edges and the merge-edge lane 2 (``s >= 0``) 31.29 m/s = ``SPEED_LIMIT_MPS``;
the ramp edge ``11553048#1`` (lane 2, ``s < 0``) 22.22 m/s; the ramp->mainline
internal connector ``:1780739017_0_0`` 26.75 m/s (observable only through the
pre-junction lane, so the tensor imagination path uses the mainline value and
the residual gap is reported by the replay). No hand-set behavioural
thresholds: every constant is a designed vType parameter, a network lane
attribute, or a SUMO source literal.
"""
from __future__ import annotations

import math

BACKEND_CONTRACT = "always_commanded_ctrl0/1"

#: SUMO step length of the frozen runs (``TS``).
IDM_DT_S = 0.5
#: ``SUMO_ATTR_CF_IDM_STEPPING`` default of MSCFModel_IDM.
IDM_STEPPING_S = 0.25
#: ``NUMERICAL_EPS`` (SUMO ``StdDefs.h``).
NUMERICAL_EPS = 1e-10
#: Free-road stand-in gap of ``MSCFModel_IDM::freeSpeed`` (1e6 m).
FREE_GAP_M = 1e6

#: The designed CAV vType parameters (``cav_typical``; copy of
#: ``wms.model.cav.TEXTBOOK_CAV_SETTINGS`` == ``traffic_shepherd.c4.population.
#: TEXTBOOK_CAV_PARAMS`` medians; asserted by unit tests). Not learned, not
#: hand-tuned here: they ARE the deployment executor's car-following design.
CAV_PARAMS = {"accel": 1.2, "decel": 2.0, "time_headway": 1.2, "min_gap": 2.0,
              "delta": 4.0, "speed_factor": 1.0}

#: Ramp edge ``11553048#1_0`` lane speed limit (m/s).
RAMP_SPEED_LIMIT_MPS = 22.22
#: Ramp->mainline internal connector ``:1780739017_0_0`` speed limit (m/s).
MERGE_CONNECTOR_SPEED_MPS = 26.75


def _iterations(dt):
    """``MAX2(1, int(TS / stepping + .5))`` of MSCFModel_IDM."""
    return max(1, int(float(dt) / IDM_STEPPING_S + .5))


def follow_speed(gap2pred, speed, leader_speed, allowed_speed, *,
                 params=None, dt=IDM_DT_S, respect_min_gap=True):
    """Exact ``MSCFModel_IDM::_v`` (torch, differentiable).

    ``gap2pred`` follows the SUMO internal convention (minGap subtracted when
    ``respect_min_gap``); ``allowed_speed`` is ``desSpeed`` (the lane's
    ``getVehicleMaxSpeed``). Bitwise-identical to :func:`follow_speed_scalar`.
    """
    import torch
    p = dict(CAV_PARAMS if params is None else params)
    a, b, tau, gap0 = (float(p[k]) for k in ("accel", "decel", "time_headway", "min_gap"))
    delta = float(p.get("delta", 4.0))
    iterations = _iterations(dt)
    two_sqrt_ab = 2. * math.sqrt(a * b)
    step = float(dt) / iterations
    eps = float(NUMERICAL_EPS)
    v = speed
    gap = gap2pred + (gap0 if respect_min_gap else 0.)
    for _ in range(iterations):
        desired = (v * tau + v * (v - leader_speed) / two_sqrt_ab).clamp_min(0.)
        if respect_min_gap:
            desired = desired + gap0
        g = gap.clamp_min(eps)
        free = (v / allowed_speed.clamp_min(eps)).pow(delta)
        accel = a * (1. - free - (desired * desired) / (g * g))
        v = (v + (accel * float(dt)) / iterations).clamp_min(0.)
        gap = gap - (((v - leader_speed) * float(dt)) / iterations).clamp_min(0.)
    return v.clamp_min(0.)


def allowed_speed(lane, s_m):
    """Lane speed limit x speedFactor (``getAllowedSpeed``) from the WM frame."""
    import torch
    from .contract import SPEED_LIMIT_MPS
    lane = torch.as_tensor(lane).long()
    s_m = torch.as_tensor(s_m)
    return torch.where((lane == 2) & (torch.as_tensor(s_m) < 0),
                       torch.full_like(s_m, float(RAMP_SPEED_LIMIT_MPS)),
                       torch.full_like(s_m, float(SPEED_LIMIT_MPS)))


def base_accel(speed, leader_speed, bumper_gap, allowed_speed, *, has_leader,
               params=None, dt=IDM_DT_S):
    """The exact deployment-executor base law (``sumo_native_accel``).

    Legacy double-add convention: the executor passes the bumper gap to
    ``getFollowSpeed`` (getLeader gap + minGap) and SUMO's ``_v`` adds minGap
    again; no leader means the infinite-gap free branch with the ego speed as
    the leader speed. Target clipped to ``allowed_speed`` and to >= 0.
    """
    import torch
    inf = torch.full_like(speed, float("inf"))
    gap2pred = torch.where(has_leader, bumper_gap, inf)
    vl = torch.where(has_leader, leader_speed, speed)
    target = follow_speed(gap2pred, speed, vl, allowed_speed,
                          params=params, dt=dt, respect_min_gap=True)
    return target.minimum(allowed_speed).clamp_min(0.).sub(speed).div(float(dt))


def safe_speed_cap(speed, leader_speed, bumper_gap, allowed_speed, *,
                   has_leader, lane=None, s_m=None, ramp_gap=None,
                   params=None, dt=IDM_DT_S):
    """The exact speedMode-27 executed-speed ceiling of a commanded CAV.

    SUMO realises ``max(0, min(v + issued*dt, vSafe, maxNextSpeed))`` where
    ``vSafe`` is the vehicle's own IDM follow speed with the internal
    single-minGap add, ``maxNextSpeed = min(v + accel*dt, allowed)``; the
    registered ramp lane-end stop (``MSCFModel_IDM::stopSpeed`` toward the
    lane-2 dead end at ``MERGE_S1_M``, i.e. the merge/lane-end approach that
    SUMO applies inside ``processLinkApproaches``) is retained on lane 2, or
    wherever a finite ``ramp_gap`` is supplied explicitly. Returns the ceiling
    in the acceleration domain (>= -v/dt by construction).
    Junction-internal foe yielding is not observable from a WM frame; that
    residual state error is reported by the offline replay.
    """
    import torch
    p = dict(CAV_PARAMS if params is None else params)
    gap0 = float(p["min_gap"])
    free = torch.full_like(speed, float(FREE_GAP_M))
    gap2pred = torch.where(has_leader, bumper_gap - gap0, free)
    vl = torch.where(has_leader, leader_speed, speed)
    vsafe = follow_speed(gap2pred, speed, vl, allowed_speed,
                         params=p, dt=dt, respect_min_gap=True)
    if ramp_gap is not None:
        stop_gap = torch.as_tensor(ramp_gap, dtype=speed.dtype).clamp_min(0.)
        stop = follow_speed(stop_gap, speed, torch.zeros_like(speed), allowed_speed,
                            params=p, dt=dt, respect_min_gap=False)
        stop = torch.minimum(stop, stop_gap / float(dt))
        vsafe = torch.minimum(vsafe, stop)
    elif lane is not None:
        from .contract import MERGE_S1_M
        stop_gap = (MERGE_S1_M - torch.as_tensor(s_m)).clamp_min(0.)
        stop = follow_speed(stop_gap.to(speed), speed,
                            torch.zeros_like(speed), allowed_speed,
                            params=p, dt=dt, respect_min_gap=False)
        stop = torch.minimum(stop, stop_gap.to(speed) / float(dt))
        vsafe = torch.where(torch.as_tensor(lane).long() == 2,
                            torch.minimum(vsafe, stop), vsafe)
    vmax = torch.minimum(speed + float(p["accel"]) * float(dt), allowed_speed)
    return (torch.minimum(vsafe, vmax).clamp_min(0.) - speed) / float(dt)


# ----------------------------------------------------------------------
# scalar twins: the same torch ops on 0-dim float64 tensors, so the scalar
# fixture is bitwise identical to the tensor path by construction.
# ----------------------------------------------------------------------
def _scalar(fn, *args, **kwargs):
    import torch
    tensors = [torch.as_tensor(float(a), dtype=torch.float64) for a in args]
    out = fn(*tensors, **kwargs)
    return float(out)


def follow_speed_scalar(gap2pred, speed, leader_speed, allowed_speed, **kwargs):
    return _scalar(follow_speed, gap2pred, speed, leader_speed, allowed_speed, **kwargs)


def allowed_speed_scalar(lane, s_m, *, prev_lane=None):
    """Scalar ``allowed_speed``; junction internals use the pre-junction lane."""
    from .contract import SPEED_LIMIT_MPS
    if int(lane) == 2:
        return RAMP_SPEED_LIMIT_MPS if float(s_m) < 0. else float(SPEED_LIMIT_MPS)
    if int(lane) == -1 and prev_lane is not None and int(prev_lane) == 2:
        return MERGE_CONNECTOR_SPEED_MPS
    return float(SPEED_LIMIT_MPS)


def base_accel_scalar(speed, leader_speed, bumper_gap, allowed_speed, *,
                      has_leader, params=None, dt=IDM_DT_S):
    import torch
    return _scalar(base_accel, speed, leader_speed, bumper_gap, allowed_speed,
                   has_leader=torch.as_tensor(bool(has_leader)), params=params, dt=dt)


def safe_speed_cap_scalar(speed, leader_speed, bumper_gap, allowed_speed, *,
                          has_leader, lane=None, s_m=None, ramp_gap=None,
                          params=None, dt=IDM_DT_S):
    """Scalar cap; ``ramp_gap`` (if finite) replaces the lane-2 dead-end stop."""
    import math
    import torch
    if ramp_gap is not None and not math.isfinite(float(ramp_gap)):
        ramp_gap = None
    return _scalar(safe_speed_cap, speed, leader_speed, bumper_gap, allowed_speed,
                   has_leader=torch.as_tensor(bool(has_leader)),
                   lane=lane, s_m=(None if s_m is None else torch.as_tensor(float(s_m))),
                   ramp_gap=ramp_gap, params=params, dt=dt)


def executed_accel(speed, leader_speed, bumper_gap, allowed_speed, residual,
                   lo, hi, *, has_leader, lane=None, s_m=None, ramp_gap=None,
                   params=None, dt=IDM_DT_S):
    """The composed always-commanded executed acceleration (single shared law).

    ``issued = clip(base + residual, min(lo, base), max(min(hi, 1.2), base))``
    with the exact executor base law, then the speedMode-27 outcome
    ``max(-v/dt, min(issued, cap))`` with the exact safe-speed cap. This is the
    ONE implementation of the imagination execution: the tensor path and the
    scalar fixture below both evaluate exactly these ops.
    """
    import torch
    p = dict(CAV_PARAMS if params is None else params)
    has_leader = torch.as_tensor(has_leader)
    base = base_accel(speed, leader_speed, bumper_gap, allowed_speed,
                      has_leader=has_leader, params=p, dt=dt)
    cap = safe_speed_cap(speed, leader_speed, bumper_gap, allowed_speed,
                         has_leader=has_leader, lane=lane, s_m=s_m,
                         ramp_gap=ramp_gap, params=p, dt=dt)
    upper = torch.maximum(base, torch.full_like(base, min(float(hi), 1.2)))
    lower = torch.minimum(base, torch.full_like(base, float(lo)))
    issued = torch.minimum(torch.maximum(base + residual, lower), upper)
    return torch.maximum(torch.minimum(issued, cap), -speed / float(dt))


def executed_accel_scalar(speed, leader_speed, bumper_gap, allowed_speed,
                          residual, lo, hi, *, has_leader, lane=None, s_m=None,
                          ramp_gap=None, params=None, dt=IDM_DT_S):
    import math
    import torch
    values = [speed, leader_speed, residual, lo, hi, dt]
    if not all(math.isfinite(float(x)) for x in values) or dt <= 0:
        raise ValueError("finite always-commanded execution inputs required")
    if ramp_gap is not None and not math.isfinite(float(ramp_gap)):
        ramp_gap = None
    return _scalar(executed_accel, speed, leader_speed, bumper_gap, allowed_speed,
                   residual, lo, hi, has_leader=torch.as_tensor(bool(has_leader)),
                   lane=lane, s_m=s_m, ramp_gap=ramp_gap, params=params, dt=dt)
