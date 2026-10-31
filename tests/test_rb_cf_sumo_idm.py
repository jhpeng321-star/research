"""S-RB-CF tests: SUMO 1.27.1 MSCFModel_IDM parity (wms.physics.idm_sumo).

Expected values are derived from the SUMO 1.27.1 source formulas themselves:
``_scalar_v``/``_scalar_step`` are an independent statement-by-statement plain
Python transcription of MSCFModel_IDM.cpp:225-269 (``_v``), MSCFModel_IDM.cpp:80-89
(``minNextSpeed``), MSCFModel.cpp:341-348 (``minNextSpeedEmergency``),
MSCFModel.cpp:324-326 (``maxNextSpeed``) and the finalizeSpeed composition
(MSCFModel.cpp:199-232, default actionStepLength == TS, no stops/dawdling).
The hard-coded literals were additionally verified by hand (see comments).
"""
from __future__ import annotations

import math

import pytest
import torch
from torch import tensor

from wms.physics.idm_sumo import (DEFAULT_CF_STEPPING_S, DEFAULT_IDM_DELTA,
                                  NUMERICAL_EPS, SumoIDMParams, action_step_enabled,
                                  follow_speed, free_speed, idm_iterations,
                                  insertion_follow_speed, max_next_speed,
                                  maximum_safe_stop_speed_euler, min_next_speed,
                                  min_next_speed_emergency, stop_speed,
                                  sumo_idm_step, sumo_idm_v)

A, B, TAU, S0, DELTA, DES = 2.6, 4.5, 1.1, 2.5, 4.0, 31.29
TS = 0.5


def params(**kw) -> SumoIDMParams:
    base = dict(accel=A, decel=B, headway_time=TAU, min_gap=S0, des_speed=DES,
                delta=DELTA)
    base.update(kw)
    return SumoIDMParams(**base)


# ----------------------------------------------------------------- scalar refs
def _scalar_v(v0, gap_bumper, lead, des, a, b, t, s0, delta, ts, iters,
              respect_min_gap=True):
    two_sqrt = 2.0 * math.sqrt(a * b)
    new = v0
    gap = gap_bumper                      # SUMO gap2pred + minGap == bumper gap
    for _ in range(iters):
        dv = new - lead
        s = max(0.0, new * t + new * dv / two_sqrt)
        if respect_min_gap:
            s = s + s0
        gap = max(NUMERICAL_EPS, gap)
        acc = a * (1.0 - (new / max(NUMERICAL_EPS, des)) ** delta
                   - (s * s) / (gap * gap))
        new = max(0.0, new + acc * ts / iters)
        gap -= max(0.0, (new - lead) * ts / iters)
    return max(0.0, new)


def _scalar_step(v, gap_bumper, lead, lane_max, p, ts, iters):
    emerg = float(p.decel) if p.emergency_decel is None else float(p.emergency_decel)
    v_safe = _scalar_v(v, gap_bumper, lead, lane_max, float(p.accel), float(p.decel),
                       float(p.headway_time), float(p.min_gap), float(p.delta),
                       ts, iters, True)
    cap = v + float(p.accel) * ts
    if p.vtype_max_speed is not None:
        cap = min(cap, float(p.vtype_max_speed))
    v_max = min(max(lane_max, v_safe), cap, v_safe)
    decel_eff = max(float(p.decel), min(emerg, 1.5))
    floor_c = max(0.0, v - decel_eff * ts)
    floor_e = max(0.0, v - emerg * ts)
    v_min = min(floor_c, max(v_safe, floor_e))
    v_next = max(0.0, max(v_min, v_max))
    return v_safe, v_next, (v_next - v) / ts


def _scalar_stop_euler(gap, decel, headway, ts):
    g = gap - NUMERICAL_EPS
    if g < 0.0:
        return 0.0
    bb = decel * ts
    n = math.floor(.5 - ((headway + math.sqrt(ts * ts + 4.0 * (ts * (2.0 * g / bb - headway)
                                                              + headway * headway)) * -0.5) / ts))
    h = 0.5 * n * (n - 1) * bb * ts + n * bb * headway
    r = (g - h) / (n * ts + headway)
    return n * bb + r


def _scalar_stop(v, gap, p, ts, iters, action_step=None):
    if gap < 0.01:
        return 0.0
    res = _scalar_v(v, gap, 0.0, float(p.des_speed), float(p.accel), float(p.decel),
                    float(p.headway_time), float(p.min_gap), float(p.delta),
                    ts, iters, False)
    if gap > 0 and v < NUMERICAL_EPS and res < NUMERICAL_EPS:
        res = _scalar_stop_euler(gap, float(p.decel),
                                 ts if action_step is None else action_step, ts)
    if gap >= 0:
        res = min(res, gap / ts)
    return res


# ------------------------------------------------------------------- iterators
def test_idm_iterations_match_sumo_rounding():
    # MAX2(1, int(TS / stepping + 0.5))
    assert idm_iterations(0.5) == 2
    assert idm_iterations(1.0) == 4
    assert idm_iterations(0.25) == 1
    assert idm_iterations(0.5, stepping_s=1.0) == 1
    assert idm_iterations(0.5, stepping_s=0.1) == 5
    with pytest.raises(ValueError):
        idm_iterations(0.0)
    with pytest.raises(ValueError):
        idm_iterations(0.5, stepping_s=0.0)


def test_default_stepping_is_sumo_default():
    assert DEFAULT_CF_STEPPING_S == 0.25
    assert DEFAULT_IDM_DELTA == 4.0


# ----------------------------------------------------------------- safe speed
def test_follow_speed_hand_derived_anchor():
    # v=20, leader 15 m/s, bumper gap 30 m (hand-derived: two _v iterations give
    # 18.953013...; accel (v_next - v)/dt = -2.093974).
    v, gap, lead = tensor([20.0], dtype=torch.float64), tensor([30.0], dtype=torch.float64), tensor([15.0], dtype=torch.float64)
    p = params()
    out = follow_speed(v, gap, lead, tensor([DES], dtype=torch.float64), p, dt_s=TS, iterations=2)
    assert out.item() == pytest.approx(18.9530131733, abs=5e-7)


def test_sumo_idm_step_composition_anchors():
    # case 1: same scene as above; the composed step returns the IDM speed (no
    # bound binds): v_next = 18.953013, accel = -2.093974.
    v, gap, lead = tensor([20.0], dtype=torch.float64), tensor([30.0], dtype=torch.float64), tensor([15.0], dtype=torch.float64)
    v_next, accel = sumo_idm_step(v, gap, lead, tensor([DES], dtype=torch.float64), params(),
                                  dt_s=TS, iterations=2)
    assert v_next.item() == pytest.approx(18.9530131733, abs=5e-7)
    assert accel.item() == pytest.approx(-2.0939736533, abs=5e-7)
    # case 2: closing at 20 m/s on a stopped leader 10 m ahead with the SUMO
    # default emergencyDecel == decel: the IDM demands 0.609375 but the
    # comfortable-decel floor binds (17.75 = 20 - 4.5*0.5): SUMO keeps 4.5 m/s^2
    # braking this step ("IDM may violate minGap on occasion").
    v_next, accel = sumo_idm_step(tensor([20.0], dtype=torch.float64), tensor([10.0], dtype=torch.float64), tensor([0.0], dtype=torch.float64),
                                  tensor([DES], dtype=torch.float64), params(), dt_s=TS, iterations=2)
    assert v_next.item() == pytest.approx(17.75, abs=1e-9)
    assert accel.item() == pytest.approx(-4.5, abs=1e-9)
    # case 3: emergencyDecel 9 allows braking down to v - 9*ts = 15.5 -> -9 m/s^2.
    v_next, accel = sumo_idm_step(tensor([20.0], dtype=torch.float64), tensor([10.0], dtype=torch.float64), tensor([0.0], dtype=torch.float64),
                                  tensor([DES], dtype=torch.float64), params(emergency_decel=9.0),
                                  dt_s=TS, iterations=2)
    assert v_next.item() == pytest.approx(15.5, abs=1e-9)
    assert accel.item() == pytest.approx(-9.0, abs=1e-9)


def test_v_iterates_internal_substeps_against_scalar():
    rng_vals = [(v, g, lv) for v in (0.0, 3.7, 12.5, 20.0, 28.9)
                for g in (0.05, 1.3, 8.0, 25.0, 120.0, 260.0)
                for lv in (0.0, 5.0, 14.9, 22.0, 31.0)]
    v = tensor([x[0] for x in rng_vals], dtype=torch_dtype())
    gap = tensor([x[1] for x in rng_vals], dtype=torch_dtype())
    lead = tensor([x[2] for x in rng_vals], dtype=torch_dtype())
    for iters, ts in ((2, 0.5), (4, 1.0)):
        out = sumo_idm_v(v, gap, lead, tensor([DES] * len(rng_vals), dtype=torch_dtype()), params(),
                         dt_s=ts, iterations=iters)
        ref = [_scalar_v(x[0], x[1], x[2], DES, A, B, TAU, S0, DELTA, ts, iters)
               for x in rng_vals]
        torch.testing.assert_close(out, tensor(ref, dtype=out.dtype),
                                   rtol=0.0, atol=5e-7)


def test_step_composition_grid_against_scalar():
    cases = [(v, g, lv, emerg, vmax) for v in (0.0, 8.0, 17.3, 24.0, 30.5)
             for g in (0.05, 2.0, 9.0, 33.0, 140.0, 240.0)
             for lv in (0.0, 7.7, 18.0, 26.0)
             for emerg in (None, 9.0)
             for vmax in (None, 33.34)]
    v = tensor([c[0] for c in cases], dtype=torch_dtype())
    gap = tensor([c[1] for c in cases], dtype=torch_dtype())
    lead = tensor([c[2] for c in cases], dtype=torch_dtype())
    des = tensor([DES] * len(cases), dtype=torch_dtype())
    for emerg in (None, 9.0):
        for vmax in (None, 33.34):
            sel = [i for i, c in enumerate(cases) if c[3] == emerg and c[4] == vmax]
            p = params(emergency_decel=emerg, vtype_max_speed=vmax)
            v_next, accel = sumo_idm_step(v[sel], gap[sel], lead[sel], des[sel], p,
                                          dt_s=TS, iterations=2)
            ref = [_scalar_step(cases[i][0], cases[i][1], cases[i][2], DES, p, TS, 2)
                   for i in sel]
            torch.testing.assert_close(
                v_next, tensor([r[1] for r in ref], dtype=v_next.dtype),
                rtol=0.0, atol=5e-7)
            torch.testing.assert_close(
                accel, tensor([r[2] for r in ref], dtype=accel.dtype),
                rtol=0.0, atol=5e-7)


def torch_dtype():
    import torch
    return torch.float64


# --------------------------------------------------------------------- bounds
def test_min_next_speed_emergency_slack_cap():
    # decel_eff = max(decel, min(emergency, 1.5)): emergencyDecel defaults to
    # decel, so a low decel only rises to 1.5 when emergencyDecel >= 1.5.
    v = tensor([20.0], dtype=torch.float64)
    assert min_next_speed(v, params(), dt_s=TS).item() == pytest.approx(17.75)
    assert min_next_speed(v, params(decel=1.0, emergency_decel=9.0),
                          dt_s=TS).item() == pytest.approx(20.0 - 1.5 * TS)
    assert min_next_speed(v, params(decel=1.0, emergency_decel=1.2),
                          dt_s=TS).item() == pytest.approx(20.0 - 1.2 * TS)
    assert min_next_speed(v, params(decel=1.0), dt_s=TS).item() \
        == pytest.approx(20.0 - 1.0 * TS)
    assert min_next_speed_emergency(v, params(), dt_s=TS).item() \
        == pytest.approx(17.75)
    assert min_next_speed_emergency(v, params(emergency_decel=9.0),
                                    dt_s=TS).item() == pytest.approx(15.5)


def test_max_next_speed_caps_accel_and_vtype_max():
    v = tensor([20.0], dtype=torch.float64)
    assert max_next_speed(v, params(), dt_s=TS).item() == pytest.approx(21.3)
    out = max_next_speed(v, params(vtype_max_speed=20.5), dt_s=TS)
    assert out.item() == pytest.approx(20.5)


# ------------------------------------------------------------------ stop speed
def test_maximum_safe_stop_speed_euler_hand_derived():
    # gap 2.0, decel 4.5, headway 0.5 (= actionStepLength), TS 0.5:
    # n = 1, h = 1.125, r = 0.865 -> x = 3.115 (exact stop at the obstacle).
    out = maximum_safe_stop_speed_euler(tensor([2.0], dtype=torch.float64), tensor([4.5], dtype=torch.float64), tensor([0.5], dtype=torch.float64),
                                        dt_s=TS)
    assert out.item() == pytest.approx(3.115, abs=1e-9)
    # gap below NUMERICAL_EPS is unreachable -> 0.
    out = maximum_safe_stop_speed_euler(tensor([0.005], dtype=torch.float64), tensor([4.5], dtype=torch.float64), tensor([0.5], dtype=torch.float64),
                                        dt_s=TS)
    assert out.item() == 0.0


def test_stop_speed_anchors():
    # standing vehicle (v < 0.01, IDM result ~0) -> Euler fallback; the result
    # is additionally capped at gap/TS (0.5/0.5 = 1.0).
    out = stop_speed(tensor([0.0], dtype=torch.float64), tensor([8.0]), params(), dt_s=TS, iterations=2)
    assert out.item() == pytest.approx(1.2936153127, abs=5e-7)
    out = stop_speed(tensor([0.0], dtype=torch.float64), tensor([0.5], dtype=torch.float64), params(), dt_s=TS, iterations=2)
    assert out.item() == pytest.approx(0.98, abs=1e-9)
    out = stop_speed(tensor([0.0], dtype=torch.float64), tensor([0.005], dtype=torch.float64), params(), dt_s=TS, iterations=2)
    assert out.item() == 0.0
    # moving vehicle: plain _v without minGap, capped at gap/TS.
    out = stop_speed(tensor([3.0], dtype=torch.float64), tensor([3.0], dtype=torch.float64), params(), dt_s=TS, iterations=2)
    assert out.item() == pytest.approx(1.8205889336, abs=5e-7)
    ref = _scalar_stop(3.0, 3.0, params(), TS, 2)
    assert out.item() == pytest.approx(ref, abs=1e-9)


def test_stop_speed_matches_scalar_grid():
    cases = [(v, g) for v in (0.0, 0.005, 0.02, 1.4, 6.0)
             for g in (0.004, 0.2, 1.0, 4.0, 17.0, 90.0)]
    v = tensor([c[0] for c in cases], dtype=torch_dtype())
    gap = tensor([c[1] for c in cases], dtype=torch_dtype())
    out = stop_speed(v, gap, params(), dt_s=TS, iterations=2)
    ref = [_scalar_stop(c[0], c[1], params(), TS, 2) for c in cases]
    torch.testing.assert_close(out, tensor(ref, dtype=out.dtype),
                               rtol=0.0, atol=5e-7)


# ---------------------------------------------------------------- free speed
def test_free_speed_matches_scalar():
    def scalar_free(v, seen, new_limit, p, ts, iters):
        lane = float(p.des_speed)
        two_sqrt = 2.0 * math.sqrt(float(p.accel) * float(p.decel))
        sec_gap = max(0.0, new_limit * float(p.headway_time)
                      + new_limit * new_limit / two_sqrt)
        if v <= new_limit:
            v_safe = _scalar_v(v, 1.0e6, new_limit, lane, float(p.accel),
                               float(p.decel), float(p.headway_time),
                               float(p.min_gap), float(p.delta), ts, iters, False)
        else:
            v_safe = _scalar_v(v, max(seen, sec_gap), 0.0, lane, float(p.accel),
                               float(p.decel), float(p.headway_time),
                               float(p.min_gap), float(p.delta), ts, iters, False)
        if seen < sec_gap:
            v_safe = min(v_safe, new_limit)
        return v_safe

    p = params()
    cases = [(v, seen, lim) for v in (5.0, 20.0, 29.0)
             for seen in (5.0, 40.0, 150.0)
             for lim in (11.0, 22.0, 31.29)]
    v = tensor([c[0] for c in cases], dtype=torch_dtype())
    seen = tensor([c[1] for c in cases], dtype=torch_dtype())
    lim = tensor([c[2] for c in cases], dtype=torch_dtype())
    out = free_speed(v, seen, lim, p, dt_s=TS, iterations=2)
    ref = [scalar_free(c[0], c[1], c[2], p, TS, 2) for c in cases]
    torch.testing.assert_close(out, tensor(ref, dtype=out.dtype),
                               rtol=0.0, atol=5e-7)


# ------------------------------------------------------------------ insertion
def test_insertion_follow_speed_anchors():
    # v=25, leader 20, bumper gap 40: the moving equilibrium needs 45.77 m so
    # followSpeed relaxes the candidate to 23.368420; at gap 60 the equilibrium
    # is covered and the desired speed is kept.
    out = insertion_follow_speed(25.0, 40.0, 20.0, params(), dt_s=TS, iterations=2)
    assert out == pytest.approx(23.3684198014, abs=5e-7)
    out = insertion_follow_speed(25.0, 60.0, 20.0, params(), dt_s=TS, iterations=2)
    assert out == pytest.approx(25.0, abs=1e-9)


# ---------------------------------------------------------------- action step
def test_action_step_enabled_validation():
    assert action_step_enabled(None, TS) is False
    assert action_step_enabled(TS, TS) is False
    assert action_step_enabled(1.0, TS) is True
    with pytest.raises(ValueError):
        action_step_enabled(0.25, TS)


def test_from_driver_params_maps_encoder_settings():
    dp = {"v0": tensor([28.0]), "time_headway": tensor([0.9]),
          "min_gap": tensor([2.0], dtype=torch.float64), "accel": tensor([2.2]),
          "decel": tensor([4.0])}
    p = SumoIDMParams.from_driver_params(dp)
    torch.testing.assert_close(p.des_speed.double(), tensor([28.0], dtype=torch.float64))
    torch.testing.assert_close(p.headway_time.double(), tensor([0.9], dtype=torch.float64))
    torch.testing.assert_close(p.min_gap.double(), tensor([2.0], dtype=torch.float64))
    torch.testing.assert_close(p.accel.double(), tensor([2.2], dtype=torch.float64))
    torch.testing.assert_close(p.decel.double(), tensor([4.0], dtype=torch.float64))
    assert p.emergency_decel is None
