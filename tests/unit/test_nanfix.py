"""S-NANFIX: finite gradients of the always-commanded execution law.

Culprit: ``sumo_idm.follow_speed`` evaluated with the free-road stand-in
``gap2pred == +inf`` (``base_accel`` of a CAV without a leader). Forward
``desired^2 / (inf*inf) == 0`` is finite, but the backward of ``g * g`` is
``(-0) * inf == NaN``, which poisons d/d(speed) of every free-road CAV and so
the first clip_grad_norm_ step ("total norm ... is non-finite").
The fix (double where) leaves the forward bitwise unchanged. CPU only.
"""
import math

import pytest
import torch

from wms.control import alwayscmd, sumo_idm
from wms.control.sumo_idm import CAV_PARAMS, NUMERICAL_EPS, _iterations

torch.set_num_threads(1)


def _legacy_follow_speed(gap2pred, speed, leader_speed, allowed_speed, *,
                         params=None, dt=0.5, respect_min_gap=True):
    """Verbatim pre-S-NANFIX body (forward reference)."""
    p = dict(CAV_PARAMS if params is None else params)
    a, b, tau, gap0 = (float(p[k]) for k in ("accel", "decel", "time_headway", "min_gap"))
    delta = float(p.get("delta", 4.0))
    iterations = _iterations(dt)
    two_sqrt_ab = 2. * math.sqrt(a * b)
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_forward_bitwise_unchanged_including_infinite_gap(dtype):
    gen = torch.Generator().manual_seed(7)
    n = 4000
    v = torch.rand(n, generator=gen, dtype=dtype) * 33.
    lv = torch.rand(n, generator=gen, dtype=dtype) * 35.
    gap = torch.rand(n, generator=gen, dtype=dtype) * 60. - 2.
    gap[::7] = float("inf")
    gap[1::11] = 0.
    gap[2::13] = 1e-7
    gap[3::17] = 5e-5
    allowed = torch.full((n,), 31.29, dtype=dtype)
    for respect in (True, False):
        new = sumo_idm.follow_speed(gap, v, lv, allowed, respect_min_gap=respect)
        old = _legacy_follow_speed(gap, v, lv, allowed, respect_min_gap=respect)
        assert torch.equal(new, old)


RISKY = {  # name: (speed, leader_speed, gap, has_leader, lane, s)
    "no_leader": (25., 24., 30., False, 0, 100.),
    "no_leader_speed0": (0., 0., 5., False, 0, 100.),
    "no_leader_ramp": (10., 0., 5., False, 2, -50.),
    "gap0": (25., 24., 0., True, 0, 100.),
    "gap_negative": (25., 24., -3., True, 1, 100.),
    "gap_tiny": (25., 24., 1e-6, True, 0, 100.),
    "speed0_leader": (0., 0., 5., True, 0, 100.),
    "lane2_dead_end": (20., 0., 30., True, 2, 75.44),
    "lane2_past_end": (20., 0., 30., True, 2, 90.),
    "mainline_past_merge": (25., 25., 30., True, 0, 1500.),
}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name", sorted(RISKY))
@pytest.mark.parametrize("law", ["base", "cap", "executed"])
def test_risky_fixture_gradients_are_finite(name, law, dtype):
    v, lv, gap, has, lane, s = RISKY[name]
    leaves = [torch.tensor(x, dtype=dtype, requires_grad=True) for x in (v, lv, gap, 0.3)]
    sv, lvt, gp, res = leaves
    sm = torch.tensor(s, dtype=dtype)
    ln = torch.tensor(lane)
    h = torch.tensor(has)
    allowed = sumo_idm.allowed_speed(ln, sm)
    with torch.autograd.detect_anomaly():
        if law == "base":
            y = sumo_idm.base_accel(sv, lvt, gp, allowed, has_leader=h)
        elif law == "cap":
            y = sumo_idm.safe_speed_cap(sv, lvt, gp, allowed, has_leader=h, lane=ln, s_m=sm)
        else:
            y = sumo_idm.executed_accel(sv, lvt, gp, allowed, res, -3., 1.2,
                                        has_leader=h, lane=ln, s_m=sm)
        assert torch.isfinite(y)
        y.backward()
    for t in leaves:
        assert t.grad is None or torch.isfinite(t.grad)


def test_free_road_gradient_equals_huge_finite_gap_gradient():
    """No leader: d(base)/d(speed) equals the derivative with a 1e9 m finite gap."""
    allowed = torch.tensor(31.29, dtype=torch.float64)

    def grad_with(gap):
        v = torch.tensor(25., dtype=torch.float64, requires_grad=True)
        z = sumo_idm.follow_speed(torch.tensor(gap, dtype=torch.float64), v,
                                  v.detach(), allowed)
        (z.minimum(allowed).sub(v) / 0.5).backward()
        return float(v.grad)

    g_inf = grad_with(float("inf"))
    assert math.isfinite(g_inf)
    assert math.isclose(g_inf, grad_with(1e9), rel_tol=1e-6)


def _batch_state(dtype, *, with_grad):
    """B=2, K=6: free-road CAV, near-zero gaps, speed 0, absent slot, junction."""
    s = torch.tensor([[10., 40., 41., 50., 20., 0.],
                      [-30., 5., 60., 75.44, 90., 100.]], dtype=dtype)
    v = torch.tensor([[25., 20., 0., 15., 0., 0.],
                      [10., 12., 3., 20., 25., 30.]], dtype=dtype)
    lane = torch.tensor([[0, 0, 0, 1, -1, 0], [2, 2, 2, 2, 0, 1]])
    present = torch.tensor([[1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    junction = torch.tensor([[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]], dtype=torch.bool)
    carry = torch.tensor([[-1, -1, -1, -1, -1, 0], [-1, -1, -1, -1, -1, -1]])
    if with_grad:
        s = s.clone().requires_grad_(True)
        v = v.clone().requires_grad_(True)
    return {"s_m": s, "v_mps": v, "a_mps2": torch.zeros_like(v), "lane": lane,
            "junction": junction, "lane_carry": carry, "present": present,
            "is_cav": present.clone(), "length_m": torch.full_like(v, 5.)}


@pytest.mark.parametrize("rule", ["0", "1"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batched_alwayscmd_executed_gradients_finite(monkeypatch, rule, dtype):
    monkeypatch.setenv("C4_SUMO_LEADER_RULE", rule)
    state = _batch_state(dtype, with_grad=True)
    residual = torch.full(state["v_mps"].shape, 0.2, dtype=dtype, requires_grad=True)
    with torch.autograd.detect_anomaly():
        a = alwayscmd.executed(state, residual)
        base = alwayscmd.inferred_base(state)
        cap = alwayscmd.inferred_cap(state)
        assert torch.isfinite(a).all() and torch.isfinite(base).all()
        assert torch.isfinite(cap).all()
        (a.sum() + base.sum() + cap.sum()).backward()
    for t in (state["s_m"], state["v_mps"], residual):
        assert torch.isfinite(t.grad).all()
    # the fixture must contain free-road rows (the original culprit)
    assert not alwayscmd._wm_geometry(state)[3].all()


def test_clip_grad_norm_finite_on_free_road_state(monkeypatch):
    monkeypatch.setenv("C4_SUMO_LEADER_RULE", "1")
    state = _batch_state(torch.float32, with_grad=False)
    state["v_mps"] = state["v_mps"].clone().requires_grad_(True)
    w = torch.nn.Parameter(torch.zeros(state["v_mps"].shape))
    a = alwayscmd.executed(state, w)
    a.sum().backward()
    # the trainer's clip: raises on a non-finite total norm
    torch.nn.utils.clip_grad_norm_([w], 1.0, error_if_nonfinite=True)
