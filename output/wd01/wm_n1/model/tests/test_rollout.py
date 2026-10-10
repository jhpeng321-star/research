"""Validity layer and rollout tests (design sections 2.2/2.3/2.5)."""
from __future__ import annotations

import torch

from model import constants as C
from model.net import WmN1Net
from model.rollout import project_step, rollout
from synth_model import make_batch, small_cfg


def test_validity_clips_accel_and_speed():
    s = torch.zeros(1, 2)
    lane = torch.tensor([[1, 1]])
    v = torch.tensor([[1.0, 10.0]])
    length = torch.full((1, 2), 5.0)
    mask = torch.ones(1, 2, dtype=torch.bool)
    accel = torch.tensor([[-30.0, 50.0]])
    cls = torch.zeros(1, 2, dtype=torch.long)
    s2, l2, v2, a2, counts = project_step(s, lane, v, length, mask, accel, cls)
    assert counts.accel_clip == 2
    assert float(v2.min()) >= 0.0
    assert counts.speed_clip == 1
    assert float(a2.min()) >= -9.0 - 1e-6 and float(a2.max()) <= 3.0 + 1e-6


def test_validity_rejects_lane_change_into_occupied_gap():
    s = torch.tensor([[0.0, 1.0]])
    lane = torch.tensor([[0, 1]])
    v = torch.tensor([[5.0, 5.0]])
    length = torch.full((1, 2), 5.0)
    mask = torch.ones(1, 2, dtype=torch.bool)
    accel = torch.zeros(1, 2)
    cls = torch.tensor([[2, 0]])            # vehicle 0 wants right into lane 1
    s2, l2, v2, a2, counts = project_step(s, lane, v, length, mask, accel, cls)
    assert counts.lane_reject == 1
    assert int(l2[0, 0]) == 0               # stayed


def test_validity_lane_bounds():
    s = torch.zeros(1, 2)
    lane = torch.tensor([[0, 2]])
    v = torch.full((1, 2), 5.0)
    length = torch.full((1, 2), 5.0)
    mask = torch.ones(1, 2, dtype=torch.bool)
    accel = torch.zeros(1, 2)
    cls = torch.tensor([[1, 1]])            # left from lane 0 and lane 2 invalid
    s2, l2, v2, a2, counts = project_step(s, lane, v, length, mask, accel, cls)
    assert counts.lane_reject == 2
    assert l2.tolist() == [[0, 2]]


def test_validity_gap_projection_keeps_nonnegative_gap():
    s = torch.tensor([[0.0, 2.0]])
    lane = torch.tensor([[1, 1]])
    v = torch.tensor([[5.0, 5.0]])
    length = torch.full((1, 2), 5.0)
    mask = torch.ones(1, 2, dtype=torch.bool)
    accel = torch.zeros(1, 2)
    cls = torch.zeros(1, 2, dtype=torch.long)
    s2, l2, v2, a2, counts = project_step(s, lane, v, length, mask, accel, cls)
    assert counts.gap_projection == 1
    order = sorted([float(s2[0, 0]), float(s2[0, 1])])
    assert order[1] - order[0] >= 5.0 - 1e-6
    assert float(v2.min()) >= 0.0


def test_rollout_deterministic_under_fixed_seed():
    net = WmN1Net(small_cfg()).eval()
    b = make_batch(B=1, K=6)
    r1 = rollout(net, b, n_steps=3, n_samples=1, seed=123)
    r2 = rollout(net, b, n_steps=3, n_samples=1, seed=123)
    assert torch.equal(r1["positions"], r2["positions"])
    assert torch.equal(r1["speeds"], r2["speeds"])
    assert torch.equal(r1["lanes"], r2["lanes"])


def test_rollout_shapes_and_safety():
    net = WmN1Net(small_cfg()).eval()
    b = make_batch(B=2, K=8)
    r = rollout(net, b, n_steps=3, n_samples=2, seed=7)
    assert r["positions"].shape == (2, 4, 2, 8)
    assert r["speeds"].shape == (2, 4, 2, 8)
    assert r["arrivals"].shape == (2, 3, 2, 2)
    assert float(r["speeds"].min()) >= 0.0
    assert int(r["arrivals"].min()) >= 0
    assert len(r["validity"]) == 2
    assert all(v.accel_clip >= 0 for v in r["validity"])


# ------------------------------------------------- D1 integration guards ----
def test_position_advances_with_zero_accel():
    """D1: zero predicted acceleration must still advance s by v*dt."""
    s = torch.tensor([[0.0, 100.0, 200.0]])
    lane = torch.tensor([[1, 1, 1]])
    v = torch.tensor([[8.0, 12.0, 5.0]])
    length = torch.full((1, 3), 5.0)
    mask = torch.ones(1, 3, dtype=torch.bool)
    accel = torch.zeros(1, 3)
    cls = torch.zeros(1, 3, dtype=torch.long)
    s2, l2, v2, a2, counts = project_step(s, lane, v, length, mask, accel, cls)
    assert torch.allclose(s2, s + v * C.DT_S, atol=1e-4)
    assert torch.allclose(v2, v)
    assert counts.gap_projection == 0


def test_constant_accel_matches_trapezoidal_closed_form():
    """D1: constant a -> s_n = s0 + n*v0*dt + 0.5*a*(n*dt)^2."""
    dt, v0, a, n = C.DT_S, 4.0, 1.5, 6
    s = torch.tensor([[0.0]])
    lane = torch.tensor([[1]])
    v = torch.tensor([[v0]])
    length = torch.full((1, 1), 5.0)
    mask = torch.ones(1, 1, dtype=torch.bool)
    cls = torch.zeros(1, 1, dtype=torch.long)
    for _ in range(n):
        s, lane, v, a_used, _ = project_step(
            s, lane, v, length, mask, torch.full((1, 1), a), cls)
    expected_s = v0 * n * dt + 0.5 * a * (n * dt) ** 2
    assert abs(float(s[0, 0]) - expected_s) < 1e-4
    assert abs(float(v[0, 0]) - (v0 + a * n * dt)) < 1e-4


def test_braking_vehicle_stops_and_never_reverses():
    """D1: hard braking clamps at v=0 and the position never decreases."""
    s = torch.tensor([[0.0]])
    lane = torch.tensor([[1]])
    v = torch.tensor([[3.0]])
    length = torch.full((1, 1), 5.0)
    mask = torch.ones(1, 1, dtype=torch.bool)
    cls = torch.zeros(1, 1, dtype=torch.long)
    prev = 0.0
    for _ in range(8):
        s, lane, v, a_used, counts = project_step(
            s, lane, v, length, mask, torch.full((1, 1), -9.0), cls)
        assert float(s[0, 0]) >= prev - 1e-6
        assert float(v[0, 0]) >= 0.0
        prev = float(s[0, 0])
    assert float(v[0, 0]) == 0.0


def test_no_same_lane_overlap_random_scenes():
    """D1: after the position update, rear bumper gap >= min gap holds."""
    gen = torch.Generator().manual_seed(2024)
    B, K = 4, 12
    for _ in range(5):
        s = torch.rand(B, K, generator=gen) * 100.0
        lane = torch.randint(0, C.N_LANES, (B, K), generator=gen)
        v = torch.rand(B, K, generator=gen) * 20.0
        length = 4.0 + torch.rand(B, K, generator=gen) * 2.0
        mask = torch.ones(B, K, dtype=torch.bool)
        accel = torch.randn(B, K, generator=gen) * 3.0
        cls = torch.zeros(B, K, dtype=torch.long)
        s2, l2, v2, a2, counts = project_step(
            s, lane, v, length, mask, accel, cls)
        assert float(v2.min()) >= 0.0
        for b in range(B):
            for L in range(C.N_LANES):
                idx = [i for i in range(K) if int(l2[b, i]) == L]
                idx.sort(key=lambda i: float(s2[b, i]), reverse=True)
                for front, back in zip(idx[:-1], idx[1:]):
                    gap = float(s2[b, front]) - float(s2[b, back])
                    min_gap = 0.5 * (float(length[b, front])
                                     + float(length[b, back]))
                    assert gap >= min_gap - 1e-4


def test_rollout_positions_advance():
    """D1: the autoregressive rollout is a real trajectory generator."""
    net = WmN1Net(small_cfg()).eval()
    b = make_batch(B=1, K=6)
    r = rollout(net, b, n_steps=4, n_samples=1, seed=11)
    p = r["positions"][0]
    assert p.shape == (5, 1, 6)                 # shape unchanged
    assert float((p[-1] - p[0]).abs().max()) > 1e-3
