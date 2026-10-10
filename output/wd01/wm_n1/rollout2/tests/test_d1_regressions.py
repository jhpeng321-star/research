"""P3 D1 regression set (INTERFACES_V2 Mismatch 10 coverage).

The D1 regression tests live on ``zcode/wmn1d1:model/tests/test_rollout.py``
and are NOT on this branch; this module adds the equivalent coverage for
``rollout2.validity``: zero-accel v*dt, constant-accel closed form,
braking no-reverse, random-scene no-overlap, determinism and shapes.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from model.constants import A_MAX, A_MIN
from rollout2 import (ValidityCounts, integrate_ballistic, integrate_d1,
                      project_rollout_step, project_same_lane_gaps)

DT = 0.5


def test_zero_accel_moves_by_v_dt():
    s = torch.tensor([[100.0, 0.0, -30.0]], dtype=torch.float64)
    v = torch.tensor([[25.0, 0.0, 3.0]], dtype=torch.float64)
    a = torch.zeros_like(v)
    s_new, v_new = integrate_d1(s, v, a, dt_s=DT)
    assert torch.equal(v_new, v)
    assert torch.allclose(s_new, s + v * DT)


def test_constant_accel_closed_form():
    """Exact for constant a when the speed does not clamp:
    s + v*dt + 0.5*a*dt^2 and v + a*dt."""
    v0 = torch.tensor([[10.0, 30.0]], dtype=torch.float64)
    a = torch.tensor([[2.0, -1.5]], dtype=torch.float64)
    s = torch.zeros_like(v0)
    s_new, v_new = integrate_d1(s, v0, a, dt_s=DT)
    assert torch.allclose(v_new, v0 + a * DT)
    assert torch.allclose(s_new, v0 * DT + 0.5 * a * DT ** 2)


def test_braking_never_reverses():
    """v_new = max(0, v + a*dt); a stopped vehicle never moves backwards."""
    s = torch.tensor([[5.0, 5.0]], dtype=torch.float64)
    v = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    a = torch.tensor([[-9.0, -9.0]], dtype=torch.float64)
    s_new, v_new = integrate_d1(s, v, a, dt_s=DT)
    assert torch.equal(v_new, torch.zeros_like(v_new))
    # row 0 travels the average-speed distance 0.5*(1+0)*0.5; row 1 stays
    assert torch.allclose(s_new, torch.tensor([[5.25, 5.0]],
                                              dtype=torch.float64))


def test_ballistic_integrator_is_the_v2g_form():
    v = torch.tensor([[20.0, 2.0]], dtype=torch.float64)
    v_new = torch.tensor([[22.0, 0.0]], dtype=torch.float64)
    s = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    s_ball = integrate_ballistic(s, v_new, dt_s=DT)
    assert torch.allclose(s_ball, v_new * DT)
    s_trap, _ = integrate_d1(s, v, (v_new - v) / DT, dt_s=DT)
    assert torch.allclose(s_trap - s_ball,
                          0.5 * (v - v_new) * DT)  # the documented delta


def test_project_clips_to_validity_bounds():
    v = torch.full((1, 3), 10.0, dtype=torch.float64)
    s = torch.zeros((1, 3), dtype=torch.float64)
    accel = torch.tensor([[+50.0, -50.0, 0.0]], dtype=torch.float64)
    lane = torch.zeros((1, 3), dtype=torch.int64)
    length = torch.full((1, 3), 5.0, dtype=torch.float64)
    present = torch.ones((1, 3), dtype=torch.bool)
    cls = torch.zeros((1, 3), dtype=torch.int64)
    _s, _l, v_new, a_used, _c, counts = project_rollout_step(
        s, lane, v, length, present, accel, cls, dt_s=DT)
    assert counts.accel_clip == 2
    assert a_used[0, 0].item() == float(A_MAX)
    assert a_used[0, 1].item() == float(A_MIN)
    assert v_new.min() >= 0.0


def test_lane_change_bounds_and_overlap_rejects():
    """A lane change into an occupied target gap is rejected (counted);
    into a free gap it is applied; a class with no valid target from the
    current lane (e.g. left from lane 0) is rejected (v1 rule)."""
    b, k = 2, 4
    s = torch.tensor([[10.0, 10.0, 50.0, 0.0],
                      [10.0, 100.0, 50.0, 0.0]], dtype=torch.float64)
    lane = torch.tensor([[1, 2, 0, -1], [1, 2, 0, -1]],
                        dtype=torch.int64)
    length = torch.full((b, k), 5.0, dtype=torch.float64)
    present = torch.tensor([[True, True, True, False],
                            [True, True, True, False]])
    v = torch.zeros((b, k), dtype=torch.float64)
    accel = torch.zeros((b, k), dtype=torch.float64)
    cls = torch.zeros((b, k), dtype=torch.int64)
    cls[0, 0] = 2   # right (1 -> 2): slot 1 occupies lane 2 at s=10 -> reject
    cls[1, 0] = 1   # left (1 -> 0): nearest lane-0 vehicle at s=50 -> free
    cls[0, 2] = 1   # left requested from lane 0 -> no valid target -> reject
    _s, lane_new, _v, _a, _cls, counts = project_rollout_step(
        s, lane, v, length, present, accel, cls, dt_s=DT)
    assert int(lane_new[0, 0]) == 1          # overlap reject -> stay
    assert int(lane_new[1, 0]) == 0          # free change -> applied
    assert int(lane_new[0, 2]) == 0          # invalid target -> stay
    assert counts.lane_reject == 2
    assert counts.gap_projection >= 0


def test_random_scene_no_overlap_after_projection():
    rng = np.random.default_rng(11)
    for trial in range(8):
        b, k = 2, 12
        s = torch.from_numpy(
            np.sort(rng.uniform(0, 60, size=(b, k)), axis=1).astype(
                np.float64))
        v = torch.from_numpy(rng.uniform(0, 30, size=(b, k)))
        accel = torch.from_numpy(rng.uniform(-6, 2, size=(b, k)))
        lane = torch.from_numpy(rng.integers(0, 3, size=(b, k)))
        length = torch.full((b, k), 5.0, dtype=torch.float64)
        present = torch.ones((b, k), dtype=torch.bool)
        cls = torch.zeros((b, k), dtype=torch.int64)
        s_new, lane_new, v_new, _a, _cls, counts = project_rollout_step(
            s, lane, v, length, present, accel, cls, dt_s=DT)
        assert v_new.min() >= 0.0
        assert int(lane_new.min()) >= 0 and int(lane_new.max()) <= 2
        # same-lane adjacent bumper gaps must be >= 0 (projection ran)
        for bb in range(b):
            for ln in range(3):
                idx = [i for i in range(k)
                       if bool(present[bb, i]) and int(lane_new[bb, i]) == ln]
                idx.sort(key=lambda i: float(s_new[bb, i]))
                for f, bk in zip(idx[:-1], idx[1:]):
                    gap = (float(s_new[bb, bk]) - 5.0
                           - float(s_new[bb, f]))
                    assert gap >= -1e-9, (trial, bb, ln, gap)
        assert counts.gap_projection >= 0


def test_project_determinism():
    rng = np.random.default_rng(3)
    s = torch.from_numpy(rng.uniform(0, 100, size=(1, 8)))
    v = torch.from_numpy(rng.uniform(5, 30, size=(1, 8)))
    accel = torch.from_numpy(rng.uniform(-4, 2, size=(1, 8)))
    lane = torch.from_numpy(rng.integers(0, 3, size=(1, 8)))
    length = torch.full((1, 8), 5.0, dtype=torch.float64)
    present = torch.ones((1, 8), dtype=torch.bool)
    cls = torch.from_numpy(rng.integers(0, 3, size=(1, 8)))
    out1 = project_rollout_step(s, lane, v, length, present, accel, cls)
    out2 = project_rollout_step(s, lane, v, length, present, accel, cls)
    for x, y in zip(out1[:5], out2[:5]):
        assert torch.equal(x, y)
    assert out1[5].as_dict() == out2[5].as_dict()


def test_shapes_and_counts_type():
    s = torch.zeros((3, 6), dtype=torch.float32)
    v = torch.full((3, 6), 10.0, dtype=torch.float32)
    accel = torch.zeros_like(v)
    lane = torch.zeros((3, 6), dtype=torch.int64)
    length = torch.full((3, 6), 5.0, dtype=torch.float32)
    present = torch.ones((3, 6), dtype=torch.bool)
    cls = torch.zeros((3, 6), dtype=torch.int64)
    out = project_rollout_step(s, lane, v, length, present, accel, cls)
    assert len(out) == 6
    assert isinstance(out[5], ValidityCounts)
    assert out[0].dtype == torch.float32      # dtype is preserved
    counts = ValidityCounts()
    s2, v2, counts = project_same_lane_gaps(
        s, lane, v, length, present, counts=counts)
    assert s2.shape == s.shape and v2.shape == v.shape


def test_gap_projection_prevents_overlap_but_counts():
    """A follower projected onto its leader keeps the bumper gap >= 0
    and the intervention is counted."""
    # lane 1: stopped leader at s=100; follower at s=96 (v=8, braking)
    # integrates to s=99.25 -> bumper gap -5.75 -> projected to s=95
    s = torch.tensor([[100.0, 96.0]], dtype=torch.float64)
    v = torch.tensor([[0.0, 8.0]], dtype=torch.float64)
    accel = torch.tensor([[0.0, -6.0]], dtype=torch.float64)
    lane = torch.ones((1, 2), dtype=torch.int64)
    length = torch.full((1, 2), 5.0, dtype=torch.float64)
    present = torch.ones((1, 2), dtype=torch.bool)
    cls = torch.zeros((1, 2), dtype=torch.int64)
    s_new, _l, v_new, _a, _cls, counts = project_rollout_step(
        s, lane, v, length, present, accel, cls, dt_s=DT)
    # leader: v_new=0, s_new=100; follower limit = 100 - 5 = 95
    assert float(s_new[0, 0]) == 100.0
    assert float(s_new[0, 1]) == pytest.approx(95.0)
    # v2g/v1 bumper gap: leader rear minus follower reference
    gap = float(s_new[0, 0]) - 5.0 - float(s_new[0, 1])
    assert gap >= -1e-9
    assert counts.gap_projection == 1
    assert float(v_new[0, 1]) <= 8.0
