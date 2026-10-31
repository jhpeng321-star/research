"""S-WMFIX-C fallback 1: TRAINING-ONLY smoothed admission relaxation (CPU).

The surrogate must (a) agree with central finite differences on a small
fixture (the gate the BLOCKED real feasibility could not measure), (b) stay an
exact identity when the flag is OFF, and (c) bracket the hard events: an open
gate drains the queue, a closed gate reproduces the exact no-service ledger
integral.
"""
import numpy as np
import pytest
import torch

from tools.wmfixmerge.runtime import TrackedEntries, admission_threshold_series
from wms.model.entries import (ENTRY_S_M, NOMINAL_LENGTH_M, SUMO_POSITION_EPS_M,
                               SUMO_V2_MIN_GAP_M, EntranceHistory, EntryConfig,
                               TorchEntryProcessV2)
from wms.model.learned_admission import LearnedAdmission
from wms.train.merge_response import (entrance_gap_series, smooth_admission_source)
from tools.errsrc.inference import LearnedMix


def state():
    return dict(s_m=torch.zeros(1, 4), v_mps=torch.zeros(1, 4), a_mps2=torch.zeros(1, 4),
        lane=torch.zeros(1, 4, dtype=torch.long), length_m=torch.ones(1, 4) * 5,
        present=torch.zeros(1, 4, dtype=torch.bool), is_cav=torch.zeros(1, 4, dtype=torch.bool))


def inferred():
    return {n: torch.tensor([[v, v * 1.1, v, v]]) for n, v in
            (("v0", 17.), ("time_headway", 1.7), ("decel", 2.3), ("min_gap", 2.1))}


def cav_pool_row():
    return torch.tensor([[21.0, 2.0, 3.0, 1.5]])


def learned_process(mix, seed=17):
    cfg = EntryConfig(conserve_queue=True, learned_admission=LearnedAdmission(mix),
                      insert_rule="sumo_like_v2")
    return TorchEntryProcessV2(history=EntranceHistory.defaults(), cfg=cfg,
                               row_seeds=[seed], couple_draws=True)


def fixture_series(theta):
    """Tiny differentiable rollout: [B=1, T=4, K=3], one vehicle per lane."""
    base = torch.tensor([[[-378.66, -2638., -2636.]]], dtype=theta.dtype).repeat(1, 4, 1)
    drift = torch.tensor([[[.0, 3., 2.], [1., 6., 4.], [2., 9., 6.], [3., 12., 8.]]],
                         dtype=theta.dtype)
    pre_pos = (base + theta * drift).clone().requires_grad_(True)
    pre_speeds = torch.full_like(pre_pos, 4.0)
    pre_present = torch.ones_like(pre_pos, dtype=torch.bool)
    pre_lanes = torch.tensor([[[2, 0, 1]]]).repeat(1, 4, 1)
    pre_length = torch.full_like(pre_pos, 5.0)
    return pre_pos, pre_speeds, pre_present, pre_lanes, pre_length


def surrogate(theta, width=1.0):
    import wms.train.merge_response as mr
    pre = fixture_series(theta)
    gap = entrance_gap_series(*pre)
    queue0 = torch.tensor([[2., 3.]], dtype=theta.dtype)
    due = torch.tensor([[[1., 0.], [0., 1.], [1., 0.], [0., 1.]]], dtype=theta.dtype)
    min_gap = torch.full_like(gap, 2.0)
    return smooth_admission_source(queue0, due, gap, min_gap, width=width)[0], gap


def test_smooth_admission_source_matches_central_finite_differences():
    torch.set_num_threads(1)
    for width in (0.5, 1.0, 2.0):
        theta = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        source, gap = surrogate(theta, width)
        analytic = torch.autograd.grad(source.sum(), theta)[0]
        assert torch.isfinite(analytic) and analytic.abs() > 0
        for eps in (1e-4, 1e-5, 1e-6):
            values = []
            for sign in (1, -1):
                point = torch.tensor(3.0 + sign * eps, dtype=torch.float64)
                values.append(surrogate(point, width)[0].detach())
            fd = ((values[0] - values[1]) / (2 * eps)).sum()
            torch.testing.assert_close(analytic, fd, rtol=1e-5, atol=1e-9)
        # the gate itself is differentiable through the gaps, finite everywhere
        assert torch.isfinite(gap).all()


def test_smooth_admission_source_brackets_the_hard_events():
    queue0 = torch.tensor([[2., 3.]])
    due = torch.tensor([[[1., 0.], [0., 1.], [1., 0.], [0., 1.]]])
    min_gap = torch.full_like(due, 2.0)
    # open gate (gap >> min gap): everything is served after the first attempt
    open_gap = torch.full_like(due, 1.0e9)
    source, queue = smooth_admission_source(queue0, due, open_gap, min_gap)
    assert source.item() == pytest.approx(0.5 * 5.0)          # only Q0 is left waiting
    assert (queue[:, 1:] == 0).all()
    # closed gate (gap << min gap): the exact no-service ledger integral
    closed_gap = torch.full_like(due, -1.0e9)
    source, queue = smooth_admission_source(queue0, due, closed_gap, min_gap)
    expected = 0.5 * sum((queue0 + due[:, :t].sum(1)).sum() for t in range(4))
    assert source.item() == pytest.approx(expected.item())
    with pytest.raises(ValueError):
        smooth_admission_source(queue0, due, closed_gap, min_gap, width=0.0)


def test_gradient_flows_only_through_the_gate():
    queue0 = torch.tensor([[1., 0.]])
    due = torch.zeros(1, 3, 2)
    min_gap = torch.full_like(due, 2.0)
    gap = torch.full_like(due, 2.0).requires_grad_(True)      # right at the boundary
    source, _ = smooth_admission_source(queue0, due, gap, min_gap)
    grad = torch.autograd.grad(source.sum(), gap)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    far = torch.full_like(due, 1.0e6).requires_grad_(True)    # saturated gate: zero grad
    source, _ = smooth_admission_source(queue0, due, far, min_gap)
    assert torch.autograd.grad(source.sum(), far)[0].abs().sum() == 0


def test_entrance_gap_series_geometry_and_end_of_step_credit():
    pre_pos = torch.tensor([[[-373.46, -2637., -2635.]]], dtype=torch.float64).repeat(1, 2, 1)
    pre_speeds = torch.full_like(pre_pos, 2.0)
    pre_present = torch.ones_like(pre_pos, dtype=torch.bool)
    pre_lanes = torch.tensor([[[2, 0, 1]]]).repeat(1, 2, 1)
    pre_length = torch.full_like(pre_pos, 5.0)
    gap = entrance_gap_series(pre_pos, pre_speeds, pre_present, pre_lanes, pre_length,
                              dt_s=0.5)
    # ramp: rear at -378.46, front at entry+5+eps, plus 2 m/s * 0.5 s credit
    front = ENTRY_S_M[2] + NOMINAL_LENGTH_M + SUMO_POSITION_EPS_M
    assert gap[0, 0, 1].item() == pytest.approx(-378.46 - front + 1.0)
    # mainline: the LARGER of the two lane gaps (incumbent "any" order)
    lane0 = -2637. - 5. - (ENTRY_S_M[0] + NOMINAL_LENGTH_M + SUMO_POSITION_EPS_M) + 1.0
    lane1 = -2635. - 5. - (ENTRY_S_M[1] + NOMINAL_LENGTH_M + SUMO_POSITION_EPS_M) + 1.0
    assert gap[0, 0, 0].item() == pytest.approx(max(lane0, lane1))
    # an empty entrance lane is an infinite gap (the hard path always admits)
    empty = torch.zeros(1, 1, 1).double()
    gap = entrance_gap_series(empty, empty, torch.zeros(1, 1, 1, dtype=torch.bool),
                              torch.zeros(1, 1, 1, dtype=torch.long),
                              torch.zeros(1, 1, 1).double(), dt_s=0.5)
    assert torch.isinf(gap).all()


def test_tracking_off_is_exact_identity_and_on_records_thresholds():
    mix = LearnedMix(inferred(), torch.ones(1, 4, 1, dtype=torch.bool), 5, ["world"],
                     cav_pools=[cav_pool_row()])
    proc = learned_process(mix, seed=43)
    tracked = TrackedEntries(proc)
    tracked_on = TrackedEntries(learned_process(mix, seed=43), track_admission=True)
    for step, blocked in enumerate((True, False)):
        s = state()
        if blocked:  # a stopped blocker 1 m past the ramp entrance: tested, refused
            front = float(proc.entry_s[2]) + 5.0 + 0.1
            s["present"][0, 3], s["lane"][0, 3] = True, 2
            s["s_m"][0, 3], s["v_mps"][0, 3] = front + 1.0 + 5.0, 0.0
        proc._emit_due = tracked.process._emit_due = tracked_on.process._emit_due = (
            lambda b: [1])
        a, b = tracked.insert(s), s  # noqa: F841 (a marks the newborn slots)
        s2 = {k: v.clone() if torch.is_tensor(v) else v for k, v in s.items()}
        tracked_on.insert(s2)
    # OFF: no tracking state on the process, identical tracked series
    assert "track_admission" not in vars(tracked.process)
    assert "last_admission_min_gaps" not in vars(tracked.process)
    assert len(tracked.present) == len(tracked_on.present) == 2
    for off, on in zip(tracked.queues, tracked_on.queues):
        assert np.array_equal(off, on)
    # ON: the hard path recorded the SAME inferred min-gaps it tested (the
    # blocked step tests candidate 0; the free step also tests candidate 1)
    assert tracked_on.admission_thresholds
    tested = {value for row in tracked_on.admission_thresholds for value in row.values()}
    admission = LearnedAdmission(mix)
    candidates = {admission.candidate(0, 1, k)["min_gap"] for k in range(3)}
    assert tested <= candidates
    assert admission.candidate(0, 1, 0)["min_gap"] in tested
    assert tracked_on.admission_thresholds[0][(0, 1)] == pytest.approx(
        admission.candidate(0, 1, 0)["min_gap"])


def test_admission_threshold_series_defaults_and_forward_fill():
    steps_rows = [{(0, 1): 2.7}, {}, {(0, 0): 3.1}, {}]
    series = admission_threshold_series(steps_rows, 1, 4)
    default = float(SUMO_V2_MIN_GAP_M[0])
    assert series.shape == (1, 4, 2)
    assert series[0, 0, 0] == default and series[0, 0, 1] == 2.7
    assert series[0, 1, 1] == 2.7            # ffill of the last tested value
    assert series[0, 2, 0] == 3.1 and series[0, 3, 0] == 3.1
    assert not series.requires_grad          # detached constants only
