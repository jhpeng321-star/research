"""Regression test for the RUN2 traffic-loss magnitude blow-up.

Run 2's traffic term (weight 0.1, ramp 30-120 s) reached 1e7-1e9 because the
per-metric ``metric_scales`` divided by numbers 50-100x *smaller* than the
metric ranges, and a diverging rollout had no saturation cap.  This test pins
the corrected, bounded behaviour; see ``_out/RUN2_TRAFFIC_LOSS_DIAG.md``.
"""
from __future__ import annotations

import torch

from output.wd01.wm_n1.wms.traffloss import TrafficLossConfig, traffic_bin_loss

LEGACY_SCALES = {"mean_speed": 0.1, "density": 0.01, "stopped_vehicles": 0.2,
                 "merge_queue_length": 0.02, "normalized_delay": 1.0 / 60.0}


def _diverging(T_steps: int, stop_frac: float, K: int = 30):
    """Recorded stays free-flow; predicted merge vehicles slow and pile up."""
    base_s = torch.linspace(-120.0, 300.0, K).view(1, 1, K).repeat(1, T_steps, 1)
    ramp = torch.linspace(0.0, 1.0, T_steps).view(1, T_steps, 1)
    merge = (base_s >= 0.0) & (base_s <= 75.0)
    s = torch.where(merge, base_s - ramp * stop_frac * 75.0, base_s)
    v = torch.where(merge, 12.0 * (1.0 - ramp * stop_frac),
                    torch.full_like(base_s, 12.0))
    p = torch.ones(1, T_steps, K, dtype=torch.bool)
    return s, v, p


def _loss(cfg, T_s=300.0, stop_frac=1.0):
    T_steps = int(T_s / 0.5)
    ps, pv, pp = _diverging(T_steps, stop_frac)
    ts, tv, tp = _diverging(T_steps, 0.0)
    idx = torch.arange(T_steps)
    return traffic_bin_loss(ps, pv, pp, ts, tv, tp, step_indices=idx, cfg=cfg)


def test_default_scales_keep_divergent_loss_bounded():
    loss, _ = _loss(TrafficLossConfig())
    assert float(loss) < 30.0
    assert torch.isfinite(loss)


def test_legacy_scales_are_the_documented_blow_up():
    legacy = TrafficLossConfig(metric_scales=dict(LEGACY_SCALES), residual_cap=None)
    loss, _ = _loss(legacy)
    assert float(loss) > 1.0e5


def test_corrected_scales_are_two_orders_below_legacy():
    fixed, _ = _loss(TrafficLossConfig())
    legacy, _ = _loss(TrafficLossConfig(metric_scales=dict(LEGACY_SCALES),
                                        residual_cap=None))
    assert float(legacy) / float(fixed) > 100.0


def test_residual_cap_saturates_a_blow_up():
    small = TrafficLossConfig(residual_cap=5.0)
    huge = TrafficLossConfig(residual_cap=5.0)
    a, _ = _loss(small, stop_frac=0.5)
    b, _ = _loss(huge, stop_frac=1.0)
    # Past the cap the metric residual cannot keep growing: the loss difference
    # stays within one order of magnitude instead of exploding ~100x.
    assert float(b) < 10.0 * float(a)


def test_default_cap_is_positive_and_metadata_stable():
    cfg = TrafficLossConfig()
    assert cfg.metric_scales["normalized_delay"] == 10.0
    assert cfg.metric_scales["merge_queue_length"] == 100.0
    assert cfg.residual_cap == 5.0
    try:
        TrafficLossConfig(residual_cap=0.0)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("non-positive residual_cap must raise")


def test_identical_prediction_is_exactly_zero_under_new_defaults():
    T_steps = 60
    s, v, p = _diverging(T_steps, 0.0)
    idx = torch.arange(T_steps)
    loss, diag = traffic_bin_loss(s, v, p, s.clone(), v.clone(), p.clone(),
                                  step_indices=idx, cfg=TrafficLossConfig())
    assert float(loss) < 1e-9
    assert all(value < 1e-6 for value in diag.values())


def test_gradients_finite_and_bounded_under_new_defaults():
    T_steps = 120
    ps, pv, pp = _diverging(T_steps, 1.0)
    ts, tv, tp = _diverging(T_steps, 0.0)
    ps.requires_grad_(True)
    pv.requires_grad_(True)
    idx = torch.arange(T_steps)
    loss, _ = traffic_bin_loss(ps, pv, pp, ts, tv, tp, step_indices=idx,
                               cfg=TrafficLossConfig())
    loss.backward()
    assert torch.isfinite(ps.grad).all() and torch.isfinite(pv.grad).all()
    assert float(ps.grad.abs().max()) < 1.0
