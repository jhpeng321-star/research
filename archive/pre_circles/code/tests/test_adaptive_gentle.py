"""Tests for adaptive gentleness: fair_balance factor modulated by the
trailing corridor-TTC EMA.

Smooth corridor (EMA <= ttc_lo) -> gentle factor (closer to 1.0, milder
slowdown); chaotic corridor (EMA >= ttc_hi) -> full-strength fair_factor;
linear interpolation between. The EMA ticks once per env step inside
update_stream_delay_rates (the existing once-per-step hook).
"""

from env.actions import ShepherdActionEngine
from tests.test_shepherd_actions import _make_engine


def _engine(**mode_overrides) -> ShepherdActionEngine:
    mc = {
        "fair_balance_factor": 0.90,
        "adaptive_gentle": True,
        "adaptive_gentle_factor": 0.93,
        "adaptive_gentle_ttc_lo": 0.5,
        "adaptive_gentle_ttc_hi": 2.0,
        "adaptive_gentle_ema_alpha": 0.5,  # fast EMA for tests
    }
    mc.update(mode_overrides)
    return _make_engine(mode_config=mc)


def _tick(engine: ShepherdActionEngine, count: int, steps: int = 1) -> None:
    engine.corridor_ttc_fn = lambda: count
    for _ in range(steps):
        engine.update_stream_delay_rates(0.2, 0.1)


def test_disabled_passthrough():
    engine = _engine(adaptive_gentle=False)
    _tick(engine, 0)
    assert engine._current_fair_factor() == 0.90


def test_no_fn_passthrough():
    engine = _engine()
    engine.corridor_ttc_fn = None
    engine.update_stream_delay_rates(0.2, 0.1)
    assert engine._current_fair_factor() == 0.93  # EMA stays at 0 (smooth)


def test_smooth_corridor_is_gentle():
    engine = _engine()
    _tick(engine, 0, steps=5)  # EMA -> 0 <= lo 0.5
    assert abs(engine._current_fair_factor() - 0.93) < 1e-9


def test_chaotic_corridor_full_strength():
    engine = _engine()
    _tick(engine, 10, steps=20)  # EMA -> ~10 >= hi 2.0
    assert abs(engine._current_fair_factor() - 0.90) < 1e-9


def test_interpolation_midpoint():
    engine = _engine(adaptive_gentle_ema_alpha=1.0)  # EMA = last count
    _tick(engine, 0)  # seed fn, then set exact midpoint count
    engine.corridor_ttc_fn = lambda: 1.25  # midpoint of [0.5, 2.0]
    engine.update_stream_delay_rates(0.2, 0.1)
    mid = 0.93 + 0.5 * (0.90 - 0.93)
    assert abs(engine._current_fair_factor() - mid) < 1e-9


def test_ema_converges_not_jumps():
    engine = _engine(adaptive_gentle_ema_alpha=0.5)
    _tick(engine, 0, steps=5)
    engine.corridor_ttc_fn = lambda: 4
    engine.update_stream_delay_rates(0.2, 0.1)
    # after one tick at alpha 0.5: ema = 0.5*0 + 0.5*4 = 2.0 -> boundary hi
    assert abs(engine._corridor_ttc_ema - 2.0) < 1e-9
    assert abs(engine._current_fair_factor() - 0.90) < 1e-9


def test_gentle_factor_clamped_to_unit_interval():
    # Factor is clamped into [0.3, 1.0] (both firmer and gentler than the
    # base factor are legal — direction is an empirical question).
    engine = _engine(adaptive_gentle_factor=0.10)
    assert engine.adaptive_gentle_factor == 0.3
    engine = _engine(adaptive_gentle_factor=1.10)
    assert engine.adaptive_gentle_factor == 1.0


def test_reset_clears_ema():
    engine = _engine()
    _tick(engine, 10, steps=20)
    assert engine._corridor_ttc_ema > 2.0
    engine.reset()
    assert engine._corridor_ttc_ema == 0.0
    assert abs(engine._current_fair_factor() - 0.93) < 1e-9


def test_fn_exception_safe():
    engine = _engine()

    def _boom():
        raise RuntimeError("traci down")

    engine.corridor_ttc_fn = _boom
    engine.update_stream_delay_rates(0.2, 0.1)  # must not raise
    assert engine._corridor_ttc_ema == 0.0
