"""Tests for the recover_speed TTC gate.

Recover is load-bearing on chaotic corridors but injects TTC events into
smooth flow (the policy adds corridor TTC exactly where the no_op was
smooth). With the gate on, recover is eligible only when the trailing
corridor-TTC EMA is at or above ``recover_ttc_lo``.
"""

from unittest.mock import patch

from tests.test_shepherd_actions import _make_engine


def _engine(**mode_overrides):
    mc = {"recover_ttc_gate": True, "recover_ttc_lo": 0.5}
    mc.update(mode_overrides)
    return _make_engine(mode_config=mc)


def _eligible(engine, gap=0.10):
    """_recover_eligible with lane/speed queries stubbed to 'all clear'."""
    engine.ema_main_delay = gap  # sign gate passes: gap > tau 0.05
    engine.ema_ramp_delay = 0.0
    with (
        patch("env.actions._lane_role", return_value=(engine.pre_edge, 0, None)),
        patch("env.actions._edge_mean_speed", return_value=25.0),
    ):
        return engine._recover_eligible("cav0", current_speed=5.0)


def test_smooth_corridor_bans_recover():
    engine = _engine()
    engine._corridor_ttc_ema = 0.0
    assert _eligible(engine) is False
    engine._corridor_ttc_ema = 0.49
    assert _eligible(engine) is False


def test_chaotic_corridor_keeps_recover():
    engine = _engine()
    engine._corridor_ttc_ema = 0.5  # at threshold -> eligible
    assert _eligible(engine) is True
    engine._corridor_ttc_ema = 2.3
    assert _eligible(engine) is True


def test_gate_off_restores_legacy():
    engine = _engine(recover_ttc_gate=False)
    engine._corridor_ttc_ema = 0.0
    assert _eligible(engine) is True


def test_threshold_configurable():
    engine = _engine(recover_ttc_lo=1.5)
    engine._corridor_ttc_ema = 1.0
    assert _eligible(engine) is False
    engine._corridor_ttc_ema = 1.5
    assert _eligible(engine) is True


def test_ema_ticks_with_gate_alone():
    # The corridor-TTC EMA must tick when ONLY the gate is on (no adaptive).
    engine = _engine()
    assert engine.adaptive_gentle is False
    engine.corridor_ttc_fn = lambda: 4
    engine.update_stream_delay_rates(0.2, 0.1)
    assert engine._corridor_ttc_ema > 0.0


def test_sign_gate_still_applies():
    # TTC gate passes but sign gate blocks on fair corridor.
    engine = _engine()
    engine._corridor_ttc_ema = 3.0
    assert _eligible(engine, gap=0.0) is False
