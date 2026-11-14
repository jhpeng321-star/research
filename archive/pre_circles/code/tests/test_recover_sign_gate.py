"""Tests for the recover_speed sign gate.

Recover is a corrective tool for a MAIN-disadvantaged corridor: it should
only be eligible when the stream gap is positive beyond tau. Firing it on a
fair or ramp-disadvantaged corridor widens the gap (structural-harm
mechanism found in the v10c autopsy).
"""

from unittest.mock import patch

from tests.test_shepherd_actions import _make_engine


def _eligible(engine, gap):
    """_recover_eligible with lane/speed queries stubbed to 'all clear'."""
    engine.ema_main_delay = gap  # ramp at 0.0 keeps subtraction exact
    engine.ema_ramp_delay = 0.0 if gap is not None else None
    with (
        patch("env.actions._lane_role", return_value=(engine.pre_edge, 0, None)),
        patch("env.actions._edge_mean_speed", return_value=25.0),
    ):
        return engine._recover_eligible("cav0", current_speed=5.0)


def test_recover_eligible_when_main_disadvantaged():
    engine = _make_engine()
    assert _eligible(engine, 0.10) is True  # gap > tau 0.05
    assert _eligible(engine, 0.06) is True


def test_recover_ineligible_when_corridor_fair():
    engine = _make_engine()
    assert _eligible(engine, 0.05) is False  # at tau -> not eligible
    assert _eligible(engine, 0.02) is False
    assert _eligible(engine, 0.0) is False


def test_recover_ineligible_when_ramp_disadvantaged():
    engine = _make_engine()
    assert _eligible(engine, -0.10) is False  # main advantaged: recover unfair-making


def test_recover_unchanged_when_gap_unknown():
    engine = _make_engine()
    assert _eligible(engine, None) is True  # EMAs unset: legacy behavior


def test_gate_off_restores_legacy():
    engine = _make_engine(mode_config={"recover_gap_sign_gate": False})
    assert _eligible(engine, -0.10) is True


def test_gap_tau_configurable():
    engine = _make_engine(mode_config={"recover_gap_tau": 0.20})
    assert _eligible(engine, 0.10) is False  # below custom tau
    assert _eligible(engine, 0.25) is True
