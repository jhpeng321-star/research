"""Tests for the hybrid mask: hard ban only in the ultra-fair zone.

When |gap| <= easy_intervention_mask_hard_below, easy-gated modes are hard
banned (mask bit 0 + apply() remaps to coast) regardless of step/graduated
mode; above it the normal soft/graduated behavior applies.
"""

from unittest.mock import patch

from tests.test_easy_intervention_mask import (
    EASY_MODES,
    IDX,
    _engine,
    _force_all_eligible,
    _set_gap,
)
from tests.test_shepherd_actions import _mock_traci


def _hybrid(hard_below=0.02, discount=0.1, graduated=True):
    return _engine(
        discount=discount,
        gap_abs=0.05,
        easy_intervention_mask_graduated=graduated,
        easy_intervention_mask_gap_abs=0.05,
        easy_intervention_mask_gap_outer=0.10,
        easy_intervention_mask_hard_below=hard_below,
    )


def test_hard_zone_returns_zero():
    engine = _hybrid(hard_below=0.02)
    _set_gap(engine, 0.01)
    assert engine.easy_intervention_discount() == 0.0
    _set_gap(engine, 0.02)  # at boundary -> hard
    assert engine.easy_intervention_discount() == 0.0


def test_soft_zone_above_hard_below():
    engine = _hybrid(hard_below=0.02)
    _set_gap(engine, 0.03)  # above hard zone, inside graduated inner zone
    d = engine.easy_intervention_discount()
    assert d is not None and abs(d - 0.1) < 1e-6
    _set_gap(engine, 0.075)  # graduated fade region
    d = engine.easy_intervention_discount()
    assert abs(d - 0.55) < 1e-6
    _set_gap(engine, 0.20)  # beyond outer -> ungated
    assert engine.easy_intervention_discount() is None


def test_hard_zone_bans_bits_and_remaps():
    engine = _hybrid(hard_below=0.02)
    _set_gap(engine, 0.01)
    _force_all_eligible(engine)
    mock_traci = _mock_traci(speed=20.0)
    with patch("env.actions.traci", mock_traci):
        mask = engine.eligibility_mask("cav0")
    assert mask[IDX["coast"]] == 1.0
    assert mask[IDX["guard"]] == 1.0
    for name in EASY_MODES:
        assert mask[IDX[name]] == 0.0, name
    with patch("env.actions.traci", _mock_traci(speed=20.0)) as mt:
        masked, _ = engine.apply("cav0", IDX["recover"])
    assert masked is True
    mt.vehicle.setSpeed.assert_called_with("cav0", -1)


def test_hard_below_off_keeps_soft_behavior():
    engine = _hybrid(hard_below=0.0)
    _set_gap(engine, 0.01)
    d = engine.easy_intervention_discount()
    assert d is not None and abs(d - 0.1) < 1e-6


def test_hard_zone_also_applies_in_step_mode():
    engine = _hybrid(hard_below=0.02, graduated=False)
    _set_gap(engine, 0.01)
    assert engine.easy_intervention_discount() == 0.0
