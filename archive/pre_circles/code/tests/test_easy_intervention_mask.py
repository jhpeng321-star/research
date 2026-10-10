"""Tests for mask-v1: easy-intervention (soft) action masking.

When the corridor is already REALLY fair in absolute terms
(|gap| <= easy_intervention_mask_gap_abs), discretionary flow-hurting modes
are discounted in the eligibility mask. ``easy_intervention_mask_discount``
interpolates: 0.0 = hard ban (bit 0 + apply() remaps to coast),
(0,1) = soft logit penalty (bit = discount, apply() does NOT remap),
1.0 = no effect. gap_abs <= 0 falls back to the no_op-relative condition
(|gap| <= gap_ref). protect is exempt (safety); coast is never gated.
"""

from unittest.mock import MagicMock, patch

from env.actions import MODE_NAMES, ShepherdActionEngine
from tests.test_shepherd_actions import _make_engine, _mock_traci

EASY_MODES = ["fair_balance", "recover", "vacate_drop"]
IDX = {name: MODE_NAMES.index(name) for name in MODE_NAMES}


def _engine(discount=0.1, gap_abs=0.05, **mode_overrides) -> ShepherdActionEngine:
    mc = {
        "easy_intervention_mask": True,
        "easy_intervention_mask_modes": EASY_MODES,
        "easy_intervention_mask_discount": discount,
        "easy_intervention_mask_gap_abs": gap_abs,
    }
    mc.update(mode_overrides)
    return _make_engine(mode_config=mc)


def _set_easy_corridor(engine: ShepherdActionEngine) -> None:
    """|gap| = 0.03 <= gap_abs 0.05 -> really fair in absolute terms."""
    engine.set_stream_delay_gap_ref(0.10)
    engine.ema_main_delay = 0.18
    engine.ema_ramp_delay = 0.15


def _set_hard_corridor(engine: ShepherdActionEngine) -> None:
    """|gap| = 0.35 > gap_abs 0.05 -> genuinely unfair, intervention free."""
    engine.set_stream_delay_gap_ref(0.10)
    engine.ema_main_delay = 0.50
    engine.ema_ramp_delay = 0.15


def _force_all_eligible(engine: ShepherdActionEngine) -> None:
    engine._fair_balance_eligible = lambda cav, spd: True
    engine._recover_eligible = lambda cav, spd: True
    engine._vacate_eligible = lambda cav, spd: True
    engine._protect_eligible = lambda cav, spd: True


def test_blocked_helper_conditions():
    engine = _engine()
    # gap unknown (EMAs unset) -> not blocked
    engine.set_stream_delay_gap_ref(0.10)
    assert engine.easy_intervention_blocked() is False
    _set_easy_corridor(engine)
    assert engine.easy_intervention_blocked() is True
    _set_hard_corridor(engine)
    assert engine.easy_intervention_blocked() is False
    # absolute rule ignores the no_op ref entirely
    engine.set_stream_delay_gap_ref(0.0)
    assert engine.easy_intervention_blocked() is False
    _set_easy_corridor(engine)
    engine.set_stream_delay_gap_ref(0.0)
    assert engine.easy_intervention_blocked() is True


def test_absolute_gate_not_noop_relative():
    """User steer: mask only when REALLY fair, not when merely fair vs no_op.

    |gap| = 0.10 <= ref 0.16 (better than no_op) but > gap_abs 0.05
    (not really fair) -> NOT blocked.
    """
    engine = _engine(gap_abs=0.05)
    engine.set_stream_delay_gap_ref(0.16)
    engine.ema_main_delay = 0.25
    engine.ema_ramp_delay = 0.15  # |gap| = 0.10
    assert engine.easy_intervention_blocked() is False
    _force_all_eligible(engine)
    mock_traci = _mock_traci(speed=20.0)
    with patch("env.actions.traci", mock_traci):
        mask = engine.eligibility_mask("cav0")
    assert all(float(mask[i]) == 1.0 for i in range(len(mask)))


def test_gap_abs_zero_falls_back_to_ref():
    """gap_abs <= 0 keeps the legacy no_op-relative condition."""
    engine = _engine(gap_abs=0.0)
    engine.set_stream_delay_gap_ref(0.10)
    engine.ema_main_delay = 0.20
    engine.ema_ramp_delay = 0.15  # |gap| = 0.05 <= ref 0.10
    assert engine.easy_intervention_blocked() is True
    engine.ema_main_delay = 0.50  # |gap| = 0.35 > ref
    assert engine.easy_intervention_blocked() is False


def test_soft_mask_discounts_bits_but_does_not_remap():
    engine = _engine(discount=0.1)
    _set_easy_corridor(engine)
    _force_all_eligible(engine)
    mock_traci = _mock_traci(speed=20.0)
    with patch("env.actions.traci", mock_traci):
        mask = engine.eligibility_mask("cav0")
    assert mask[IDX["coast"]] == 1.0
    assert mask[IDX["guard"]] == 1.0  # safety exempt
    for name in EASY_MODES:
        assert abs(float(mask[IDX[name]]) - 0.1) < 1e-6, name

    # apply() must NOT remap in soft mode: recover_speed actually runs.
    engine._recover_speed = MagicMock(return_value=False)
    mock_traci = _mock_traci(speed=20.0)
    with patch("env.actions.traci", mock_traci):
        engine.apply("cav0", IDX["recover"])
    engine._recover_speed.assert_called_once()
    # coast remap would have issued setSpeed(-1); it must not have.
    for call in mock_traci.vehicle.setSpeed.call_args_list:
        assert call.args[1] != -1


def test_hard_mask_zeroes_bits_and_remaps_to_coast():
    engine = _engine(discount=0.0)
    _set_easy_corridor(engine)
    _force_all_eligible(engine)
    mock_traci = _mock_traci(speed=20.0)
    with patch("env.actions.traci", mock_traci):
        mask = engine.eligibility_mask("cav0")
    assert mask[IDX["coast"]] == 1.0
    assert mask[IDX["guard"]] == 1.0
    for name in EASY_MODES:
        assert mask[IDX[name]] == 0.0, name

    mock_traci = _mock_traci(speed=20.0)
    with patch("env.actions.traci", mock_traci):
        masked, _ = engine.apply("cav0", IDX["recover"])
    assert masked is True
    mock_traci.vehicle.setSpeed.assert_called_with("cav0", -1)


def test_no_gating_when_corridor_unfair():
    engine = _engine(discount=0.1)
    _set_hard_corridor(engine)
    _force_all_eligible(engine)
    mock_traci = _mock_traci(speed=20.0)
    with patch("env.actions.traci", mock_traci):
        mask = engine.eligibility_mask("cav0")
    assert all(float(mask[i]) == 1.0 for i in range(len(mask)))


def test_no_gating_when_gap_unknown():
    engine = _engine(discount=0.1)
    engine.set_stream_delay_gap_ref(0.10)
    assert engine.stream_delay_gap() is None
    _force_all_eligible(engine)
    mock_traci = _mock_traci(speed=20.0)
    with patch("env.actions.traci", mock_traci):
        mask = engine.eligibility_mask("cav0")
    assert all(float(mask[i]) == 1.0 for i in range(len(mask)))


def test_flag_off_restores_old_behavior():
    engine = _engine(discount=0.0, easy_intervention_mask=False)
    _set_easy_corridor(engine)
    _force_all_eligible(engine)
    mock_traci = _mock_traci(speed=20.0)
    with patch("env.actions.traci", mock_traci):
        mask = engine.eligibility_mask("cav0")
    assert all(float(mask[i]) == 1.0 for i in range(len(mask)))


def test_discount_clamped_to_unit_interval():
    engine = _engine(discount=1.7)
    assert engine.easy_intervention_mask_discount == 1.0
    engine = _engine(discount=-0.5)
    assert engine.easy_intervention_mask_discount == 0.0


def _grad_engine(discount=0.1, gap_abs=0.05, gap_outer=0.10) -> ShepherdActionEngine:
    return _engine(
        discount=discount,
        gap_abs=gap_abs,
        easy_intervention_mask_graduated=True,
        easy_intervention_mask_gap_abs=gap_abs,
        easy_intervention_mask_gap_outer=gap_outer,
    )


def _set_gap(engine: ShepherdActionEngine, g: float) -> None:
    engine.ema_main_delay = 0.15 + g
    engine.ema_ramp_delay = 0.15


def test_graduated_discount_curve():
    engine = _grad_engine(discount=0.1)
    _set_gap(engine, 0.03)  # below inner -> full strength
    assert abs(engine.easy_intervention_discount() - 0.1) < 1e-6
    _set_gap(engine, 0.05)  # at inner -> full strength
    assert abs(engine.easy_intervention_discount() - 0.1) < 1e-6
    _set_gap(engine, 0.075)  # halfway -> ~0.55
    assert abs(engine.easy_intervention_discount() - 0.55) < 1e-6
    _set_gap(engine, 0.10)  # at outer -> ungated
    assert engine.easy_intervention_discount() is None
    _set_gap(engine, 0.20)  # beyond outer -> ungated
    assert engine.easy_intervention_discount() is None
    # unknown gap -> ungated
    engine.ema_main_delay = None
    engine.ema_ramp_delay = None
    assert engine.easy_intervention_discount() is None


def test_graduated_mask_bits_track_fairness():
    engine = _grad_engine(discount=0.1)
    _set_gap(engine, 0.075)
    _force_all_eligible(engine)
    mock_traci = _mock_traci(speed=20.0)
    with patch("env.actions.traci", mock_traci):
        mask = engine.eligibility_mask("cav0")
    assert mask[IDX["coast"]] == 1.0
    assert mask[IDX["guard"]] == 1.0  # safety exempt
    for name in EASY_MODES:
        assert abs(float(mask[IDX[name]]) - 0.55) < 1e-6, name


def test_graduated_never_remaps_when_discount_positive():
    engine = _grad_engine(discount=0.1)
    _set_gap(engine, 0.03)  # strongest penalty, still soft
    engine._recover_speed = MagicMock(return_value=False)
    mock_traci = _mock_traci(speed=20.0)
    with patch("env.actions.traci", mock_traci):
        engine.apply("cav0", IDX["recover"])
    engine._recover_speed.assert_called_once()
    for call in mock_traci.vehicle.setSpeed.call_args_list:
        assert call.args[1] != -1


def test_graduated_hard_discount_still_bans_at_inner():
    engine = _grad_engine(discount=0.0)
    _set_gap(engine, 0.03)
    _force_all_eligible(engine)
    mock_traci = _mock_traci(speed=20.0)
    with patch("env.actions.traci", mock_traci):
        mask = engine.eligibility_mask("cav0")
    for name in EASY_MODES:
        assert mask[IDX[name]] == 0.0, name
    with patch("env.actions.traci", _mock_traci(speed=20.0)) as mt:
        masked, _ = engine.apply("cav0", IDX["recover"])
    assert masked is True
    mt.vehicle.setSpeed.assert_called_with("cav0", -1)
    # between inner and outer with d=0 -> interpolated (soft, not a ban)
    _set_gap(engine, 0.075)
    assert abs(engine.easy_intervention_discount() - 0.5) < 1e-6


def test_graduated_off_keeps_step_behavior():
    engine = _engine(discount=0.1)  # graduated defaults off
    _set_easy_corridor(engine)
    assert abs(engine.easy_intervention_discount() - 0.1) < 1e-6
    _set_hard_corridor(engine)
    assert engine.easy_intervention_discount() is None
