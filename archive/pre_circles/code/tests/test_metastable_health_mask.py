"""Tests for the Experiment C metastable-health graduated action-space mask.

The RSU computes merge-zone health h in [0,1] (Krauss-Wagner-Gawron 1997
§V hysteresis + §VI pre-breakdown variance) and the engine turns it into a
graduated soft mask on the discretionary modes: healthy merge -> strong coast
pressure (low discount); sick merge -> discount rises to 1.0 (ungated), so the
policy can meter/recover to restore metastable.
"""

from __future__ import annotations

from unittest.mock import patch


from env.actions import MODE_NAMES
from tests.test_shepherd_actions import _make_engine, _mock_traci

FAIR = MODE_NAMES.index("fair_balance")
RECOVER = MODE_NAMES.index("recover")
PROTECT = MODE_NAMES.index("guard")
COAST = MODE_NAMES.index("coast")


def _health_engine(h: float):
    eng = _make_engine(
        mode_config={
            "metastable_health_mask": True,
            "metastable_health_mask_modes": ["fair_balance", "recover"],
            "metastable_health_mask_discount": 0.05,
            "metastable_health_mask_h_lo": 0.2,
            "metastable_health_mask_h_hi": 0.7,
            "mode_count": 5,
        }
    )
    eng.met_health_fn = lambda: h
    return eng


def test_health_discount_sick_merge_is_open():
    # h = 0 (collapse imminent): ungated (None) — metering/recover fully legal
    # so the CAV can meter to restore metastable.
    eng = _health_engine(0.0)
    assert eng.met_health_discount() is None


def test_health_discount_healthy_merge_strong_penalty():
    # h = 1 (deep metastable): discount at the configured floor (0.05) — strong
    # logit penalty pushing the policy to coast.
    eng = _health_engine(1.0)
    assert abs(eng.met_health_discount() - 0.05) < 1e-9


def test_health_discount_linear_fade_between():
    # h midpoint of [0.2, 0.7] = 0.45 -> halfway between 1.0 (open) and 0.05.
    eng = _health_engine(0.45)
    mid = 1.0 - 0.95 * (0.45 - 0.2) / 0.5
    assert abs(eng.met_health_discount() - mid) < 1e-9
    # h at the lo threshold -> open (None).
    eng2 = _health_engine(0.2)
    assert eng2.met_health_discount() is None


def test_health_mask_disabled_returns_none():
    eng = _make_engine(mode_config={"metastable_health_mask": False, "mode_count": 5})
    eng.met_health_fn = lambda: 0.0
    assert eng.met_health_discount() is None


def test_health_mask_missing_fn_returns_none():
    eng = _make_engine(
        mode_config={
            "metastable_health_mask": True,
            "metastable_health_mask_modes": ["fair_balance"],
            "mode_count": 5,
        }
    )
    assert eng.met_health_discount() is None


def test_eligibility_mask_healthy_merge_discounts_fair_but_not_protect():
    eng = _health_engine(1.0)
    with (
        patch.object(eng, "_fair_balance_eligible", return_value=True),
        patch.object(eng, "_recover_eligible", return_value=True),
        patch.object(eng, "_protect_eligible", return_value=True),
        patch.object(eng, "_current_speed", return_value=20.0),
        patch.object(eng, "coast_force_armed", return_value=False),
    ):
        m = eng.eligibility_mask("cav0")
    # coast always legal; fair_balance/recover discounted; protect full.
    assert m[COAST] == 1.0
    assert abs(m[FAIR] - 0.05) < 1e-9
    assert abs(m[RECOVER] - 0.05) < 1e-9
    assert m[PROTECT] == 1.0


def test_eligibility_mask_sick_merge_fully_open():
    eng = _health_engine(0.0)
    with (
        patch.object(eng, "_fair_balance_eligible", return_value=True),
        patch.object(eng, "_recover_eligible", return_value=True),
        patch.object(eng, "_protect_eligible", return_value=True),
        patch.object(eng, "_current_speed", return_value=20.0),
        patch.object(eng, "coast_force_armed", return_value=False),
    ):
        m = eng.eligibility_mask("cav0")
    assert m[FAIR] == 1.0
    assert m[RECOVER] == 1.0


def test_apply_healthy_merge_soft_not_hard():
    # discount > 0 -> soft: the mode stays physically allowed (no coast remap),
    # the penalty lives in the logits. Only a hard 0.0 remaps to coast.
    eng = _health_engine(1.0)
    mock_traci = _mock_traci(speed=20.0)
    with (
        patch("env.actions.traci", mock_traci),
        patch.object(eng, "_fair_balance_eligible", return_value=True),
        patch.object(eng, "_cmd_speed") as cmd,
        patch.object(eng, "_apply_coast") as coast,
    ):
        fired, ineligible = eng.apply("cav0", FAIR, mag=1.0)
    assert fired is False and ineligible is False
    cmd.assert_called_once()
    coast.assert_not_called()


def test_apply_hard_health_zero_remaps_to_coast():
    # Healthy merge (h=1.0) with discount forced to 0.0 -> hard ban: the mode
    # is remapped to coast and mask_fired=True.
    eng = _health_engine(1.0)
    eng.met_health_mask_discount = 0.0  # force hard-ban semantics
    eng.met_health_mask_h_lo = 0.2
    eng.met_health_mask_h_hi = 0.7
    mock_traci = _mock_traci(speed=20.0)
    with (
        patch("env.actions.traci", mock_traci),
        patch.object(eng, "_fair_balance_eligible", return_value=True),
        patch.object(eng, "_apply_coast") as coast,
    ):
        fired, _ = eng.apply("cav0", FAIR, mag=1.0)
    assert fired is True
    coast.assert_called_once()
