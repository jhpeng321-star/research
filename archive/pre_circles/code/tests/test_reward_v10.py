"""tests/test_reward_v10.py — v10 learned-abstention reward terms."""

from env.rewards import (
    _easy_intervention_tax,
    _fair_win,
    compute_shepherd_reward,
)


def test_fair_win_pays_only_strictly_below_ref():
    assert _fair_win(None) == 0.0  # unknown gap
    assert _fair_win(0.10, gap_scale=0.40, gap_ref=0.10) == 0.0  # tie: no pay
    assert _fair_win(-0.10, gap_scale=0.40, gap_ref=0.10) == 0.0  # sign-insensitive
    assert _fair_win(0.30, gap_scale=0.40, gap_ref=0.10) == 0.0  # above ref
    assert abs(_fair_win(0.0, gap_scale=0.40, gap_ref=0.20) - 0.5) < 1e-9
    assert _fair_win(0.0, gap_scale=0.40, gap_ref=0.99) == 1.0  # clipped


def test_easy_intervention_tax():
    assert _easy_intervention_tax(0, 0.01, 0.10) == 0.0  # coast never taxed
    assert _easy_intervention_tax(1, None, 0.10) == 0.0  # unknown gap: no tax
    assert _easy_intervention_tax(1, 0.05, 0.10) == -1.0  # easy + act: taxed
    assert _easy_intervention_tax(4, -0.05, 0.10) == -1.0  # |gap| used
    assert _easy_intervention_tax(1, 0.50, 0.10) == 0.0  # hard state: free
    assert _easy_intervention_tax(1, 0.05, 0.0) == 0.0  # ref 0 -> off


def _v9_zero_weights():
    # Every key compute_shepherd_reward reads, zeroed, plus the v10 keys.
    return {
        "efficiency": 0.0,
        "safety": 0.0,
        "stability": 0.0,
        "fairness": 0.0,
        "delay_budget": 0.0,
        "delay_win": 0.0,
        "control_cost": 0.0,
        "mask_penalty": 0.0,
        "ineligible_penalty": 0.0,
        "mode_use_bonus": 0.0,
        "flow": 0.0,
        "tail": 0.0,
        "fair_win": 0.0,
        "easy_intervention_tax": 0.0,
    }


def test_compute_shepherd_reward_v10_terms_additive():
    # All v9 weights zeroed; helpers that would touch traci return 0 without a
    # connection (their exceptions are caught internally), so the total is
    # exactly the two v10 terms.
    w = _v9_zero_weights()
    w["fair_win"] = 0.20
    w["easy_intervention_tax"] = 0.15
    r = compute_shepherd_reward(
        "cav1",
        1,
        35.0,
        w,
        200.0,
        cav_ids={"cav1"},
        stream_delay_gap=0.05,
        stream_fairness_gap_scale=0.40,
        stream_delay_gap_ref=0.10,
    )
    # fair_win = (0.10 - 0.05)/0.40 = 0.125 -> 0.20 * 0.125 = 0.025
    # tax = -1 -> 0.15 * -1 = -0.15 ; total = -0.125
    assert abs(r - (-0.125)) < 1e-9


def test_v10_terms_default_off():
    # Zeroed v9 weights, v10 keys absent -> reward exactly 0 (backward compat).
    w = _v9_zero_weights()
    del w["fair_win"]
    del w["easy_intervention_tax"]
    r = compute_shepherd_reward(
        "cav1",
        1,
        35.0,
        w,
        200.0,
        cav_ids={"cav1"},
        stream_delay_gap=0.05,
        stream_fairness_gap_scale=0.40,
        stream_delay_gap_ref=0.10,
    )
    assert r == 0.0
