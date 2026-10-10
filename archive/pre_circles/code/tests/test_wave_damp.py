"""Tests for wave_damp (research_v2 flow_stabilization port, opt-in 6th mode).

The CAV matches the EMA of ahead-band lane speeds with smooth rate-limited
commands — absorbing compression waves instead of amplifying them. Opt-in
only: default runs keep 5 modes (champion compatibility); mode_count=6
opts in. wave_damp is always eligible and exempt from the easy tax.
"""

from unittest.mock import patch


from env.rewards import compute_shepherd_reward
from tests.test_shepherd_actions import _make_engine, _mock_traci


def _engine(**overrides):
    return _make_engine(mode_config=overrides)


def test_ema_smoothing():
    engine = _engine(flow_stab_ema_alpha=0.3)
    engine.ema_state["cav0"] = 20.0
    with (
        patch.object(engine, "_ahead_band_mean_speed", return_value=10.0),
        patch("env.actions.traci", _mock_traci(speed=20.0)),
        patch.object(engine, "_set_speed_safe", create=True)
        if False
        else patch("env.actions._set_speed_safe"),
    ):
        engine._wave_damp("cav0", 20.0)
    # v_smooth = 0.3*10 + 0.7*20 = 17.0 (band 10 < 20 - 0.5 -> preview -0.5 -> 9.5:
    # v_smooth = 0.3*9.5 + 0.7*20 = 16.85)
    assert abs(engine.ema_state["cav0"] - 16.85) < 1e-6


def test_preview_anticipation():
    engine = _engine(flow_stab_deadband=0.5, flow_stab_preview_gain=0.5, flow_stab_ema_alpha=1.0)
    with (
        patch.object(engine, "_ahead_band_mean_speed", return_value=10.0),
        patch("env.actions.traci", _mock_traci(speed=20.0)),
        patch("env.actions._set_speed_safe"),
    ):
        engine._wave_damp("cav0", 20.0)
    # band slower than ego by > deadband -> v_down = 10 - 0.5 preview = 9.5
    assert abs(engine.ema_state["cav0"] - 9.5) < 1e-6


def test_no_preview_when_band_faster():
    engine = _engine(flow_stab_deadband=0.5, flow_stab_preview_gain=0.5, flow_stab_ema_alpha=1.0)
    with (
        patch.object(engine, "_ahead_band_mean_speed", return_value=25.0),
        patch("env.actions.traci", _mock_traci(speed=20.0)),
        patch("env.actions._set_speed_safe"),
    ):
        engine._wave_damp("cav0", 20.0)
    assert abs(engine.ema_state["cav0"] - 25.0) < 1e-6


def test_no_info_holds_speed():
    engine = _engine(flow_stab_ema_alpha=1.0)
    with (
        patch.object(engine, "_ahead_band_mean_speed", return_value=None),
        patch("env.actions.traci", _mock_traci(speed=20.0)),
        patch("env.actions._set_speed_safe"),
    ):
        engine._wave_damp("cav0", 20.0)
    assert abs(engine.ema_state["cav0"] - 20.0) < 1e-6


def test_default_mask_is_5_bits_champion_compat():
    engine = _engine()
    mock = _mock_traci(speed=20.0)
    with patch("env.actions.traci", mock):
        mask = engine.eligibility_mask("cav0")
    assert mask.shape == (5,)


def test_optin_mask_is_6_bits_with_wave_damp_legal():
    engine = _engine(mode_count=6)
    mock = _mock_traci(speed=20.0)
    with patch("env.actions.traci", mock):
        mask = engine.eligibility_mask("cav0")
    assert mask.shape == (6,)
    assert mask[5] == 1.0  # wave_damp always legal


def test_easy_tax_exempt():
    weights = {
        "efficiency": 0.0,
        "safety": 0.0,
        "stability": 0.0,
        "fairness": 0.0,
        "control_cost": 0.0,
        "easy_intervention_tax": 1.0,
    }
    # wave_damp (5) on an easy corridor: no tax
    r = compute_shepherd_reward(
        "cav0",
        5,
        33.0,
        weights,
        120.0,
        cav_ids={"cav0"},
        stream_delay_gap=0.0,
        stream_delay_gap_ref=0.0,
    )
    assert r == 0.0
    # fair_balance (1) on the same corridor: pays -1
    r = compute_shepherd_reward(
        "cav0",
        1,
        33.0,
        weights,
        120.0,
        cav_ids={"cav0"},
        stream_delay_gap=0.0,
        stream_delay_gap_ref=0.0,
    )
    assert r == -1.0


def test_wave_damp_is_cost_free():
    # The do-no-harm mode pays no control cost (same as coast); without this
    # the policy rationally never picks it (6-mode run showed 0.0 usage).
    weights = {
        "efficiency": 0.0,
        "safety": 0.0,
        "stability": 0.0,
        "fairness": 0.0,
        "control_cost": 1.0,
    }
    r_damp = compute_shepherd_reward("cav0", 5, 33.0, weights, 120.0, cav_ids={"cav0"})
    r_coast = compute_shepherd_reward("cav0", 0, 33.0, weights, 120.0, cav_ids={"cav0"})
    r_fair = compute_shepherd_reward("cav0", 1, 33.0, weights, 120.0, cav_ids={"cav0"})
    assert r_damp == r_coast == 0.0
    assert r_fair == -1.0
