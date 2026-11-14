"""Tests for Shepherd / Flat-MARL human-centric reward (incl. fairness)."""

from unittest.mock import patch

from env.rewards import (
    MergeOutflowTracker,
    TimeLossTracker,
    _control_cost,
    _efficiency,
    _fairness,
    _percentile,
    _safety,
    compute_shepherd_reward,
)


def test_control_cost_zero_only_for_mode_zero():
    assert _control_cost(0) == 0.0
    assert _control_cost(1) == -1.0
    assert _control_cost(4) == -1.0


def test_control_cost_need_gated_scales_with_health():
    # Coast still free.
    assert _control_cost(0, health=1.0, need_gated=True, c_min=0.25) == 0.0
    # Calm traffic (health=1): full -1.
    assert abs(_control_cost(2, health=1.0, need_gated=True, c_min=0.25) - (-1.0)) < 1e-6
    # Fully jammed (health=0): only -c_min.
    assert abs(_control_cost(2, health=0.0, need_gated=True, c_min=0.25) - (-0.25)) < 1e-6
    # Mid health=0.5, c_min=0.25 → -(0.25 + 0.75*0.5) = -0.625.
    assert abs(_control_cost(1, health=0.5, need_gated=True, c_min=0.25) - (-0.625)) < 1e-6
    # need_gated but no health → flat -1 (safe fallback).
    assert _control_cost(3, health=None, need_gated=True, c_min=0.25) == -1.0


def test_control_health_clips_upstream_s_norm():
    from env.rewards import _control_health

    class _T:
        def s_norm(self, v_ref):
            return 1.4  # above parity → health 1.0

        def sys_norm(self, q_ref, v_ref):
            return 0.2

    assert _control_health(_T(), source="upstream") == 1.0
    assert abs(_control_health(_T(), source="sys") - 0.2) < 1e-6
    assert _control_health(None) is None


class _FakeFlowTracker:
    """MergeOutflowTracker stand-in returning a fixed sys_norm."""

    def __init__(self, sys_norm):
        self._sys_norm = sys_norm

    def sys_norm(self, q_ref, v_ref):
        return self._sys_norm


def _run_efficiency(sys_norm, w_flow=1.0):
    """_efficiency is F_flow only (ego speed removed)."""
    tracker = _FakeFlowTracker(sys_norm) if sys_norm is not None else None
    return _efficiency("cav1", 35.0, flow_tracker=tracker, w_flow=w_flow)


def test_efficiency_positive_slope_above_parity():
    # sys_norm 1.3 -> F_flow = +0.3
    r = _run_efficiency(1.3)
    assert abs(r - 0.3) < 1e-6


def test_efficiency_f_flow_clips_at_plus_one():
    # sys_norm 2.5 -> clip(2.5, 0, 2) - 1 = +1.0
    r = _run_efficiency(2.5)
    assert abs(r - 1.0) < 1e-6


def test_efficiency_negative_f_flow_in_jam():
    # sys_norm 0.4 -> F_flow = -0.6
    r = _run_efficiency(0.4)
    assert abs(r - (-0.6)) < 1e-6


def test_efficiency_neutral_without_tracker():
    # No tracker -> F_flow = 0 (neutral).
    r = _run_efficiency(None)
    assert abs(r - 0.0) < 1e-6


def test_efficiency_mid_range_above_parity():
    # sys_norm 1.5 -> F_flow = +0.5
    r = _run_efficiency(1.5)
    assert abs(r - 0.5) < 1e-6


def test_efficiency_w_flow_scales():
    r = _run_efficiency(1.5, w_flow=0.5)
    assert abs(r - 0.25) < 1e-6


def test_merge_outflow_tracker_clips_at_two():
    """q_norm/s_norm clip at 2.0 (not 1.2) so sys_norm can exceed parity."""
    t = MergeOutflowTracker(edge_id="e", alpha=0.5, speed_edge_id="s")
    t._ema = 3.0
    t._v_ema = 60.0
    assert t.q_norm(1.0) == 2.0  # 3.0 clipped at 2.0, NOT 1.2
    assert t.s_norm(20.0) == 2.0  # 3.0 clipped at 2.0, NOT 1.2
    assert t.q_norm(6.0) == 0.5  # below the cap
    assert t.s_norm(120.0) == 0.5


def test_percentile_basic():
    assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 90.0) == 5.0
    assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 90.0) == 9.0
    assert _percentile([5.0], 90.0) == 5.0
    assert _percentile([], 90.0) == 0.0


def test_fairness_with_time_loss_tracker():
    """Fairness: signed mean HDV-delay improvement vs baseline rate r_ref."""
    tracker = TimeLossTracker()

    # Simulate two HDVs with different delay rates
    tracker._rates = {"hdv1": 0.8, "hdv2": 0.2}
    tracker._prev_time_loss = {"hdv1": 5.0, "hdv2": 2.0}

    with patch("env.rewards._local_vehicles", return_value=["hdv1", "hdv2"]):
        r = _fairness(
            "cav1",
            200.0,
            max_speed=35.0,
            cav_ids={"cav1"},
            time_loss_tracker=tracker,
        )
        # rates = [0.8, 0.2], mean = 0.5 == r_ref -> neutral 0.0
        assert r == 0.0


def test_fairness_includes_zero_rate_hdvs():
    """Zero-delay HDVs count toward the mean (no rate>0 filter)."""
    tracker = TimeLossTracker()
    tracker._rates = {"hdv1": 0.8, "hdv2": 0.0}

    with patch("env.rewards._local_vehicles", return_value=["hdv1", "hdv2"]):
        r = _fairness(
            "cav1",
            200.0,
            max_speed=35.0,
            cav_ids={"cav1"},
            time_loss_tracker=tracker,
        )
        # mean = 0.4 < r_ref = 0.5 -> (0.5 - 0.4)/0.5 = +0.2
        assert abs(r - 0.2) < 1e-6


def test_fairness_zero_when_no_hdvs():
    tracker = TimeLossTracker()
    with patch("env.rewards._local_vehicles", return_value=["cav2"]):
        r = _fairness(
            "cav1",
            200.0,
            max_speed=35.0,
            cav_ids={"cav1", "cav2"},
            time_loss_tracker=tracker,
        )
        assert r == 0.0


def test_fairness_zero_without_tracker():
    """Without a TimeLossTracker, fairness returns 0 (no crash)."""
    with patch("env.rewards._local_vehicles", return_value=["hdv1"]):
        r = _fairness("cav1", 200.0, max_speed=35.0, cav_ids={"cav1"})
        assert r == 0.0


def test_safety_includes_ttc():
    with (
        patch("env.rewards.traci") as mock_traci,
        patch("env.rewards._local_vehicles", return_value=[]),
        patch("env.rewards.leader_ttc", return_value=1.0),
    ):
        mock_traci.vehicle.getAcceleration.return_value = 0.0
        mock_traci.simulation.getCollidingVehiclesIDList.return_value = []
        r = _safety("cav1", 200.0)
        assert r == -1.0  # low TTC only


def test_stream_fairness_relative_to_gap_ref():
    from env.rewards import _stream_fairness

    # |gap|=0.10, ref=0.10 → no excess
    assert abs(_stream_fairness(0.10, gap_scale=0.40, gap_ref=0.10)) < 1e-9
    # |gap|=0.30, ref=0.10, scale=0.40 → excess=0.20 → -0.5
    assert abs(_stream_fairness(0.30, gap_scale=0.40, gap_ref=0.10) - (-0.5)) < 1e-6
    # absolute mode (ref=0) still saturates at |gap|>=scale
    assert abs(_stream_fairness(0.50, gap_scale=0.40, gap_ref=0.0) - (-1.0)) < 1e-6


def test_delay_budget_one_sided():
    from env.rewards import _delay_budget

    # Better than ref → 0
    assert abs(_delay_budget(0.20, 0.30, scale=0.20)) < 1e-9
    # Worse by 0.10 with scale 0.20 → -0.5
    assert abs(_delay_budget(0.40, 0.30, scale=0.20) - (-0.5)) < 1e-6
    # Missing inputs → 0
    assert _delay_budget(None, 0.3) == 0.0
    assert _delay_budget(0.4, None) == 0.0


def test_delay_win_one_sided():
    from env.rewards import _delay_win

    # Worse / equal → 0
    assert abs(_delay_win(0.40, 0.30, scale=0.20)) < 1e-9
    assert abs(_delay_win(0.30, 0.30, scale=0.20)) < 1e-9
    # Better by 0.10 with scale 0.20 → +0.5
    assert abs(_delay_win(0.20, 0.30, scale=0.20) - 0.5) < 1e-6
    # Clip at 1
    assert abs(_delay_win(0.0, 0.50, scale=0.20) - 1.0) < 1e-6
    assert _delay_win(None, 0.3) == 0.0
    assert _delay_win(0.2, None) == 0.0


def test_compute_reward_includes_delay_budget_weight():
    weights = {
        "efficiency": 0.0,
        "safety": 0.0,
        "stability": 0.0,
        "fairness": 0.0,
        "delay_budget": 2.0,
        "control_cost": 0.0,
    }
    with (
        patch("env.rewards._efficiency", return_value=0.0),
        patch("env.rewards._safety", return_value=0.0),
        patch("env.rewards._stability", return_value=0.0),
        patch("env.rewards._fairness", return_value=0.0),
        patch("env.rewards._delay_budget", return_value=-0.5),
        patch("env.rewards._delay_win", return_value=0.0),
        patch("env.rewards._control_cost", return_value=0.0),
    ):
        r = compute_shepherd_reward("cav1", 0, 35.0, weights, 200.0, cav_ids={"cav1"})
        assert abs(r - (-1.0)) < 1e-6


def test_compute_reward_includes_delay_win_weight():
    weights = {
        "efficiency": 0.0,
        "safety": 0.0,
        "stability": 0.0,
        "fairness": 0.0,
        "delay_budget": 0.0,
        "delay_win": 0.5,
        "control_cost": 0.0,
    }
    with (
        patch("env.rewards._efficiency", return_value=0.0),
        patch("env.rewards._safety", return_value=0.0),
        patch("env.rewards._stability", return_value=0.0),
        patch("env.rewards._fairness", return_value=0.0),
        patch("env.rewards._delay_budget", return_value=0.0),
        patch("env.rewards._delay_win", return_value=0.8),
        patch("env.rewards._control_cost", return_value=0.0),
    ):
        r = compute_shepherd_reward("cav1", 0, 35.0, weights, 200.0, cav_ids={"cav1"})
        assert abs(r - 0.4) < 1e-6


def test_compute_reward_ineligible_penalty():
    weights = {
        "efficiency": 0.0,
        "safety": 0.0,
        "stability": 0.0,
        "fairness": 0.0,
        "control_cost": 0.0,
        "ineligible_penalty": 0.2,
    }
    with (
        patch("env.rewards._efficiency", return_value=0.0),
        patch("env.rewards._safety", return_value=0.0),
        patch("env.rewards._stability", return_value=0.0),
        patch("env.rewards._fairness", return_value=0.0),
        patch("env.rewards._control_cost", return_value=0.0),
    ):
        r_ok = compute_shepherd_reward(
            "cav1", 3, 35.0, weights, 200.0, cav_ids={"cav1"}, ineligible=False
        )
        r_bad = compute_shepherd_reward(
            "cav1", 3, 35.0, weights, 200.0, cav_ids={"cav1"}, ineligible=True
        )
        assert abs(r_ok - 0.0) < 1e-6
        assert abs(r_bad - (-0.2)) < 1e-6


def test_compute_reward_mode_use_bonus_non_coast():
    weights = {
        "efficiency": 0.0,
        "safety": 0.0,
        "stability": 0.0,
        "fairness": 0.0,
        "control_cost": 0.0,
        "mode_use_bonus": 0.1,
    }
    with (
        patch("env.rewards._efficiency", return_value=0.0),
        patch("env.rewards._safety", return_value=0.0),
        patch("env.rewards._stability", return_value=0.0),
        patch("env.rewards._fairness", return_value=0.0),
        patch("env.rewards._control_cost", return_value=0.0),
    ):
        # Legacy: mode_use_bonus_modes=None → any eligible non-coast.
        for mode in (1, 2, 3, 4):
            assert (
                abs(
                    compute_shepherd_reward(
                        "cav1",
                        mode,
                        35.0,
                        weights,
                        200.0,
                        cav_ids={"cav1"},
                        ineligible=False,
                        mode_use_bonus_modes=None,
                    )
                    - 0.1
                )
                < 1e-6
            )
        # Targeted: only recover(2) / vacate(3).
        targeted = ["recover", "vacate_drop"]
        assert (
            abs(
                compute_shepherd_reward(
                    "cav1",
                    2,
                    35.0,
                    weights,
                    200.0,
                    cav_ids={"cav1"},
                    ineligible=False,
                    mode_use_bonus_modes=targeted,
                )
                - 0.1
            )
            < 1e-6
        )
        assert (
            abs(
                compute_shepherd_reward(
                    "cav1",
                    1,
                    35.0,
                    weights,
                    200.0,
                    cav_ids={"cav1"},
                    ineligible=False,
                    mode_use_bonus_modes=targeted,
                )
                - 0.0
            )
            < 1e-6
        )
        assert (
            abs(
                compute_shepherd_reward(
                    "cav1",
                    0,
                    35.0,
                    weights,
                    200.0,
                    cav_ids={"cav1"},
                    ineligible=False,
                    mode_use_bonus_modes=targeted,
                )
                - 0.0
            )
            < 1e-6
        )
        assert (
            abs(
                compute_shepherd_reward(
                    "cav1",
                    3,
                    35.0,
                    weights,
                    200.0,
                    cav_ids={"cav1"},
                    ineligible=True,
                    mode_use_bonus_modes=targeted,
                )
                - 0.0
            )
            < 1e-6
        )


def test_mode_use_bonus_does_not_outrank_flow_delta():
    """If recover hurts F_flow by more than the bonus, coast still wins."""
    weights = {
        "efficiency": 1.0,
        "safety": 0.0,
        "stability": 0.0,
        "fairness": 0.0,
        "control_cost": 0.0,
        "flow": 1.0,
        "mode_use_bonus": 0.1,
    }
    with (
        patch("env.rewards._safety", return_value=0.0),
        patch("env.rewards._stability", return_value=0.0),
        patch("env.rewards._fairness", return_value=0.0),
        patch("env.rewards._control_cost", return_value=0.0),
        patch("env.rewards._tail_penalty", return_value=0.0),
    ):
        with patch("env.rewards._efficiency", return_value=0.0):
            r_coast = compute_shepherd_reward(
                "cav1",
                0,
                35.0,
                weights,
                200.0,
                cav_ids={"cav1"},
                ineligible=False,
                mode_use_bonus_modes=None,
            )
        with patch("env.rewards._efficiency", return_value=-0.15):
            r_recover_hurts_flow = compute_shepherd_reward(
                "cav1",
                2,
                35.0,
                weights,
                200.0,
                cav_ids={"cav1"},
                ineligible=False,
                mode_use_bonus_modes=None,
            )
        # -0.15 + 0.1 bonus = -0.05 < coast at 0
        assert r_recover_hurts_flow < r_coast
        assert abs(r_coast - 0.0) < 1e-6
