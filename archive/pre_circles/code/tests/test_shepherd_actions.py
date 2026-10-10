"""Tests for Shepherd action helpers: ahead-band, rate-limit, gentle modes."""

from unittest.mock import MagicMock, patch

from env.actions import (
    _rate_limit,
    MODE_NAMES,
    RL_SPEED_MODE,
    ShepherdActionEngine,
)
from env.traci_query import ahead_lane_speeds as _ahead_speeds


def test_rate_limit_deceleration():
    """Braking should be capped at a_max_dec * dt."""
    v_cmd = _rate_limit(20.0, 10.0, dt=1.0, a_max_dec=2.0, a_max_acc=1.0, deadband=0.5)
    # diff = -10, max dec = 2 * 1 = 2, so v_cmd = 20 - 2 = 18
    assert abs(v_cmd - 18.0) < 1e-6


def test_rate_limit_acceleration():
    """Acceleration should be capped at a_max_acc * dt (slower than decel)."""
    v_cmd = _rate_limit(10.0, 20.0, dt=1.0, a_max_dec=2.0, a_max_acc=1.0, deadband=0.5)
    # diff = +10, max acc = 1 * 1 = 1, so v_cmd = 10 + 1 = 11
    assert abs(v_cmd - 11.0) < 1e-6


def test_rate_limit_deadband():
    """Within deadband, speed should not change."""
    v_cmd = _rate_limit(15.0, 15.2, dt=1.0, a_max_dec=2.0, a_max_acc=1.0, deadband=0.5)
    assert abs(v_cmd - 15.0) < 1e-6


def test_rate_limit_full_brake_within_limit():
    """Small speed diff within rate limit should reach target."""
    v_cmd = _rate_limit(10.0, 8.0, dt=1.0, a_max_dec=2.0, a_max_acc=1.0, deadband=0.5)
    # diff = -2, max dec = 2, so v_cmd = max(8, 10-2) = 8
    assert abs(v_cmd - 8.0) < 1e-6


def test_ahead_speeds_with_mock():
    """Ahead-band selection: only vehicles with positive offset on same lane."""
    from unittest.mock import patch, MagicMock

    mock_traci = MagicMock()
    mock_traci.vehicle.getIDList.return_value = ["ego", "ahead1", "ahead2", "behind1", "other_edge"]

    def get_lane(vid):
        return {
            "ego": "lane_0",
            "ahead1": "lane_0",
            "ahead2": "lane_0",
            "behind1": "lane_0",
            "other_edge": "lane_1",
        }[vid]

    def get_lane_pos(vid):
        return {
            "ego": 100.0,
            "ahead1": 260.0,
            "ahead2": 180.0,
            "behind1": 50.0,
            "other_edge": 200.0,
        }[vid]

    def get_road(vid):
        return {
            "ego": "edge1",
            "ahead1": "edge1",
            "ahead2": "edge1",
            "behind1": "edge1",
            "other_edge": "edge2",
        }[vid]

    def get_speed(vid):
        return {"ego": 25.0, "ahead1": 15.0, "ahead2": 20.0, "behind1": 30.0, "other_edge": 10.0}[
            vid
        ]

    mock_traci.vehicle.getLaneID.side_effect = get_lane
    mock_traci.vehicle.getLanePosition.side_effect = get_lane_pos
    mock_traci.vehicle.getRoadID.side_effect = get_road
    mock_traci.vehicle.getSpeed.side_effect = get_speed

    with patch("env.traci_query.traci", mock_traci):
        # Primary band [150, 200]: ahead1 at 160 (260-100), ahead2 at 80 (180-100)
        # ahead1 is in [150, 200] -> primary = [15.0]
        # ahead2 at 80 is in [0, 200] but not [150, 200] -> fallback
        speeds = _ahead_speeds("ego", near=150.0, far=200.0)
        assert 15.0 in speeds  # ahead1 in primary band
        assert len(speeds) == 1  # only ahead1 in primary


def test_ahead_speeds_fallback_band():
    """When primary band is empty, fall back to [0, far]."""
    from unittest.mock import patch, MagicMock

    mock_traci = MagicMock()
    mock_traci.vehicle.getIDList.return_value = ["ego", "close_ahead"]

    mock_traci.vehicle.getLaneID.side_effect = lambda vid: "lane_0"
    mock_traci.vehicle.getLanePosition.side_effect = lambda vid: {
        "ego": 100.0,
        "close_ahead": 120.0,
    }[vid]
    mock_traci.vehicle.getRoadID.side_effect = lambda vid: "edge1"
    mock_traci.vehicle.getSpeed.side_effect = lambda vid: {"ego": 25.0, "close_ahead": 18.0}[vid]

    with patch("env.traci_query.traci", mock_traci):
        # close_ahead at 20m (120-100) - not in [150, 200], but in [0, 200]
        speeds = _ahead_speeds("ego", near=150.0, far=200.0)
        assert 18.0 in speeds  # fallback to [0, 200]


def test_ahead_speeds_empty():
    """No vehicles ahead returns empty list."""
    from unittest.mock import patch, MagicMock

    mock_traci = MagicMock()
    mock_traci.vehicle.getIDList.return_value = ["ego"]
    mock_traci.vehicle.getLaneID.side_effect = lambda vid: "lane_0"
    mock_traci.vehicle.getLanePosition.side_effect = lambda vid: 100.0
    mock_traci.vehicle.getRoadID.side_effect = lambda vid: "edge1"
    mock_traci.vehicle.getSpeed.side_effect = lambda vid: 25.0

    with patch("env.traci_query.traci", mock_traci):
        speeds = _ahead_speeds("ego", near=150.0, far=200.0)
        assert speeds == []


def test_mode_names_gated_menu():
    # The DEFAULT run's mode set is the original 5 (champion compat);
    # wave_damp is registered globally but opt-in only.
    from config.defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["shepherd_modes"] == [
        "coast",
        "fair_balance",
        "recover",
        "vacate_drop",
        "guard",
    ]
    assert MODE_NAMES[:5] == DEFAULT_CONFIG["shepherd_modes"]
    assert MODE_NAMES[5] == "wave_damp"


def test_rl_speed_mode():
    assert RL_SPEED_MODE & 1 == 1
    assert RL_SPEED_MODE & 0b110 == 0b110


def _make_engine(**overrides) -> ShepherdActionEngine:
    cfg = dict(
        base_tau=1.0,
        tau_deltas={
            "coast": 0.0,
            "fair_balance": 0.25,
            "recover": -0.2,
            "vacate_drop": 0.0,
            "guard": 0.5,
        },
        speed_factors={
            "coast": None,
            "fair_balance": 0.90,
            "recover": None,
            "vacate_drop": 0.90,
            "guard": 0.85,
        },
        max_speed=35.0,
        dt=1.0,
        mode_config={
            "max_decel": 1.0,
            "max_accel": 1.25,
            "deadband": 0.5,
            "guard_speed": 8.0,
            "fair_balance_gap_tau": 0.05,
            "fair_balance_tip": 120.0,
            "fair_balance_factor": 0.90,
            "fair_balance_floor": 8.0,
            "recover_merge_clear": 10.0,
            "recover_down_clear": 18.0,
            "recover_ego_max": 12.0,
            "recover_push": 1.5,
            "recover_v_cap": 18.0,
            "protect_ttc": 1.5,
            "protect_factor": 0.85,
            "protect_tau_delta": 0.5,
        },
    )
    if "mode_config" in overrides:
        base_mc = cfg["mode_config"]
        base_mc.update(overrides.pop("mode_config"))
        overrides["mode_config"] = base_mc
    cfg.update(overrides)
    return ShepherdActionEngine(**cfg)


def _mock_traci(speed: float = 20.0, ids=("cav0",)) -> MagicMock:
    mock_traci = MagicMock()
    mock_traci.vehicle.getIDList.return_value = list(ids)
    mock_traci.vehicle.getSpeed.side_effect = lambda vid: speed
    mock_traci.vehicle.getTau.return_value = 1.0
    mock_traci.vehicle.getLaneID.return_value = "11553048#1_0"  # ramp
    mock_traci.vehicle.getLanePosition.return_value = 100.0
    mock_traci.lane.getLength.return_value = 378.0
    mock_traci.vehicle.getRoadID.return_value = "11553048#1"
    mock_traci.edge.getLastStepVehicleIDs.return_value = ["a", "b", "c", "d", "e"]
    return mock_traci


def _set_speed_args(mock_traci):
    return [c.args[1] for c in mock_traci.vehicle.setSpeed.call_args_list]


def test_coast_restores_sumo_control():
    engine = _make_engine()
    mock_traci = _mock_traci(speed=20.0)
    with patch("env.actions.traci", mock_traci):
        engine.apply("cav0", 0)
        mock_traci.vehicle.setSpeed.assert_called_with("cav0", -1)
        mock_traci.vehicle.setTau.assert_called_with("cav0", 1.0)


def test_fair_balance_coasts_without_gap():
    engine = _make_engine()
    mock_traci = _mock_traci(speed=20.0)
    # rem = 378-100 = 278 > tip 120 → also ineligible; set near tip
    mock_traci.lane.getLength.return_value = 150.0
    mock_traci.vehicle.getLanePosition.return_value = 50.0  # rem=100
    with (
        patch("env.actions.traci", mock_traci),
        patch("env.actions.leader_info", return_value=None),
    ):
        engine.apply("cav0", 1)  # fair_balance, no EMA gap
        mock_traci.vehicle.setSpeed.assert_called_with("cav0", -1)


def test_fair_balance_slows_ramp_when_main_more_delayed():
    engine = _make_engine()
    engine.update_stream_delay_rates(0.20, 0.05)  # gap = +0.15 > tau
    mock_traci = _mock_traci(speed=20.0)
    mock_traci.lane.getLength.return_value = 150.0
    mock_traci.vehicle.getLanePosition.return_value = 50.0  # rem=100 < tip
    with (
        patch("env.actions.traci", mock_traci),
        patch("env.actions.leader_info", return_value=None),
    ):
        engine.apply("cav0", 1)
        issued = _set_speed_args(mock_traci)
        assert issued and all(v != -1 for v in issued)
        # 0.90*20=18, rate-limited by max_decel 1.0 → 19.0
        assert abs(issued[-1] - 19.0) < 1e-6


def test_fair_balance_blocked_in_reward_dead_band():
    """|G| above absolute tau but at/below G_ref+tau → coast (Phase 1 gate)."""
    engine = _make_engine()
    engine.set_stream_delay_gap_ref(0.16)  # threshold = 0.21
    engine.update_stream_delay_rates(0.26, 0.10)  # |G|=0.16 <= 0.21
    mock_traci = _mock_traci(speed=20.0)
    mock_traci.lane.getLength.return_value = 150.0
    mock_traci.vehicle.getLanePosition.return_value = 50.0
    with (
        patch("env.actions.traci", mock_traci),
        patch("env.actions.leader_info", return_value=None),
    ):
        mask, inelig = engine.apply("cav0", 1)
        assert inelig is True
        mock_traci.vehicle.setSpeed.assert_called_with("cav0", -1)


def test_fair_balance_allowed_above_ref_plus_tau():
    engine = _make_engine()
    engine.set_stream_delay_gap_ref(0.16)  # threshold = 0.21
    engine.update_stream_delay_rates(0.40, 0.10)  # |G|=0.30 > 0.21
    mock_traci = _mock_traci(speed=20.0)
    mock_traci.lane.getLength.return_value = 150.0
    mock_traci.vehicle.getLanePosition.return_value = 50.0
    with (
        patch("env.actions.traci", mock_traci),
        patch("env.actions.leader_info", return_value=None),
    ):
        mask, inelig = engine.apply("cav0", 1)
        assert inelig is False
        issued = _set_speed_args(mock_traci)
        assert issued and all(v != -1 for v in issued)


def test_fair_balance_slows_main_when_ramp_more_delayed():
    engine = _make_engine()
    engine.update_stream_delay_rates(0.05, 0.20)  # gap = -0.15
    mock_traci = _mock_traci(speed=20.0)
    mock_traci.vehicle.getLaneID.return_value = "1191554636_1"
    mock_traci.vehicle.getRoadID.return_value = "1191554636"
    mock_traci.lane.getLength.return_value = 200.0
    mock_traci.vehicle.getLanePosition.return_value = 100.0  # rem=100
    with (
        patch("env.actions.traci", mock_traci),
        patch("env.actions.leader_info", return_value=None),
    ):
        engine.apply("cav0", 1)
        issued = _set_speed_args(mock_traci)
        assert issued and all(v != -1 for v in issued)


def test_fair_balance_coasts_on_wrong_stream():
    """When main is more delayed, ramp should slow — main CAV coasts."""
    engine = _make_engine()
    engine.update_stream_delay_rates(0.20, 0.05)
    mock_traci = _mock_traci(speed=20.0)
    mock_traci.vehicle.getLaneID.return_value = "1191554636_1"
    mock_traci.vehicle.getRoadID.return_value = "1191554636"
    mock_traci.lane.getLength.return_value = 200.0
    mock_traci.vehicle.getLanePosition.return_value = 100.0
    with patch("env.actions.traci", mock_traci):
        engine.apply("cav0", 1)
        mock_traci.vehicle.setSpeed.assert_called_with("cav0", -1)


def test_recover_speed_pushes_when_clear_and_slow():
    engine = _make_engine()
    mock_traci = _mock_traci(speed=10.0)
    mock_traci.vehicle.getLaneID.return_value = "1191554636_1"  # pre

    def edge_spd(edge, min_n=3):
        return {"1487606179": 12.0, "1487606178": 22.0, "1191554636": 10.0}.get(edge)

    with (
        patch("env.actions.traci", mock_traci),
        patch("env.actions.leader_info", return_value=None),
        patch("env.actions._edge_mean_speed", side_effect=edge_spd),
    ):
        engine.apply("cav0", 2)  # recover_speed
        issued = _set_speed_args(mock_traci)
        assert issued and issued[-1] > 10.0


def test_recover_speed_coasts_when_merge_jammed():
    engine = _make_engine()
    mock_traci = _mock_traci(speed=10.0)
    mock_traci.vehicle.getLaneID.return_value = "1191554636_1"

    def edge_spd(edge, min_n=3):
        return {"1487606179": 5.0, "1487606178": 22.0}.get(edge)

    with (
        patch("env.actions.traci", mock_traci),
        patch("env.actions._edge_mean_speed", side_effect=edge_spd),
    ):
        engine.apply("cav0", 2)
        mock_traci.vehicle.setSpeed.assert_called_with("cav0", -1)


def test_vacate_drop_requests_lane_change():
    engine = _make_engine()
    mock_traci = _mock_traci(speed=15.0)
    mock_traci.vehicle.getLaneID.return_value = "1487606179_0"
    with (
        patch("env.actions.traci", mock_traci),
        patch("env.actions.leader_info", return_value=None),
    ):
        engine.apply("cav0", 3)
        mock_traci.vehicle.changeLane.assert_called()
        issued = _set_speed_args(mock_traci)
        assert issued and all(v != -1 for v in issued)


def test_vacate_drop_coasts_off_drop_lane():
    engine = _make_engine()
    mock_traci = _mock_traci(speed=15.0)
    mock_traci.vehicle.getLaneID.return_value = "1487606179_1"
    with patch("env.actions.traci", mock_traci):
        mask, inelig = engine.apply("cav0", 3)
        mock_traci.vehicle.setSpeed.assert_called_with("cav0", -1)
        mock_traci.vehicle.changeLane.assert_not_called()
        assert mask is False
        assert inelig is True


def test_vacate_drop_eligible_not_ineligible():
    engine = _make_engine()
    mock_traci = _mock_traci(speed=15.0)
    mock_traci.vehicle.getLaneID.return_value = "1487606179_0"
    with (
        patch("env.actions.traci", mock_traci),
        patch("env.actions.leader_info", return_value=None),
    ):
        engine.apply_all({"cav0": 3})
        assert engine.last_mask_fired["cav0"] is False
        assert engine.last_ineligible["cav0"] is False


def test_apply_all_records_ineligible():
    engine = _make_engine()
    mock_traci = _mock_traci(speed=15.0)
    mock_traci.vehicle.getLaneID.return_value = "1487606179_1"
    with patch("env.actions.traci", mock_traci):
        engine.apply_all({"cav0": 3})
        assert engine.last_ineligible["cav0"] is True
        assert engine.last_mask_fired["cav0"] is False


def test_protect_coasts_without_risk():
    engine = _make_engine()
    mock_traci = _mock_traci(speed=20.0)
    with (
        patch("env.actions.traci", mock_traci),
        patch("env.actions.leader_ttc", return_value=10.0),
        patch("env.actions.side_ttc", return_value=10.0),
        patch("env.actions.merge_arrival_pet", return_value=10.0),
    ):
        engine.apply("cav0", 4)
        mock_traci.vehicle.setSpeed.assert_called_with("cav0", -1)


def test_protect_slows_on_low_ttc():
    engine = _make_engine()
    mock_traci = _mock_traci(speed=20.0)
    with (
        patch("env.actions.traci", mock_traci),
        patch("env.actions.leader_info", return_value=None),
        patch("env.actions.leader_ttc", return_value=0.8),
        patch("env.actions.side_ttc", return_value=10.0),
        patch("env.actions.merge_arrival_pet", return_value=10.0),
    ):
        engine.apply("cav0", 4)
        issued = _set_speed_args(mock_traci)
        assert issued and all(v != -1 for v in issued)
        assert issued[-1] < 20.0


def test_coast_restores_original_vtype_tau():
    engine = _make_engine()
    mock_traci = _mock_traci(speed=20.0)
    mock_traci.vehicle.getTau.return_value = 1.5
    with patch("env.actions.traci", mock_traci):
        engine.apply("cav0", 0)
        mock_traci.vehicle.setTau.assert_called_once_with("cav0", 1.5)


def test_disabled_mode_remaps_to_coast():
    engine = _make_engine(mode_config={"disabled_modes": ["fair_balance"]})
    engine.update_stream_delay_rates(0.2, 0.05)
    mock_traci = _mock_traci(speed=20.0)
    mock_traci.lane.getLength.return_value = 150.0
    mock_traci.vehicle.getLanePosition.return_value = 50.0
    with patch("env.actions.traci", mock_traci):
        engine.apply("cav0", 1)
        mock_traci.vehicle.setSpeed.assert_called_with("cav0", -1)


def test_coast_force_blocks_when_corridor_healthy():
    engine = _make_engine(
        mode_config={
            "coast_force": True,
            "coast_force_delay_slack": 0.0,
            "coast_force_modes": None,  # all non-coast
            "coast_force_hold_steps": 1,
        }
    )
    engine.set_stream_delay_gap_ref(0.16)
    engine.set_delay_rate_ref(0.30)
    # |G|=0.10 <= 0.21, mean delay 0.25 <= 0.30 → healthy
    engine.update_stream_delay_rates(0.30, 0.20)
    assert engine.corridor_is_healthy() is True
    assert engine.coast_force_armed() is True
    mock_traci = _mock_traci(speed=20.0)
    mock_traci.lane.getLength.return_value = 150.0
    mock_traci.vehicle.getLanePosition.return_value = 50.0
    with (
        patch("env.actions.traci", mock_traci),
        patch("env.actions.leader_info", return_value=None),
        patch("env.actions.leader_ttc", return_value=10.0),
        patch("env.actions.side_ttc", return_value=10.0),
        patch("env.actions.merge_arrival_pet", return_value=10.0),
    ):
        # fair_balance blocked when healthy (protect would be exempt if TTC low)
        mask, _ = engine.apply("cav0", 1)
        assert mask is True
        mock_traci.vehicle.setSpeed.assert_called_with("cav0", -1)
        elig = engine.eligibility_mask("cav0")
        assert elig[0] == 1.0
        assert elig[1] == 0.0  # fair_balance blocked


def test_coast_force_scoped_modes_allows_recover():
    engine = _make_engine(
        mode_config={
            "coast_force": True,
            "coast_force_modes": ["fair_balance", "guard"],
            "coast_force_hold_steps": 1,
        }
    )
    engine.set_stream_delay_gap_ref(0.16)
    engine.set_delay_rate_ref(0.30)
    engine.update_stream_delay_rates(0.30, 0.20)
    assert engine.coast_force_armed() is True
    assert engine._coast_force_blocks("guard", "cav0", 20.0) is True
    assert engine._coast_force_blocks("recover", "cav0", 20.0) is False


def test_coast_force_exempts_protect_when_ttc_low():
    engine = _make_engine(
        mode_config={
            "coast_force": True,
            "coast_force_modes": ["fair_balance", "guard"],
            "coast_force_hold_steps": 1,
        }
    )
    engine.set_stream_delay_gap_ref(0.16)
    engine.set_delay_rate_ref(0.30)
    engine.update_stream_delay_rates(0.30, 0.20)
    assert engine.coast_force_armed() is True
    with (
        patch("env.actions.leader_ttc", return_value=0.5),
        patch("env.actions.side_ttc", return_value=10.0),
        patch("env.actions.merge_arrival_pet", return_value=10.0),
    ):
        assert engine._coast_force_blocks("guard", "cav0", 20.0) is False
        assert engine._coast_force_blocks("fair_balance", "cav0", 20.0) is True
    mock_traci = _mock_traci(speed=20.0)
    with (
        patch("env.actions.traci", mock_traci),
        patch("env.actions.leader_info", return_value=None),
        patch("env.actions.leader_ttc", return_value=0.5),
        patch("env.actions.side_ttc", return_value=10.0),
        patch("env.actions.merge_arrival_pet", return_value=10.0),
    ):
        mask, inelig = engine.apply("cav0", 4)  # protect
        assert mask is False
        assert inelig is False
        issued = _set_speed_args(mock_traci)
        assert issued and all(v != -1 for v in issued)


def test_coast_force_requires_hold_streak():
    engine = _make_engine(
        mode_config={
            "coast_force": True,
            "coast_force_modes": ["guard"],
            "coast_force_hold_steps": 3,
        }
    )
    engine.set_stream_delay_gap_ref(0.16)
    engine.set_delay_rate_ref(0.30)
    engine.update_stream_delay_rates(0.30, 0.20)
    assert engine.coast_force_armed() is False  # streak=1
    engine.update_stream_delay_rates(0.30, 0.20)
    assert engine.coast_force_armed() is False  # streak=2
    engine.update_stream_delay_rates(0.30, 0.20)
    assert engine.coast_force_armed() is True  # streak=3


def test_coast_force_min_step_delays_arming():
    engine = _make_engine(
        mode_config={
            "coast_force": True,
            "coast_force_modes": ["guard"],
            "coast_force_hold_steps": 1,
            "coast_force_min_step": 3,
        }
    )
    engine.set_stream_delay_gap_ref(0.16)
    engine.set_delay_rate_ref(0.30)
    engine.update_stream_delay_rates(0.30, 0.20)  # step=1
    assert engine.coast_force_armed() is False
    engine.update_stream_delay_rates(0.30, 0.20)  # step=2
    assert engine.coast_force_armed() is False
    engine.update_stream_delay_rates(0.30, 0.20)  # step=3
    assert engine.coast_force_armed() is True


def test_coast_force_allows_when_gap_unhealthy():
    engine = _make_engine(
        mode_config={
            "coast_force": True,
            "coast_force_modes": None,
            "coast_force_hold_steps": 1,
        }
    )
    engine.set_stream_delay_gap_ref(0.16)
    engine.set_delay_rate_ref(0.30)
    engine.update_stream_delay_rates(0.50, 0.10)  # |G|=0.40 > 0.21
    assert engine.corridor_is_healthy() is False
    mock_traci = _mock_traci(speed=20.0)
    mock_traci.lane.getLength.return_value = 150.0
    mock_traci.vehicle.getLanePosition.return_value = 50.0
    with (
        patch("env.actions.traci", mock_traci),
        patch("env.actions.leader_info", return_value=None),
    ):
        # fair_balance allowed by relative gate
        mask, inelig = engine.apply("cav0", 1)
        assert mask is False
        assert inelig is False
        issued = _set_speed_args(mock_traci)
        assert issued and all(v != -1 for v in issued)
