"""Tests for the metastable-control feasibility probe (design spec
docs/superpowers/specs/2026-08-12-metastable-feasibility-probe-design.md).

Covers: env.metastable_health_detail accessor, the pure rule decision
function, the scripted-intervention policy, and the health-recording wrapper.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from config.constants import MODE_NAMES, RAMP_EDGE
from env.core import TrafficShepherdMultiAgentEnv

COAST = MODE_NAMES.index("coast")
FAIR = MODE_NAMES.index("fair_balance")
PROTECT = MODE_NAMES.index("guard")


def test_metastable_health_detail_accessor():
    env = TrafficShepherdMultiAgentEnv(
        {
            "sumo_binary": "sumo",
            "sumo_config": "osm.light.sumocfg",
            "cav_penetration": 0.20,
            "max_episode_steps": 30,
            "simulation_step": 1.0,
        }
    )
    obs, _ = env.reset(seed=123)
    detail = None
    for _ in range(5):
        obs, _, _, _, _ = env.step({cav_id: 0 for cav_id in obs.keys()})
        detail = env.metastable_health_detail()
        if detail is not None:
            break
    env.close()
    if detail is not None:
        h, rho, v_mean = detail
        assert 0.0 <= h <= 1.0
        assert rho > 0.0
        assert v_mean >= 0.0
        # consistent with the Experiment-C private wrapper (same snapshot)
        assert env._merge_metastable_health() == h


from policies.metastable_rule import (  # noqa: E402
    DEFAULT_RULE_CFG,
    HealthRecordingWrapper,
    MetastableRulePolicy,
    ScriptedInterventionPolicy,
    _new_state,
    metastable_rule_decision,
)


def _cfg(**over):
    c = dict(DEFAULT_RULE_CFG)
    c.update(over)
    return c


def _fresh():
    return _new_state()


# ---- pure decision function (smoothness off for exact dial assertions) ----


def test_decision_healthy_coasts():
    mode, dial, _ = metastable_rule_decision(
        _fresh(), 0.8, 3.0, "ramp", None, _cfg(smoothness_on=False)
    )
    assert (mode, dial) == (COAST, 0.0)


def test_decision_transition_band_paces_mainline_only():
    s = _fresh()
    m, d, _ = metastable_rule_decision(s, 0.5, 3.0, "mainline", None, _cfg(smoothness_on=False))
    assert (m, d) == (PROTECT, 0.2)
    m, d, _ = metastable_rule_decision(_fresh(), 0.5, 3.0, "ramp", None, _cfg(smoothness_on=False))
    assert (m, d) == (COAST, 0.0)


def test_decision_sick_meters_ramp_by_rho_excess():
    s = _fresh()
    m, d, _ = metastable_rule_decision(s, 0.2, 5.5, "ramp", None, _cfg(smoothness_on=False))
    assert m == FAIR
    assert abs(d - 0.3) < 1e-9  # dial at rho == rho_crit
    m2, d2, _ = metastable_rule_decision(
        _fresh(), 0.2, 6.5, "ramp", None, _cfg(smoothness_on=False)
    )
    assert m2 == FAIR
    assert abs(d2 - 0.7) < 1e-9  # saturates at rho_dial_hi


def test_decision_sick_low_rho_no_meter():
    m, _, _ = metastable_rule_decision(_fresh(), 0.2, 3.0, "ramp", None, _cfg(smoothness_on=False))
    assert m == COAST


def test_decision_sick_paces_mainline():
    m, d, _ = metastable_rule_decision(
        _fresh(), 0.2, 6.0, "mainline", None, _cfg(smoothness_on=False)
    )
    assert (m, d) == (PROTECT, 0.2)


def test_decision_hysteresis_release_needs_sustained_recovery():
    cfg = _cfg(smoothness_on=False)
    s = _fresh()
    # start metering
    m, _, s = metastable_rule_decision(s, 0.2, 6.0, "ramp", None, cfg)
    assert m == FAIR and s["metering"] is True
    # h recovered but only briefly: still metering
    for _ in range(4):
        m, _, s = metastable_rule_decision(s, 0.7, 6.0, "ramp", None, cfg)
        assert m == FAIR, "must hold metering during the release window"
    # 5th consecutive recovered step -> release
    m, _, s = metastable_rule_decision(s, 0.7, 6.0, "ramp", None, cfg)
    assert s["metering"] is False
    assert m == COAST


def test_decision_ttc_floor_downgrades_meter_to_protect():
    cfg = _cfg(smoothness_on=False)
    m, d, _ = metastable_rule_decision(_fresh(), 0.2, 6.0, "ramp", 1.5, cfg)
    assert (m, d) == (PROTECT, 0.2)
    # floor disabled -> meter again
    cfg2 = _cfg(smoothness_on=False, ttc_floor_on=False)
    m, _, _ = metastable_rule_decision(_fresh(), 0.2, 6.0, "ramp", 1.5, cfg2)
    assert m == FAIR


def test_decision_smoothness_caps_dial_rate():
    s = _fresh()
    s["prev_dial"] = 0.7
    s["prev_mode"] = FAIR
    cfg = _cfg()  # smoothness on
    m, d, _ = metastable_rule_decision(s, 0.5, 3.0, "mainline", None, cfg)  # target 0.2
    assert 0.65 <= d <= 0.7  # capped at 0.05/step


def test_decision_smoothness_min_dwell():
    s = _fresh()
    cfg = _cfg()  # smoothness on
    m, d, s = metastable_rule_decision(s, 0.8, 3.0, "ramp", None, cfg)  # coast
    assert (m, d) == (COAST, 0.0)
    m, _, s = metastable_rule_decision(s, 0.2, 6.0, "ramp", None, cfg)  # wants meter
    assert m == COAST, "dwell: mode must be held >= min_dwell steps"
    m, _, s = metastable_rule_decision(s, 0.2, 6.0, "ramp", None, cfg)
    assert m == COAST
    m, _, s = metastable_rule_decision(s, 0.2, 6.0, "ramp", None, cfg)
    assert m == FAIR, "dwell satisfied after 3 steps"


# ---- policy-level wiring (mock env, no SUMO) ----


class _FakeEnv:
    def __init__(self, detail=(0.2, 6.0, 8.0)):
        self._detail = detail

    def metastable_health_detail(self):
        return self._detail


def test_rule_policy_meters_ramp_paces_mainline():
    env = _FakeEnv((0.2, 6.0, 8.0))
    snap = MagicMock()
    snap.road.side_effect = lambda v: RAMP_EDGE if v == "cav_r" else "some_edge"
    with (
        patch("policies.metastable_rule.get_active_snapshot", return_value=snap),
        patch("policies.metastable_rule.leader_ttc", return_value=float("inf")),
    ):
        pol = MetastableRulePolicy()
        pol.cfg["smoothness_on"] = False  # isolate wiring; dwell covered by pure tests
        pol.reset(env)
        acts = pol.act({"cav_r": np.zeros(10), "cav_m": np.zeros(10)})
    assert acts["cav_r"][0] == FAIR
    assert acts["cav_m"][0] == PROTECT


def test_scripted_burst_triggers_on_low_h_and_cycles():
    pol = ScriptedInterventionPolicy(FAIR, 0.6, on_steps=2, off_steps=2, trigger_h=0.4)
    pol.reset(_FakeEnv((0.2, 6.0, 8.0)))
    a1 = pol.act({"cav0": np.zeros(4)})
    assert a1["cav0"] == (FAIR, 0.6), "trigger fires immediately"
    a2 = pol.act({"cav0": np.zeros(4)})
    assert a2["cav0"] == (FAIR, 0.6), "second on-step"
    a3 = pol.act({"cav0": np.zeros(4)})
    assert a3["cav0"] == (COAST, 0.0), "burst exhausted -> coast"
    pol.act({"cav0": np.zeros(4)})  # off-step 1
    pol.act({"cav0": np.zeros(4)})  # off-step 2
    a6 = pol.act({"cav0": np.zeros(4)})
    assert a6["cav0"] == (FAIR, 0.6), "re-triggers after off window"


def test_scripted_continuous_always_on():
    pol = ScriptedInterventionPolicy(PROTECT, 0.2, trigger_h=None)
    pol.reset(_FakeEnv((0.8, 2.0, 25.0)))
    a = pol.act({"cav0": np.zeros(4)})
    assert a["cav0"] == (PROTECT, 0.2)


def test_recording_wrapper_records_health_per_step():
    inner = MagicMock()
    inner.direct_control = False
    inner.act.return_value = {"cav0": 0}
    env = _FakeEnv((0.6, 3.0, 20.0))
    w = HealthRecordingWrapper(inner)
    w.reset(env)
    w.act({"cav0": np.zeros(4)})
    w.act({"cav0": np.zeros(4)})
    assert len(w.history) == 2
    assert w.history[0][:3] == (0.6, 3.0, 20.0)  # (h, rho, v_mean, n_veh)
    assert w.history[0][3] is None  # no snapshot in the mock env


# ---- retry features (2026-08-12): health-in-obs + health reward ----

from env.features import raw_obs_dim  # noqa: E402
from env.rewards import compute_shepherd_reward  # noqa: E402
from train import _build_env_config  # noqa: E402


def _reward_kwargs_health(**over):
    w = {
        "efficiency": 0.0,
        "safety": 0.0,
        "stability": 0.0,
        "fairness": 0.0,
        "delay_budget": 0.0,
        "delay_win": 0.0,
        "fair_win": 0.0,
        "easy_intervention_tax": 0.0,
        "control_cost": 0.0,
        "mask_penalty": 0.0,
        "ineligible_penalty": 0.0,
        "mode_use_bonus": 0.0,
        "flow": 0.0,
        "tail": 0.0,
        "corridor_ttc_penalty": 0.0,
    }
    w.update(over)
    return w


def test_health_reward_ramps_with_h():
    base = compute_shepherd_reward(
        "cav0",
        0,
        35.0,
        _reward_kwargs_health(health=1.0),
        200.0,
        metastable_health=0.2,
    )
    at1 = compute_shepherd_reward(
        "cav0",
        0,
        35.0,
        _reward_kwargs_health(health=1.0),
        200.0,
        metastable_health=1.0,
    )
    mid = compute_shepherd_reward(
        "cav0",
        0,
        35.0,
        _reward_kwargs_health(health=1.0),
        200.0,
        metastable_health=0.6,
    )
    below = compute_shepherd_reward(
        "cav0",
        0,
        35.0,
        _reward_kwargs_health(health=1.0),
        200.0,
        metastable_health=0.0,
    )
    # R_health = clip((h-0.2)/0.8, 0, 1): h=1 -> +1.0, h=0.6 -> +0.5,
    # h<=0.2 -> +0.0
    assert abs((at1 - base) - 1.0) < 1e-9
    assert abs((mid - base) - 0.5) < 1e-9
    assert abs(below - base) < 1e-9


def test_health_reward_zero_weight_ignores_h():
    a = compute_shepherd_reward(
        "cav0", 0, 35.0, _reward_kwargs_health(), 200.0, metastable_health=1.0
    )
    b = compute_shepherd_reward(
        "cav0", 0, 35.0, _reward_kwargs_health(), 200.0, metastable_health=0.0
    )
    assert abs(a - b) < 1e-9


def test_raw_obs_dim_health():
    base = raw_obs_dim(20)
    assert raw_obs_dim(20, include_health=True) == base + 1
    assert (
        raw_obs_dim(20, include_cav_count=True, include_health=True)
        == raw_obs_dim(20, include_cav_count=True) + 1
    )


def test_raw_obs_dim_dv():
    base = raw_obs_dim(20)
    assert raw_obs_dim(20, include_dv=True) == base + 1
    assert (
        raw_obs_dim(20, include_health=True, include_dv=True)
        == raw_obs_dim(20, include_health=True) + 1
    )


def test_env_dv_obs_dim_and_value():
    env = TrafficShepherdMultiAgentEnv(
        {
            "sumo_binary": "sumo",
            "sumo_config": "osm.light.sumocfg",
            "cav_penetration": 0.20,
            "max_episode_steps": 20,
            "simulation_step": 1.0,
            "use_health_obs": True,
            "use_dv_obs": True,
        }
    )
    feat_dim = raw_obs_dim(20, include_health=True, include_dv=True)
    mask_dim = int(getattr(env, "_action_mask_dim", 0))
    want = feat_dim + mask_dim
    obs, _ = env.reset(seed=123)
    for o in obs.values():
        assert o.shape[0] == want
    for _ in range(5):
        obs, _, _, _, _ = env.step({cav_id: 0 for cav_id in obs.keys()})
    for o in obs.values():
        assert o.shape[0] == want
        # h is the 2nd-to-last feature slot, dv the last
        assert 0.0 <= o[feat_dim - 2] <= 1.0
        assert o[feat_dim - 1] >= 0.0, "injection dv magnitude is non-negative"
    env.close()


class _TrainArgs:
    density = "medium"
    curriculum_densities = None
    multi_density = False
    multi_penetration = False
    sumo_binary = "sumo"
    cav_penetration = 0.2
    max_episode_steps = 1200
    max_neighbors = 20
    max_cavs_critic = 32
    hdv_mixture = None
    baseline = "shepherd"


def test_build_env_config_health_mask_includes_protect():
    args = _TrainArgs()
    args.metastable_health_mask = True
    args.health_mask_includes_protect = True
    cfg = _build_env_config(args)
    assert cfg["metastable_health_mask"] is True
    assert cfg["metastable_health_mask_modes"] == [
        "fair_balance",
        "recover",
        "guard",
    ]


def test_env_health_obs_dim_and_value():
    env = TrafficShepherdMultiAgentEnv(
        {
            "sumo_binary": "sumo",
            "sumo_config": "osm.light.sumocfg",
            "cav_penetration": 0.20,
            "max_episode_steps": 20,
            "simulation_step": 1.0,
            "use_health_obs": True,
        }
    )
    # Obs = raw features (incl. the h slot) + action-mask bits (default
    # action_masking=True in DEFAULT_CONFIG).
    feat_dim = raw_obs_dim(20, include_health=True)
    mask_dim = int(getattr(env, "_action_mask_dim", 0))
    want = feat_dim + mask_dim
    obs, _ = env.reset(seed=123)
    for o in obs.values():
        assert o.shape[0] == want
    for _ in range(5):
        obs, _, _, _, _ = env.step({cav_id: 0 for cav_id in obs.keys()})
    for o in obs.values():
        assert o.shape[0] == want
        # h is the last feature slot (0.5 neutral when the merge is still
        # empty, else h in [0,1])
        assert 0.0 <= o[feat_dim - 1] <= 1.0
    env.close()


# ---- Stage 3 (2026-08-13): PREVENTIVE injection control ----

from policies.metastable_rule import (  # noqa: E402
    DEFAULT_PREVENTIVE_CFG,
    PreventiveInjectionPolicy,
    preventive_decision,
    preventive_vc,
)


def _pcfg(**over):
    c = dict(DEFAULT_PREVENTIVE_CFG)
    c.update(over)
    return c


def test_preventive_vc_linear_calibration():
    cfg = _pcfg()
    assert abs(preventive_vc(2.0, cfg) - 25.0) < 1e-9
    assert abs(preventive_vc(5.5, cfg) - 12.0) < 1e-9
    assert abs(preventive_vc(3.75, cfg) - 18.5) < 1e-9
    assert preventive_vc(None, cfg) is None


def test_preventive_rho_setpoint_meters_before_collapse():
    cfg = _pcfg(smoothness_on=False, levers=("rho",))
    # rho = 4.5 (still on the upper branch: rho_h = 0.29) -> preventive hold
    m, d, _ = preventive_decision(_fresh(), 0.5, 4.5, "ramp", None, None, None, cfg)
    assert m == FAIR
    assert abs(d - 0.433333) < 1e-4  # dial 0.3 + 0.4*(4.5-4.0)/(5.5-4.0)
    # rho below setpoint -> coast
    m, _, _ = preventive_decision(_fresh(), 0.7, 3.5, "ramp", None, None, None, cfg)
    assert m == COAST


def test_preventive_dv_gate_holds_slow_ramp_injection():
    cfg = _pcfg(smoothness_on=False, levers=("dv",))
    # rho=3.0 (metastable band) -> v_c = 20.71; v_ramp=10 -> 10 < 18.71 -> hold
    m, d, _ = preventive_decision(_fresh(), 0.6, 3.0, "ramp", None, 10.0, None, cfg)
    assert m == FAIR
    assert abs(d - 0.3) < 1e-9  # gentle dv hold dial
    # v_ramp at/above v_c - margin -> merge freely
    m, _, _ = preventive_decision(_fresh(), 0.6, 3.0, "ramp", None, 20.0, None, cfg)
    assert m == COAST


def test_preventive_dv_gate_inactive_below_rho_free():
    # Below rho_free (2.0) the free branch is unconditionally stable — a slow
    # injection cannot nucleate breakdown, so no hold (no pointless queue).
    cfg = _pcfg(smoothness_on=False, levers=("dv",))
    m, _, _ = preventive_decision(_fresh(), 0.9, 1.5, "ramp", None, 8.0, None, cfg)
    assert m == COAST


def test_preventive_lever_ablation_isolates():
    # rho-lever-only ignores the velocity condition
    cfg = _pcfg(smoothness_on=False, levers=("rho",))
    m, _, _ = preventive_decision(_fresh(), 0.6, 3.0, "ramp", None, 10.0, None, cfg)
    assert m == COAST
    # dv-lever-only ignores the density condition
    cfg2 = _pcfg(smoothness_on=False, levers=("dv",))
    m, _, _ = preventive_decision(_fresh(), 0.5, 4.5, "ramp", None, 30.0, None, cfg2)
    assert m == COAST


def test_preventive_fairness_gate_blocks_when_ramp_worse():
    cfg = _pcfg(smoothness_on=False, levers=("rho",))
    # gap = ema_main - ema_ramp = -0.2 -> the ramp is already worse: no hold
    m, _, _ = preventive_decision(_fresh(), 0.5, 4.5, "ramp", None, None, -0.2, cfg)
    assert m == COAST
    # ramp not worse -> hold allowed
    m, _, _ = preventive_decision(_fresh(), 0.5, 4.5, "ramp", None, None, 0.0, cfg)
    assert m == FAIR
    # gate disabled -> hold regardless
    cfg2 = _pcfg(smoothness_on=False, levers=("rho",), fairness_gate=False)
    m, _, _ = preventive_decision(_fresh(), 0.5, 4.5, "ramp", None, None, -0.2, cfg2)
    assert m == FAIR


def test_preventive_hysteresis_release():
    cfg = _pcfg(smoothness_on=False, levers=("rho",))
    s = _fresh()
    m, _, s = preventive_decision(s, 0.5, 4.5, "ramp", None, None, None, cfg)
    assert m == FAIR
    for _ in range(4):
        m, _, s = preventive_decision(s, 0.7, 3.5, "ramp", None, None, None, cfg)
        assert m == FAIR, "hold must persist during the release window"
    m, _, s = preventive_decision(s, 0.7, 3.5, "ramp", None, None, None, cfg)
    assert m == COAST


def test_preventive_ttc_floor_downgrades_hold_to_protect():
    cfg = _pcfg(smoothness_on=False, levers=("rho",))
    m, d, _ = preventive_decision(_fresh(), 0.5, 4.5, "ramp", 1.5, None, None, cfg)
    assert (m, d) == (PROTECT, 0.2)


def test_preventive_mainline_coasts_by_default():
    # Protect is threat-gated in the engine (emergency headway only), so the
    # controller does NOT pace the mainline — it coasts. Explicit mainline_pace
    # enables the (gentle) protect branch for completeness.
    cfg = _pcfg(smoothness_on=False)
    m, _, _ = preventive_decision(_fresh(), 0.5, 4.0, "mainline", None, None, None, cfg)
    assert m == COAST
    cfg2 = _pcfg(smoothness_on=False, mainline_pace=True)
    m, d, _ = preventive_decision(_fresh(), 0.5, 4.0, "mainline", None, None, None, cfg2)
    assert (m, d) == (PROTECT, 0.2)
    m, _, _ = preventive_decision(_fresh(), 0.8, 3.0, "mainline", None, None, None, cfg2)
    assert m == COAST


def test_preventive_policy_wiring():
    env = _FakeEnv((0.5, 4.5, 15.0))
    snap = MagicMock()
    snap.road.side_effect = lambda v: RAMP_EDGE if v == "cav_r" else "some_edge"
    snap.speed.return_value = 10.0
    snap.ordered_ids.return_value = []
    with (
        patch("policies.metastable_rule.get_active_snapshot", return_value=snap),
        patch("policies.metastable_rule.leader_ttc", return_value=float("inf")),
    ):
        pol = PreventiveInjectionPolicy(levers=("rho",))
        pol.cfg["smoothness_on"] = False
        pol.reset(env)
        acts = pol.act({"cav_r": np.zeros(10), "cav_m": np.zeros(10)})
    assert acts["cav_r"][0] == FAIR, "ramp CAV holds at rho above setpoint"
    assert acts["cav_m"][0] == COAST, "mainline coasts (protect is threat-gated)"


def test_wrapper_records_injection_dv_for_any_policy():
    env = _FakeEnv((0.5, 4.5, 15.0))
    snap = MagicMock()
    snap.ordered_ids.return_value = ["r1"]
    snap.road.side_effect = lambda v: RAMP_EDGE if v == "r1" else "other"
    snap.lane_pos.return_value = 300.0  # near the junction (RAMP_LANE_LENGTH 378.66)
    snap.speed.return_value = 10.0  # ramp approach at 10 m/s, merge at 15 -> |dv|=5
    inner = MagicMock()
    inner.direct_control = False
    inner.act.return_value = {"cav0": 0}
    with patch("policies.metastable_rule.get_active_snapshot", return_value=snap):
        w = HealthRecordingWrapper(inner)
        w.reset(env)
        w.act({"cav0": np.zeros(4)})
    assert w.dv_history == [5.0]
    # no ramp vehicles near the junction -> no sample
    snap.ordered_ids.return_value = []
    w2 = HealthRecordingWrapper(inner)
    w2.reset(env)
    with patch("policies.metastable_rule.get_active_snapshot", return_value=snap):
        w2.act({"cav0": np.zeros(4)})
    assert w2.dv_history == []


# ---- Stage 4 (2026-08-13): sweep features ----


def test_preventive_corridor_ttc_block_suppresses_hold():
    cfg = _pcfg(smoothness_on=False, levers=("rho",), corridor_ttc_block=True)
    # corridor has a sub-threshold TTC pair this step -> no hold
    m, _, _ = preventive_decision(_fresh(), 0.5, 4.5, "ramp", None, None, None, cfg, corr_ttc=1)
    assert m == COAST
    # corridor clean -> hold allowed
    m, _, _ = preventive_decision(_fresh(), 0.5, 4.5, "ramp", None, None, None, cfg, corr_ttc=0)
    assert m == FAIR
    # block disabled -> hold regardless
    cfg2 = _pcfg(smoothness_on=False, levers=("rho",))
    m, _, _ = preventive_decision(_fresh(), 0.5, 4.5, "ramp", None, None, None, cfg2, corr_ttc=3)
    assert m == FAIR


def test_preventive_setpoint_override():
    cfg = _pcfg(smoothness_on=False, levers=("rho",), rho_set=3.0)
    m, _, _ = preventive_decision(_fresh(), 0.6, 3.2, "ramp", None, None, None, cfg)
    assert m == FAIR, "holds at rho >= 3.0 (custom setpoint)"
    m, _, _ = preventive_decision(_fresh(), 0.7, 2.5, "ramp", None, None, None, cfg)
    assert m == COAST


def test_preventive_maindamp_inverts_metered_role():
    env = _FakeEnv((0.5, 4.5, 15.0))
    snap = MagicMock()
    snap.road.side_effect = lambda v: RAMP_EDGE if v == "cav_ramp" else "pre_edge"
    snap.speed.return_value = 20.0
    snap.ordered_ids.return_value = []
    with (
        patch("policies.metastable_rule.get_active_snapshot", return_value=snap),
        patch("policies.metastable_rule.leader_ttc", return_value=float("inf")),
    ):
        pol = PreventiveInjectionPolicy(
            levers=("rho",), meter_side="mainline", cfg_overrides={"rho_set": 3.5}
        )
        pol.cfg["smoothness_on"] = False
        pol.reset(env)
        acts = pol.act({"cav_main": np.zeros(10), "cav_ramp": np.zeros(10)})
    # mainline is the metered role -> holds; ramp coasts
    assert acts["cav_main"][0] == FAIR
    assert acts["cav_ramp"][0] == COAST


# ---- Stage 6 (2026-08-13): platoon-breaking VSL + always-on ----

from config.constants import PRE_MERGE_EDGE  # noqa: E402


def _vsl_snap(pre_speed: float) -> MagicMock:
    snap = MagicMock()
    snap.ordered_ids.return_value = ["p1"]
    snap.road.side_effect = lambda v: PRE_MERGE_EDGE if v == "p1" else "main_edge"
    snap.speed.side_effect = lambda v: pre_speed if v == "p1" else 20.0
    return snap


def test_vsl_pre_edge_speed_trigger():
    env = _FakeEnv((0.6, 3.0, 20.0))
    with (
        patch("policies.metastable_rule.get_active_snapshot", return_value=_vsl_snap(15.0)),
        patch("policies.metastable_rule.leader_ttc", return_value=float("inf")),
    ):
        pol = PreventiveInjectionPolicy(
            levers=("rho",),
            meter_side="mainline",
            trigger_src="pre_v",
            cfg_overrides={"v_trig": 18.0, "meter_dial_fixed": 0.4, "rho_set": 3.5},
        )
        pol.cfg["smoothness_on"] = False
        pol.reset(env)
        acts = pol.act({"cav_main": np.zeros(10)})
    assert acts["cav_main"][0] == FAIR, "pre-edge speed below v_trig -> mainline cuts"
    assert abs(acts["cav_main"][1] - 0.4) < 1e-9, "fixed VSL dial"


def test_vsl_idle_when_upstream_fast():
    env = _FakeEnv((0.6, 3.0, 20.0))
    with (
        patch("policies.metastable_rule.get_active_snapshot", return_value=_vsl_snap(25.0)),
        patch("policies.metastable_rule.leader_ttc", return_value=float("inf")),
    ):
        pol = PreventiveInjectionPolicy(
            levers=("rho",),
            meter_side="mainline",
            trigger_src="pre_v",
            cfg_overrides={"v_trig": 18.0, "meter_dial_fixed": 0.4, "rho_set": 3.5},
        )
        pol.cfg["smoothness_on"] = False
        pol.reset(env)
        acts = pol.act({"cav_main": np.zeros(10)})
    assert acts["cav_main"][0] == COAST, "free-flowing upstream -> no cut"


def test_always_on_gentle_damping():
    env = _FakeEnv((0.7, 2.5, 25.0))
    snap = MagicMock()
    snap.road.return_value = "main_edge"
    snap.speed.return_value = 20.0
    snap.ordered_ids.return_value = []
    with (
        patch("policies.metastable_rule.get_active_snapshot", return_value=snap),
        patch("policies.metastable_rule.leader_ttc", return_value=float("inf")),
    ):
        pol = PreventiveInjectionPolicy(
            levers=("rho",),
            meter_side="mainline",
            cfg_overrides={"rho_set": 0.0, "meter_dial_fixed": 0.15},
        )
        pol.cfg["smoothness_on"] = False
        pol.reset(env)
        acts = pol.act({"cav_main": np.zeros(10)})
    assert acts["cav_main"][0] == FAIR, "always-on damping holds regardless of rho"
    assert abs(acts["cav_main"][1] - 0.15) < 1e-9


# ---- Stage 7 (2026-08-14): inverted-dv probes ----


def test_match_v_paces_fast_ramp_approach():
    # ramp approach at 20 m/s into a 14 m/s merge -> hold (pace down)
    env = _FakeEnv((0.3, 5.0, 14.0))
    snap = MagicMock()
    snap.ordered_ids.return_value = ["r1"]
    snap.road.return_value = RAMP_EDGE  # both the CAV and the probe vehicle are on the ramp
    snap.lane_pos.return_value = 300.0
    snap.speed.return_value = 20.0
    with (
        patch("policies.metastable_rule.get_active_snapshot", return_value=snap),
        patch("policies.metastable_rule.leader_ttc", return_value=float("inf")),
    ):
        pol = PreventiveInjectionPolicy(
            levers=("rho",),
            meter_side="ramp",
            trigger_src="match_v",
            cfg_overrides={"meter_dial_fixed": 0.3, "match_v_margin": 3.0},
        )
        pol.cfg["smoothness_on"] = False
        pol.reset(env)
        acts = pol.act({"cav_r": np.zeros(10)})
    assert acts["cav_r"][0] == FAIR, "fast ramp approach into slow merge -> pace down"


def test_match_v_idle_when_ramp_close_to_stream():
    # ramp approach at 15 m/s vs 14 m/s merge -> within margin, no hold
    env = _FakeEnv((0.3, 5.0, 14.0))
    snap = MagicMock()
    snap.ordered_ids.return_value = ["r1"]
    snap.road.return_value = RAMP_EDGE
    snap.lane_pos.return_value = 300.0
    snap.speed.return_value = 15.0
    with (
        patch("policies.metastable_rule.get_active_snapshot", return_value=snap),
        patch("policies.metastable_rule.leader_ttc", return_value=float("inf")),
    ):
        pol = PreventiveInjectionPolicy(
            levers=("rho",),
            meter_side="ramp",
            trigger_src="match_v",
            cfg_overrides={"meter_dial_fixed": 0.3, "match_v_margin": 3.0},
        )
        pol.cfg["smoothness_on"] = False
        pol.reset(env)
        acts = pol.act({"cav_r": np.zeros(10)})
    assert acts["cav_r"][0] == COAST


# ---- Phase 2 (2026-08-13): feasibility margin feature + mask ----


def test_raw_obs_dim_margin():
    base = raw_obs_dim(20)
    assert raw_obs_dim(20, include_margin=True) == base + 1
    assert (
        raw_obs_dim(20, include_health=True, include_dv=True, include_margin=True)
        == raw_obs_dim(20, include_health=True, include_dv=True) + 1
    )


def test_feasibility_discount_math():
    from env.actions import ShepherdActionEngine

    def _feas_engine():
        mode_config = dict(
            max_decel=1.0,
            max_accel=1.0,
            deadband=0.5,
            guard_speed=8.0,
            disabled_modes=[],
            metastable_health_mask=True,
            metastable_health_mask_modes=["fair_balance", "recover"],
            feasibility_mask=True,
        )
        return ShepherdActionEngine(
            base_tau=1.0,
            tau_deltas={
                "coast": 0.0,
                "fair_balance": 0.25,
                "recover": -0.2,
                "vacate_drop": 0.0,
                "guard": 0.5,
                "wave_damp": 0.0,
                "merge_match": 0.0,
                "jad_absorb": 0.0,
                "jad_support": 0.0,
            },
            speed_factors={},
            max_speed=35.0,
            mode_config=mode_config,
        )

    eng = _feas_engine()
    eng.feas_fn = lambda: -500.0  # infeasible (negative margin)
    assert eng.feasibility_discount() == 0.05
    eng.feas_fn = lambda: 1000.0  # feasible
    assert eng.feasibility_discount() is None
    eng.feas_fn = lambda: 150.0  # inside the fade band (0..300)
    d = eng.feasibility_discount()
    assert 0.05 < d < 1.0
    # disabled -> ungated
    eng2 = _feas_engine()
    eng2.feasibility_mask_enabled = False
    eng2.feas_fn = lambda: -500.0
    assert eng2.feasibility_discount() is None


def test_env_margin_obs_dim_and_value():
    env = TrafficShepherdMultiAgentEnv(
        {
            "sumo_binary": "sumo",
            "sumo_config": "osm.light.sumocfg",
            "cav_penetration": 0.20,
            "max_episode_steps": 20,
            "simulation_step": 1.0,
            "use_health_obs": True,
            "use_dv_obs": True,
            "use_margin_obs": True,
        }
    )
    feat_dim = raw_obs_dim(20, include_health=True, include_dv=True, include_margin=True)
    mask_dim = int(getattr(env, "_action_mask_dim", 0))
    want = feat_dim + mask_dim
    obs, _ = env.reset(seed=123)
    for o in obs.values():
        assert o.shape[0] == want
    for _ in range(5):
        obs, _, _, _, _ = env.step({cav_id: 0 for cav_id in obs.keys()})
    for o in obs.values():
        assert o.shape[0] == want
        # margin is the last feature slot; with default L_min = 0 it equals
        # the pre-edge length (~1144 m) once rho is available
        assert o[feat_dim - 1] >= 0.0
    env.close()
