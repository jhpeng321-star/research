"""CPU-only C4 contract, local sensing, safety and imagination integration tests."""
from __future__ import annotations

import math

import pytest
import torch

from wms.control.bridge import make_cav_controller
from wms.control.contract import (ACCEL_MAX_MPS2, ACCEL_MIN_MPS2, F_OWN, F_REPORT,
                                  GROUP_SIGNAL_DIM, MAX_REPORTS, MERGE_S1_M,
                                  OWN_FEATURES, PHYSICS_DT_S, REPORT_FEATURES,
                                  SENSOR_RANGE_M, SPEED_LIMIT_MPS, RewardWeights)
from wms.control.obs import build_observations
from wms.control.policy import CavPolicy, RsuGroupEncoder, TeamCritic
from wms.control.reward import episode_summary, team_reward
from wms.control.safety import safe_accel
from wms.model.cav import direct_command_motion
from wms.model.residual_net import HdvConfig
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def state(s=(0.0, 30.0, -20.0, 50.0), v=(10.0, 12.0, 8.0, 15.0),
          lane=(1, 1, 1, 0), cav=(True, False, True, True), present=None,
          batch=1, dtype=torch.float32):
    k = len(s)
    return {"s_m": torch.tensor([s], dtype=dtype).repeat(batch, 1),
            "v_mps": torch.tensor([v], dtype=dtype).repeat(batch, 1),
            "a_mps2": torch.zeros(batch, k, dtype=dtype),
            "lane": torch.tensor([lane], dtype=torch.long).repeat(batch, 1),
            "present": torch.tensor([present or (True,) * k]).repeat(batch, 1),
            "is_cav": torch.tensor([cav]).repeat(batch, 1),
            "length_m": torch.full((batch, k), 5.0, dtype=dtype)}


def test_observation_shapes_feature_order_and_local_sensing():
    x = state(batch=2)
    obs = build_observations(x)
    assert obs.own.shape == (2, 4, F_OWN)
    assert obs.reports.shape == (2, 4, MAX_REPORTS, F_REPORT)
    assert obs.report_mask.shape == (2, 4, MAX_REPORTS)
    assert obs.report_mask.dtype == obs.cav_mask.dtype == torch.bool
    expected = torch.tensor([10, 0, 0, 1, 0, MERGE_S1_M, 25, 2, 1, 15, -2, 1, 1])
    torch.testing.assert_close(obs.own[0, 0], expected)
    assert not obs.own[:, 1].any()  # HDV neighbour is sensed but never an actor.
    # Closest CAV report is rear slot 2; its front neighbour is ego slot 0.
    expected_report = torch.tensor([-20, -2, 0, 0, 1, 0, 8, 15, 2, 1, 0, 0, 0],  # cmd, intent = 0
                                   dtype=torch.float32)
    torch.testing.assert_close(obs.reports[0, 0, 0], expected_report)
    assert obs.report_mask[0, 0].sum() == 2


def test_sensor_range_lane_absence_and_padding():
    x = state(s=(0, 101, -101, 1), lane=(1, 1, 1, 0),
              cav=(True, False, False, True), present=(True, True, True, False))
    # Arbitrary padding must not leak into observations, including NaNs.
    for name in ("s_m", "v_mps", "a_mps2", "length_m"):
        x[name][0, 3] = float("nan")
    obs = build_observations(x)
    assert torch.isfinite(obs.own).all() and torch.isfinite(obs.reports).all()
    for prefix in ("front", "rear"):
        assert obs.own[0, 0, OWN_FEATURES.index(prefix + "_gap_m")] == SENSOR_RANGE_M
        assert obs.own[0, 0, OWN_FEATURES.index(prefix + "_dv_mps")] == 0
        assert obs.own[0, 0, OWN_FEATURES.index(prefix + "_present")] == 0
    assert not obs.report_mask.any()
    assert not obs.reports.any()
    assert not obs.own[0, 1:].any()


def test_sensor_range_inclusive_and_report_source_age():
    x = state(s=(0, 100, -100, 101), v=(10, 11, 9, 12))
    x["is_cav"][:] = True
    age = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    obs = build_observations(x, age)
    assert obs.own[0, 0, OWN_FEATURES.index("front_gap_m")] == 95
    assert obs.own[0, 0, OWN_FEATURES.index("rear_gap_m")] == 95
    torch.testing.assert_close(obs.reports[0, 0, :3, REPORT_FEATURES.index("age_s")],
                               torch.tensor([0.2, 0.3, 0.4]))
    recipient_ages = age[:, None, :].expand(1, 4, 4) + 1
    obs2 = build_observations(x, recipient_ages)
    torch.testing.assert_close(obs2.reports[0, 0, :3, REPORT_FEATURES.index("age_s")], torch.tensor([1.2, 1.3, 1.4]))


def test_equal_position_overlap_is_sensed_and_charged_for_closing_pair():
    x = state(s=(0, 0), v=(10, 0), lane=(1, 1), cav=(True, False))
    obs = build_observations(x)
    assert obs.own[0, 0, OWN_FEATURES.index("front_gap_m")] == 0
    assert obs.own[0, 0, OWN_FEATURES.index("front_present")] == 1
    metrics = team_reward(x, x, torch.zeros_like(x["s_m"]), RewardWeights())
    torch.testing.assert_close(metrics["cost_ttc"], torch.tensor([3.0]))


def test_report_order_range_nearest_sixteen_and_ties():
    positions = (0, 400, -400, 401, -401) + tuple(range(19, 0, -1))
    n = len(positions)
    x = state(s=positions, v=(10,) * n, lane=(1,) * n, cav=(True,) * n)
    obs = build_observations(x)
    torch.testing.assert_close(obs.reports[0, 0, :, 0], torch.arange(1.0, 17.0))
    assert obs.report_mask[0, 0].all()
    x = state(s=(0, 400, -400, 401), cav=(True,) * 4)
    obs = build_observations(x)
    torch.testing.assert_close(obs.reports[0, 0, :2, 0], torch.tensor([400.0, -400.0]))
    assert obs.report_mask[0, 0].sum() == 2


def test_observation_position_speed_gradients():
    x = state()
    x["s_m"].requires_grad_()
    x["v_mps"].requires_grad_()
    obs = build_observations(x)
    loss = obs.own.square().sum() + obs.reports.square().sum()
    loss.backward()
    for name in ("s_m", "v_mps"):
        assert torch.isfinite(x[name].grad).all()
        assert x[name].grad.abs().sum() > 0


@pytest.mark.parametrize("all_absent", [False, True])
def test_network_shapes_masking_finiteness_and_parameter_budget(all_absent):
    x = state(batch=2)
    x["is_cav"][:, 2:] = False
    if all_absent:
        x["present"][:] = False
    obs = build_observations(x)
    actor, rsu, critic = CavPolicy(), RsuGroupEncoder(), TeamCritic()
    signal = rsu(obs, x["s_m"])
    accel, log_prob = actor(obs, signal)
    value = critic(x)
    assert signal.shape == (2, 4, GROUP_SIGNAL_DIM)
    assert accel.shape == log_prob.shape == (2, 4)
    assert value.shape == (2,)
    for tensor in (signal, accel, log_prob, value):
        assert torch.isfinite(tensor).all()
    assert not signal[~obs.cav_mask].any()
    assert not accel[~obs.cav_mask].any() and not log_prob[~obs.cav_mask].any()
    assert (accel >= ACCEL_MIN_MPS2).all() and (accel <= actor.accel_max).all()
    if all_absent:
        assert not value.any()
    for module in (actor, rsu, critic):
        assert sum(p.numel() for p in module.parameters()) < 1_000_000


def test_seed_determinism_and_tanh_log_density():
    x = state()
    obs = build_observations(x)
    torch.manual_seed(77)
    actor, rsu = CavPolicy(accel_max=ACCEL_MAX_MPS2), RsuGroupEncoder()   # legacy [-3, 1.5] formula
    group = rsu(obs, x["s_m"])
    torch.manual_seed(9)
    first = actor(obs, group)
    torch.manual_seed(9)
    second = actor(obs, group)
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    # Force N(0,1), then deterministic z=0 has log density including affine scale.
    with torch.no_grad():
        actor.head[-1].weight.zero_()
        actor.head[-1].bias.zero_()
    accel, log_prob = actor(obs, group, deterministic=True)
    torch.testing.assert_close(accel[obs.cav_mask], torch.full((3,), -0.75))
    expected = -0.5 * math.log(2 * math.pi) - math.log(2.25)
    torch.testing.assert_close(log_prob[obs.cav_mask], torch.full((3,), expected))
    with torch.no_grad():
        actor.head[-1].bias[0] = 30.0
    assert torch.isfinite(actor(obs, group, deterministic=True)[1]).all()


def test_group_zone_conditioning_and_set_equivariance():
    torch.manual_seed(4)
    x = state(s=(-150, 20, 180, 200), cav=(True,) * 4)
    obs = build_observations(x)
    actor, rsu, critic = CavPolicy(), RsuGroupEncoder(), TeamCritic()
    signal = rsu(obs, x["s_m"])
    # Identical local features still get position-conditioned group signals.
    obs.own[:] = obs.own[:, :1].clone()
    conditioned = rsu(obs, x["s_m"])
    assert not torch.allclose(conditioned[:, 0], conditioned[:, 2])
    obs = build_observations(x)
    permutation = torch.tensor([2, 0, 3, 1])
    shuffled = {key: val[:, permutation] for key, val in x.items()}
    shuffled_obs = build_observations(shuffled)
    shuffled_signal = rsu(shuffled_obs, shuffled["s_m"])
    torch.testing.assert_close(shuffled_signal, signal[:, permutation], atol=1e-6, rtol=1e-5)
    original_action = actor(obs, signal, deterministic=True)[0]
    shuffled_action = actor(shuffled_obs, shuffled_signal, deterministic=True)[0]
    torch.testing.assert_close(shuffled_action, original_action[:, permutation], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(critic(shuffled), critic(x), atol=1e-6, rtol=1e-5)


def test_critic_uses_hdvs_and_ignores_padding():
    torch.manual_seed(44)
    x = state(cav=(False,) * 4, present=(True, True, True, False))
    critic = TeamCritic().double()
    x = {name: value.double() if value.is_floating_point() else value for name, value in x.items()}
    x["v_mps"].requires_grad_()
    critic(x).sum().backward()
    assert x["v_mps"].grad[0, :3].abs().sum() > 0
    assert x["v_mps"].grad[0, 3] == 0
    before = critic(x)
    x["s_m"][0, 3] = 1e6
    torch.testing.assert_close(critic(x), before)


def test_reward_numeric_components_normalization_and_empty_batch():
    x = state(s=(MERGE_S1_M - 1, 0, 0, 0), v=(0, SPEED_LIMIT_MPS, 10, 10),
              present=(True, True, False, False))
    nxt = {**x, "s_m": x["s_m"] + torch.tensor([[2, 0, 0, 0]])}
    accel = torch.tensor([[-4.0, 2.0, 100.0, 100.0]])
    weights = RewardWeights()
    metrics = team_reward(x, nxt, accel, weights)
    assert set(metrics) == {"reward", "cost_ttc", "delay", "throughput", "smooth",
                            "hard_brake", "fairness_delay", "source_wait"}
    assert all(t.shape == (1,) and torch.isfinite(t).all() for t in metrics.values())
    torch.testing.assert_close(metrics["delay"], torch.tensor([0.5]))
    torch.testing.assert_close(metrics["smooth"], torch.tensor([20.0]))
    torch.testing.assert_close(metrics["hard_brake"], torch.tensor([1.0]))
    expected_crossing = torch.sigmoid(torch.tensor(1.0)) - torch.sigmoid(torch.tensor(-1.0))
    torch.testing.assert_close(metrics["throughput"], expected_crossing[None])
    expected = ((-metrics["delay"] + 0.5 * metrics["throughput"] - 0.05 * metrics["smooth"]
                 - metrics["hard_brake"]) / 2 - metrics["fairness_delay"])
    torch.testing.assert_close(metrics["reward"], expected)
    x["present"][:] = False
    empty = team_reward(x, {**nxt, "present": x["present"]}, accel, weights)
    assert all(not t.any() for t in empty.values())


def test_fairness_responds_to_worst_delays_and_has_gradients():
    n = 20
    x = state(s=tuple(range(n)), v=(15,) * n, lane=(0,) * n, cav=(False,) * n,
              dtype=torch.float64)
    equal = team_reward(x, x, torch.zeros_like(x["s_m"]), RewardWeights())["fairness_delay"]
    expected = PHYSICS_DT_S * (1 - 15 / SPEED_LIMIT_MPS)
    torch.testing.assert_close(equal, torch.tensor([expected], dtype=torch.float64))
    nxt = {**x, "v_mps": x["v_mps"].clone()}
    nxt["v_mps"][0, 0] = 0
    nxt["v_mps"].requires_grad_()
    worse = team_reward(x, nxt, torch.zeros_like(x["s_m"]), RewardWeights())["fairness_delay"]
    assert worse > equal
    worse.sum().backward()
    assert nxt["v_mps"].grad[0, 0].abs() > nxt["v_mps"].grad[0, 1].abs()
    assert torch.isfinite(nxt["v_mps"].grad).all()


def test_ttc_closing_same_lane_pairs_and_gradient():
    x = state(s=(0, 15, 100, 30), v=(20, 10, 10, 0), lane=(1, 1, 0, 1),
              present=(True, True, True, False))
    x["s_m"].requires_grad_()
    x["v_mps"].requires_grad_()
    cost = team_reward(x, x, torch.zeros_like(x["s_m"]), RewardWeights())["cost_ttc"]
    # Bumper gap 10, closing speed 10: TTC=1, soft relu(3-1)=2.
    torch.testing.assert_close(cost, torch.tensor([2.0]), atol=1e-5, rtol=1e-5)
    cost.sum().backward()
    assert x["v_mps"].grad[0, 0] > 0
    assert x["s_m"].grad[0, 1] < 0
    stationary = {**x, "v_mps": torch.full_like(x["v_mps"], 10)}
    assert not team_reward(stationary, stationary, torch.zeros_like(x["s_m"]),
                           RewardWeights())["cost_ttc"].any()


def test_reward_position_speed_gradients_and_episode_logs():
    x = state(s=(MERGE_S1_M - 2, 30, -20, 50))
    nxt = {**x, "s_m": (x["s_m"] + 1).requires_grad_(),
           "v_mps": x["v_mps"].clone().requires_grad_()}
    metrics = team_reward(x, nxt, torch.zeros_like(x["s_m"]), RewardWeights())
    metrics["reward"].sum().backward()
    assert nxt["s_m"].grad.abs().sum() > 0
    assert nxt["v_mps"].grad.abs().sum() > 0
    summary = episode_summary([metrics, metrics])
    assert summary["n_steps"] == 2 and summary["duration_s"] == 1
    assert summary["delay"] == pytest.approx(2 * metrics["delay"].item())
    assert all(isinstance(value, float) for value in summary.values())
    assert episode_summary([])["duration_s"] == 0


@pytest.mark.parametrize("leader_speed", [0.0, 8.0])
def test_safety_stops_with_two_metre_gap_behind_braking_leader(leader_speed):
    x = state(s=(0, 35), v=(12, leader_speed), lane=(1, 1), cav=(True, False))
    minimum_gap = float("inf")
    for _ in range(50):
        proposal = torch.tensor([[1.5, -3.0]])
        a = safe_accel(x, proposal)
        assert (a >= -3).all()
        assert a[0, 1] == -3  # HDV proposal is preserved.
        command = torch.stack((a, x["v_mps"] + 1.5 * PHYSICS_DT_S), -1)
        s, v, actual, _ = direct_command_motion(x["s_m"], x["v_mps"], command,
                                               PHYSICS_DT_S, -3, 1.5)
        x = {**x, "s_m": s, "v_mps": v, "a_mps2": actual}
        minimum_gap = min(minimum_gap, (s[0, 1] - s[0, 0] - 5).item())
    assert minimum_gap >= 2.0 - 1e-5
    assert x["v_mps"][0, 0] < 1e-3


def test_safety_gradient_no_leader_and_actuator_floor():
    x = state(s=(0, 35), v=(12, 0), lane=(1, 1), cav=(True, False))
    x["s_m"].requires_grad_()
    proposal = torch.tensor([[1.5, 0.0]], requires_grad=True)
    filtered = safe_accel(x, proposal)
    filtered.sum().backward()
    assert torch.isfinite(x["s_m"].grad).all() and x["s_m"].grad.abs().sum() > 0
    assert torch.isfinite(proposal.grad).all()
    infeasible = {**x, "s_m": torch.tensor([[0.0, 6.0]])}
    assert safe_accel(infeasible, proposal)[0, 0] == ACCEL_MIN_MPS2
    solo = state(s=(0,), v=(10,), lane=(1,), cav=(True,))
    torch.testing.assert_close(safe_accel(solo, torch.tensor([[2.0]])), torch.tensor([[1.5]]))


def test_bridge_decision_clock_hold_reset_record_and_gradients():
    calls = []

    class CountingPolicy:
        def __call__(self, obs, group):
            calls.append(True)
            action = torch.full_like(obs.own[..., 0], 0.1 * len(calls), requires_grad=True)
            return action, action * 0

    x = state(s=(0,), v=(10,), lane=(1,), cav=(True,))
    records = []
    controller = make_cav_controller(CountingPolicy(), RsuGroupEncoder(), safe_accel, records)
    actions = []
    for index in range(4):
        motion = controller({**x, "physics_step_index": torch.full((1, 1), index)})
        assert motion.shape == (1, 1, 4)
        actions.append(motion[..., 3])
        x = {**x, "s_m": motion[..., 0], "v_mps": motion[..., 2], "a_mps2": motion[..., 3]}
    assert len(calls) == len(records) == 2
    torch.testing.assert_close(actions[0], actions[1])
    torch.testing.assert_close(actions[2], actions[3])
    x["s_m"].sum().backward()
    assert records[0]["actions"].grad.abs().sum() > 0
    assert records[1]["actions"].grad.abs().sum() > 0
    controller({**x, "physics_step_index": 0})
    assert len(calls) == 3


@pytest.mark.parametrize("front_s,rear_s,ego_s,expected_lane", [
    (31, 9, 20, 1), (30.9, 9, 20, 2), (31, 9.1, 20, 2),
    (31, 9, -1, 2), (100, 50, 80, 2), (20, 0, 20, 2),
])
def test_bridge_ramp_merge_gap_and_zone(front_s, rear_s, ego_s, expected_lane):
    x = state(s=(ego_s, front_s, rear_s, 40), v=(0,) * 4,
              lane=(2, 1, 1, 0), cav=(True, False, False, True))
    actor = CavPolicy()
    controller = make_cav_controller(actor, RsuGroupEncoder(), safe_accel)
    result = controller({**x, "physics_step_index": 0})
    assert result[0, 0, 1] == expected_lane
    assert result[0, 3, 1] == 0


def test_reward_of_ten_step_wms_rollout_reaches_actor_and_group_parameters():
    torch.manual_seed(2026)
    x = state(s=(-100, -40, -140, 0), v=(12, 15, 10, 13),
              lane=(1, 1, 1, 0), cav=(True, False, True, False))
    actor, rsu = CavPolicy(hidden_dim=32, lane_head=False), RsuGroupEncoder(hidden_dim=32)  # accel-only bridge path
    records = []
    controller = make_cav_controller(actor, rsu, safe_accel, records)
    model = WmSModel(ModelSpec(history_hidden=16, k_hidden=8),
                     hdv=HdvConfig(d_model=16, n_heads=2, n_layers=1, ffn=32))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    rollout = model.rollout(x, n_steps=10, cav_controller=controller)
    rewards = []
    current = x
    for i in range(10):
        nxt = {**x, "s_m": rollout["positions"][:, i], "v_mps": rollout["speeds"][:, i],
               "lane": rollout["lanes"][:, i], "a_mps2": rollout["accel"][:, i],
               "present": rollout["masks"][:, i]}
        rewards.append(team_reward(current, nxt, rollout["accel"][:, i], RewardWeights())["reward"])
        current = nxt
    (-torch.stack(rewards).sum()).backward()
    assert len(records) == 5
    for module in (actor, rsu):
        grads = [p.grad for p in module.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum().item() for g in grads) > 0
    # A final return reaches earlier decisions through the actual WM-S graph.
    assert rollout["positions"].requires_grad and rollout["speeds"].requires_grad


def test_bridge_simultaneous_merges_preserve_target_lane_six_metre_gaps():
    x = state(s=(20, 28), v=(0, 0), lane=(2, 2), cav=(True, True))
    controller = make_cav_controller(CavPolicy(), RsuGroupEncoder(), safe_accel)
    result = controller({**x, "physics_step_index": 0})
    torch.testing.assert_close(result[..., 1], torch.tensor([[2.0, 1.0]]))
    # At an eleven-metre front-bumper separation both merges are admissible.
    x["s_m"] = torch.tensor([[20.0, 31.0]])
    result = controller({**x, "physics_step_index": 0})
    torch.testing.assert_close(result[..., 1], torch.tensor([[1.0, 1.0]]))
