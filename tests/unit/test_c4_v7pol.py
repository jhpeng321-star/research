"""V7POL (owner-approved 2026-10-17): Dreamer-policy training fixes from docs/C4_DIAGNOSIS_20261016.md.

Every switch defaults off and is then bit-identical:
1. ``reward.ttc_pairs = "cav"`` (TTC cost only over pairs with a CAV follower or leader) and
   ``trainer.ttc_budget_mode = "native_relative"`` (+ ``ttc_native_margin``; per-start excess over the
   counterfactual all-native base row, absolute budget without one) and ``trainer.lagrange_scaled``;
2. ``trainer.alpha_min`` (floor on the learned entropy temperatures, accel and lane);
3. ``policy.lane_control = "native"`` (actor without a lane head; no lane commands in SUMO);
4. ``reward.merge_discharge`` (un-normalised merge-end crossing count);
5. ``configs/c4_policy_v7.toml`` resolves and runs a 2-update CLI smoke (fake WM).
CPU only; no SUMO, no files from runs/."""
from __future__ import annotations

import importlib.util
import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wms.control.contract import MERGE_S1_M, PHYSICS_DT_S, RewardWeights
from wms.control.counterfactual import CounterfactualConfig
from wms.control.dreamer import DreamerConfig, DreamerTrainer
from wms.control.policy import CavPolicy, PolicyExecutor, RsuGroupEncoder, TeamCritic
from wms.control.reward import _ttc_cost, team_reward
from wms.control.start_states import split_start_states

REPO = Path(__file__).resolve().parents[2]
V7_KEYS = {"ttc_budget_mode", "ttc_native_starts", "ttc_excess", "ttc_penalty", "ttc_native_cost",
           "ttc_policy_cost_native_starts"}


def _helpers(name):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_v7pol_helpers_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D = _helpers("test_c4_dreamer")
P2 = _helpers("test_c4_policy2")


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _close_pairs_state():
    """CAV 0 closes on HDV 1 (lane 1, TTC 2.5 s); HDV 2 closes on HDV 3 (lane 0, TTC 2.0 s);
    CAVs 4 / 5 far upstream (a counterfactual group of two)."""
    return P2.state([-500.0, -470.0, -600.0, -575.0, -900.0, -800.0], [1, 1, 0, 0, 0, 1],
                    v=[25.0, 15.0, 25.0, 15.0, 20.0, 20.0],
                    cav=[True, False, False, False, True, True])


def _cost_trainer(*, policy=None, lanes=False, counterfactual=True, reward=None, hidden=16, **cfg):
    from wms.control.bridge import make_cav_controller
    from wms.control.lanes import LaneRules
    from wms.control.obs import build_observations
    from wms.control.safety import safe_accel
    from wms.control.train_cli import GroupSignal

    settings = dict(batch_size=2, horizon=3, checkpoint_every=0, validation_every=0,
                    actor_lr=1e-3, critic_lr=1e-3)
    settings.update(cfg)
    return DreamerTrainer(
        [P2.LaneWorldModel()], policy if policy is not None else CavPolicy(hidden_dim=hidden),
        GroupSignal(RsuGroupEncoder(hidden_dim=hidden)), TeamCritic(hidden_dim=hidden),
        P2.Starts(_close_pairs_state()), observation_builder=build_observations,
        reward_fn=team_reward, safety=safe_accel, controller_factory=make_cav_controller,
        config=DreamerConfig(**settings), reward_weights=reward or RewardWeights(),
        lane_rules=LaneRules() if lanes else None,
        counterfactual=CounterfactualConfig(enabled=counterfactual, n_individual=2, n_group=1,
                                            n_flip=1, group_max_size=3))


def _native_policy(hidden=16):
    policy = CavPolicy(hidden_dim=hidden, lane_head=False)
    policy.lane_control = "native"
    return policy


# ------------------------------------------------------------------ 1. TTC pairs
def test_ttc_pairs_cav_drops_human_human_pairs_and_keeps_cav_pairs():
    x = _close_pairs_state()
    every = _ttc_cost(x, 3.0)
    cav = _ttc_cost(x, 3.0, "cav")
    assert torch.equal(every, _ttc_cost(x, 3.0, "all"))
    hdv_only = {**x, "is_cav": torch.zeros_like(x["is_cav"])}
    assert float(_ttc_cost(hdv_only, 3.0, "cav")) == 0.0          # no CAV -> no pair
    def exposure(ttc):                                            # softplus relu(3 - TTC)
        return 0.05 * math.log1p(math.exp((3.0 - ttc) / 0.05))
    assert float(every) == pytest.approx(exposure(2.5) + exposure(2.0), rel=1e-5)
    assert float(cav) == pytest.approx(exposure(2.5), rel=1e-5)   # CAV follower -> HDV leader only
    # A CAV LEADER also qualifies: HDV 2 behind CAV 3.
    lead = {**x, "is_cav": torch.tensor([[False, False, False, True, False, False]])}
    assert float(_ttc_cost(lead, 3.0, "cav")) == pytest.approx(exposure(2.0), rel=1e-5)
    with pytest.raises(ValueError, match="ttc_pairs"):
        _ttc_cost(x, 3.0, "hdv")
    with pytest.raises(ValueError, match="is_cav"):
        _ttc_cost({k: v for k, v in x.items() if k != "is_cav"}, 3.0, "cav")
    metrics = team_reward(x, x, torch.zeros_like(x["s_m"]), RewardWeights(ttc_pairs="cav"))
    assert torch.equal(metrics["cost_ttc"], cav)


def test_reward_defaults_are_bit_identical():
    x = _close_pairs_state()
    nxt = {**x, "s_m": x["s_m"] + 10.0, "v_mps": x["v_mps"] - 1.0}
    accel = torch.full_like(x["s_m"], -0.5)
    a = team_reward(x, nxt, accel, RewardWeights())
    b = team_reward(x, nxt, accel, RewardWeights(ttc_pairs="all", merge_discharge=0.0))
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), key


# ------------------------------------------------------------------ 4. merge discharge
def test_merge_discharge_adds_unnormalised_crossings_with_gradient():
    x = P2.state([MERGE_S1_M - 0.5, -300.0, -200.0], [1, 1, 0], v=[20.0, 20.0, 20.0],
                 cav=[True, False, False])
    next_s = (x["s_m"] + PHYSICS_DT_S * x["v_mps"]).requires_grad_(True)
    nxt = {**x, "s_m": next_s}
    accel = torch.zeros_like(x["s_m"])
    base = team_reward(x, nxt, accel, RewardWeights())
    with_term = team_reward(x, nxt, accel, RewardWeights(merge_discharge=0.8))
    crossing = float(base["throughput"].detach())
    assert 0.5 < crossing < 1.0                                    # one vehicle crosses s = 75.44 m
    assert float((with_term["reward"] - base["reward"]).detach()) == pytest.approx(0.8 * crossing, rel=1e-6)
    grad, = torch.autograd.grad((with_term["reward"] - base["reward"]).sum(), next_s)
    assert grad[0, 0] > 0 and not grad[0, 1:].any()               # only the crossing vehicle
    far = P2.state([-500.0], [1])
    still = {**far, "s_m": far["s_m"] + 10.0}
    zero = torch.zeros_like(far["s_m"])
    assert float(team_reward(far, still, zero, RewardWeights(merge_discharge=0.8))["reward"]
                 - team_reward(far, still, zero, RewardWeights())["reward"]) == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------------ config validation
def test_config_defaults_and_validation():
    cfg = DreamerConfig()
    assert (cfg.ttc_budget_mode, cfg.ttc_native_margin, cfg.lagrange_scaled, cfg.alpha_min) == \
        ("absolute", 0.0, False, 0.0)
    for bad in (dict(ttc_budget_mode="relative"), dict(ttc_native_margin=-1.0),
                dict(alpha_min=-1e-3), dict(lagrange_scaled=1), dict(alpha_min=math.inf)):
        with pytest.raises(ValueError):
            DreamerConfig(**bad)
    DreamerConfig(ttc_budget_mode="native_relative", ttc_native_margin=0.2, lagrange_scaled=True,
                  alpha_min=1e-3)


# ------------------------------------------------------------------ default off = bit-identical
def test_defaults_off_leave_updates_and_checkpoints_bit_identical(tmp_path, monkeypatch):
    monkeypatch.setattr("wms.control.dreamer.checked_path", lambda p, **k: Path(p))
    torch.manual_seed(11)
    base = _cost_trainer(lanes=True, entropy_target=0.5, alpha_lr=1e-2)
    a = base.update()
    torch.manual_seed(11)
    explicit = _cost_trainer(lanes=True, entropy_target=0.5, alpha_lr=1e-2,
                             ttc_budget_mode="absolute", ttc_native_margin=0.0,
                             lagrange_scaled=False, alpha_min=0.0)
    b = explicit.update()
    a.pop("wall_time_s"), b.pop("wall_time_s")
    assert a == b and not V7_KEYS & a.keys()
    base.save_checkpoint(tmp_path / "a.pt")
    explicit.save_checkpoint(tmp_path / "b.pt")
    payload_a = torch.load(tmp_path / "a.pt", weights_only=False)
    payload_b = torch.load(tmp_path / "b.pt", weights_only=False)
    D._assert_equal(payload_a, payload_b)
    assert "lane_control" not in payload_a and payload_a["lane_head"] is True


# ------------------------------------------------------------------ native-relative budget
def test_native_relative_terms_arithmetic_and_fallback():
    owner = SimpleNamespace(config=DreamerConfig(ttc_budget_mode="native_relative", ttc_budget=3.0,
                                                 ttc_native_margin=0.5))
    # [E=1, H=2, N=5]: rows 0..2 = all-CAV rows of starts 0..2, rows 3 / 4 = base rows of starts 0 / 1.
    costs = torch.tensor([[[2.0, 1.0, 4.0, 1.0, 2.0], [4.0, 1.0, 6.0, 1.0, 2.0]]], requires_grad=True)
    imagined = {"n_base": 3, "costs": costs,
                "variants": SimpleNamespace(base_row={0: 3, 1: 4})}
    penalty, dual, budget, extra = DreamerTrainer._safety_terms(owner, imagined, costs[..., :3].mean())
    # per start: c = (3, 1, 5); ref = (1 + 0.5, 2 + 0.5, absolute 3.0) -> excess (1.5, -1.5, 2.0)
    assert float(penalty) == pytest.approx((1.5 + 0.0 + 2.0) / 3)  # relu: no credit below native
    assert float(dual) == pytest.approx(0.0) and budget == 0.0    # dual: native starts only
    assert extra["ttc_native_starts"] == 2 and extra["ttc_native_cost"] == pytest.approx(1.5)
    assert extra["ttc_policy_cost_native_starts"] == pytest.approx(2.0)
    penalty.backward()
    assert costs.grad[..., 3:].abs().sum() == 0                   # native rows are a fixed reference
    assert costs.grad[..., 1].abs().sum() == 0 and costs.grad[..., 0].abs().sum() > 0
    # No base row at all -> absolute budget everywhere, the dual over all starts.
    imagined = {"n_base": 3, "costs": costs.detach(), "variants": None}
    penalty, dual, budget, extra = DreamerTrainer._safety_terms(owner, imagined, None)
    assert float(dual) == pytest.approx(((3 - 3.0) + (1 - 3.0) + (5 - 3.0)) / 3)
    assert float(penalty) == pytest.approx(2.0 / 3) and extra["ttc_native_starts"] == 0
    # absolute mode returns the original terms
    owner.config = DreamerConfig()
    mean_cost = costs.detach()[..., :3].mean()
    out = DreamerTrainer._safety_terms(owner, {}, mean_cost)
    assert out[0] is mean_cost and float(out[1]) == float(mean_cost) and out[2] is None


def test_native_relative_update_uses_base_rows_and_moves_the_dual():
    torch.manual_seed(3)
    trainer = _cost_trainer(ttc_budget_mode="native_relative", dual_lr=0.5,
                            reward=RewardWeights(ttc_pairs="cav"))
    metrics = trainer.update()
    assert metrics["ttc_native_starts"] == 2 and metrics["ttc_budget_mode"] == "native_relative"
    assert metrics["ttc_native_cost"] > 0                       # CAV 0 closing on HDV 1
    assert trainer.lagrange_multiplier == pytest.approx(max(0.0, 0.5 * metrics["ttc_excess"]))
    assert math.isfinite(metrics["actor_grad_norm"])


def test_lagrange_scaled_divides_the_penalty_by_the_return_scale():
    losses = {}
    for scaled in (False, True):
        torch.manual_seed(4)
        trainer = _cost_trainer(counterfactual=False, lagrange_init=2.0, dual_lr=0.0,
                                entropy_weight=0.0, return_ema=1.0, lagrange_scaled=scaled)
        trainer.return_low, trainer.return_high = 0.0, 10.0       # fixed scale 10
        trainer.actor_objective = lambda imagined, scale, weights: (torch.zeros(()), {})
        metrics = trainer.update()
        assert metrics["return_scale"] == pytest.approx(10.0) and metrics["cost_ttc"] > 0
        losses[scaled] = (metrics["actor_loss"], metrics["cost_ttc"])
    assert losses[False][0] == pytest.approx(2.0 * losses[False][1], rel=1e-5)
    assert losses[True][0] == pytest.approx(2.0 * losses[True][1] / 10.0, rel=1e-5)


# ------------------------------------------------------------------ 2. entropy floor
def test_alpha_min_floors_the_accel_temperature():
    torch.manual_seed(7)
    free = D._trainer(entropy_weight=3e-3, entropy_target=-50.0, alpha_lr=1.0)
    floored = D._trainer(entropy_weight=3e-3, entropy_target=-50.0, alpha_lr=1.0, alpha_min=1e-3)
    for trainer in (free, floored):
        with torch.no_grad():
            trainer.log_alpha.fill_(-20.0)                       # collapsed temperature
    assert float(floored._temperature(floored.log_alpha, 0.0)) == pytest.approx(1e-3)
    assert float(free._temperature(free.log_alpha, 0.0)) == pytest.approx(math.exp(-20.0))
    metrics = floored.update()
    assert float(floored.log_alpha) == pytest.approx(math.log(1e-3))
    assert metrics["alpha"] == pytest.approx(1e-3)
    free.update()
    assert float(free.log_alpha) < -20.0                         # keeps falling without the floor


def test_alpha_min_floors_the_lane_temperature():
    torch.manual_seed(8)
    trainer = _cost_trainer(lanes=True, lane_entropy_target=-50.0, alpha_lr=1.0, alpha_min=1e-3)
    with torch.no_grad():
        trainer.log_lane_alpha.fill_(-20.0)
    metrics = trainer.update()
    assert float(trainer.log_lane_alpha) == pytest.approx(math.log(1e-3))
    assert metrics["lane_alpha"] == pytest.approx(1e-3)


# ------------------------------------------------------------------ 3. lane control
def test_lane_control_native_requires_no_lane_head():
    from wms.control.train_cli import policy_settings

    assert policy_settings({}) == policy_settings({"policy": {"lane_control": "learned"}})
    assert policy_settings({"policy": {"lane_control": "native"}})["lane_control"] == "native"
    with pytest.raises(ValueError, match="lane_control"):
        policy_settings({"policy": {"lane_control": "sumo"}})
    bad = CavPolicy(hidden_dim=16)
    bad.lane_control = "native"
    with pytest.raises(ValueError, match="lane head"):
        _cost_trainer(policy=bad)


def test_lane_control_native_imagination_keeps_mainline_cavs_in_lane():
    torch.manual_seed(9)
    trainer = _cost_trainer(policy=_native_policy())
    assert not trainer.use_lanes and trainer.lane_control == "native"
    metrics = trainer.update()
    assert "lane_entropy" not in metrics and math.isfinite(metrics["actor_grad_norm"])
    batch = trainer.start_states.sample(2)
    imagined = trainer.imagine(batch)
    start_lanes = batch.state["lane"]
    for state in imagined["states"][0]:
        assert torch.equal(state["lane"], start_lanes)            # no lane changes in imagination


def test_lane_control_native_checkpoint_and_executor_issue_no_lane_commands(tmp_path, monkeypatch):
    from wms.control import policy as policy_module

    monkeypatch.setattr("wms.control.dreamer.checked_path", lambda p, **k: Path(p))
    torch.manual_seed(10)
    trainer = _cost_trainer(policy=_native_policy(hidden=64), hidden=64)
    path = tmp_path / "native.pt"
    trainer.save_checkpoint(path)
    payload = torch.load(path, weights_only=True)
    assert payload["lane_control"] == "native" and payload["lane_head"] is False
    assert not any(key.startswith("lane_head.") for key in payload["policy"])
    exe = policy_module.load_policy_fn(str(path))
    assert exe.native_lanes and not exe.has_lane_head
    x = P2.state([-500.0, -900.0, -300.0], [0, 1, 1], cav=[True, True, False])
    accel, lanes = exe(x, with_lanes=True)
    assert lanes is None and accel.shape == (1, 3) and math.isnan(accel[0, 2])
    hold = PolicyExecutor(exe.policy, None, hold=True)
    assert not hold.has_lane_head and hold(x, with_lanes=True)[1] is None
    # A learned-lane executor (no lane_control attribute) is unchanged: lanes array.
    learned = PolicyExecutor(CavPolicy(), RsuGroupEncoder())
    assert learned.has_lane_head and learned(x, with_lanes=True)[1].shape == (1, 3)


def test_sumo_runner_passes_no_lanes_for_a_native_lane_policy(tmp_path, monkeypatch):
    from wms.control import policy as policy_module
    from traffic_shepherd.c4.baselines import run_policy_episode

    monkeypatch.setattr("wms.control.dreamer.checked_path", lambda p, **k: Path(p))
    trainer = _cost_trainer(policy=_native_policy(hidden=64), hidden=64)
    trainer.save_checkpoint(tmp_path / "native.pt")
    exe = policy_module.load_policy_fn(str(tmp_path / "native.pt"))
    x = P2.state([-500.0, -900.0, -300.0], [0, 1, 1], cav=[True, True, False])
    calls = []

    class Env:
        def reset(self, **kwargs):
            return x

        def step(self, accel, **kwargs):
            calls.append(kwargs)
            return x, {"done": len(calls) == 2, "summary": {"ok": True}}

    assert run_policy_episode(Env(), exe, seed=0, demand_vph=2400, penetration=0.2, warmup_s=0,
                              horizon_s=2) == {"ok": True}
    # step(accel) without ``lanes``: CavSumoEnv then never sets laneChangeMode 513 on the CAVs.
    assert calls == [{}, {}]


# ------------------------------------------------------------------ 5. configs + CLI smoke
def test_v7_configs_resolve():
    from wms.control.train_cli import _read_config, policy_settings

    raw = _read_config(REPO / "configs" / "c4_policy_v7.toml")
    cfg = DreamerConfig(**raw["trainer"])
    assert (cfg.ttc_budget_mode, cfg.lagrange_scaled, cfg.alpha_min) == ("native_relative", True, 1e-3)
    assert cfg.credit_profile_every == 0 and raw["counterfactual"]["max_starts"] == 3
    weights = RewardWeights(**raw["reward"])
    assert weights.ttc_pairs == "cav" and weights.merge_discharge == pytest.approx(0.8)
    assert policy_settings(raw)["lane_control"] == "native"
    assert raw["data"]["wm_config"] == "configs/c4_wm_v5m_refresh.toml"
    assert raw["data"]["start_sources"] == ["native_v5", "genrand_plain"]
    assert "start_source_weights" not in raw["data"]
    smoke = _read_config(REPO / "configs" / "c4_policy_v7_smoke.toml")
    assert smoke["run"]["updates"] == 2 and smoke["run"]["device"] == "cpu"
    assert DreamerConfig(**smoke["trainer"]).ttc_budget_mode == "native_relative"
    # v6 stays default off
    v6 = _read_config(REPO / "configs" / "c4_policy_v6_lowmem.toml")
    assert DreamerConfig(**v6["trainer"]).ttc_budget_mode == "absolute"
    assert RewardWeights(**v6["reward"]).ttc_pairs == "all"
    assert "lane_control" not in policy_settings(v6)


class DecayWorldModel(D.FakeWorldModel):
    """FakeWorldModel with the EXECCAP and V6 decay switches the v6/v7 trainer configs use."""

    cav_exec_cap = "off"
    grad_decay_per_step = 1.0


def test_cli_smoke_with_the_v7_config(monkeypatch):
    from wms.control import train_cli
    from wms.control.bridge import make_cav_controller
    from wms.control.obs import build_observations
    from wms.control.safety import safe_accel
    from wms.control.train_cli import GroupSignal

    hidden = 16
    monkeypatch.setattr(train_cli, "_load_components", lambda: {
        "policy": CavPolicy(hidden_dim=hidden), "group_encoder": GroupSignal(
            RsuGroupEncoder(hidden_dim=hidden)),
        "critic": TeamCritic(hidden_dim=hidden), "observation_builder": build_observations,
        "reward_fn": team_reward, "safety": safe_accel,
        "controller_factory": make_cav_controller})
    monkeypatch.setattr(train_cli, "_load_world_models", lambda paths, device: [DecayWorldModel()])
    monkeypatch.setattr(train_cli, "_load_starts", lambda data, models, seed, entries=False:
                        split_start_states(D._dataset(), seed=seed))
    with tempfile.TemporaryDirectory(prefix="c4-v7-cli-", dir=Path(__file__).parent) as tmp:
        output = Path(tmp)
        config = output / "v7.toml"
        config.write_text(
            f'extends = "{(REPO / "configs" / "c4_policy_v7.toml").as_posix()}"\n\n'
            "[run]\nthreads = 1\n", encoding="utf-8")
        arguments = ["--config", str(config), "--wm-ckpt", str(output / "fake.pt"),
                     "--smoke", "--output-dir", str(output)]
        assert train_cli.main(arguments) == 0, (output / "RUN_SUMMARY.json").read_text()
        summary = json.loads((output / "RUN_SUMMARY.json").read_text())
        assert summary["status"] == "completed" and summary["updates"] == 2
        assert summary["lane_control"] == "native" and summary["lane_head"] is False
        assert summary["config"]["ttc_budget_mode"] == "native_relative"
        logs = [json.loads(line) for line in (output / "updates.jsonl").read_text().splitlines()]
        assert len(logs) == 2
        for entry in logs:
            assert entry["ttc_budget_mode"] == "native_relative"
            assert entry["alpha"] >= 1e-3 and math.isfinite(entry["entropy"])
            assert "lane_entropy" not in entry
        payload = torch.load(output / "policy_latest.pt", weights_only=True)
        assert payload["lane_control"] == "native"
        assert payload["reward_weights"]["ttc_pairs"] == "cav"
        assert payload["reward_weights"]["merge_discharge"] == pytest.approx(0.8)
