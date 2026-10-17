"""V8POL (2026-10-17): Dreamer-policy fixes after the v7 run (docs/C4_MORNING_REPORT_20261017.md).

Every switch defaults off and is then bit-identical:
1. ``reward.ttc_pairs = "cav_follower"``: TTC cost only over pairs whose FOLLOWER (closing vehicle) is a CAV;
2. ``trainer.lagrange_max`` (cap on the TTC multiplier, logged) and ``trainer.lagrange_decay``;
3. ``reward.free_speed``: free-road CAV speed-shortfall incentive (reward term, not a controller);
4. ``configs/c4_policy_v8.toml`` resolves and runs a 2-update CLI smoke (fake WM).
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

from wms.control.contract import PHYSICS_DT_S, SPEED_LIMIT_MPS, RewardWeights
from wms.control.dreamer import DreamerConfig, DreamerTrainer
from wms.control.policy import CavPolicy, RsuGroupEncoder, TeamCritic
from wms.control.reward import _free_speed_shortfall, _ttc_cost, team_reward
from wms.control.start_states import split_start_states

REPO = Path(__file__).resolve().parents[2]
V8_KEYS = {"free_speed_shortfall", "lagrange_capped", "lagrange_pre_cap"}


def _helpers(name):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_v8pol_helpers_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D = _helpers("test_c4_dreamer")
P2 = _helpers("test_c4_policy2")
V7 = _helpers("test_c4_v7pol")


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _exposure(ttc, limit=3.0):                      # softplus relu(limit - TTC), 0.05 s temperature
    return 0.05 * math.log1p(math.exp((limit - ttc) / 0.05))


# ------------------------------------------------------------------ 1. TTC pairs: CAV follower only
def test_cav_follower_two_vehicle_example_both_ways():
    # Vehicle 0 at -500 m, 25 m/s closes on vehicle 1 at -470 m, 15 m/s: gap 25 m, closing 10 m/s, TTC 2.5 s.
    def pair(cav):
        return P2.state([-500.0, -470.0], [1, 1], v=[25.0, 15.0], cav=cav)
    cav_follows = pair([True, False])        # CAV follower (closing), HDV leader
    cav_leads = pair([False, True])          # HDV follower closing on a slow CAV leader
    expected = _exposure(2.5)
    assert float(_ttc_cost(cav_follows, 3.0, "cav_follower")) == pytest.approx(expected, rel=1e-5)
    assert float(_ttc_cost(cav_leads, 3.0, "cav_follower")) == 0.0
    # v7 "cav" counts both; "all" ignores the type.
    assert float(_ttc_cost(cav_leads, 3.0, "cav")) == pytest.approx(expected, rel=1e-5)
    assert float(_ttc_cost(cav_follows, 3.0, "cav")) == pytest.approx(expected, rel=1e-5)
    assert float(_ttc_cost(cav_leads, 3.0, "all")) == pytest.approx(expected, rel=1e-5)
    # Reversed slot order: same physics, same answer (no dependence on slot index).
    flipped = P2.state([-470.0, -500.0], [1, 1], v=[15.0, 25.0], cav=[False, True])
    assert float(_ttc_cost(flipped, 3.0, "cav_follower")) == pytest.approx(expected, rel=1e-5)
    flipped_lead = P2.state([-470.0, -500.0], [1, 1], v=[15.0, 25.0], cav=[True, False])
    assert float(_ttc_cost(flipped_lead, 3.0, "cav_follower")) == 0.0
    # Gradient: the CAV follower's speed raises the cost, the leader's speed lowers it.
    v = cav_follows["v_mps"].clone().requires_grad_(True)
    grad, = torch.autograd.grad(_ttc_cost({**cav_follows, "v_mps": v}, 3.0, "cav_follower"), v)
    assert grad[0, 0] > 0 and grad[0, 1] < 0
    with pytest.raises(ValueError, match="is_cav"):
        _ttc_cost({k: x for k, x in cav_follows.items() if k != "is_cav"}, 3.0, "cav_follower")


def test_cav_follower_on_the_v7_fixture_and_in_team_reward():
    x = V7._close_pairs_state()               # CAV 0 -> HDV 1 (TTC 2.5 s); HDV 2 -> HDV 3 (TTC 2.0 s)
    assert float(_ttc_cost(x, 3.0, "cav_follower")) == pytest.approx(_exposure(2.5), rel=1e-5)
    lead = {**x, "is_cav": torch.tensor([[False, False, False, True, False, False]])}
    assert float(_ttc_cost(lead, 3.0, "cav")) == pytest.approx(_exposure(2.0), rel=1e-5)
    assert float(_ttc_cost(lead, 3.0, "cav_follower")) == 0.0    # HDV 2 behind CAV 3: not charged
    metrics = team_reward(x, x, torch.zeros_like(x["s_m"]), RewardWeights(ttc_pairs="cav_follower"))
    assert torch.equal(metrics["cost_ttc"], _ttc_cost(x, 3.0, "cav_follower"))


def test_reward_defaults_are_bit_identical():
    x = V7._close_pairs_state()
    nxt = {**x, "s_m": x["s_m"] + 10.0, "v_mps": x["v_mps"] - 1.0}
    accel = torch.full_like(x["s_m"], -0.5)
    a = team_reward(x, nxt, accel, RewardWeights())
    b = team_reward(x, nxt, accel, RewardWeights(free_speed=0.0, free_speed_time_gap_s=1.0,
                                                 free_speed_range_m=5.0))
    assert a.keys() == b.keys() and "free_speed_shortfall" not in a
    for key in a:
        assert torch.equal(a[key], b[key]), key
    w = RewardWeights()
    assert (w.free_speed, w.free_speed_time_gap_s, w.free_speed_range_m) == (0.0, 3.0, 100.0)


# ------------------------------------------------------------------ 3. free-road speed incentive
def _free_state():
    """Positions are front bumpers, length 5 m (gap = ds - 5).
    Lane 1: CAV 0 alone (free, 20 m/s); CAV 1 at 20 m/s, gap 15 m behind HDV 2 (0.75 s -> not free);
    lane 0: CAV 3 at 20 m/s, gap 70 m behind HDV 4 (3.5 s > 3 -> free);
    CAV 5 at 2 m/s, gap 20 m behind HDV 6 (20 / max(2, 10) = 2 s -> NOT free, although 10 s at its
    own speed); HDV 7 slow and alone (no term)."""
    s = [1000.0, -500.0, -480.0, -900.0, -825.0, -1500.0, -1475.0, -2500.0]
    lane = [1, 1, 1, 0, 0, 0, 0, 0]
    v = [20.0, 20.0, 20.0, 20.0, 20.0, 2.0, 0.0, 5.0]
    cav = [True, True, False, True, False, True, False, False]
    return P2.state(s, lane, v=v, cav=cav)


def _next(x, **override):
    return {**x, "s_m": x["s_m"] + PHYSICS_DT_S * x["v_mps"], **override}


def test_free_speed_value_mask_and_normalisation():
    x = _free_state()
    nxt = _next(x)
    out = _free_speed_shortfall(x, nxt, 3.0, 100.0)
    per = PHYSICS_DT_S * (1.0 - 20.0 / SPEED_LIMIT_MPS)
    assert float(out) == pytest.approx(2 * per / 4, rel=1e-6)    # CAVs 0 and 3 free, 4 CAVs
    # a 10 m range frees CAVs 1 and 3 by distance (and CAV 5 by its 20 m gap; 10 s gap rule off)
    per5 = PHYSICS_DT_S * (1.0 - 2.0 / SPEED_LIMIT_MPS)
    assert float(_free_speed_shortfall(x, nxt, 1e9, 10.0)) == pytest.approx((3 * per + per5) / 4, rel=1e-6)
    # no CAVs -> 0; at the speed limit -> 0
    hdv = {**x, "is_cav": torch.zeros_like(x["is_cav"])}
    assert float(_free_speed_shortfall(hdv, {**nxt, "is_cav": hdv["is_cav"]}, 3.0, 100.0)) == 0.0
    fast = {**nxt, "v_mps": torch.full_like(nxt["v_mps"], SPEED_LIMIT_MPS)}
    assert float(_free_speed_shortfall(x, fast, 3.0, 100.0)) == 0.0
    with pytest.raises(ValueError, match="is_cav"):
        _free_speed_shortfall({k: t for k, t in x.items() if k != "is_cav"}, nxt, 3.0, 100.0)


def test_free_speed_reward_term_gradient_only_on_free_cavs():
    x = _free_state()
    v_next = x["v_mps"].clone().requires_grad_(True)
    nxt = _next(x, v_mps=v_next)
    accel = torch.zeros_like(x["s_m"])
    base = team_reward(x, nxt, accel, RewardWeights())
    on = team_reward(x, nxt, accel, RewardWeights(free_speed=0.8))
    diff = on["reward"] - base["reward"]
    assert float(diff.detach()) == pytest.approx(-0.8 * float(on["free_speed_shortfall"].detach()),
                                                 rel=1e-6)
    for key in base:                                            # every other component unchanged
        if key != "reward":
            assert torch.equal(base[key], on[key]), key
    grad, = torch.autograd.grad(diff.sum(), v_next)
    expected = 0.8 * PHYSICS_DT_S / SPEED_LIMIT_MPS / 4
    assert float(grad[0, 0]) == pytest.approx(expected, rel=1e-5)
    assert float(grad[0, 3]) == pytest.approx(expected, rel=1e-5)
    assert not grad[0, [1, 2, 4, 5, 6, 7]].any()               # non-free CAVs and all HDVs: zero


# ------------------------------------------------------------------ 2. lagrange cap / decay
def _dual_owner(**cfg):
    return SimpleNamespace(config=DreamerConfig(**cfg), lagrange_multiplier=0.0, update_count=0)


def test_lagrange_cap_and_decay_arithmetic(capsys):
    owner = _dual_owner(dual_lr=1.0, lagrange_max=2.0)
    assert DreamerTrainer.update_lagrange(owner, 1.5, 0.0) == pytest.approx(1.5)
    assert owner._lagrange_capped is False and owner._lagrange_pre_cap == pytest.approx(1.5)
    assert DreamerTrainer.update_lagrange(owner, 1.5, 0.0) == pytest.approx(2.0)     # 3.0 -> cap
    assert owner._lagrange_capped is True and owner._lagrange_pre_cap == pytest.approx(3.0)
    assert "[lagrange-cap]" in capsys.readouterr().out
    DreamerTrainer.update_lagrange(owner, 1.0, 0.0)                                   # still capped
    assert "[lagrange-cap]" not in capsys.readouterr().out                            # printed once
    assert DreamerTrainer.update_lagrange(owner, -0.5, 0.0) == pytest.approx(1.5)     # falls freely
    assert owner._lagrange_capped is False
    leaky = _dual_owner(dual_lr=1.0, lagrange_decay=0.1)
    leaky.lagrange_multiplier = 4.0
    assert DreamerTrainer.update_lagrange(leaky, 0.5, 0.0) == pytest.approx(0.9 * 4.0 + 0.5)
    assert DreamerTrainer.update_lagrange(leaky, -10.0, 0.0) == 0.0                   # still >= 0
    plain = _dual_owner(dual_lr=0.5)
    plain.lagrange_multiplier = 7.0
    assert DreamerTrainer.update_lagrange(plain, 5.0, 1.0) == 7.0 + 0.5 * 4.0         # default: uncapped
    assert not hasattr(plain, "_lagrange_capped")


def test_config_defaults_and_validation():
    cfg = DreamerConfig()
    assert (cfg.lagrange_max, cfg.lagrange_decay) == (None, 0.0)
    for bad in (dict(lagrange_max=-1.0), dict(lagrange_max=math.inf), dict(lagrange_max=True),
                dict(lagrange_max="2"), dict(lagrange_decay=-0.1), dict(lagrange_decay=1.0)):
        with pytest.raises(ValueError):
            DreamerConfig(**bad)
    DreamerConfig(lagrange_max=2.0, lagrange_decay=0.01)
    DreamerConfig(lagrange_max=0)


def test_defaults_off_leave_updates_and_checkpoints_bit_identical(tmp_path, monkeypatch):
    monkeypatch.setattr("wms.control.dreamer.checked_path", lambda p, **k: Path(p))
    torch.manual_seed(11)
    base = V7._cost_trainer(lanes=True, entropy_target=0.5, alpha_lr=1e-2)
    a = base.update()
    torch.manual_seed(11)
    explicit = V7._cost_trainer(lanes=True, entropy_target=0.5, alpha_lr=1e-2,
                                lagrange_max=None, lagrange_decay=0.0,
                                reward=RewardWeights(free_speed=0.0))
    b = explicit.update()
    a.pop("wall_time_s"), b.pop("wall_time_s")
    assert a == b and not V8_KEYS & a.keys()
    base.save_checkpoint(tmp_path / "a.pt")
    explicit.save_checkpoint(tmp_path / "b.pt")
    D._assert_equal(torch.load(tmp_path / "a.pt", weights_only=False),
                    torch.load(tmp_path / "b.pt", weights_only=False))


def test_update_logs_cap_and_free_speed():
    torch.manual_seed(5)
    trainer = V7._cost_trainer(ttc_budget=0.0, dual_lr=50.0, lagrange_max=0.25,
                               reward=RewardWeights(ttc_pairs="cav_follower", free_speed=0.8))
    metrics = trainer.update()
    assert metrics["cost_ttc"] > 0                              # CAV 0 closing on HDV 1
    assert metrics["lagrange_capped"] is True and metrics["lagrange_pre_cap"] > 0.25
    assert trainer.lagrange_multiplier == pytest.approx(0.25)
    assert metrics["lagrange_multiplier"] == pytest.approx(0.25)
    assert math.isfinite(metrics["free_speed_shortfall"]) and metrics["free_speed_shortfall"] >= 0
    assert math.isfinite(metrics["actor_grad_norm"])


# ------------------------------------------------------------------ 4. configs + CLI smoke
def test_v8_configs_resolve():
    from wms.control.train_cli import _read_config, policy_settings

    raw = _read_config(REPO / "configs" / "c4_policy_v8.toml")
    v7 = _read_config(REPO / "configs" / "c4_policy_v7.toml")
    cfg = DreamerConfig(**raw["trainer"])
    assert cfg.lagrange_max == pytest.approx(2.0) and cfg.lagrange_decay == 0.0
    assert (cfg.ttc_budget_mode, cfg.lagrange_scaled, cfg.alpha_min) == ("native_relative", True, 1e-3)
    weights = RewardWeights(**raw["reward"])
    assert weights.ttc_pairs == "cav_follower" and weights.free_speed == pytest.approx(0.8)
    assert weights.merge_discharge == pytest.approx(0.8)
    assert policy_settings(raw)["lane_control"] == "native"
    assert raw["run"]["output_dir"] == "runs/active/C4-POLICY-V8"
    # everything else is the v7 configuration
    for section in set(raw) | set(v7):
        if section not in ("trainer", "reward", "run"):
            assert raw[section] == v7[section], section
    assert {k: x for k, x in raw["trainer"].items() if k != "lagrange_max"} == v7["trainer"]
    assert {k: x for k, x in raw["reward"].items() if k not in ("ttc_pairs", "free_speed")} == \
        {k: x for k, x in v7["reward"].items() if k != "ttc_pairs"}
    assert {k: x for k, x in raw["run"].items() if k != "output_dir"} == \
        {k: x for k, x in v7["run"].items() if k != "output_dir"}
    smoke = _read_config(REPO / "configs" / "c4_policy_v8_smoke.toml")
    assert smoke["run"]["updates"] == 2 and smoke["run"]["device"] == "cpu"
    assert DreamerConfig(**smoke["trainer"]).lagrange_max == pytest.approx(2.0)
    assert DreamerConfig(**v7["trainer"]).lagrange_max is None           # v7 stays default off
    assert RewardWeights(**v7["reward"]).free_speed == 0.0


def test_cli_smoke_with_the_v8_config(monkeypatch):
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
    monkeypatch.setattr(train_cli, "_load_world_models", lambda paths, device: [V7.DecayWorldModel()])
    monkeypatch.setattr(train_cli, "_load_starts", lambda data, models, seed, entries=False:
                        split_start_states(D._dataset(), seed=seed))
    with tempfile.TemporaryDirectory(prefix="c4-v8-cli-", dir=Path(__file__).parent) as tmp:
        output = Path(tmp)
        config = output / "v8.toml"
        config.write_text(
            f'extends = "{(REPO / "configs" / "c4_policy_v8.toml").as_posix()}"\n\n'
            "[run]\nthreads = 1\n", encoding="utf-8")
        arguments = ["--config", str(config), "--wm-ckpt", str(output / "fake.pt"),
                     "--smoke", "--output-dir", str(output)]
        assert train_cli.main(arguments) == 0, (output / "RUN_SUMMARY.json").read_text()
        summary = json.loads((output / "RUN_SUMMARY.json").read_text())
        assert summary["status"] == "completed" and summary["updates"] == 2
        assert summary["config"]["lagrange_max"] == pytest.approx(2.0)
        assert summary["lane_control"] == "native" and summary["lane_head"] is False
        logs = [json.loads(line) for line in (output / "updates.jsonl").read_text().splitlines()]
        assert len(logs) == 2
        for entry in logs:
            assert entry["ttc_budget_mode"] == "native_relative"
            assert isinstance(entry["lagrange_capped"], bool)
            assert entry["lagrange_multiplier"] <= 2.0
            assert math.isfinite(entry["free_speed_shortfall"])
        payload = torch.load(output / "policy_latest.pt", weights_only=True)
        assert payload["reward_weights"]["ttc_pairs"] == "cav_follower"
        assert payload["reward_weights"]["free_speed"] == pytest.approx(0.8)
