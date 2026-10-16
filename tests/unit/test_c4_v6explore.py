"""V6EXPLORE (2026-10-15): stop the actor's premature entropy collapse.

1. ``trainer.entropy_target`` / ``lane_entropy_target`` (SAC-style learned ``log_alpha``): default
   off is bit-identical (no temperature keys anywhere), alpha starts at the fixed
   ``entropy_weight``, rises when the actor's entropy is below the target and falls above it,
   is logged each update and checkpointed / resumable exactly;
2. ``policy.init_zero_accel``: the fresh actor's decoded mean acceleration is ~0 m/s^2 on random
   inputs (the random default decodes near the [-3, 1.2] midpoint -0.9 and mostly brakes);
3. ``configs/c4_policy_v6_explore.toml`` resolves (v5 unchanged, defaults off) and the CLI smoke
   run carries the targets and the zero-mean start into the trainer digest.
CPU only; no SUMO, no checkpoint files from runs/."""
from __future__ import annotations

import importlib.util
import json
import math
import tempfile
from pathlib import Path

import pytest
import torch

from wms.control.contract import POLICY_ACCEL_MAX_MPS2
from wms.control.dreamer import DreamerConfig
from wms.control.policy import CavPolicy
from wms.control.start_states import split_start_states

REPO = Path(__file__).resolve().parents[2]
TEMPERATURE_KEYS = {"alpha", "entropy_target", "lane_alpha", "lane_entropy_target"}


def _helpers(name):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_v6explore_helpers_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D = _helpers("test_c4_dreamer")
P2 = _helpers("test_c4_policy2")


class ExecCapWorldModel(D.FakeWorldModel):
    """FakeWorldModel with the EXECCAP switch (the v5/v6 trainer configs set cav_exec_cap)."""

    cav_exec_cap = "off"


def alpha_of(trainer, lane=False):
    return float((trainer.log_lane_alpha if lane else trainer.log_alpha).detach())


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


# ------------------------------------------------------------------ config defaults
def test_config_defaults_off_and_validation():
    cfg = DreamerConfig()
    assert cfg.entropy_target is None and cfg.lane_entropy_target is None
    assert cfg.alpha_lr == pytest.approx(3e-4)
    assert CavPolicy().head[-1].bias.shape == (2,) and not CavPolicy().init_zero_accel
    for bad in (dict(entropy_target=True), dict(entropy_target=math.inf),
                dict(lane_entropy_target="high"), dict(alpha_lr=0.0), dict(alpha_lr=-1e-4)):
        with pytest.raises(ValueError):
            DreamerConfig(**bad)
    DreamerConfig(entropy_target=-1.0, lane_entropy_target=0.5)   # negative targets are entropy


# ------------------------------------------------------------------ default off = bit-identical
def test_defaults_off_leave_updates_and_checkpoints_bit_identical():
    torch.manual_seed(9)
    base = D._trainer(ensemble=2)
    a = base.update()
    torch.manual_seed(9)
    explicit = D._trainer(ensemble=2, entropy_target=None, lane_entropy_target=None,
                          alpha_lr=3e-4)
    b = explicit.update()
    for trainer in (base, explicit):
        assert trainer.log_alpha is None and trainer.alpha_optimizer is None
        assert trainer.log_lane_alpha is None and trainer.lane_alpha_optimizer is None
    a.pop("wall_time_s"), b.pop("wall_time_s")
    assert a == b and not TEMPERATURE_KEYS & a.keys()
    with tempfile.TemporaryDirectory(prefix="c4-v6-", dir=Path(__file__).parent) as tmp:
        first, second = Path(tmp) / "a.pt", Path(tmp) / "b.pt"
        base.save_checkpoint(first)
        explicit.save_checkpoint(second)
        payload_a = torch.load(first, map_location="cpu", weights_only=False)
        payload_b = torch.load(second, map_location="cpu", weights_only=False)
        D._assert_equal(payload_a, payload_b)
        assert not {"log_alpha", "alpha_optimizer", "log_lane_alpha",
                    "lane_alpha_optimizer"} & payload_a.keys()
        assert payload_a["config"]["entropy_target"] is None


# ------------------------------------------------------------------ the entropy temperature
def test_alpha_starts_at_the_fixed_weight_and_moves_toward_the_target():
    torch.manual_seed(7)
    rising = D._trainer(entropy_weight=3e-3, entropy_target=5.0, alpha_lr=0.1)
    assert alpha_of(rising) == pytest.approx(math.log(3e-3))    # continues from the bonus
    before = alpha_of(rising)
    metrics = rising.update()
    assert metrics["entropy"] < 5.0                                    # toy actor is too sure
    assert alpha_of(rising) > before, "alpha must rise below the target"
    assert metrics["alpha"] == pytest.approx(math.exp(alpha_of(rising)))
    assert metrics["entropy_target"] == pytest.approx(5.0)
    torch.manual_seed(7)
    falling = D._trainer(entropy_weight=3e-3, entropy_target=-50.0, alpha_lr=0.1)
    before = alpha_of(falling)
    metrics = falling.update()
    assert metrics["entropy"] > -50.0
    assert alpha_of(falling) < before, "alpha must fall above the target"


def test_alpha_is_checkpointed_and_resumes_exactly():
    torch.manual_seed(11)
    trainer = D._trainer(entropy_weight=3e-3, entropy_target=5.0, alpha_lr=0.05, ensemble=2)
    D._assert_finite_metrics(trainer.update())
    with tempfile.TemporaryDirectory(prefix="c4-v6-resume-", dir=Path(__file__).parent) as tmp:
        checkpoint = Path(tmp) / "trainer.pt"
        trainer.save_checkpoint(checkpoint)
        expected_metrics = trainer.update()
        expected_alpha = alpha_of(trainer)
        expected = Path(tmp) / "expected-next.pt"
        trainer.save_checkpoint(expected)
        D._seed(999)
        resumed = D._trainer(entropy_weight=3e-3, entropy_target=5.0, alpha_lr=0.05, ensemble=2)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert "log_alpha" in payload and "alpha_optimizer" in payload
        resumed.load_checkpoint(checkpoint)
        assert alpha_of(resumed) == float(payload["log_alpha"])
        actual_metrics = resumed.update()
        assert TEMPERATURE_KEYS & expected_metrics.keys()
        assert alpha_of(resumed) == pytest.approx(expected_alpha)
        expected_metrics.pop("wall_time_s"), actual_metrics.pop("wall_time_s")
        assert actual_metrics == expected_metrics
        actual = Path(tmp) / "actual-next.pt"
        resumed.save_checkpoint(actual)
        D._assert_equal(torch.load(expected, map_location="cpu", weights_only=False),
                        torch.load(actual, map_location="cpu", weights_only=False))


def test_lane_alpha_moves_toward_the_target_and_is_logged():
    torch.manual_seed(8)
    rising = P2._real_trainer(lane_entropy_target=2.0, alpha_lr=0.1)
    before = alpha_of(rising, lane=True)
    metrics = rising.update()
    assert 0.0 < metrics["lane_entropy"] < 2.0                          # uniform head: log 2..log 3
    assert alpha_of(rising, lane=True) > before
    assert metrics["lane_alpha"] == pytest.approx(math.exp(alpha_of(rising, lane=True)))
    assert metrics["lane_entropy_target"] == pytest.approx(2.0)
    torch.manual_seed(8)
    falling = P2._real_trainer(lane_entropy_target=-5.0, alpha_lr=0.1)
    before = alpha_of(falling, lane=True)
    metrics = falling.update()
    assert metrics["lane_entropy"] > -5.0
    assert alpha_of(falling, lane=True) < before
    with pytest.raises(ValueError, match="lane head"):
        D._trainer(lane_entropy_target=0.5)                             # no lanes in this trainer


# ------------------------------------------------------------------ zero-mean acceleration start
def _decoded_means(policy, bounds, trials=16):
    policy.eval()
    policy.set_accel_bounds(*bounds)
    generator = torch.Generator().manual_seed(3)
    means = []
    with torch.no_grad():
        for _ in range(trials):
            speeds = torch.rand((1, 4), generator=generator) * 25.0 + 1.0
            x = P2.state([-600.0, -570.0, -540.0, -510.0], [0, 1, 1, 1], v=speeds.squeeze(0).tolist())
            obs, signal = P2._obs_and_signal(x)
            accel, _ = policy(obs, signal, deterministic=True)
            means.append(accel[0])
    return torch.stack(means)


def test_init_zero_accel_decodes_zero_mean_on_random_inputs():
    torch.manual_seed(0)
    policy = CavPolicy(accel_min=-3.0, accel_max=1.2, init_zero_accel=True)
    assert policy.init_zero_accel
    decoded = _decoded_means(policy, (-3.0, 1.2))
    assert decoded.abs().max() < 0.05, decoded.abs().max()            # |mean| ~ 0 m/s^2
    # a configured cap re-centres the bias (fresh actors stay zero-mean)
    policy.set_accel_bounds(-3.0, 2.6)
    decoded = _decoded_means(policy, (-3.0, 2.6))
    assert decoded.abs().max() < 0.05
    # only the MEAN row is initialised: the log-std row keeps its random values
    assert policy.head[-1].weight[1].abs().sum() > 0
    # the default random start decodes near the midpoint (-0.9) and mostly brakes: the asymmetry
    # the switch removes (v5 evidence: initial action mean -0.67)
    torch.manual_seed(0)
    default = CavPolicy(accel_min=-3.0, accel_max=1.2)
    assert not default.init_zero_accel
    decoded = _decoded_means(default, (-3.0, 1.2))
    assert decoded.mean() < -0.5


# ------------------------------------------------------------------ config + CLI digest
def test_v6_explore_config_resolves_and_v5_stays_default_off():
    from wms.control.train_cli import _read_config, policy_settings

    raw = _read_config(REPO / "configs" / "c4_policy_v6_explore.toml")
    cfg = DreamerConfig(**raw["trainer"])
    assert cfg.entropy_target == pytest.approx(-1.0) and cfg.alpha_lr == pytest.approx(3e-4)
    assert cfg.lane_entropy_target is None
    assert policy_settings(raw) == {"accel_max": 1.2, "init_zero_accel": True}
    v5 = _read_config(REPO / "configs" / "c4_policy_v5.toml")
    assert DreamerConfig(**v5["trainer"]).entropy_target is None
    assert policy_settings(v5) == {"accel_max": 1.2}
    assert policy_settings({}) == {"accel_max": POLICY_ACCEL_MAX_MPS2}
    with pytest.raises(ValueError, match="init_zero_accel"):
        policy_settings({"policy": {"init_zero_accel": "yes"}})


def test_cli_smoke_digest_carries_targets_alpha_logging_and_zero_mean(monkeypatch):
    from wms.control import train_cli
    from wms.control.bridge import make_cav_controller
    from wms.control.obs import build_observations
    from wms.control.policy import RsuGroupEncoder, TeamCritic
    from wms.control.reward import team_reward
    from wms.control.safety import safe_accel
    from wms.control.train_cli import GroupSignal

    hidden = 16
    monkeypatch.setattr(train_cli, "_load_components", lambda: {
        "policy": CavPolicy(hidden_dim=hidden), "group_encoder": GroupSignal(
            RsuGroupEncoder(hidden_dim=hidden)),
        "critic": TeamCritic(hidden_dim=hidden), "observation_builder": build_observations,
        "reward_fn": team_reward, "safety": safe_accel,
        "controller_factory": make_cav_controller})
    monkeypatch.setattr(train_cli, "_load_world_models", lambda paths, device: [ExecCapWorldModel()])
    monkeypatch.setattr(train_cli, "_load_starts", lambda data, models, seed, entries=False:
                        split_start_states(D._dataset(), seed=seed))
    with tempfile.TemporaryDirectory(prefix="c4-v6-cli-", dir=Path(__file__).parent) as tmp:
        output = Path(tmp)
        config = output / "v6.toml"
        config.write_text(
            f'extends = "{(REPO / "configs" / "c4_policy_v6_explore.toml").as_posix()}"\n\n'
            "[run]\nthreads = 1\n", encoding="utf-8")
        arguments = ["--config", str(config), "--wm-ckpt", str(output / "fake.pt"),
                     "--smoke", "--output-dir", str(output)]
        assert train_cli.main(arguments) == 0
        summary = json.loads((output / "RUN_SUMMARY.json").read_text())
        assert summary["status"] == "completed" and summary["updates"] == 2
        assert summary["config"]["entropy_target"] == pytest.approx(-1.0)
        assert summary["policy_init_zero_accel"] is True
        assert summary["policy_accel_bounds"] == [-3.0, 1.2]
        logs = [json.loads(line) for line in (output / "updates.jsonl").read_text().splitlines()]
        assert len(logs) == 2
        for entry in logs:
            assert entry["entropy_target"] == pytest.approx(-1.0)
            assert math.isfinite(entry["alpha"]) and entry["alpha"] > 0
            assert math.isfinite(entry["entropy"])
