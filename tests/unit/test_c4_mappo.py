"""CPU unit tests for the C4 MAPPO baseline (fake env, no SUMO, no partner modules).

Covers: GAE math, PPO loss signs (clipped surrogate, value clip, Lagrangian),
a 2-iteration trainer smoke with the fake env, checkpoint resume determinism,
the lazy worker-spec path and the CLI --smoke pipeline on the real TOML config.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from wms.control.contract import ACCEL_MAX_MPS2, ACCEL_MIN_MPS2, RewardWeights
from wms.control.mappo.buffer import (
    ReturnScaler,
    compute_gae,
    discounted_returns,
    episode_gae,
    flatten_episodes,
)
from wms.control.mappo.config_helpers import build_trainer, config_from_dict, merge_override
from wms.control.mappo.eval_bridge import load_policy_fn
from wms.control.mappo.fakeenv import (
    FakeActor,
    FakeCavEnv,
    FakeGroupEncoder,
    FakeTeamCritic,
    fake_build_observations,
    fake_safe_accel,
    fake_team_reward,
)
from wms.control.mappo.ppo import (
    PPOConfig,
    PPOUpdater,
    clipped_surrogate,
    clipped_value_loss,
    update_lagrange,
)
from wms.control.mappo.rollout import build_rollout_deps, collect_episode, run_job
from wms.control.mappo.trainer import Budget, MappoConfig, MappoTrainer, RolloutConfig
from wms.core.toml import loads

REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO / "configs" / "c4_mappo.toml"


def _fake_spec() -> dict:
    return {
        "env_name": "fake",
        "env_kwargs": {},
        "obs_builder_name": "fake_obs",
        "group_encoder_name": "fake_encoder",
        "actor_name": "fake_actor",
        "actor_kwargs": {},
        "group_encoder_kwargs": {},
        "safety_name": "fake_safety",
        "team_reward_name": "fake_reward",
        "reward_weights": {},
    }


def _trainer(
    out_dir: Path,
    *,
    seed: int = 7,
    iterations: int = 2,
    episodes: int = 2,
    horizon: float = 6.0,
    warmup: float = 0.0,
    max_sumo_seconds: float = 1.0e9,
) -> MappoTrainer:
    torch.manual_seed(11)
    cfg = MappoConfig(
        ppo=PPOConfig(epochs=2, minibatch_size=64, target_kl=0.0),
        rollout=RolloutConfig(
            seed=seed,
            iterations=iterations,
            episodes_per_iteration=episodes,
            horizon_s=horizon,
            warmup_s=warmup,
            penetrations=(0.2, 0.4),
            demands_vph=(2400,),
            max_sumo_seconds=max_sumo_seconds,
        ),
    )
    return MappoTrainer(
        out_dir=out_dir,
        config=cfg,
        env=FakeCavEnv(),
        obs_builder=fake_build_observations,
        group_encoder=FakeGroupEncoder(),
        actor=FakeActor(),
        critic=FakeTeamCritic(),
        safety_layer=fake_safe_accel,
        team_reward=fake_team_reward,
        meta={"names": _fake_spec()},
    )


def _collect(seed: int = 3, horizon: float = 4.0):
    torch.manual_seed(seed)
    env = FakeCavEnv()
    return collect_episode(
        env,
        obs_builder=fake_build_observations,
        group_encoder=FakeGroupEncoder(),
        actor=FakeActor(),
        safety_layer=fake_safe_accel,
        team_reward=fake_team_reward,
        weights=RewardWeights(),
        seed=seed,
        demand_vph=2600,
        penetration=0.2,
        warmup_s=0.0,
        horizon_s=horizon,
    )


def test_gae_matches_hand_computation():
    rewards = torch.tensor([1.0, 2.0, 3.0])
    values = torch.tensor([0.5, 1.0, 0.5])
    adv, ret = compute_gae(rewards, values, gamma=0.9, lam=0.5, bootstrap_value=torch.tensor(2.0))
    expected = torch.tensor([2.92325, 3.385, 4.3])
    assert torch.allclose(adv, expected, atol=1e-4)
    assert torch.allclose(ret, adv + values, atol=1e-4)


def test_gae_termination_drops_bootstrap():
    episode = _collect()
    batch = flatten_episodes([episode])
    values = torch.linspace(0.1, 0.5, batch.n_frames)
    finals = torch.tensor([99.0])
    adv_boot, _, _ = episode_gae(batch, values, finals, gamma=0.9, lam=0.5)
    manual, _ = compute_gae(
        batch.rewards, values, gamma=0.9, lam=0.5, bootstrap_value=torch.tensor(0.0)
    )
    batch.terminated[0] = True
    adv_term, _, _ = episode_gae(batch, values, finals, gamma=0.9, lam=0.5)
    assert torch.allclose(adv_term, manual, atol=1e-6)  # terminated: no bootstrap
    assert not torch.allclose(adv_boot, manual, atol=1e-3)  # truncated: bootstrapped


def test_discounted_returns():
    out = discounted_returns(torch.tensor([1.0, 1.0, 1.0]), 0.5)
    assert torch.allclose(out, torch.tensor([1.75, 1.5, 1.0]))


def test_return_scaler_roundtrip_and_stats():
    scaler = ReturnScaler()
    scaler.update(torch.arange(1, 5, dtype=torch.float32))
    assert abs(scaler.mean - 2.5) < 1e-6
    assert torch.allclose(scaler.scale(torch.tensor(2.5)), torch.tensor(0.0), atol=1e-6)
    x = torch.tensor([3.0, 4.0])
    assert torch.allclose(scaler.unscale(scaler.scale(x)), x, atol=1e-5)
    clone = ReturnScaler.from_state(scaler.state_dict())
    assert torch.allclose(clone.scale(x), scaler.scale(x))


def test_clipped_surrogate_signs_and_clipping():
    eps = 0.2
    a_pos = torch.tensor([1.0])
    a_neg = torch.tensor([-1.0])
    surr_pos = [
        float(clipped_surrogate(torch.tensor([r]), a_pos, eps)) for r in (0.8, 1.0, 1.1, 1.25, 1.4)
    ]
    assert surr_pos == sorted(surr_pos)  # rising with the ratio
    assert abs(surr_pos[3] - surr_pos[4]) < 1e-6  # flat beyond 1 + eps
    surr_neg = [
        float(clipped_surrogate(torch.tensor([r]), a_neg, eps)) for r in (0.5, 0.75, 1.0, 1.3)
    ]
    assert surr_neg == sorted(surr_neg, reverse=True)  # ratio pushed below 1
    assert abs(surr_neg[0] - surr_neg[1]) < 1e-6  # flat below 1 - eps


def test_clipped_value_loss_blocks_oversized_improvement():
    eps = 0.2
    inside = clipped_value_loss(torch.tensor([0.1]), torch.tensor([0.0]), torch.tensor([0.0]), eps)
    assert abs(float(inside) - 0.5 * 0.01) < 1e-6  # unclipped branch
    jumped = clipped_value_loss(torch.tensor([0.5]), torch.tensor([0.0]), torch.tensor([1.0]), eps)
    assert abs(float(jumped) - 0.5 * 0.64) < 1e-6  # capped at v_old + eps


def test_lagrange_multiplier_dual_ascent():
    up = update_lagrange(1.0, mean_cost=0.1, cost_limit=0.02, lr=0.05, alpha_max=50.0)
    assert up > 1.0
    down = update_lagrange(0.01, mean_cost=0.0, cost_limit=0.02, lr=1.0, alpha_max=50.0)
    assert down == 0.0  # clamped at zero
    cap = update_lagrange(49.99, mean_cost=10.0, cost_limit=0.0, lr=1.0, alpha_max=50.0)
    assert cap == 50.0


def test_ppo_update_pushes_logp_toward_advantage():
    episode = _collect()
    batch = flatten_episodes([episode])
    torch.manual_seed(5)
    actor, critic, encoder = FakeActor(), FakeTeamCritic(), FakeGroupEncoder()
    updater = PPOUpdater(
        actor=actor,
        critic=critic,
        group_encoder=encoder,
        obs_builder=fake_build_observations,
        config=PPOConfig(epochs=4, minibatch_size=16, lr_actor=0.01, target_kl=0.0),
    )
    n = batch.n_frames
    adv = 1.0 + 0.01 * torch.arange(n, dtype=torch.float32)
    diag = updater.update(
        batch,
        advantages=adv,
        returns=torch.zeros(n),
        cost_returns=torch.zeros(n),
        scaler=ReturnScaler(),
    )
    assert diag["finite"]
    with torch.no_grad():
        obs = fake_build_observations(batch.states)
        logp_new, _ = actor.evaluate_actions(obs, encoder(obs), batch.actions)
    rows, cols = (
        batch.cav_mask.nonzero(as_tuple=False)[:, 0],
        batch.cav_mask.nonzero(as_tuple=False)[:, 1],
    )
    delta = logp_new[rows, cols] - batch.logp[rows, cols]
    advantage = adv[rows] - adv[rows].mean()
    corr = float((delta * advantage).sum() / (delta.norm() * advantage.norm() + 1e-8))
    assert corr > 0.1, (corr, float(delta.mean()))


def test_two_iteration_smoke_artifacts(tmp_path):
    out = tmp_path / "run"
    trainer = _trainer(out)
    init = {k: v.clone() for k, v in trainer.actor.state_dict().items()}
    result = trainer.run()
    assert result["status"] == "completed"
    assert result["iterations_completed"] == 2
    lines = [
        json.loads(x) for x in out.joinpath("log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert lines[0]["type"] == "config"
    assert [x["type"] for x in lines[1:]] == ["iter", "iter"]
    assert all(x["pi_loss"] == x["pi_loss"] for x in lines[1:])
    for name in ("latest.pt", "ckpt_00001.pt", "ckpt_00002.pt"):
        assert (out / "checkpoints" / name).is_file()
    summary = json.loads(out.joinpath("RUN_SUMMARY.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert summary["budget"]["used_sumo_seconds"] == 4 * 6.0
    assert summary["budget"]["used_env_steps"] == 24
    final = trainer.actor.state_dict()
    assert any(not torch.allclose(init[k], final[k]) for k in init)
    policy_fn = load_policy_fn(out / "checkpoints" / "latest.pt")
    state = FakeCavEnv().reset(
        seed=1, demand_vph=2400, penetration=0.2, warmup_s=0.0, horizon_s=10.0
    )
    accel = policy_fn(state)
    assert accel.shape == (1, state["v_mps"].shape[1])
    assert float(accel.min()) >= ACCEL_MIN_MPS2 - 1e-6
    assert float(accel.max()) <= ACCEL_MAX_MPS2 + 1e-6


def test_budget_blocks_oversized_episodes():
    budget = Budget(max_sumo_seconds=13.0, max_env_steps=None)
    assert budget.can_start(6.0, 0.0, False)
    budget.used_sumo_seconds = 8.0
    assert not budget.can_start(6.0, 0.0, False)  # 8 + 6 > 13
    assert budget.can_start(5.0, 0.0, False)
    charged = Budget(max_sumo_seconds=100.0, max_env_steps=None)
    episode = _collect(horizon=4.0)
    charged.charge(episode, warmup_s=2.0, charge_warmup=False)
    assert charged.used_sumo_seconds == 4.0 and charged.used_env_steps == 4
    charged.charge(episode, warmup_s=2.0, charge_warmup=True)
    assert charged.used_sumo_seconds == 10.0


def test_resume_matches_straight_run(tmp_path):
    straight_dir, split_dir = tmp_path / "straight", tmp_path / "split"
    straight = _trainer(straight_dir)
    straight.run()
    first = _trainer(split_dir)
    first.run(1)
    resumed = _trainer(split_dir)
    result = resumed.run(1)
    assert result["iterations_completed"] == 2
    for module in ("actor", "group_encoder", "critic"):
        params_a = getattr(straight, module).state_dict()
        params_b = getattr(resumed, module).state_dict()
        assert set(params_a) == set(params_b)
        assert all(torch.allclose(params_a[k], params_b[k], atol=1e-6) for k in params_a)
    assert straight.budget.used_sumo_seconds == resumed.budget.used_sumo_seconds
    lines = [
        json.loads(x)
        for x in split_dir.joinpath("log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [x["type"] for x in lines] == ["config", "iter", "iter"]
    assert (
        json.loads(split_dir.joinpath("RUN_SUMMARY.json").read_text(encoding="utf-8"))[
            "iterations_completed"
        ]
        == 2
    )


def test_worker_spec_path_builds_and_runs():
    spec = _fake_spec()
    deps = build_rollout_deps(spec)
    assert isinstance(deps["env"], FakeCavEnv)
    assert callable(deps["obs_builder"]) and callable(deps["team_reward"])
    job = {
        "spec": spec,
        "actor_state": deps["actor"].state_dict(),
        "encoder_state": deps["group_encoder"].state_dict(),
        "seed": 5,
        "demand_vph": 2600,
        "penetration": 0.1,
        "warmup_s": 0.0,
        "horizon_s": 4.0,
    }
    row = run_job(job)
    episode = row["episode"]
    assert episode.steps == 4 and len(episode.states) == 5
    assert episode.logp.shape == (4, episode.actions.shape[1])
    assert bool(episode.cav_mask.any())
    assert torch.isfinite(episode.logp).all()


def test_config_toml_defaults_and_smoke_merge(tmp_path):
    cfg = loads(CONFIG_PATH.read_text(encoding="utf-8"))
    parsed = config_from_dict(cfg)
    assert parsed.ppo.gamma == 0.99 and parsed.ppo.gae_lambda == 0.95
    assert parsed.ppo.clip_eps == 0.2 and parsed.ppo.clip_value_eps == 0.2
    assert parsed.rollout.penetrations == (0.05, 0.1, 0.2, 0.4)
    assert parsed.rollout.demands_vph == (2400, 2600, 2700)
    assert parsed.rollout.max_env_steps is None  # 0 in TOML = no cap
    assert parsed.rollout.max_sumo_seconds > 0
    merged = config_from_dict(merge_override(cfg, cfg["smoke"]))
    assert merged.rollout.horizon_s == 6.0
    assert merged.ppo.epochs == 2
    trainer = build_trainer(
        merge_override(cfg, cfg["smoke"]), workers=1, out_dir=tmp_path / "cfg_run"
    )
    result = trainer.run()
    assert result["status"] == "completed"
    assert result["budget"]["used_sumo_seconds"] <= 64.0


def test_cli_smoke_on_real_config(tmp_path):
    from wms.control.mappo.train import main

    assert (
        main(
            [
                "--config",
                str(CONFIG_PATH),
                "--workers",
                "1",
                "--out",
                str(tmp_path / "cli_run"),
                "--smoke",
                "--max-sumo-seconds",
                "24",
            ]
        )
        == 0
    )
    summary = json.loads((tmp_path / "cli_run" / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    assert summary["status"] in ("completed", "budget_exhausted")
    assert summary["budget"]["used_sumo_seconds"] <= 24.0
    assert summary["budget"]["max_sumo_seconds"] == 24.0
